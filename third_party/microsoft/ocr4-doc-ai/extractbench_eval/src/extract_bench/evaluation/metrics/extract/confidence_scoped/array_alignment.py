"""Array-row alignment for the confidence-scoped scorer.

One algorithm for every array size: propose candidate pred/GT row pairs, score
them with full-row similarity, then solve the assignment optimally per
connected component. Array size only changes how candidates are proposed.
Small arrays enumerate every pair. Large arrays use a bounded sparse candidate
set from shared row features, blocking keys, and index proximity before solving
the same assignment problem over the candidates.
"""

from __future__ import annotations

import logging
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .gt_normalization import REPEATED_STRUCTURE_IDENTITY_SPELLINGS
from .paths import (
    field_name,
    format_path,
    iter_leaf_values,
    path_without_indices,
    value_at,
)
from .types import ArrayAlignment
from .value_matching import (
    canonical_text,
    compare_values,
    parse_single_number,
    token_containment_score,
    tokens_for,
)

logger = logging.getLogger(__name__)

# Full pairwise fuzzy alignment (every pred x GT pair scored) runs when the
# array is small enough; larger arrays use the sparse-candidate path below.
MAX_FUZZY_ARRAY_PAIRS = 40000
# Secondary budget on total leaf-comparison work (pred leaves x GT leaves),
# so a few hundred very wide rows cannot sneak past the pair budget.
MAX_FUZZY_ARRAY_WORK = 800000
# Row-similarity discounts: indirect evidence (best-leaf fuzzy match, whole-row
# token overlap) is weighted below direct same-suffix leaf agreement.
FUZZY_LEAF_SCORE_WEIGHT = 0.92
ROW_TEXT_SCORE_WEIGHT = 0.9
# Similarity is averaged over fields BOTH rows populate, letting a near-empty
# fragment outscore a complete row; scale by coverage of the fuller row.
ROW_COVERAGE_WEIGHT_BASE = 0.5
# Floor fuzzy assignments relative to the array's own accepted-score median:
# equal-length 1:1 assignment otherwise force-marries an extra predicted row
# to a missed GT row. Feature/identity modes keep their own guards.
ADAPTIVE_FLOOR_MIN_ALIGNMENTS = 5
ADAPTIVE_FLOOR_MEDIAN_FRACTION = 0.75
# Mode labels for the pure-similarity assignments (the only floored modes).
FUZZY_SIMILARITY_MODE = "fuzzy_similarity_array_row"
SPARSE_FUZZY_MODE = "sparse_fuzzy_array_row"
_ADAPTIVE_FLOOR_MODES = frozenset({FUZZY_SIMILARITY_MODE, SPARSE_FUZZY_MODE})
# Minimum full-row similarity for any fuzzy/sparse pairing to enter assignment.
FUZZY_ROW_SIMILARITY_MIN = 0.25
# Minimum full-row similarity to accept a blocked-identity alignment (guards
# against a shared key value that is actually two different entities).
BLOCK_MIN_SIMILARITY = 0.5
# An inferred identity key must be near-unique on both sides to be trusted.
IDENTITY_UNIQUE_RATIO_MIN = 0.9
# ...and its values must overlap on at least max(2, 20% of the smaller side).
IDENTITY_MIN_OVERLAP = 2
IDENTITY_MIN_OVERLAP_RATIO = 0.2
# Minimum combined rarity score for a composite-feature row pairing.
COMPOSITE_FEATURE_SCORE_MIN = 0.75
# A scalar value is "distinctive" (usable as a row fingerprint) at this size.
DISTINCTIVE_VALUE_MIN_CHARS = 30
DISTINCTIVE_VALUE_MIN_TOKENS = 5
# A scalar value is "informative" (usable as a composite feature) at this size.
INFORMATIVE_TEXT_MIN_CHARS = 20
INFORMATIVE_TEXT_MIN_TOKENS = 3
# Sparse candidate generation: a shared feature proposes pairs only when it is
# rare enough on both sides (common values like a state code propose nothing).
SPARSE_FEATURE_OCCURRENCE_LIMIT = 50
SPARSE_FEATURE_PAIR_LIMIT = 400
# Positional prior: for mostly-in-order arrays up to this many rows, pred[i]
# is also paired with gt[i +/- window].
SPARSE_INDEX_WINDOW = 2
SPARSE_INDEX_WINDOW_MAX_ROWS = 200
# Blocking-key quality gates: decent coverage and strong distinctness so a
# constant-ish column (state, country) or a weak shared term cannot block.
BLOCKING_KEY_MIN_COVERAGE = 0.5
BLOCKING_KEY_MIN_DISTINCT = 0.7
BLOCKING_KEY_IDENTITY_BOOST = 1.6


@dataclass
class PreparedRow:
    """Precomputed per-row views so candidate generation and scoring do not
    re-canonicalize the same values for every candidate pair."""

    index: int
    row: dict[str, Any]
    leaves: list[tuple[str, Any]]
    scalar: dict[str, Any]
    canonical_by_key: dict[str, str]
    numeric_by_key: dict[str, float]
    tokens_by_key: dict[str, set[str]]
    informative_features: set[tuple[str, str]]
    distinctive_values: list[str]
    row_text: str


def _row_coverage_weight(pred_leaves: list[tuple[str, Any]], gt_leaves: list[tuple[str, Any]]) -> float:
    pred_nonnull = sum(1 for _, value in pred_leaves if value is not None)
    gt_nonnull = sum(1 for _, value in gt_leaves if value is not None)
    fuller = max(pred_nonnull, gt_nonnull)
    coverage = min(pred_nonnull, gt_nonnull) / fuller if fuller else 0.0
    return ROW_COVERAGE_WEIGHT_BASE + (1 - ROW_COVERAGE_WEIGHT_BASE) * coverage


def _combined_row_similarity(
    direct_scores: list[float],
    fuzzy_scores: list[float],
    row_text: float,
    pred_leaves: list[tuple[str, Any]],
    gt_leaves: list[tuple[str, Any]],
) -> float:
    direct = sum(direct_scores) / len(direct_scores) if direct_scores else 0.0
    fuzzy = sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0.0
    base = max(direct, fuzzy * FUZZY_LEAF_SCORE_WEIGHT, row_text * ROW_TEXT_SCORE_WEIGHT)
    return base * _row_coverage_weight(pred_leaves, gt_leaves)


def row_similarity(predicted_row: Any, gt_row: Any, *, schema: Any, parent_path: str) -> float:
    if not isinstance(predicted_row, dict) or not isinstance(gt_row, dict):
        return 0.0

    pred_leaves = iter_leaf_values(predicted_row)
    gt_leaves = iter_leaf_values(gt_row)
    if not pred_leaves or not gt_leaves:
        return 0.0

    direct_scores: list[float] = []
    gt_by_suffix = dict(gt_leaves)
    for suffix, pred_value in pred_leaves:
        if pred_value is None:
            continue
        if suffix in gt_by_suffix:
            ok, score, _ = compare_values(
                gt_by_suffix[suffix],
                pred_value,
                schema=schema,
                predicted_path=f"{parent_path}[0].{suffix}",
                expected_path=f"{parent_path}[0].{suffix}",
            )
            direct_scores.append(1.0 if ok else score)

    fuzzy_scores: list[float] = []
    for _, pred_value in pred_leaves:
        if pred_value is None:
            continue
        best = 0.0
        for _, gt_value in gt_leaves:
            if gt_value is None:
                continue
            ok, score, _ = compare_values(
                gt_value,
                pred_value,
                schema=schema,
                predicted_path=parent_path,
                expected_path=parent_path,
            )
            best = max(best, 1.0 if ok else score, token_containment_score(gt_value, pred_value))
        fuzzy_scores.append(best)

    row_text = token_containment_score(
        " ".join(str(v) for _, v in gt_leaves if v is not None),
        " ".join(str(v) for _, v in pred_leaves if v is not None),
    )
    return _combined_row_similarity(direct_scores, fuzzy_scores, row_text, pred_leaves, gt_leaves)


def prepare_row(index: int, row: Any) -> PreparedRow | None:
    if not isinstance(row, dict):
        return None
    leaves = iter_leaf_values(row)
    scalar = {key: value for key, value in leaves if not isinstance(value, (dict, list))}
    canonical_by_key: dict[str, str] = {}
    numeric_by_key: dict[str, float] = {}
    tokens_by_key: dict[str, set[str]] = {}
    informative_features_set: set[tuple[str, str]] = set()
    distinctive_values_list: list[str] = []
    for key, value in scalar.items():
        text = canonical_text(value)
        if text:
            canonical_by_key[key] = text
            tokens = tokens_for(text)
            tokens_by_key[key] = tokens
            leaf = field_name(key).casefold()
            include = False
            if looks_like_identity_key(key) or "reference" in leaf:
                include = True
            elif "date" in leaf and re.search(r"\d", text):
                include = True
            elif len(text) >= INFORMATIVE_TEXT_MIN_CHARS and len(tokens) >= INFORMATIVE_TEXT_MIN_TOKENS:
                include = True
            if include:
                informative_features_set.add((key, text))
            if len(text) >= DISTINCTIVE_VALUE_MIN_CHARS and len(tokens) >= DISTINCTIVE_VALUE_MIN_TOKENS:
                distinctive_values_list.append(text)
        number = parse_single_number(value)
        if number is not None:
            numeric_by_key[key] = number
    row_text = " ".join(str(v) for _, v in leaves if v is not None)
    return PreparedRow(
        index=index,
        row=row,
        leaves=leaves,
        scalar=scalar,
        canonical_by_key=canonical_by_key,
        numeric_by_key=numeric_by_key,
        tokens_by_key=tokens_by_key,
        informative_features=informative_features_set,
        distinctive_values=distinctive_values_list,
        row_text=row_text,
    )


def prepare_rows(rows: list[Any]) -> list[PreparedRow]:
    prepared: list[PreparedRow] = []
    for index, row in enumerate(rows):
        item = prepare_row(index, row)
        if item is not None:
            prepared.append(item)
    return prepared


def row_similarity_prepared(predicted_row: PreparedRow, gt_row: PreparedRow, *, schema: Any, parent_path: str) -> float:
    if not predicted_row.leaves or not gt_row.leaves:
        return 0.0

    direct_scores: list[float] = []
    gt_by_suffix = dict(gt_row.leaves)
    for suffix, pred_value in predicted_row.leaves:
        if pred_value is None:
            continue
        if suffix in gt_by_suffix:
            ok, score, _ = compare_values(
                gt_by_suffix[suffix],
                pred_value,
                schema=schema,
                predicted_path=f"{parent_path}[0].{suffix}",
                expected_path=f"{parent_path}[0].{suffix}",
            )
            direct_scores.append(1.0 if ok else score)

    fuzzy_scores: list[float] = []
    for _, pred_value in predicted_row.leaves:
        if pred_value is None:
            continue
        best = 0.0
        for _, gt_value in gt_row.leaves:
            if gt_value is None:
                continue
            ok, score, _ = compare_values(
                gt_value,
                pred_value,
                schema=schema,
                predicted_path=parent_path,
                expected_path=parent_path,
            )
            best = max(best, 1.0 if ok else score, token_containment_score(gt_value, pred_value))
        fuzzy_scores.append(best)

    row_text = token_containment_score(gt_row.row_text, predicted_row.row_text)
    return _combined_row_similarity(direct_scores, fuzzy_scores, row_text, predicted_row.leaves, gt_row.leaves)


def scalar_leaf_map(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    return {key: value for key, value in iter_leaf_values(obj) if not isinstance(value, (dict, list))}


def identity_keys_for_array(schema: Any, parent_path: str) -> list[str]:
    """Schema-declared identity keys (``data_schema.repeated_structure``);
    exact path preferred, terminal field name as fallback."""
    repeated = schema.get("repeated_structure") if isinstance(schema, dict) else None
    if not isinstance(repeated, dict):
        return []
    for candidate in (parent_path, path_without_indices(parent_path), field_name(parent_path)):
        spec = repeated.get(candidate)
        if not isinstance(spec, dict):
            continue
        keys = next((spec[s] for s in REPEATED_STRUCTURE_IDENTITY_SPELLINGS if spec.get(s)), None)
        if isinstance(keys, str):
            return [key.strip() for key in keys.split(",") if key.strip()]
        if isinstance(keys, list):
            return [str(key).strip() for key in keys if str(key).strip()]
    return []


def rule_identity_keys_for_array(rule_identity_keys: dict[str, list[str]] | None, parent_path: str) -> list[str]:
    """Identity keys declared via ``match_by:`` structural rule annotations
    (``rule_identity_keys``, keyed by rule field path)."""
    if not rule_identity_keys:
        return []
    for candidate in (parent_path, path_without_indices(parent_path), field_name(parent_path)):
        keys = rule_identity_keys.get(candidate)
        if keys:
            return list(keys)
    return []


def value_for_identity_key(row: Any, key: str) -> Any:
    if not isinstance(row, dict):
        return None
    cursor: Any = row
    for part in key.split("."):
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
    return cursor


def identity_tuple(row: Any, keys: list[str]) -> tuple[str, ...] | None:
    values = tuple(canonical_text(value_for_identity_key(row, key)) for key in keys)
    if not values or not any(values):
        return None
    return values


def build_declared_identity_alignment(
    predicted_list: list[Any],
    gt_list: list[Any],
    *,
    keys: list[str],
) -> tuple[ArrayAlignment, set[int], set[int]]:
    """Align rows by schema-declared identity keys; duplicates claim in list
    order, empty tuples fall through to later passes."""
    out: ArrayAlignment = {}
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    if not keys:
        return out, used_pred, used_gt

    pred_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for pred_index_num, predicted_row in enumerate(predicted_list):
        key = identity_tuple(predicted_row, keys)
        if key is not None:
            pred_index[key].append(pred_index_num)

    for gt_index, gt_row in enumerate(gt_list):
        key = identity_tuple(gt_row, keys)
        if key is None:
            continue
        for pred_index_num in pred_index.get(key, []):
            if pred_index_num in used_pred:
                continue
            out[pred_index_num] = (gt_index, 1.0, "declared_identity_array_row")
            used_pred.add(pred_index_num)
            used_gt.add(gt_index)
            break
    return out, used_pred, used_gt


def add_gated_rule_identity_alignment(
    out: ArrayAlignment,
    used_pred: set[int],
    used_gt: set[int],
    predicted_list: list[Any],
    gt_list: list[Any],
    *,
    keys: list[str],
) -> None:
    """Join rows on ``match_by:`` rule keys only when the key tuple is
    globally unique on BOTH sides and no shared non-empty scalar contradicts.

    Rule keys are often a single repeatable column; the duplicate-claiming
    tuple pass mis-lots repeat-key rows, so these make only unambiguous joins
    and everything else falls through to the content passes.
    """
    if not keys:
        return
    pred_by_tuple: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for pred_index, predicted_row in enumerate(predicted_list):
        key = identity_tuple(predicted_row, keys)
        if key is not None:
            pred_by_tuple[key].append(pred_index)
    gt_by_tuple: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for gt_index, gt_row in enumerate(gt_list):
        key = identity_tuple(gt_row, keys)
        if key is not None:
            gt_by_tuple[key].append(gt_index)
    for key, pred_hits in pred_by_tuple.items():
        gt_hits = gt_by_tuple.get(key, [])
        if len(pred_hits) != 1 or len(gt_hits) != 1:
            continue
        pred_index, gt_index = pred_hits[0], gt_hits[0]
        if pred_index in used_pred or gt_index in used_gt:
            continue
        pred_map = scalar_leaf_map(predicted_list[pred_index])
        gt_map = scalar_leaf_map(gt_list[gt_index])
        conflict = False
        for shared_key in set(pred_map) & set(gt_map):
            pred_text = canonical_text(pred_map[shared_key])
            gt_text = canonical_text(gt_map[shared_key])
            if pred_text and gt_text and pred_text != gt_text:
                conflict = True
                break
        if conflict:
            continue
        out[pred_index] = (gt_index, 1.0, "rule_identity_array_row")
        used_pred.add(pred_index)
        used_gt.add(gt_index)


def looks_like_identity_key(key: str) -> bool:
    leaf = field_name(key).casefold()
    if leaf in {
        "description",
        "value",
        "amount",
        "date",
        "type",
        "class",
        "term",
        "verbatim_term",
    }:
        return False
    if leaf in {
        "id",
        "identifier",
        "number",
        "line_number",
        "fact_id",
        "event_id",
        "claim_number",
        "check_number",
        "account_number",
        "patient_account_number",
        "control_number",
        "cusip",
        "isin",
        "sedol",
        "figi",
        "vin",
        "code",
        "name",
        "name_of_issuer",
    }:
        return True
    return (
        leaf.endswith("_id")
        or leaf.endswith("_number")
        or leaf.endswith("_no")
        or leaf.endswith("_code")
        or leaf.endswith("_name")
        or leaf.endswith("_reference")
    )


def best_inferred_identity_key(predicted_list: list[Any], gt_list: list[Any]) -> str | None:
    gt_values_by_key: dict[str, list[str]] = defaultdict(list)
    pred_values_by_key: dict[str, list[str]] = defaultdict(list)
    for row in gt_list:
        for key, value in scalar_leaf_map(row).items():
            text = canonical_text(value)
            if text:
                gt_values_by_key[key].append(text)
    for row in predicted_list:
        for key, value in scalar_leaf_map(row).items():
            text = canonical_text(value)
            if text:
                pred_values_by_key[key].append(text)

    best_key = None
    best_score = 0.0
    for key in sorted(set(gt_values_by_key) & set(pred_values_by_key)):
        if not looks_like_identity_key(key):
            continue
        gt_values = gt_values_by_key[key]
        pred_values = pred_values_by_key[key]
        if not gt_values or not pred_values:
            continue
        gt_unique_ratio = len(set(gt_values)) / len(gt_values)
        pred_unique_ratio = len(set(pred_values)) / len(pred_values)
        if gt_unique_ratio < IDENTITY_UNIQUE_RATIO_MIN or pred_unique_ratio < IDENTITY_UNIQUE_RATIO_MIN:
            continue
        overlap = len(set(gt_values) & set(pred_values))
        min_observed = min(len(set(gt_values)), len(set(pred_values)))
        if overlap < max(IDENTITY_MIN_OVERLAP, math.ceil(min_observed * IDENTITY_MIN_OVERLAP_RATIO)):
            continue
        score = overlap * min(gt_unique_ratio, pred_unique_ratio)
        if score > best_score:
            best_key = key
            best_score = score
    return best_key


def add_inferred_identity_alignment(
    out: ArrayAlignment,
    used_pred: set[int],
    used_gt: set[int],
    predicted_list: list[Any],
    gt_list: list[Any],
) -> None:
    key = best_inferred_identity_key(predicted_list, gt_list)
    if not key:
        return
    index: dict[str, list[int]] = defaultdict(list)
    for gt_index, gt_row in enumerate(gt_list):
        if gt_index in used_gt:
            continue
        value = scalar_leaf_map(gt_row).get(key)
        if value is not None:
            index[canonical_text(value)].append(gt_index)
    for pred_index, predicted_row in enumerate(predicted_list):
        if pred_index in used_pred or not isinstance(predicted_row, dict):
            continue
        pred_value = scalar_leaf_map(predicted_row).get(key)
        if pred_value is None:
            continue
        for gt_index in index.get(canonical_text(pred_value), []):
            if gt_index not in used_gt:
                used_gt.add(gt_index)
                used_pred.add(pred_index)
                out[pred_index] = (gt_index, 1.0, "inferred_identity_array_row")
                break


def distinctive_row_values(row: Any) -> list[str]:
    values: list[str] = []
    for value in scalar_leaf_map(row).values():
        text = canonical_text(value)
        if len(text) >= DISTINCTIVE_VALUE_MIN_CHARS and len(tokens_for(text)) >= DISTINCTIVE_VALUE_MIN_TOKENS:
            values.append(text)
    return values


def informative_row_features(row: Any) -> set[tuple[str, str]]:
    features: set[tuple[str, str]] = set()
    for key, value in scalar_leaf_map(row).items():
        text = canonical_text(value)
        if not text:
            continue
        leaf = field_name(key).casefold()
        include = False
        if looks_like_identity_key(key) or "reference" in leaf:
            include = True
        elif "date" in leaf and re.search(r"\d", text):
            include = True
        elif len(text) >= INFORMATIVE_TEXT_MIN_CHARS and len(tokens_for(text)) >= INFORMATIVE_TEXT_MIN_TOKENS:
            include = True
        if include:
            features.add((key, text))
    return features


def add_distinctive_value_alignment(
    out: ArrayAlignment,
    used_pred: set[int],
    used_gt: set[int],
    predicted_list: list[Any],
    gt_list: list[Any],
) -> None:
    gt_occurrences: dict[str, list[int]] = defaultdict(list)
    pred_occurrences: dict[str, list[int]] = defaultdict(list)
    for gt_index, gt_row in enumerate(gt_list):
        if gt_index in used_gt:
            continue
        for value in distinctive_row_values(gt_row):
            gt_occurrences[value].append(gt_index)
    for pred_index, predicted_row in enumerate(predicted_list):
        if pred_index in used_pred:
            continue
        for value in distinctive_row_values(predicted_row):
            pred_occurrences[value].append(pred_index)

    candidates: list[tuple[int, int, int]] = []
    for value in sorted(set(gt_occurrences) & set(pred_occurrences)):
        gt_matches = gt_occurrences[value]
        pred_matches = pred_occurrences[value]
        if len(gt_matches) == 1 and len(pred_matches) == 1:
            candidates.append((len(value), pred_matches[0], gt_matches[0]))

    candidates.sort(reverse=True)
    for _, pred_index, gt_index in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        out[pred_index] = (gt_index, 1.0, "distinctive_value_array_row")


def add_composite_feature_alignment(
    out: ArrayAlignment,
    used_pred: set[int],
    used_gt: set[int],
    predicted_list: list[Any],
    gt_list: list[Any],
) -> None:
    gt_features = [informative_row_features(row) for row in gt_list]
    pred_features = [informative_row_features(row) for row in predicted_list]
    gt_occurrences = Counter(feature for features in gt_features for feature in features)
    pred_occurrences = Counter(feature for features in pred_features for feature in features)
    gt_by_feature: dict[tuple[str, str], list[int]] = defaultdict(list)
    for gt_index, features in enumerate(gt_features):
        if gt_index in used_gt:
            continue
        for feature in features:
            gt_by_feature[feature].append(gt_index)

    pair_features: dict[tuple[int, int], set[tuple[str, str]]] = defaultdict(set)
    for pred_index, features in enumerate(pred_features):
        if pred_index in used_pred:
            continue
        for feature in features:
            for gt_index in gt_by_feature.get(feature, []):
                if gt_index not in used_gt:
                    pair_features[(pred_index, gt_index)].add(feature)

    candidates: list[tuple[float, int, int, int]] = []
    for (pred_index, gt_index), features in pair_features.items():
        if len(features) < 2:
            continue
        score = sum(1.0 / math.sqrt(max(gt_occurrences[feature], pred_occurrences[feature])) for feature in features)
        if score >= COMPOSITE_FEATURE_SCORE_MIN:
            candidates.append((score, len(features), pred_index, gt_index))

    candidates.sort(reverse=True)
    for score, _, pred_index, gt_index in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        out[pred_index] = (gt_index, min(score, 1.0), "composite_feature_array_row")


def best_blocking_key(predicted_list: list[Any], gt_list: list[Any]) -> str | None:
    """Pick the scalar column best suited to *block* rows of a large keyless array.

    Unlike inferred identity this tolerates duplicates (block, then
    disambiguate within the block by row similarity), but needs decent
    coverage and distinctness so constant-ish columns cannot block.
    """
    pred_vals: dict[str, list[str]] = defaultdict(list)
    gt_vals: dict[str, list[str]] = defaultdict(list)
    for row in predicted_list:
        for key, value in scalar_leaf_map(row).items():
            text = canonical_text(value)
            if text:
                pred_vals[key].append(text)
    for row in gt_list:
        for key, value in scalar_leaf_map(row).items():
            text = canonical_text(value)
            if text:
                gt_vals[key].append(text)
    n_pred = max(len(predicted_list), 1)
    n_gt = max(len(gt_list), 1)
    best_key = None
    best_score = 0.0
    for key in set(pred_vals) & set(gt_vals):
        pv = pred_vals[key]
        gv = gt_vals[key]
        coverage = min(len(pv) / n_pred, len(gv) / n_gt)
        if coverage < BLOCKING_KEY_MIN_COVERAGE:
            continue
        distinct = min(len(set(pv)) / len(pv), len(set(gv)) / len(gv))
        # A weak key blocks rows that then mis-pair within the block; with no
        # discriminative field, leave the array to the multi-feature passes.
        if distinct < BLOCKING_KEY_MIN_DISTINCT:
            continue
        score = coverage * distinct * (BLOCKING_KEY_IDENTITY_BOOST if looks_like_identity_key(key) else 1.0)
        if score > best_score:
            best_key = key
            best_score = score
    return best_key


def add_blocked_fuzzy_alignment(
    out: ArrayAlignment,
    used_pred: set[int],
    used_gt: set[int],
    predicted_list: list[Any],
    gt_list: list[Any],
    *,
    schema: Any,
    parent_path: str,
) -> None:
    """Block rows by a discriminative (not necessarily unique) key, then align
    within each block by full-row similarity — recovers giant keyless arrays
    the identity passes leave unmatched."""
    key = best_blocking_key(predicted_list, gt_list)
    if not key:
        return
    gt_blocks: dict[str, list[int]] = defaultdict(list)
    for gt_index, gt_row in enumerate(gt_list):
        if gt_index in used_gt or not isinstance(gt_row, dict):
            continue
        value = canonical_text(scalar_leaf_map(gt_row).get(key))
        if value:
            gt_blocks[value].append(gt_index)
    pred_blocks: dict[str, list[int]] = defaultdict(list)
    for pred_index, pred_row in enumerate(predicted_list):
        if pred_index in used_pred or not isinstance(pred_row, dict):
            continue
        value = canonical_text(scalar_leaf_map(pred_row).get(key))
        if value:
            pred_blocks[value].append(pred_index)

    for value, pred_indexes in pred_blocks.items():
        gt_indexes = gt_blocks.get(value)
        if not gt_indexes:
            continue
        # Shared key: accept even when other fields differ (real field
        # errors), but only above BLOCK_MIN_SIMILARITY so an ambiguous
        # duplicate cannot marry a different entity.
        if len(pred_indexes) == 1 and len(gt_indexes) == 1:
            i, j = pred_indexes[0], gt_indexes[0]
            if i in used_pred or j in used_gt:
                continue
            score = row_similarity(predicted_list[i], gt_list[j], schema=schema, parent_path=parent_path)
            if score < BLOCK_MIN_SIMILARITY:
                continue
            used_pred.add(i)
            used_gt.add(j)
            out[i] = (j, score, "blocked_identity_array_row")
            continue
        if len(pred_indexes) * len(gt_indexes) > SPARSE_FEATURE_PAIR_LIMIT:
            continue  # weak key produced a large block; leave to later passes
        matrix = [
            [row_similarity(predicted_list[i], gt_list[j], schema=schema, parent_path=parent_path) for j in gt_indexes]
            for i in pred_indexes
        ]
        for a, b in optimal_assignment(matrix):
            i, j = pred_indexes[a], gt_indexes[b]
            if i in used_pred or j in used_gt:
                continue
            if matrix[a][b] < BLOCK_MIN_SIMILARITY:
                continue
            used_pred.add(i)
            used_gt.add(j)
            out[i] = (j, matrix[a][b], "blocked_identity_array_row")


def number_feature(value: float) -> str:
    return f"{value:.12g}"


def row_sparse_features(row: PreparedRow) -> set[tuple[str, str, str]]:
    features: set[tuple[str, str, str]] = set()
    for key, text in row.canonical_by_key.items():
        if len(text) >= 2:
            features.add(("exact", key, text))
        if looks_like_identity_key(key):
            features.add(("identity", key, text))
    for key, number in row.numeric_by_key.items():
        features.add(("number", key, number_feature(number)))
    for key, text in row.informative_features:
        features.add(("informative", key, text))
    for text in row.distinctive_values:
        features.add(("distinctive", "", text))
    return features


def add_indexed_candidate_pairs(
    pairs: set[tuple[int, int]],
    pred_rows: list[PreparedRow],
    gt_rows: list[PreparedRow],
    used_pred: set[int],
    used_gt: set[int],
) -> None:
    gt_by_feature: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    pred_by_feature: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for gt_row in gt_rows:
        if gt_row.index in used_gt:
            continue
        for feature in row_sparse_features(gt_row):
            gt_by_feature[feature].append(gt_row.index)
    for pred_row in pred_rows:
        if pred_row.index in used_pred:
            continue
        for feature in row_sparse_features(pred_row):
            pred_by_feature[feature].append(pred_row.index)

    for feature in set(gt_by_feature) & set(pred_by_feature):
        pred_indexes = pred_by_feature[feature]
        gt_indexes = gt_by_feature[feature]
        if (
            len(pred_indexes) > SPARSE_FEATURE_OCCURRENCE_LIMIT
            or len(gt_indexes) > SPARSE_FEATURE_OCCURRENCE_LIMIT
            or len(pred_indexes) * len(gt_indexes) > SPARSE_FEATURE_PAIR_LIMIT
        ):
            continue
        for pred_index in pred_indexes:
            for gt_index in gt_indexes:
                if pred_index not in used_pred and gt_index not in used_gt:
                    pairs.add((pred_index, gt_index))


def add_block_candidate_pairs(
    pairs: set[tuple[int, int]],
    pred_rows: list[PreparedRow],
    gt_rows: list[PreparedRow],
    used_pred: set[int],
    used_gt: set[int],
) -> None:
    key = best_blocking_key([row.row for row in pred_rows], [row.row for row in gt_rows])
    if not key:
        return
    gt_blocks: dict[str, list[int]] = defaultdict(list)
    pred_blocks: dict[str, list[int]] = defaultdict(list)
    for gt_row in gt_rows:
        if gt_row.index in used_gt:
            continue
        value = gt_row.canonical_by_key.get(key)
        if value:
            gt_blocks[value].append(gt_row.index)
    for pred_row in pred_rows:
        if pred_row.index in used_pred:
            continue
        value = pred_row.canonical_by_key.get(key)
        if value:
            pred_blocks[value].append(pred_row.index)
    for value, pred_indexes in pred_blocks.items():
        gt_indexes = gt_blocks.get(value)
        if not gt_indexes or len(pred_indexes) * len(gt_indexes) > SPARSE_FEATURE_PAIR_LIMIT:
            continue
        for pred_index in pred_indexes:
            for gt_index in gt_indexes:
                if pred_index not in used_pred and gt_index not in used_gt:
                    pairs.add((pred_index, gt_index))


def add_index_window_candidate_pairs(
    pairs: set[tuple[int, int]],
    pred_rows: list[PreparedRow],
    gt_rows: list[PreparedRow],
    used_pred: set[int],
    used_gt: set[int],
) -> None:
    if max(len(pred_rows), len(gt_rows)) > SPARSE_INDEX_WINDOW_MAX_ROWS:
        return
    gt_indexes = {row.index for row in gt_rows if row.index not in used_gt}
    for pred_row in pred_rows:
        if pred_row.index in used_pred:
            continue
        for delta in range(-SPARSE_INDEX_WINDOW, SPARSE_INDEX_WINDOW + 1):
            gt_index = pred_row.index + delta
            if gt_index in gt_indexes:
                pairs.add((pred_row.index, gt_index))


def generate_sparse_candidate_pairs(
    pred_rows: list[PreparedRow],
    gt_rows: list[PreparedRow],
    used_pred: set[int],
    used_gt: set[int],
    *,
    include_all_pairs: bool,
) -> set[tuple[int, int]]:
    remaining_pred = [row.index for row in pred_rows if row.index not in used_pred]
    remaining_gt = [row.index for row in gt_rows if row.index not in used_gt]
    if include_all_pairs:
        return {(pred_index, gt_index) for pred_index in remaining_pred for gt_index in remaining_gt}

    pairs: set[tuple[int, int]] = set()
    add_indexed_candidate_pairs(pairs, pred_rows, gt_rows, used_pred, used_gt)
    add_block_candidate_pairs(pairs, pred_rows, gt_rows, used_pred, used_gt)
    add_index_window_candidate_pairs(pairs, pred_rows, gt_rows, used_pred, used_gt)
    return pairs


def connected_components(
    scored_pairs: dict[tuple[int, int], float],
) -> list[tuple[set[int], set[int], list[tuple[int, int]]]]:
    pred_to_gt: dict[int, set[int]] = defaultdict(set)
    gt_to_pred: dict[int, set[int]] = defaultdict(set)
    for pred_index, gt_index in scored_pairs:
        pred_to_gt[pred_index].add(gt_index)
        gt_to_pred[gt_index].add(pred_index)

    components: list[tuple[set[int], set[int], list[tuple[int, int]]]] = []
    seen_pred: set[int] = set()
    seen_gt: set[int] = set()
    for start_pred in sorted(pred_to_gt):
        if start_pred in seen_pred:
            continue
        comp_pred: set[int] = set()
        comp_gt: set[int] = set()
        queue: list[tuple[str, int]] = [("p", start_pred)]
        while queue:
            kind, index = queue.pop()
            if kind == "p":
                if index in seen_pred:
                    continue
                seen_pred.add(index)
                comp_pred.add(index)
                for gt_index in pred_to_gt.get(index, set()):
                    if gt_index not in seen_gt:
                        queue.append(("g", gt_index))
            else:
                if index in seen_gt:
                    continue
                seen_gt.add(index)
                comp_gt.add(index)
                for pred_index in gt_to_pred.get(index, set()):
                    if pred_index not in seen_pred:
                        queue.append(("p", pred_index))
        comp_pairs = [
            (pred_index, gt_index)
            for pred_index in comp_pred
            for gt_index in pred_to_gt.get(pred_index, set())
            if gt_index in comp_gt
        ]
        components.append((comp_pred, comp_gt, comp_pairs))
    return components


def optimal_assignment(score_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """(row, col) pairs maximizing total similarity: optimal matching (scipy)
    so a locally-high pairing cannot steal a row another prediction needs
    more; greedy fallback without scipy."""
    if not score_matrix or not score_matrix[0]:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        matrix = np.array(score_matrix, dtype=float)
        rows, cols = linear_sum_assignment(-matrix)
        return list(zip(rows.tolist(), cols.tolist(), strict=True))
    except ImportError:
        ranked = sorted(
            ((score_matrix[a][b], a, b) for a in range(len(score_matrix)) for b in range(len(score_matrix[0]))),
            reverse=True,
        )
        used_a: set[int] = set()
        used_b: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for _, a, b in ranked:
            if a in used_a or b in used_b:
                continue
            used_a.add(a)
            used_b.add(b)
            pairs.append((a, b))
        return pairs


def assign_sparse_candidate_pairs(
    out: ArrayAlignment,
    used_pred: set[int],
    used_gt: set[int],
    pred_rows: list[PreparedRow],
    gt_rows: list[PreparedRow],
    candidate_pairs: set[tuple[int, int]],
    *,
    schema: Any,
    parent_path: str,
    mode: str,
) -> tuple[int, int, int]:
    pred_by_index = {row.index: row for row in pred_rows}
    gt_by_index = {row.index: row for row in gt_rows}
    scored_pairs: dict[tuple[int, int], float] = {}
    scored_count = 0
    for pred_index, gt_index in sorted(candidate_pairs):
        if pred_index in used_pred or gt_index in used_gt:
            continue
        pred_row = pred_by_index.get(pred_index)
        gt_row = gt_by_index.get(gt_index)
        if pred_row is None or gt_row is None:
            continue
        scored_count += 1
        score = row_similarity_prepared(pred_row, gt_row, schema=schema, parent_path=parent_path)
        if score >= FUZZY_ROW_SIMILARITY_MIN:
            scored_pairs[(pred_index, gt_index)] = score

    largest_component_pairs = 0
    for pred_set, gt_set, comp_pairs in connected_components(scored_pairs):
        largest_component_pairs = max(largest_component_pairs, len(comp_pairs))
        pred_indexes = sorted(pred_set)
        gt_indexes = sorted(gt_set)
        pred_pos = {pred_index: pos for pos, pred_index in enumerate(pred_indexes)}
        gt_pos = {gt_index: pos for pos, gt_index in enumerate(gt_indexes)}
        matrix = [[0.0 for _ in gt_indexes] for _ in pred_indexes]
        for pred_index, gt_index in comp_pairs:
            matrix[pred_pos[pred_index]][gt_pos[gt_index]] = scored_pairs[(pred_index, gt_index)]
        for a, b in optimal_assignment(matrix):
            pred_index = pred_indexes[a]
            gt_index = gt_indexes[b]
            if pred_index in used_pred or gt_index in used_gt:
                continue
            score = matrix[a][b]
            if score < FUZZY_ROW_SIMILARITY_MIN:
                continue
            used_pred.add(pred_index)
            used_gt.add(gt_index)
            out[pred_index] = (gt_index, score, mode)
    return scored_count, len(scored_pairs), largest_component_pairs


def build_sparse_array_alignment(
    predicted_list: list[Any],
    gt_list: list[Any],
    *,
    schema: Any,
    parent_path: str,
    include_all_pairs: bool,
    rule_identity_keys: dict[str, list[str]] | None = None,
) -> ArrayAlignment:
    started = time.perf_counter()
    predicted_row_count = len(predicted_list)
    gt_row_count = len(gt_list)
    possible_pairs = predicted_row_count * gt_row_count
    declared_keys = identity_keys_for_array(schema, parent_path)
    out, used_pred, used_gt = build_declared_identity_alignment(
        predicted_list,
        gt_list,
        keys=declared_keys,
    )
    if not declared_keys:
        add_gated_rule_identity_alignment(
            out,
            used_pred,
            used_gt,
            predicted_list,
            gt_list,
            keys=rule_identity_keys_for_array(rule_identity_keys, parent_path),
        )

    if not include_all_pairs:
        add_inferred_identity_alignment(out, used_pred, used_gt, predicted_list, gt_list)
        add_distinctive_value_alignment(out, used_pred, used_gt, predicted_list, gt_list)
        add_composite_feature_alignment(out, used_pred, used_gt, predicted_list, gt_list)
        # Blocked runs LAST so it cannot perturb the pool the composite/
        # distinctive passes rely on; sparse fuzzy candidates mop up the rest.
        add_blocked_fuzzy_alignment(
            out, used_pred, used_gt, predicted_list, gt_list, schema=schema, parent_path=parent_path
        )

    pred_rows = prepare_rows(predicted_list)
    gt_rows_prepared = prepare_rows(gt_list)
    candidate_pairs = generate_sparse_candidate_pairs(
        pred_rows,
        gt_rows_prepared,
        used_pred,
        used_gt,
        include_all_pairs=include_all_pairs,
    )
    scored_count, eligible_count, largest_component_pairs = assign_sparse_candidate_pairs(
        out,
        used_pred,
        used_gt,
        pred_rows,
        gt_rows_prepared,
        candidate_pairs,
        schema=schema,
        parent_path=parent_path,
        mode=FUZZY_SIMILARITY_MODE if include_all_pairs else SPARSE_FUZZY_MODE,
    )
    if logger.isEnabledFor(logging.DEBUG):
        modes = Counter(mode for _, _, mode in out.values())
        logger.debug(
            "sparse-align path=%s rows=%dx%d all_pairs=%d possible=%d candidates=%d scored=%d "
            "eligible=%d largest_component_pairs=%d matched=%d modes=%s elapsed_s=%.4f",
            parent_path,
            predicted_row_count,
            gt_row_count,
            int(include_all_pairs),
            possible_pairs,
            len(candidate_pairs),
            scored_count,
            eligible_count,
            largest_component_pairs,
            len(out),
            dict(sorted(modes.items())),
            time.perf_counter() - started,
        )
    return out


def build_array_alignment(
    predicted_list: Any,
    gt_list: Any,
    *,
    schema: Any,
    parent_path: str,
    rule_identity_keys: dict[str, list[str]] | None = None,
) -> ArrayAlignment:
    if not isinstance(predicted_list, list) or not isinstance(gt_list, list):
        return {}
    if not predicted_list or not gt_list:
        return {}
    if not all(isinstance(item, dict) for item in predicted_list if item is not None):
        return {}
    if not all(isinstance(item, dict) for item in gt_list if item is not None):
        return {}

    possible_pairs = len(predicted_list) * len(gt_list)
    total_pred_leaves = sum(len(iter_leaf_values(row)) for row in predicted_list if isinstance(row, dict))
    total_gt_leaves = sum(len(iter_leaf_values(row)) for row in gt_list if isinstance(row, dict))
    include_all_pairs = (
        possible_pairs <= MAX_FUZZY_ARRAY_PAIRS and total_pred_leaves * total_gt_leaves <= MAX_FUZZY_ARRAY_WORK
    )
    alignment = build_sparse_array_alignment(
        predicted_list,
        gt_list,
        schema=schema,
        parent_path=parent_path,
        include_all_pairs=include_all_pairs,
        rule_identity_keys=rule_identity_keys,
    )
    if len(alignment) >= ADAPTIVE_FLOOR_MIN_ALIGNMENTS:
        floor = ADAPTIVE_FLOOR_MEDIAN_FRACTION * statistics.median(score for (_, score, _) in alignment.values())
        alignment = {
            pred_index: (gt_index, score, mode)
            for pred_index, (gt_index, score, mode) in alignment.items()
            if mode not in _ADAPTIVE_FLOOR_MODES or score >= floor
        }
    return alignment


def get_array_alignment_pair(
    *,
    predicted_doc: Any,
    gt_doc: Any,
    schema: Any,
    pred_parent_tokens: list[str | int],
    gt_parent_tokens: list[str | int],
    cache: dict[str, ArrayAlignment],
    rule_identity_keys: dict[str, list[str]] | None = None,
) -> ArrayAlignment:
    """Alignment between a predicted array and a (possibly remapped) GT array.

    Unlike ``get_array_alignment`` the two sides may live at different paths:
    outer object-array indices on the GT side are routed through their own row
    alignments first (nested arrays)."""
    key = f"{format_path(pred_parent_tokens)}=>{format_path(gt_parent_tokens)}"
    if key in cache:
        return cache[key]
    ok_pred, pred_list = value_at(predicted_doc, pred_parent_tokens)
    ok_gt, gt_list = value_at(gt_doc, gt_parent_tokens)
    cache[key] = build_array_alignment(
        pred_list if ok_pred else None,
        gt_list if ok_gt else None,
        schema=schema,
        parent_path=format_path(pred_parent_tokens),
        rule_identity_keys=rule_identity_keys,
    )
    return cache[key]
