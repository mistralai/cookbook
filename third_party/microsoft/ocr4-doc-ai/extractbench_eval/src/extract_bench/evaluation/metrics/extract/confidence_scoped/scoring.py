from __future__ import annotations

import logging
import math
from typing import Any, cast

from extract_bench.evaluation.metrics.field_grounding.evidence_comparator import parse_match_by_keys

from .array_alignment import get_array_alignment_pair
from .gt_normalization import lift_repeated_structure_gt, lift_repeated_structure_schema
from .paths import (
    field_name,
    format_path,
    iter_leaf_values,
    parent_path_for_array_item,
    path_tokens,
    path_without_indices,
    schema_description,
    schema_path_exists,
    value_at,
)
from .types import ConfidenceFieldRow, EmittedField, EvaluationContext, MatchResult
from .value_matching import (
    canonical_text,
    compare_values,
    is_effectively_null,
    looks_temporal,
    numeric_tokens_for,
    token_containment_score,
    tokens_for,
    value_supported_by_citation,
)

logger = logging.getLogger(__name__)

# Token-containment level at which a cross-field candidate counts as a match.
CROSS_FIELD_TOKEN_CONTAINMENT_MIN = 0.95
# Cross-field containment applies temporal, token-mass, and numeric-token-
# equality guards (a subset of the in-path containment gates).
CROSS_FIELD_FORWARD_MIN_TOKENS = 3
CROSS_FIELD_REVERSE_MIN_TOKENS = 4
# Alignment score above which a matched parent row is treated as GT-supported.
ALIGNMENT_SUPPORT_MIN = 0.25
# Sibling-bleed guard: the containment surplus must be at least this many
# tokens, and the sibling value it reproduces at least this many.
SIBLING_BLEED_MIN_SURPLUS_TOKENS = 2
SIBLING_BLEED_MIN_SIBLING_TOKENS = 2
# Cross-field GT matching is a guarded fallback for generic "section" schemas
# where extractors legitimately place the same text under sibling fields;
# only these roots/leaf names ever attempt it.
CROSS_FIELD_SECTION_ROOTS = frozenset(
    {
        "other",
        "certifications",
        "certificationsAndAwards",
        "awards",
        "licenses",
    }
)
CROSS_FIELD_SECTION_FIELD_NAMES = frozenset(
    {"content", "description", "organization", "date", "sectionTitle", "title", "name"}
)

# Extra / duplicated / fragment prediction row: no GT row, no fallback credit.
_UNMATCHED_ROW = MatchResult(False, "", None, "unmatched_row", "unmatched_prediction", 0.0)


def clamp_confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        return None
    return max(0.0, min(1.0, v))


def citation_text(meta: Any) -> str:
    if not isinstance(meta, dict):
        return ""
    citation = meta.get("citation")
    if isinstance(citation, list) and citation and isinstance(citation[0], dict):
        return str(citation[0].get("matching_text") or "")
    return ""


def flatten_scalar_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    return [(path, leaf) for path, leaf in iter_leaf_values(value, prefix) if leaf is not None]


def best_cross_field_match(
    predicted: Any,
    gt_doc: Any,
    *,
    schema: Any,
    predicted_path: str,
) -> tuple[str, Any, bool, float, str]:
    best_path = ""
    best_value = None
    best_score = 0.0
    best_mode = "missing"
    for gt_path, gt_value in flatten_scalar_values(gt_doc):
        ok, score, mode = compare_values(
            gt_value,
            predicted,
            schema=schema,
            predicted_path=predicted_path,
            expected_path=gt_path,
        )
        containment = 0.0
        if not (looks_temporal(gt_value) or looks_temporal(predicted)) and numeric_tokens_for(
            gt_value
        ) == numeric_tokens_for(predicted):
            forward = token_containment_score(gt_value, predicted)
            if forward >= CROSS_FIELD_TOKEN_CONTAINMENT_MIN and len(tokens_for(gt_value)) >= (
                CROSS_FIELD_FORWARD_MIN_TOKENS
            ):
                containment = forward
            reverse = token_containment_score(predicted, gt_value)
            if reverse >= CROSS_FIELD_TOKEN_CONTAINMENT_MIN and len(tokens_for(predicted)) >= (
                CROSS_FIELD_REVERSE_MIN_TOKENS
            ):
                containment = max(containment, reverse)
        weighted_score = 1.0 if ok else max(score, containment)
        if weighted_score > best_score:
            best_path, best_value, best_score, best_mode = gt_path, gt_value, weighted_score, mode
        if ok:
            return gt_path, gt_value, True, 1.0, f"cross_field_{mode}"
    if best_score >= CROSS_FIELD_TOKEN_CONTAINMENT_MIN:
        return best_path, best_value, True, best_score, "cross_field_token_containment"
    return best_path, best_value, False, best_score, best_mode


def containment_sibling_bleed(predicted: Any, expected: Any, gt_row: Any, matched_suffix: str) -> bool:
    """True when a superset containment pass owes its surplus tokens to a
    sibling field's GT value: the prediction swallowed an adjacent field."""
    if not isinstance(gt_row, dict):
        return False
    surplus = tokens_for(predicted) - tokens_for(expected)
    if len(surplus) < SIBLING_BLEED_MIN_SURPLUS_TOKENS:
        return False
    for key, value in iter_leaf_values(gt_row):
        if key == matched_suffix or value is None:
            continue
        sibling_tokens = tokens_for(value)
        if len(sibling_tokens) >= SIBLING_BLEED_MIN_SIBLING_TOKENS and sibling_tokens <= surplus:
            return True
    return False


def containment_ambiguous_across_rows(predicted: Any, gt_parent: list[Any], gt_index: int, suffix: str) -> bool:
    """A pred-subset containment pass is unearned when other rows' values for
    the same column also contain every predicted token — the truncation
    destroyed the discriminating tokens."""
    predicted_tokens = tokens_for(predicted)
    if not predicted_tokens:
        return False
    suffix_tokens = path_tokens(suffix) if suffix else []
    matched_tokens: set[str] | None = None
    if 0 <= gt_index < len(gt_parent):
        ok_matched, matched_value = value_at(gt_parent[gt_index], suffix_tokens)
        if ok_matched and matched_value is not None:
            # Exact equality with the matched row's own value (or an element
            # of it) fully identifies the value: never ambiguous.
            predicted_canonical = canonical_text(predicted)
            if predicted_canonical:
                candidates = matched_value if isinstance(matched_value, list) else [matched_value]
                for candidate in candidates:
                    if not isinstance(candidate, (dict, list)) and canonical_text(candidate) == predicted_canonical:
                        return False
            matched_tokens = tokens_for(matched_value)
    for index, row in enumerate(gt_parent):
        if index == gt_index:
            continue
        ok, value = value_at(row, suffix_tokens)
        if not ok or value is None or isinstance(value, (dict, list)):
            continue
        value_tokens = tokens_for(value)
        if value_tokens == matched_tokens:
            # An identical value on another row loses no discrimination.
            continue
        if predicted_tokens <= value_tokens:
            return True
    return False


def continuation_fold_match(
    field_path: str,
    predicted_value: Any,
    predicted_doc: Any,
    gt_parent: Any,
    gt_index: int,
) -> str | None:
    """Fold equivalence across continuation siblings (name/name_continuation,
    line1/line2), either direction; credit only on whole-string canonical
    equality of the joined parts."""
    leaf = field_name(field_path)
    parent_tokens = path_tokens(field_path)[:-1]
    ok, pred_row = value_at(predicted_doc, parent_tokens)
    if not ok or not isinstance(pred_row, dict):
        return None
    gt_row = gt_parent[gt_index] if isinstance(gt_parent, list) and 0 <= gt_index < len(gt_parent) else None
    if not isinstance(gt_row, dict):
        return None
    pairs: list[tuple[str, str]] = [(leaf, f"{leaf}_continuation")]
    if leaf.endswith("_continuation"):
        pairs.append((leaf[: -len("_continuation")], leaf))
    if leaf and leaf[-1] == "1":
        pairs.append((leaf, f"{leaf[:-1]}2"))
    if leaf and leaf[-1] == "2":
        pairs.append((f"{leaf[:-1]}1", leaf))
    for base, cont in pairs:
        pred_base = pred_row.get(base)
        pred_cont = pred_row.get(cont)
        gt_base = gt_row.get(base)
        gt_cont = gt_row.get(cont)
        # Prediction splits what GT folds into the base field.
        if (
            isinstance(pred_base, str)
            and pred_base.strip()
            and isinstance(pred_cont, str)
            and pred_cont.strip()
            and gt_base is not None
            and (gt_cont is None or (isinstance(gt_cont, str) and not gt_cont.strip()))
            and canonical_text(f"{pred_base} {pred_cont}") == canonical_text(gt_base)
        ):
            return "continuation_fold_equivalent"
        # Prediction folds what GT splits.
        if (
            leaf == base
            and isinstance(predicted_value, str)
            and predicted_value.strip()
            and gt_base is not None
            and isinstance(gt_cont, str)
            and gt_cont.strip()
            and canonical_text(predicted_value) == canonical_text(f"{gt_base} {gt_cont}")
        ):
            return "continuation_fold_equivalent"
    return None


def category_allowed_by_schema(schema: Any, path: str, predicted: Any) -> bool:
    if field_name(path) != "category" or predicted is None:
        return False
    desc = schema_description(schema, path).casefold()
    if "only return this if" in desc:
        return False
    return canonical_text(predicted) in canonical_text(desc)


def should_attempt_cross_field(path: str) -> bool:
    root = path.split(".", 1)[0].split("[", 1)[0]
    return root in CROSS_FIELD_SECTION_ROOTS and field_name(path) in CROSS_FIELD_SECTION_FIELD_NAMES


def matched_parent_has_gt_or_citation(cited_text: str, alignment_score: float) -> bool:
    if alignment_score >= ALIGNMENT_SUPPORT_MIN:
        return True
    return bool(cited_text)


def rule_identity_keys_by_path(field_rules: list[Any]) -> dict[str, list[str]]:
    """Row identity keys from ``match_by:`` structural rule annotations — the
    declaration channel for datasets without ``repeated_structure``."""
    out: dict[str, list[str]] = {}
    for rule in field_rules:
        structural = getattr(rule, "structural", None)
        field_path = getattr(rule, "field_path", None)
        if not isinstance(structural, str) or not structural.startswith("match_by:"):
            continue
        if not isinstance(field_path, str) or not field_path:
            continue
        keys = parse_match_by_keys(structural.split(":", 1)[1])
        if keys:
            out[field_path] = keys
    return out


def remap_gt_prefix_tokens(
    tokens: list[str | int],
    *,
    predicted_doc: Any,
    gt_doc: Any,
    schema: Any,
    context: EvaluationContext,
    rule_identity_keys: dict[str, list[str]] | None = None,
) -> tuple[list[str | int] | None, bool]:
    """Map predicted-path tokens to GT-side tokens, routing every object-array
    index through that array's row alignment (outer rows of nested arrays
    included).

    Returns ``(gt_tokens, any_alignment)``; ``gt_tokens`` is None when a
    governing alignment assigns the predicted row no GT row. Scalar-array
    indices and unaligned arrays keep the positional index."""
    gt_tokens: list[str | int] = []
    any_alignment = False
    for pos, tok in enumerate(tokens):
        if not isinstance(tok, int):
            gt_tokens.append(tok)
            continue
        ok_p, pred_list = value_at(predicted_doc, tokens[:pos])
        ok_g, gt_list = value_at(gt_doc, gt_tokens)
        is_object_array = (
            ok_p
            and ok_g
            and isinstance(pred_list, list)
            and isinstance(gt_list, list)
            and not (pred_list and all(not isinstance(x, dict) for x in pred_list if x is not None))
        )
        if not is_object_array:
            gt_tokens.append(tok)
            continue
        alignment = get_array_alignment_pair(
            predicted_doc=predicted_doc,
            gt_doc=gt_doc,
            schema=schema,
            pred_parent_tokens=list(tokens[:pos]),
            gt_parent_tokens=list(gt_tokens),
            cache=context.alignments,
            rule_identity_keys=rule_identity_keys,
        )
        if not alignment:
            gt_tokens.append(tok)
            continue
        any_alignment = True
        if tok not in alignment:
            return None, True
        gt_tokens.append(alignment[tok][0])
    return gt_tokens, any_alignment


def evaluate_field(
    field: EmittedField,
    *,
    predicted_doc: Any,
    gt_doc: Any,
    schema: Any,
    context: EvaluationContext,
    rule_identity_keys: dict[str, list[str]] | None = None,
) -> MatchResult:
    tokens = path_tokens(field.path)

    # GT-side path with every outer object-array index routed through its row
    # alignment.
    gt_full_tokens, path_has_alignment = remap_gt_prefix_tokens(
        tokens,
        predicted_doc=predicted_doc,
        gt_doc=gt_doc,
        schema=schema,
        context=context,
        rule_identity_keys=rule_identity_keys,
    )
    if gt_full_tokens is None:
        # Some governing row alignment assigns this predicted row no GT row:
        # extra / duplicated / fragment row. No positional fallback.
        return _UNMATCHED_ROW

    ok_exact, exact_gt = value_at(gt_doc, gt_full_tokens)
    if not ok_exact and gt_full_tokens:
        # A missing final key inside an existing object is semantically null.
        ok_parent, parent = value_at(gt_doc, gt_full_tokens[:-1])
        if ok_parent and isinstance(parent, dict) and isinstance(gt_full_tokens[-1], str):
            ok_exact = True
            exact_gt = None
    exact_gt_path = format_path(gt_full_tokens)

    array_info = parent_path_for_array_item(field.path)
    aligned_object_array = False
    gt_parent_tokens: list[str | int] | None = None
    gt_parent_path = ""
    ok_gt_parent = False
    gt_parent: Any = None
    ok_pred_parent = False
    pred_parent: Any = None
    if array_info is not None:
        parent_path, predicted_index, suffix = array_info
        gt_parent_tokens, _parent_has_alignment = remap_gt_prefix_tokens(
            path_tokens(parent_path),
            predicted_doc=predicted_doc,
            gt_doc=gt_doc,
            schema=schema,
            context=context,
            rule_identity_keys=rule_identity_keys,
        )
        if gt_parent_tokens is None:
            return _UNMATCHED_ROW
        gt_parent_path = format_path(gt_parent_tokens)
        ok_gt_parent, gt_parent = value_at(gt_doc, gt_parent_tokens)
        ok_pred_parent, pred_parent = value_at(predicted_doc, path_tokens(parent_path))
        if (
            ok_pred_parent
            and ok_gt_parent
            and isinstance(pred_parent, list)
            and isinstance(gt_parent, list)
            and not (pred_parent and all(not isinstance(item, dict) for item in pred_parent if item is not None))
        ):
            aligned_object_array = bool(
                get_array_alignment_pair(
                    predicted_doc=predicted_doc,
                    gt_doc=gt_doc,
                    schema=schema,
                    pred_parent_tokens=path_tokens(parent_path),
                    gt_parent_tokens=gt_parent_tokens,
                    cache=context.alignments,
                    rule_identity_keys=rule_identity_keys,
                )
            )

    # Exact-path shortcut, disabled whenever a row alignment governs the
    # path — the aligner's assignment is authoritative there.
    pre_pass_mode: str | None = None
    if ok_exact and not aligned_object_array and not path_has_alignment:
        correct, _score, mode = compare_values(
            exact_gt,
            field.value,
            schema=schema,
            predicted_path=field.path,
            expected_path=exact_gt_path,
        )
        if correct and mode == "token_containment" and len(gt_full_tokens) > 1:
            ok_row, gt_row = value_at(gt_doc, gt_full_tokens[:-1])
            if ok_row and containment_sibling_bleed(field.value, exact_gt, gt_row, str(gt_full_tokens[-1])):
                correct, mode = False, "token_containment_sibling_bleed"
        if correct:
            return MatchResult(True, exact_gt_path, exact_gt, mode, "exact_path", 1.0)
        pre_pass_mode = mode

    if array_info is not None:
        assert gt_parent_tokens is not None  # set above or returned unmatched_row
        if ok_gt_parent and isinstance(gt_parent, dict):
            # Shape projection: prediction is a scalar-array item, GT a grouped
            # object. GT leaves are claimed exclusively, and a nearest-miss is
            # recorded only when the pair shares a token.
            claimed_grouped = context.claimed_grouped.setdefault(gt_parent_path, set())
            best_path = ""
            best_value = None
            best_score = 0.0
            best_mode = "missing"
            matched_claimed = False
            pred_tokens = tokens_for(field.value)
            for rel_path, gt_value in flatten_scalar_values(gt_parent):
                gt_path = f"{gt_parent_path}.{rel_path}" if rel_path else gt_parent_path
                correct, score, mode = compare_values(
                    gt_value,
                    field.value,
                    schema=schema,
                    predicted_path=field.path,
                    expected_path=gt_path,
                )
                if gt_path in claimed_grouped:
                    if correct:
                        matched_claimed = True
                    continue
                if correct:
                    claimed_grouped.add(gt_path)
                    return MatchResult(
                        True, gt_path, gt_value, f"grouped_collection_{mode}", "grouped_gt_projection", 1.0
                    )
                if score > best_score and pred_tokens & tokens_for(gt_value):
                    best_path, best_value, best_score, best_mode = gt_path, gt_value, score, mode
            if matched_claimed:
                return MatchResult(False, "", None, "grouped_scalar_surplus", "grouped_gt_projection", 0.0)
            if best_score > 0:
                return MatchResult(False, best_path, best_value, best_mode, "grouped_gt_projection", best_score)

        if ok_pred_parent and ok_gt_parent and isinstance(pred_parent, list) and isinstance(gt_parent, list):
            if pred_parent and all(not isinstance(item, dict) for item in pred_parent if item is not None):
                # Scalar arrays are set-like against the ROW-ALIGNED GT array;
                # element claims are exclusive, so surplus copies beyond GT's
                # multiplicity score False.
                claimed_scalar = context.claimed_scalar.setdefault(gt_parent_path, set())
                unclaimed_indices = [i for i in range(len(gt_parent)) if i not in claimed_scalar]
                if not unclaimed_indices:
                    # Empty or fully-claimed GT array: surplus element; must
                    # not attach to a claimed element or book as "missing".
                    return MatchResult(False, "", None, "scalar_array_surplus", "scalar_array_membership", 0.0)
                best_path = ""
                best_value = None
                best_score = 0.0
                best_mode = "missing"
                matched_claimed = False
                for gt_index, gt_value in enumerate(gt_parent):
                    correct, score, mode = compare_values(
                        gt_value,
                        field.value,
                        schema=schema,
                        predicted_path=field.path,
                        expected_path=f"{gt_parent_path}[{gt_index}]",
                    )
                    if gt_index in claimed_scalar:
                        if correct:
                            matched_claimed = True
                        continue
                    weighted_score = 1.0 if correct else score
                    if weighted_score > best_score:
                        best_path = f"{gt_parent_path}[{gt_index}]"
                        best_value = gt_value
                        best_score = weighted_score
                        best_mode = mode
                    if correct:
                        claimed_scalar.add(gt_index)
                        return MatchResult(
                            True,
                            f"{gt_parent_path}[{gt_index}]",
                            gt_value,
                            f"scalar_array_{mode}",
                            "scalar_array_membership",
                            1.0,
                        )
                if matched_claimed:
                    # Equal to an already-claimed element only: surplus copy.
                    return MatchResult(False, "", None, "scalar_array_surplus", "scalar_array_membership", 0.0)
                return MatchResult(False, best_path, best_value, best_mode, "scalar_array_membership", best_score)

            alignment = get_array_alignment_pair(
                predicted_doc=predicted_doc,
                gt_doc=gt_doc,
                schema=schema,
                pred_parent_tokens=path_tokens(parent_path),
                gt_parent_tokens=gt_parent_tokens,
                cache=context.alignments,
                rule_identity_keys=rule_identity_keys,
            )
            if predicted_index in alignment:
                gt_index, alignment_score, alignment_mode = alignment[predicted_index]
                expected_path = f"{gt_parent_path}[{gt_index}].{suffix}" if suffix else f"{gt_parent_path}[{gt_index}]"
                ok_gt, gt_value = value_at(gt_doc, path_tokens(expected_path))
                if ok_gt:
                    correct, score, mode = compare_values(
                        gt_value,
                        field.value,
                        schema=schema,
                        predicted_path=field.path,
                        expected_path=expected_path,
                    )
                    if correct and suffix:
                        gt_row = gt_parent[gt_index] if 0 <= gt_index < len(gt_parent) else None
                        if mode == "token_containment" and containment_sibling_bleed(
                            field.value, gt_value, gt_row, suffix
                        ):
                            correct, score, mode = False, 0.0, "token_containment_sibling_bleed"
                        elif mode == "concise_string_contained_in_expected" and containment_ambiguous_across_rows(
                            field.value, gt_parent, gt_index, suffix
                        ):
                            correct, score, mode = False, 0.0, "containment_ambiguous_across_rows"
                    if correct:
                        return MatchResult(True, expected_path, gt_value, mode, alignment_mode, alignment_score)

                    if should_attempt_cross_field(field.path):
                        gt_path, cross_gt_value, cross_correct, cross_score, cross_mode = best_cross_field_match(
                            field.value,
                            gt_doc,
                            schema=schema,
                            predicted_path=field.path,
                        )
                        if cross_correct:
                            return MatchResult(
                                True,
                                gt_path,
                                cross_gt_value,
                                cross_mode,
                                f"{alignment_mode}_cross_field_gt_value",
                                max(alignment_score, cross_score),
                            )
                    fold_mode = continuation_fold_match(field.path, field.value, predicted_doc, gt_parent, gt_index)
                    if fold_mode:
                        return MatchResult(True, expected_path, gt_value, fold_mode, alignment_mode, alignment_score)
                    return MatchResult(False, expected_path, gt_value, mode, alignment_mode, alignment_score)

                if should_attempt_cross_field(field.path):
                    gt_path, cross_gt_value, cross_correct, cross_score, cross_mode = best_cross_field_match(
                        field.value,
                        gt_doc,
                        schema=schema,
                        predicted_path=field.path,
                    )
                    if cross_correct:
                        return MatchResult(
                            True,
                            gt_path,
                            cross_gt_value,
                            cross_mode,
                            f"{alignment_mode}_cross_field_gt_value",
                            max(alignment_score, cross_score),
                        )

                if field_name(field.path) == "sectionTitle" and value_supported_by_citation(
                    field.value, field.cited_text
                ):
                    return MatchResult(
                        True,
                        f"{gt_parent_path}[{gt_index}].{suffix}",
                        None,
                        "schema_extra_cited_section_title",
                        f"{alignment_mode}_extra_schema_field",
                        alignment_score,
                    )

                if category_allowed_by_schema(schema, field.path, field.value) and matched_parent_has_gt_or_citation(
                    field.cited_text, alignment_score
                ):
                    return MatchResult(
                        True,
                        f"{gt_parent_path}[{gt_index}].{suffix}",
                        None,
                        "schema_extra_inferred_category",
                        f"{alignment_mode}_extra_schema_field",
                        alignment_score,
                    )
            elif aligned_object_array:
                # Aligned array but this row got no GT row (extra/duplicate/
                # fragment): no positional fallback, no cross-field credit.
                return _UNMATCHED_ROW

    # Explicit canonicalization for location city/state duplicates such as
    # Dubai, UAE where GT duplicated city into state without a schema mandate.
    if field.path.endswith(".state") and field.value is None:
        base = field.path.rsplit(".", 1)[0]
        _, gt_state = value_at(gt_doc, path_tokens(field.path))
        _, gt_city = value_at(gt_doc, path_tokens(f"{base}.city"))
        _, pred_city = value_at(predicted_doc, path_tokens(f"{base}.city"))
        _, gt_country = value_at(gt_doc, path_tokens(f"{base}.country"))
        _, pred_country = value_at(predicted_doc, path_tokens(f"{base}.country"))
        if gt_state and gt_state == gt_city == pred_city and gt_country == pred_country:
            return MatchResult(
                True,
                field.path,
                gt_state,
                "city_state_duplicate_absent",
                "exact_path_canonicalized",
                1.0,
            )

    if not ok_exact and field.value is not None and should_attempt_cross_field(field.path):
        gt_path, gt_value, correct, score, mode = best_cross_field_match(
            field.value,
            gt_doc,
            schema=schema,
            predicted_path=field.path,
        )
        if correct:
            return MatchResult(True, gt_path, gt_value, mode, "cross_field_gt_value", score)
        if field_name(field.path) == "sectionTitle" and value_supported_by_citation(field.value, field.cited_text):
            return MatchResult(True, "", None, "schema_extra_cited_section_title", "extra_schema_field", 0.0)
        if category_allowed_by_schema(schema, field.path, field.value) and field.cited_text:
            return MatchResult(True, "", None, "schema_extra_inferred_category", "extra_schema_field", 0.0)
        return MatchResult(False, gt_path, gt_value, mode, "unmatched_prediction", score)

    if ok_exact and exact_gt is None and is_effectively_null(field.value):
        # Null prediction vs a GT row that omits the key entirely (sparse
        # claim schemas): both assert "no value".
        return MatchResult(True, exact_gt_path, None, "null_missing_key_equivalent", "exact_path", 1.0)

    expected_value = exact_gt if ok_exact else None
    fallback_mode = "missing"
    if ok_exact and field.value is not None and exact_gt is None:
        # Prediction asserts a value where GT is null: a spurious value, not
        # a GT-side miss — keep the direction visible for error breakdowns.
        fallback_mode = "spurious_value"
    elif ok_exact and pre_pass_mode is not None and field.value is not None and exact_gt is not None:
        # Both sides present and the pre-pass compared them: preserve the
        # informative comparator mode instead of the overloaded "missing".
        fallback_mode = pre_pass_mode
    # alignment_score reports ALIGNMENT quality, not value correctness: an
    # exact-path row whose value mismatches is still a perfectly-aligned row.
    return MatchResult(
        False,
        exact_gt_path if ok_exact else "",
        expected_value,
        fallback_mode,
        "exact_path" if ok_exact else "unmatched_prediction",
        1.0 if ok_exact else 0.0,
    )


def _citation_data(citation: Any) -> dict[str, Any]:
    if hasattr(citation, "model_dump"):
        data = citation.model_dump(mode="json")
        if isinstance(data, dict):
            return cast(dict[str, Any], data)
        return {}
    if isinstance(citation, dict):
        return cast(dict[str, Any], citation)
    return {
        "field_path": getattr(citation, "field_path", None),
        "confidence": getattr(citation, "confidence", None),
        "reference_text": getattr(citation, "reference_text", None),
        "metadata": getattr(citation, "metadata", None),
    }


def _citation_reference_text(data: dict[str, Any]) -> str:
    reference_text = data.get("reference_text")
    if isinstance(reference_text, str):
        return reference_text
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        for key in ("reference_text", "matching_text", "text"):
            value = metadata.get(key)
            if isinstance(value, str):
                return value
        citation = metadata.get("citation")
        if isinstance(citation, list) and citation and isinstance(citation[0], dict):
            return str(citation[0].get("matching_text") or "")
    return citation_text(data)


def _collect_emitted_fields_from_citations(extracted_data: Any, field_citations: list[Any]) -> list[EmittedField]:
    by_path: dict[str, EmittedField] = {}
    for citation in field_citations:
        data = _citation_data(citation)
        field_path = data.get("field_path")
        confidence = clamp_confidence(data.get("confidence"))
        if not isinstance(field_path, str) or not field_path or confidence is None:
            continue
        found, value = value_at(extracted_data, path_tokens(field_path))
        if not found:
            continue
        cited_text = _citation_reference_text(data)
        current = by_path.get(field_path)
        if current is None or confidence > current.confidence:
            by_path[field_path] = EmittedField(field_path, value, confidence, cited_text)
    return list(by_path.values())


def _set_path_value(root: Any, field_path: str, value: Any) -> Any:
    tokens = path_tokens(field_path)
    if not tokens:
        return root
    if root is None:
        root = [] if isinstance(tokens[0], int) else {}
    cursor = root
    for index, token in enumerate(tokens):
        is_last = index == len(tokens) - 1
        next_token = tokens[index + 1] if not is_last else None
        if isinstance(token, int):
            if not isinstance(cursor, list):
                logger.debug("Dropping rule value at %s: list index into non-list container", field_path)
                return root
            while len(cursor) <= token:
                cursor.append(None)
            if is_last:
                cursor[token] = value
                return root
            if cursor[token] is None:
                cursor[token] = [] if isinstance(next_token, int) else {}
            cursor = cursor[token]
            continue
        if not isinstance(cursor, dict):
            logger.debug("Dropping rule value at %s: key into non-dict container", field_path)
            return root
        if is_last:
            cursor[token] = value
            return root
        if token not in cursor or cursor[token] is None:
            cursor[token] = [] if isinstance(next_token, int) else {}
        cursor = cursor[token]
    return root


def _expected_value_from_rule(rule: Any) -> Any:
    evidence = getattr(rule, "evidence", None)
    if evidence:
        values = [getattr(entry, "value", None) for entry in evidence if getattr(entry, "value", None) is not None]
        if len(values) == 1:
            return values[0]
        if len(values) > 1:
            return values
    return getattr(rule, "expected_value", None)


def expected_doc_from_rules(field_rules: list[Any], skip_field_paths: set[str]) -> Any:
    root: Any = {}
    for rule in field_rules:
        field_path = getattr(rule, "field_path", None)
        if not isinstance(field_path, str) or field_path in skip_field_paths:
            continue
        root = _set_path_value(root, field_path, _expected_value_from_rule(rule))
    return root


def _verified_paths(field_rules: list[Any]) -> set[str]:
    return {
        rule.field_path
        for rule in field_rules
        if isinstance(getattr(rule, "field_path", None), str) and bool(getattr(rule, "verified", False))
    }


def _field_pattern_for_row(path: str) -> str:
    return path_without_indices(path) if path else ""


def build_confidence_field_rows(
    *,
    extracted_data: Any,
    field_rules: list[Any],
    field_citations: list[Any],
    data_schema: dict[str, Any] | None = None,
    skip_field_paths: set[str] | None = None,
    expected_output: Any | None = None,
) -> list[ConfidenceFieldRow]:
    """Score confidence-bearing emitted fields against GT.

    The row set is scoped to fields the extractor emitted WITH confidence
    metadata (not to GT rules), then each emitted value is matched against the
    ground-truth document by a fixed ladder: exact path, declared/inferred
    array-row alignment, then the guarded cross-field / schema-extra fallback
    modes. Every row records which rung matched (``comparison_mode`` /
    ``alignment_mode``) so downstream consumers can filter by match kind.
    """
    skip_set = skip_field_paths or set()
    gt_source = expected_output if expected_output is not None else expected_doc_from_rules(field_rules, skip_set)
    # Normalize lifted repeated structures on BOTH sides before any lookup:
    # extraction emits declared keys top-level while some GTs nest the same
    # rows under a parent array, and schema_valid must see the lifted paths.
    schema = lift_repeated_structure_schema(data_schema or {})
    gt_doc = lift_repeated_structure_gt(gt_source, schema)
    emitted = _collect_emitted_fields_from_citations(extracted_data, field_citations)
    verified_paths = _verified_paths(field_rules)
    rule_identity_keys = rule_identity_keys_by_path(field_rules)
    context = EvaluationContext()
    # Grant exclusive claims highest-confidence-first so the surplus False lands
    # on the low-confidence duplicate; ties and output rows keep citation order.
    scoring_order = sorted(range(len(emitted)), key=lambda index: -emitted[index].confidence)
    rows_by_emitted_index: dict[int, ConfidenceFieldRow] = {}
    for emitted_index in scoring_order:
        field = emitted[emitted_index]
        if field.path in skip_set:
            continue
        match = evaluate_field(
            field,
            predicted_doc=extracted_data,
            gt_doc=gt_doc,
            schema=schema,
            context=context,
            rule_identity_keys=rule_identity_keys,
        )
        field_path = match.expected_path or field.path
        # comparison_score: correctness first, alignment support otherwise; a
        # positionally-resolved wrong value carries no support even though its
        # diagnostic alignment_score reads 1.0.
        if match.correct:
            score = 1.0
        elif match.alignment_mode == "exact_path":
            score = 0.0
        else:
            score = match.alignment_score
        rows_by_emitted_index[emitted_index] = ConfidenceFieldRow(
            field_path=field_path,
            matched_gt_path=match.expected_path,
            field_pattern=_field_pattern_for_row(field_path),
            predicted_path=field.path,
            expected_value=match.expected_value,
            predicted_value=field.value,
            confidence=field.confidence,
            correct=match.correct,
            comparison_score=score,
            comparison_mode=match.comparison_mode,
            verified=field_path in verified_paths,
            alignment_mode=match.alignment_mode,
            alignment_score=match.alignment_score,
            cited_text=field.cited_text,
            schema_valid=True if not schema else schema_path_exists(schema, field.path),
        )
    return [rows_by_emitted_index[index] for index in sorted(rows_by_emitted_index)]
