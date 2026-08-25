"""Table Record Match Metric.

Evaluates semantic equivalence of tables by treating them as collections
of key-value record objects.  Each HTML table is converted to a list of
dicts, then predicted records are matched to GT records via the Hungarian
algorithm with graduated cell-level scoring.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from extract_bench.evaluation.metrics.parse.table_extraction import (
    ExtractedTable,
    extract_html_tables,
)
from extract_bench.evaluation.metrics.parse.table_parsing import (
    TableData,
    parse_html_tables,
)
from extract_bench.evaluation.metrics.parse.table_title_stripping import (
    HeaderHints,
    HeaderInfo,
    extract_header_info,
    strip_title_rows,
)
from extract_bench.evaluation.metrics.parse.utils import (
    _normalize_sub_sup_for_table as _shared_normalize_sub_sup,
)
from extract_bench.evaluation.metrics.parse.utils import normalize_text
from extract_bench.schemas.evaluation import MetricValue

logger = logging.getLogger(__name__)

# ===========================================================================
# Phase 1: Normalization
# ===========================================================================

SYNTHETIC_KEY_PREFIX = "col_"
KEY_SEPARATOR = " "

# Minimum header similarity to consider a column pair matched
_COLUMN_MATCH_THRESHOLD = 0.9

_EMPTY_SENTINELS = frozenset({"", "nan", "NaN", "NAN", "none", "None", "-", "—", "n/a", "N/A"})

# Sup/sub conversion is now defined once in utils.py and shared between
# GriTS normalize_cell_text and TRM normalize_table. The local name is
# preserved as a thin alias so the existing import path
# (`from ...table_record_match_metric import _normalize_sub_sup_for_table`)
# in tests and other modules continues to work.
_normalize_sub_sup_for_table = _shared_normalize_sub_sup


def _is_all_empty_record(record: dict[str, str]) -> bool:
    """Check if every value in a record is an empty string."""
    return all(v.strip() == "" for v in record.values())


def _is_empty_or_nan(value: str) -> bool:
    """Check if a value represents an empty/missing cell."""
    return str(value).strip() in _EMPTY_SENTINELS


def _normalize_trm_cell_text(text: str) -> str:
    """Extra cell-level normalization for table record match scoring.

    Applied only inside cell_score() — NOT in the shared normalize_text()
    which is used by TEDS, GriTS, header accuracy, and text similarity.

    Rules:
    1. Strip leader dots ("........" or ". . . .") — visual fillers whose
       exact count is meaningless and varies between GT and prediction.
    2. Strip whitespace adjacent to $, %, (, )
       - "$ 5,061" → "$5,061"
       - "50 %"    → "50%"
       - "word (n)" → "word(n)"
    3. Strip commas from numeric contexts (dollars, percentages, plain numbers)
       - "$5,061"  → "$5061"
       - "1,000%"  → "1000%"
       - "1,000,000" → "1000000"
       - But NOT "Smith, John" (commas between non-digit chars are preserved)
    """
    # --- leader dots (e.g. "Item ......... 5" or "Item . . . . 5") ---
    # These are visual fillers whose exact count is meaningless; strip them
    # so ground-truth and prediction don't diverge on dot count.
    text = re.sub(r"(?:\.\s){2,}\.?", "", text)  # ". . . ." style
    text = re.sub(r"\.{2,}", "", text)  # "........" style
    text = re.sub(r"  +", " ", text).strip()  # collapse leftover whitespace

    # --- whitespace around punctuation ---
    text = re.sub(r"\$\s+", "$", text)
    text = re.sub(r"\s+%", "%", text)
    text = re.sub(r"\s+\(", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    # Strip whitespace before the registered-trademark symbol so "Apple ® Inc"
    # and "Apple® Inc" compare equal.
    text = re.sub(r"\s+®", "®", text)

    # --- comma removal in numeric contexts ---
    # Remove commas that sit between digits (thousands separators).
    # This covers "$1,000", "1,000%", "(1,000)", and plain "1,000,000"
    # but leaves "Smith, John" untouched because the comma is not
    # between two digits.
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    return text


def normalize_table(table: TableData) -> TableData:
    """Normalize all cell text in a table for comparison.

    Applies normalize_text() + _normalize_trm_cell_text() to every cell in
    table.data and to col_headers text, producing a new TableData with
    normalized values throughout. Columns whose data cells AND header
    entries are all empty strings after normalization are dropped, and
    remaining column indices are renumbered to stay contiguous.
    """
    if table.data.size == 0:
        return table
    n_rows, n_cols = table.data.shape
    normalized = np.empty_like(table.data)
    for r in range(n_rows):
        for c in range(n_cols):
            val = str(table.data[r, c])
            val = _normalize_sub_sup_for_table(val)
            val = normalize_text(val)
            val = _normalize_trm_cell_text(val)
            normalized[r, c] = val
    # Normalize col_headers and row_headers metadata to match data cell normalization
    normalized_col_headers: dict[int, list[tuple[int, str]]] = {}
    for col_idx, entries in table.col_headers.items():
        normalized_col_headers[col_idx] = [
            (row_idx, _normalize_trm_cell_text(normalize_text(_normalize_sub_sup_for_table(text))))
            for row_idx, text in entries
        ]
    normalized_row_headers: dict[int, list[tuple[int, str]]] = {}
    for row_idx, entries in table.row_headers.items():
        normalized_row_headers[row_idx] = [
            (col_idx, _normalize_trm_cell_text(normalize_text(_normalize_sub_sup_for_table(text))))
            for col_idx, text in entries
        ]

    # Drop columns whose data cells AND header entries are all literally
    # empty strings after normalization. Such columns add noise to header
    # alignment and inflate the union denominator used by _record_similarity
    # without contributing any signal. Sentinel values like "-" or "n/a"
    # count as real content and are NOT dropped.
    #
    # This runs independently on GT and pred, so the two normalized tables
    # may end up with *different* column counts when (say) GT has an empty
    # column 2 but pred has real content there. That's intentional: record
    # matching is content-driven — ``_record_similarity`` compares columns
    # via the ``column_mapping`` keyed by header text, not by positional
    # index, so differing shapes are fine and the unmapped column on the
    # longer side penalizes the score via ``max(n_gt_cols, n_pred_cols)``.
    def _col_is_empty(c: int) -> bool:
        for r in range(n_rows):
            if str(normalized[r, c]) != "":
                return False
        for _row_idx, text in normalized_col_headers.get(c, []):
            if text != "":
                return False
        return True

    keep_cols = [c for c in range(n_cols) if not _col_is_empty(c)]
    if len(keep_cols) != n_cols:
        old_to_new = {old: new for new, old in enumerate(keep_cols)}
        normalized = normalized[:, keep_cols]
        normalized_col_headers = {
            old_to_new[c]: normalized_col_headers[c] for c in keep_cols if c in normalized_col_headers
        }
        header_cols = {old_to_new[c] for c in table.header_cols if c in old_to_new}
        header_cells = {(r, old_to_new[c]) for (r, c) in table.header_cells if c in old_to_new}
        normalized_row_headers = {
            row_idx: [(old_to_new[c], text) for c, text in entries if c in old_to_new]
            for row_idx, entries in normalized_row_headers.items()
        }
    else:
        header_cols = table.header_cols
        header_cells = table.header_cells

    return TableData(
        data=normalized,
        header_rows=table.header_rows,
        header_cols=header_cols,
        col_headers=normalized_col_headers,
        row_headers=normalized_row_headers,
        header_cells=header_cells,
        context_before=table.context_before,
        context_after=table.context_after,
    )


def cell_score(expected: str, actual: str) -> float:
    """Score similarity between two cell values: 1.0 for match, 0.0 otherwise.

    Assumes both values have already been normalized by normalize_table()
    (normalize_text + _normalize_trm_cell_text applied upfront).
    """
    if _is_empty_or_nan(expected) and _is_empty_or_nan(actual):
        return 1.0
    if _is_empty_or_nan(expected) or _is_empty_or_nan(actual):
        return 0.0
    return 1.0 if expected == actual else 0.0


# ===========================================================================
# Phase 2: Header Extraction & Column Alignment
# ===========================================================================


def _disambiguate_keys(raw_keys: list[str]) -> list[str]:
    """Append _1, _2, … suffixes to duplicate key strings."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for k in raw_keys:
        if k in seen:
            seen[k] += 1
            result.append(f"{k}_{seen[k]}")
        else:
            seen[k] = 0
            result.append(k)
    # Retroactively fix first occurrence if duplicated
    first_idx: dict[str, int] = {}
    for i, k in enumerate(raw_keys):
        if k not in first_idx:
            first_idx[k] = i
    for k, count in seen.items():
        if count > 0 and first_idx[k] < len(result):
            result[first_idx[k]] = f"{k}_0"
    return result


def _build_column_keys(
    table: TableData,
    col_header_rows: set[int],
    title_rows: set[int],
) -> tuple[list[str], frozenset[str]]:
    """Build column key strings from header rows, excluding title rows.

    Multi-level headers are flattened by joining with KEY_SEPARATOR.
    Columns without headers get synthetic keys (col_0, col_1, ...).
    Duplicate keys are disambiguated with _0, _1, ... suffixes.

    Returns (keys, synthetic_keys) where synthetic_keys is the set of
    key strings that are positional placeholders rather than real header text.
    """
    n_cols = table.data.shape[1]
    raw_keys: list[str] = []
    synthetic_indices: set[int] = set()
    for c in range(n_cols):
        if c in table.col_headers and table.col_headers[c]:
            parts = [
                (row_idx, text)
                for row_idx, text in table.col_headers[c]
                if row_idx in col_header_rows and row_idx not in title_rows
            ]
            parts.sort(key=lambda t: t[0])
            key = KEY_SEPARATOR.join(text for _, text in parts if text.strip())
            if not key:
                key = f"{SYNTHETIC_KEY_PREFIX}{c}"
                synthetic_indices.add(c)
        else:
            key = f"{SYNTHETIC_KEY_PREFIX}{c}"
            synthetic_indices.add(c)
        raw_keys.append(key)

    keys = _disambiguate_keys(raw_keys)
    synthetic_keys = frozenset(keys[c] for c in synthetic_indices)
    return keys, synthetic_keys


def extract_header_info_from_hints(table: TableData, hints: HeaderHints) -> HeaderInfo:
    """Build a HeaderInfo from a trimmed table and its precomputed hints.

    The strip stage already removed top ``<td>`` and top ``<th>`` title
    rows from the grid, so ``td_title_rows`` is always empty in trimmed
    coordinates. ``col_header_rows``, ``th_title_rows``, and
    ``stripped_titles`` are taken verbatim from ``hints``; only ``keys``
    is re-derived here. ``stripped_titles`` was already collected from
    a normalized table inside the strip stage, so re-applying the
    per-cell normalization chain here would double-normalize — and the
    chain is not idempotent for inputs like numeric strings with
    separators (e.g. ``"1,2,3" → "12,3" → "123"``), which would break
    prefix-stripping in ``_align_columns_header``.
    """
    if table.data.size == 0:
        return HeaderInfo(
            keys=[],
            synthetic_keys=frozenset(),
            col_header_rows=set(),
            th_title_rows=set(),
            td_title_rows=set(),
            stripped_titles=set(),
        )

    col_header_rows = set(hints.col_header_rows)
    th_title_rows = set(hints.th_title_rows)
    keys, synthetic_keys = _build_column_keys(table, col_header_rows, th_title_rows)
    stripped_titles = {t for t in hints.stripped_titles if t}

    return HeaderInfo(
        keys=keys,
        synthetic_keys=synthetic_keys,
        col_header_rows=col_header_rows,
        th_title_rows=th_title_rows,
        td_title_rows=set(),
        stripped_titles=stripped_titles,
    )


def _data_row_indices(header: HeaderInfo, n_rows: int) -> list[int]:
    """Return row indices that contain data (not headers or titles).

    Trailing title rows at the end of a multi-row header block are
    included as data rows (e.g. section headers like "SPINAL DISORDERS").
    """
    emit_title_rows: set[int] = set()
    if len(header.col_header_rows) > 1 and header.th_title_rows:
        last_header = max(header.col_header_rows)
        if last_header in header.th_title_rows:
            emit_title_rows.add(last_header)

    return [
        r
        for r in range(n_rows)
        if r not in (header.col_header_rows - emit_title_rows) and r not in header.td_title_rows
    ]


def align_columns(
    gt_keys: list[str],
    pred_keys: list[str],
    *,
    gt_synthetic: frozenset[str] = frozenset(),
    pred_synthetic: frozenset[str] = frozenset(),
    gt_titles: set[str] | None = None,
    pred_titles: set[str] | None = None,
) -> tuple[dict[str, str], float]:
    """Align predicted columns to GT columns using header-based fuzzy matching."""
    if not gt_keys or not pred_keys:
        return {}, 0.0
    return _align_columns_header(
        gt_keys,
        pred_keys,
        gt_synthetic=gt_synthetic,
        pred_synthetic=pred_synthetic,
        gt_titles=gt_titles,
        pred_titles=pred_titles,
    )


def _align_columns_header(
    gt_keys: list[str],
    pred_keys: list[str],
    *,
    gt_synthetic: frozenset[str] = frozenset(),
    pred_synthetic: frozenset[str] = frozenset(),
    gt_titles: set[str] | None = None,
    pred_titles: set[str] | None = None,
) -> tuple[dict[str, str], float]:
    """Header-based column alignment using fuzzy key matching.

    If the initial alignment matches fewer than half the columns, checks
    whether a title stripped from one side is a prefix of the other side's
    keys.  If so, strips that prefix and retries alignment.

    This handles the case where GT has "Months of Development 3" (title
    kept as hierarchical header) but pred has just "3" (title stripped),
    or vice versa.
    """
    mapping, alignment_score = _align_columns_header_core(
        gt_keys, pred_keys, gt_synthetic=gt_synthetic, pred_synthetic=pred_synthetic
    )

    # If we matched most columns, no fallback needed
    min_cols = min(len(gt_keys), len(pred_keys))
    if min_cols <= 1 or len(mapping) >= min_cols * 0.5:
        return mapping, alignment_score

    # Try prefix-stripping fallback, but ONLY using titles that were
    # actually stripped from the OTHER side's header rows.
    best_mapping, best_score = mapping, alignment_score

    # If pred stripped a title, that title might be a prefix in GT keys
    # (GT kept it hierarchical, pred didn't).  Try stripping it from GT.
    for title in pred_titles or ():
        prefix = title + KEY_SEPARATOR
        if not any(k.startswith(prefix) for k in gt_keys):
            continue
        stripped_gt = [k[len(prefix) :] if k.startswith(prefix) else k for k in gt_keys]
        # Stripped keys inherit synthetic status from their originals
        stripped_gt_synthetic = frozenset(sk for sk, ok in zip(stripped_gt, gt_keys, strict=True) if ok in gt_synthetic)
        trial_mapping, trial_score = _align_columns_header_core(
            stripped_gt, pred_keys, gt_synthetic=stripped_gt_synthetic, pred_synthetic=pred_synthetic
        )
        # Remap back to original GT keys
        gt_by_stripped = dict(zip(stripped_gt, gt_keys, strict=True))
        trial_mapping = {gt_by_stripped[sk]: pv for sk, pv in trial_mapping.items()}
        if len(trial_mapping) > len(best_mapping):
            best_mapping, best_score = trial_mapping, trial_score

    # If GT stripped a title, that title might be a prefix in pred keys.
    # Try stripping it from pred.
    for title in gt_titles or ():
        prefix = title + KEY_SEPARATOR
        if not any(k.startswith(prefix) for k in pred_keys):
            continue
        stripped_pred = [k[len(prefix) :] if k.startswith(prefix) else k for k in pred_keys]
        stripped_pred_synthetic = frozenset(
            sk for sk, ok in zip(stripped_pred, pred_keys, strict=True) if ok in pred_synthetic
        )
        trial_mapping, trial_score = _align_columns_header_core(
            gt_keys, stripped_pred, gt_synthetic=gt_synthetic, pred_synthetic=stripped_pred_synthetic
        )
        # Remap back to original pred keys
        pred_by_stripped = dict(zip(stripped_pred, pred_keys, strict=True))
        trial_mapping = {gk: pred_by_stripped[sv] for gk, sv in trial_mapping.items()}
        if len(trial_mapping) > len(best_mapping):
            best_mapping, best_score = trial_mapping, trial_score

    return best_mapping, best_score


_HIERARCHY_DASH_RE = re.compile(r"\s+[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212]\s+")


def _normalize_key_for_match(key: str) -> str:
    """Collapse a dash-like separator surrounded by spaces into a single space.

    Used only when comparing column keys for alignment, so that a flattened
    hierarchical header like ``"Group A B"`` matches a model's dash-joined
    rendering ``"Group A - B"``. Real keys are stored unchanged; only the
    similarity input is normalized. Covers ASCII hyphen plus the common
    Unicode hyphen/dash characters (en dash, em dash, minus, etc.).
    """
    return _HIERARCHY_DASH_RE.sub(" ", key).lower()


def _align_columns_header_core(
    gt_keys: list[str],
    pred_keys: list[str],
    *,
    gt_synthetic: frozenset[str] = frozenset(),
    pred_synthetic: frozenset[str] = frozenset(),
) -> tuple[dict[str, str], float]:
    """Core header-based column alignment using fuzzy key matching."""
    n_gt = len(gt_keys)
    n_pred = len(pred_keys)
    gt_norm = [_normalize_key_for_match(k) for k in gt_keys]
    pred_norm = [_normalize_key_for_match(k) for k in pred_keys]
    sim = np.zeros((n_gt, n_pred))
    for i, gk in enumerate(gt_keys):
        for j, pk in enumerate(pred_keys):
            gk_empty = _is_empty_or_nan(gk) or gk in gt_synthetic
            pk_empty = _is_empty_or_nan(pk) or pk in pred_synthetic
            if gk_empty != pk_empty:
                sim[i, j] = 0.0  # empty vs non-empty = no match
            else:
                sim[i, j] = fuzz.ratio(gt_norm[i], pred_norm[j]) / 100.0

    row_ind, col_ind = linear_sum_assignment(-sim)

    mapping: dict[str, str] = {}
    scores: list[float] = []
    for r, c in zip(row_ind, col_ind, strict=True):
        if sim[r, c] >= _COLUMN_MATCH_THRESHOLD:
            mapping[gt_keys[r]] = pred_keys[c]
            scores.append(sim[r, c])

    alignment_score = float(np.mean(scores)) if scores else 0.0
    return mapping, alignment_score


# ===========================================================================
# Phase 3: Record Generation
# ===========================================================================


def build_records(table: TableData, header: HeaderInfo) -> list[dict[str, str]]:
    """Build data records from non-header, non-title rows.

    Uses _data_row_indices for row selection.
    """
    if table.data.size == 0:
        return []
    n_rows, n_cols = table.data.shape
    data_rows = _data_row_indices(header, n_rows)

    records: list[dict[str, str]] = []
    for r in data_rows:
        record: dict[str, str] = {}
        for col_idx, key in zip(range(n_cols), header.keys, strict=True):
            record[key] = str(table.data[r, col_idx])
        records.append(record)

    # Remove records where every cell is empty — these are padding artifacts
    # (e.g., trailing rows in merged tables) and should not affect scoring.
    return [r for r in records if not _is_all_empty_record(r)]


def table_to_records(
    table: TableData,
) -> tuple[list[str], list[dict[str, str]], set[str]]:
    """Convert a TableData into (keys, records, stripped_titles).

    Convenience wrapper combining extract_header_info + build_records.
    """
    header = extract_header_info(table)
    records = build_records(table, header)
    return header.keys, records, header.stripped_titles


# ===========================================================================
# Phase 4: Record Comparison & Scoring
# ===========================================================================


def _record_similarity(
    gt: dict[str, str],
    pred: dict[str, str],
    column_mapping: dict[str, str],
    *,
    gt_keys: list[str] | None = None,
    pred_keys: list[str] | None = None,
    zero_columns: set[str] | None = None,
) -> float:
    """Compute similarity between two records using the column mapping.

    Scores over the union of GT and pred column keys (matched columns
    counted once, unmatched columns from either side counted once each)
    so that both unmatched GT columns and extra pred columns penalize
    the score symmetrically.  Matched columns get their cell_score;
    unmatched columns contribute 0.0.  Columns in zero_columns are
    forced to 0 (empty header penalty).
    """
    if not column_mapping:
        return 0.0

    # Union size = matched columns + unmatched GT columns + extra pred columns.
    # (fall back to mapping size for backward compatibility when
    # gt_keys/pred_keys aren't provided)
    n_gt_cols = len(gt_keys) if gt_keys is not None else len(column_mapping)
    n_pred_cols = len(pred_keys) if pred_keys is not None else len(column_mapping)
    n_matched = len(column_mapping)
    n_total = n_matched + (n_gt_cols - n_matched) + (n_pred_cols - n_matched)
    if n_total == 0:
        return 0.0

    # Score matched columns
    matched_score = 0.0
    for gt_key, pred_key in column_mapping.items():
        if zero_columns and gt_key in zero_columns:
            continue  # Force 0 for this column
        gt_val = gt.get(gt_key, "")
        pred_val = pred.get(pred_key, "")
        matched_score += cell_score(gt_val, pred_val)

    # Unmatched columns (both sides) implicitly contribute 0.0
    return matched_score / n_total


def match_records(
    gt_records: list[dict[str, str]],
    pred_records: list[dict[str, str]],
    column_mapping: dict[str, str],
    *,
    gt_keys: list[str] | None = None,
    pred_keys: list[str] | None = None,
    zero_columns: set[str] | None = None,
) -> tuple[list[tuple[int, int, float]], float]:
    """Match GT records to predicted records and compute overall score.

    Args:
        gt_keys: All GT column keys — unmatched columns penalize the score.
        pred_keys: All pred column keys — extra columns penalize the score.
        zero_columns: GT columns forced to 0 score (empty header penalty).

    Returns:
        matches: list of (gt_idx, pred_idx, pair_score)
        score: sum(pair_scores) / max(n_gt, n_pred)
    """
    n_gt = len(gt_records)
    n_pred = len(pred_records)

    if n_gt == 0 and n_pred == 0:
        return [], 1.0
    if n_gt == 0 or n_pred == 0:
        return [], 0.0

    # Build record similarity matrix
    sim = np.zeros((n_gt, n_pred))
    for i, g in enumerate(gt_records):
        for j, p in enumerate(pred_records):
            sim[i, j] = _record_similarity(
                g, p, column_mapping, gt_keys=gt_keys, pred_keys=pred_keys, zero_columns=zero_columns
            )

    # Match records via Hungarian algorithm
    matches: list[tuple[int, int, float]] = []
    row_ind, col_ind = linear_sum_assignment(-sim)
    for r, c in zip(row_ind, col_ind, strict=True):
        matches.append((r, c, sim[r, c]))

    total_score = sum(s for _, _, s in matches)
    score = total_score / max(n_gt, n_pred)
    return matches, score


def build_record_details(
    gt_records: list[dict[str, str]],
    pred_records: list[dict[str, str]],
    column_mapping: dict[str, str],
    matches: list[tuple[int, int, float]],
    *,
    gt_keys: list[str] | None = None,
    pred_keys: list[str] | None = None,
    zero_columns: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build per-record detail dicts for diagnostics.

    Returns a list of dicts, one per matched pair plus entries for
    unmatched GT/pred records.  Each dict contains the expected values,
    predicted values, per-cell scores, and the overall record score.
    """
    # Reverse mapping: pred_key → gt_key for finding extra pred columns
    reverse_mapping = {v: k for k, v in column_mapping.items()}

    # Unmatched GT keys (no corresponding pred column)
    all_gt_keys = gt_keys if gt_keys is not None else list(column_mapping)
    unmatched_gt_keys = [k for k in all_gt_keys if k not in column_mapping]

    # Extra pred keys (no corresponding GT column)
    all_pred_keys = pred_keys if pred_keys is not None else list(column_mapping.values())
    extra_pred_keys = [k for k in all_pred_keys if k not in reverse_mapping]

    details: list[dict[str, Any]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for gt_idx, pred_idx, pair_score in matches:
        gt_idx = int(gt_idx)
        pred_idx = int(pred_idx)
        pair_score = float(pair_score)
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        gt_rec = gt_records[gt_idx]
        pred_rec = pred_records[pred_idx]

        cells: list[dict[str, Any]] = []
        # Matched columns
        for gt_key, pred_key in column_mapping.items():
            gt_val = gt_rec.get(gt_key, "")
            pred_val = pred_rec.get(pred_key, "")
            if zero_columns and gt_key in zero_columns:
                cs = 0.0
            else:
                cs = cell_score(gt_val, pred_val)
            cells.append(
                {
                    "column": gt_key,
                    "expected": gt_val,
                    "actual": pred_val,
                    "score": cs,
                }
            )
        # Unmatched GT columns (missing from pred)
        for gt_key in unmatched_gt_keys:
            gt_val = gt_rec.get(gt_key, "")
            cells.append(
                {
                    "column": gt_key,
                    "expected": gt_val,
                    "actual": None,
                    "score": 0.0,
                }
            )
        # Extra pred columns (not in GT)
        for pred_key in extra_pred_keys:
            pred_val = pred_rec.get(pred_key, "")
            cells.append(
                {
                    "column": f"[extra] {pred_key}",
                    "expected": None,
                    "actual": pred_val,
                    "score": 0.0,
                }
            )

        details.append(
            {
                "type": "matched",
                "gt_index": gt_idx,
                "pred_index": pred_idx,
                "score": pair_score,
                "cells": cells,
            }
        )

    for i, gt_rec in enumerate(gt_records):
        if i not in matched_gt:
            details.append(
                {
                    "type": "unmatched_gt",
                    "gt_index": i,
                    "pred_index": None,
                    "score": 0.0,
                    "cells": [
                        {"column": k, "expected": gt_rec.get(k, ""), "actual": None, "score": 0.0}
                        for k in column_mapping
                    ],
                }
            )

    for j, pred_rec in enumerate(pred_records):
        if j not in matched_pred:
            details.append(
                {
                    "type": "unmatched_pred",
                    "gt_index": None,
                    "pred_index": j,
                    "score": 0.0,
                    "cells": [
                        {"column": pk, "expected": None, "actual": pred_rec.get(pk, ""), "score": 0.0}
                        for pk in column_mapping.values()
                    ],
                }
            )

    return details


def _build_detail_strings(per_table_details: list[dict[str, Any]]) -> list[str]:
    """Build human-readable detail strings from per-table record details.

    Produces ``[SECTION:...]`` markers recognised by the detailed report JS
    to render two nested collapsibles: perfect and imperfect matches.
    """
    multi_table = len(per_table_details) > 1
    perfect_lines: list[str] = []
    perfect_columns: list[str] = []
    imperfect_lines: list[str] = []

    for td in per_table_details:
        table_idx: int = td.get("gt_table_index", 0)
        prefix = f"Table {table_idx + 1}: " if multi_table else ""
        record_details: list[dict[str, Any]] = td.get("record_details", [])

        # Report entirely unmatched tables (no pred pairing, no column matches, no GT records)
        if not record_details and td.get("score", 0.0) == 0.0:
            reason = td.get("reason", "no prediction")
            pred_idx = td.get("pred_table_index")
            gt_cols: list[str] = td.get("gt_columns", [])
            n_gt_recs: int = td.get("n_gt_records", 0)

            if pred_idx is None:
                line = f"{prefix}GT table {table_idx + 1} — unmatched ({reason})"
            else:
                line = f"{prefix}GT table {table_idx + 1} ↔ Pred table {pred_idx + 1} — {reason}"

            if gt_cols:
                line += f", columns: [{', '.join(gt_cols)}]"
            if n_gt_recs > 0:
                line += f", {n_gt_recs} record{'s' if n_gt_recs != 1 else ''}"

            imperfect_lines.append(line)
            continue

        for rd in record_details:
            rtype = rd.get("type", "")
            score = rd.get("score", 0.0)
            gt_idx = rd.get("gt_index")
            pred_idx = rd.get("pred_index")
            cells: list[dict[str, Any]] = rd.get("cells", [])

            if rtype == "unmatched_gt":
                imperfect_lines.append(f"{prefix}GT#{gt_idx} \u2014 unmatched (no prediction)")
            elif rtype == "unmatched_pred":
                imperfect_lines.append(f"{prefix}Pred#{pred_idx} \u2014 extra prediction (no GT match)")
            elif score == 1.0:
                if not perfect_columns and cells:
                    perfect_columns = [c.get("column", "?") for c in cells]
                vals = " | ".join(f'"{c.get("expected", "")}"' for c in cells)
                perfect_lines.append(f"{prefix}GT#{gt_idx} \u2194 Pred#{pred_idx} ({score:.3f}) \u2014 {vals}")
            else:
                # Show only imperfect cells to keep lines compact
                diffs = []
                for c in cells:
                    if c.get("score", 1.0) < 1.0:
                        exp = c.get("expected", "")
                        act = c.get("actual", "")
                        cscore = c.get("score", 0.0)
                        col = c.get("column", "?")
                        diffs.append(f'{col}: "{exp}"\u2192"{act}" ({cscore:.2f})')
                diff_str = " | ".join(diffs) if diffs else "all cells differ"
                imperfect_lines.append(f"{prefix}GT#{gt_idx} \u2194 Pred#{pred_idx} ({score:.3f}) \u2014 {diff_str}")

    lines: list[str] = []
    if perfect_lines:
        lines.append(f"[SECTION:Perfect Record Matches ({len(perfect_lines)})]")
        if perfect_columns:
            lines.append("Columns: " + " | ".join(perfect_columns))
        lines.extend(perfect_lines)
    if imperfect_lines:
        lines.append(f"[SECTION:Imperfect Record Matches ({len(imperfect_lines)})]")
        lines.extend(imperfect_lines)
    return lines


# ===========================================================================
# Header Recovery, Demotion & Promotion Helpers
# ===========================================================================


def _try_section_header_promotion(
    table: TableData,
    header: HeaderInfo,
) -> HeaderInfo | None:
    """Try promoting the row right below the header block to a header row.

    Returns a new HeaderInfo with the row promoted, or None if no candidate exists.
    The candidate row must be a section-header-like row (spanning, single value)
    immediately after the header block.
    """
    if not header.col_header_rows:
        return None
    n_rows = table.data.shape[0]
    candidate_row = max(header.col_header_rows) + 1
    if candidate_row >= n_rows:
        return None
    # Check if it's a section-header-like row (width-spanning, single value)
    n_cols = table.data.shape[1]
    row_values = [str(table.data[candidate_row, c]).strip() for c in range(n_cols)]
    unique_nonempty = {v for v in row_values if v}
    if len(unique_nonempty) != 1:
        return None
    # Rebuild header info with this row included in col_header_rows
    new_col_header_rows = header.col_header_rows | {candidate_row}
    new_keys, new_synthetic = _build_column_keys(table, new_col_header_rows, header.th_title_rows)
    return HeaderInfo(
        keys=new_keys,
        synthetic_keys=new_synthetic,
        col_header_rows=new_col_header_rows,
        th_title_rows=header.th_title_rows,
        td_title_rows=header.td_title_rows,
        stripped_titles=header.stripped_titles,
    )


def _try_recover_pred_header(
    gt_header: HeaderInfo,
    pred_table: TableData,
    pred_header: HeaderInfo,
) -> HeaderInfo | None:
    """Try to recover pred headers from first data row when pred has no real headers.

    If GT has real headers and pred has all synthetic keys, check if pred's
    first data row matches any GT header. If any column matches, promote
    the entire first row to a header row.

    Returns new HeaderInfo with recovered headers, or None if recovery fails.
    """
    # Only trigger when GT has real headers and pred has all synthetic
    gt_has_real = bool(set(gt_header.keys) - gt_header.synthetic_keys)
    pred_all_synthetic = set(pred_header.keys) <= pred_header.synthetic_keys
    if not gt_has_real or not pred_all_synthetic:
        return None

    n_rows, n_cols = pred_table.data.shape
    data_rows = _data_row_indices(pred_header, n_rows)
    if not data_rows:
        return None

    first_data_row = data_rows[0]
    first_row_values = [str(pred_table.data[first_data_row, c]).strip() for c in range(n_cols)]

    # Check if any value in the first row matches a GT header key
    gt_real_keys = {k.lower() for k in gt_header.keys if k not in gt_header.synthetic_keys}
    has_match = any(
        fuzz.ratio(val.lower(), gk) / 100.0 >= _COLUMN_MATCH_THRESHOLD
        for val in first_row_values
        if val
        for gk in gt_real_keys
    )

    if not has_match:
        return None

    # Promote first data row to header
    new_col_header_rows = pred_header.col_header_rows | {first_data_row}
    raw_keys = [val if val else f"{SYNTHETIC_KEY_PREFIX}{c}" for c, val in enumerate(first_row_values)]
    new_keys = _disambiguate_keys(raw_keys)
    new_synthetic = frozenset(k for c, k in enumerate(new_keys) if not first_row_values[c])

    return HeaderInfo(
        keys=new_keys,
        synthetic_keys=new_synthetic,
        col_header_rows=new_col_header_rows,
        th_title_rows=pred_header.th_title_rows,
        td_title_rows=pred_header.td_title_rows,
        stripped_titles=pred_header.stripped_titles,
    )


def _demote_pred_headers(
    gt_header: HeaderInfo,
    pred_table: TableData,
    pred_header: HeaderInfo,
) -> HeaderInfo:
    """Demote pred headers to data when GT has no real headers.

    GT is authoritative — if GT says no headers, pred's <th> tags are irrelevant.
    Returns new HeaderInfo with all header rows cleared and synthetic keys.
    """
    gt_all_synthetic = set(gt_header.keys) <= gt_header.synthetic_keys
    pred_has_real = bool(set(pred_header.keys) - pred_header.synthetic_keys)

    if not gt_all_synthetic or not pred_has_real:
        return pred_header  # No demotion needed

    n_cols = pred_table.data.shape[1]
    demoted_keys = [f"{SYNTHETIC_KEY_PREFIX}{c}" for c in range(n_cols)]

    return HeaderInfo(
        keys=demoted_keys,
        synthetic_keys=frozenset(demoted_keys),
        col_header_rows=set(),  # No header rows
        th_title_rows=set(),  # Title rows also become data
        td_title_rows=set(),  # Title rows also become data
        stripped_titles=set(),
    )


# ===========================================================================
# Phase 5: Side-by-Side Table Splitting
# ===========================================================================


def _resolve_header_row_values(
    table: TableData,
    header: HeaderInfo,
) -> list[list[str]]:
    """Extract per-column header values for each header row, resolving colspan.

    Returns a list of rows, where each row is a list of normalized cell texts
    (one per column). Colspan'd cells appear with their text repeated for each
    column they cover (since col_headers already stores per-column entries).

    Only includes rows in header.col_header_rows, sorted top-to-bottom.
    """
    n_cols = table.data.shape[1]

    # Build (col, row) → text lookup for O(1) access
    header_lookup: dict[tuple[int, int], str] = {}
    for col_idx, entries in table.col_headers.items():
        for row_idx, text in entries:
            header_lookup[(col_idx, row_idx)] = _normalize_trm_cell_text(normalize_text(text.strip()))

    result: list[list[str]] = []
    for row_idx in sorted(header.col_header_rows):
        result.append([header_lookup.get((c, row_idx), "") for c in range(n_cols)])
    return result


# ===========================================================================
# Metric Orchestrator
# ===========================================================================


class TableRecordMatchMetric:
    """Compute table_record_match: semantic equivalence of HTML tables as record collections."""

    def _score_table_pair(
        self,
        gt_table: TableData,
        pred_table: TableData,
        table_index: int,
        *,
        gt_hints: HeaderHints,
        pred_hints: HeaderHints,
    ) -> tuple[float, dict[str, Any]]:
        """Score a single GT/pred table pair.

        Pipeline:
        1. Normalize cell text in both tables
        2. Build header info from precomputed hints, align columns
        3. Generate records from both tables
        4. Match and score records
        """
        # Phase 1: Normalize
        gt_table = normalize_table(gt_table)
        pred_table = normalize_table(pred_table)

        # Phase 2: Header extraction & column alignment
        gt_header = extract_header_info_from_hints(gt_table, gt_hints)
        pred_header = extract_header_info_from_hints(pred_table, pred_hints)

        # Header recovery: if GT has headers and pred doesn't, try promoting pred's first row
        recovered = _try_recover_pred_header(gt_header, pred_table, pred_header)
        if recovered is not None:
            pred_header = recovered

        # Header demotion: if GT has no headers, demote pred's headers to data
        pred_header = _demote_pred_headers(gt_header, pred_table, pred_header)

        col_mapping, alignment_score = align_columns(
            gt_header.keys,
            pred_header.keys,
            gt_synthetic=gt_header.synthetic_keys,
            pred_synthetic=pred_header.synthetic_keys,
            gt_titles=gt_header.stripped_titles,
            pred_titles=pred_header.stripped_titles,
        )

        # Try section header promotion for both tables
        for table, header, is_gt in [(gt_table, gt_header, True), (pred_table, pred_header, False)]:
            promoted = _try_section_header_promotion(table, header)
            if promoted is None:
                continue

            if is_gt:
                trial_gt_keys = promoted.keys
                trial_pred_keys = pred_header.keys
                trial_gt_synthetic = promoted.synthetic_keys
                trial_pred_synthetic = pred_header.synthetic_keys
                trial_gt_titles = promoted.stripped_titles
                trial_pred_titles = pred_header.stripped_titles
            else:
                trial_gt_keys = gt_header.keys
                trial_pred_keys = promoted.keys
                trial_gt_synthetic = gt_header.synthetic_keys
                trial_pred_synthetic = promoted.synthetic_keys
                trial_gt_titles = gt_header.stripped_titles
                trial_pred_titles = promoted.stripped_titles

            trial_mapping, trial_score = align_columns(
                trial_gt_keys,
                trial_pred_keys,
                gt_synthetic=trial_gt_synthetic,
                pred_synthetic=trial_pred_synthetic,
                gt_titles=trial_gt_titles,
                pred_titles=trial_pred_titles,
            )
            if len(trial_mapping) > len(col_mapping) or (
                len(trial_mapping) == len(col_mapping) and trial_score > alignment_score
            ):
                col_mapping, alignment_score = trial_mapping, trial_score
                if is_gt:
                    gt_header = promoted
                else:
                    pred_header = promoted

        # Hard gate: no columns matched → 0
        if not col_mapping:
            gt_real_keys = [k for k in gt_header.keys if k not in gt_header.synthetic_keys]
            n_gt_recs = len(build_records(gt_table, gt_header))
            return 0.0, {
                "gt_table_index": table_index,
                "pred_table_index": table_index,
                "score": 0.0,
                "reason": "no column matches",
                "gt_columns": gt_real_keys if gt_real_keys else gt_header.keys,
                "n_gt_records": n_gt_recs,
            }

        # Compute zero_columns: GT columns with real headers matched to empty/synthetic pred headers
        zero_columns: set[str] = set()
        for gt_key, pred_key in col_mapping.items():
            gt_is_real = gt_key not in gt_header.synthetic_keys and not _is_empty_or_nan(gt_key)
            pred_is_empty = pred_key in pred_header.synthetic_keys or _is_empty_or_nan(pred_key)
            if gt_is_real and pred_is_empty:
                zero_columns.add(gt_key)

        # Phase 3: Record generation
        gt_records = build_records(gt_table, gt_header)
        pred_records = build_records(pred_table, pred_header)

        if not gt_records:
            return 0.0, {
                "gt_table_index": table_index,
                "pred_table_index": table_index,
                "score": 0.0,
                "reason": "no GT records",
                "gt_columns": gt_header.keys,
                "n_gt_records": 0,
            }

        # Phase 4: Record comparison & scoring
        matches, score = match_records(
            gt_records,
            pred_records,
            col_mapping,
            gt_keys=gt_header.keys,
            pred_keys=pred_header.keys,
            zero_columns=zero_columns if zero_columns else None,
        )
        record_details = build_record_details(
            gt_records,
            pred_records,
            col_mapping,
            matches,
            gt_keys=gt_header.keys,
            pred_keys=pred_header.keys,
            zero_columns=zero_columns if zero_columns else None,
        )

        return score, {
            "gt_table_index": table_index,
            "pred_table_index": table_index,
            "score": score,
            "alignment_score": alignment_score,
            "n_gt_records": len(gt_records),
            "n_pred_records": len(pred_records),
            "n_matched_columns": len(col_mapping),
            "column_mapping": col_mapping,
            "record_details": record_details,
        }

    @property
    def metric_name(self) -> str:
        return "table_record_match"

    @staticmethod
    def _extract_grits_table_pairing(
        grits_results: list[MetricValue] | None,
    ) -> list[tuple[int, int | None]] | None:
        """Extract GT→pred table pairing from GriTS metadata.

        Returns list of (gt_idx, pred_idx_or_None) sorted by gt_idx,
        or None if GriTS results aren't available.
        """
        if not grits_results:
            return None
        for r in grits_results:
            if r.metadata and "per_table_details" in r.metadata:
                details = r.metadata["per_table_details"]
                return [(d["gt_table_index"], d.get("pred_table_index")) for d in details]
        return None

    @staticmethod
    def _compute_grits_table_pairing(
        expected_tables: list[str],
        actual_tables: list[str],
    ) -> list[tuple[int, int | None]]:
        """Compute GT→pred table pairing via GriTS-Con Hungarian matching.

        Runs GriTS-Con on all (GT × pred) table pairs, then uses the
        Hungarian algorithm to find the optimal 1:1 assignment.
        """
        from extract_bench.evaluation.metrics.parse.grits_metric import (
            grits_from_html,
        )

        n_gt = len(expected_tables)
        n_pred = len(actual_tables)
        cost_matrix = np.zeros((n_gt, n_pred))

        for i, gt_html in enumerate(expected_tables):
            for j, pred_html in enumerate(actual_tables):
                result = grits_from_html(gt_html, pred_html)
                score = result["grits_con"] if result else 0.0
                cost_matrix[i, j] = -score

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched: dict[int, int] = {}
        for r, c in zip(row_ind, col_ind, strict=True):
            matched[int(r)] = int(c)

        pairing: list[tuple[int, int | None]] = []
        for i in range(n_gt):
            pairing.append((i, matched.get(i)))
        return pairing

    def compute_extracted(
        self,
        expected_tables: list[Any],
        actual_tables: list[Any],
        *,
        pairing: list[tuple[int, int | None]],
    ) -> list[MetricValue]:
        """Score TRM from pre-extracted ExtractedTable lists + GriTS pairing.

        New entry point for the unified table-extraction pipeline (P3+).
        TRM still owns its normalization — ``normalize_table`` is applied
        per pair inside ``_score_table_pair``, so scores are numerically
        equal to the legacy ``compute(str, str)`` path.
        """
        # The strip_title_rows stage runs upstream in the evaluator and
        # populates ``header_hints``. If a caller bypasses that stage,
        # apply it lazily here so this entry point stays self-contained.
        expected_tables = [et if et.header_hints is not None else strip_title_rows(et) for et in expected_tables]
        actual_tables = [et if et.header_hints is not None else strip_title_rows(et) for et in actual_tables]

        gt_parsed = [et.table_data for et in expected_tables]
        pred_parsed = [et.table_data for et in actual_tables]
        gt_hints_list: list[HeaderHints] = [et.header_hints for et in expected_tables]
        pred_hints_list: list[HeaderHints] = [et.header_hints for et in actual_tables]

        if not gt_parsed:
            return []

        if not pred_parsed:
            return [
                MetricValue(
                    metric_name=self.metric_name,
                    value=0.0,
                    metadata={
                        "tables_predicted": False,
                        "n_gt_tables": len(gt_parsed),
                        "n_pred_tables": 0,
                    },
                ),
                MetricValue(
                    metric_name=f"{self.metric_name}_perfect",
                    value=0.0,
                    metadata={
                        "tables_predicted": False,
                        "n_perfect": 0,
                        "n_gt_tables": len(gt_parsed),
                    },
                ),
            ]

        n_gt = len(gt_parsed)
        n_pred = len(pred_parsed)

        # Splitting and any pairing override now happen upstream in the
        # evaluator (see ``split_ambiguous_merged_pred``); TRM consumes the
        # GriTS-supplied pairing uniformly, including on split sub-tables.
        local_pairing: list[tuple[int, int | None]] = list(pairing)

        per_table_scores: list[float] = []
        per_table_details: list[dict[str, Any]] = []

        for gt_idx, pred_idx in local_pairing:
            if pred_idx is None or pred_idx >= n_pred or gt_idx >= n_gt:
                per_table_scores.append(0.0)
                unmatched_detail: dict[str, Any] = {
                    "gt_table_index": gt_idx,
                    "pred_table_index": None,
                    "score": 0.0,
                    "reason": "no prediction",
                }
                if gt_idx < n_gt:
                    gt_t = normalize_table(gt_parsed[gt_idx])
                    gt_h = extract_header_info_from_hints(gt_t, gt_hints_list[gt_idx])
                    gt_r = build_records(gt_t, gt_h)
                    real_keys = [k for k in gt_h.keys if k not in gt_h.synthetic_keys]
                    unmatched_detail["gt_columns"] = real_keys if real_keys else gt_h.keys
                    unmatched_detail["n_gt_records"] = len(gt_r)
                per_table_details.append(unmatched_detail)
                continue

            score, detail = self._score_table_pair(
                gt_parsed[gt_idx],
                pred_parsed[pred_idx],
                gt_idx,
                gt_hints=gt_hints_list[gt_idx],
                pred_hints=pred_hints_list[pred_idx],
            )
            detail["pred_table_index"] = pred_idx
            per_table_scores.append(score)
            per_table_details.append(detail)

        avg_score = float(np.mean(per_table_scores)) if per_table_scores else 0.0
        n_perfect = sum(1 for s in per_table_scores if s == 1.0)
        perfect_rate = n_perfect / len(per_table_scores) if per_table_scores else 0.0

        return [
            MetricValue(
                metric_name=self.metric_name,
                value=avg_score,
                metadata={
                    "tables_predicted": True,
                    "n_gt_tables": n_gt,
                    "n_pred_tables": n_pred,
                    "per_table_scores": per_table_scores,
                    "per_table_details": per_table_details,
                },
                details=_build_detail_strings(per_table_details),
            ),
            MetricValue(
                metric_name=f"{self.metric_name}_perfect",
                value=perfect_rate,
                metadata={
                    "tables_predicted": True,
                    "n_perfect": n_perfect,
                    "n_gt_tables": n_gt,
                },
            ),
        ]

    def compute(self, expected: str, actual: str, **kwargs: Any) -> list[MetricValue]:
        """Score all tables in expected vs actual HTML/markdown content.

        If ``grits_results`` is passed in kwargs, reuses GriTS's Hungarian
        table matching.  Otherwise computes GriTS-Con matching directly.
        Skips matching when there's only one table on each side.

        Returns list with a single MetricValue for the metric.
        """
        # Use lxml-based parsing as the gatekeeper — it handles malformed
        # HTML (e.g. missing </table>) more reliably than string scanning.
        gt_parsed = parse_html_tables(expected)
        pred_parsed = parse_html_tables(actual)

        if not gt_parsed:
            return []

        if not pred_parsed:
            return [
                MetricValue(
                    metric_name=self.metric_name,
                    value=0.0,
                    metadata={"tables_predicted": False, "n_gt_tables": len(gt_parsed), "n_pred_tables": 0},
                ),
                MetricValue(
                    metric_name=f"{self.metric_name}_perfect",
                    value=0.0,
                    metadata={"tables_predicted": False, "n_perfect": 0, "n_gt_tables": len(gt_parsed)},
                ),
            ]

        # String-scanned tables are still needed for GriTS-based pairing
        expected_tables = extract_html_tables(expected)
        actual_tables = extract_html_tables(actual)

        # Wrap each parsed table in an ExtractedTable and run the strip
        # stage so this legacy entry point feeds ``_score_table_pair`` the
        # same trimmed-table + hints payload that ``compute_extracted``
        # does. Keeps TRM detector-free.
        gt_extracted = [strip_title_rows(ExtractedTable(raw_html="", table_data=t)) for t in gt_parsed]
        pred_extracted = [strip_title_rows(ExtractedTable(raw_html="", table_data=t)) for t in pred_parsed]
        gt_parsed = [et.table_data for et in gt_extracted]
        pred_parsed = [et.table_data for et in pred_extracted]
        gt_hints_list: list[HeaderHints] = [et.header_hints for et in gt_extracted]  # type: ignore[misc]
        pred_hints_list: list[HeaderHints] = [et.header_hints for et in pred_extracted]  # type: ignore[misc]

        n_gt = len(gt_parsed)
        n_pred = len(pred_parsed)

        # Determine table pairing via GriTS Hungarian matching (reused from
        # the evaluator if available, otherwise computed locally). Splitting
        # of ambiguous merged-pred tables happens upstream in the evaluator
        # via ``split_ambiguous_merged_pred``; this legacy ``compute(str, str)``
        # entry point no longer attempts it.
        grits_results: list[MetricValue] | None = kwargs.get("grits_results")
        pairing: list[tuple[int, int | None]] | None = self._extract_grits_table_pairing(grits_results)
        if pairing is None:
            pairing = self._compute_grits_table_pairing(expected_tables, actual_tables)

        per_table_scores: list[float] = []
        per_table_details: list[dict[str, Any]] = []

        for gt_idx, pred_idx in pairing:
            if pred_idx is None or pred_idx >= n_pred or gt_idx >= n_gt:
                per_table_scores.append(0.0)
                unmatched_detail: dict[str, Any] = {
                    "gt_table_index": gt_idx,
                    "pred_table_index": None,
                    "score": 0.0,
                    "reason": "no prediction",
                }
                # Include GT table info for the detail report
                if gt_idx < n_gt:
                    gt_t = normalize_table(gt_parsed[gt_idx])
                    gt_h = extract_header_info_from_hints(gt_t, gt_hints_list[gt_idx])
                    gt_r = build_records(gt_t, gt_h)
                    real_keys = [k for k in gt_h.keys if k not in gt_h.synthetic_keys]
                    unmatched_detail["gt_columns"] = real_keys if real_keys else gt_h.keys
                    unmatched_detail["n_gt_records"] = len(gt_r)
                per_table_details.append(unmatched_detail)
                continue

            score, detail = self._score_table_pair(
                gt_parsed[gt_idx],
                pred_parsed[pred_idx],
                gt_idx,
                gt_hints=gt_hints_list[gt_idx],
                pred_hints=pred_hints_list[pred_idx],
            )
            detail["pred_table_index"] = pred_idx
            per_table_scores.append(score)
            per_table_details.append(detail)

        avg_score = float(np.mean(per_table_scores)) if per_table_scores else 0.0
        n_perfect = sum(1 for s in per_table_scores if s == 1.0)
        perfect_rate = n_perfect / len(per_table_scores) if per_table_scores else 0.0

        return [
            MetricValue(
                metric_name=self.metric_name,
                value=avg_score,
                metadata={
                    "tables_predicted": True,
                    "n_gt_tables": n_gt,
                    "n_pred_tables": n_pred,
                    "per_table_scores": per_table_scores,
                    "per_table_details": per_table_details,
                },
                details=_build_detail_strings(per_table_details),
            ),
            MetricValue(
                metric_name=f"{self.metric_name}_perfect",
                value=perfect_rate,
                metadata={
                    "tables_predicted": True,
                    "n_perfect": n_perfect,
                    "n_gt_tables": n_gt,
                },
            ),
        ]
