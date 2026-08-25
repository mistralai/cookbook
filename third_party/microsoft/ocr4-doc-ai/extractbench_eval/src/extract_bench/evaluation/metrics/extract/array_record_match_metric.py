"""Order-insensitive record matching metric for long extract arrays."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from extract_bench.evaluation.metrics.extract.json_subset_match import (
    _is_nullable_numeric_field,
    _normalize_nullable_numeric,
    normalize_date_string,
)
from extract_bench.schemas.evaluation import MetricValue

# Above this many residual cells (unmatched_actual x unmatched_expected, after the
# exact-row peel) the dense assignment matrix is skipped to bound eval memory.
# Matches unified_evidence's _GROUNDED_MAX_CELLS; a 77.4k-row array would need
# ~67 GB otherwise and OOM-kill the runner.
_MAX_RESIDUAL_ASSIGNMENT_CELLS = 100_000_000

_PUNCT_RE = re.compile(r"[^\w\s]", re.U)
_WS_RE = re.compile(r"\s+", re.U)

# Reserved output keys that carry attribution metadata, not schema content, and must
# NEVER be scored as extracted cells. Kept in sync with the codegen provider's
# ``PROVENANCE_KEY`` (table_codegen/schema_utils.py): a record may carry
# ``_provenance: {"page": N}`` for source attribution, which the validity gate tolerates
# and these value metrics ignore (nested record subfields already come from the schema /
# expected rows, so the only leak was the top-level union walk over actual keys).
RESERVED_OUTPUT_KEYS: frozenset[str] = frozenset({"_provenance"})

DEFAULT_FUZZY_FIELD_THRESHOLDS: dict[str, float] = {
    # Matches the public LongArray-Extract reference scorer.
    "description": 0.95,
    "court": 0.85,
}


@dataclass(frozen=True)
class ArrayRecordMatchCounts:
    correct: int
    expected_total: int
    predicted_total: int
    expected_rows: int
    predicted_rows: int
    arrays_scored: int
    scalar_fields_scored: int

    @property
    def false_positive(self) -> int:
        return max(self.predicted_total - self.correct, 0)

    @property
    def false_negative(self) -> int:
        return max(self.expected_total - self.correct, 0)


@dataclass(frozen=True)
class _RowAssignment:
    pairs: list[tuple[int, int]]
    unmatched_actual_indices: list[int]
    unmatched_expected_indices: list[int]


def _normalize_text(value: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", value)).strip().lower()


def _unwrap_value(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"value"}:
        return value["value"]
    return value


def _normalize_dates_deep(value: Any) -> Any:
    """Canonicalize date-like strings to ISO format everywhere in a JSON value.

    Applied once to both sides up front (not per cell pair) so the n×m
    assignment cost matrix stays cheap. `normalize_date_string` only rewrites
    strings that parse as a whole date, so IDs, amounts, and descriptions
    that merely contain digits pass through untouched.
    """
    if isinstance(value, str):
        return normalize_date_string(value)
    if isinstance(value, dict):
        return {k: _normalize_dates_deep(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_dates_deep(v) for v in value]
    return value


def _normalize_ws(value: str) -> str:
    """Collapse internal whitespace runs so only inter-word spacing matters.

    Multi-line cells (postal addresses, wrapped headers) carry hard line breaks
    whose position depends on render column width, not content, so a prediction
    that re-wraps the same text is correct.

    Deliberately whitespace-RUN-only: whitespace hugging punctuation ("3, 4" vs
    "3,4") stays significant here because this helper defines the default cell
    equality (and ``_cell_key`` interning) for EVERY extract dataset. Datasets
    that want punctuation-spacing leniency opt in per field via the
    ``punctuation_spacing`` normalizer (unified_evidence_metric).
    """
    return _WS_RE.sub(" ", value).strip()


def _cell_match(
    expected: Any,
    actual: Any,
    field: str,
    fuzzy_field_thresholds: dict[str, float],
    field_schema: Any = None,
) -> bool:
    threshold = fuzzy_field_thresholds.get(field)
    if threshold is not None and isinstance(expected, str) and isinstance(actual, str):
        expected_norm = _normalize_text(expected)
        actual_norm = _normalize_text(actual)
        return expected_norm == actual_norm or (
            bool(expected_norm) and bool(actual_norm) and fuzz.ratio(expected_norm, actual_norm) >= threshold * 100.0
        )
    if isinstance(expected, str) and isinstance(actual, str):
        return _normalize_ws(expected) == _normalize_ws(actual)
    # Nullable numeric fields treat 0 / 0.0 and None as equivalent. Gated on
    # the JSON Schema shape so plain numeric fields keep strict semantics.
    if _is_nullable_numeric_field(field_schema):
        expected = _normalize_nullable_numeric(expected)
        actual = _normalize_nullable_numeric(actual)
    return bool(expected == actual)


_UNHASHABLE = object()


def _eq_key(value: Any) -> Any:
    """Hashable snapshot of Python ``==`` for intern (not the string-cell rule).

    JSON containers freeze to tagged tuples so intern keys stay plain hashable
    values (no FrozenDict) and JSON-dump after replacing tuples with lists:
    lists as ``("l", items...)`` in order, dicts as ``("d", sorted (key, value)
    pairs)`` because ``dict ==`` is key-order independent. Nested strings stay
    exact: ``["Moscow "]`` and ``["Moscow"]`` are different keys. The top-level
    string branch of ``_cell_key`` still whitespace-folds. Anything that cannot
    freeze (mixed incomparable dict keys, a set, ...) stays ``_UNHASHABLE`` so
    that column falls back to pairwise compare.
    """
    if isinstance(value, list):
        parts = tuple(_eq_key(item) for item in value)
        if any(part is _UNHASHABLE for part in parts):
            return _UNHASHABLE
        return ("l", parts)
    if isinstance(value, dict):
        items: list[tuple[Any, Any]] = []
        for key, item in value.items():
            frozen_key = _eq_key(key)
            frozen_item = _eq_key(item)
            if frozen_key is _UNHASHABLE or frozen_item is _UNHASHABLE:
                return _UNHASHABLE
            items.append((frozen_key, frozen_item))
        try:
            items.sort()
        except TypeError:
            return _UNHASHABLE
        return ("d", tuple(items))
    try:
        hash(value)
    except TypeError:
        return _UNHASHABLE
    return ("v", value)


def _cell_key(value: Any, field_schema: Any = None) -> Any:
    """Hashable interning key that mirrors ``_cell_match`` for non-fuzzy fields.

    Two cells share a key iff ``_cell_match`` returns True (strings compared
    whitespace-insensitively, everything else by ``==``). Strings are tagged
    apart from non-strings so a normalized string can never collide with a
    look-alike scalar, exactly as ``_cell_match`` keeps its str/str branch
    distinct from the ``==`` fallback. List and dict cells freeze to tagged
    tuples of exact nested values so opaque JSON arrays and objects intern
    the same way scalars already do. Returns ``_UNHASHABLE`` when a cell still
    cannot be interned -> caller scores that column pairwise.

    When ``field_schema`` is a nullable-numeric shape, ``0`` / ``0.0`` collapse
    to ``None`` in the key so that null-vs-zero cells intern to the same slot,
    matching the equality relaxation in ``_cell_match``.
    """
    if isinstance(value, str):
        return ("s", _normalize_ws(value))
    if _is_nullable_numeric_field(field_schema):
        value = _normalize_nullable_numeric(value)
    if isinstance(value, (list, dict)):
        return _eq_key(value)
    try:
        hash(value)
    except TypeError:
        return _UNHASHABLE
    return ("v", value)


def _intern_field(
    actual_list: list[Any],
    expected_list: list[Any],
    field: str,
    field_schema: Any = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Map each row's ``field`` cell to an int id shared across both sides.

    Equal cells (per ``_cell_match``) receive the same id, so a single
    broadcasted integer ``!=`` reproduces the per-pair mismatch test while
    normalizing each cell once, not once per pair. Returns ``None`` if any
    cell cannot be interned so the caller scores that column pairwise.
    """
    keymap: dict[Any, int] = {}
    out: list[np.ndarray] = []
    for rows in (actual_list, expected_list):
        ids = np.empty(len(rows), dtype=np.int64)
        for idx, row in enumerate(rows):
            key = _cell_key(row.get(field), field_schema) if isinstance(row, dict) else _cell_key(None, field_schema)
            if key is _UNHASHABLE:
                return None
            slot = keymap.get(key)
            if slot is None:
                slot = len(keymap)
                keymap[key] = slot
            ids[idx] = slot
        out.append(ids)
    return out[0], out[1]


def _row_match_key(
    row: Any,
    subfields: list[str],
    fuzzy_field_thresholds: dict[str, float],
    field_schemas: dict[str, Any] | None = None,
) -> tuple[Any, ...] | None:
    """Hashable full-row key whose equality implies zero assignment cost."""
    row_dict = row if isinstance(row, dict) else {}
    parts: list[Any] = []
    for field in subfields:
        value = row_dict.get(field)
        cell_key = _cell_key(value, (field_schemas or {}).get(field))
        if cell_key is _UNHASHABLE:
            return None
        parts.append(cell_key)
    return tuple(parts)


def _can_peel_exact_rows(subfields: list[str], fuzzy_field_thresholds: dict[str, float]) -> bool:
    return all(fuzzy_field_thresholds.get(field) is None for field in subfields)


def _peel_exact_row_matches(
    actual_list: list[Any],
    expected_list: list[Any],
    subfields: list[str],
    fuzzy_field_thresholds: dict[str, float],
    field_schemas: dict[str, Any] | None = None,
) -> _RowAssignment:
    """Pre-align zero-cost rows before building an expensive residual matrix.

    Long-array predictions are often mostly exact with a few inserted, dropped,
    or shifted rows. Removing exact full-row matches first preserves the value
    score while shrinking the dense Hungarian problem to the true residual.
    """
    if not _can_peel_exact_rows(subfields, fuzzy_field_thresholds):
        return _RowAssignment(
            pairs=[],
            unmatched_actual_indices=list(range(len(actual_list))),
            unmatched_expected_indices=list(range(len(expected_list))),
        )

    actual_by_key: defaultdict[tuple[Any, ...], deque[int]] = defaultdict(deque)
    for actual_idx, row in enumerate(actual_list):
        key = _row_match_key(row, subfields, fuzzy_field_thresholds, field_schemas)
        if key is not None:
            actual_by_key[key].append(actual_idx)

    pairs: list[tuple[int, int]] = []
    matched_actual: set[int] = set()
    unmatched_expected_indices: list[int] = []
    for expected_idx, row in enumerate(expected_list):
        key = _row_match_key(row, subfields, fuzzy_field_thresholds, field_schemas)
        bucket = actual_by_key.get(key) if key is not None else None
        if bucket:
            actual_idx = bucket.popleft()
            pairs.append((actual_idx, expected_idx))
            matched_actual.add(actual_idx)
        else:
            unmatched_expected_indices.append(expected_idx)

    unmatched_actual_indices = [idx for idx in range(len(actual_list)) if idx not in matched_actual]
    return _RowAssignment(
        pairs=pairs,
        unmatched_actual_indices=unmatched_actual_indices,
        unmatched_expected_indices=unmatched_expected_indices,
    )


def _mismatch_cost_matrix(
    actual_list: list[Any],
    expected_list: list[Any],
    subfields: list[str],
    fuzzy_field_thresholds: dict[str, float],
    field_schemas: dict[str, Any] | None = None,
) -> np.ndarray:
    """Vectorized ``(n_actual, n_expected)`` matrix of mismatched-subfield counts.

    Bit-identical to the per-pair ``sum(not _cell_match(...))`` build but without
    the n*m*k Python loop: each exact-match subfield is interned to ints and
    compared with one broadcasted ``!=`` (each cell normalized once, not once per
    pair). Fuzzy fields and unhashable cells fall back to the original pairwise
    compare for that column only. The matrix is ``int16`` (counts are bounded by
    ``len(subfields)``) -- 4x smaller than ``float64``, which bounds the build
    time and memory of very large matrices.
    """
    na, ne = len(actual_list), len(expected_list)
    cost = np.zeros((na, ne), dtype=np.int16)
    if na == 0 or ne == 0:
        return cost
    for field in subfields:
        field_schema = (field_schemas or {}).get(field)
        interned = None
        if fuzzy_field_thresholds.get(field) is None:
            interned = _intern_field(actual_list, expected_list, field, field_schema)
        if interned is not None:
            act_ids, exp_ids = interned
            cost += act_ids[:, None] != exp_ids[None, :]
            continue
        # Fuzzy threshold or unhashable cell: original per-pair compare, this column only.
        for i, actual_row in enumerate(actual_list):
            actual_dict = actual_row if isinstance(actual_row, dict) else {}
            actual_value = actual_dict.get(field)
            cost_row = cost[i]
            for j, expected_row in enumerate(expected_list):
                expected_dict = expected_row if isinstance(expected_row, dict) else {}
                if not _cell_match(expected_dict.get(field), actual_value, field, fuzzy_field_thresholds, field_schema):
                    cost_row[j] += 1
    return cost


def _assign_rows(
    actual_list: list[Any],
    expected_list: list[Any],
    subfields: list[str],
    fuzzy_field_thresholds: dict[str, float],
    field_schemas: dict[str, Any] | None = None,
) -> list[tuple[int, int]]:
    assignment = _peel_exact_row_matches(actual_list, expected_list, subfields, fuzzy_field_thresholds, field_schemas)
    pairs = list(assignment.pairs)
    if not assignment.unmatched_actual_indices or not assignment.unmatched_expected_indices:
        return pairs

    residual_actual = [actual_list[idx] for idx in assignment.unmatched_actual_indices]
    residual_expected = [expected_list[idx] for idx in assignment.unmatched_expected_indices]
    # Memory ceiling on the residual assignment matrix (mirrors the
    # unified_evidence value-matrix guard). After the exact-row peel, a divergent
    # giant array leaves a residual ~= the full array; the dense int16 cost
    # matrix plus scipy's internal float64 copy is tens of GB at ~50k+ rows (a
    # 77,400-row array needs ~67 GB) and OOM-kills the eval runner. Over the cell
    # budget, skip the residual assignment: those rows stay unmatched (scored as
    # misses) rather than crashing the whole run -- a peel-only lower bound.
    if len(residual_actual) * len(residual_expected) > _MAX_RESIDUAL_ASSIGNMENT_CELLS:
        return pairs
    cost = _mismatch_cost_matrix(residual_actual, residual_expected, subfields, fuzzy_field_thresholds, field_schemas)
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs.extend(
        (assignment.unmatched_actual_indices[int(actual_idx)], assignment.unmatched_expected_indices[int(expected_idx)])
        for actual_idx, expected_idx in zip(row_ind, col_ind, strict=True)
    )
    return pairs


def _is_array_schema(field_schema: dict[str, Any], expected_value: Any) -> bool:
    field_type = field_schema.get("type")
    return field_type == "array" or isinstance(expected_value, list)


def _array_subfields(field_schema: dict[str, Any], expected_rows: Any) -> list[str]:
    item_props = field_schema.get("items", {}).get("properties", {})
    if item_props:
        return list(item_props.keys())
    if isinstance(expected_rows, list):
        keys: set[str] = set()
        for row in expected_rows:
            if isinstance(row, dict):
                keys.update(row.keys())
        return sorted(keys)
    return []


def _score_array(
    expected_rows: Any,
    actual_rows: Any,
    subfields: list[str],
    fuzzy_field_thresholds: dict[str, float],
    field_schemas: dict[str, Any] | None = None,
) -> tuple[int, int, int, int, int]:
    expected_list = expected_rows if isinstance(expected_rows, list) else []
    actual_list = actual_rows if isinstance(actual_rows, list) else []
    expected_total = len(expected_list) * len(subfields)
    predicted_total = len(actual_list) * len(subfields)

    if not expected_list or not subfields or not actual_list:
        return 0, expected_total, predicted_total, len(expected_list), len(actual_list)

    correct = 0
    for actual_idx, expected_idx in _assign_rows(
        actual_list, expected_list, subfields, fuzzy_field_thresholds, field_schemas
    ):
        actual_dict = actual_list[actual_idx] if isinstance(actual_list[actual_idx], dict) else {}
        expected_dict = expected_list[expected_idx] if isinstance(expected_list[expected_idx], dict) else {}
        correct += sum(
            _cell_match(
                expected_dict.get(field),
                actual_dict.get(field),
                field,
                fuzzy_field_thresholds,
                (field_schemas or {}).get(field),
            )
            for field in subfields
        )

    return correct, expected_total, predicted_total, len(expected_list), len(actual_list)


def compute_array_record_match_counts(
    expected: Any,
    actual: Any,
    data_schema: dict[str, Any] | None = None,
    fuzzy_field_thresholds: dict[str, float] | None = None,
    normalize_dates: bool = True,
) -> ArrayRecordMatchCounts | None:
    """Compute order-insensitive record matching counts for top-level arrays.

    The denominator follows the public LongArray-Extract scorer: every expected
    scalar field is one data point, and every expected array row subfield is one
    data point. Array row order is ignored through optimal one-to-one matching.

    When ``normalize_dates`` is set (the default), date-like strings on both
    sides are canonicalized to ISO format before comparison, aligned with the
    generic ``accuracy`` metric (json_subset_match).
    """
    expected = _unwrap_value(expected)
    actual = _unwrap_value(actual)
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return None
    if normalize_dates:
        expected = _normalize_dates_deep(expected)
        actual = _normalize_dates_deep(actual)

    schema_props = (data_schema or {}).get("properties", {})
    fuzzy = dict(DEFAULT_FUZZY_FIELD_THRESHOLDS if fuzzy_field_thresholds is None else fuzzy_field_thresholds)

    correct = 0
    expected_total = 0
    predicted_total = 0
    expected_rows = 0
    predicted_rows = 0
    arrays_scored = 0
    scalar_fields_scored = 0

    for field in sorted((set(expected) | set(actual)) - RESERVED_OUTPUT_KEYS):
        field_schema = schema_props.get(field, {})
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if _is_array_schema(field_schema, expected_value):
            subfields = _array_subfields(field_schema, expected_value)
            if not subfields:
                continue
            item_props = field_schema.get("items", {}).get("properties", {}) if isinstance(field_schema, dict) else {}
            field_schemas = item_props if isinstance(item_props, dict) else {}
            (
                array_correct,
                array_expected_total,
                array_predicted_total,
                array_expected_rows,
                array_predicted_rows,
            ) = _score_array(expected_value, actual_value, subfields, fuzzy, field_schemas)
            correct += array_correct
            expected_total += array_expected_total
            predicted_total += array_predicted_total
            expected_rows += array_expected_rows
            predicted_rows += array_predicted_rows
            arrays_scored += 1
        else:
            expected_total += 1
            # An omitted key is an implicit null prediction and always counts in
            # the precision denominator (an omitted GT-null field matches via
            # ``_cell_match(None, None)``, so gating on ``field in actual`` let
            # correct exceed predicted_total and precision run past 1.0).
            predicted_total += 1
            scalar_fields_scored += 1
            if _cell_match(expected_value, actual_value, field, fuzzy, field_schema):
                correct += 1

    if arrays_scored == 0 or expected_total <= 0:
        return None

    return ArrayRecordMatchCounts(
        correct=correct,
        expected_total=expected_total,
        predicted_total=predicted_total,
        expected_rows=expected_rows,
        predicted_rows=predicted_rows,
        arrays_scored=arrays_scored,
        scalar_fields_scored=scalar_fields_scored,
    )


class ArrayRecordMatchMetric:
    """Metric bundle for long-array extraction quality."""

    def __init__(self, fuzzy_field_thresholds: dict[str, float] | None = None, normalize_dates: bool = True):
        self._fuzzy_field_thresholds = fuzzy_field_thresholds
        self._normalize_dates = normalize_dates

    def compute(self, expected: Any, actual: Any, **kwargs: Any) -> list[MetricValue]:
        data_schema = kwargs.get("data_schema")
        fuzzy_field_thresholds = kwargs.get("fuzzy_field_thresholds", self._fuzzy_field_thresholds)
        normalize_dates = kwargs.get("normalize_dates", self._normalize_dates)
        counts = compute_array_record_match_counts(
            expected=expected,
            actual=actual,
            data_schema=data_schema,
            fuzzy_field_thresholds=fuzzy_field_thresholds,
            normalize_dates=normalize_dates,
        )
        if counts is None:
            return []

        accuracy = counts.correct / counts.expected_total if counts.expected_total else 0.0
        precision = counts.correct / counts.predicted_total if counts.predicted_total else 0.0
        recall = accuracy
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        row_count_ratio = counts.predicted_rows / counts.expected_rows if counts.expected_rows else 0.0

        common_metadata = {
            "correct": counts.correct,
            "expected_total": counts.expected_total,
            "predicted_total": counts.predicted_total,
            "expected_rows": counts.expected_rows,
            "predicted_rows": counts.predicted_rows,
            "arrays_scored": counts.arrays_scored,
            "scalar_fields_scored": counts.scalar_fields_scored,
            "fuzzy_field_thresholds": dict(
                DEFAULT_FUZZY_FIELD_THRESHOLDS if fuzzy_field_thresholds is None else fuzzy_field_thresholds
            ),
            "normalize_dates": normalize_dates,
        }
        prf_metadata = {
            **common_metadata,
            "tp": counts.correct,
            "fp": counts.false_positive,
            "fn": counts.false_negative,
        }

        return [
            MetricValue(
                metric_name="array_record_accuracy",
                value=accuracy,
                metadata={
                    **common_metadata,
                    "passed": counts.correct,
                    "total": counts.expected_total,
                    "denominator": "expected_data_points",
                },
            ),
            MetricValue(metric_name="array_record_precision", value=precision, metadata=prf_metadata),
            MetricValue(metric_name="array_record_recall", value=recall, metadata=prf_metadata),
            MetricValue(metric_name="array_record_f1", value=f1, metadata=prf_metadata),
            MetricValue(
                metric_name="array_record_row_count_ratio",
                value=row_count_ratio,
                metadata={
                    **common_metadata,
                    "count": counts.predicted_rows,
                },
            ),
        ]
