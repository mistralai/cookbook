"""v0.2 evidence-list value comparators.

This module is intentionally separate from the legacy source-export
``value_compare`` helpers. v0.2 comparator names describe normalization
policies for evidence values, not only JSON-schema primitive types.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from datetime import date, datetime
from typing import Any

from dateutil import parser as date_parser
from rapidfuzz.distance import JaroWinkler

from extract_bench.evaluation.metrics.field_grounding.core import ValueComparison
from extract_bench.test_cases.schema import ComparatorSpec

_DEFAULT_COMPARATOR = "case_insensitive"
_STRING_THRESHOLD = 0.90


def compare_evidence_value(
    expected: Any,
    actual: Any,
    comparator: ComparatorSpec | None = None,
) -> ValueComparison:
    """Compare one expected evidence value against one prediction."""
    spec = comparator or _DEFAULT_COMPARATOR
    if isinstance(spec, dict):
        return _compare_object(expected, actual, spec)
    return _compare_scalar(expected, actual, spec)


def compare_evidence_array(
    expected_items: list[Any],
    actual: Any,
    *,
    comparator: ComparatorSpec | None = None,
    structural: str | None = None,
    exhaustive: bool | None = None,
    expected_min: int | None = None,
) -> ValueComparison:
    """Compare an array-valued field using v0.2 structural semantics."""
    if not isinstance(actual, list):
        return ValueComparison(False, 0.0, structural or "array", "actual_not_array")

    min_count = len(expected_items) if expected_min is None else expected_min
    if len(expected_items) < min_count:
        return ValueComparison(False, 0.0, structural or "array", "insufficient_expected_items")

    mode = structural or "ordered"
    is_exhaustive = True if exhaustive is None else exhaustive
    if mode.startswith("match_by:"):
        key = mode.split(":", 1)[1]
        return _compare_match_by(expected_items, actual, key, comparator, is_exhaustive, expected_min=expected_min)
    if mode == "set":
        return _compare_unordered(
            expected_items,
            actual,
            comparator,
            exhaustive=is_exhaustive,
            multiset=False,
            expected_min=expected_min,
        )
    if mode == "multiset":
        return _compare_unordered(
            expected_items,
            actual,
            comparator,
            exhaustive=is_exhaustive,
            multiset=True,
            expected_min=expected_min,
        )
    return _compare_ordered(expected_items, actual, comparator, exhaustive=is_exhaustive)


def _compare_object(expected: Any, actual: Any, comparator: dict[str, str]) -> ValueComparison:
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return ValueComparison(False, 0.0, "object", "object_type_mismatch")
    if not comparator:
        return ValueComparison(True, 1.0, "object", "pass")

    scores: list[float] = []
    for key, key_comparator in comparator.items():
        if key not in expected:
            continue
        if key not in actual:
            return ValueComparison(False, 0.0, "object", f"missing_key:{key}")
        comparison = compare_evidence_value(expected[key], actual[key], key_comparator)
        if not comparison.passed:
            return ValueComparison(False, comparison.score, f"object.{key_comparator}", f"{key}:{comparison.reason}")
        scores.append(comparison.score)
    score = sum(scores) / len(scores) if scores else 1.0
    return ValueComparison(True, score, "object", "pass")


def _compare_scalar(expected: Any, actual: Any, comparator: str) -> ValueComparison:
    name = comparator.casefold().strip()
    if expected is None:
        passed = actual is None or _normalized_text(actual) == ""
        return ValueComparison(passed, 1.0 if passed else 0.0, "null", "pass" if passed else "non_null_actual")

    if name in {"exact", "identity"}:
        return _exact(expected, actual)
    if name in {"case_insensitive", "enum", "name", "address"}:
        return _case_insensitive(expected, actual, mode=name)
    if name in {"string_substring", "substring", "contains"}:
        return _substring(expected, actual)
    if name in {"number", "number_with_unit", "currency"}:
        return _number(expected, actual, mode=name)
    if name == "date":
        return _date(expected, actual)
    if name == "phone":
        return _phone(expected, actual)
    if name == "boolean":
        return _boolean(expected, actual)

    return ValueComparison(False, 0.0, name or "unknown", f"unknown_comparator:{comparator}")


def _exact(expected: Any, actual: Any) -> ValueComparison:
    if isinstance(expected, str) or isinstance(actual, str):
        passed = _collapse_ws(str(expected)) == _collapse_ws("" if actual is None else str(actual))
    else:
        passed = expected == actual
    return ValueComparison(passed, 1.0 if passed else 0.0, "exact", "pass" if passed else "exact_mismatch")


def _case_insensitive(expected: Any, actual: Any, *, mode: str) -> ValueComparison:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return _number(expected, actual, mode="number")
    if isinstance(expected, bool):
        return _boolean(expected, actual)
    expected_norm = _normalized_text(expected)
    actual_norm = _normalized_text(actual)
    passed = expected_norm == actual_norm
    score = 1.0 if passed else float(JaroWinkler.normalized_similarity(expected_norm, actual_norm))
    if not passed and score >= _STRING_THRESHOLD:
        return ValueComparison(True, score, mode, "pass")
    return ValueComparison(passed, score, mode, "pass" if passed else "case_insensitive_mismatch")


def _substring(expected: Any, actual: Any) -> ValueComparison:
    expected_norm = _normalized_text(expected)
    actual_norm = _normalized_text(actual)
    passed = bool(expected_norm) and (expected_norm in actual_norm or actual_norm in expected_norm)
    score = 1.0 if passed else float(JaroWinkler.normalized_similarity(expected_norm, actual_norm))
    return ValueComparison(passed, score, "string_substring", "pass" if passed else "substring_mismatch")


def _number(expected: Any, actual: Any, *, mode: str) -> ValueComparison:
    expected_num = _parse_number(expected)
    actual_num = _parse_number(actual)
    passed = (
        expected_num is not None
        and actual_num is not None
        and math.isclose(
            expected_num,
            actual_num,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    )
    return ValueComparison(passed, 1.0 if passed else 0.0, mode, "pass" if passed else "number_mismatch")


def _date(expected: Any, actual: Any) -> ValueComparison:
    expected_date = _parse_date(expected)
    actual_date = _parse_date(actual)
    passed = expected_date is not None and actual_date is not None and expected_date == actual_date
    return ValueComparison(passed, 1.0 if passed else 0.0, "date", "pass" if passed else "date_mismatch")


def _phone(expected: Any, actual: Any) -> ValueComparison:
    expected_digits = re.sub(r"\D+", "", "" if expected is None else str(expected))
    actual_digits = re.sub(r"\D+", "", "" if actual is None else str(actual))
    if len(expected_digits) >= 10 and len(actual_digits) >= 10:
        passed = expected_digits[-10:] == actual_digits[-10:]
    else:
        passed = expected_digits == actual_digits and bool(expected_digits)
    return ValueComparison(passed, 1.0 if passed else 0.0, "phone", "pass" if passed else "phone_mismatch")


def _boolean(expected: Any, actual: Any) -> ValueComparison:
    expected_bool = _parse_bool(expected)
    actual_bool = _parse_bool(actual)
    passed = expected_bool is not None and actual_bool is not None and expected_bool is actual_bool
    return ValueComparison(passed, 1.0 if passed else 0.0, "boolean", "pass" if passed else "boolean_mismatch")


def _compare_ordered(
    expected_items: list[Any],
    actual_items: list[Any],
    comparator: ComparatorSpec | None,
    *,
    exhaustive: bool,
) -> ValueComparison:
    if exhaustive and len(expected_items) != len(actual_items):
        return ValueComparison(False, 0.0, "ordered", "array_length_mismatch")
    if len(actual_items) < len(expected_items):
        return ValueComparison(False, 0.0, "ordered", "missing_array_items")
    scores: list[float] = []
    for expected, actual in zip(expected_items, actual_items, strict=False):
        comparison = compare_evidence_value(expected, actual, comparator)
        if not comparison.passed:
            return ValueComparison(False, comparison.score, "ordered", comparison.reason)
        scores.append(comparison.score)
    return ValueComparison(True, _avg(scores), "ordered", "pass")


def _compare_unordered(
    expected_items: list[Any],
    actual_items: list[Any],
    comparator: ComparatorSpec | None,
    *,
    exhaustive: bool,
    multiset: bool,
    expected_min: int | None = None,
) -> ValueComparison:
    if multiset and not comparator:
        expected_counter = Counter(_stable_key(item) for item in expected_items)
        actual_counter = Counter(_stable_key(item) for item in actual_items)
        if exhaustive and expected_counter != actual_counter:
            return ValueComparison(False, 0.0, "multiset", "multiset_mismatch")
        passed = all(actual_counter[key] >= count for key, count in expected_counter.items())
        return ValueComparison(passed, 1.0 if passed else 0.0, "multiset", "pass" if passed else "missing_array_items")

    # v0.2 evidence semantics: each entry in `expected_items` is an OR-acceptable variant
    # (e.g. the same logical value cited at multiple PDF locations, or a fine vs coarse
    # surface form of the same item). Coverage is measured over the predictions: how many
    # actual items are covered by *some* expected variant. Pass when matched_count meets
    # the floor, which is `expected_min` if explicitly set, otherwise the number of
    # distinct expected values (computed via `_stable_key`).
    mode = "multiset" if multiset else "set"
    scores: list[float] = []
    matched_actual: list[int] = []
    extra_actual: list[int] = []
    for index, actual in enumerate(actual_items):
        best: ValueComparison | None = None
        for expected in expected_items:
            comparison = compare_evidence_value(expected, actual, comparator)
            if comparison.passed and (best is None or comparison.score > best.score):
                best = comparison
        if best is not None:
            matched_actual.append(index)
            scores.append(best.score)
        else:
            extra_actual.append(index)

    if expected_min is not None:
        threshold = expected_min
    else:
        threshold = len({_stable_key(e) for e in expected_items})

    matched_count = len(matched_actual)
    avg_score = _avg(scores) if scores else 0.0

    if matched_count < threshold:
        return ValueComparison(False, avg_score, mode, "missing_array_items")
    if exhaustive and extra_actual:
        return ValueComparison(False, avg_score, mode, "extra_array_items")
    return ValueComparison(True, avg_score, mode, "pass")


def parse_match_by_keys(key_spec: str) -> list[str]:
    """Split a ``match_by:`` key spec into identity keys.

    The GT builder emits composite identities as a comma-joined list
    (``match_by:cusip,title_of_class``); a row matches only when every key
    passes its per-key comparator.
    """
    return [key.strip() for key in key_spec.split(",") if key.strip()]


def _match_by_row_passes_keys(
    expected: dict[str, Any],
    actual: dict[str, Any],
    keys: list[str],
    comparator: ComparatorSpec | None,
) -> bool:
    for key in keys:
        key_comparator = comparator.get(key) if isinstance(comparator, dict) else None
        # Absent and null are equivalent in extraction output: providers may
        # omit keys whose value is null, so a null gold cell matches a missing
        # actual key.
        if not compare_evidence_value(expected[key], actual.get(key), key_comparator).passed:
            return False
    return True


def _compare_match_by(
    expected_items: list[Any],
    actual_items: list[Any],
    key: str,
    comparator: ComparatorSpec | None,
    exhaustive: bool,
    expected_min: int | None = None,
) -> ValueComparison:
    keys = parse_match_by_keys(key)
    if not keys:
        return ValueComparison(False, 0.0, "match_by", "missing_match_key")

    # Validate gold shape
    for expected in expected_items:
        if not isinstance(expected, dict):
            return ValueComparison(False, 0.0, "match_by", f"expected_missing_key:{keys[0]}")
        for sub_key in keys:
            if sub_key not in expected:
                return ValueComparison(False, 0.0, "match_by", f"expected_missing_key:{sub_key}")

    # v0.2 evidence semantics: expected_items is a list of OR-acceptable variants keyed by `key`.
    # Multiple entries with the same `key` value (different surface forms or different PDF locations)
    # describe the same logical row. Coverage is measured over predictions: each actual is "covered"
    # if some expected entry's `key` matches AND the full object compare passes. Pass when matched
    # count meets the floor (expected_min if set, else number of distinct expected key values).
    scores: list[float] = []
    matched_actual: list[int] = []
    extra_actual: list[int] = []
    for index, actual in enumerate(actual_items):
        if not isinstance(actual, dict):
            extra_actual.append(index)
            continue
        best: ValueComparison | None = None
        for expected in expected_items:
            if not _match_by_row_passes_keys(expected, actual, keys, comparator):
                continue
            # Null cells in a gold row mean "not consensus-verified", not "assert
            # null" — the GT builder nulls out cells it could not verify while
            # the corresponding per-leaf rules carry any real null assertions.
            # Grade the row only on its verified (non-null) cells.
            verified_expected = {cell: value for cell, value in expected.items() if value is not None}
            full_cmp = compare_evidence_value(verified_expected, actual, comparator)
            if full_cmp.passed and (best is None or full_cmp.score > best.score):
                best = full_cmp
        if best is not None:
            matched_actual.append(index)
            scores.append(best.score)
        else:
            extra_actual.append(index)

    if expected_min is not None:
        threshold = expected_min
    else:
        threshold = len({tuple(_stable_key(e[sub_key]) for sub_key in keys) for e in expected_items})

    matched_count = len(matched_actual)
    avg_score = _avg(scores) if scores else 0.0

    if matched_count < threshold:
        return ValueComparison(False, avg_score, "match_by", f"missing_match:{key}")
    if exhaustive and extra_actual:
        return ValueComparison(False, avg_score, "match_by", "extra_array_items")
    return ValueComparison(True, avg_score, "match_by", "pass")


def _parse_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    negative = bool(re.search(r"\([^)]*\)", text))
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return None
    try:
        number = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return -abs(number) if negative else number


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        parsed: datetime = date_parser.parse(str(value), fuzzy=True)
        return parsed.date()
    except (ValueError, OverflowError, TypeError):
        return None


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _normalized_text(value)
    if text in {"true", "yes", "y", "1", "checked", "x"}:
        return True
    if text in {"false", "no", "n", "0", "unchecked", ""}:
        return False
    return None


def _normalized_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.casefold().split())


def _collapse_ws(value: str) -> str:
    return " ".join(value.split())


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def _stable_key(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_stable_key(value[key])}" for key in sorted(value)) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_stable_key(item) for item in value) + "]"
    return _normalized_text(value)


def stable_value_key(value: Any) -> str:
    """Normalized identity used for set/multiset/match_by dedup and row alignment."""
    return _stable_key(value)
