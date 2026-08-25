"""Typed attribution comparison helpers for field grounding metrics."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Literal, cast

from extract_bench.evaluation.metrics.field_grounding.core import (
    STRING_MATCH_THRESHOLD,
    ValueComparison,
)
from extract_bench.evaluation.metrics.field_grounding.evidence_comparator import (
    compare_evidence_array,
    compare_evidence_value,
)
from extract_bench.test_cases.bbox_value_strict_comparator import (
    COMPARATOR_VERSION,
    ExpectedType,
    ExtractionSource,
)
from extract_bench.test_cases.bbox_value_strict_comparator import (
    compare as compare_bbox_value,
)
from extract_bench.test_cases.schema import ExtractFieldTestRule

AttributionSource = Literal["native", "ocr", "structured_value_no_citation_text"]

_DIAGNOSTIC_ONLY_MODES = frozenset({"annotation_truncated", "ocr_noise_prefix"})
_STRING_FALLBACK_TYPES = frozenset({"string", "date"})


def compare_attributed_value(
    expected_value: Any,
    actual_text: Any,
    *,
    expected_type: ExpectedType | None = None,
    source_kind: AttributionSource = "native",
    allow_diagnostic_equivalences: bool = False,
) -> ValueComparison:
    """Compare one expected field value against selected attribution text.

    The strict source-export comparator is the primary authority for typed
    equivalences. A Jaro-Winkler fallback is retained for string-shaped
    values, matching the field-grounding metric contract, but substring
    containment is intentionally never a passing mode here.
    """
    resolved_type = expected_type or infer_expected_type(expected_value)
    extraction_source: ExtractionSource = "ocr" if source_kind == "ocr" else "native"
    verdict = compare_bbox_value(
        expected_value,
        resolved_type,
        "" if actual_text is None else str(actual_text),
        extraction_source=extraction_source,
    )

    diagnostic_only = verdict.equivalence_used in _DIAGNOSTIC_ONLY_MODES and not allow_diagnostic_equivalences
    if verdict.verified and not diagnostic_only:
        return ValueComparison(
            passed=True,
            score=1.0,
            mode=verdict.equivalence_used,
            reason="pass",
        )

    score = float(verdict.similarity_score or 0.0)
    if resolved_type in _STRING_FALLBACK_TYPES and score >= STRING_MATCH_THRESHOLD:
        return ValueComparison(
            passed=True,
            score=score,
            mode="jaro_winkler",
            reason="pass",
        )

    reason = verdict.reason
    if diagnostic_only:
        reason = f"{verdict.equivalence_used}_diagnostic_only"
    return ValueComparison(
        passed=False,
        score=score,
        mode=verdict.equivalence_used if verdict.equivalence_used != "none" else "strict",
        reason=reason or "no_equivalence_rule_matched",
    )


def infer_expected_type(expected_value: Any) -> ExpectedType:
    """Infer a strict comparator type when schema metadata is unavailable."""
    if expected_value is None:
        return "null"
    if isinstance(expected_value, bool):
        return "boolean"
    if isinstance(expected_value, (int, float)):
        return "number"
    if isinstance(expected_value, str) and _looks_like_iso_date(expected_value):
        return "date"
    return "string"


def expected_type_for_field_path(
    data_schema: dict[str, Any] | None,
    field_path: str,
    expected_value: Any,
) -> ExpectedType:
    """Resolve a field's expected type from JSON schema, falling back safely."""
    schema_type = _schema_type_for_field_path(_freeze_schema(data_schema), field_path) if data_schema else None
    if schema_type in {"string", "number", "integer", "boolean", "null"}:
        if schema_type == "integer":
            return "number"
        return cast(ExpectedType, schema_type)
    return infer_expected_type(expected_value)


@lru_cache(maxsize=4096)
def _schema_type_for_field_path(schema_key: tuple[Any, ...], field_path: str) -> str | None:
    schema = _thaw_schema(schema_key)
    tokens = _parse_field_path_tokens(field_path)
    cursor: Any = schema

    for token in tokens:
        cursor = _descend_schema(cursor, token)
        if cursor is None:
            return None

    schema_type = cursor.get("type") if isinstance(cursor, dict) else None
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        return str(non_null[0]) if non_null else "null"
    return str(schema_type) if schema_type is not None else None


def _descend_schema(schema: Any, token: str | int) -> Any:
    if not isinstance(schema, dict):
        return None

    schema_type = schema.get("type")
    if isinstance(token, int):
        if schema_type == "array" or "items" in schema:
            return schema.get("items")
        return None

    if schema_type == "array" or ("items" in schema and "properties" not in schema):
        schema = schema.get("items")
        if not isinstance(schema, dict):
            return None

    properties = schema.get("properties")
    if isinstance(properties, dict) and token in properties:
        return properties[token]
    return None


def _parse_field_path_tokens(field_path: str) -> tuple[str | int, ...]:
    tokens: list[str | int] = []
    for part in field_path.split("."):
        if not part:
            continue
        match = re.match(r"^([^\[]+)", part)
        if match:
            tokens.append(match.group(1))
        for index in re.findall(r"\[(\d+)\]", part):
            tokens.append(int(index))
    return tuple(tokens)


def _looks_like_iso_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()))


def _freeze_schema(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_schema(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_schema(item) for item in value)
    return (value,)


def _thaw_schema(value: tuple[Any, ...]) -> Any:
    if not value:
        return None
    if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
        return {key: _thaw_schema(cast(tuple[Any, ...], item)) for key, item in value}
    if len(value) == 1 and not isinstance(value[0], tuple):
        return value[0]
    return [_thaw_schema(cast(tuple[Any, ...], item)) for item in value]


def compare_field_with_rule(
    rule: ExtractFieldTestRule | None,
    expected_value: Any,
    actual_text: Any,
    *,
    expected_type: ExpectedType | None = None,
    source_kind: AttributionSource = "native",
    allow_diagnostic_equivalences: bool = False,
) -> ValueComparison:
    """v0.2 dispatch shim — extract paths only.

    When ``rule.comparator`` is set, dispatches to a per-comparator implementation;
    otherwise falls through to ``compare_attributed_value`` (today's extract-side
    behavior).

    The shim accepts ``rule=None`` so legacy callers that don't have a rule object
    (record-level metrics, raw value comparison) can opt in incrementally without
    threading rule context everywhere.
    """
    if rule is not None and rule.evidence is not None:
        return compare_evidence_value(
            expected_value,
            actual_text,
            rule.comparator or "case_insensitive",
        )

    return compare_attributed_value(
        expected_value,
        actual_text,
        expected_type=expected_type,
        source_kind=source_kind,
        allow_diagnostic_equivalences=allow_diagnostic_equivalences,
    )


def candidate_values_for_rule(rule: ExtractFieldTestRule | None) -> list[Any]:
    """v0.2: yield the deduped set of candidate values for a rule (OR-over-evidence).

    When ``rule.evidence`` is set, returns only evidence values. Legacy rules
    return a single-element list with ``expected_value``. Empty rule returns
    ``[None]`` so null-expected rules still roundtrip through the comparator.
    """
    if rule is None:
        return [None]
    seen: set[Any] = set()
    candidates: list[Any] = []

    def _push(value: Any) -> None:
        try:
            key = ("hashable", value)
            hash(key)
            if key in seen:
                return
            seen.add(key)
        except TypeError:
            pass
        candidates.append(value)

    if rule.evidence is not None:
        for entry in rule.evidence:
            _push(entry.value)
        return candidates
    _push(rule.expected_value)
    return candidates


def compare_value_against_rule(
    rule: ExtractFieldTestRule | None,
    prediction: Any,
    *,
    expected_type: ExpectedType | None = None,
    source_kind: AttributionSource = "native",
    allow_diagnostic_equivalences: bool = False,
) -> ValueComparison:
    """Best-of comparison across a rule's candidate values (OR-over-evidence).

    For legacy rules with no v0.2 evidence list this collapses to a single
    ``compare_field_with_rule`` call against ``rule.expected_value`` — fully
    backward-compatible. For v0.2 rules with multiple evidence entries the result
    is the highest-scoring (passing > non-passing, then by score) comparison
    across all candidate values.
    """
    candidates = candidate_values_for_rule(rule)
    best: ValueComparison | None = None
    for candidate in candidates:
        comparison = compare_field_with_rule(
            rule,
            candidate,
            prediction,
            expected_type=expected_type,
            source_kind=source_kind,
            allow_diagnostic_equivalences=allow_diagnostic_equivalences,
        )
        if best is None:
            best = comparison
            continue
        if (comparison.passed and not best.passed) or (
            comparison.passed == best.passed and comparison.score > best.score
        ):
            best = comparison
    if best is None:
        return ValueComparison(passed=False, score=0.0, mode="missing", reason="no_candidates")
    return best


def compare_array_against_rule(
    rule: ExtractFieldTestRule,
    prediction: Any,
) -> ValueComparison:
    """Compare an array-valued v0.2 rule against a structured prediction."""
    if rule.evidence is None:
        return ValueComparison(passed=False, score=0.0, mode="array", reason="missing_evidence")
    expected_items = [entry.value for entry in rule.evidence]
    return compare_evidence_array(
        expected_items,
        prediction,
        comparator=rule.comparator or "case_insensitive",
        structural=rule.structural,
        exhaustive=rule.exhaustive,
        expected_min=rule.expected_min,
    )


__all__ = [
    "COMPARATOR_VERSION",
    "AttributionSource",
    "candidate_values_for_rule",
    "compare_array_against_rule",
    "compare_attributed_value",
    "compare_field_with_rule",
    "compare_value_against_rule",
    "expected_type_for_field_path",
    "infer_expected_type",
]
