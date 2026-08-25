"""Normalize list-rooted per_table_row extract predictions for evaluation.

This module is a pure **shape adapter**. It does not emit any metrics of
its own. The existing ``precision``, ``recall``, ``f1``, ``accuracy``, and
``extract_field_value_pass_rate`` metrics are what score correctly once
the prediction is normalized; downstream metadata on those metrics carries
normalization state for debugging and dashboard drill-down.

The v0.5 test cases were authored for ``extraction_target=per_doc``, so every
``ExtractFieldTestRule.field_path`` is dict-rooted (e.g. ``personnel[0].name``,
``client_id``). ``extraction_target=per_table_row`` may emit a bare row list::

    extracted_data = [{"name": "Alice", ...}, {"name": "Bob", ...}]

or, when the API is given the original per-doc schema, a list of document-shaped
wrappers where each wrapper contains the inferred array field::

    extracted_data = [
        {"client_id": "C-1", "personnel": [{"name": "Alice"}]},
        {"client_id": "C-1", "personnel": [{"name": "Bob"}]},
    ]

A full source-export corpus run also exposed singleton scalar-document lists and
multi-array wrapper lists. The extract evaluator's path walkers are all
dict-rooted, so this module projects those list-rooted shapes back into the
per-doc schema shape before scoring.

Scope
-----
- Dict-rooted predictions are a no-op.
- Singleton scalar-document lists become the singleton dict.
- Wrapper rows merge all top-level list fields and preserve representative
  scalar fields.
- Bare row lists still require one canonical array prefix; scalar rules are
  skipped only in this mode because bare rows cannot structurally emit them.
- Case-only rule aliases are skipped when the JSON schema has a unique
  canonical top-level key.

Accuracy caveat
---------------
The whole-JSON ``JsonSubsetMatchMetric`` intentionally still operates on the
unmodified ``expected_output`` vs the unwrapped ``extracted_data``. Accuracy
will honestly drop on per_table_row runs because scalar fields like
``client_id`` may be missing from bare-row predictions. That drop is a
correct signal for whole-JSON accuracy, and is separate from the rule-level
P/R/F1 this adapter fixes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from extract_bench.test_cases.extract_field_paths import parse_field_path
from extract_bench.test_cases.schema import ExtractFieldTestRule


@dataclass(frozen=True)
class ListPredictionNormalization:
    """Result of projecting a list-rooted prediction into evaluator shape."""

    extracted_data: Any
    applied: bool
    mode: str
    skipped_field_paths: list[str] = field(default_factory=list)
    alias_skipped_field_paths: list[str] = field(default_factory=list)
    normalized_top_level_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def infer_array_field(rules: Iterable[ExtractFieldTestRule]) -> str | None:
    """Return the single top-level array-prefix used by all array-rooted rules.

    For paths like ``personnel[0].name`` and ``personnel[1].net_pay``, returns
    ``"personnel"``. Returns ``None`` if rules span multiple array prefixes or
    if no rule is array-rooted.
    """
    array_prefixes: set[str] = set()
    for rule in rules:
        try:
            tokens = parse_field_path(rule.field_path)
        except ValueError:
            continue
        # Array-rooted rule: first token is a string, second is an int.
        if len(tokens) >= 2 and isinstance(tokens[0], str) and isinstance(tokens[1], int):
            array_prefixes.add(tokens[0])
    if len(array_prefixes) == 1:
        return next(iter(array_prefixes))
    return None


def _parse_tokens(field_path: str) -> list[str | int] | None:
    try:
        return list(parse_field_path(field_path))
    except ValueError:
        return None


def _top_level_field(field_path: str) -> str | None:
    tokens = _parse_tokens(field_path)
    if tokens and isinstance(tokens[0], str):
        return tokens[0]
    return None


def _is_scalar_rooted_path(field_path: str) -> bool:
    """Return True if ``field_path`` doesn't traverse any array.

    Top-level scalars (``client_id``) and nested-dict-only paths
    (``buyer.company``) are considered scalar-rooted. Any path containing a
    numeric index token is array-rooted.
    """
    tokens = _parse_tokens(field_path)
    if tokens is None:
        return False
    return not any(isinstance(token, int) for token in tokens)


def _is_array_rooted_path(field_path: str) -> bool:
    tokens = _parse_tokens(field_path)
    return bool(tokens and len(tokens) >= 2 and isinstance(tokens[0], str) and isinstance(tokens[1], int))


def _schema_canonical_key_map(data_schema: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(data_schema, dict):
        return {}
    properties = data_schema.get("properties")
    if not isinstance(properties, dict):
        return {}

    by_casefold: dict[str, list[str]] = {}
    for key in properties:
        if isinstance(key, str):
            by_casefold.setdefault(key.casefold(), []).append(key)
    return {folded: keys[0] for folded, keys in by_casefold.items() if len(keys) == 1}


def _canonicalize_key(key: str, canonical_keys: dict[str, str]) -> str:
    return canonical_keys.get(key.casefold(), key)


def _array_prefixes(
    rules: Iterable[ExtractFieldTestRule],
    canonical_keys: dict[str, str],
) -> set[str]:
    prefixes: set[str] = set()
    for rule in rules:
        if not _is_array_rooted_path(rule.field_path):
            continue
        top_level = _top_level_field(rule.field_path)
        if top_level is not None:
            prefixes.add(_canonicalize_key(top_level, canonical_keys))
    return prefixes


def _alias_skipped_field_paths(
    rules: Iterable[ExtractFieldTestRule],
    canonical_keys: dict[str, str],
) -> list[str]:
    skipped: list[str] = []
    for rule in rules:
        top_level = _top_level_field(rule.field_path)
        if top_level is None:
            continue
        canonical = _canonicalize_key(top_level, canonical_keys)
        if canonical != top_level:
            skipped.append(rule.field_path)
    return skipped


def _all_items_are_dicts(extracted_data: list[Any]) -> bool:
    return all(isinstance(item, dict) for item in extracted_data)


def _has_list_valued_field(extracted_data: list[Any]) -> bool:
    return any(isinstance(value, list) for item in extracted_data if isinstance(item, dict) for value in item.values())


def _merge_wrapper_rows(
    extracted_data: list[Any],
    canonical_keys: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    merged: dict[str, Any] = {}
    scalar_values: dict[str, Any] = {}
    scalar_conflicts: dict[str, set[str]] = {}

    for item in extracted_data:
        if not isinstance(item, dict):
            continue
        for raw_key, value in item.items():
            if not isinstance(raw_key, str):
                continue
            key = _canonicalize_key(raw_key, canonical_keys)
            if isinstance(value, list):
                existing = merged.setdefault(key, [])
                if isinstance(existing, list):
                    existing.extend(row for row in value if row is not None)
                continue

            if _is_empty_scalar(value):
                continue
            if key not in scalar_values:
                scalar_values[key] = value
            elif scalar_values[key] != value:
                scalar_conflicts.setdefault(key, {repr(scalar_values[key])}).add(repr(value))

    for key, value in scalar_values.items():
        merged.setdefault(key, value)

    warnings = [
        f"conflicting scalar values for {key}: {sorted(values)}" for key, values in sorted(scalar_conflicts.items())
    ]
    return merged, warnings


def _is_empty_scalar(value: Any) -> bool:
    return value is None or value == ""


def normalize_list_prediction(
    extracted_data: Any,
    rules: Iterable[ExtractFieldTestRule],
    *,
    data_schema: dict[str, Any] | None = None,
) -> ListPredictionNormalization:
    """Project list-rooted predictions into the dict-rooted evaluator shape."""
    if not isinstance(extracted_data, list):
        return ListPredictionNormalization(
            extracted_data=extracted_data,
            applied=False,
            mode="no_op",
            normalized_top_level_keys=sorted(extracted_data.keys()) if isinstance(extracted_data, dict) else [],
        )

    rules_list = list(rules)
    canonical_keys = _schema_canonical_key_map(data_schema)
    alias_skipped = _alias_skipped_field_paths(rules_list, canonical_keys)
    alias_skipped_set = set(alias_skipped)
    scoreable_rules = [rule for rule in rules_list if rule.field_path not in alias_skipped_set]
    array_prefixes = _array_prefixes(scoreable_rules, canonical_keys)

    if not extracted_data:
        if len(array_prefixes) == 1:
            array_field = next(iter(array_prefixes))
            return ListPredictionNormalization(
                extracted_data={array_field: []},
                applied=True,
                mode="bare_rows",
                skipped_field_paths=[
                    rule.field_path for rule in scoreable_rules if _is_scalar_rooted_path(rule.field_path)
                ],
                alias_skipped_field_paths=alias_skipped,
                normalized_top_level_keys=[array_field],
            )
        return ListPredictionNormalization(
            extracted_data=extracted_data,
            applied=False,
            mode="no_op",
            alias_skipped_field_paths=alias_skipped,
        )

    if (
        (rules_list or canonical_keys)
        and not array_prefixes
        and len(extracted_data) == 1
        and isinstance(extracted_data[0], dict)
        and not _has_list_valued_field(extracted_data)
    ):
        normalized = {
            _canonicalize_key(key, canonical_keys): value
            for key, value in extracted_data[0].items()
            if isinstance(key, str)
        }
        return ListPredictionNormalization(
            extracted_data=normalized,
            applied=True,
            mode="singleton_doc",
            alias_skipped_field_paths=alias_skipped,
            normalized_top_level_keys=sorted(normalized.keys()),
        )

    if _all_items_are_dicts(extracted_data) and _has_list_valued_field(extracted_data):
        merged, warnings = _merge_wrapper_rows(extracted_data, canonical_keys)
        return ListPredictionNormalization(
            extracted_data=merged,
            applied=True,
            mode="wrapper_merge",
            alias_skipped_field_paths=alias_skipped,
            normalized_top_level_keys=sorted(merged.keys()),
            warnings=warnings,
        )

    if len(array_prefixes) == 1:
        array_field = next(iter(array_prefixes))
        skipped = [rule.field_path for rule in scoreable_rules if _is_scalar_rooted_path(rule.field_path)]
        return ListPredictionNormalization(
            extracted_data={array_field: extracted_data},
            applied=True,
            mode="bare_rows",
            skipped_field_paths=skipped,
            alias_skipped_field_paths=alias_skipped,
            normalized_top_level_keys=[array_field],
        )

    return ListPredictionNormalization(
        extracted_data=extracted_data,
        applied=False,
        mode="no_op",
        alias_skipped_field_paths=alias_skipped,
    )


def unwrap_list_prediction(
    extracted_data: Any,
    rules: Iterable[ExtractFieldTestRule],
    *,
    data_schema: dict[str, Any] | None = None,
) -> tuple[Any, bool, list[str]]:
    """Return ``(wrapped_data, unwrap_applied, skipped_field_paths)``.

    If ``extracted_data`` is a list and the rules share a single array-prefix,
    return a dict rooted at that prefix. Bare row lists become
    ``{prefix: extracted_data}``; wrapper-per-row lists become
    ``{prefix: flattened_rows}``. Also return the list of field_paths that
    don't touch any array (those can't be scored against a list-rooted
    prediction — caller should exclude them from denominators).

    If ``extracted_data`` is not a list or no single array-prefix is inferable,
    returns ``(extracted_data, False, [])`` unchanged. In particular, this is
    a no-op for the common per_doc case where predictions are already
    dict-rooted.
    """
    normalized = normalize_list_prediction(extracted_data, rules, data_schema=data_schema)
    return (
        normalized.extracted_data,
        normalized.applied,
        [*normalized.skipped_field_paths, *normalized.alias_skipped_field_paths],
    )
