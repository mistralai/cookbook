"""Helpers for selecting extract_field rules for evaluation."""

from collections.abc import Iterable

from extract_bench.test_cases.schema import ExtractFieldTestRule


def filter_extract_field_rules(
    rules: Iterable[ExtractFieldTestRule],
    *,
    require_bboxes: bool = False,
) -> list[ExtractFieldTestRule]:
    """Return extract_field rules matching evaluator-level rule filters."""
    filtered: list[ExtractFieldTestRule] = []
    for rule in rules:
        if require_bboxes and not rule.bboxes:
            continue
        filtered.append(rule)
    return filtered
