"""Generic test-rule filters shared by evaluation runners."""

from __future__ import annotations

from typing import Any

from extract_bench.test_cases.schema import TestCase


def is_verified_rule(rule: Any) -> bool:
    """Return False only for rules explicitly marked ``verified=False``."""
    if isinstance(rule, dict):
        return rule.get("verified", True) is not False

    getter = getattr(rule, "get", None)
    if callable(getter):
        return getter("verified", True) is not False

    return getattr(rule, "verified", True) is not False


def filter_verified_test_rules(test_case: TestCase) -> TestCase:
    """Return a copy of ``test_case`` with unverified test rules removed."""
    rules = getattr(test_case, "test_rules", None)
    if not rules:
        return test_case

    filtered_rules = [rule for rule in rules if is_verified_rule(rule)]
    if len(filtered_rules) == len(rules):
        return test_case

    return test_case.model_copy(update={"test_rules": filtered_rules})
