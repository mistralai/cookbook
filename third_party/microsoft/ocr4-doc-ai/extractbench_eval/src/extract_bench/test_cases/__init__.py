"""Test case management for inference and evaluation."""

from extract_bench.test_cases.loader import load_test_case, load_test_cases
from extract_bench.test_cases.rule_filters import filter_verified_test_rules, is_verified_rule
from extract_bench.test_cases.schema import (
    BaseTestCase,
    ExtractFieldBbox,
    ExtractFieldTestRule,
    ExtractTestCase,
    ParseTestCase,
    TestCase,
)

__all__ = [
    "BaseTestCase",
    "ExtractFieldBbox",
    "ExtractFieldTestRule",
    "ExtractTestCase",
    "ParseTestCase",
    "TestCase",
    "filter_verified_test_rules",
    "is_verified_rule",
    "load_test_case",
    "load_test_cases",
]
