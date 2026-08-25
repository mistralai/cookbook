"""Tests for extract_field_paths module."""

from __future__ import annotations

import time

import pytest

from extract_bench.test_cases.extract_field_paths import (
    get_path,
    inflate_expected_output,
    parse_field_path,
    set_path,
    validate_rules_match_expected_output,
)
from extract_bench.test_cases.schema import ExtractFieldTestRule

# -----------------------------------------------------------------------------
# parse_field_path
# -----------------------------------------------------------------------------


def test_parse_simple_key() -> None:
    assert parse_field_path("a") == ["a"]


def test_parse_dotted_keys() -> None:
    assert parse_field_path("a.b") == ["a", "b"]


def test_parse_array_index() -> None:
    assert parse_field_path("a[0]") == ["a", 0]


def test_parse_array_then_key() -> None:
    assert parse_field_path("a[0].b") == ["a", 0, "b"]


def test_parse_nested_arrays() -> None:
    assert parse_field_path("a[0][1]") == ["a", 0, 1]


def test_parse_complex_path() -> None:
    assert parse_field_path("line_items[0].subitems[2].description") == [
        "line_items",
        0,
        "subitems",
        2,
        "description",
    ]


def test_parse_empty_raises() -> None:
    with pytest.raises(ValueError):
        parse_field_path("")


# -----------------------------------------------------------------------------
# set_path
# -----------------------------------------------------------------------------


def test_set_path_simple_key() -> None:
    target: dict = {}
    set_path(target, ["a"], 1)
    assert target == {"a": 1}


def test_set_path_nested_keys() -> None:
    target: dict = {}
    set_path(target, ["a", "b", "c"], 42)
    assert target == {"a": {"b": {"c": 42}}}


def test_set_path_array_element() -> None:
    target: dict = {}
    set_path(target, ["a", 0, "b"], "hi")
    assert target == {"a": [{"b": "hi"}]}


def test_set_path_auto_pads_with_none() -> None:
    target: dict = {}
    set_path(target, ["a", 3, "b"], "x")
    assert target == {"a": [None, None, None, {"b": "x"}]}


def test_set_path_does_not_clobber_siblings() -> None:
    target: dict = {}
    set_path(target, ["a", "x"], 1)
    set_path(target, ["a", "y"], 2)
    assert target == {"a": {"x": 1, "y": 2}}


def test_set_path_does_not_clobber_array_siblings() -> None:
    target: dict = {}
    set_path(target, ["items", 0, "name"], "A")
    set_path(target, ["items", 1, "name"], "B")
    assert target == {"items": [{"name": "A"}, {"name": "B"}]}


def test_set_path_type_mismatch_raises() -> None:
    target: dict = {"a": "scalar"}
    with pytest.raises(TypeError):
        set_path(target, ["a", "b"], 1)


def test_set_path_type_mismatch_int_into_dict_raises() -> None:
    target: dict = {"a": {"not_a_list": 1}}
    with pytest.raises(TypeError):
        set_path(target, ["a", 0], "x")


# -----------------------------------------------------------------------------
# get_path
# -----------------------------------------------------------------------------


def test_get_path_found() -> None:
    source = {"a": {"b": [10, 20, 30]}}
    assert get_path(source, ["a", "b", 1]) == 20


def test_get_path_missing_returns_default() -> None:
    source: dict = {"a": {}}
    assert get_path(source, ["a", "b", "c"]) is None
    assert get_path(source, ["a", "b", "c"], default="missing") == "missing"


def test_get_path_out_of_bounds_returns_default() -> None:
    source = {"a": [1, 2]}
    assert get_path(source, ["a", 5]) is None
    assert get_path(source, ["a", -1]) is None


def test_get_path_empty_tokens_returns_source() -> None:
    source = {"a": 1}
    assert get_path(source, []) == source


# -----------------------------------------------------------------------------
# inflate_expected_output — round-trip
# -----------------------------------------------------------------------------


def test_inflate_round_trip_exact() -> None:
    target = {
        "po_number": "PO-1",
        "buyer": {"company": "Acme"},
        "line_items": [
            {"item_number": "A1", "quantity": 2},
            {"item_number": "B2", "quantity": 5},
        ],
    }
    rules = [
        ExtractFieldTestRule(field_path="po_number", expected_value="PO-1"),
        ExtractFieldTestRule(field_path="buyer.company", expected_value="Acme"),
        ExtractFieldTestRule(field_path="line_items[0].item_number", expected_value="A1"),
        ExtractFieldTestRule(field_path="line_items[0].quantity", expected_value=2),
        ExtractFieldTestRule(field_path="line_items[1].item_number", expected_value="B2"),
        ExtractFieldTestRule(field_path="line_items[1].quantity", expected_value=5),
    ]
    inflated = inflate_expected_output(rules)
    assert inflated == target


def test_inflate_accepts_plain_dicts_as_rules() -> None:
    rules = [
        {"type": "extract_field", "field_path": "a.b", "expected_value": 1},
        {"type": "extract_field", "field_path": "a.c", "expected_value": 2},
    ]
    assert inflate_expected_output(rules) == {"a": {"b": 1, "c": 2}}


def test_inflate_skips_rule_without_field_path() -> None:
    rules = [
        ExtractFieldTestRule(field_path="a", expected_value=1),
        {"type": "other", "text": "no field path here"},
    ]
    assert inflate_expected_output(rules) == {"a": 1}


def test_inflate_auto_pads_sparse_array() -> None:
    rules = [
        ExtractFieldTestRule(field_path="items[2].name", expected_value="C"),
    ]
    assert inflate_expected_output(rules) == {"items": [None, None, {"name": "C"}]}


def test_inflate_duplicate_path_keeps_first_non_none() -> None:
    rules = [
        ExtractFieldTestRule(field_path="a", expected_value=None),
        ExtractFieldTestRule(field_path="a", expected_value="second"),
        ExtractFieldTestRule(field_path="a", expected_value="third"),
    ]
    # First rule sets None; second replaces it with "second" (existing was None);
    # third is skipped (existing is truthy/non-None).
    assert inflate_expected_output(rules) == {"a": "second"}


# -----------------------------------------------------------------------------
# validate_rules_match_expected_output — drift detection
# -----------------------------------------------------------------------------


def test_validate_matches_returns_empty() -> None:
    rules = [
        ExtractFieldTestRule(field_path="po_number", expected_value="PO-1"),
        ExtractFieldTestRule(field_path="buyer.company", expected_value="Acme"),
    ]
    expected = {"po_number": "PO-1", "buyer": {"company": "Acme"}}
    assert validate_rules_match_expected_output(rules, expected) == []


def test_validate_value_mismatch() -> None:
    rules = [ExtractFieldTestRule(field_path="po_number", expected_value="PO-1")]
    drifts = validate_rules_match_expected_output(rules, {"po_number": "PO-99"})
    assert len(drifts) == 1
    assert "po_number" in drifts[0]
    assert "value mismatch" in drifts[0]


def test_validate_missing_key_in_expected() -> None:
    rules = [
        ExtractFieldTestRule(field_path="a", expected_value=1),
        ExtractFieldTestRule(field_path="b", expected_value=2),
    ]
    drifts = validate_rules_match_expected_output(rules, {"a": 1})
    # b is missing in expected_output
    assert any("missing in expected_output" in d for d in drifts)


def test_validate_missing_key_in_rules() -> None:
    rules = [ExtractFieldTestRule(field_path="a", expected_value=1)]
    drifts = validate_rules_match_expected_output(rules, {"a": 1, "b": 2})
    assert any("missing in rules-derived" in d for d in drifts)


def test_validate_array_length_mismatch() -> None:
    rules = [
        ExtractFieldTestRule(field_path="items[0]", expected_value="x"),
    ]
    drifts = validate_rules_match_expected_output(rules, {"items": ["x", "y", "z"]})
    assert any("items[1]" in d for d in drifts)
    assert any("items[2]" in d for d in drifts)


def test_validate_expected_output_none_returns_empty() -> None:
    rules = [ExtractFieldTestRule(field_path="a", expected_value=1)]
    assert validate_rules_match_expected_output(rules, None) == []


# -----------------------------------------------------------------------------
# Performance: large array
# -----------------------------------------------------------------------------


def test_inflate_large_array_under_500ms() -> None:
    rules = [ExtractFieldTestRule(field_path=f"items[{i}].value", expected_value=i) for i in range(1000)]
    start = time.perf_counter()
    result = inflate_expected_output(rules)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert len(result["items"]) == 1000
    assert result["items"][999] == {"value": 999}
    # Local dev is ~10–20ms; shared CI runners can spike to a few hundred ms.
    assert elapsed_ms < 500, f"inflate_expected_output took {elapsed_ms:.1f}ms, expected < 500ms"
