from __future__ import annotations

import pytest

from extract_bench.evaluation.metrics.field_grounding.value_compare import (
    compare_attributed_value,
    expected_type_for_field_path,
)


def test_schema_type_resolution_for_nested_array_field() -> None:
    schema = {
        "type": "object",
        "properties": {
            "employees": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "post": {"type": "string"},
                        "total_work_hours": {"type": "number"},
                        "is_active": {"type": "boolean"},
                    },
                },
            }
        },
    }

    assert expected_type_for_field_path(schema, "employees[17].post", "Security Guard") == "string"
    assert expected_type_for_field_path(schema, "employees[17].total_work_hours", 48.5) == "number"
    assert expected_type_for_field_path(schema, "employees[17].is_active", False) == "boolean"


def test_schema_type_resolution_falls_back_to_expected_value() -> None:
    assert expected_type_for_field_path({}, "missing.path", "2024-12-25") == "date"
    assert expected_type_for_field_path(None, "missing.path", True) == "boolean"


def test_typed_date_uses_expected_to_disambiguate_us_and_eu_forms() -> None:
    comparison = compare_attributed_value("1979-12-06", "12/06/1979", expected_type="date")

    assert comparison.passed is True
    assert comparison.mode == "date"


def test_unicode_compatibility_fold_passes_micro_symbol() -> None:
    comparison = compare_attributed_value("50uL", "50µL*", expected_type="string")

    assert comparison.passed is True
    assert comparison.mode == "unicode_compat_fold"


def test_ocr_icelandic_fold_is_source_gated() -> None:
    native_comparison = compare_attributed_value("bjonusta", "þjónusta", expected_type="string", source_kind="native")
    ocr_comparison = compare_attributed_value("bjonusta", "þjónusta", expected_type="string", source_kind="ocr")

    assert native_comparison.passed is False
    assert ocr_comparison.passed is True
    assert ocr_comparison.mode == "icelandic_diacritic_fold"


def test_diagnostic_truncation_is_not_an_attribution_pass_by_default() -> None:
    comparison = compare_attributed_value(
        "Absorbent Underpads",
        "Absorbent Underpads, 24 x 20 in, Protector Sheets, 50/Pack",
        expected_type="string",
    )

    assert comparison.passed is False
    assert comparison.mode == "annotation_truncated"
    assert comparison.reason == "annotation_truncated_diagnostic_only"


def test_substring_is_not_a_passing_mode() -> None:
    comparison = compare_attributed_value(
        "462 N Rodeo Dr, Beverly Hills, CA 90210",
        "B Partner's name, address, city, state, and ZIP code: "
        "BW Portfolio Limited 462 N Rodeo Dr, Beverly Hills, CA 90210",
        expected_type="string",
    )

    assert comparison.mode != "substring"
    assert comparison.passed is False


@pytest.mark.parametrize(
    ("expected", "actual", "expected_type"),
    [
        (False, "LI", "boolean"),
        (2671.43, "2 671 43", "number"),
        (None, "—", "null"),
        ("—", "—", "number"),
        ("-", "-", "number"),
    ],
)
def test_source_export_typed_equivalences(expected: object, actual: str, expected_type: str) -> None:
    comparison = compare_attributed_value(expected, actual, expected_type=expected_type)  # type: ignore[arg-type]

    assert comparison.passed is True


def test_dash_placeholder_uses_null_empty_mode_for_numeric_schema_fields() -> None:
    comparison = compare_attributed_value("—", "—", expected_type="number")

    assert comparison.passed is True
    assert comparison.mode == "null_empty"
