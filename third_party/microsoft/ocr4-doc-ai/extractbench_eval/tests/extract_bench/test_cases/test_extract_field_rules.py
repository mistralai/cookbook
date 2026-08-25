"""Tests for ExtractFieldBbox, ExtractFieldTestRule, and ExtractTestCase coercion."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from extract_bench.test_cases.schema import (
    ExtractFieldBbox,
    ExtractFieldTestRule,
    ExtractTestCase,
    FieldEvidence,
)


def _make_extract_test_case(**overrides) -> dict:
    """Minimal payload to build an ExtractTestCase via model_validate."""
    payload: dict = {
        "test_id": "group/doc",
        "group": "group",
        "file_path": "some/doc.pdf",
        "schema": {"type": "object"},
    }
    payload.update(overrides)
    return payload


# -----------------------------------------------------------------------------
# ExtractFieldBbox
# -----------------------------------------------------------------------------


def test_bbox_defaults_source_bbox_index_to_none() -> None:
    box = ExtractFieldBbox(page=1, bbox=[0.1, 0.2, 0.3, 0.4])
    assert box.source_bbox_index is None


def test_bbox_accepts_source_bbox_index() -> None:
    box = ExtractFieldBbox(page=2, bbox=[0.0, 0.0, 1.0, 1.0], source_bbox_index=5)
    assert box.source_bbox_index == 5


def test_bbox_page_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        ExtractFieldBbox(page=0, bbox=[0.1, 0.1, 0.1, 0.1])


def test_bbox_source_bbox_index_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        ExtractFieldBbox(page=1, bbox=[0.0, 0.0, 0.1, 0.1], source_bbox_index=-1)


def test_bbox_accepts_any_length_list() -> None:
    # Plan does not constrain to 4 floats; just any list of floats.
    box = ExtractFieldBbox(page=1, bbox=[0.1, 0.2, 0.3, 0.4, 0.5])
    assert box.bbox == [0.1, 0.2, 0.3, 0.4, 0.5]


# -----------------------------------------------------------------------------
# ExtractFieldTestRule
# -----------------------------------------------------------------------------


def test_rule_verified_defaults_to_true() -> None:
    rule = ExtractFieldTestRule(field_path="po_number", expected_value="PO-1")
    assert rule.verified is True


def test_rule_empty_bboxes_is_valid() -> None:
    rule = ExtractFieldTestRule(field_path="po_number", expected_value="PO-1")
    assert rule.bboxes == []


def test_rule_type_discriminator_is_fixed() -> None:
    rule = ExtractFieldTestRule(field_path="po_number", expected_value="PO-1")
    assert rule.type == "extract_field"


def test_rule_with_bboxes_round_trip() -> None:
    rule = ExtractFieldTestRule(
        field_path="line_items[0].description",
        expected_value="Fall Harness",
        bboxes=[
            ExtractFieldBbox(page=1, bbox=[0.1, 0.2, 0.3, 0.4], source_bbox_index=0),
            ExtractFieldBbox(page=1, bbox=[0.1, 0.5, 0.3, 0.4], source_bbox_index=1),
        ],
        verified=False,
        tags=["source_export"],
    )
    payload = rule.model_dump()
    restored = ExtractFieldTestRule.model_validate(payload)
    assert restored == rule


def test_rule_accepts_bool_expected_value() -> None:
    rule = ExtractFieldTestRule(field_path="is_active", expected_value=True)
    assert rule.expected_value is True


def test_rule_accepts_value_only_evidence_without_page() -> None:
    rule = ExtractFieldTestRule(
        field_path="stock_list[0].issuer",
        expected_value=None,
        evidence=[FieldEvidence(page=None, value={"issuer": "Acme"})],
    )

    assert rule.evidence is not None
    assert rule.evidence[0].page is None


# -----------------------------------------------------------------------------
# ExtractTestCase coercion
# -----------------------------------------------------------------------------


def test_extract_test_case_coerces_dict_to_typed_rule() -> None:
    rule_dict = {
        "type": "extract_field",
        "field_path": "po_number",
        "expected_value": "PO-1",
        "bboxes": [{"page": 1, "bbox": [0.1, 0.2, 0.3, 0.4], "source_bbox_index": 0}],
        "verified": True,
    }
    tc = ExtractTestCase.model_validate(_make_extract_test_case(test_rules=[rule_dict]))
    assert tc.test_rules is not None
    assert len(tc.test_rules) == 1
    rule = tc.test_rules[0]
    assert isinstance(rule, ExtractFieldTestRule)
    assert rule.field_path == "po_number"
    assert rule.expected_value == "PO-1"
    assert len(rule.bboxes) == 1
    assert rule.bboxes[0].source_bbox_index == 0


def test_extract_test_case_preserves_unknown_rules_as_dicts() -> None:
    extract_rule = {
        "type": "extract_field",
        "field_path": "po_number",
        "expected_value": "PO-1",
    }
    opaque_rule = {"type": "present", "text": "Hello world"}

    tc = ExtractTestCase.model_validate(_make_extract_test_case(test_rules=[extract_rule, opaque_rule]))
    assert tc.test_rules is not None
    assert len(tc.test_rules) == 2
    assert isinstance(tc.test_rules[0], ExtractFieldTestRule)
    # Unknown rule type passed through unchanged.
    assert isinstance(tc.test_rules[1], dict)
    assert tc.test_rules[1] == opaque_rule


def test_extract_test_case_test_rules_none_stays_none() -> None:
    tc = ExtractTestCase.model_validate(_make_extract_test_case())
    assert tc.test_rules is None


def test_get_extract_field_rules_filters_only_typed() -> None:
    extract_rule = {
        "type": "extract_field",
        "field_path": "po_number",
        "expected_value": "PO-1",
    }
    opaque_rule = {"type": "present", "text": "Hello world"}

    tc = ExtractTestCase.model_validate(_make_extract_test_case(test_rules=[extract_rule, opaque_rule, extract_rule]))
    typed = tc.get_extract_field_rules()
    assert len(typed) == 2
    assert all(isinstance(r, ExtractFieldTestRule) for r in typed)


def test_get_extract_field_rules_when_none() -> None:
    tc = ExtractTestCase.model_validate(_make_extract_test_case())
    assert tc.get_extract_field_rules() == []


def test_preexisting_typed_rule_passes_through() -> None:
    typed_rule = ExtractFieldTestRule(field_path="po_number", expected_value="PO-1")
    # Pydantic accepts typed instances in model_validate input.
    tc = ExtractTestCase.model_validate(_make_extract_test_case(test_rules=[typed_rule]))
    assert tc.test_rules is not None
    assert isinstance(tc.test_rules[0], ExtractFieldTestRule)
    assert tc.test_rules[0].field_path == "po_number"


def test_file_path_coerced_from_string() -> None:
    # Sanity check on the base test-case machinery — used by the above test scaffold.
    tc = ExtractTestCase.model_validate(_make_extract_test_case())
    assert isinstance(tc.file_path, Path)


def test_extract_field_rule_rejects_unknown_normalizers() -> None:
    """A typo'd normalizer must fail at load time, not silently no-op in the
    metrics (the field would score strictly with no warning)."""
    with pytest.raises(ValidationError, match="Unknown normalizer"):
        ExtractFieldTestRule(field_path="f", normalizers=["case-insensitive"])
    rule = ExtractFieldTestRule(field_path="f", normalizers=["case_insensitive", "lenient_date"])
    assert rule.normalizers == ["case_insensitive", "lenient_date"]
