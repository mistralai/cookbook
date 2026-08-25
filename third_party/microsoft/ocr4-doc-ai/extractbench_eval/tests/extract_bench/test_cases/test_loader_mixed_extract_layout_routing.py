"""Loader routing and coercion for mixed Parse + Extract corpora."""

from __future__ import annotations

import json
from pathlib import Path

from extract_bench.test_cases.loader import load_test_case
from extract_bench.test_cases.schema import ExtractFieldTestRule, ExtractTestCase, LayoutTestRule, ParseTestCase


def _write_doc(tmp_path: Path, test_rules: list[dict]) -> Path:
    doc_dir = tmp_path / "group"
    doc_dir.mkdir()
    pdf_path = doc_dir / "doc1.pdf"
    pdf_path.write_bytes(b"%PDF-1.0 dummy")
    payload = {
        "data_schema": {"type": "object", "properties": {"po_number": {"type": "string"}}},
        "expected_output": {"po_number": "PO-1234"},
        "test_rules": test_rules,
    }
    pdf_path.with_suffix(".test.json").write_text(json.dumps(payload))
    return pdf_path


def test_parse_hint_loads_mixed_layout_and_extract_field_as_parse_case(tmp_path: Path) -> None:
    pdf_path = _write_doc(
        tmp_path,
        [
            {
                "type": "layout",
                "page": 1,
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "canonical_class": "Text",
                "text": "PO-1234",
                "id": "layout-1",
            },
            {
                "type": "extract_field",
                "field_path": "po_number",
                "expected_value": "PO-1234",
                "id": "field-1",
                "bboxes": [{"page": 1, "bbox": [0.1, 0.1, 0.2, 0.2]}],
            },
        ],
    )

    result = load_test_case(pdf_path, product_type_hint="PARSE")

    assert isinstance(result, ParseTestCase)
    assert len(result.get_layout_rules()) == 1
    assert len(result.get_extract_field_rules()) == 1
    assert isinstance(result.get_layout_rules()[0], LayoutTestRule)
    assert isinstance(result.get_extract_field_rules()[0], ExtractFieldTestRule)


def test_parse_hint_keeps_extract_only_corpus_on_extract_case(tmp_path: Path) -> None:
    pdf_path = _write_doc(
        tmp_path,
        [
            {
                "type": "extract_field",
                "field_path": "po_number",
                "expected_value": "PO-1234",
                "id": "field-1",
                "bboxes": [{"page": 1, "bbox": [0.1, 0.1, 0.2, 0.2]}],
            },
        ],
    )

    result = load_test_case(pdf_path, product_type_hint="PARSE")

    assert isinstance(result, ExtractTestCase)
    assert len(result.get_extract_field_rules()) == 1


def test_parse_hint_uses_parse_case_for_layout_only_extract_corpus(tmp_path: Path) -> None:
    pdf_path = _write_doc(
        tmp_path,
        [
            {
                "type": "layout",
                "page": 1,
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "canonical_class": "Text",
                "text": "PO-1234",
                "id": "layout-1",
            },
        ],
    )

    result = load_test_case(pdf_path, product_type_hint="PARSE")

    assert isinstance(result, ParseTestCase)
    assert len(result.get_layout_rules()) == 1


def test_parse_hint_loads_table_rules_from_extract_annotated_corpus(tmp_path: Path) -> None:
    pdf_path = _write_doc(
        tmp_path,
        [
            {
                "type": "extract_field",
                "field_path": "po_number",
                "expected_value": "PO-1234",
                "id": "field-1",
            },
            {
                "type": "table",
                "cell": "[yes]",
                "top_heading": "Vehicle 1",
                "left_heading": "Navigation",
                "id": "table-1",
            },
        ],
    )

    result = load_test_case(pdf_path, product_type_hint="PARSE")

    assert isinstance(result, ParseTestCase)
    assert len(result.get_extract_field_rules()) == 1
    assert [rule.type for rule in result.test_rules or []] == ["extract_field", "table"]


def test_parse_hint_loads_expected_markdown_from_extract_annotated_corpus(tmp_path: Path) -> None:
    pdf_path = _write_doc(tmp_path, [])
    pdf_path.with_suffix(".md").write_text("| Option | Vehicle 1 |\n| --- | --- |\n| Navigation | [yes] |\n")

    result = load_test_case(pdf_path, product_type_hint="PARSE")

    assert isinstance(result, ParseTestCase)
    assert result.expected_markdown == "| Option | Vehicle 1 |\n| --- | --- |\n| Navigation | [yes] |\n"


def test_extract_hint_keeps_mixed_table_corpus_on_extract_case(tmp_path: Path) -> None:
    pdf_path = _write_doc(
        tmp_path,
        [
            {
                "type": "extract_field",
                "field_path": "po_number",
                "expected_value": "PO-1234",
                "id": "field-1",
            },
            {
                "type": "table",
                "cell": "[yes]",
                "top_heading": "Vehicle 1",
                "left_heading": "Navigation",
                "id": "table-1",
            },
        ],
    )

    result = load_test_case(pdf_path, product_type_hint="EXTRACT")

    assert isinstance(result, ExtractTestCase)
