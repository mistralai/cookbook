"""Tests for the loader drift warning on extract_field rules vs expected_output."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from extract_bench.test_cases.loader import load_test_case
from extract_bench.test_cases.schema import ExtractTestCase


def _write_doc(tmp_path: Path) -> Path:
    doc_dir = tmp_path / "group"
    doc_dir.mkdir()
    pdf_path = doc_dir / "doc1.pdf"
    pdf_path.write_bytes(b"%PDF-1.0 dummy")
    return pdf_path


def test_drift_warning_emitted_when_rules_disagree(tmp_path: Path, caplog) -> None:
    pdf_path = _write_doc(tmp_path)

    # expected_output says PO-1; rule says PO-2 → should warn.
    test_json = {
        "data_schema": {"type": "object"},
        "expected_output": {"po_number": "PO-1"},
        "test_rules": [
            {
                "type": "extract_field",
                "field_path": "po_number",
                "expected_value": "PO-2",
            }
        ],
    }
    (pdf_path.with_suffix(".test.json")).write_text(json.dumps(test_json))

    with caplog.at_level(logging.WARNING, logger="extract_bench.test_cases.loader"):
        result = load_test_case(pdf_path)
    assert isinstance(result, ExtractTestCase)
    # Test case still loaded — warning is non-fatal.
    drift_records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and ("disagree" in r.getMessage() or "drift" in r.getMessage().lower())
    ]
    assert drift_records, f"Expected a drift WARNING log, got: {[r.getMessage() for r in caplog.records]}"


def test_no_warning_when_rules_match_expected_output(tmp_path: Path, caplog) -> None:
    pdf_path = _write_doc(tmp_path)
    test_json = {
        "data_schema": {"type": "object"},
        "expected_output": {"po_number": "PO-1"},
        "test_rules": [
            {
                "type": "extract_field",
                "field_path": "po_number",
                "expected_value": "PO-1",
            }
        ],
    }
    (pdf_path.with_suffix(".test.json")).write_text(json.dumps(test_json))

    with caplog.at_level(logging.WARNING, logger="extract_bench.test_cases.loader"):
        result = load_test_case(pdf_path)
    assert isinstance(result, ExtractTestCase)
    drift_records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and ("disagree" in r.getMessage() or "drift" in r.getMessage())
    ]
    assert not drift_records, f"Unexpected drift WARNING: {[r.getMessage() for r in drift_records]}"


def test_no_warning_when_expected_output_absent(tmp_path: Path, caplog) -> None:
    pdf_path = _write_doc(tmp_path)
    test_json = {
        "data_schema": {"type": "object"},
        "test_rules": [
            {
                "type": "extract_field",
                "field_path": "po_number",
                "expected_value": "PO-1",
            }
        ],
    }
    (pdf_path.with_suffix(".test.json")).write_text(json.dumps(test_json))

    with caplog.at_level(logging.WARNING, logger="extract_bench.test_cases.loader"):
        result = load_test_case(pdf_path)
    assert isinstance(result, ExtractTestCase)
    drift_records = [r for r in caplog.records if r.levelno == logging.WARNING and "disagree" in r.getMessage()]
    assert not drift_records
