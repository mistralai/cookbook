"""Tests for comparison extract citation overlay wiring."""

from extract_bench.analysis.comparison_core import (
    normalize_field_citations_for_overlay,
    normalize_gt_evidence_for_overlay,
)


def test_comparison_overlay_wrappers_use_compact_payload() -> None:
    result = {
        "output": {
            "field_citations": [
                {
                    "field_path": "invoice.number",
                    "page": 1,
                    "bbox": [0.1, 0.2, 0.3, 0.1],
                    "reference_text": "INV-001",
                },
                {"field_path": "page-only", "page": 2},
            ]
        }
    }
    test_data = {
        "_field_rules": {
            "invoice.number": {
                "verified": True,
                "evidence": [{"page": 1, "bbox": [0.1, 0.2, 0.3, 0.1], "quote": "INV-001"}],
            }
        }
    }

    assert normalize_field_citations_for_overlay(result) == [
        {"fieldPath": "invoice.number", "page": 1, "bbox": [0.1, 0.2, 0.3, 0.1]}
    ]
    assert normalize_gt_evidence_for_overlay(test_data) == [
        {"fieldPath": "invoice.number", "page": 1, "bbox": [0.1, 0.2, 0.3, 0.1]}
    ]


def test_comparison_js_wires_extract_citation_overlay() -> None:
    from importlib.resources import files

    js = files("extract_bench.analysis").joinpath("static", "comparison_report.js").read_text(encoding="utf-8")
    assert "drawExtractComparisonOverlay" in js
    assert "toggleExtractCitationLayer" in js
    assert "field_citations" in js
    assert "gt_annotations" in js
    assert "Ground truth" in js
