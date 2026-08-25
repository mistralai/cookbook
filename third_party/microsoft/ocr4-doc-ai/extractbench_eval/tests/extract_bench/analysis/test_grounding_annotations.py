"""Tests for shared extract grounding annotation loaders."""

from __future__ import annotations

from extract_bench.analysis.grounding_annotations import (
    load_gt_grounding_annotations,
    load_pred_grounding_citations,
)


def test_load_pred_citations_detailed_defaults_keep_page_only_and_text() -> None:
    result_data = {
        "output": {
            "field_citations": [
                {
                    "field_path": "invoice.number",
                    "page": 1,
                    "bbox": [0.1, 0.2, 0.3, 0.1],
                    "reference_text": "INV-001",
                },
                {"field_path": "missing-page", "bbox": [0.0, 0.0, 0.1, 0.1]},
                {"field_path": "page-only", "page": 2},
            ]
        }
    }

    annotations = load_pred_grounding_citations(result_data)

    assert annotations == [
        {
            "fieldPath": "invoice.number",
            "page": 1,
            "bbox": [0.1, 0.2, 0.3, 0.1],
            "referenceText": "INV-001",
        },
        {"fieldPath": "page-only", "page": 2},
    ]


def test_load_pred_citations_overlay_mode_requires_bbox_and_drops_text() -> None:
    result_data = {
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

    annotations = load_pred_grounding_citations(
        result_data,
        require_bbox=True,
        include_reference_text=False,
    )

    assert annotations == [
        {
            "fieldPath": "invoice.number",
            "page": 1,
            "bbox": [0.1, 0.2, 0.3, 0.1],
        }
    ]


def test_load_gt_annotations_detailed_defaults_include_metadata() -> None:
    test_data = {
        "_field_rules": {
            "invoice.number": {
                "verified": True,
                "evidence": [
                    {"page": 1, "bbox": [0.1, 0.2, 0.3, 0.1], "quote": "INV-001"},
                    {"page": 2, "quote": "page-only"},
                ],
            }
        }
    }

    annotations = load_gt_grounding_annotations(test_data)

    assert len(annotations) == 1
    assert annotations[0] == {
        "fieldPath": "invoice.number",
        "page": 1,
        "bbox": [0.1, 0.2, 0.3, 0.1],
        "verified": True,
        "quote": "INV-001",
    }


def test_load_gt_annotations_from_test_rules_list() -> None:
    test_data = {
        "test_rules": [
            {
                "type": "extract_field",
                "field_path": "jurisdiction",
                "verified": False,
                "evidence": [
                    {"page": 9, "bbox": [0.32, 0.07, 0.10, 0.02], "quote": "Goshen County"},
                ],
            },
            {
                "type": "layout",
                "field_path": "ignored",
                "evidence": [{"page": 1, "bbox": [0.0, 0.0, 0.1, 0.1]}],
            },
        ]
    }

    annotations = load_gt_grounding_annotations(test_data)

    assert len(annotations) == 1
    assert annotations[0]["fieldPath"] == "jurisdiction"
    assert annotations[0]["page"] == 9
    assert annotations[0]["verified"] is False


def test_load_gt_annotations_overlay_mode_strips_metadata() -> None:
    test_data = {
        "_field_rules": {
            "invoice.number": {
                "verified": True,
                "evidence": [
                    {"page": 1, "bbox": [0.1, 0.2, 0.3, 0.1], "quote": "INV-001"},
                ],
            }
        }
    }

    annotations = load_gt_grounding_annotations(
        test_data,
        include_verified=False,
        include_quote=False,
    )

    assert annotations == [
        {
            "fieldPath": "invoice.number",
            "page": 1,
            "bbox": [0.1, 0.2, 0.3, 0.1],
        }
    ]


def test_loaders_handle_missing_payloads() -> None:
    assert load_pred_grounding_citations(None) == []
    assert load_pred_grounding_citations({}) == []
    assert load_gt_grounding_annotations(None) == []
    assert load_gt_grounding_annotations({}) == []


def test_load_gt_annotations_falls_back_to_legacy_bboxes() -> None:
    """v0.1 rules carry the same page/COCO pairs under ``bboxes``, not ``evidence``."""
    test_data = {
        "_field_rules": {
            "invoice.total": {
                "verified": True,
                "expected_value": "42.00",
                "bboxes": [{"page": 3, "bbox": [0.5, 0.6, 0.2, 0.05]}],
            }
        }
    }

    annotations = load_gt_grounding_annotations(test_data)

    assert len(annotations) == 1
    assert annotations[0]["fieldPath"] == "invoice.total"
    assert annotations[0]["page"] == 3
    assert annotations[0]["bbox"] == [0.5, 0.6, 0.2, 0.05]
    assert annotations[0]["verified"] is True


def test_load_gt_annotations_prefers_evidence_over_legacy_bboxes() -> None:
    test_data = {
        "_field_rules": {
            "invoice.total": {
                "evidence": [{"page": 1, "bbox": [0.1, 0.1, 0.1, 0.1]}],
                "bboxes": [{"page": 9, "bbox": [0.9, 0.9, 0.05, 0.05]}],
            }
        }
    }

    assert [a["page"] for a in load_gt_grounding_annotations(test_data)] == [1]


def test_legacy_bboxes_reach_the_comparison_overlay_payload() -> None:
    """The comparison overlay uses the same loader, so v0.1 GT must reach it too."""
    test_data = {
        "_field_rules": {
            "invoice.total": {"bboxes": [{"page": 3, "bbox": [0.5, 0.6, 0.2, 0.05]}]},
        }
    }

    overlay = load_gt_grounding_annotations(test_data, include_verified=False, include_quote=False)

    assert overlay == [{"fieldPath": "invoice.total", "page": 3, "bbox": [0.5, 0.6, 0.2, 0.05]}]
