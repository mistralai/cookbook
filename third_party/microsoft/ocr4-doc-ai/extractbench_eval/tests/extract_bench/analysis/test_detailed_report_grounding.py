"""Tests for grounding annotation loading in the detailed report."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from extract_bench.analysis.detailed_report import (
    _lookup_keys_for_file,
    _normalize_grounding_rule,
    _select_grounding_rules,
)


def test_normalize_grounding_rule_maps_keys_and_bboxes() -> None:
    rule = {
        "field_path": "vendor.name",
        "value_pass": True,
        "page_pass": False,
        "bbox_iou": 0.42,
        "bbox_iou_pass": 1,
        "matched_pred_bboxes": [[0.1, 0.2, 0.3, 0.4, 99], "bad", [0.5]],
    }

    normalized = _normalize_grounding_rule(rule)

    assert normalized["fieldPath"] == "vendor.name"
    assert normalized["valuePass"] is True
    assert normalized["pagePass"] is False
    assert normalized["bboxIou"] == 0.42
    assert normalized["bboxIouPass"] == 1
    assert normalized["matchedPredBboxes"] == [[0.1, 0.2, 0.3, 0.4]]
    assert normalized["passed"] is True


def test_select_grounding_rules_prefers_priority_metric() -> None:
    low_priority = SimpleNamespace(
        metric_name="extract_attribution_pass_rate",
        metadata={
            "rule_results": [
                {"field_path": "from_attr", "value_pass": True},
            ]
        },
    )
    high_priority = SimpleNamespace(
        metric_name="extract_evidence_bbox_IOU_pass_rate",
        metadata={
            "rule_results": [
                {"field_path": "from_iou", "value_pass": False, "bbox_iou_pass": False},
            ]
        },
    )

    selected = _select_grounding_rules([low_priority, high_priority])

    assert len(selected) == 1
    assert selected[0]["fieldPath"] == "from_iou"


def test_select_grounding_rules_falls_back_to_any_field_path_rules() -> None:
    other = SimpleNamespace(
        metric_name="some_other_metric",
        metadata={
            "rule_results": [
                {"path": "fallback.field", "element_pass": True},
            ]
        },
    )

    selected = _select_grounding_rules([other])

    assert len(selected) == 1
    assert selected[0]["fieldPath"] == "fallback.field"
    assert selected[0]["passed"] is True


def test_lookup_keys_for_file_includes_group_and_stem() -> None:
    root = Path("/tmp/output")
    file_path = root / "short" / "invoice_001.result.json"

    keys = _lookup_keys_for_file(file_path, root, suffix=".result.json")

    assert keys == ["short/invoice_001", "invoice_001"]


def test_lookup_keys_for_file_stem_only_at_root() -> None:
    root = Path("/tmp/cases")
    file_path = root / "invoice_001.test.json"

    keys = _lookup_keys_for_file(file_path, root, suffix=".test.json")

    assert keys == ["invoice_001"]
