from __future__ import annotations

import pytest

from extract_bench.evaluation.metrics.field_grounding.core import (
    BBox,
    bbox_recall,
    compare_field_value,
    compute_bbox_metrics,
    field_iou,
    normalize_text,
)


def test_normalize_text_preserves_visible_unicode() -> None:
    assert normalize_text("  ＡＢＣ\t中文  😀\u200b ") == "abc 中文 😀"


def test_string_comparison_uses_jaro_winkler_threshold() -> None:
    assert compare_field_value("Acme Corporation", " acme   corporation ").passed
    near = compare_field_value("Service revenue", "Service revnue")
    assert near.score >= 0.9
    assert near.passed
    far = compare_field_value("Service revenue", "Equipment sales")
    assert far.score < 0.9
    assert not far.passed


def test_typed_value_comparison() -> None:
    assert compare_field_value(10, "10").passed
    assert not compare_field_value(10, "10.5").passed
    assert compare_field_value(10.25, "10.2500001").passed
    assert compare_field_value(True, "checked").passed
    assert not compare_field_value(False, "yes").passed
    assert compare_field_value("2024-12-25", "December 25, 2024").passed
    assert not compare_field_value("2024-12-25", "December 26, 2024").passed


def test_date_comparison_handles_period_suffixed_tokens() -> None:
    # Common cell formats like "Mon. Jan. 02 2023" must parse as dates rather
    # than fall through to Jaro-Winkler against the ISO ground truth.
    cmp = compare_field_value("2023-01-02", "Mon. Jan. 02 2023")
    assert cmp.passed
    assert cmp.mode == "date"

    assert compare_field_value("2023-01-02", "Jan. 02 2023").passed
    assert compare_field_value("2023-01-02", "Monday January 02, 2023").passed
    # Mismatched day still fails through the date branch.
    assert not compare_field_value("2023-01-02", "Mon. Jan. 03 2023").passed


def test_field_iou_differs_from_standard_iou() -> None:
    gt = [BBox(page=1, group="a", bbox=(0.0, 0.0, 0.2, 0.2))]
    pred = [BBox(page=1, group="a", bbox=(0.0, 0.0, 1.0, 1.0))]
    assert field_iou(gt, pred) == pytest.approx(1.0)
    # Standard IoU would be gt_area / pred_area = 0.04.


def test_bbox_recall_deduplicates_overlapping_predictions() -> None:
    gt = [BBox(page=1, group="a", bbox=(0.0, 0.0, 0.2, 0.2))]
    pred = [
        BBox(page=1, group="a", bbox=(0.0, 0.0, 0.15, 0.2)),
        BBox(page=1, group="a", bbox=(0.05, 0.0, 0.15, 0.2)),
    ]
    metrics = compute_bbox_metrics(gt, pred)
    assert metrics.bbox_recall == pytest.approx(1.0)
    # Best single intersection only covers 75% of the GT.
    assert metrics.iou == pytest.approx(0.75)


def test_bboxes_are_scoped_by_page_and_group() -> None:
    gt = [
        BBox(page=1, group="a", bbox=(0.0, 0.0, 0.2, 0.2)),
        BBox(page=2, group="a", bbox=(0.0, 0.0, 0.2, 0.2)),
    ]
    pred = [
        BBox(page=1, group="a", bbox=(0.0, 0.0, 0.2, 0.2)),
        BBox(page=2, group="b", bbox=(0.0, 0.0, 0.2, 0.2)),
    ]
    assert field_iou(gt, pred) == pytest.approx(0.5)
    assert bbox_recall(gt, pred) == pytest.approx(0.5)


def test_empty_gt_bbox_metrics_are_zero() -> None:
    assert compute_bbox_metrics([], [BBox(page=1, bbox=(0.0, 0.0, 1.0, 1.0))]).iou == 0.0
