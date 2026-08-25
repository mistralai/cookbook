"""Tests for parse/extract rule_results display normalization."""

from __future__ import annotations

from extract_bench.analysis.detailed_report import (
    _coerce_rule_pass,
    _is_non_counting_metric,
    _normalize_rule_result_for_table,
    _rule_tally_from_metadata,
)

# One v0.2 evidence rule, as the adapter emits it: the value matched, but on the
# wrong page and with no qualifying bbox. The same dict backs every
# extract_evidence_* metric, so each must read its own key.
_EVIDENCE_RULE = {
    "field_path": "operator_name",
    "value_pass": True,
    "page_pass": False,
    "page_qualified": False,
    "bbox_qualified": False,
    "bbox_covered_pass": False,
    "bbox_iou": 0.0,
    "bbox_iou_pass": 0.0,
    "mode": "address",
    "reason": "pass",
}


def test_coerce_rule_pass_prefers_explicit_passed() -> None:
    assert _coerce_rule_pass({"passed": True, "value_pass": False}) is True
    assert _coerce_rule_pass({"passed": False}) is False


def test_coerce_rule_pass_extract_value_pass() -> None:
    assert _coerce_rule_pass({"value_pass": True, "bbox_iou_pass": 0.93}) is True
    assert _coerce_rule_pass({"value_pass": False, "bbox_iou_pass": 0.93}) is False


def test_coerce_rule_pass_ignores_float_bbox_iou_pass() -> None:
    assert _coerce_rule_pass({"bbox_iou_pass": 0.93}) is None
    assert _coerce_rule_pass({"bbox_iou_pass": 1}) is True
    assert _coerce_rule_pass({"bbox_iou_pass": 0}) is False
    # 0.0 == 0 and 1.0 == 1 in Python, so the guard has to check the type.
    assert _coerce_rule_pass({"bbox_iou_pass": 0.0}) is None
    assert _coerce_rule_pass({"bbox_iou_pass": 1.0}) is None


def test_coerce_rule_pass_reads_the_key_its_metric_counts() -> None:
    assert _coerce_rule_pass(_EVIDENCE_RULE, "extract_evidence_value_pass_rate") is True
    assert _coerce_rule_pass(_EVIDENCE_RULE, "extract_evidence_page_pass_rate") is False
    assert _coerce_rule_pass(_EVIDENCE_RULE, "extract_evidence_page_covered_pass_rate") is False
    assert _coerce_rule_pass(_EVIDENCE_RULE, "extract_evidence_bbox_covered_pass_rate") is False


def test_coerce_rule_pass_metric_specific_keys_for_parse_and_localization() -> None:
    rule = {"field_path": "x", "loc_pass": True, "cls_pass": False, "attr_pass": True, "element_pass": False}
    assert _coerce_rule_pass(rule, "extract_localization_pass_rate") is True
    assert _coerce_rule_pass(rule, "extract_field_classification_pass_rate") is False
    assert _coerce_rule_pass(rule, "extract_attribution_pass_rate") is True
    assert _coerce_rule_pass(rule, "extract_element_pass_rate") is False


def test_coerce_rule_pass_unknown_metric_falls_back_to_generic_order() -> None:
    assert _coerce_rule_pass(_EVIDENCE_RULE, "some_unmapped_metric") is True


def test_non_counting_metrics_are_excluded_from_rule_tallies() -> None:
    assert _is_non_counting_metric("extract_evidence_bbox_IOU_pass_rate") is True
    assert _is_non_counting_metric("extract_evidence_bbox_IOU_alignment") is True
    assert _is_non_counting_metric("extract_evidence_bbox_coverage") is True
    assert _is_non_counting_metric("extract_evidence_value_pass_rate") is False


def test_normalize_rule_result_for_table_parse_shape() -> None:
    assert _normalize_rule_result_for_table(
        {
            "type": "form_field",
            "name": "invoice_total",
            "passed": True,
            "explanation": "matched",
        }
    ) == {
        "type": "form_field",
        "passed": True,
        "id": "invoice_total",
        "message": "matched",
    }


def test_normalize_rule_result_for_table_extract_shape() -> None:
    assert _normalize_rule_result_for_table(
        {
            "field_path": "operator_name",
            "mode": "address",
            "value_pass": True,
            "reason": "pass",
            "bbox_iou_pass": 0.93,
        }
    ) == {
        "type": "address",
        "passed": True,
        "id": "operator_name",
        "message": "pass",
    }


# --- conditional denominators -------------------------------------------------
#
# Several evidence metrics share one ``rule_results`` list but score over a
# subset of it. The evaluator records the subset size it used, so the chip has
# to read that rather than ``len(rule_results)``.


def test_tally_uses_full_rule_count_for_unconditional_metrics() -> None:
    meta = {"tp": 3, "total": 5, "denominator": "graded_verified_v02_rules"}
    assert _rule_tally_from_metadata(meta) == (3, 5)


def test_tally_uses_page_qualified_subset_for_page_covered() -> None:
    # 2 of 55 rules are page-qualified and 1 of those passes: the metric is 0.5,
    # so the chip must read 1/2, not 1/55.
    meta = {"tp": 1, "total": 55, "denominator": "page_qualified_rules", "covered_total": 2}
    assert _rule_tally_from_metadata(meta) == (1, 2)


def test_tally_uses_gt_bbox_subset_for_bbox_covered() -> None:
    # Only GT-bbox-bearing leaves are gradeable; whether a leaf had a GT bbox is
    # not on the per-rule dicts at all, so counting rules cannot recover this.
    meta = {"tp": 4, "total": 55, "denominator": "gt_bbox_bearing_rules", "bbox_gt_total": 6}
    assert _rule_tally_from_metadata(meta) == (4, 6)


def test_tally_reports_zero_over_the_conditional_subset() -> None:
    meta = {"tp": 0, "total": 55, "denominator": "gt_bbox_bearing_rules", "bbox_gt_total": 3}
    assert _rule_tally_from_metadata(meta) == (0, 3)


def test_tally_falls_back_when_bookkeeping_is_absent() -> None:
    assert _rule_tally_from_metadata({"total": 5}) is None
    assert _rule_tally_from_metadata({"tp": 3}) is None
    # Conditional metric missing its subset count — do not silently use ``total``.
    assert _rule_tally_from_metadata({"tp": 1, "total": 55, "denominator": "gt_bbox_bearing_rules"}) is None


def test_tally_rejects_inconsistent_or_non_count_bookkeeping() -> None:
    # A numerator larger than its denominator means we misread the schema.
    assert _rule_tally_from_metadata({"tp": 9, "total": 5}) is None
    # ``bbox_IOU_pass_rate`` sums IoU floats into tp-shaped slots; not a count.
    assert _rule_tally_from_metadata({"tp": 2.5, "total": 5}) is None
    # ``isinstance(True, int)`` is True, so bools must be rejected explicitly.
    assert _rule_tally_from_metadata({"tp": True, "total": 5}) is None
