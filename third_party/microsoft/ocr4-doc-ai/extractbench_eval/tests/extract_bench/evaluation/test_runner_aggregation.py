from __future__ import annotations

from pathlib import Path

import pytest

from extract_bench.evaluation.runner import EvaluationRunner
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue


def test_runner_uses_avg_for_macro_and_micro_for_pooled_extract_metrics() -> None:
    runner = EvaluationRunner(output_dir=Path("/tmp/unused"))
    results = [
        EvaluationResult(
            test_id="a",
            example_id="a",
            pipeline_name="p",
            product_type="extract",
            success=True,
            metrics=[
                MetricValue(metric_name="extract_value_precision", value=0.5, metadata={"tp": 1, "fp": 1, "fn": 1}),
                MetricValue(metric_name="extract_value_recall", value=0.5, metadata={"tp": 1, "fp": 1, "fn": 1}),
                MetricValue(metric_name="extract_value_f1", value=0.5, metadata={"tp": 1, "fp": 1, "fn": 1}),
                MetricValue(
                    metric_name="extract_element_pass_rate",
                    value=0.5,
                    metadata={"passed": 1, "total": 2, "tp": 1, "fp": 1, "fn": 0},
                ),
                MetricValue(
                    metric_name="extract_bbox_iou",
                    value=0.25,
                    metadata={
                        "score_sum": 0.5,
                        "score_count": 2,
                        "intersection_area": 1.0,
                        "union_area": 4.0,
                    },
                ),
                MetricValue(
                    metric_name="extract_bbox_recall",
                    value=0.5,
                    metadata={
                        "score_sum": 1.0,
                        "score_count": 2,
                        "covered_gt_area": 2.0,
                        "gt_area": 4.0,
                    },
                ),
                MetricValue(
                    metric_name="parse_field_iou",
                    value=0.25,
                    metadata={
                        "score_sum": 0.5,
                        "score_count": 2,
                        "intersection_area": 100.0,
                        "union_area": 100.0,
                    },
                ),
                MetricValue(
                    metric_name="parse_field_bbox_recall",
                    value=0.5,
                    metadata={
                        "score_sum": 1.0,
                        "score_count": 2,
                        "covered_gt_area": 100.0,
                        "gt_area": 100.0,
                    },
                ),
                MetricValue(
                    metric_name="parse_field_text_similarity",
                    value=0.5,
                    metadata={"string_rule_count": 1, "total_rule_count": 2},
                ),
            ],
        ),
        EvaluationResult(
            test_id="b",
            example_id="b",
            pipeline_name="p",
            product_type="extract",
            success=True,
            metrics=[
                MetricValue(metric_name="extract_value_precision", value=1.0, metadata={"tp": 3, "fp": 0, "fn": 0}),
                MetricValue(metric_name="extract_value_recall", value=1.0, metadata={"tp": 3, "fp": 0, "fn": 0}),
                MetricValue(metric_name="extract_value_f1", value=1.0, metadata={"tp": 3, "fp": 0, "fn": 0}),
                MetricValue(
                    metric_name="extract_element_pass_rate",
                    value=1.0,
                    metadata={"passed": 3, "total": 3, "tp": 3, "fp": 0, "fn": 0},
                ),
                MetricValue(
                    metric_name="extract_bbox_iou",
                    value=1.0,
                    metadata={
                        "score_sum": 3.0,
                        "score_count": 3,
                        "intersection_area": 9.0,
                        "union_area": 9.0,
                    },
                ),
                MetricValue(
                    metric_name="extract_bbox_recall",
                    value=1.0,
                    metadata={
                        "score_sum": 3.0,
                        "score_count": 3,
                        "covered_gt_area": 3.0,
                        "gt_area": 3.0,
                    },
                ),
                MetricValue(
                    metric_name="parse_field_iou",
                    value=1.0,
                    metadata={
                        "score_sum": 3.0,
                        "score_count": 3,
                        "intersection_area": 0.0,
                        "union_area": 100.0,
                    },
                ),
                MetricValue(
                    metric_name="parse_field_bbox_recall",
                    value=1.0,
                    metadata={
                        "score_sum": 3.0,
                        "score_count": 3,
                        "covered_gt_area": 0.0,
                        "gt_area": 100.0,
                    },
                ),
                MetricValue(
                    metric_name="parse_field_text_similarity",
                    value=1.0,
                    metadata={"string_rule_count": 3, "total_rule_count": 3},
                ),
            ],
        ),
    ]

    aggregate = runner._aggregate_metrics(results)

    assert aggregate["avg_extract_value_f1"] == 0.75
    assert aggregate["micro_extract_value_precision"] == pytest.approx(0.8)
    assert aggregate["micro_extract_value_recall"] == pytest.approx(0.8)
    assert aggregate["micro_extract_value_f1"] == pytest.approx(0.8)
    assert aggregate["avg_extract_element_pass_rate"] == 0.75
    assert aggregate["micro_extract_element_pass_rate"] == pytest.approx(0.8)
    assert aggregate["avg_extract_bbox_iou"] == 0.625
    assert aggregate["micro_extract_bbox_iou"] == pytest.approx(3.5 / 5.0)
    assert aggregate["micro_extract_bbox_iou"] != pytest.approx(10.0 / 13.0)
    assert aggregate["avg_extract_bbox_recall"] == 0.75
    assert aggregate["micro_extract_bbox_recall"] == pytest.approx(4.0 / 5.0)
    assert aggregate["micro_extract_bbox_recall"] != pytest.approx(5.0 / 7.0)
    assert aggregate["avg_parse_field_iou"] == 0.625
    assert aggregate["micro_parse_field_iou"] == pytest.approx(3.5 / 5.0)
    assert aggregate["micro_parse_field_iou"] != pytest.approx(100.0 / 200.0)
    assert aggregate["avg_parse_field_bbox_recall"] == 0.75
    assert aggregate["micro_parse_field_bbox_recall"] == pytest.approx(4.0 / 5.0)
    assert aggregate["micro_parse_field_bbox_recall"] != pytest.approx(100.0 / 200.0)
    assert aggregate["avg_parse_field_text_similarity"] == 0.75
    assert aggregate["micro_parse_field_text_similarity"] == pytest.approx(0.875)
    assert "macro_extract_element_pass_rate" not in aggregate


# ---------------------------------------------------------------------------
# Count-as-zero padding (genuine failures drag avg_* scores down)
# ---------------------------------------------------------------------------


def _success(test_id: str, metrics: list[MetricValue], product_type: str = "parse") -> EvaluationResult:
    return EvaluationResult(
        test_id=test_id,
        example_id=test_id,
        pipeline_name="p",
        product_type=product_type,
        success=True,
        metrics=metrics,
    )


def _failure(
    test_id: str,
    error: str = "provider produced no usable output",
    product_type: str = "parse",
) -> EvaluationResult:
    return EvaluationResult(
        test_id=test_id,
        example_id=test_id,
        pipeline_name="p",
        product_type=product_type,
        success=False,
        metrics=[],
        error=error,
    )


def test_genuine_failures_pad_score_metrics_with_zeros() -> None:
    runner = EvaluationRunner(output_dir=Path("/tmp/unused"))
    results = [
        _success("a", [MetricValue(metric_name="rule_pass_rate", value=0.8)]),
        _success("b", [MetricValue(metric_name="rule_pass_rate", value=0.6)]),
        _failure("c"),
        _failure("d"),
    ]

    aggregate = runner._aggregate_metrics(results)

    # Average over all 4 attempted examples, not just the 2 survivors
    assert aggregate["avg_rule_pass_rate"] == pytest.approx(0.35)
    assert aggregate["min_rule_pass_rate"] == 0.0
    assert aggregate["max_rule_pass_rate"] == 0.8


def test_skipped_results_do_not_pad() -> None:
    runner = EvaluationRunner(output_dir=Path("/tmp/unused"))
    results = [
        _success("a", [MetricValue(metric_name="mean_f1", value=0.9)], product_type="layout_detection"),
        _success("b", [MetricValue(metric_name="mean_f1", value=0.7)], product_type="layout_detection"),
        # Cross-eval page the provider was never meant to be scored on
        _failure("c", error="No layout data for page 0", product_type="layout_detection"),
    ]

    aggregate = runner._aggregate_metrics(results)

    assert aggregate["avg_mean_f1"] == pytest.approx(0.8)
    assert aggregate["min_mean_f1"] == 0.7


def test_infra_failures_do_not_pad() -> None:
    runner = EvaluationRunner(output_dir=Path("/tmp/unused"))
    results = [
        _success("a", [MetricValue(metric_name="rule_pass_rate", value=0.8)]),
        _success("b", [MetricValue(metric_name="rule_pass_rate", value=0.6)]),
        # Harness-side errors: must not move the provider's score
        _failure("c", error="Worker error: evaluator crashed"),
        _failure("d", error="Evaluation error: division by zero"),
        _failure("e", error="Task execution error: cancelled"),
    ]

    aggregate = runner._aggregate_metrics(results)

    assert aggregate["avg_rule_pass_rate"] == pytest.approx(0.7)
    assert aggregate["min_rule_pass_rate"] == 0.6


def test_unadaptable_layout_output_pads_as_genuine_failure() -> None:
    runner = EvaluationRunner(output_dir=Path("/tmp/unused"))
    results = [
        _success("a", [MetricValue(metric_name="rule_pass_rate", value=0.8)]),
        _success("b", [MetricValue(metric_name="rule_pass_rate", value=0.6)]),
        # Provider produced output that can't be adapted to a LayoutOutput.
        # Surfaces as a "Worker error:" but is a provider failure, not a
        # harness crash, so it must count as a 0 in the denominator.
        _failure("c", error="Worker error: Inference output is not LayoutOutput and no provider adapter matched."),
    ]

    aggregate = runner._aggregate_metrics(results)

    # (0.8 + 0.6 + 0) / 3
    assert aggregate["avg_rule_pass_rate"] == pytest.approx(0.4667, abs=1e-3)
    assert aggregate["min_rule_pass_rate"] == 0.0


def test_diagnostic_count_metrics_are_not_padded() -> None:
    runner = EvaluationRunner(output_dir=Path("/tmp/unused"))
    # Count/lower-is-better values that happen to all fall in [0, 1]: a
    # synthetic 0 would either be meaningless or make failures look better.
    diag = [
        MetricValue(metric_name="tables_unmatched_expected", value=1.0),
        MetricValue(metric_name="num_predictions", value=1.0),
        MetricValue(metric_name="null_hallucination_rate", value=1.0),
        MetricValue(metric_name="rule_pass_rate", value=1.0),  # control: a real score
    ]
    diag_zero = [
        MetricValue(metric_name="tables_unmatched_expected", value=0.0),
        MetricValue(metric_name="num_predictions", value=0.0),
        MetricValue(metric_name="null_hallucination_rate", value=0.0),
        MetricValue(metric_name="rule_pass_rate", value=0.0),
    ]
    results = [
        _success("a", diag),
        _success("b", diag_zero),
        _failure("c"),
        _failure("d"),
    ]

    aggregate = runner._aggregate_metrics(results)

    # Diagnostics unchanged by failures
    assert aggregate["avg_tables_unmatched_expected"] == pytest.approx(0.5)
    assert aggregate["avg_num_predictions"] == pytest.approx(0.5)
    assert aggregate["avg_null_hallucination_rate"] == pytest.approx(0.5)
    # The real score is padded
    assert aggregate["avg_rule_pass_rate"] == pytest.approx(0.25)


def test_count_metadata_metrics_are_not_padded() -> None:
    runner = EvaluationRunner(output_dir=Path("/tmp/unused"))
    results = [
        _success("a", [MetricValue(metric_name="custom_elements", value=1.0, metadata={"count": 1})]),
        _success("b", [MetricValue(metric_name="custom_elements", value=0.0, metadata={"count": 0})]),
        _failure("c"),
    ]

    aggregate = runner._aggregate_metrics(results)

    assert aggregate["avg_custom_elements"] == pytest.approx(0.5)
    assert aggregate["total_custom_elements"] == 1.0


def test_padding_is_scoped_to_failing_product_type() -> None:
    runner = EvaluationRunner(output_dir=Path("/tmp/unused"))
    results = [
        _success("a", [MetricValue(metric_name="rule_pass_rate", value=1.0)], product_type="parse"),
        _success("b", [MetricValue(metric_name="mean_f1", value=1.0)], product_type="layout_detection"),
        _failure("c", product_type="layout_detection"),
    ]

    aggregate = runner._aggregate_metrics(results)

    # The layout failure pads layout metrics only, never parse metrics
    assert aggregate["avg_mean_f1"] == pytest.approx(0.5)
    assert aggregate["avg_rule_pass_rate"] == pytest.approx(1.0)


def test_metrics_outside_unit_range_are_not_padded() -> None:
    runner = EvaluationRunner(output_dir=Path("/tmp/unused"))
    results = [
        _success("a", [MetricValue(metric_name="some_measure", value=0.5)]),
        _success("b", [MetricValue(metric_name="some_measure", value=3.0)]),
        _failure("c"),
    ]

    aggregate = runner._aggregate_metrics(results)

    assert aggregate["avg_some_measure"] == pytest.approx(1.75)


def test_no_failures_is_a_no_op() -> None:
    runner = EvaluationRunner(output_dir=Path("/tmp/unused"))
    results = [
        _success("a", [MetricValue(metric_name="rule_pass_rate", value=0.8)]),
        _success("b", [MetricValue(metric_name="rule_pass_rate", value=0.6)]),
    ]

    aggregate = runner._aggregate_metrics(results)

    assert aggregate["avg_rule_pass_rate"] == pytest.approx(0.7)
    assert aggregate["min_rule_pass_rate"] == 0.6


# ---------------------------------------------------------------------------
# Synthesized zero for layout-detection cases with no usable inference output
# ---------------------------------------------------------------------------


def _layout_case(test_id: str, with_annotations: bool):
    from extract_bench.test_cases.schema import LayoutDetectionTestCase, LayoutTestRule

    rules = [LayoutTestRule(page=1, bbox=[0.1, 0.1, 0.2, 0.2], canonical_class="text")] if with_annotations else []
    return LayoutDetectionTestCase(
        test_id=test_id,
        group="layout",
        file_path=Path("/tmp/nonexistent.pdf"),
        test_rules=rules,
    )


def test_missing_layout_output_is_synthesized_as_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tc = _layout_case("layout/doc1", with_annotations=True)
    monkeypatch.setattr("extract_bench.evaluation.runner.load_test_cases", lambda **kwargs: [tc])

    runner = EvaluationRunner(output_dir=tmp_path, test_cases_dir=tmp_path)
    summary = runner.run_evaluation(use_rich=False)

    assert summary.failed == 1
    assert summary.successful == 0
    assert len(summary.per_example_results) == 1
    synthesized = summary.per_example_results[0]
    assert synthesized.success is False
    # Tagged PARSE so the synthetic 0 lands in the padding scope of the
    # cross-eval parse metrics the layout group actually reports.
    assert synthesized.product_type == "parse"
    assert synthesized.error == "No usable inference output for this example"


def test_layout_case_without_annotations_is_not_scored_as_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tc = _layout_case("layout/doc2", with_annotations=False)
    monkeypatch.setattr("extract_bench.evaluation.runner.load_test_cases", lambda **kwargs: [tc])

    runner = EvaluationRunner(output_dir=tmp_path, test_cases_dir=tmp_path)
    summary = runner.run_evaluation(use_rich=False)

    # Unscorable GT (no annotations) is a data issue, not a provider failure
    assert summary.failed == 0
    assert summary.per_example_results == []
