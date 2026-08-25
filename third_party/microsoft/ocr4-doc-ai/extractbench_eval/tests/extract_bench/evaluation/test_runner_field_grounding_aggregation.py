from __future__ import annotations

import pytest

from extract_bench.evaluation.runner import EvaluationRunner
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue


def _result(metrics: list[MetricValue]) -> EvaluationResult:
    return EvaluationResult(
        test_id="test",
        example_id="example",
        pipeline_name="pipeline",
        product_type="extract",
        success=True,
        metrics=metrics,
    )


def _result_with_diagnostics(diagnostic_metrics: list[MetricValue]) -> EvaluationResult:
    result = _result([])
    result.diagnostic_metrics = diagnostic_metrics
    return result


def test_diagnostic_metrics_are_aggregated_separately(tmp_path) -> None:
    runner = object.__new__(EvaluationRunner)

    aggregate_metrics = runner._aggregate_metrics(
        [
            _result([MetricValue(metric_name="accuracy", value=1.0)]),
            _result([MetricValue(metric_name="accuracy", value=0.0)]),
        ]
    )
    aggregate_diagnostics = runner._aggregate_metrics(
        [
            _result_with_diagnostics([MetricValue(metric_name="field_accuracy_employees", value=1.0)]),
            _result_with_diagnostics([MetricValue(metric_name="field_accuracy_employees", value=0.0)]),
        ],
        metric_list_name="diagnostic_metrics",
    )

    assert aggregate_metrics["avg_accuracy"] == pytest.approx(0.5)
    assert "avg_field_accuracy_employees" not in aggregate_metrics
    assert aggregate_diagnostics["avg_field_accuracy_employees"] == pytest.approx(0.5)


def test_precision_recall_f1_are_micro_aggregated_from_counts(tmp_path) -> None:
    runner = object.__new__(EvaluationRunner)
    summary = runner._aggregate_metrics(
        [
            _result(
                [
                    MetricValue(metric_name="precision", value=1.0, metadata={"tp": 1, "fp": 0, "fn": 9}),
                    MetricValue(metric_name="recall", value=0.1, metadata={"tp": 1, "fp": 0, "fn": 9}),
                    MetricValue(metric_name="f1", value=0.1818, metadata={"tp": 1, "fp": 0, "fn": 9}),
                ]
            ),
            _result(
                [
                    MetricValue(metric_name="precision", value=0.5, metadata={"tp": 1, "fp": 1, "fn": 0}),
                    MetricValue(metric_name="recall", value=1.0, metadata={"tp": 1, "fp": 1, "fn": 0}),
                    MetricValue(metric_name="f1", value=0.6667, metadata={"tp": 1, "fp": 1, "fn": 0}),
                ]
            ),
        ]
    )

    assert summary["avg_precision"] == pytest.approx(0.75)
    assert summary["avg_recall"] == pytest.approx(0.55)
    assert summary["avg_f1"] == pytest.approx((0.1818 + 0.6667) / 2)
    assert summary["micro_precision"] == pytest.approx(2 / 3)
    assert summary["micro_recall"] == pytest.approx(2 / 11)
    assert summary["micro_f1"] == pytest.approx(2 * (2 / 3) * (2 / 11) / ((2 / 3) + (2 / 11)))
    assert summary["total_precision_tp"] == 2.0
    assert summary["total_precision_fp"] == 1.0
    assert summary["total_precision_fn"] == 9.0


def test_extract_element_pass_rate_reports_document_weighted_and_micro_metrics(tmp_path) -> None:
    runner = object.__new__(EvaluationRunner)
    summary = runner._aggregate_metrics(
        [
            _result(
                [
                    MetricValue(
                        metric_name="extract_element_pass_rate",
                        value=0.0,
                        metadata={"passed": 0, "total": 1, "verified_passed": 0, "verified_total": 1},
                    ),
                ]
            ),
            _result(
                [
                    MetricValue(
                        metric_name="extract_element_pass_rate",
                        value=0.8,
                        metadata={"passed": 8, "total": 10, "verified_passed": 4, "verified_total": 4},
                    ),
                ]
            ),
        ]
    )

    assert summary["avg_extract_element_pass_rate"] == pytest.approx(0.4)
    assert summary["micro_extract_element_pass_rate"] == pytest.approx(8 / 11)
    assert summary["micro_verified_extract_element_pass_rate"] == pytest.approx(4 / 5)
    assert "headline_extract_element_pass_rate" not in summary
    assert summary["total_extract_element_pass_rate_passed"] == 8.0
    assert summary["total_extract_element_pass_rate_evaluated"] == 11.0
    assert summary["total_verified_extract_element_pass_rate_passed"] == 4.0
    assert summary["total_verified_extract_element_pass_rate_evaluated"] == 5.0


def test_extract_field_pass_rates_report_document_weighted_and_micro_metrics(tmp_path) -> None:
    runner = object.__new__(EvaluationRunner)
    summary = runner._aggregate_metrics(
        [
            _result(
                [
                    MetricValue(
                        metric_name=metric_name,
                        value=0.0,
                        metadata={"passed": 0, "total": total, "verified_passed": 0, "verified_total": total},
                    )
                    for metric_name, total in (
                        ("extract_field_element_pass_rate", 1),
                        ("extract_field_rule_pass_rate", 3),
                        ("extract_field_localization_pass_rate", 1),
                        ("extract_field_classification_pass_rate", 1),
                        ("extract_field_attribution_pass_rate", 1),
                    )
                ]
            ),
            _result(
                [
                    MetricValue(
                        metric_name=metric_name,
                        value=passed / total,
                        metadata={
                            "passed": passed,
                            "total": total,
                            "verified_passed": verified_passed,
                            "verified_total": verified_total,
                        },
                    )
                    for metric_name, passed, total, verified_passed, verified_total in (
                        ("extract_field_element_pass_rate", 8, 10, 4, 4),
                        ("extract_field_rule_pass_rate", 24, 30, 12, 12),
                        ("extract_field_localization_pass_rate", 9, 10, 4, 4),
                        ("extract_field_classification_pass_rate", 10, 10, 4, 4),
                        ("extract_field_attribution_pass_rate", 8, 10, 4, 4),
                    )
                ]
            ),
        ]
    )

    assert summary["avg_extract_field_element_pass_rate"] == pytest.approx(0.4)
    assert summary["micro_extract_field_element_pass_rate"] == pytest.approx(8 / 11)
    assert summary["micro_verified_extract_field_element_pass_rate"] == pytest.approx(4 / 5)
    assert summary["avg_extract_field_rule_pass_rate"] == pytest.approx(0.4)
    assert summary["micro_extract_field_rule_pass_rate"] == pytest.approx(24 / 33)
    assert summary["micro_verified_extract_field_rule_pass_rate"] == pytest.approx(12 / 15)
    assert summary["micro_extract_field_localization_pass_rate"] == pytest.approx(9 / 11)
    assert summary["micro_verified_extract_field_localization_pass_rate"] == pytest.approx(4 / 5)
    assert summary["micro_extract_field_classification_pass_rate"] == pytest.approx(10 / 11)
    assert summary["micro_verified_extract_field_classification_pass_rate"] == pytest.approx(4 / 5)
    assert summary["micro_extract_field_attribution_pass_rate"] == pytest.approx(8 / 11)
    assert summary["micro_verified_extract_field_attribution_pass_rate"] == pytest.approx(4 / 5)


def test_extract_element_pass_rate_omits_headline_when_verified_counts_absent(tmp_path) -> None:
    runner = object.__new__(EvaluationRunner)
    summary = runner._aggregate_metrics(
        [
            _result(
                [
                    MetricValue(
                        metric_name="extract_element_pass_rate",
                        value=1.0,
                        metadata={"passed": 1, "total": 1},
                    ),
                ]
            ),
            _result(
                [
                    MetricValue(
                        metric_name="extract_element_pass_rate",
                        value=0.5,
                        metadata={"passed": 1, "total": 2},
                    ),
                ]
            ),
        ]
    )

    assert summary["avg_extract_element_pass_rate"] == pytest.approx(0.75)
    assert summary["micro_extract_element_pass_rate"] == pytest.approx(2 / 3)
    assert "headline_extract_element_pass_rate" not in summary


def test_record_metrics_report_macro_and_micro_aggregates(tmp_path) -> None:
    runner = object.__new__(EvaluationRunner)
    summary = runner._aggregate_metrics(
        [
            _result(
                [
                    MetricValue(metric_name="record_precision", value=1.0, metadata={"tp": 1, "fp": 0, "fn": 9}),
                    MetricValue(metric_name="record_recall", value=0.1, metadata={"tp": 1, "fp": 0, "fn": 9}),
                    MetricValue(metric_name="record_f1", value=0.1818, metadata={"tp": 1, "fp": 0, "fn": 9}),
                    MetricValue(metric_name="record_accuracy", value=0.1, metadata={"tp": 1, "fp": 0, "fn": 9}),
                ]
            ),
            _result(
                [
                    MetricValue(metric_name="record_precision", value=0.5, metadata={"tp": 1, "fp": 1, "fn": 0}),
                    MetricValue(metric_name="record_recall", value=1.0, metadata={"tp": 1, "fp": 1, "fn": 0}),
                    MetricValue(metric_name="record_f1", value=0.6667, metadata={"tp": 1, "fp": 1, "fn": 0}),
                    MetricValue(metric_name="record_accuracy", value=0.5, metadata={"tp": 1, "fp": 1, "fn": 0}),
                ]
            ),
        ]
    )

    assert summary["avg_record_precision"] == pytest.approx(0.75)
    assert summary["avg_record_recall"] == pytest.approx(0.55)
    assert summary["avg_record_f1"] == pytest.approx((0.1818 + 0.6667) / 2)
    assert summary["avg_record_accuracy"] == pytest.approx(0.3)
    assert summary["micro_record_precision"] == pytest.approx(2 / 3)
    assert summary["micro_record_recall"] == pytest.approx(2 / 11)
    assert summary["micro_record_f1"] == pytest.approx(2 * (2 / 3) * (2 / 11) / ((2 / 3) + (2 / 11)))
    assert summary["micro_record_accuracy"] == pytest.approx(2 / 12)
