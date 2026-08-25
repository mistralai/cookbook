"""Tests for avg_/min_/max_ aggregate metric grouping helpers."""

from __future__ import annotations

from extract_bench.analysis.detailed_report import _build_data_blob
from extract_bench.analysis.metric_aggregates import group_avg_min_max, to_agg_metric_records
from extract_bench.analysis.metric_definitions import EXTRACT_HEADLINE_METRICS
from extract_bench.schemas.evaluation import EvaluationSummary


def test_group_avg_min_max_groups_complete_triples() -> None:
    flat = {
        "avg_accuracy": 0.9,
        "min_accuracy": 0.8,
        "max_accuracy": 1.0,
        "avg_value_f1": 0.7,
        "min_value_f1": 0.5,
        "max_value_f1": 0.9,
    }
    groups = group_avg_min_max(flat)
    assert groups == {
        "accuracy": {"avg": 0.9, "min": 0.8, "max": 1.0},
        "value_f1": {"avg": 0.7, "min": 0.5, "max": 0.9},
    }


def test_group_avg_min_max_ignores_non_prefixed_and_excludes() -> None:
    flat = {
        "avg_accuracy": 0.9,
        "min_accuracy": 0.8,
        "max_accuracy": 1.0,
        "avg_tables_expected": 3.0,
        "min_tables_expected": 1.0,
        "max_tables_expected": 5.0,
        "example_count": 2.0,
        "micro_array_record_f1": 0.85,
        "total_array_record_f1_tp": 10.0,
    }
    groups = group_avg_min_max(flat, exclude={"tables_expected"})
    assert set(groups) == {"accuracy"}
    assert "example_count" not in groups
    assert "micro_array_record_f1" not in groups


def test_group_avg_min_max_partial_triple() -> None:
    groups = group_avg_min_max({"avg_accuracy": 0.9, "max_accuracy": 1.0})
    assert groups == {"accuracy": {"avg": 0.9, "max": 1.0}}


def test_to_agg_metric_records_defaults_and_sort() -> None:
    flat = {
        "avg_zzz": 0.5,
        "min_zzz": 0.4,
        "max_zzz": 0.6,
        "avg_accuracy": 0.95,
        "min_accuracy": 0.9,
        "max_accuracy": 1.0,
    }
    # Include a headline metric so order_metrics puts it first.
    headline = EXTRACT_HEADLINE_METRICS[0]
    flat[f"avg_{headline}"] = 0.8
    flat[f"min_{headline}"] = 0.7
    flat[f"max_{headline}"] = 0.9

    ordered = to_agg_metric_records(flat)
    assert [r["name"] for r in ordered] == [headline, "accuracy", "zzz"]
    assert ordered[1]["displayName"]
    assert ordered[1]["avg"] == 0.95
    assert ordered[1]["min"] == 0.9
    assert ordered[1]["max"] == 1.0

    by_avg = to_agg_metric_records(flat, sort_by_avg=True)
    assert [r["name"] for r in by_avg] == ["accuracy", headline, "zzz"]


def test_build_data_blob_tag_metrics_are_grouped() -> None:
    summary = EvaluationSummary(
        total_examples=1,
        successful=1,
        failed=0,
        skipped=0,
        aggregate_metrics={
            "avg_accuracy": 0.9,
            "min_accuracy": 0.9,
            "max_accuracy": 0.9,
            "micro_array_record_f1": 0.8,
        },
        tag_metrics={
            "domain:D1": {
                "avg_accuracy": 0.9,
                "min_accuracy": 0.85,
                "max_accuracy": 0.95,
                "example_count": 2.0,
                "micro_array_record_f1": 0.8,
                "total_array_record_f1_tp": 4.0,
            }
        },
    )
    blob = _build_data_blob(summary)
    tag = blob["tagMetrics"]["domain:D1"]
    assert tag["exampleCount"] == 2
    assert [m["name"] for m in tag["metrics"]] == ["accuracy"]
    assert tag["metrics"][0]["avg"] == 0.9
    assert tag["metrics"][0]["min"] == 0.85
    assert tag["metrics"][0]["max"] == 0.95
    # Flat leftover keys must not leak into the report payload.
    assert "micro_array_record_f1" not in tag
    assert not any(k.startswith("avg_") for k in tag)


def test_build_data_blob_default_metric_prefers_headline() -> None:
    headline = EXTRACT_HEADLINE_METRICS[0]
    summary = EvaluationSummary(
        total_examples=1,
        successful=1,
        failed=0,
        skipped=0,
        aggregate_metrics={
            "avg_array_record_row_count_ratio": 1.5,
            "min_array_record_row_count_ratio": 1.0,
            "max_array_record_row_count_ratio": 2.0,
            f"avg_{headline}": 0.8,
            f"min_{headline}": 0.7,
            f"max_{headline}": 0.9,
        },
    )
    blob = _build_data_blob(summary)
    # aggMetrics stays avg-sorted (row count ratio first), but the Examples
    # selector should open on the tag-table / headline metric.
    assert blob["aggMetrics"][0]["name"] == "array_record_row_count_ratio"
    assert blob["defaultMetric"] == headline
