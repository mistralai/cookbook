"""CSV report generation for evaluation results."""

import csv
from pathlib import Path

from extract_bench.schemas.evaluation import EvaluationSummary


def export_csv(summary: EvaluationSummary, report_dir: Path) -> Path:
    """Export evaluation results to CSV."""
    csv_path = report_dir / "_evaluation_results.csv"

    # Collect all unique metric names and stat names
    all_metric_names: set[str] = set()
    all_stat_names: set[str] = set()
    for result in summary.per_example_results:
        for metric in result.metrics:
            all_metric_names.add(metric.metric_name)
        for stat in result.stats:
            all_stat_names.add(stat.name)
    sorted_metric_names = sorted(all_metric_names)
    sorted_stat_names = sorted(all_stat_names)

    # Write CSV
    fieldnames = (
        [
            "test_id",
            "example_id",
            "pipeline_name",
            "product_type",
            "success",
            "error",
            "tags",
        ]
        + sorted_stat_names
        + sorted_metric_names
    )

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in summary.per_example_results:
            row: dict[str, object] = {
                "test_id": result.test_id,
                "example_id": result.example_id,
                "pipeline_name": result.pipeline_name,
                "product_type": result.product_type,
                "success": result.success,
                "error": result.error or "",
                "tags": ",".join(result.tags),
            }
            # Add stat values
            stat_dict = {s.name: s.value for s in result.stats}
            for stat_name in sorted_stat_names:
                row[stat_name] = stat_dict.get(stat_name, "")
            # Add metric values
            metric_dict = {m.metric_name: m.value for m in result.metrics}
            for metric_name in sorted_metric_names:
                row[metric_name] = metric_dict.get(metric_name, "")

            writer.writerow(row)

    return csv_path
