"""Rule-level CSV export for evaluation summaries."""

from __future__ import annotations

from pathlib import Path

from extract_bench.schemas.evaluation import EvaluationSummary


def export_rule_csv(
    summary: EvaluationSummary,
    report_dir: Path,
    dataset_dir: Path | None = None,
) -> Path:
    """Export normalized rule outcomes to a CSV file.

    Note: The triage module (which provided build_rule_outcomes) has been removed.
    This function now writes an empty file as a placeholder.
    """
    csv_path = report_dir / "_evaluation_rule_results.csv"
    csv_path.write_text("")
    return csv_path
