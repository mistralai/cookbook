"""Markdown report generation for evaluation results."""

from pathlib import Path

from extract_bench.schemas.evaluation import EvaluationSummary


def export_markdown(summary: EvaluationSummary, report_dir: Path) -> Path:
    """Export evaluation summary to markdown."""
    md_path = report_dir / "_evaluation_report.md"

    lines = [
        "# Evaluation Report",
        "",
        (f"**Generated:** {summary.completed_at.isoformat() if summary.completed_at else 'N/A'}"),
        "",
        "## Summary",
        "",
        f"- **Total Examples:** {summary.total_examples}",
        f"- **Successful:** {summary.successful}",
        f"- **Failed:** {summary.failed}",
        f"- **Skipped:** {summary.skipped}",
        "",
    ]

    # Add operational stats sections if available
    if summary.aggregate_stats:
        for stat_name, agg in sorted(summary.aggregate_stats.items()):
            unit = agg.get("unit", "")
            display_name = stat_name.replace("_", " ").title()
            # Use more decimal places for small-value stats (cost, per-page)
            is_currency = "$" in unit
            fmt = ".6f" if is_currency else ".1f"
            fmt_total = ".4f" if is_currency else ".0f"
            lines.extend(
                [
                    f"## {display_name} Statistics",
                    "",
                    f"- **Average:** {agg['avg']:{fmt}}{unit}",
                    f"- **Total:** {agg['total']:{fmt_total}}{unit}",
                    f"- **Min:** {agg['min']:{fmt}}{unit}",
                    f"- **Max:** {agg['max']:{fmt}}{unit}",
                    f"- **P50:** {agg['p50']:{fmt}}{unit}",
                    f"- **P95:** {agg['p95']:{fmt}}{unit}",
                    f"- **P99:** {agg['p99']:{fmt}}{unit}",
                    f"- **Count:** {agg['count']}",
                    "",
                ]
            )

    if summary.aggregate_metrics:
        lines.extend(
            [
                "## Aggregate Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ]
        )
        metric_display_names = {
            "teds": "TEDS (All)",
            "teds_predicted": "TEDS (Among Predicted Tables)",
            "teds_struct": "TEDS-Struct (All)",
            "teds_struct_predicted": "TEDS-Struct (Among Predicted Tables)",
            "teds_struct_bool": "TEDS-Struct+BoolContent (All)",
            "teds_struct_bool_predicted": "TEDS-Struct+BoolContent (Among Predicted Tables)",
            "grits_con": "GriTS Con (All)",
            "grits_con_predicted": "GriTS Con (Among Predicted Tables)",
            "ref_grits_con": "Ref GriTS Con (All)",
            "ref_grits_con_predicted": "Ref GriTS Con (Among Predicted Tables)",
            "rule_pass_rate": "Rule Pass Rate",
            "text_similarity": "Text Similarity",
            "accuracy": "Accuracy",
            "qa_answer_match": "QA Match",
            "layout_reading_order_pass_rate": "Layout Reading Order Pass Rate",
        }
        for metric_name, value in sorted(summary.aggregate_metrics.items()):
            if metric_name.startswith("avg_"):
                base_name = metric_name.replace("avg_", "")
                display_name = metric_display_names.get(base_name, base_name.replace("_", " ").title())
                lines.append(f"| {display_name} | {value:.4f} |")
        lines.append("")

    if summary.failed > 0:
        lines.extend(
            [
                "## Errors",
                "",
            ]
        )
        failed_results = [r for r in summary.per_example_results if not r.success]
        for result in failed_results:
            lines.append(f"### {result.test_id}")
            lines.append(f"- **Error:** {result.error}")
            lines.append("")

    md_path.write_text("\n".join(lines))
    return md_path
