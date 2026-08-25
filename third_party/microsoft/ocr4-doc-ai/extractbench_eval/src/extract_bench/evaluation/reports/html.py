"""HTML report generation for evaluation results."""

from pathlib import Path

from extract_bench.schemas.evaluation import EvaluationSummary


def export_html(summary: EvaluationSummary, report_dir: Path) -> Path:
    """Export evaluation summary to HTML."""
    html_path = report_dir / "_evaluation_report.html"

    # Build HTML content
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "    <title>Evaluation Report</title>",
        "    <style>",
        "        * { margin: 0; padding: 0; box-sizing: border-box; }",
        "        body {",
        (
            "            font-family: -apple-system, BlinkMacSystemFont, "
            "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;"
        ),
        "            line-height: 1.6;",
        "            color: #333;",
        "            background: #f5f5f5;",
        "            padding: 20px;",
        "        }",
        "        .container {",
        "            max-width: 1200px;",
        "            margin: 0 auto;",
        "            background: white;",
        "            padding: 30px;",
        "            border-radius: 8px;",
        "            box-shadow: 0 2px 4px rgba(0,0,0,0.1);",
        "        }",
        "        h1 {",
        "            color: #2c3e50;",
        "            border-bottom: 3px solid #3498db;",
        "            padding-bottom: 10px;",
        "            margin-bottom: 20px;",
        "        }",
        "        h2 {",
        "            color: #34495e;",
        "            margin-top: 30px;",
        "            margin-bottom: 15px;",
        "            border-left: 4px solid #3498db;",
        "            padding-left: 10px;",
        "        }",
        "        .summary-grid {",
        "            display: grid;",
        "            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));",
        "            gap: 15px;",
        "            margin: 20px 0;",
        "        }",
        "        .summary-card {",
        "            background: #f8f9fa;",
        "            padding: 15px;",
        "            border-radius: 6px;",
        "            border-left: 4px solid #3498db;",
        "        }",
        "        .summary-card h3 {",
        "            font-size: 14px;",
        "            color: #7f8c8d;",
        "            margin-bottom: 5px;",
        "            text-transform: uppercase;",
        "            letter-spacing: 0.5px;",
        "        }",
        "        .summary-card .value {",
        "            font-size: 28px;",
        "            font-weight: bold;",
        "            color: #2c3e50;",
        "        }",
        "        .summary-card.success .value { color: #27ae60; }",
        "        .summary-card.failed .value { color: #e74c3c; }",
        "        .summary-card.skipped .value { color: #95a5a6; }",
        "        table {",
        "            width: 100%;",
        "            border-collapse: collapse;",
        "            margin: 20px 0;",
        "            background: white;",
        "        }",
        "        th, td {",
        "            padding: 12px;",
        "            text-align: left;",
        "            border-bottom: 1px solid #ddd;",
        "        }",
        "        th {",
        "            background: #3498db;",
        "            color: white;",
        "            font-weight: 600;",
        "        }",
        "        tr:hover { background: #f8f9fa; }",
        "        .metric-value {",
        "            font-weight: 600;",
        "        }",
        "        .metric-value.high { color: #27ae60; }",
        "        .metric-value.medium { color: #f39c12; }",
        "        .metric-value.low { color: #e74c3c; }",
        "        .timestamp {",
        "            color: #7f8c8d;",
        "            font-size: 14px;",
        "            margin-bottom: 20px;",
        "        }",
        "        .error-section {",
        "            background: #fff5f5;",
        "            border-left: 4px solid #e74c3c;",
        "            padding: 15px;",
        "            margin: 20px 0;",
        "            border-radius: 4px;",
        "        }",
        "        .error-item {",
        "            margin: 10px 0;",
        "            padding: 10px;",
        "            background: white;",
        "            border-radius: 4px;",
        "        }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class='container'>",
        "        <h1>Evaluation Report</h1>",
        (
            f"        <div class='timestamp'>Generated: "
            f"{summary.completed_at.isoformat() if summary.completed_at else 'N/A'}</div>"
        ),
        "",
        "        <h2>Summary</h2>",
        "        <div class='summary-grid'>",
        "            <div class='summary-card'>",
        "                <h3>Total Examples</h3>",
        f"                <div class='value'>{summary.total_examples}</div>",
        "            </div>",
        "            <div class='summary-card success'>",
        "                <h3>Successful</h3>",
        f"                <div class='value'>{summary.successful}</div>",
        "            </div>",
        "            <div class='summary-card failed'>",
        "                <h3>Failed</h3>",
        f"                <div class='value'>{summary.failed}</div>",
        "            </div>",
        "            <div class='summary-card skipped'>",
        "                <h3>Skipped</h3>",
        f"                <div class='value'>{summary.skipped}</div>",
        "            </div>",
    ]

    # Add avg cards for each stat in the summary grid
    for stat_name, agg in sorted(summary.aggregate_stats.items()):
        unit = agg.get("unit", "")
        display_name = stat_name.replace("_", " ").title()
        fmt = ".6f" if "$" in unit else ".0f"
        html_lines.extend(
            [
                "            <div class='summary-card'>",
                f"                <h3>Avg {display_name}</h3>",
                f"                <div class='value'>{agg['avg']:{fmt}}{unit}</div>",
                "            </div>",
            ]
        )

    html_lines.append("        </div>")

    # Add detailed stats sections
    for stat_name, agg in sorted(summary.aggregate_stats.items()):
        unit = agg.get("unit", "")
        display_name = stat_name.replace("_", " ").title()
        is_currency = "$" in unit
        fmt = ".6f" if is_currency else ".1f"
        fmt_total = ".4f" if is_currency else ".0f"
        html_lines.extend(
            [
                "",
                f"        <h2>{display_name} Statistics</h2>",
                "        <div class='summary-grid'>",
                "            <div class='summary-card'>",
                "                <h3>Total</h3>",
                f"                <div class='value'>{agg['total']:{fmt_total}}{unit}</div>",
                "            </div>",
                "            <div class='summary-card'>",
                "                <h3>Average</h3>",
                f"                <div class='value'>{agg['avg']:{fmt}}{unit}</div>",
                "            </div>",
                "            <div class='summary-card'>",
                "                <h3>Min</h3>",
                f"                <div class='value'>{agg['min']:{fmt}}{unit}</div>",
                "            </div>",
                "            <div class='summary-card'>",
                "                <h3>Max</h3>",
                f"                <div class='value'>{agg['max']:{fmt}}{unit}</div>",
                "            </div>",
                "            <div class='summary-card'>",
                "                <h3>P50</h3>",
                f"                <div class='value'>{agg['p50']:{fmt}}{unit}</div>",
                "            </div>",
                "            <div class='summary-card'>",
                "                <h3>P95</h3>",
                f"                <div class='value'>{agg['p95']:{fmt}}{unit}</div>",
                "            </div>",
                "            <div class='summary-card'>",
                "                <h3>P99</h3>",
                f"                <div class='value'>{agg['p99']:{fmt}}{unit}</div>",
                "            </div>",
                "        </div>",
            ]
        )

    # Add aggregate metrics table
    if summary.aggregate_metrics:
        html_lines.extend(
            [
                "",
                "        <h2>Aggregate Metrics</h2>",
                "        <table>",
                "            <thead>",
                "                <tr>",
                "                    <th>Metric</th>",
                "                    <th>Average</th>",
                "                    <th>Min</th>",
                "                    <th>Max</th>",
                "                </tr>",
                "            </thead>",
                "            <tbody>",
            ]
        )

        # Group metrics by base name (avg_, min_, max_)
        metric_groups: dict[str, dict[str, float]] = {}
        for metric_name, value in summary.aggregate_metrics.items():
            if metric_name.startswith("avg_"):
                base_name = metric_name.replace("avg_", "")
                if base_name not in metric_groups:
                    metric_groups[base_name] = {}
                metric_groups[base_name]["avg"] = value
            elif metric_name.startswith("min_"):
                base_name = metric_name.replace("min_", "")
                if base_name not in metric_groups:
                    metric_groups[base_name] = {}
                metric_groups[base_name]["min"] = value
            elif metric_name.startswith("max_"):
                base_name = metric_name.replace("max_", "")
                if base_name not in metric_groups:
                    metric_groups[base_name] = {}
                metric_groups[base_name]["max"] = value

        # Sort by average value (descending) for main metrics
        sorted_metrics = sorted(
            metric_groups.items(),
            key=lambda x: x[1].get("avg", 0),
            reverse=True,
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

        for base_name, values in sorted_metrics:
            avg_val = values.get("avg", 0)
            min_val = values.get("min", 0)
            max_val = values.get("max", 0)

            # Determine color class based on value
            if avg_val >= 0.9:
                color_class = "high"
            elif avg_val >= 0.7:
                color_class = "medium"
            else:
                color_class = "low"

            display_name = metric_display_names.get(
                base_name,
                base_name.replace("_", " ").replace("field accuracy ", "").title(),
            )
            html_lines.append(
                f"                <tr>"
                f"<td>{display_name}</td>"
                f"<td><span class='metric-value {color_class}'>"
                f"{avg_val:.4f}</span></td>"
                f"<td>{min_val:.4f}</td>"
                f"<td>{max_val:.4f}</td>"
                f"</tr>"
            )

        html_lines.extend(
            [
                "            </tbody>",
                "        </table>",
            ]
        )

    # Add errors section if any
    if summary.failed > 0:
        failed_results = [r for r in summary.per_example_results if not r.success]
        html_lines.extend(
            [
                "",
                "        <h2>Errors</h2>",
                "        <div class='error-section'>",
            ]
        )
        for result in failed_results:
            html_lines.extend(
                [
                    "            <div class='error-item'>",
                    f"                <strong>{result.test_id}</strong><br>",
                    f"                <span style='color: #e74c3c;'>{result.error}</span>",
                    "            </div>",
                ]
            )
        html_lines.append("        </div>")

    html_lines.extend(
        [
            "    </div>",
            "</body>",
            "</html>",
        ]
    )

    html_path.write_text("\n".join(html_lines))
    return html_path
