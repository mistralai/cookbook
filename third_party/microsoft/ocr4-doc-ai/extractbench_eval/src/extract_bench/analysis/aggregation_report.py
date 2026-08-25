"""Aggregation dashboard report for multi-category benchmark runs.

Generates a self-contained HTML dashboard showing all categories side-by-side,
with per-category metric selectors, pipeline metadata, and links to detailed reports.

Uses the same design system (Newsreader / Plus Jakarta Sans / JetBrains Mono,
warm editorial palette) as the detailed evaluation reports.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extract_bench.analysis.cost_summary import summarize_documents, summarize_splits
from extract_bench.analysis.metric_definitions import (
    EXTRACT_DEFAULT_METRIC,
    METRIC_GROUP_SEPARATOR,
    display_name,
    order_metrics,
    tooltip_dict,
)
from extract_bench.analysis.report_static_assets import aggregation_report_script, aggregation_report_style
from extract_bench.schemas.evaluation import EvaluationSummary


def _load_category_summary(report_json: Path) -> EvaluationSummary | None:
    """Load an EvaluationSummary from a per-category report JSON."""
    try:
        data = json.loads(report_json.read_text(encoding="utf-8"))
        return EvaluationSummary.model_validate(data)
    except Exception:
        return None


# Default "main metric" per category type.  Everything else falls back to rule_pass_rate.
# The extract splits are the ExtractBench length taxonomy; they open on the
# paper's headline metric rather than on whatever sorts first alphabetically.
_DEFAULT_METRICS: dict[str, str] = {
    "table": "grits_trm_composite",
    "layout": "layout_element_rule_pass_rate",
    "text_content": "content_faithfulness",
    "text_formatting": "semantic_formatting",
    "form": "rule_form_field_pass_rate",
    "short": EXTRACT_DEFAULT_METRIC,
    "medium": EXTRACT_DEFAULT_METRIC,
    "long": EXTRACT_DEFAULT_METRIC,
}


def _extract_category_data(name: str, summary: EvaluationSummary) -> dict[str, Any]:
    """Extract display data for a single category from its EvaluationSummary."""
    metrics = summary.aggregate_metrics

    # Build metric list from avg_* keys only
    available: dict[str, float] = {}
    for key in metrics:
        if not key.startswith("avg_"):
            continue
        metric_name = key[len("avg_") :]
        # Skip _predicted duplicates and _judge duplicates
        if "_predicted" in metric_name or "_judge" in metric_name:
            continue
        available[metric_name] = metrics[key]

    # Headline metrics first, then a divider, then the rest alphabetically.
    metric_list: list[dict[str, Any]] = []
    for metric_name in order_metrics(available):
        if metric_name == METRIC_GROUP_SEPARATOR:
            metric_list.append({"name": METRIC_GROUP_SEPARATOR, "displayName": "", "value": None})
            continue
        metric_list.append(
            {
                "name": metric_name,
                "displayName": display_name(metric_name),
                "value": available[metric_name],  # raw 0-1 float
            }
        )

    # Determine default metric for this category
    default_metric = _DEFAULT_METRICS.get(name, "rule_pass_rate")
    # Fall back if default isn't available in the metrics list
    if default_metric not in available:
        if "rule_pass_rate" in available:
            default_metric = "rule_pass_rate"
        else:
            # order_metrics puts the headline block first, so this picks the
            # best available metric rather than the alphabetically first one.
            ordered = order_metrics(available, separator=False)
            default_metric = ordered[0] if ordered else ""

    return {
        "name": name,
        "displayName": name.replace("_", " ").title(),
        "files": summary.total_examples,
        # Documents the pipeline produced no usable result for. They are scored
        # zero rather than dropped, so they already drag every mean down; the
        # count is what tells a bad extraction apart from a crashed run.
        "failed": summary.failed,
        "defaultMetric": default_metric,
        "metrics": metric_list,
    }


def generate_aggregation_report(
    pipeline_output_dir: Path,
    groups: list[str],
    pipeline_name: str = "",
) -> Path:
    """Generate an aggregation dashboard HTML showing all categories side-by-side.

    Args:
        pipeline_output_dir: Directory containing per-category subdirectories with
            _evaluation_report.json files.
        groups: List of category/group names to include.
        pipeline_name: Pipeline name for display in the report header.

    Returns:
        Path to the generated HTML file.
    """
    # Load pipeline metadata
    pipeline_metadata: dict[str, Any] = {}
    metadata_path = pipeline_output_dir / "_metadata.json"
    if metadata_path.exists():
        try:
            pipeline_metadata = json.loads(metadata_path.read_text(encoding="utf-8")).get("pipeline", {})
        except Exception:
            pass

    if not pipeline_name and pipeline_metadata.get("pipeline_name"):
        pipeline_name = pipeline_metadata["pipeline_name"]

    categories: list[dict[str, Any]] = []
    per_split_examples: dict[str, list[Any]] = {}
    for group_name in groups:
        report_path = pipeline_output_dir / group_name / "_evaluation_report.json"
        summary = _load_category_summary(report_path)
        if summary is not None:
            cat_data = _extract_category_data(group_name, summary)
            examples = summary.per_example_results or []
            per_split_examples[group_name] = list(examples)
            cat_data["cost"] = summarize_documents(examples).as_dict()
            categories.append(cat_data)

    total_files = sum(c["files"] for c in categories)
    # Splits are disjoint by construction -- a document's split IS its length
    # tag -- so failures add up across them the same way the file counts do.
    total_failed = sum(c["failed"] for c in categories)
    # Overall cost pools every document once across splits, rather than
    # averaging the three split means — a 20-document split must not weigh the
    # same as a 252-document one.
    overall_cost = summarize_splits(per_split_examples)

    # One selection governs every card, so the dropdown offers the union of what
    # the splits report, in the canonical headline-first order. A split missing
    # the selection renders as "—": a zero there would read as a real score.
    union: dict[str, float] = {}
    for cat in categories:
        for m in cat["metrics"]:
            if m["name"] != METRIC_GROUP_SEPARATOR:
                union.setdefault(m["name"], 0.0)
    unified_metrics: list[dict[str, Any]] = []
    for metric_name in order_metrics(union):
        if metric_name == METRIC_GROUP_SEPARATOR:
            unified_metrics.append({"name": METRIC_GROUP_SEPARATOR, "displayName": ""})
        else:
            unified_metrics.append({"name": metric_name, "displayName": display_name(metric_name)})
    # Start on whichever per-split default the splits already agreed on.
    defaults = [c["defaultMetric"] for c in categories if c.get("defaultMetric")]
    unified_default = max(set(defaults), key=defaults.count) if defaults else ""
    if unified_default not in union:
        unified_default = "rule_pass_rate" if "rule_pass_rate" in union else (next(iter(union), ""))

    data_blob = {
        "pipelineName": pipeline_name,
        "pipelineMetadata": pipeline_metadata,
        "generatedAt": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "totalFiles": total_files,
        "totalFailed": total_failed,
        "categories": categories,
        "unifiedMetrics": unified_metrics,
        "defaultMetric": unified_default,
        "overallCost": overall_cost.as_dict(),
        "metricTooltips": tooltip_dict(),
    }

    data_json = json.dumps(data_blob, default=str, ensure_ascii=False)
    data_json = data_json.replace("</script>", "<\\/script>")
    data_json = data_json.replace("<!--", "<\\!--")

    # Build full HTML by concatenation (same pattern as detailed_report.py)
    parts: list[str] = []
    parts.append(_HTML_HEAD)
    parts.append(aggregation_report_style())
    parts.append("\n</head>\n")
    parts.append(_HTML_BODY)

    # Data blob
    parts.append("\n<script>\nconst DATA = ")
    parts.append(data_json)
    parts.append(";\n</script>\n")

    parts.append(aggregation_report_script())
    parts.append("\n</body>\n</html>\n")

    html = "".join(parts)
    output_path = pipeline_output_dir / "_evaluation_report_dashboard.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# HTML template parts — uses same design system as detailed_report.py
# ---------------------------------------------------------------------------

_FONT_URL = (
    "https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@"
    "0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400"
    "&family=Plus+Jakarta+Sans:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500&display=swap"
)

_HTML_HEAD = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{_FONT_URL}" rel="stylesheet">
"""

_HTML_BODY = """\
<body>
<div class="report-container">
  <header class="report-header">
    <h1 id="report-title">Evaluation Report</h1>
    <p class="subtitle" id="subtitle"></p>
  </header>

  <div class="summary-row" id="summary-cards"></div>

  <div class="section-header">
    <h2 class="section-title">Categories</h2>
    <select class="metric-selector" id="metric-select"></select>
  </div>
  <div class="categories-grid" id="categories-grid"></div>
</div>
"""
