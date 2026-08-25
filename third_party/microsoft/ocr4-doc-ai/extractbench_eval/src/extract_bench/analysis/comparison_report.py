"""HTML report generator for pipeline comparisons.

Generates a rich, interactive comparison report matching the dashboard's
warm editorial design system (Newsreader / Plus Jakarta Sans / JetBrains Mono).
"""

import base64
import html
import json
from pathlib import Path
from typing import Any

from extract_bench.analysis.metric_definitions import (
    display_name_dict,
    tooltip_dict,
)
from extract_bench.analysis.report_static_assets import comparison_report_script, comparison_report_style


def _get_file_data_url(file_path: Path) -> str | None:
    """Convert a file to a data URL for embedding in HTML."""
    if not file_path.exists():
        return None

    try:
        # Determine MIME type
        suffix = file_path.suffix.lower()
        mime_types = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
        }
        mime_type = mime_types.get(suffix, "application/octet-stream")

        # Read file and encode
        with open(file_path, "rb") as f:
            file_data = f.read()
            encoded = base64.b64encode(file_data).decode("utf-8")
            return f"data:{mime_type};base64,{encoded}"
    except Exception:
        return None


METRIC_DISPLAY_NAMES: dict[str, str] = display_name_dict()


def _format_pct(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def resolve_comparison_pdf_base_url(
    test_cases_dir: Path | None,
    report_path: Path,
    explicit_url: str | None = None,
) -> str:
    """Return a URL path from the comparison report HTML to the test PDF directory.

    Unlike the detailed report (which appends ``{testId}.pdf`` under an optional
    ``pdfs/`` root), comparison previews join ``pdf_base_url`` with
    ``input_file_rel``, which is already relative to ``test_cases_dir``. Using
    the test-cases directory itself avoids a doubled ``pdfs/`` segment.
    """
    if explicit_url:
        return explicit_url.rstrip("/")
    if test_cases_dir is None or not test_cases_dir.exists():
        return ""

    import os

    pdf_root = test_cases_dir.resolve()
    try:
        return os.path.relpath(pdf_root, report_path.parent.resolve()).replace("\\", "/")
    except ValueError:
        return ""


def _embed_input_files(
    matched_results: list[dict[str, Any]],
    test_cases_dir: Path | None,
    original_base_path: str,
) -> None:
    """Embed image input files as base64 data URLs and compute relative paths (in-place)."""
    import os

    # Use original_base_path as the base for relative path computation
    # (may differ from test_cases_dir which requires local existence)
    base_for_rel = original_base_path or (str(test_cases_dir) if test_cases_dir else "")

    for result in matched_results:
        input_file = result.get("input_file")
        if not input_file:
            continue
        fp = Path(input_file)
        if not fp.exists() and test_cases_dir:
            rel = fp.name
            candidate = test_cases_dir / rel
            if candidate.exists():
                fp = candidate

        # Compute relative path for PDF.js URL resolution
        # Try file-system-based relpath first, fall back to string manipulation
        if test_cases_dir and fp.exists():
            try:
                result["input_file_rel"] = os.path.relpath(str(fp.resolve()), str(test_cases_dir.resolve())).replace(
                    "\\", "/"
                )
            except ValueError:
                result["input_file_rel"] = str(fp)
        elif base_for_rel and input_file.startswith(base_for_rel):
            # String-based relative path (for CI paths that don't exist locally)
            rel_path = input_file[len(base_for_rel) :]
            if rel_path.startswith("/"):
                rel_path = rel_path[1:]
            result["input_file_rel"] = rel_path

        # Embed images as base64 (not PDFs — they're too large, use PDF.js instead)
        if fp.exists() and fp.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif"):
            data_url = _get_file_data_url(fp)
            if data_url:
                result["input_data_url"] = data_url


def generate_comparison_html(comparison_data: dict[str, Any], output_path: Path | None = None) -> str | Path:
    """
    Generate an interactive HTML report for pipeline comparison.

    :param comparison_data: Comparison data from compare_pipelines()
    :param output_path: Optional path to save the HTML report. If None, returns HTML string.
    :return: HTML string if output_path is None, otherwise Path to the generated file
    """
    matched_results = comparison_data["matched_results"]
    stats = comparison_data["stats"]
    product_type = comparison_data.get("product_type", "extract")
    comparison_metric = comparison_data.get("comparison_metric", "accuracy")
    original_base_path = comparison_data.get("original_base_path", "")
    pdf_base_url = comparison_data.get("pdf_base_url", "")

    metric_display_name = METRIC_DISPLAY_NAMES.get(comparison_metric, comparison_metric)

    # Embed input images as data URLs and compute relative paths
    test_cases_dir_str = comparison_data.get("original_base_path", "")
    test_cases_dir = Path(test_cases_dir_str) if test_cases_dir_str else None
    _embed_input_files(matched_results, test_cases_dir, original_base_path)

    pipeline_a_name = html.escape(stats["pipeline_a_name"])
    pipeline_b_name = html.escape(stats["pipeline_b_name"])

    # Sort matched results by test_id
    matched_results_sorted = sorted(matched_results, key=lambda r: r["test_id"])

    html_content = _build_html(
        matched_results_sorted,
        stats,
        product_type,
        comparison_metric,
        metric_display_name,
        pipeline_a_name,
        pipeline_b_name,
        original_base_path,
        pdf_base_url,
    )

    if output_path is None:
        return html_content

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html_content)
    return output_path


def _build_html(
    matched_results: list[dict[str, Any]],
    stats: dict[str, Any],
    product_type: str,
    comparison_metric: str,
    metric_display_name: str,
    pipeline_a_name: str,
    pipeline_b_name: str,
    original_base_path: str,
    pdf_base_url: str = "",
) -> str:
    """Build the full HTML document."""

    # Pre-compute result rows HTML
    rows_html = []
    for i, result in enumerate(matched_results):
        rows_html.append(_build_result_row(result, stats, i))

    title = f"Pipeline Comparison: {pipeline_a_name} vs {pipeline_b_name}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    {comparison_report_style()}
</head>
<body>
    <header class="page-header">
        <div class="header-inner">
            <h1>Pipeline Comparison</h1>
            <p class="subtitle">{pipeline_a_name} <span class="vs">vs</span> {pipeline_b_name}</p>
            <div class="metric-selector">
                <label class="metric-selector-label" for="metricSelect">Primary metric:</label>
                <select id="metricSelect" onchange="switchMetric(this.value)"></select>
            </div>
        </div>
    </header>

    <main class="container">
        {_build_stats_section(stats, pipeline_a_name, pipeline_b_name)}

        <div class="path-config">
            <div class="path-config-header">
                <span class="path-config-title">Data Path Configuration</span>
                <button class="path-config-toggle" onclick="togglePathConfig()">Configure</button>
            </div>
            <div class="path-config-body" id="pathConfigBody">
                <p class="hint">If you received this report from someone else, set your local path to the test data folder:</p>
                <input type="text" id="dataBasePath" placeholder="e.g., /Users/yourname/data/financial_tables" onchange="updateBasePath()" />
                <p class="current-path">Current: <span id="currentBasePath">(using original paths)</span></p>
            </div>
        </div>

        {_build_filter_bar(stats, pipeline_a_name, pipeline_b_name)}

        <div class="results-list" id="resultsList">
            <div class="results-table-header">
                <div class="col-id">Test ID</div>
                <div class="col-metric">{pipeline_a_name}</div>
                <div class="col-metric">{pipeline_b_name}</div>
                <div class="col-delta">Delta</div>
                <div class="col-category">Category</div>
            </div>
            {"".join(rows_html)}
        </div>
    </main>

    <script>
        const comparisonData = {json.dumps(matched_results)};
        const pipelineAName = {json.dumps(stats["pipeline_a_name"])};
        const pipelineBName = {json.dumps(stats["pipeline_b_name"])};
        const productType = {json.dumps(product_type)};
        let comparisonMetric = {json.dumps(comparison_metric)};
        let metricDisplayName = {json.dumps(METRIC_DISPLAY_NAMES.get(comparison_metric, comparison_metric))};
        const originalBasePath = {json.dumps(original_base_path)};
        const pdfBaseUrl = {json.dumps(pdf_base_url)};
        const metricTooltips = {json.dumps(tooltip_dict())};
        const metricDisplayNames = {json.dumps(METRIC_DISPLAY_NAMES)};
    </script>
    {comparison_report_script()}
</body>
</html>"""


def _build_result_row(result: dict[str, Any], stats: dict[str, Any], index: int) -> str:
    """Build a single result row with expandable detail area."""
    test_id = html.escape(result["test_id"])
    metric_a = result["pipeline_a"].get("metric_value")
    metric_b = result["pipeline_b"].get("metric_value")
    category = result["category"]

    def fmt(val: float | None) -> str:
        if val is None:
            return '<span class="na">N/A</span>'
        pct = val * 100
        css = _metric_color_class(val)
        return f'<span class="metric-val {css}">{pct:.1f}%</span>'

    def delta_str(a: float | None, b: float | None) -> str:
        if a is None or b is None:
            return '<span class="na">&mdash;</span>'
        d = (a - b) * 100
        sign = "+" if d > 0 else ""
        css = "delta-pos" if d > 0 else ("delta-neg" if d < 0 else "delta-zero")
        return f'<span class="{css}">{sign}{d:.1f}pp</span>'

    category_labels = {
        "a_better": f"{stats['pipeline_a_name']} Better",
        "b_better": f"{stats['pipeline_b_name']} Better",
        "both_bad": "Both Bad",
        "tie": "Tie",
    }
    cat_label = html.escape(category_labels.get(category, category))

    return f"""
            <div class="result-row" data-category="{category}" data-index="{index}">
                <div class="row-summary" onclick="toggleRow({index})">
                    <div class="col-id"><span class="expand-icon" id="icon-{index}">&#9654;</span> {test_id}</div>
                    <div class="col-metric">{fmt(metric_a)}</div>
                    <div class="col-metric">{fmt(metric_b)}</div>
                    <div class="col-delta">{delta_str(metric_a, metric_b)}</div>
                    <div class="col-category"><span class="badge badge-{category.replace("_", "-")}">{cat_label}</span></div>
                </div>
                <div class="row-detail" id="detail-{index}"></div>
            </div>"""


def _build_stats_section(stats: dict[str, Any], a_name: str, b_name: str) -> str:
    return f"""
        <section class="stats-grid">
            <div class="stat-card" data-filter="all" onclick="filterFromCard(this)">
                <div class="stat-value">{stats["total_matched"]}</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat-card stat-a-better" data-filter="a_better" onclick="filterFromCard(this)">
                <div class="stat-value">{stats["a_better"]}</div>
                <div class="stat-label">{a_name} Better</div>
            </div>
            <div class="stat-card stat-b-better" data-filter="b_better" onclick="filterFromCard(this)">
                <div class="stat-value">{stats["b_better"]}</div>
                <div class="stat-label">{b_name} Better</div>
            </div>
            <div class="stat-card stat-tie" data-filter="tie" onclick="filterFromCard(this)">
                <div class="stat-value">{stats["tie"]}</div>
                <div class="stat-label">Tie</div>
            </div>
            <div class="stat-card stat-bad" data-filter="both_bad" onclick="filterFromCard(this)">
                <div class="stat-value">{stats["both_bad"]}</div>
                <div class="stat-label">Both Bad</div>
            </div>
        </section>"""


def _build_filter_bar(stats: dict[str, Any], a_name: str, b_name: str) -> str:
    return f"""
        <div class="filter-bar">
            <button class="filter-btn active" data-filter="all">All ({stats["total_matched"]})</button>
            <button class="filter-btn" data-filter="a_better">{a_name} Better ({stats["a_better"]})</button>
            <button class="filter-btn" data-filter="b_better">{b_name} Better ({stats["b_better"]})</button>
            <button class="filter-btn" data-filter="tie">Tie ({stats["tie"]})</button>
            <button class="filter-btn" data-filter="both_bad">Both Bad ({stats["both_bad"]})</button>
        </div>"""


def _metric_color_class(val: float | None) -> str:
    if val is None:
        return "metric-na"
    if val >= 0.9:
        return "metric-high"
    if val >= 0.7:
        return "metric-mid"
    if val >= 0.5:
        return "metric-low"
    return "metric-bad"
