"""Multi-pipeline leaderboard report.

Generates a self-contained HTML leaderboard comparing all pipelines in the
output directory side-by-side, with per-category metric selectors, best-score
highlighting, and links to individual pipeline dashboards.

Uses the same design system (Newsreader / Plus Jakarta Sans / JetBrains Mono,
warm editorial palette) as the other reports.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extract_bench.analysis.aggregation_report import _DEFAULT_METRICS
from extract_bench.analysis.cost_summary import summarize_documents, summarize_splits
from extract_bench.analysis.metric_definitions import METRIC_GROUP_SEPARATOR, order_metrics
from extract_bench.analysis.metric_definitions import display_name as _display_name
from extract_bench.schemas.evaluation import EvaluationSummary


def _load_pipeline_data(pipeline_dir: Path) -> dict[str, Any] | None:
    """Load pipeline metadata and per-category avg metrics from a pipeline output dir."""
    metadata_path = pipeline_dir / "_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    pm = metadata.get("pipeline", {})
    pipeline_name = pm.get("pipeline_name", pipeline_dir.name)

    # Discover categories (subdirs with _evaluation_report.json)
    categories: list[dict[str, Any]] = []
    per_split_examples: dict[str, list[Any]] = {}
    for subdir in sorted(pipeline_dir.iterdir()):
        if not subdir.is_dir():
            continue
        report_path = subdir / "_evaluation_report.json"
        if not report_path.exists():
            continue
        try:
            summary = EvaluationSummary.model_validate(json.loads(report_path.read_text(encoding="utf-8")))
        except Exception:
            continue

        # Extract avg metrics, same filtering as aggregation_report
        metrics_dict: dict[str, float] = {}
        for key in sorted(summary.aggregate_metrics.keys()):
            if not key.startswith("avg_"):
                continue
            metric_name = key[len("avg_") :]
            if "_predicted" in metric_name or "_judge" in metric_name:
                continue
            metrics_dict[metric_name] = summary.aggregate_metrics[key]

        examples = list(summary.per_example_results or [])
        per_split_examples[subdir.name] = examples

        # Documents actually contributing each metric. Needed because the
        # overall is a mean over documents, and a category's mean can only be
        # reweighted correctly if we know how many documents it was taken over.
        # It is NOT `total_examples`: the grounding metrics are withheld on
        # documents whose ground truth carries no verified boxes, so weighting
        # those by the full document count would inflate them.
        metric_counts: dict[str, int] = {}
        for example in examples:
            seen_here = set()
            for metric in getattr(example, "metrics", None) or ():
                name = metric.metric_name if hasattr(metric, "metric_name") else metric.get("metric_name")
                if name and name not in seen_here:
                    seen_here.add(name)
                    metric_counts[name] = metric_counts.get(name, 0) + 1

        categories.append(
            {
                "name": subdir.name,
                "files": summary.total_examples,
                # Documents the pipeline produced no usable result for. They are
                # scored zero rather than dropped, so they already drag the mean
                # down; the count tells a bad extraction from a crashed run.
                "failed": summary.failed,
                "metrics": metrics_dict,
                "metricCounts": metric_counts,
                "cost": summarize_documents(examples).as_dict(),
            }
        )

    if not categories:
        return None

    return {
        "name": pipeline_name,
        "dirName": pipeline_dir.name,
        "displayName": pipeline_name.replace("_", " ").title(),
        "provider": pm.get("provider_name", ""),
        "productType": pm.get("product_type", ""),
        "config": pm.get("config", {}),
        "categories": categories,
        # Pooled over every document once, not the mean of the split means.
        "overallCost": summarize_splits(per_split_examples).as_dict(),
    }


def generate_leaderboard_report(
    output_dir: Path,
    pipeline_names: list[str] | None = None,
    output_file: Path | None = None,
) -> Path:
    """Generate a leaderboard HTML comparing multiple pipelines.

    Args:
        output_dir: Parent directory containing pipeline subdirectories.
        pipeline_names: Optional list of pipeline dir names to include.
            If None, auto-discovers all subdirs with _metadata.json.
        output_file: Path for the output HTML. Defaults to output_dir/_leaderboard.html.

    Returns:
        Path to the generated HTML file.
    """
    output_dir = Path(output_dir)

    # Discover or filter pipelines
    if pipeline_names:
        dirs = [output_dir / name for name in pipeline_names]
    else:
        dirs = sorted(d for d in output_dir.iterdir() if d.is_dir() and (d / "_metadata.json").exists())

    pipelines: list[dict[str, Any]] = []
    for d in dirs:
        data = _load_pipeline_data(d)
        if data is not None:
            pipelines.append(data)

    if not pipelines:
        raise ValueError(f"No valid pipeline results found in {output_dir}")

    # Collect union of categories and metrics
    all_categories: list[str] = []
    seen_cats: set[str] = set()
    for p in pipelines:
        for cat in p["categories"]:
            if cat["name"] not in seen_cats:
                all_categories.append(cat["name"])
                seen_cats.add(cat["name"])

    # Build scores matrix and collect per-category metrics
    scores: dict[str, dict[str, dict[str, float]]] = {}
    metric_counts: dict[str, dict[str, dict[str, int]]] = {}
    category_files: dict[str, dict[str, int]] = {}
    category_failures: dict[str, dict[str, int]] = {}
    category_metrics: dict[str, list[str]] = {}
    all_metric_names: set[str] = set()

    for cat_name in all_categories:
        scores[cat_name] = {}
        metric_counts[cat_name] = {}
        category_files[cat_name] = {}
        category_failures[cat_name] = {}
        metric_set: set[str] = set()
        for p in pipelines:
            cat_data = next((c for c in p["categories"] if c["name"] == cat_name), None)
            if cat_data:
                scores[cat_name][p["name"]] = cat_data["metrics"]
                metric_counts[cat_name][p["name"]] = cat_data.get("metricCounts", {})
                category_files[cat_name][p["name"]] = cat_data["files"]
                category_failures[cat_name][p["name"]] = cat_data.get("failed", 0)
                metric_set.update(cat_data["metrics"].keys())
                all_metric_names.update(cat_data["metrics"].keys())
            else:
                scores[cat_name][p["name"]] = {}
                metric_counts[cat_name][p["name"]] = {}
                category_files[cat_name][p["name"]] = 0
                category_failures[cat_name][p["name"]] = 0
        # Headline metrics first, then a divider, then the rest alphabetically.
        category_metrics[cat_name] = order_metrics(metric_set)

    # Build metric display names
    metric_names_map: dict[str, str] = {}
    for m in all_metric_names:
        metric_names_map[m] = _display_name(m)

    # Build default metrics per category
    default_metrics: dict[str, str] = {}
    for cat_name in all_categories:
        # The separator is a divider, never a selectable default.
        selectable = [m for m in category_metrics.get(cat_name, []) if m != METRIC_GROUP_SEPARATOR]
        default = _DEFAULT_METRICS.get(cat_name, "rule_pass_rate")
        if default not in selectable:
            if "rule_pass_rate" in selectable:
                default = "rule_pass_rate"
            else:
                # `selectable` is headline-first, so this falls back to the best
                # available metric rather than the alphabetically first one.
                default = selectable[0] if selectable else ""
        default_metrics[cat_name] = default

    # One selection governs every row, so the dropdown offers the union of what
    # the splits report, in the canonical headline-first order. A split missing
    # the selection renders as "—": a zero there would read as a real score.
    unified_metrics = order_metrics(all_metric_names)
    unified_selectable = [m for m in unified_metrics if m != METRIC_GROUP_SEPARATOR]
    # Start on whichever per-split default the splits already agreed on.
    unified_default = ""
    if default_metrics:
        unified_default = max(set(default_metrics.values()), key=list(default_metrics.values()).count)
    if unified_default not in unified_selectable:
        if "rule_pass_rate" in unified_selectable:
            unified_default = "rule_pass_rate"
        else:
            unified_default = unified_selectable[0] if unified_selectable else ""

    data_blob = {
        "generatedAt": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "defaultMetrics": default_metrics,
        "unifiedMetrics": unified_metrics,
        "defaultMetric": unified_default,
        "pipelines": [
            {
                "name": p["name"],
                "dirName": p["dirName"],
                "displayName": p["displayName"],
                "provider": p["provider"],
                "productType": p["productType"],
                "config": p["config"],
                "dashboardUrl": p["dirName"] + "/_evaluation_report_dashboard.html",
            }
            for p in pipelines
        ],
        "categories": all_categories,
        "categoryDisplayNames": {c: c.replace("_", " ").title() for c in all_categories},
        "categoryFiles": category_files,
        # failures[category][pipeline] and overallFailures[pipeline], against the
        # same denominators the file counts use. Splits are disjoint by
        # construction, so the overall is the sum across them.
        "categoryFailures": category_failures,
        "overallFailures": {p["name"]: sum(c.get("failed", 0) for c in p["categories"]) for p in pipelines},
        "overallFiles": {p["name"]: sum(c["files"] for c in p["categories"]) for p in pipelines},
        "scores": scores,
        # Documents behind each category/pipeline/metric mean, so the overall
        # can be reweighted into a mean over documents.
        "metricCounts": metric_counts,
        "metricNames": metric_names_map,
        "categoryMetrics": category_metrics,
        # cost[category][pipeline] and overallCost[pipeline]. Every figure is a
        # mean over documents counted once; overall pools documents across
        # splits rather than averaging the per-split means.
        "cost": {
            cat_name: {
                p["name"]: (next((c for c in p["categories"] if c["name"] == cat_name), {}) or {}).get("cost", {})
                for p in pipelines
            }
            for cat_name in all_categories
        },
        "overallCost": {p["name"]: p.get("overallCost", {}) for p in pipelines},
    }

    data_json = json.dumps(data_blob, default=str, ensure_ascii=False)
    data_json = data_json.replace("</script>", "<\\/script>")
    data_json = data_json.replace("<!--", "<\\!--")

    parts: list[str] = []
    parts.append(_HTML_HEAD)
    parts.append(_CSS)
    parts.append("</style>\n</head>\n")
    parts.append(_HTML_BODY)
    parts.append("\n<script>\nconst DATA = ")
    parts.append(data_json)
    parts.append(";\n</script>\n")
    parts.append("<script>\n")
    parts.append(_JS)
    parts.append("\n</script>\n")
    parts.append("</body>\n</html>\n")

    html = "".join(parts)
    if output_file is None:
        output_file = output_dir / "_leaderboard.html"
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")
    return output_file


# ---------------------------------------------------------------------------
# HTML template
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
<title>Benchmark Leaderboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{_FONT_URL}" rel="stylesheet">
<style>
"""

_CSS = """\
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
:root {
    --bg: #f8f7f4;
    --fg: #1c1917;
    --card: #ffffff;
    --border: #e7e5e4;
    --muted: #78716c;
    --muted-light: #a8a29e;
    --cream: #faf9f6;
    --emerald: #059669;
    --emerald-bg: #ecfdf5;
    --emerald-light: #d1fae5;
    --amber: #d97706;
    --amber-bg: #fffbeb;
    --red: #dc2626;
    --red-bg: #fef2f2;
    --blue: #2563eb;
    --blue-bg: #eff6ff;
    --gold: #b8860b;
    --gold-bg: #fef9e7;
    --gold-border: #e6c547;
    --font-heading: 'Newsreader', Georgia, serif;
    --font-body: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
    --shadow-sm: 0 1px 2px rgba(28,25,23,0.05);
    --shadow-md: 0 4px 6px -1px rgba(28,25,23,0.07), 0 2px 4px -2px rgba(28,25,23,0.05);
    --shadow-lg: 0 10px 25px -5px rgba(28,25,23,0.1), 0 4px 10px -4px rgba(28,25,23,0.06);
    --radius: 12px;
    --radius-sm: 6px;
}
html { font-size: 15px; }
body {
    font-family: var(--font-body);
    background: var(--bg);
    color: var(--fg);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--muted-light); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

.report-container {
    max-width: 1600px;
    margin: 0 auto;
    padding: 40px 32px 80px;
}

/* ───── Header ───── */
.report-header { margin-bottom: 40px; }
.report-header h1 {
    font-family: var(--font-heading);
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--fg);
    line-height: 1.15;
}
.report-header .subtitle {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 8px;
    letter-spacing: 0.01em;
}

/* ───── Table wrapper ───── */
.leaderboard-wrap {
    overflow-x: auto;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-lg);
}
.leaderboard-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 600px;
}

/* ───── Cells ───── */
.leaderboard-table th,
.leaderboard-table td {
    padding: 18px 24px;
    border-bottom: 1px solid var(--border);
    text-align: center;
    vertical-align: middle;
    transition: background 0.12s ease;
}
.leaderboard-table th {
    background: var(--cream);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    position: sticky;
    top: 0;
    z-index: 2;
    padding: 20px 24px;
    border-bottom: 2px solid var(--border);
    vertical-align: bottom;
}

/* Sticky first column */
.leaderboard-table th:first-child,
.leaderboard-table td:first-child {
    text-align: left;
    position: sticky;
    left: 0;
    z-index: 1;
    background: var(--card);
    border-right: 1px solid var(--border);
    min-width: 220px;
    padding-left: 28px;
}
.leaderboard-table th:first-child {
    background: var(--cream);
    z-index: 3;
    vertical-align: bottom;
}
.category-header-label {
    font-family: var(--font-heading);
    font-size: 1rem;
    font-weight: 600;
    color: var(--fg);
    text-transform: none;
    letter-spacing: -0.01em;
}
.leaderboard-table tbody tr:last-child td {
    border-bottom: none;
}

/* Row hover (category label column only) */
.leaderboard-table tbody tr:not(.overall-row):hover td:first-child {
    background: #f5f4f1;
}

/* ───── Pipeline header ───── */
.pipeline-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    min-width: 140px;
}
.pipeline-header .pipeline-crown {
    font-size: 1.3rem;
    line-height: 1;
    filter: drop-shadow(0 1px 2px rgba(184,134,11,0.3));
}
.pipeline-header .pipeline-name {
    font-family: var(--font-heading);
    font-size: 1rem;
    font-weight: 700;
    color: var(--fg);
    text-transform: none;
    letter-spacing: -0.01em;
    line-height: 1.3;
}
.pipeline-header .pipeline-name a {
    color: inherit;
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: border-color 0.15s, color 0.15s;
}
.pipeline-header .pipeline-name a:hover {
    color: var(--blue);
    border-bottom-color: var(--blue);
}
.pipeline-header .pipeline-tier {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 500;
    color: var(--muted-light);
    text-transform: none;
    letter-spacing: 0.02em;
    background: var(--bg);
    padding: 2px 8px;
    border-radius: 99px;
    border: 1px solid var(--border);
}

/* Winner column header glow */
.pipeline-header.is-winner .pipeline-name a {
    color: var(--gold);
}
.pipeline-header.is-winner .pipeline-tier {
    background: var(--gold-bg);
    border-color: var(--gold-border);
    color: var(--gold);
}

/* ───── Category cell ───── */
.category-cell {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.category-name {
    font-family: var(--font-heading);
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--fg);
    line-height: 1.3;
}
.category-name .file-count {
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--muted-light);
    margin-left: 2px;
}
.category-selector {
    width: 100%;
    max-width: 210px;
    padding: 5px 8px;
    font-family: var(--font-body);
    font-size: 0.73rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--cream);
    color: var(--muted);
    cursor: pointer;
    outline: none;
    transition: border-color 0.15s;
}
.category-selector:focus { border-color: var(--blue); }
.category-selector:hover { border-color: var(--muted-light); }
/* The 0.73rem above sized a dropdown repeated inside each narrow row cell.
   The single selector is the table's primary control, so it reads at the
   size of the "Category" label it sits under. */
#metric-select {
    margin-top: 8px;
    max-width: none;
    padding: 7px 10px;
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--fg);
}

/* ───── Column hover: simple bounding box ───── */
.leaderboard-table th[data-col],
.leaderboard-table td[data-col] {
    cursor: pointer;
}
.leaderboard-table th[data-col].col-hover {
    box-shadow: inset 2px 0 0 var(--muted), inset -2px 0 0 var(--muted), inset 0 2px 0 var(--muted);
}
.leaderboard-table .overall-row td[data-col].col-hover {
    box-shadow: inset 2px 0 0 var(--muted), inset -2px 0 0 var(--muted), inset 0 -2px 0 var(--muted);
}
.leaderboard-table tbody tr:not(.overall-row) td[data-col].col-hover {
    box-shadow: inset 2px 0 0 var(--muted), inset -2px 0 0 var(--muted);
}

/* ───── Score cell ───── */
.score-wrap {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    min-width: 90px;
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    transition: background 0.15s ease;
}
.score-wrap.is-best {
    background: var(--emerald-bg);
}
.score-number {
    font-family: var(--font-mono);
    font-size: 1rem;
    font-weight: 500;
    white-space: nowrap;
    line-height: 1;
}
.score-wrap.is-best .score-number {
    font-weight: 700;
    font-size: 1.05rem;
}
.score-bar-track {
    width: 100%;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}
.bar-emerald { background: var(--emerald); }
.bar-amber { background: var(--amber); }
.bar-red { background: var(--red); }
.score-badge {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--emerald);
    line-height: 1;
}
.score-na {
    color: var(--muted-light);
    font-size: 0.8rem;
    font-style: italic;
}
.color-emerald { color: var(--emerald); }
.color-amber { color: var(--amber); }
.color-red { color: var(--red); }

/* ───── Overall row ───── */
/* ───── Cost, inline under the score it belongs to ───── */
.cell-cost {
    display: flex;
    flex-direction: column;
    align-items: center;
    font-family: var(--font-mono);
    line-height: 1.3;
    margin-top: 6px;
}
.cell-cost-page {
    font-size: 0.8rem;
    font-weight: 500;
    color: #6b6759;
}
.cell-cost-unit {
    font-size: 0.68rem;
    font-weight: 400;
    color: #8a8577;
}
.cell-cost-total {
    font-size: 0.66rem;
    font-weight: 400;
    color: #a5a094;
}
/* ───── Failures, under the cost they invalidate ───── */
/* Shown on every cell, not only the failing ones: a missing line would leave
   "nothing failed" indistinguishable from "nobody counted". */
.cell-fail, .cell-fail-none {
    display: block;
    font-family: var(--font-mono);
    font-size: 0.66rem;
    line-height: 1.3;
    margin-top: 4px;
}
.cell-fail {
    font-weight: 500;
    color: var(--red);
}
.cell-fail-none { color: #a5a094; }

.overall-row td {
    border-top: 2px solid var(--border);
    border-bottom: none;
    background: var(--cream);
    padding-top: 20px;
    padding-bottom: 20px;
}
.overall-row td:first-child {
    background: var(--cream) !important;
    border-right-color: var(--border);
}
.overall-row:hover td,
.overall-row:hover td:first-child {
    background: #f3f1ec !important;
}
.overall-label {
    font-family: var(--font-heading);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--fg);
    letter-spacing: -0.01em;
}
.overall-sublabel {
    font-size: 0.7rem;
    font-weight: 400;
    color: var(--muted);
    display: block;
    margin-top: 2px;
}
/* Overall score cells */
.overall-row .score-wrap {
    background: transparent;
}
.overall-row .score-wrap.is-best {
    background: var(--emerald-bg);
}
.overall-row .score-number {
    color: var(--fg);
}
.overall-row .score-wrap.is-best .score-number {
    color: var(--emerald);
    font-weight: 600;
}
.overall-row .score-bar-track {
    background: var(--border);
}
.overall-row .score-badge {
    color: var(--emerald);
}
.overall-row .score-na {
    color: var(--muted-light);
}

@media (max-width: 768px) {
    .report-container { padding: 20px 16px 48px; }
    .report-header h1 { font-size: 1.8rem; }
}
"""

_HTML_BODY = """\
<body>
<div class="report-container">
  <header class="report-header">
    <h1>Benchmark Leaderboard</h1>
    <p class="subtitle" id="subtitle"></p>
  </header>

  <div class="leaderboard-wrap">
    <table class="leaderboard-table" id="leaderboard-table">
      <thead id="table-head"></thead>
      <tbody id="table-body"></tbody>
    </table>
  </div>
</div>
"""

_JS = """\
(function() {
  function colorClass(rate) {
    if (rate >= 80) return 'emerald';
    if (rate >= 50) return 'amber';
    return 'red';
  }

  function pct(val) { return val.toFixed(1) + '%'; }

  function esc(s) {
    if (s == null) return '';
    var d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
  }

  // ─── State ───
  // Divider entry between the paper's headline metrics and the rest.
  var SEPARATOR = '__separator__';

  // Cost formatters. Null means no priced document in the pool — an em dash,
  // never a misleading $0.00.
  function usd(v) {
    if (v == null) return '—';
    return v < 1 ? '$' + v.toFixed(4) : '$' + v.toFixed(2);
  }
  function cents(v) {
    if (v == null) return '—';
    return (v * 100).toFixed(2) + '¢';
  }


  // One metric for the whole table: every row and the overall move together.
  var selectedMetric = DATA.defaultMetric;

  // Subtitle
  document.getElementById('subtitle').textContent =
    DATA.pipelines.length + ' pipeline' + (DATA.pipelines.length !== 1 ? 's' : '') +
    ' across ' + DATA.categories.length + ' categories';

  // ─── Helpers ───
  function getScore(category, pipelineName) {
    var cs = DATA.scores[category];
    if (!cs) return null;
    var ps = cs[pipelineName];
    if (!ps) return null;
    var v = ps[selectedMetric];
    return (v !== undefined && v !== null) ? v : null;
  }

  function findBest(category) {
    var bestVal = -1, bestNames = [];
    for (var i = 0; i < DATA.pipelines.length; i++) {
      var v = getScore(category, DATA.pipelines[i].name);
      if (v === null) continue;
      if (v > bestVal) { bestVal = v; bestNames = [DATA.pipelines[i].name]; }
      else if (v === bestVal) { bestNames.push(DATA.pipelines[i].name); }
    }
    return bestNames;
  }

  // Mean over DOCUMENTS, not over categories: each category's mean is weighted
  // by the number of documents it was taken over, so a 20-document split cannot
  // weigh as much as a 252-document one. Weights come from metricCounts rather
  // than the file count, because grounding metrics are withheld on documents
  // whose ground truth carries no boxes.
  function getOverallScore(pipelineName) {
    var weighted = 0, docs = 0, plainSum = 0, plainCount = 0;
    for (var i = 0; i < DATA.categories.length; i++) {
      var cat = DATA.categories[i];
      var v = getScore(cat, pipelineName);
      if (v === null) continue;
      plainSum += v; plainCount++;
      var counts = ((DATA.metricCounts || {})[cat] || {})[pipelineName] || {};
      var n = counts[selectedMetric];
      if (typeof n === 'number' && n > 0) { weighted += v * n; docs += n; }
    }
    if (docs > 0) return weighted / docs;
    // No per-metric counts (older reports): fall back to the category mean
    // rather than showing nothing.
    return plainCount > 0 ? plainSum / plainCount : null;
  }

  function getOverallWinners() {
    var best = -1, names = [];
    for (var i = 0; i < DATA.pipelines.length; i++) {
      var v = getOverallScore(DATA.pipelines[i].name);
      if (v === null) continue;
      if (v > best) { best = v; names = [DATA.pipelines[i].name]; }
      else if (v === best) { names.push(DATA.pipelines[i].name); }
    }
    return names;
  }

  function getMaxFiles(category) {
    var files = DATA.categoryFiles[category] || {};
    var max = 0;
    for (var p in files) { if (files[p] > max) max = files[p]; }
    return max;
  }

  // Extract a clean tier/model label from config
  function getTierLabel(p) {
    var cfg = p.config || {};
    if (cfg.tier) return cfg.tier;
    if (cfg.model) return cfg.model;
    if (cfg.ocr_system) return cfg.ocr_system;
    // Fallback: use first config value that's a short string
    for (var k in cfg) {
      var v = cfg[k];
      if (typeof v === 'string' && v.length < 30) return v;
    }
    return p.productType || '';
  }

  // Build a score cell with progress bar
  // Cost rides in the same cell as the score it belongs to: per-page is the
  // comparable figure so it leads, with the run total beneath it as context.
  function costLines(c) {
    if (!c || !c.hasCost) return '';
    var h = '<div class="cell-cost">';
    h += '<span class="cell-cost-page">' + cents(c.meanPerPageUsd)
      + '<span class="cell-cost-unit"> / page</span></span>';
    h += '<span class="cell-cost-total">' + usd(c.totalUsd) + ' total</span>';
    h += '</div>';
    return h;
  }

  // Documents the pipeline produced no usable result for. They are scored zero
  // rather than dropped, so the score above already carries the penalty — this
  // line separates "extracted badly" from "never returned an answer". Sits
  // outside costLines because a wholly failed run reports no cost at all, and
  // that is exactly when the count matters most.
  function failLines(failed, total) {
    if (!total) return '';
    if (!failed) return '<span class="cell-fail-none">0 / ' + total + ' failed</span>';
    return '<span class="cell-fail">' + failed + ' / ' + total + ' failed ('
      + pct(100 * failed / total) + ')</span>';
  }

  function buildScoreCell(v, isBest, isOverall, cost, failed, total) {
    var footer = costLines(cost) + failLines(failed, total);
    if (v === null) return '<span class="score-na">N/A</span>' + footer;
    var pctVal = v * 100;
    var c = colorClass(pctVal);
    var cls = 'score-wrap' + (isBest ? ' is-best' : '');
    var h = '<div class="' + cls + '">';
    h += '<span class="score-number color-' + c + '">' + pct(pctVal) + '</span>';
    h += '<div class="score-bar-track"><div class="score-bar-fill bar-' + c
      + '" style="width:' + Math.min(pctVal, 100).toFixed(1) + '%"></div></div>';
    if (isBest) h += '<span class="score-badge">Best</span>';
    h += footer;
    h += '</div>';
    return h;
  }

  // ─── Render ───
  function renderHead() {
    var winners = getOverallWinners();
    var thead = document.getElementById('table-head');
    // The metric selector heads the row axis it governs, rather than repeating
    // once per row.
    var html = '<tr><th><span class="category-header-label">Category</span>';
    html += '<select class="category-selector" id="metric-select">';
    var uMetrics = DATA.unifiedMetrics || [];
    for (var mi = 0; mi < uMetrics.length; mi++) {
      var um = uMetrics[mi];
      if (um === SEPARATOR) {
        html += '<option disabled>──────────</option>';
        continue;
      }
      var usel = um === selectedMetric ? ' selected' : '';
      html += '<option value="' + esc(um) + '"' + usel + '>' + esc(DATA.metricNames[um] || um) + '</option>';
    }
    html += '</select></th>';
    for (var i = 0; i < DATA.pipelines.length; i++) {
      var p = DATA.pipelines[i];
      var isWinner = winners.indexOf(p.name) >= 0;
      var tierLabel = getTierLabel(p);
      html += '<th data-col="' + i + '" data-url="' + esc(p.dashboardUrl)
        + '"><div class="pipeline-header' + (isWinner ? ' is-winner' : '') + '">';
      if (isWinner) html += '<span class="pipeline-crown">\\ud83d\\udc51</span>';
      html += '<span class="pipeline-name">' + esc(p.displayName) + '</span>';
      var sub = p.provider || '';
      if (tierLabel && tierLabel !== p.provider) sub += (sub ? ' / ' : '') + tierLabel;
      if (sub) html += '<span class="pipeline-tier">' + esc(sub) + '</span>';
      html += '</div></th>';
    }
    html += '</tr>';
    thead.innerHTML = html;
  }

  function renderBody() {
    var tbody = document.getElementById('table-body');
    var html = '';

    // Category rows
    for (var ci = 0; ci < DATA.categories.length; ci++) {
      var cat = DATA.categories[ci];
      var bestPipelines = findBest(cat);
      var files = getMaxFiles(cat);

      html += '<tr>';
      html += '<td><div class="category-cell">';
      html += '<span class="category-name">' + esc(DATA.categoryDisplayNames[cat] || cat);
      html += ' <span class="file-count">' + files + ' files</span></span>';
      html += '</div></td>';

      var catCost = ((DATA.cost || {})[cat]) || {};
      var catFail = ((DATA.categoryFailures || {})[cat]) || {};
      var catFiles = ((DATA.categoryFiles || {})[cat]) || {};
      for (var pi = 0; pi < DATA.pipelines.length; pi++) {
        var pName = DATA.pipelines[pi].name;
        var v = getScore(cat, pName);
        var isBest = bestPipelines.indexOf(pName) >= 0;
        html += '<td data-col="' + pi + '" data-url="' + esc(DATA.pipelines[pi].dashboardUrl)
          + '">' + buildScoreCell(v, isBest, false, catCost[pName], catFail[pName], catFiles[pName]) + '</td>';
      }
      html += '</tr>';
    }

    // Overall row
    var overallWinners = getOverallWinners();

    html += '<tr class="overall-row">';
    html += '<td><span class="overall-label">Overall'
      + '<span class="overall-sublabel">Mean over all documents</span></span></td>';
    for (var opi = 0; opi < DATA.pipelines.length; opi++) {
      var opName = DATA.pipelines[opi].name;
      var ov = getOverallScore(opName);
      var oIsBest = overallWinners.indexOf(opName) >= 0;
      // Overall cost pools every document once across splits — NOT the mean of
      // the per-category costs, which would let a 20-document split weigh as
      // much as a 252-document one.
      html += '<td data-col="' + opi + '" data-url="' + esc(DATA.pipelines[opi].dashboardUrl)
        + '">' + buildScoreCell(ov, oIsBest, true, (DATA.overallCost || {})[opName],
          (DATA.overallFailures || {})[opName], (DATA.overallFiles || {})[opName]) + '</td>';
    }
    html += '</tr>';

    tbody.innerHTML = html;
  }

  // The selector lives in the head, which render() rebuilds, so rebind there.
  function bindMetricSelector() {
    var select = document.getElementById('metric-select');
    if (!select) return;
    select.addEventListener('change', function(e) {
      selectedMetric = e.target.value;
      render();
      var again = document.getElementById('metric-select');
      if (again) again.focus();
    });
  }

  function bindColumnInteractions() {
    var table = document.getElementById('leaderboard-table');
    var lastCol = null;

    function highlightCol(colIdx) {
      if (colIdx === lastCol) return;
      clearCol();
      if (colIdx === null) return;
      lastCol = colIdx;
      var cells = table.querySelectorAll('[data-col="' + colIdx + '"]');
      for (var i = 0; i < cells.length; i++) cells[i].classList.add('col-hover');
    }

    function clearCol() {
      if (lastCol === null) return;
      var cells = table.querySelectorAll('[data-col="' + lastCol + '"]');
      for (var i = 0; i < cells.length; i++) cells[i].classList.remove('col-hover');
      lastCol = null;
    }

    table.addEventListener('mouseover', function(e) {
      var cell = e.target.closest('[data-col]');
      if (cell) {
        highlightCol(cell.getAttribute('data-col'));
      }
    });

    table.addEventListener('mouseleave', function() {
      clearCol();
    });

    table.addEventListener('click', function(e) {
      // Don't navigate if clicking a dropdown
      if (e.target.tagName === 'SELECT' || e.target.tagName === 'OPTION') return;
      var cell = e.target.closest('[data-col]');
      if (cell && cell.getAttribute('data-url')) {
        window.location.href = cell.getAttribute('data-url');
      }
    });
  }

  // Best overall on the left. Re-sorted on every render because the overall
  // depends on the metric selected per category, so changing a dropdown can
  // legitimately reorder the board. Pipelines with no overall score sink to the
  // right rather than being dropped.
  function sortPipelinesByOverall() {
    DATA.pipelines.sort(function(a, b) {
      var av = getOverallScore(a.name);
      var bv = getOverallScore(b.name);
      if (av === null && bv === null) return a.name.localeCompare(b.name);
      if (av === null) return 1;
      if (bv === null) return -1;
      if (bv !== av) return bv - av;
      return a.name.localeCompare(b.name);
    });
  }

  function render() {
    sortPipelinesByOverall();
    renderHead();
    renderBody();
    bindColumnInteractions();
    bindMetricSelector();
  }

  render();
})();
"""
