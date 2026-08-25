"""Detailed HTML report generation for evaluation results.

Generates a self-contained, interactive HTML evaluation report with:
- Summary cards with key metrics
- Aggregate metrics panel with color-coded score bars
- Collapsible aggregate stats (latency, cost, tokens)
- Interactive examples table with metric selector, filters, sort, search, pagination
- Detail panel with per-example metrics, rule results, PDF viewer, and stats

This module provides the fancy interactive report (_evaluation_report_detailed.html).
It should be run as a separate step after evaluation to explore results in detail.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import bleach
import markdown2

from extract_bench.analysis.grounding_annotations import (
    load_gt_grounding_annotations,
    load_pred_grounding_citations,
)
from extract_bench.analysis.metric_aggregates import to_agg_metric_records
from extract_bench.analysis.metric_definitions import (
    EXTRACT_DEFAULT_METRIC,
    display_name,
    order_metrics,
    tooltip_dict,
)
from extract_bench.analysis.report_static_assets import detailed_report_script, detailed_report_style
from extract_bench.schemas.evaluation import EvaluationSummary

# Per-doc table count metrics are bookkeeping, not quality scores -- exclude
# them from the detailed report's aggregate / tag metric panels.
_HIDDEN_TABLE_COUNT_METRICS = {
    "tables_expected",
    "tables_actual",
    "tables_paired",
    "tables_unmatched_expected",
    "tables_unmatched_pred",
    "tables_unparseable_pred",
}


def _render_markdown_to_html(md_text: str) -> str:
    """Render markdown to sanitised HTML, preserving HTML tables with colspan/rowspan."""
    if not md_text:
        return ""

    # Extract HTML tables before markdown2 processing (it can mangle colspan/rowspan)
    html_table_pattern = r"<table[^>]*>.*?</table>"
    processed_md = md_text
    table_placeholders: dict[str, str] = {}
    matches = list(re.finditer(html_table_pattern, md_text, re.DOTALL | re.IGNORECASE))
    for i, match in enumerate(reversed(matches)):
        placeholder = f"<!--HTMLTABLE_{len(matches) - 1 - i}-->"
        table_placeholders[placeholder] = match.group(0)
        s, e = match.span()
        processed_md = processed_md[:s] + placeholder + processed_md[e:]

    rendered = markdown2.markdown(processed_md, extras=["tables", "fenced-code-blocks", "break-on-newline"])

    # Restore original HTML tables
    for placeholder, table_html in table_placeholders.items():
        rendered = rendered.replace(placeholder, table_html)

    allowed_tags = bleach.sanitizer.ALLOWED_TAGS | {
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "caption",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "br",
        "hr",
        "pre",
        "code",
        "img",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "div",
        "span",
        "sup",
        "sub",
    }
    allowed_attrs = {
        **bleach.sanitizer.ALLOWED_ATTRIBUTES,
        "th": ["colspan", "rowspan", "scope"],
        "td": ["colspan", "rowspan"],
        "img": ["src", "alt", "width", "height"],
        "code": ["class"],
        "pre": ["class"],
    }
    return str(bleach.clean(rendered, tags=allowed_tags, attributes=allowed_attrs))


_GROUNDING_RULE_METRIC_PRIORITY = (
    "extract_evidence_bbox_IOU_pass_rate",
    "extract_evidence_bbox_covered_pass_rate",
    "extract_localization_pass_rate",
    "extract_attribution_pass_rate",
    "parse_field_grounding_pass_rate",
)


def _lookup_keys_for_file(file_path: Path, root: Path | None, *, suffix: str) -> list[str]:
    """Return lookup keys for a sidecar file: ``group/stem`` then ``stem``."""
    stem = file_path.name[: -len(suffix)] if file_path.name.endswith(suffix) else file_path.stem
    keys = [stem]
    if root is not None:
        try:
            rel = file_path.parent.relative_to(root)
            if rel.parts:
                keys.insert(0, "/".join((*rel.parts, stem)))
        except ValueError:
            pass
    return keys


def _store_map_entry(store: dict[str, Any], keys: list[str], value: Any) -> None:
    for key in keys:
        store[key] = value


def _load_gt_grounding_annotations(test_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Load GT evidence boxes from ``_field_rules`` and legacy ``test_rules``."""
    return load_gt_grounding_annotations(test_data)


def _load_pred_grounding_citations(result_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Load predicted field citations from an inference result file."""
    return load_pred_grounding_citations(result_data)


# Which rule_results key backs each metric's numerator. The v0.2 evidence
# adapter hangs one rule_results list off several metrics, so without this the
# generic fallback below would print the same value-pass count beside all of
# them. Longest suffix first: matching is by ``endswith``.
_METRIC_RULE_PASS_KEYS: tuple[tuple[str, str], ...] = (
    ("page_covered_pass_rate", "page_qualified"),
    ("bbox_covered_pass_rate", "bbox_covered_pass"),
    ("classification_pass_rate", "cls_pass"),
    ("localization_pass_rate", "loc_pass"),
    ("attribution_pass_rate", "attr_pass"),
    ("element_pass_rate", "element_pass"),
    ("value_pass_rate", "value_pass"),
    ("page_pass_rate", "page_pass"),
)

# Metrics whose numerator is a summed score rather than a count of passing
# rules. An "x/y rules" chip beside them would be a category error.
_NON_COUNTING_METRIC_SUFFIXES: tuple[str, ...] = (
    "bbox_IOU_pass_rate",
    "bbox_IOU_alignment",
    "bbox_coverage",
)

# Where a metric's own denominator count lives in its metadata, keyed by the
# ``denominator`` label the evaluator records. Several evidence metrics are
# conditional: ``page_covered_pass_rate`` is scored over page-qualified rules
# and the bbox metrics over GT-bbox-bearing rules, both proper subsets of
# ``rule_results``. Anything unlisted is scored over the full rule list.
_DENOMINATOR_META_KEYS: dict[str, str] = {
    "page_qualified_rules": "covered_total",
    "gt_bbox_bearing_rules": "bbox_gt_total",
}


def _is_non_counting_metric(metric_name: str) -> bool:
    """True when an ``x/y rules`` tally would misrepresent the metric."""
    return metric_name.endswith(_NON_COUNTING_METRIC_SUFFIXES)


def _is_count(value: Any) -> bool:
    """True for a plain non-negative int. ``bool`` is an ``int`` — exclude it."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _rule_tally_from_metadata(metadata: dict[str, Any]) -> tuple[int, int] | None:
    """Return ``(passed, total)`` from a metric's own tp/denominator bookkeeping.

    The evaluator already records the numerator (``tp``) and which denominator
    it scored against, so read those rather than recounting ``rule_results``.
    Re-deriving eligibility here would mean mirroring the adapter's gating —
    and for the bbox metrics that is not even possible, because whether a
    leaf's GT carried a bbox is not exposed on the per-rule dicts.

    Returns ``None`` when the bookkeeping is absent or inconsistent, leaving
    the caller to fall back to counting rules.
    """
    tp = metadata.get("tp")
    if not _is_count(tp):
        return None
    total_key = _DENOMINATOR_META_KEYS.get(str(metadata.get("denominator")), "total")
    total = metadata.get(total_key)
    if not _is_count(total) or total < tp:
        return None
    return tp, total


def _coerce_rule_pass(rule: dict[str, Any], metric_name: str | None = None) -> bool | None:
    """Return pass/fail for a rule_results entry across parse and extract schemas.

    Extract evidence rules often expose ``value_pass`` / ``page_pass`` instead of
    ``passed``. When ``metric_name`` is given, the key that metric actually
    counts wins, so each displayed tally agrees with its own metric.
    """
    if metric_name:
        for suffix, key in _METRIC_RULE_PASS_KEYS:
            if metric_name.endswith(suffix):
                value = rule.get(key)
                return value if isinstance(value, bool) else None

    if "passed" in rule:
        return bool(rule["passed"])
    for key in ("value_pass", "element_pass", "page_pass", "bbox_covered_pass", "loc_pass", "attr_pass"):
        if key in rule and isinstance(rule[key], bool):
            return rule[key]
    bip = rule.get("bbox_iou_pass")
    # ``bbox_iou_pass`` is the value-gated IoU float on v0.2 evidence rules, so
    # only a genuine bool or int 0/1 is a pass flag. ``isinstance(True, int)`` is
    # True, hence the bool check first; ``0.0 in (0, 1)`` is True, hence type().
    if isinstance(bip, bool):
        return bip
    if type(bip) is int and bip in (0, 1):
        return bool(bip)
    return None


def _normalize_rule_result_for_table(rule: dict[str, Any], metric_name: str | None = None) -> dict[str, Any]:
    """Normalize parse/extract rule_results for the detailed-report rules table."""
    passed = _coerce_rule_pass(rule, metric_name)
    return {
        "type": str(rule.get("type") or rule.get("mode") or ""),
        "passed": bool(passed) if passed is not None else False,
        "id": str(rule.get("id") or rule.get("field_path") or rule.get("path") or rule.get("name") or ""),
        "message": str(rule.get("message") or rule.get("explanation") or rule.get("reason") or ""),
    }


def _normalize_grounding_rule(rule: dict[str, Any]) -> dict[str, Any]:
    """Normalize rule-result dict keys to camelCase for the HTML report."""
    field_path = rule.get("field_path") or rule.get("path") or rule.get("id") or ""
    passed = _coerce_rule_pass(rule)
    normalized: dict[str, Any] = {
        "fieldPath": str(field_path),
        "passed": bool(passed) if passed is not None else False,
    }

    key_map = {
        "type": "type",
        "id": "id",
        "message": "message",
        "verified": "verified",
        "value_pass": "valuePass",
        "page_pass": "pagePass",
        "page_qualified": "pageQualified",
        "bbox_qualified": "bboxQualified",
        "bbox_iou": "bboxIou",
        "bbox_iou_pass": "bboxIouPass",
        "bbox_covered_pass": "bboxCoveredPass",
        "loc_pass": "locPass",
        "attr_pass": "attrPass",
        "element_pass": "elementPass",
        "iou": "iou",
        "mode": "mode",
        "reason": "reason",
        "localization_reason": "localizationReason",
        "matched_pred_field_path": "matchedPredFieldPath",
    }
    for src, dst in key_map.items():
        if src in rule:
            normalized[dst] = rule[src]

    if "matched_pred_bboxes" in rule and isinstance(rule["matched_pred_bboxes"], list):
        normalized["matchedPredBboxes"] = [
            [float(v) for v in box[:4]]
            for box in rule["matched_pred_bboxes"]
            if isinstance(box, list) and len(box) >= 4
        ]
    return normalized


def _select_grounding_rules(metric_values: list[Any]) -> list[dict[str, Any]]:
    """Pick the richest grounding rule-results list from evaluation metrics."""
    by_name: dict[str, Any] = {mv.metric_name: mv for mv in metric_values}

    for metric_name in _GROUNDING_RULE_METRIC_PRIORITY:
        metric = by_name.get(metric_name)
        if metric and metric.metadata and isinstance(metric.metadata.get("rule_results"), list):
            rules = metric.metadata["rule_results"]
            if rules and isinstance(rules[0], dict) and ("field_path" in rules[0] or "path" in rules[0]):
                return [_normalize_grounding_rule(r) for r in rules if isinstance(r, dict)]

    for metric in metric_values:
        metadata = metric.metadata or {}
        rules = metadata.get("rule_results")
        if not isinstance(rules, list) or not rules:
            continue
        if isinstance(rules[0], dict) and ("field_path" in rules[0] or "path" in rules[0]):
            return [_normalize_grounding_rule(r) for r in rules if isinstance(r, dict)]
    return []


def _render_output_html(text: str | None, output_format: str) -> str:
    """Return rendered HTML for markdown output; JSON is shown raw in the report."""
    if not text or output_format != "markdown":
        return ""
    return _render_markdown_to_html(text)


def _build_data_blob(
    summary: EvaluationSummary,
    output_dir: Path | None = None,
    test_cases_dir: Path | None = None,
    pdf_base_url: str = "",
) -> dict[str, Any]:
    """Build the JSON data blob that powers the client-side rendering."""

    # --- load predicted/expected output from files ---
    predicted_map: dict[str, str] = {}
    expected_map: dict[str, str] = {}
    predicted_format_map: dict[str, str] = {}
    expected_format_map: dict[str, str] = {}
    job_id_map: dict[str, str] = {}
    parse_job_logs_url_map: dict[str, str] = {}
    parse_job_logs_local_path_map: dict[str, str] = {}
    parse_job_logs_html_path_map: dict[str, str] = {}
    pred_citations_map: dict[str, list[dict[str, Any]]] = {}
    gt_grounding_map: dict[str, list[dict[str, Any]]] = {}

    if output_dir and output_dir.exists():
        for result_file in output_dir.rglob("*.result.json"):
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
                lookup_keys = _lookup_keys_for_file(result_file, output_dir, suffix=".result.json")
                output = data.get("output") or {}
                raw_output = data.get("raw_output") or {}
                # Parse output: markdown field.
                # Rare on ExtractBench (JSON extract is the product path), but kept
                # because inherited parse pipelines remain runnable for two-stage
                # extract and grounding cross-eval.
                if isinstance(output, dict) and output.get("markdown"):
                    _store_map_entry(predicted_map, lookup_keys, output["markdown"])
                    _store_map_entry(predicted_format_map, lookup_keys, "markdown")
                # Extract output: extracted_data field
                elif isinstance(output, dict) and output.get("extracted_data"):
                    _store_map_entry(
                        predicted_map,
                        lookup_keys,
                        json.dumps(output["extracted_data"], indent=2, ensure_ascii=False),
                    )
                    _store_map_entry(predicted_format_map, lookup_keys, "json")
                citations = _load_pred_grounding_citations(data)
                if citations:
                    _store_map_entry(pred_citations_map, lookup_keys, citations)
                # Job ID from output (e.g. LlamaParse)
                if isinstance(output, dict) and output.get("job_id"):
                    for key in lookup_keys:
                        job_id_map[key] = output["job_id"]
                if isinstance(raw_output, dict):
                    job_logs_url = raw_output.get("job_logs_url")
                    if not isinstance(job_logs_url, str) or not job_logs_url:
                        job_logs = raw_output.get("job_logs")
                        if isinstance(job_logs, dict):
                            nested_url = job_logs.get("url")
                            if isinstance(nested_url, str) and nested_url:
                                job_logs_url = nested_url
                    if isinstance(job_logs_url, str) and job_logs_url:
                        for key in lookup_keys:
                            parse_job_logs_url_map[key] = job_logs_url

                    job_logs_local = raw_output.get("job_logs_local_path")
                    if isinstance(job_logs_local, str) and job_logs_local:
                        for key in lookup_keys:
                            parse_job_logs_local_path_map[key] = job_logs_local

                    job_logs_html = raw_output.get("job_logs_html_local_path")
                    if isinstance(job_logs_html, str) and job_logs_html:
                        for key in lookup_keys:
                            parse_job_logs_html_path_map[key] = job_logs_html
            except Exception:
                pass

    if test_cases_dir and test_cases_dir.exists():
        for test_file in test_cases_dir.rglob("*.test.json"):
            try:
                data = json.loads(test_file.read_text(encoding="utf-8"))
                lookup_keys = _lookup_keys_for_file(test_file, test_cases_dir, suffix=".test.json")
                if data.get("expected_markdown"):
                    _store_map_entry(expected_map, lookup_keys, data["expected_markdown"])
                    _store_map_entry(expected_format_map, lookup_keys, "markdown")
                elif data.get("expected_output"):
                    _store_map_entry(
                        expected_map,
                        lookup_keys,
                        json.dumps(data["expected_output"], indent=2, ensure_ascii=False),
                    )
                    _store_map_entry(expected_format_map, lookup_keys, "json")
                gt_annotations = _load_gt_grounding_annotations(data)
                if gt_annotations:
                    _store_map_entry(gt_grounding_map, lookup_keys, gt_annotations)
            except Exception:
                pass

    # --- aggregate metrics (group avg/min/max) ---
    agg_metrics = to_agg_metric_records(
        summary.aggregate_metrics,
        exclude=_HIDDEN_TABLE_COUNT_METRICS,
        sort_by_avg=True,
    )
    # Same headline-first ordering as the Metrics by Tag table rows.
    ordered_metric_names = order_metrics([str(r["name"]) for r in agg_metrics], separator=False)
    if EXTRACT_DEFAULT_METRIC in ordered_metric_names:
        default_metric = EXTRACT_DEFAULT_METRIC
    elif ordered_metric_names:
        default_metric = ordered_metric_names[0]
    else:
        default_metric = ""

    # --- aggregate stats ---
    agg_stats = []
    for stat_name, agg in sorted(summary.aggregate_stats.items()):
        agg_stats.append(
            {
                "name": stat_name,
                "displayName": stat_name.replace("_", " ").title(),
                "unit": agg.get("unit", ""),
                "avg": agg.get("avg", 0),
                "min": agg.get("min", 0),
                "max": agg.get("max", 0),
                "p50": agg.get("p50", 0),
                "p95": agg.get("p95", 0),
                "p99": agg.get("p99", 0),
                "total": agg.get("total", 0),
                "count": agg.get("count", 0),
            }
        )

    # --- metric names lookup ---
    metric_names_map: dict[str, str] = {}
    for rec in agg_metrics:
        metric_names_map[str(rec["name"])] = str(rec["displayName"])

    # --- collect all tags ---
    all_tags: set[str] = set()
    for result in summary.per_example_results:
        all_tags.update(result.tags)

    # --- per-example data ---
    examples = []
    for result in summary.per_example_results:
        metrics_dict: dict[str, float] = {}
        rule_details: dict[str, dict[str, int]] = {}
        rule_results_map: dict[str, list[dict[str, Any]]] = {}
        metric_details_map: dict[str, list[str]] = {}

        for mv in result.metrics:
            if mv.metric_name in _HIDDEN_TABLE_COUNT_METRICS:
                continue
            metrics_dict[mv.metric_name] = mv.value
            # Add to metric_names_map if not already there
            if mv.metric_name not in metric_names_map:
                metric_names_map[mv.metric_name] = display_name(mv.metric_name)

            # Collect human-readable detail strings
            if mv.details:
                metric_details_map[mv.metric_name] = mv.details

            # Extract rule details from metadata (parse + extract schemas).
            # Scored per metric: several metrics share one rule_results list, and
            # each counts a different key over a different denominator. Prefer the
            # metric's own tp/denominator so the chip cannot disagree with the
            # value beside it; count rules only when that bookkeeping is missing.
            if "rule_results" in mv.metadata and not _is_non_counting_metric(mv.metric_name):
                raw_rules = [r for r in mv.metadata["rule_results"] if isinstance(r, dict)]
                tally = _rule_tally_from_metadata(mv.metadata)
                if tally is None:
                    passed = sum(1 for r in raw_rules if _coerce_rule_pass(r, mv.metric_name) is True)
                    total = len(raw_rules)
                else:
                    passed, total = tally
                rule_details[mv.metric_name] = {"passed": passed, "total": total}
                rule_results_map[mv.metric_name] = [
                    _normalize_rule_result_for_table(r, mv.metric_name) for r in raw_rules
                ]

        stats_dict: dict[str, float] = {}
        for s in result.stats:
            stats_dict[s.name] = s.value

        basename = result.test_id.rsplit("/", 1)[-1]
        grounding_rules = _select_grounding_rules(result.metrics)
        gt_annotations = gt_grounding_map.get(result.test_id) or gt_grounding_map.get(basename) or []
        pred_citations = pred_citations_map.get(result.test_id) or pred_citations_map.get(basename) or []
        pred_text = predicted_map.get(result.test_id) or predicted_map.get(basename) or ""
        exp_text = expected_map.get(result.test_id) or expected_map.get(basename) or ""
        pred_format = predicted_format_map.get(result.test_id) or predicted_format_map.get(basename) or ""
        exp_format = expected_format_map.get(result.test_id) or expected_format_map.get(basename) or ""

        examples.append(
            {
                "id": result.test_id,
                "success": result.success,
                "error": result.error,
                "tags": result.tags,
                "productType": result.product_type,
                "jobId": (result.job_id or job_id_map.get(result.test_id) or job_id_map.get(basename, "")),
                "parseJobId": result.parse_job_id or "",
                "parseJobLogsUrl": (
                    parse_job_logs_url_map.get(result.test_id) or parse_job_logs_url_map.get(basename, "")
                ),
                "parseJobLogsLocalPath": (
                    parse_job_logs_local_path_map.get(result.test_id) or parse_job_logs_local_path_map.get(basename, "")
                ),
                "parseJobLogsHtmlPath": (
                    parse_job_logs_html_path_map.get(result.test_id) or parse_job_logs_html_path_map.get(basename, "")
                ),
                "metrics": metrics_dict,
                "stats": stats_dict,
                "ruleDetails": rule_details,
                "ruleResults": rule_results_map,
                "metricDetails": metric_details_map,
                "grounding": {
                    "gt": gt_annotations,
                    "pred": pred_citations,
                    "rules": grounding_rules,
                },
                "predictedOutput": pred_text,
                "expectedOutput": exp_text,
                "predictedOutputFormat": pred_format,
                "expectedOutputFormat": exp_format,
                "predictedHtml": _render_output_html(pred_text, pred_format),
                "expectedHtml": _render_output_html(exp_text, exp_format),
            }
        )

    completed_at_str = ""
    if summary.completed_at is not None:
        completed_at_str = summary.completed_at.isoformat()

    return {
        "summary": {
            "total": summary.total_examples,
            "successful": summary.successful,
            "failed": summary.failed,
            "skipped": summary.skipped,
            "completedAt": completed_at_str,
        },
        "aggMetrics": agg_metrics,
        "aggStats": agg_stats,
        "metricNames": metric_names_map,
        "metricTooltips": tooltip_dict(),
        "defaultMetric": default_metric,
        "tags": sorted(all_tags),
        "tagMetrics": {
            tag: {
                "exampleCount": int(metrics.get("example_count", 0)),
                "metrics": to_agg_metric_records(
                    metrics,
                    exclude=_HIDDEN_TABLE_COUNT_METRICS,
                ),
            }
            for tag, metrics in summary.tag_metrics.items()
        },
        "examples": examples,
        "pdfBaseUrl": pdf_base_url,
    }


# ---------------------------------------------------------------------------
# HTML template parts
# ---------------------------------------------------------------------------

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
"""

_HTML_BODY = """\
</head>
<body>
<div class="report-container">
    <div class="report-header">
        <h1>Evaluation Report</h1>
        <div class="subtitle" id="report-subtitle"></div>
    </div>
    <div class="summary-row" id="summary-cards"></div>
    <div id="agg-metrics"></div>
    <div class="stats-section" id="agg-stats"></div>
    <div class="tag-metrics-section" id="tag-metrics"></div>
    <div id="examples-section">
        <h2 class="section-title">Examples</h2>
        <div class="controls-bar" id="controls"></div>
        <div class="tag-filters" id="tag-filters"></div>
        <div class="results-count" id="results-count"></div>
        <div class="table-wrap">
            <table class="examples-table">
                <thead>
                    <tr>
                        <th class="col-status"></th>
                        <th class="col-id">Test ID</th>
                        <th class="col-score">Score</th>
                        <th class="col-tags">Tags</th>
                    </tr>
                </thead>
                <tbody id="examples-tbody"></tbody>
            </table>
        </div>
        <div class="pagination" id="pagination"></div>
    </div>
</div>
"""


def generate_detailed_html_report(
    summary: EvaluationSummary,
    report_dir: Path,
    output_dir: Path | None = None,
    test_cases_dir: Path | None = None,
    pdf_base_url: str | None = None,
    pipeline_name: str | None = None,
    group: str | None = None,
) -> Path:
    """Export evaluation summary to an interactive HTML report.

    Args:
        summary: Evaluation summary data.
        report_dir: Directory to write the HTML report.
        output_dir: Directory containing inference result files (for predicted output).
        test_cases_dir: Directory containing test case files (for expected output).
        pdf_base_url: Base URL for PDF files. If not provided but test_cases_dir is set,
            falls back to the local filesystem path.
        pipeline_name: Name of the pipeline (e.g., 'llamaextract_agentic').
        group: Evaluation category/group (e.g., 'text_content').
    """
    html_path = report_dir / "_evaluation_report_detailed.html"

    # Resolve PDF base URL: explicit > relative path from report to PDF directory
    resolved_pdf_base_url = ""
    if pdf_base_url:
        resolved_pdf_base_url = pdf_base_url.rstrip("/")
    elif test_cases_dir is not None and test_cases_dir.exists():
        import os

        # JSONL datasets store PDFs under a pdfs/ subdirectory, while sidecar
        # datasets store them directly alongside test.json files. Use the pdfs/
        # subdirectory if it exists so that {baseUrl}/{testId}.pdf resolves correctly.
        pdf_root = test_cases_dir.resolve()
        if (pdf_root / "pdfs").is_dir():
            pdf_root = pdf_root / "pdfs"
        resolved_pdf_base_url = os.path.relpath(pdf_root, report_dir.resolve())

    # Load pipeline metadata if available
    metadata: dict[str, Any] = {}
    if output_dir:
        # Try pipeline output root (one level up from group report dir)
        for candidate in [output_dir / "_metadata.json", output_dir.parent / "_metadata.json"]:
            if candidate.exists():
                try:
                    metadata = json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:
                    pass
                break

    # Extract pipeline info
    pipeline_info = metadata.get("pipeline", {})
    resolved_pipeline_name = pipeline_name or pipeline_info.get("pipeline_name", "")
    provider_name = pipeline_info.get("provider_name", "")
    product_type = pipeline_info.get("product_type", "")
    pipeline_config = pipeline_info.get("config", {})

    data_blob = _build_data_blob(
        summary,
        output_dir=output_dir,
        test_cases_dir=test_cases_dir,
        pdf_base_url=resolved_pdf_base_url,
    )

    # Add run info to data blob
    data_blob["runInfo"] = {
        "pipelineName": resolved_pipeline_name,
        "providerName": provider_name,
        "productType": product_type,
        "category": group or "",
        "config": pipeline_config,
    }

    # Serialize and escape for safe embedding inside <script>
    data_json = json.dumps(data_blob, default=str, ensure_ascii=False)
    # Prevent premature script close
    data_json = data_json.replace("</script>", "<\\/script>")
    # Prevent HTML comment issues
    data_json = data_json.replace("<!--", "<\\!--")

    completed_str = ""
    if summary.completed_at is not None:
        completed_str = summary.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build full HTML by concatenation (no f-string over JS/CSS to avoid brace issues)
    parts: list[str] = []
    parts.append(_HTML_HEAD)
    parts.append(detailed_report_style())
    parts.append(_HTML_BODY)

    # Title and subtitle script — show pipeline, provider, category
    title_parts = []
    if resolved_pipeline_name:
        title_parts.append(resolved_pipeline_name.replace("_", " ").title())
    if group:
        title_parts.append(group.replace("_", " ").title())
    title_text = " — ".join(title_parts) if title_parts else "Evaluation Report"

    subtitle_parts = []
    if provider_name:
        subtitle_parts.append("Provider: " + provider_name)
    if product_type:
        subtitle_parts.append("Product: " + product_type)
    if completed_str:
        subtitle_parts.append("Generated: " + completed_str)
    subtitle_text = (
        "  |  ".join(subtitle_parts) if subtitle_parts else ("Generated: " + completed_str if completed_str else "")
    )

    parts.append("<script>")
    parts.append('document.querySelector(".report-header h1").textContent = ' + json.dumps(title_text) + ";")
    parts.append('document.getElementById("report-subtitle").textContent = ' + json.dumps(subtitle_text) + ";")
    parts.append("</script>\n")

    # Data blob
    parts.append("<script>\nconst DATA = ")
    parts.append(data_json)
    parts.append(";\n</script>\n")

    parts.append(detailed_report_script())
    parts.append("\n</body>\n</html>\n")

    html_path.write_text("".join(parts), encoding="utf-8")
    return html_path
