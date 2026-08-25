"""Evaluation runner for layout attribution metrics.

Matches output result.json files against ground truth test.json files,
computes attribution metrics, and produces a report.

Usage:
    python -m extract_bench.evaluation.metrics.attribution.evaluate \
        --output-dir data/output/llamparse_agentic_tests/ours_agentic \
        --gt-dir data/layout_attribution_50_ro \
        --ioa-threshold 0.3
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from extract_bench.evaluation.layout_adapters import create_layout_adapter_for_result
from extract_bench.evaluation.metrics.attribution.core import (
    AttributionResult,
    compute_attribution_metrics,
    parse_gt_elements,
)
from extract_bench.schemas.layout_detection_output import LayoutPrediction
from extract_bench.schemas.pipeline_io import InferenceResult


@dataclass
class PageEvaluation:
    """Evaluation result for a single page."""

    example_id: str
    category: str
    page_hash: str
    gt_path: str
    output_path: str
    result: AttributionResult
    error: str | None = None


@dataclass
class EvaluationReport:
    """Full evaluation report across all pages."""

    pages: list[PageEvaluation] = field(default_factory=list)

    # Aggregate metrics
    mean_lap: float = 0.0
    mean_lar: float = 0.0
    mean_af1: float = 0.0
    grounding_accuracy: float = 0.0  # pooled across all pages
    grounded_count: int = 0
    total_count: int = 0

    # Per-category breakdown
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)

    # Per-class breakdown (aggregated across all pages)
    per_class_lar: dict[str, float] = field(default_factory=dict)
    per_class_lap: dict[str, float] = field(default_factory=dict)
    per_class_af1: dict[str, float] = field(default_factory=dict)
    per_class_grounding: dict[str, float] = field(default_factory=dict)

    # Counts
    total_pages: int = 0
    total_gt_files: int = 0
    matched_files: int = 0
    unmatched_output_files: int = 0


def find_matching_files(
    output_dir: str | Path,
    gt_dir: str | Path,
) -> list[tuple[str, str, str, str]]:
    """Find output files that have matching ground truth files.

    Matches by hash-based filename: both output and GT use the same hash.

    :param output_dir: Root output directory (contains category subdirs)
    :param gt_dir: Root ground truth directory (contains category subdirs)
    :return: List of (category, page_hash, gt_path, output_path) tuples
    """
    output_dir = Path(output_dir)
    gt_dir = Path(gt_dir)

    # Index all GT files by hash
    gt_index: dict[str, str] = {}  # hash -> full path
    for gt_file in gt_dir.rglob("*.test.json"):
        page_hash = gt_file.stem.replace(".test", "")
        gt_index[page_hash] = str(gt_file)

    # Find matching output files
    matches = []
    for result_file in output_dir.rglob("*.result.json"):
        page_hash = result_file.stem.replace(".result", "")
        if page_hash in gt_index:
            # Determine category from path
            category = result_file.parent.name
            matches.append(
                (
                    category,
                    page_hash,
                    gt_index[page_hash],
                    str(result_file),
                )
            )

    return sorted(matches, key=lambda x: (x[0], x[1]))


def evaluate_single_page(
    gt_path: str,
    output_path: str,
    ioa_threshold: float = 0.3,
) -> tuple[AttributionResult, str | None]:
    """Evaluate attribution metrics for a single page.

    :param gt_path: Path to ground truth .test.json
    :param output_path: Path to output .result.json
    :param ioa_threshold: IoA threshold for spatial matching
    :return: (AttributionResult, error_message or None)
    """
    try:
        with open(gt_path) as f:
            gt_data = json.load(f)
        with open(output_path) as f:
            result_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return AttributionResult(), str(e)

    # Parse GT elements
    test_rules = gt_data.get("test_rules", [])
    gt_elements = parse_gt_elements(test_rules)

    try:
        inference_result = InferenceResult.model_validate(result_data)
    except Exception as exc:
        return AttributionResult(), f"Invalid inference result schema: {exc}"

    adapter = create_layout_adapter_for_result(inference_result)
    layout_output = adapter.to_layout_output(inference_result)

    page_number = _resolve_page_number(layout_output.predictions)
    pred_blocks = adapter.to_attribution_blocks(layout_output, page_number=page_number)

    # Compute metrics
    result = compute_attribution_metrics(gt_elements, pred_blocks, ioa_threshold)
    return result, None


def _resolve_page_number(predictions: list[LayoutPrediction]) -> int:
    pages: list[int] = []
    for prediction in predictions:
        if isinstance(prediction.page, int) and prediction.page > 0:
            pages.append(prediction.page)
    if not pages:
        return 1
    return min(pages)


def run_evaluation(
    output_dir: str | Path,
    gt_dir: str | Path,
    ioa_threshold: float = 0.3,
) -> EvaluationReport:
    """Run full evaluation across all matching files.

    :param output_dir: Root output directory
    :param gt_dir: Root ground truth directory
    :param ioa_threshold: IoA threshold for spatial matching
    :return: EvaluationReport
    """
    matches = find_matching_files(output_dir, gt_dir)

    report = EvaluationReport()
    report.total_gt_files = sum(1 for _ in Path(gt_dir).rglob("*.test.json"))
    report.matched_files = len(matches)

    # Count unmatched output files
    all_output_hashes = set()
    for f in Path(output_dir).rglob("*.result.json"):
        all_output_hashes.add(f.stem.replace(".result", ""))
    gt_hashes = set()
    for f in Path(gt_dir).rglob("*.test.json"):
        gt_hashes.add(f.stem.replace(".test", ""))
    report.unmatched_output_files = len(all_output_hashes - gt_hashes)

    if not matches:
        print("WARNING: No matching files found between output and GT directories.")
        return report

    for category, page_hash, gt_path, output_path in matches:
        result, error = evaluate_single_page(gt_path, output_path, ioa_threshold)

        page_eval = PageEvaluation(
            example_id=f"{category}/{page_hash}",
            category=category,
            page_hash=page_hash,
            gt_path=gt_path,
            output_path=output_path,
            result=result,
            error=error,
        )
        report.pages.append(page_eval)

    # Compute aggregates
    _aggregate_report(report)
    return report


def _aggregate_report(report: EvaluationReport) -> None:
    """Compute aggregate metrics from per-page results."""
    successful = [p for p in report.pages if p.error is None]
    report.total_pages = len(successful)

    if not successful:
        return

    report.mean_lap = sum(p.result.lap for p in successful) / len(successful)
    report.mean_lar = sum(p.result.lar for p in successful) / len(successful)
    report.mean_af1 = sum(p.result.af1 for p in successful) / len(successful)

    # Grounding accuracy: pool element counts across all pages
    report.grounded_count = sum(p.result.grounded_count for p in successful)
    report.total_count = sum(p.result.total_count for p in successful)
    report.grounding_accuracy = report.grounded_count / report.total_count if report.total_count > 0 else 1.0

    # Per-category aggregation
    categories: dict[str, list[PageEvaluation]] = {}
    for p in successful:
        categories.setdefault(p.category, []).append(p)

    for cat, pages in categories.items():
        cat_grounded = sum(p.result.grounded_count for p in pages)
        cat_total = sum(p.result.total_count for p in pages)
        report.per_category[cat] = {
            "mean_lap": sum(p.result.lap for p in pages) / len(pages),
            "mean_lar": sum(p.result.lar for p in pages) / len(pages),
            "mean_af1": sum(p.result.af1 for p in pages) / len(pages),
            "grounding_accuracy": cat_grounded / cat_total if cat_total > 0 else 1.0,
            "grounded": cat_grounded,
            "total_elements": cat_total,
            "count": len(pages),
        }

    # Per-class aggregation (simple average across pages where present)
    lar_num: dict[str, float] = {}
    lar_den: dict[str, int] = {}
    lap_num: dict[str, float] = {}
    lap_den: dict[str, int] = {}
    af1_num: dict[str, float] = {}
    af1_den: dict[str, int] = {}
    for p in successful:
        for cls, lar in p.result.per_class_lar.items():
            lar_num[cls] = lar_num.get(cls, 0.0) + lar
            lar_den[cls] = lar_den.get(cls, 0) + 1
        for cls, lap in p.result.per_class_lap.items():
            lap_num[cls] = lap_num.get(cls, 0.0) + lap
            lap_den[cls] = lap_den.get(cls, 0) + 1
        for cls, af1 in p.result.per_class_af1.items():
            af1_num[cls] = af1_num.get(cls, 0.0) + af1
            af1_den[cls] = af1_den.get(cls, 0) + 1

    for cls in lar_num:
        report.per_class_lar[cls] = lar_num[cls] / lar_den[cls]
    for cls in lap_num:
        report.per_class_lap[cls] = lap_num[cls] / lap_den[cls]
    for cls in af1_num:
        report.per_class_af1[cls] = af1_num[cls] / af1_den[cls]

    # Per-class grounding accuracy (pool element counts across pages)
    ga_class_pass: dict[str, int] = {}
    ga_class_total: dict[str, int] = {}
    for p in successful:
        for cls, count in p.result.per_class_grounded_count.items():
            ga_class_pass[cls] = ga_class_pass.get(cls, 0) + count
        for cls, count in p.result.per_class_total_count.items():
            ga_class_total[cls] = ga_class_total.get(cls, 0) + count

    for cls in ga_class_total:
        report.per_class_grounding[cls] = (
            ga_class_pass.get(cls, 0) / ga_class_total[cls] if ga_class_total[cls] > 0 else 1.0
        )


def format_report(report: EvaluationReport) -> str:
    """Format evaluation report as a readable string.

    :param report: EvaluationReport
    :return: Formatted string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  LAYOUT ATTRIBUTION EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  GT files available:   {report.total_gt_files}")
    lines.append(f"  Matched & evaluated:  {report.matched_files}")
    lines.append(f"  Unmatched outputs:    {report.unmatched_output_files}")
    lines.append("")
    pct = report.grounding_accuracy * 100
    lines.append("-" * 70)
    lines.append(f"  GROUNDING ACCURACY:    {pct:.1f}%  ({report.grounded_count}/{report.total_count} elements)")
    lines.append("-" * 70)
    lines.append("")
    if report.per_class_grounding:
        for cls in sorted(report.per_class_grounding, key=lambda c: report.per_class_grounding[c]):
            acc = report.per_class_grounding[cls]
            # Recover counts for display
            ga_pass = sum(p.result.per_class_grounded_count.get(cls, 0) for p in report.pages if p.error is None)
            ga_total = sum(p.result.per_class_total_count.get(cls, 0) for p in report.pages if p.error is None)
            lines.append(f"    {cls:20s}  {acc * 100:5.1f}%  ({ga_pass}/{ga_total})")
        lines.append("")
    lines.append("-" * 70)
    lines.append("  DETAILED METRICS")
    lines.append("-" * 70)
    lines.append(f"  LAP (precision):       {report.mean_lap:.4f}")
    lines.append(f"  LAR (recall):          {report.mean_lar:.4f}")
    lines.append(f"  AF1 (f1-score):        {report.mean_af1:.4f}")
    lines.append("")

    if report.per_category:
        lines.append("-" * 70)
        lines.append("  PER-CATEGORY BREAKDOWN")
        lines.append("-" * 70)
        for cat, metrics in sorted(report.per_category.items()):
            n = metrics["count"]
            ga_pct = metrics["grounding_accuracy"] * 100
            lines.append(
                f"  {cat} (n={n}):  Grounding={ga_pct:.1f}% ({metrics['grounded']}/{metrics['total_elements']})"
            )
            lines.append(
                f"    LAP={metrics['mean_lap']:.4f}  LAR={metrics['mean_lar']:.4f}  AF1={metrics['mean_af1']:.4f}"
            )
        lines.append("")

    if report.per_class_lar:
        lines.append("-" * 70)
        lines.append("  PER-CLASS LAR")
        lines.append("-" * 70)
        for cls, lar in sorted(report.per_class_lar.items()):
            lines.append(f"  {cls:20s}: {lar:.4f}")
        lines.append("")

    if report.per_class_lap:
        lines.append("-" * 70)
        lines.append("  PER-CLASS LAP")
        lines.append("-" * 70)
        for cls, lap in sorted(report.per_class_lap.items()):
            lines.append(f"  {cls:20s}: {lap:.4f}")
        lines.append("")

    if report.per_class_af1:
        lines.append("-" * 70)
        lines.append("  PER-CLASS AF1")
        lines.append("-" * 70)
        for cls, af1 in sorted(report.per_class_af1.items()):
            lines.append(f"  {cls:20s}: {af1:.4f}")
        lines.append("")

    if report.pages:
        lines.append("-" * 70)
        lines.append("  PER-PAGE DETAILS")
        lines.append("-" * 70)
        for p in report.pages:
            r = p.result
            status = "ERROR" if p.error else "OK"
            lines.append(f"  [{status}] {p.example_id}")
            if p.error:
                lines.append(f"    Error: {p.error}")
            else:
                ga_pct = r.grounding_accuracy * 100
                lines.append(
                    f"    Grounding: {ga_pct:.1f}% ({r.grounded_count}/{r.total_count})  "
                    f"AF1={r.af1:.4f}  LAP={r.lap:.4f}  LAR={r.lar:.4f}"
                )
                lines.append(f"    Unmatched: GT={r.unmatched_gt_elements} Pred={r.unmatched_pred_blocks}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():  # type: ignore[no-untyped-def]
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Layout Attribution Evaluation")
    parser.add_argument("--output-dir", required=True, help="Output results directory")
    parser.add_argument("--gt-dir", required=True, help="Ground truth directory")
    parser.add_argument("--ioa-threshold", type=float, default=0.3, help="IoA threshold")
    parser.add_argument("--json-output", help="Optional: save results as JSON")
    args = parser.parse_args()

    report = run_evaluation(args.output_dir, args.gt_dir, args.ioa_threshold)
    print(format_report(report))

    if args.json_output:
        # Save JSON report
        json_data = {
            "aggregate": {
                "grounding_accuracy": report.grounding_accuracy,
                "grounded_count": report.grounded_count,
                "total_count": report.total_count,
                "mean_lap": report.mean_lap,
                "mean_lar": report.mean_lar,
                "mean_af1": report.mean_af1,
                "total_pages": report.total_pages,
                "matched_files": report.matched_files,
            },
            "per_category": report.per_category,
            "per_class_grounding": report.per_class_grounding,
            "per_class_lar": report.per_class_lar,
            "per_class_lap": report.per_class_lap,
            "per_class_af1": report.per_class_af1,
            "pages": [
                {
                    "example_id": p.example_id,
                    "category": p.category,
                    "error": p.error,
                    "grounding_accuracy": p.result.grounding_accuracy,
                    "grounded_count": p.result.grounded_count,
                    "total_count": p.result.total_count,
                    "lap": p.result.lap,
                    "lar": p.result.lar,
                    "af1": p.result.af1,
                    "num_gt_elements": p.result.num_gt_elements,
                    "num_pred_blocks": p.result.num_pred_blocks,
                    "unmatched_gt_elements": p.result.unmatched_gt_elements,
                    "unmatched_pred_blocks": p.result.unmatched_pred_blocks,
                    "per_class_grounding": p.result.per_class_grounding,
                    "per_class_lar": p.result.per_class_lar,
                    "per_class_lap": p.result.per_class_lap,
                    "per_class_af1": p.result.per_class_af1,
                }
                for p in report.pages
            ],
        }
        with open(args.json_output, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nJSON report saved to: {args.json_output}")


if __name__ == "__main__":
    main()
