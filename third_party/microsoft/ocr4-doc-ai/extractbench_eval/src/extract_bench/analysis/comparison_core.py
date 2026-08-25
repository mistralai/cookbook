"""
Lightweight comparison module for evaluating two pipeline results.

Avoids Pydantic and other heavy extract_bench deps so it can run in the
dashboard deployment. Local analysis helpers without those deps are fine.
"""

import json
import re
from pathlib import Path
from typing import Any

from extract_bench.analysis.grounding_annotations import (
    load_gt_grounding_annotations,
    load_pred_grounding_citations,
)

# Metric mapping for different product types
COMPARISON_METRIC_MAP: dict[str, str] = {
    "extract": "accuracy",
    "parse": "normalized_text_score",
    "layout_detection": "mAP@[.50:.95]",
}


def load_evaluation_report(pipeline_path: Path) -> dict | None:
    """Load evaluation report JSON from a pipeline directory."""
    report_file = pipeline_path / "_evaluation_report.json"
    if not report_file.exists():
        return None
    try:
        with open(report_file) as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception:
        return None


def inference_result_candidate_paths(pipeline_path: Path, test_id: str) -> list[Path]:
    """Return likely on-disk locations for a test_id's ``.result.json``.

    ``test_id`` is often ``{group}/{stem}``. Comparisons may point at the pipeline
    root (``.../llamaextract_agentic``) or at a group directory (``.../short``), so
    both ``{group}/{stem}.result.json`` and ``{stem}.result.json`` are candidates.
    """
    basename = test_id.rsplit("/", 1)[-1]
    candidates = [
        pipeline_path / f"{test_id}.result.json",
        pipeline_path / f"{basename}.result.json",
    ]
    # Preserve order, drop duplicates (Path equality is enough here).
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def normalize_field_citations_for_overlay(result_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return camelCase field citations suitable for PDF bbox overlays.

    Drops page-only citations (no bbox) and reference text to keep comparison
    HTML payloads small while preserving everything the overlay needs.
    """
    return load_pred_grounding_citations(
        result_data,
        require_bbox=True,
        include_reference_text=False,
    )


def normalize_gt_evidence_for_overlay(test_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return GT evidence boxes from a ``.test.json`` payload for PDF overlays."""
    return load_gt_grounding_annotations(
        test_data,
        include_verified=False,
        include_quote=False,
    )


def load_test_json_for_overlay(test_cases_dir: Path | None, test_id: str) -> dict[str, Any] | None:
    """Load the ``.test.json`` sidecar for a comparison test_id, if present."""
    if test_cases_dir is None:
        return None
    basename = test_id.rsplit("/", 1)[-1]
    candidates = [
        test_cases_dir / f"{test_id}.test.json",
        test_cases_dir / f"{basename}.test.json",
    ]
    parts = test_id.split("/")
    if len(parts) == 2:
        group, stem = parts
        candidates.insert(0, test_cases_dir / group / f"{stem}.test.json")
    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else None
            except Exception:
                return None
    return None


def load_inference_result(pipeline_path: Path, test_id: str) -> dict | None:
    """Load inference result for a specific test_id."""
    basename = test_id.rsplit("/", 1)[-1]
    for result_path in inference_result_candidate_paths(pipeline_path, test_id):
        if result_path.exists():
            try:
                with open(result_path) as f:
                    return json.load(f)  # type: ignore[no-any-return]
            except Exception:
                return None

    # Fallback: search by basename only (slashes in test_id break rglob globs).
    for result_file in pipeline_path.rglob(f"{basename}.result.json"):
        try:
            with open(result_file) as f:
                return json.load(f)  # type: ignore[no-any-return]
        except Exception:
            return None
    return None


def get_metric_value(metrics_list: list, metric_name: str) -> float | None:
    """Extract a specific metric value from a metrics list."""
    for metric in metrics_list:
        if metric.get("metric_name") == metric_name:
            return metric.get("value")  # type: ignore[no-any-return]
    return None


def get_directory_suffix(pipeline_dir: Path) -> str:
    """
    Extract a distinguishing suffix from the pipeline directory path.

    Looks for run IDs, dates, or other identifying info in parent directories.
    """
    parent_name = pipeline_dir.parent.name

    # Try to extract a run ID pattern (e.g., run-21391181794)
    run_id_match = re.search(r"run-(\d+)", parent_name)
    if run_id_match:
        return f"run-{run_id_match.group(1)}"

    # Try to extract a date pattern (e.g., 2025-01-27)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", parent_name)
    if date_match:
        return date_match.group(1)

    # Fall back to the parent directory name
    if parent_name and parent_name != "output":
        return parent_name

    # Last resort: use the full parent path's last 2 components
    parts = pipeline_dir.parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])

    return str(pipeline_dir)


def get_predictions_from_inference(inference: dict | None) -> list[dict] | None:
    """Extract predictions as list of dicts from inference result."""
    if not inference:
        return None
    output = inference.get("output")
    if not output:
        return None
    core_predictions = output.get("core_predictions")
    if not core_predictions:
        return None
    return [
        {
            "bbox": p.get("bbox"),
            "class": p.get("core_class"),
            "score": p.get("score"),
        }
        for p in core_predictions
    ]


def compare_pipelines(
    path_a: Path,
    path_b: Path,
    test_cases_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Compare results from two pipeline directories.

    Args:
        path_a: Directory containing pipeline A evaluation results
        path_b: Directory containing pipeline B evaluation results
        test_cases_dir: Optional directory containing test cases (for input file paths)

    Returns:
        Dictionary with comparison data including:
        - matched_results: List of per-example comparisons
        - pipeline_a_only: Results only in pipeline A
        - pipeline_b_only: Results only in pipeline B
        - stats: Summary statistics
        - product_type: The detected product type
        - comparison_metric: The metric used for comparison
    """
    path_a = Path(path_a)
    path_b = Path(path_b)

    # Load evaluation reports
    report_a = load_evaluation_report(path_a)
    report_b = load_evaluation_report(path_b)

    if not report_a or not report_b:
        raise ValueError(
            "Could not load evaluation reports. Make sure both directories contain _evaluation_report.json files."
        )

    # Extract per-example results
    results_a = {r["test_id"]: r for r in report_a.get("per_example_results", [])}
    results_b = {r["test_id"]: r for r in report_b.get("per_example_results", [])}

    # Detect product type from first result
    product_type = "extract"
    if results_a:
        first_result = next(iter(results_a.values()))
        product_type = first_result.get("product_type", "extract").lower()

    comparison_metric = COMPARISON_METRIC_MAP.get(product_type, "accuracy")

    # Compare matched results
    matched_results: list[dict[str, Any]] = []
    pipeline_a_only: list[str] = []
    pipeline_b_only: list[str] = []

    all_test_ids = set(results_a.keys()) | set(results_b.keys())

    for test_id in all_test_ids:
        result_a = results_a.get(test_id)
        result_b = results_b.get(test_id)

        if result_a and result_b:
            # Both have results - compare
            metrics_a = result_a.get("metrics", [])
            metrics_b = result_b.get("metrics", [])

            metric_a = get_metric_value(metrics_a, comparison_metric)
            metric_b = get_metric_value(metrics_b, comparison_metric)

            # Load inference results for output data
            inference_a = load_inference_result(path_a, test_id)
            inference_b = load_inference_result(path_b, test_id)

            # Extract input file path from inference results
            input_file_a = inference_a.get("request", {}).get("source_file_path") if inference_a else None
            input_file_b = inference_b.get("request", {}).get("source_file_path") if inference_b else None

            comparison: dict[str, Any] = {
                "test_id": test_id,
                "input_file": input_file_a or input_file_b,
                "pipeline_a": {
                    "pipeline_name": result_a.get("pipeline_name", "Pipeline A"),
                    "metric_value": metric_a,
                    "success": result_a.get("success", False),
                    "error": result_a.get("error"),
                    "all_metrics": metrics_a,
                    "all_stats": result_a.get("stats", []),
                },
                "pipeline_b": {
                    "pipeline_name": result_b.get("pipeline_name", "Pipeline B"),
                    "metric_value": metric_b,
                    "success": result_b.get("success", False),
                    "error": result_b.get("error"),
                    "all_metrics": metrics_b,
                    "all_stats": result_b.get("stats", []),
                },
            }

            # Add product-type-specific output data
            if product_type == "layout_detection":
                comparison["pipeline_a"]["predictions"] = get_predictions_from_inference(inference_a)
                comparison["pipeline_b"]["predictions"] = get_predictions_from_inference(inference_b)
                # GT annotations would need test case loading which we skip for now
                comparison["gt_annotations"] = None
            elif product_type == "extract":
                output_a = inference_a.get("output", {}) if inference_a else {}
                output_b = inference_b.get("output", {}) if inference_b else {}
                comparison["pipeline_a"]["output"] = output_a.get("extracted_data")
                comparison["pipeline_b"]["output"] = output_b.get("extracted_data")
                comparison["pipeline_a"]["field_citations"] = normalize_field_citations_for_overlay(inference_a)
                comparison["pipeline_b"]["field_citations"] = normalize_field_citations_for_overlay(inference_b)
                comparison["gt_annotations"] = normalize_gt_evidence_for_overlay(
                    load_test_json_for_overlay(test_cases_dir, test_id)
                )
            elif product_type == "parse":
                output_a = inference_a.get("output", {}) if inference_a else {}
                output_b = inference_b.get("output", {}) if inference_b else {}
                comparison["pipeline_a"]["output"] = output_a.get("markdown")
                comparison["pipeline_b"]["output"] = output_b.get("markdown")

            # Determine comparison category
            if metric_a is not None and metric_b is not None:
                if metric_a > metric_b:
                    comparison["category"] = "a_better"
                elif metric_b > metric_a:
                    comparison["category"] = "b_better"
                else:
                    comparison["category"] = "tie"
            elif metric_a is None and metric_b is None:
                comparison["category"] = "both_bad"
            elif metric_a is None:
                comparison["category"] = "b_better"
            else:
                comparison["category"] = "a_better"

            matched_results.append(comparison)
        elif result_a:
            pipeline_a_only.append(test_id)
        elif result_b:
            pipeline_b_only.append(test_id)

    # Get pipeline names from results
    pipeline_a_name = "Pipeline A"
    pipeline_b_name = "Pipeline B"
    if results_a:
        first_a = next(iter(results_a.values()))
        pipeline_a_name = first_a.get("pipeline_name", path_a.name)
    if results_b:
        first_b = next(iter(results_b.values()))
        pipeline_b_name = first_b.get("pipeline_name", path_b.name)

    # Disambiguate if same name
    if pipeline_a_name == pipeline_b_name:
        suffix_a = get_directory_suffix(path_a)
        suffix_b = get_directory_suffix(path_b)

        if suffix_a != suffix_b:
            pipeline_a_name = f"{pipeline_a_name} ({suffix_a})"
            pipeline_b_name = f"{pipeline_b_name} ({suffix_b})"
        else:
            pipeline_a_name = f"{pipeline_a_name} (A)"
            pipeline_b_name = f"{pipeline_b_name} (B)"

    # Calculate statistics
    stats = {
        "total_matched": len(matched_results),
        "a_better": sum(1 for r in matched_results if r["category"] == "a_better"),
        "b_better": sum(1 for r in matched_results if r["category"] == "b_better"),
        "tie": sum(1 for r in matched_results if r["category"] == "tie"),
        "both_bad": sum(1 for r in matched_results if r["category"] == "both_bad"),
        "pipeline_a_only": len(pipeline_a_only),
        "pipeline_b_only": len(pipeline_b_only),
        "pipeline_a_name": pipeline_a_name,
        "pipeline_b_name": pipeline_b_name,
        "product_type": product_type,
        "comparison_metric": comparison_metric,
    }

    return {
        "matched_results": matched_results,
        "pipeline_a_only": pipeline_a_only,
        "pipeline_b_only": pipeline_b_only,
        "stats": stats,
        "product_type": product_type,
        "comparison_metric": comparison_metric,
        "original_base_path": str(test_cases_dir) if test_cases_dir else "",
    }
