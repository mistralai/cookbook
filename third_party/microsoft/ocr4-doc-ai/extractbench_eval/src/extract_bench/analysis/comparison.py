"""Comparison tool for evaluating two different pipeline results."""

import json
from pathlib import Path
from typing import Any

from extract_bench.analysis.comparison_core import (
    load_inference_result,
    load_test_json_for_overlay,
    normalize_field_citations_for_overlay,
    normalize_gt_evidence_for_overlay,
)
from extract_bench.evaluation.layout_adapters import create_layout_adapter_for_result
from extract_bench.evaluation.layout_label_mappers import project_layout_predictions
from extract_bench.schemas.evaluation import EvaluationResult, EvaluationSummary
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.test_cases import load_test_cases
from extract_bench.test_cases.schema import LayoutDetectionTestCase


class PipelineComparison:
    """Compare results from two different pipelines."""

    def __init__(
        self,
        pipeline_a_dir: Path,
        pipeline_b_dir: Path,
        test_cases_dir: Path | None = None,
    ):
        """
        Initialize comparison between two pipelines.

        :param pipeline_a_dir: Directory containing pipeline A evaluation results
        :param pipeline_b_dir: Directory containing pipeline B evaluation results
        :param test_cases_dir: Optional directory containing test cases
        """
        self.pipeline_a_dir = Path(pipeline_a_dir)
        self.pipeline_b_dir = Path(pipeline_b_dir)
        self.test_cases_dir = Path(test_cases_dir) if test_cases_dir else None

    # Metric mapping for different product types
    METRIC_MAP: dict[str, str] = {
        "extract": "accuracy",
        "parse": "rule_pass_rate",
        "layout_detection": "mAP@[.50:.95]",
    }

    def _detect_product_type(self, summary: EvaluationSummary) -> str:
        """Detect product type from evaluation results."""
        if summary.per_example_results:
            return summary.per_example_results[0].product_type
        return "parse"  # default fallback

    def _get_directory_suffix(self, pipeline_dir: Path) -> str:
        """
        Extract a distinguishing suffix from the pipeline directory path.

        Looks for run IDs, dates, or other identifying info in parent directories.
        Example paths:
            /output/financial_tables_run-21391181794/llamaextract_agentic -> "run-21391181794"
            /output/2025-01-27/llamaextract_agentic -> "2025-01-27"
            /output/experiment_v2/llamaextract_agentic -> "experiment_v2"
        """
        import re

        # Get the parent directory name (the run/experiment folder)
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

    def _load_evaluation_summary(self, output_dir: Path) -> EvaluationSummary | None:
        """Load evaluation summary from a directory."""
        eval_report_path = output_dir / "_evaluation_report.json"
        if not eval_report_path.exists():
            return None
        try:
            with open(eval_report_path) as f:
                data = json.load(f)
            return EvaluationSummary.model_validate(data)
        except Exception:
            return None

    def _load_inference_result(self, output_dir: Path, test_id: str) -> InferenceResult | None:
        """Load inference result for a specific test_id."""
        data = load_inference_result(output_dir, test_id)
        if data is None:
            return None
        try:
            return InferenceResult.model_validate(data)
        except Exception:
            return None

    def _get_accuracy(self, eval_result: EvaluationResult) -> float | None:
        """Extract accuracy metric from evaluation result (backward compatibility)."""
        for metric in eval_result.metrics:
            if metric.metric_name == "accuracy":
                return metric.value
        return None

    def _get_comparison_metric(self, eval_result: EvaluationResult, product_type: str) -> float | None:
        """Get the primary comparison metric based on product type."""
        target_metric = self.METRIC_MAP.get(product_type, "accuracy")

        for metric in eval_result.metrics:
            if metric.metric_name == target_metric:
                return metric.value
        return None

    def _get_predictions(self, inference: InferenceResult | None) -> list[dict] | None:
        """Extract predictions as list of dicts for JSON serialization."""
        if not inference or not inference.output:
            return None
        try:
            adapter = create_layout_adapter_for_result(inference)
            layout_output = adapter.to_layout_output(inference)
            projected = project_layout_predictions(
                inference,
                layout_output,
                evaluation_view="core",
                target_ontology="canonical",
            )
            return [
                {
                    "bbox": prediction["bbox"],
                    "class": prediction["class_name"],
                    "score": prediction["score"],
                }
                for prediction in projected
            ]
        except Exception:
            return None

    def _get_gt_annotations(self, test_case: Any) -> list[dict] | None:
        """Extract GT annotations from test case."""
        if not test_case or not isinstance(test_case, LayoutDetectionTestCase):
            return None
        annotations = test_case.get_layout_annotations()
        if not annotations:
            return None
        return [{"bbox": ann.bbox, "class": ann.canonical_class} for ann in annotations]

    def compare(self) -> dict[str, Any]:
        """
        Compare results from both pipelines.

        Returns a dictionary with comparison data including:
        - matched_results: List of comparisons
        - pipeline_a_only: Results only in pipeline A
        - pipeline_b_only: Results only in pipeline B
        - stats: Summary statistics
        - product_type: The detected product type
        - comparison_metric: The metric used for comparison
        """
        # Load evaluation summaries
        summary_a = self._load_evaluation_summary(self.pipeline_a_dir)
        summary_b = self._load_evaluation_summary(self.pipeline_b_dir)

        if not summary_a or not summary_b:
            raise ValueError(
                "Could not load evaluation summaries. "
                "Make sure both directories contain _evaluation_report.json files. "
                "Run evaluation first using: run_evaluation"
            )

        # Detect product type
        product_type = self._detect_product_type(summary_a)
        comparison_metric = self.METRIC_MAP.get(product_type, "accuracy")

        # Create mapping of test_id -> EvaluationResult
        results_a = {r.test_id: r for r in summary_a.per_example_results}
        results_b = {r.test_id: r for r in summary_b.per_example_results}

        # Load test cases if available
        test_cases: dict[str, Any] = {}
        if self.test_cases_dir and self.test_cases_dir.exists():
            test_cases_list = load_test_cases(self.test_cases_dir)
            test_cases = {tc.test_id: tc for tc in test_cases_list}

        # Compare matched results
        matched_results = []
        pipeline_a_only = []
        pipeline_b_only = []

        all_test_ids = set(results_a.keys()) | set(results_b.keys())

        for test_id in all_test_ids:
            result_a = results_a.get(test_id)
            result_b = results_b.get(test_id)

            if result_a and result_b:
                # Both have results - compare using product-type-specific metric
                metric_a = self._get_comparison_metric(result_a, product_type)
                metric_b = self._get_comparison_metric(result_b, product_type)

                # Load inference results for outputs
                inference_a = self._load_inference_result(self.pipeline_a_dir, test_id)
                inference_b = self._load_inference_result(self.pipeline_b_dir, test_id)

                # Get test case for input file and schema
                test_case = test_cases.get(test_id)

                comparison: dict[str, Any] = {
                    "test_id": test_id,
                    "pipeline_a": {
                        "pipeline_name": result_a.pipeline_name,
                        "metric_value": metric_a,
                        "success": result_a.success,
                        "error": result_a.error,
                        "all_metrics": [m.model_dump() for m in result_a.metrics],
                        "all_stats": [s.model_dump() for s in result_a.stats],
                    },
                    "pipeline_b": {
                        "pipeline_name": result_b.pipeline_name,
                        "metric_value": metric_b,
                        "success": result_b.success,
                        "error": result_b.error,
                        "all_metrics": [m.model_dump() for m in result_b.metrics],
                        "all_stats": [s.model_dump() for s in result_b.stats],
                    },
                    "input_file": str(test_case.file_path) if test_case else None,
                }

                # Add product-type-specific data
                if product_type == "layout_detection":
                    comparison["pipeline_a"]["predictions"] = self._get_predictions(inference_a)
                    comparison["pipeline_b"]["predictions"] = self._get_predictions(inference_b)
                    comparison["gt_annotations"] = self._get_gt_annotations(test_case)
                elif product_type == "extract":
                    comparison["pipeline_a"]["output"] = (
                        inference_a.output.extracted_data
                        if inference_a and hasattr(inference_a.output, "extracted_data")
                        else None
                    )
                    comparison["pipeline_b"]["output"] = (
                        inference_b.output.extracted_data
                        if inference_b and hasattr(inference_b.output, "extracted_data")
                        else None
                    )
                    comparison["pipeline_a"]["field_citations"] = normalize_field_citations_for_overlay(
                        inference_a.model_dump() if inference_a else None
                    )
                    comparison["pipeline_b"]["field_citations"] = normalize_field_citations_for_overlay(
                        inference_b.model_dump() if inference_b else None
                    )
                    # Prefer on-disk sidecar (has _field_rules); fall back empty.
                    comparison["gt_annotations"] = normalize_gt_evidence_for_overlay(
                        load_test_json_for_overlay(self.test_cases_dir, test_id)
                    )
                    comparison["schema"] = (
                        test_case.data_schema if test_case and hasattr(test_case, "data_schema") else None
                    )
                elif product_type == "parse":
                    comparison["pipeline_a"]["output"] = (
                        inference_a.output.markdown if inference_a and hasattr(inference_a.output, "markdown") else None
                    )
                    comparison["pipeline_b"]["output"] = (
                        inference_b.output.markdown if inference_b and hasattr(inference_b.output, "markdown") else None
                    )

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
                pipeline_a_only.append(result_a.test_id)
            elif result_b:
                pipeline_b_only.append(result_b.test_id)

        # Get pipeline names
        pipeline_a_name = (
            summary_a.per_example_results[0].pipeline_name if summary_a.per_example_results else "Pipeline A"
        )
        pipeline_b_name = (
            summary_b.per_example_results[0].pipeline_name if summary_b.per_example_results else "Pipeline B"
        )

        # De-duplicate pipeline names if they're the same
        if pipeline_a_name == pipeline_b_name:
            # Extract distinguishing info from directory paths
            suffix_a = self._get_directory_suffix(self.pipeline_a_dir)
            suffix_b = self._get_directory_suffix(self.pipeline_b_dir)

            if suffix_a != suffix_b:
                pipeline_a_name = f"{pipeline_a_name} ({suffix_a})"
                pipeline_b_name = f"{pipeline_b_name} ({suffix_b})"
            else:
                # Fallback to generic A/B if suffixes are also the same
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
            "original_base_path": str(self.test_cases_dir) if self.test_cases_dir else "",
        }
