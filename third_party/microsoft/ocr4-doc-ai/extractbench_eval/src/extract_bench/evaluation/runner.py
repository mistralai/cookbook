"""Evaluation runner for computing metrics on inference results."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unicodedata
from collections.abc import Callable
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,
)
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.evaluation.evaluators.layoutdet import LayoutDetectionEvaluator
from extract_bench.evaluation.evaluators.parse import ParseEvaluator
from extract_bench.evaluation.evaluators.qa import QAEvaluator
from extract_bench.evaluation.layout_adapters import create_layout_adapter_for_result
from extract_bench.evaluation.metrics.field_grounding.extract_adapter import (
    compute_failure_field_denominator,
)
from extract_bench.evaluation.stats import build_operational_stats
from extract_bench.schemas.evaluation import EvaluationResult, EvaluationSummary, MetricValue
from extract_bench.schemas.layout_detection_output import LayoutOutput
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases import load_test_cases
from extract_bench.test_cases.parse_rule_schemas import get_rule_type
from extract_bench.test_cases.rule_filters import filter_verified_test_rules
from extract_bench.test_cases.schema import (
    ExtractTestCase,
    LayoutDetectionTestCase,
    ParseTestCase,
    TestCase,
)

if TYPE_CHECKING:
    from extract_bench.schemas.parse_output import ParseOutput


def _is_skipped_result(result: EvaluationResult) -> bool:
    """Distinguish a legitimately-skipped result from a genuine failure.

    Both carry ``success=False``. A skip is a case the provider was never
    meant to be scored on (e.g. a cross-eval page for which it produced no
    layout data); a failure is a case it was meant to handle but errored on.
    Only failures should count as 0 in the aggregate denominator.
    """
    return bool(result.error and "No layout data" in result.error)


def _is_infra_failure(result: EvaluationResult) -> bool:
    """Failure caused by the evaluation harness, not by the provider's output.

    Evaluator crashes and async task errors still count as failures in the
    summary, but they must not zero-pad the provider's scores in
    ``_aggregate_metrics`` — an evaluator bug or transient harness error
    should not move the leaderboard.

    Exception: a worker error whose cause is that the provider's output could
    not be adapted to a ``LayoutOutput`` (raised in the layout adapters /
    layout-detection evaluator) is a *provider* failure, not a harness one —
    the provider simply produced nothing the layout task can score. It must
    count as a genuine 0, so it is explicitly excluded from the infra set.
    """
    if not result.error:
        return False
    if "not LayoutOutput" in result.error:
        return False
    return result.error.startswith(("Worker error:", "Evaluation error:", "Task execution error:"))


# Diagnostic metrics that are counts or lower-is-better rates. Padding a
# synthetic 0 for a failed example (see _aggregate_metrics) would be
# meaningless for a count, and for a lower-is-better rate it would make
# failures *improve* the aggregate. Metrics carrying "count" metadata are
# excluded from padding automatically; this set covers the diagnostic
# metrics that don't carry it. New higher-is-better score metrics need no
# entry here — only new diagnostic metrics do.
_NON_SCORE_METRICS = frozenset(
    {
        "num_predictions",
        "num_ground_truth",
        "tables_expected",
        "tables_actual",
        "tables_paired",
        "tables_unmatched_expected",
        "tables_unmatched_pred",
        "tables_unparseable_pred",
        "parse_field_gt_count",
        "unmatched_gt_elements",
        "unmatched_pred_elements",
        "null_hallucination_rate",  # lower is better: a synthetic 0 would reward failure
    }
)

# Extract metric names whose successful-path metadata carries pooled
# ``tp/fp/fn`` counts with the per-doc eligible rule count as denominator
# (see ``_compute_v02_evidence_metrics`` in
# ``metrics/field_grounding/extract_adapter.py``: every one of these emits
# ``tp + fp + fn == total = len(eligible)``). When a doc fails inference we
# project the failure as ``tp=0, fp=N, fn=0`` for these so the doc counts
# in pooled (micro) aggregations as ``0/N`` instead of being silently
# dropped as ``0/0``.
#
# Excluded on purpose:
#  - ``extract_evidence_page_covered_pass_rate`` — denominator is
#    ``page_qualified``, not the full rule count; mixing ``fp=N`` (=total)
#    into the pooled denominator would over-penalize via the wrong base.
#  - ``extract_evidence_bbox_IOU_alignment`` /
#    ``extract_evidence_bbox_IOU_pass_rate`` — diagnostic metrics that emit
#    ``tp=fp=fn=0`` on the success path; injecting ``fp=N`` on failure
#    would surface a brand-new ``micro_*`` aggregate that doesn't exist
#    today and isn't well-defined as a 0/N pool.
_EXTRACT_FAILURE_POOLED_METRICS: frozenset[str] = frozenset(
    {
        "extract_evidence_value_pass_rate",
        "extract_evidence_page_pass_rate",
        "extract_evidence_bbox_coverage",
        "extract_evidence_bbox_covered_pass_rate",
    }
)

# Headline metrics emitted for *every* document of a product type, used only to
# zero-fill a run in which nothing succeeded at all. In the normal partial-
# failure path the metric names come from the successful results, which is
# strictly more faithful; this is the fallback so that a pipeline failing on
# 100% of documents scores 0.0 rather than disappearing from the report
# entirely. Extract's page/grounded families are deliberately excluded — they
# are conditional on the ground truth carrying evidence boxes, so synthesizing
# them would invent a denominator the successful path would not have produced.
_FALLBACK_METRIC_NAMES: dict[str, tuple[str, ...]] = {
    "extract": (
        "extract_unified_value_precision",
        "extract_unified_value_recall",
        "extract_unified_value_f1",
    ),
}

# Metric prefixes NEVER zero-synthesized for a failed inference. The penalty
# convention exists so error-prone pipelines cannot game *accuracy* by dropping
# hard documents — extraction reliability is already captured by the value-F1
# zeros and the success counters. Confidence metrics instead measure how well
# the confidence model *ranks* an extraction's fields, which is undefined when
# no extraction happened; scoring them 0 conflates extraction reliability with
# confidence-model quality and makes one crashed document swing a small slice's
# mean by double digits. Failed documents drop out of confidence aggregates.
_FAILURE_SYNTHESIS_EXCLUDED_METRIC_PREFIXES: tuple[str, ...] = ("confidence_",)


def _eligible_for_failure_synthesis(metric_name: str) -> bool:
    return not metric_name.startswith(_FAILURE_SYNTHESIS_EXCLUDED_METRIC_PREFIXES)


def _build_failure_metric_metadata(
    metric_name: str,
    failure_metadata: dict[str, Any],
    pooled_denom: int,
) -> dict[str, Any]:
    """Per-metric metadata for an inference-failed Extract result.

    For metrics in :data:`_EXTRACT_FAILURE_POOLED_METRICS` with a positive
    denominator, inject ``tp=0, fp=N, fn=0`` so the failure pools into
    micro aggregations as ``0/N``. Other metrics keep the bare failure
    marker — the runner aggregator's PRF / pass_rate path is gated on
    ``tp+fp+fn > 0``, so omitting the counts preserves the current
    "diagnostic / wrong-denominator metric stays out of micro" behaviour.
    """
    md = dict(failure_metadata)
    if pooled_denom > 0 and metric_name in _EXTRACT_FAILURE_POOLED_METRICS:
        md.update(
            {
                "tp": 0,
                "fp": pooled_denom,
                "fn": 0,
                "total": pooled_denom,
                "denominator": "inference_failed_synthesised",
            }
        )
    return md


# Module-level worker function for ProcessPoolExecutor (must be picklable)
def _evaluate_single_worker(
    inference_result_dict: dict[str, Any],
    test_case_dict: dict[str, Any],
    test_case_type: str,
    eval_mode: str | bool,
    evaluator_type: str | None,
    default_layout_ontology: str = "basic",
    enable_teds: bool = False,
    skip_rules: bool = False,
    verified_only: bool = False,
) -> dict[str, Any]:
    """
    Worker function for parallel evaluation using ProcessPoolExecutor.

    This function runs in a separate process, so it must:
    1. Accept only picklable arguments (dicts, not Pydantic models)
    2. Create evaluators locally (they can't be pickled)
    3. Return a dict (not Pydantic model)

    :param inference_result_dict: Serialized InferenceResult
    :param test_case_dict: Serialized TestCase
    :param test_case_type: Type of test case ("parse", "layout_detection", etc.)
    :param eval_mode: "multi_task", True (cross_eval), or False (normal)
    :param evaluator_type: Type of evaluator to use (None for multi_task)
    :param default_layout_ontology: Default ontology to use when test case omits ontology
    :param enable_teds: Enable TEDS metric computation in parse evaluation
    :param skip_rules: Skip rule-based metric computation in parse evaluation
    :param verified_only: Discard test rules explicitly marked verified=false
    :return: Serialized EvaluationResult dict
    """
    # Import here to avoid circular imports and ensure fresh state in worker
    from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
    from extract_bench.evaluation.evaluators.layoutdet import LayoutDetectionEvaluator
    from extract_bench.evaluation.evaluators.parse import ParseEvaluator
    from extract_bench.evaluation.layout_adapters import (
        create_layout_adapter_for_result,
    )
    from extract_bench.schemas.evaluation import EvaluationResult
    from extract_bench.schemas.pipeline_io import InferenceResult
    from extract_bench.schemas.product import ProductType
    from extract_bench.test_cases.schema import (
        ExtractTestCase,
        LayoutDetectionTestCase,
        ParseTestCase,
    )

    try:
        # Deserialize inputs
        inference_result = InferenceResult.model_validate(inference_result_dict)

        # Deserialize test case based on type
        test_case: ExtractTestCase | LayoutDetectionTestCase | ParseTestCase
        if test_case_type == "layout_detection":
            test_case = LayoutDetectionTestCase.model_validate(test_case_dict)
        elif test_case_type == "parse":
            test_case = ParseTestCase.model_validate(test_case_dict)
        elif test_case_type == "extract":
            test_case = ExtractTestCase.model_validate(test_case_dict)
        else:
            raise ValueError(f"Unknown test_case_type: {test_case_type}")

        if verified_only:
            test_case = filter_verified_test_rules(test_case)

        # Create evaluator based on type
        evaluators: dict[
            str,
            ExtractEvaluator | ParseEvaluator | LayoutDetectionEvaluator,
        ] = {
            "extract": ExtractEvaluator(),
            "parse": ParseEvaluator(
                enable_teds=enable_teds,
                enable_rule_based=not skip_rules,
            ),
            "layout_detection": LayoutDetectionEvaluator(default_ontology=default_layout_ontology),
        }

        if eval_mode == "multi_task":
            # Multi-task evaluation needs special handling
            # For now, return error - multi_task is complex and rarely used
            # The main parallel path is for normal evaluations
            return EvaluationResult(
                test_id=test_case.test_id,
                example_id=inference_result.request.example_id,
                pipeline_name=inference_result.pipeline_name,
                product_type=inference_result.product_type.value,
                success=False,
                error="multi_task evaluation not supported in parallel mode",
            ).model_dump()

        elif eval_mode is True:  # is_cross_eval
            # Cross-evaluation: extract layout from PARSE result
            assert isinstance(test_case, LayoutDetectionTestCase)

            adapter = create_layout_adapter_for_result(inference_result)
            layout_output = adapter.to_layout_output(
                inference_result,
                page_filter=test_case.page_index + 1,
            )
            if not layout_output.predictions:
                return EvaluationResult(
                    test_id=test_case.test_id,
                    example_id=inference_result.request.example_id,
                    pipeline_name=inference_result.pipeline_name,
                    product_type="layout_detection",
                    success=False,
                    error=f"No layout data for page {test_case.page_index}",
                ).model_dump()

            # Build a synthetic LAYOUT_DETECTION result from adapted output.
            layout_inference_result = InferenceResult(
                request=inference_result.request,
                pipeline_name=inference_result.pipeline_name,
                product_type=ProductType.LAYOUT_DETECTION,
                raw_output=inference_result.raw_output,
                output=layout_output,
                started_at=inference_result.started_at,
                completed_at=inference_result.completed_at,
                latency_in_ms=inference_result.latency_in_ms,
            )
            layout_evaluator = evaluators["layout_detection"]
            result = layout_evaluator.evaluate(layout_inference_result, test_case)
            return result.model_dump()
        else:
            # Normal evaluation
            if evaluator_type is None or evaluator_type not in evaluators:
                return EvaluationResult(
                    test_id=test_case.test_id,
                    example_id=inference_result.request.example_id,
                    pipeline_name=inference_result.pipeline_name,
                    product_type=inference_result.product_type.value,
                    success=False,
                    error=f"No evaluator for type: {evaluator_type}",
                ).model_dump()
            evaluator = evaluators[evaluator_type]
            result = evaluator.evaluate(inference_result, test_case)
            return result.model_dump()

    except Exception as e:
        # Return error result
        return {
            "test_id": test_case_dict.get("test_id", "unknown"),
            "example_id": inference_result_dict.get("request", {}).get("example_id", "unknown"),
            "pipeline_name": inference_result_dict.get("pipeline_name", "unknown"),
            "product_type": inference_result_dict.get("product_type", "unknown"),
            "success": False,
            "error": f"Worker error: {str(e)}",
            "metrics": [],
            "stats": [],
        }


def _scale_layout_output_coordinates(
    layout_output: LayoutOutput,
    target_width: int,
    target_height: int,
) -> LayoutOutput:
    """
    Scale layout output coordinates from source space to target space.

    :param layout_output: Layout output with predictions in source coordinate space
    :param target_width: Target image width (ground truth dimensions)
    :param target_height: Target image height (ground truth dimensions)
    :return: New LayoutOutput with scaled coordinates
    """
    if layout_output.image_width == 0 or layout_output.image_height == 0:
        return layout_output

    # Calculate scale factors
    x_scale = target_width / layout_output.image_width
    y_scale = target_height / layout_output.image_height

    # If no scaling needed, return as-is
    if abs(x_scale - 1.0) < 0.001 and abs(y_scale - 1.0) < 0.001:
        return layout_output

    def scale_bbox(bbox: list[float]) -> list[float]:
        """Scale bbox [x1, y1, x2, y2] to target space."""
        return [
            bbox[0] * x_scale,
            bbox[1] * y_scale,
            bbox[2] * x_scale,
            bbox[3] * y_scale,
        ]

    # Scale raw predictions
    scaled_predictions = []
    for pred in layout_output.predictions:
        scaled_pred = pred.model_copy(update={"bbox": scale_bbox(pred.bbox)})
        scaled_predictions.append(scaled_pred)

    return LayoutOutput(
        task_type=layout_output.task_type,
        example_id=layout_output.example_id,
        pipeline_name=layout_output.pipeline_name,
        model=layout_output.model,
        image_width=target_width,
        image_height=target_height,
        predictions=scaled_predictions,
    )


class EvaluationRunner:
    """
    Runs evaluation on saved inference results.

    Loads inference results from output directory, matches them with test cases,
    and computes metrics using product-specific evaluators.
    """

    def __init__(
        self,
        output_dir: Path,
        test_cases_dir: Path | None = None,
        multi_task: bool = True,
        enable_teds: bool = False,
        skip_rules: bool = False,
        layout_ontology: str = "basic",
        verified_only: bool = False,
    ):
        """
        Initialize the evaluation runner.

        :param output_dir: Directory containing inference results
        :param test_cases_dir: Optional directory containing test cases (if different from data)
        :param multi_task: Enable multi-task evaluation for mixed rule types
        :param enable_teds: Enable TEDS metric computation in parse evaluation
        :param skip_rules: Skip rule-based metric computation in parse evaluation
        :param layout_ontology: Default layout ontology when test case does not specify one
        :param verified_only: Discard test rules explicitly marked verified=false
        """
        self.output_dir = Path(output_dir)
        self.test_cases_dir = Path(test_cases_dir) if test_cases_dir else None
        self.multi_task = multi_task
        self.enable_teds = enable_teds
        self.skip_rules = skip_rules
        self.layout_ontology = layout_ontology
        self.verified_only = verified_only

        # Register default evaluators
        self._evaluators: dict[str, Any] = {}
        # Register ParseEvaluator for PARSE product type
        self.register_evaluator(
            "parse",
            ParseEvaluator(
                enable_teds=enable_teds,
                enable_rule_based=not skip_rules,
            ),
        )
        # Register QAEvaluator for PARSE product type with QA test cases
        self.register_evaluator("qa", QAEvaluator())
        # Register LayoutDetectionEvaluator for LAYOUT_DETECTION product type
        self.register_evaluator(
            "layout_detection",
            LayoutDetectionEvaluator(default_ontology=self.layout_ontology),
        )
        self.register_evaluator("extract", ExtractEvaluator())

    def register_evaluator(self, product_type: str, evaluator: Any) -> None:
        """
        Register a product-specific evaluator.

        :param product_type: Product type (e.g., 'extract', 'parse')
        :param evaluator: Evaluator instance implementing BaseEvaluator
        """
        self._evaluators[product_type] = evaluator

    def _load_inference_result(self, result_path: Path) -> InferenceResult | None:
        """
        Load an inference result from a JSON file.

        :param result_path: Path to the result JSON file
        :return: InferenceResult or None if loading fails
        """
        try:
            with open(result_path) as f:
                data = json.load(f)
            return InferenceResult.model_validate(data)
        except Exception:
            return None

    def _find_result_files(self, output_dir: Path) -> list[Path]:
        """
        Find all result JSON files in the output directory.

        :param output_dir: Directory to search
        :return: List of paths to result JSON files
        """
        result_files = []
        # Look for .result.json files (normalized results)
        for result_file in output_dir.rglob("*.result.json"):
            result_files.append(result_file)
        return sorted(result_files)

    def _evaluate_single(
        self,
        inference_result: InferenceResult,
        test_case: TestCase,
        evaluator: Any,
        eval_mode: str | bool,
    ) -> EvaluationResult:
        """
        Evaluate a single test case (thread-safe helper for parallel execution).

        :param inference_result: The inference result to evaluate
        :param test_case: The test case with expected values
        :param evaluator: The evaluator to use (None for multi_task mode)
        :param eval_mode: "multi_task", True (cross_eval), or False (normal)
        :return: EvaluationResult
        """
        try:
            if eval_mode == "multi_task":
                # Multi-task evaluation: split rules and run both evaluators
                assert isinstance(test_case, (LayoutDetectionTestCase, ParseTestCase))
                return self._evaluate_multi_task(inference_result, test_case)
            elif eval_mode is True:  # is_cross_eval
                # Cross-evaluation: extract layout from PARSE result and evaluate
                assert isinstance(test_case, LayoutDetectionTestCase)

                adapter = create_layout_adapter_for_result(inference_result)
                layout_output = adapter.to_layout_output(
                    inference_result,
                    page_filter=test_case.page_index + 1,
                )
                if not layout_output.predictions:
                    return EvaluationResult(
                        test_id=test_case.test_id,
                        example_id=inference_result.request.example_id,
                        pipeline_name=inference_result.pipeline_name,
                        product_type="layout_detection",
                        success=False,
                        error=f"No layout data for page {test_case.page_index}",
                    )

                # Create a synthetic InferenceResult with layout output
                layout_inference_result = InferenceResult(
                    request=inference_result.request,
                    pipeline_name=inference_result.pipeline_name,
                    product_type=ProductType.LAYOUT_DETECTION,
                    raw_output=inference_result.raw_output,
                    output=layout_output,
                    started_at=inference_result.started_at,
                    completed_at=inference_result.completed_at,
                    latency_in_ms=inference_result.latency_in_ms,
                )
                return evaluator.evaluate(layout_inference_result, test_case)  # type: ignore[no-any-return]
            else:
                return evaluator.evaluate(inference_result, test_case)  # type: ignore[no-any-return]
        except Exception as e:
            return EvaluationResult(
                test_id=test_case.test_id,
                example_id=inference_result.request.example_id,
                pipeline_name=inference_result.pipeline_name,
                product_type=inference_result.product_type.value,
                success=False,
                error=f"Evaluation error: {str(e)}",
            )

    def _match_result_with_test_case(
        self,
        inference_result: InferenceResult,
        test_cases: dict[str, TestCase],
    ) -> TestCase | None:
        """
        Match an inference result with a test case by example_id/test_id.

        :param inference_result: The inference result
        :param test_cases: Dictionary mapping test_id to TestCase
        :return: Matching TestCase or None
        """
        example_id = inference_result.request.example_id
        # Try direct match first
        if example_id in test_cases:
            return test_cases[example_id]
        # Fall back to NFC-normalized comparison so that filenames whose
        # accented characters were stored as NFD on one side and NFC on
        # the other (common after a macOS round-trip) still match.
        example_id_nfc = unicodedata.normalize("NFC", example_id)
        example_id_nfc_stem = example_id_nfc.rsplit(".", 1)[0]
        for test_id, test_case in test_cases.items():
            test_id_nfc = unicodedata.normalize("NFC", test_id)
            if test_id_nfc == example_id_nfc or test_id_nfc == example_id_nfc_stem:
                return test_case
        return None

    def _match_result_with_test_cases_multi(
        self,
        inference_result: InferenceResult,
        test_cases: dict[str, TestCase],
    ) -> list[TestCase]:
        """
        Match an inference result with multiple test cases by example_id prefix.

        Used for cross-evaluation where one PARSE result can match multiple
        LAYOUT_DETECTION test cases (one per page).

        :param inference_result: The inference result
        :param test_cases: Dictionary mapping test_id to TestCase
        :return: List of matching TestCases
        """
        example_id = inference_result.request.example_id
        matches = []

        # Try direct match first
        if example_id in test_cases:
            matches.append(test_cases[example_id])
            return matches

        # NFC-normalized fallback for accented filenames whose Unicode form
        # differs between sides (e.g. NFD result vs NFC test on disk).
        example_id_nfc = unicodedata.normalize("NFC", example_id)
        if example_id_nfc != example_id:
            for test_id, test_case in test_cases.items():
                if unicodedata.normalize("NFC", test_id) == example_id_nfc:
                    matches.append(test_case)
                    return matches

        # For multi-page: match test_ids that start with example_id + "/"
        # e.g., example_id="pdfs/uber" matches "pdfs/uber/page_0", "pdfs/uber/page_1"
        prefix_nfc = example_id_nfc + "/"
        for test_id, test_case in test_cases.items():
            if unicodedata.normalize("NFC", test_id).startswith(prefix_nfc):
                matches.append(test_case)

        return matches

    def run_evaluation(
        self,
        product_type: str | None = None,
        pipeline_name: str | None = None,
        group: str | None = None,
        verbose: bool = False,
        use_rich: bool | None = None,
        max_workers: int | None = None,
    ) -> EvaluationSummary:
        """
        Run evaluation on all inference results in the output directory.

        :param product_type: Optional filter by product type
        :param pipeline_name: Optional filter by pipeline name
        :param group: Optional filter by group name
        :param verbose: Show detailed information about skipped results
        :param use_rich: Whether to use Rich for progress indication (default: auto-detect)
        :param max_workers: Number of worker threads for parallel evaluation (default: CPU count)
        :return: Evaluation summary with aggregated metrics
        """
        # Auto-detect Rich usage if not specified
        if use_rich is None:
            use_rich = sys.stdout.isatty() and not verbose
        console = Console() if use_rich else None
        # Load test cases if test_cases_dir is provided
        test_cases_dict: dict[str, TestCase] = {}
        if self.test_cases_dir:
            test_cases = load_test_cases(
                root_dir=self.test_cases_dir,
                require_test_json=False,
                product_type=None if product_type == "parse" else product_type,
            )
            # Filter by group if specified
            if group:
                original_count = len(test_cases)
                test_cases = [tc for tc in test_cases if tc.group == group]
                if verbose:
                    print(
                        f"📋 Filtered to {len(test_cases)} test cases in group '{group}' (from {original_count} total)"
                    )
            if self.verified_only:
                test_cases = [filter_verified_test_rules(tc) for tc in test_cases]
            test_cases_dict = {tc.test_id: tc for tc in test_cases}
            if verbose:
                print(f"📋 Loaded {len(test_cases_dict)} test cases")
                if test_cases_dict:
                    sample_ids = list(test_cases_dict.keys())[:3]
                    print(f"   Sample test_ids: {sample_ids}")

        # Find all result files
        result_files = self._find_result_files(self.output_dir)
        if verbose:
            print(f"📁 Found {len(result_files)} result files")

        # Filter by group if specified
        # Result files are saved as: output_dir/group/test_id.result.json
        # So we can filter by checking the parent directory name
        # text_content and text_formatting share inference results in text/
        _INFERENCE_DIR = {"text_content": "text", "text_formatting": "text"}
        if group:
            original_file_count = len(result_files)
            match_dir = _INFERENCE_DIR.get(group, group)
            result_files = [f for f in result_files if f.parent.name == match_dir]
            if verbose:
                print(f"   Filtered to {len(result_files)} files in group '{group}' (from {original_file_count} total)")

        # Filter by pipeline if specified
        if pipeline_name:
            # Match a whole path component, not a substring: pipeline names in
            # the registry are prefixes of one another (`llamaextract_agentic`
            # vs `llamaextract_agentic_plus`, `extend_extract` vs
            # `extend_extract_max`), and a substring test silently pulls the
            # longer pipeline's results into the shorter one's scores.
            result_files = [f for f in result_files if pipeline_name in f.parent.parts]
            if verbose:
                print(f"   Filtered to {len(result_files)} files for pipeline '{pipeline_name}'")

        # Load and evaluate each result
        evaluation_results: list[EvaluationResult] = []
        successful = 0
        failed = 0
        skipped = 0

        # Separate QA and non-QA evaluations
        qa_evaluation_tasks: list[tuple[InferenceResult, ParseTestCase, QAEvaluator]] = []
        # (inference_result, test_case, evaluator, eval_mode)
        # eval_mode: True = cross_eval, False = normal, "multi_task" = multi-task eval
        non_qa_evaluations: list[tuple[InferenceResult, TestCase, Any, bool | str]] = []

        # First pass: collect all evaluations and separate QA from non-QA
        for result_file in result_files:
            inference_result = self._load_inference_result(result_file)
            if not inference_result:
                skipped += 1
                if verbose:
                    print(f"⚠️  Skipped {result_file.name}: Failed to load inference result")
                continue

            # Filter by product type if specified
            # Allow cross-evaluation: PARSE results can be evaluated against LAYOUT_DETECTION tests
            is_cross_eval_allowed = (
                product_type == "layout_detection" and inference_result.product_type == ProductType.PARSE
            )
            if product_type and inference_result.product_type.value != product_type:
                if not is_cross_eval_allowed:
                    skipped += 1
                    if verbose:
                        print(
                            f"⚠️  Skipped {result_file.name}: Product type mismatch "
                            f"({inference_result.product_type.value} != {product_type})"
                        )
                    continue

            # Check for cross-evaluation: PARSE result against LAYOUT_DETECTION tests
            is_cross_eval_candidate = is_cross_eval_allowed and inference_result.product_type == ProductType.PARSE

            if is_cross_eval_candidate:
                # Cross-evaluation: match multiple layout test cases (one per page)
                matched_test_cases = self._match_result_with_test_cases_multi(inference_result, test_cases_dict)
                if not matched_test_cases:
                    skipped += 1
                    if verbose:
                        print(
                            f"⚠️  Skipped {result_file.name}: No matching layout test cases found "
                            f"(example_id: {inference_result.request.example_id})"
                        )
                    continue

                evaluator = self._evaluators.get("layout_detection")
                if not evaluator:
                    skipped += 1
                    if verbose:
                        print(f"⚠️  Skipped {result_file.name}: No layout_detection evaluator for cross-evaluation")
                    continue

                # Add one evaluation task per matched test case (per page)
                for test_case in matched_test_cases:
                    non_qa_evaluations.append((inference_result, test_case, evaluator, True))  # True = is_cross_eval
                continue

            # Regular matching: single test case
            test_case = self._match_result_with_test_case(inference_result, test_cases_dict)  # type: ignore[assignment]
            if not test_case:
                skipped += 1
                if verbose:
                    print(
                        f"⚠️  Skipped {result_file.name}: No matching test case found "
                        f"(example_id: {inference_result.request.example_id})"
                    )
                continue

            # Get appropriate evaluator
            # Expand qa_configs (plural) into per-question QA evaluation tasks
            has_qa_configs = isinstance(test_case, ParseTestCase) and test_case.qa_configs
            if has_qa_configs:
                evaluator = self._evaluators.get("qa")
                if evaluator:
                    assert isinstance(test_case, ParseTestCase)
                    for i, qc in enumerate(test_case.qa_configs, 1):  # type: ignore[arg-type]
                        per_q_tc = test_case.model_copy(
                            update={
                                "test_id": f"{test_case.test_id}#q{i}",
                                "qa_config": qc,
                                "qa_configs": None,
                            }
                        )
                        if evaluator.can_evaluate(inference_result, per_q_tc):
                            qa_evaluation_tasks.append((inference_result, per_q_tc, evaluator))
                continue

            is_qa_test = isinstance(test_case, ParseTestCase) and test_case.qa_config is not None

            if is_qa_test:
                evaluator = self._evaluators.get("qa")
                if not evaluator:
                    skipped += 1
                    if verbose:
                        print(f"⚠️  Skipped {result_file.name}: No QA evaluator registered")
                    continue
                if not evaluator.can_evaluate(inference_result, test_case):
                    skipped += 1
                    if verbose:
                        print(
                            f"⚠️  Skipped {result_file.name}: QA evaluator cannot handle this case "
                            f"(test_id: {test_case.test_id})"
                        )
                    continue
                qa_evaluation_tasks.append((inference_result, test_case, evaluator))  # type: ignore[arg-type]
            else:
                # Check for multi-task evaluation: test case has mixed rule types
                # Multi-task works with PARSE results, or LAYOUT_DETECTION results
                # that contain LlamaParse data (pages with markdown)
                is_llamaparse_output = self._is_llamaparse_output(inference_result)
                has_mixed = self._has_mixed_rules(test_case)
                is_multi_task_eval = (
                    self.multi_task
                    and (
                        inference_result.product_type == ProductType.PARSE
                        or (inference_result.product_type == ProductType.LAYOUT_DETECTION and is_llamaparse_output)
                    )
                    and has_mixed
                )

                if is_multi_task_eval:
                    # Multi-task evaluation: split rules and run both evaluators
                    # Use None evaluator as marker; actual evaluators called
                    # in _evaluate_multi_task
                    non_qa_evaluations.append((inference_result, test_case, None, "multi_task"))
                    continue

                # Check for cross-evaluation: PARSE result against LayoutDetectionTestCase
                is_cross_eval = (
                    isinstance(test_case, LayoutDetectionTestCase)
                    and inference_result.product_type == ProductType.PARSE
                )

                if is_cross_eval:
                    # Cross-evaluation: extract layout from PARSE result
                    evaluator = self._evaluators.get("layout_detection")
                    if not evaluator:
                        skipped += 1
                        if verbose:
                            print(f"⚠️  Skipped {result_file.name}: No layout_detection evaluator for cross-evaluation")
                        continue
                    # Mark this as cross-evaluation for special handling later
                    non_qa_evaluations.append((inference_result, test_case, evaluator, True))  # True = is_cross_eval
                else:
                    result_product_type = inference_result.product_type.value
                    evaluator = self._evaluators.get(result_product_type)
                    if not evaluator:
                        skipped += 1
                        if verbose:
                            print(
                                f"⚠️  Skipped {result_file.name}: No evaluator registered for "
                                f"product type: {result_product_type}"
                            )
                        continue
                    if not evaluator.can_evaluate(inference_result, test_case):
                        skipped += 1
                        if verbose:
                            reason = "Evaluator cannot evaluate this case"
                            print(
                                f"⚠️  Skipped {result_file.name}: {reason} "
                                f"(test_id: {test_case.test_id}, "
                                f"example_id: {inference_result.request.example_id})"
                            )
                        continue
                    non_qa_evaluations.append((inference_result, test_case, evaluator, False))  # False = not cross-eval

        # Score test cases with no inference result as blank output (0.0).
        # Without this, tools that fail to parse hard documents have those
        # docs silently dropped from the aggregate averages, inflating their
        # scores relative to tools that produce output for every document.
        if test_cases_dict:
            from extract_bench.schemas.parse_output import ParseOutput

            covered_ids = {tc.test_id for _, tc, _, _ in non_qa_evaluations}
            covered_ids.update(tc.test_id.split("#q")[0] for _, tc, _ in qa_evaluation_tasks)
            parse_evaluator = self._evaluators.get(ProductType.PARSE.value)
            synthesized = 0
            for test_id, test_case in test_cases_dict.items():
                if test_id in covered_ids:
                    continue
                # Layout-detection GT with no usable inference output: the
                # provider produced nothing to localize, so this is a genuine 0,
                # not a vanished example. A blank parse output can't be routed
                # through the layout evaluator (and its empty-prediction path
                # would be classified as a skip), so emit a failed result
                # directly — it is then counted as 0 by the count-as-zero
                # aggregation. This also covers result files that exist but
                # could not be evaluated (e.g. an error result with no output),
                # hence "usable". GT cases with no layout annotations are
                # exempt: no provider could score on them, so a 0 would punish
                # the provider for a ground-truth data issue.
                if isinstance(test_case, LayoutDetectionTestCase):
                    if not test_case.get_layout_annotations():
                        continue
                    # Tag this synthetic 0 as PARSE, not LAYOUT_DETECTION: the
                    # layout group scores cross-eval'd parse output, so its
                    # metrics (rule_pass_rate, etc.) are emitted under the PARSE
                    # product type. Per-product-type padding only zero-pads a
                    # metric with failures of a product type that produced it,
                    # so a LAYOUT_DETECTION-tagged failure here would never reach
                    # the parse metrics and the provider's no-output pages would
                    # silently drop out of the denominator (inflating the score).
                    evaluation_results.append(
                        EvaluationResult(
                            test_id=test_id,
                            example_id=test_id,
                            pipeline_name=pipeline_name or self.output_dir.name,
                            product_type=ProductType.PARSE.value,
                            success=False,
                            metrics=[],
                            error="No usable inference output for this example",
                        )
                    )
                    failed += 1
                    synthesized += 1
                    continue
                # Only plain parse test cases get a blank parse evaluation; QA /
                # extract cases have evaluator-specific requirements a blank can't satisfy.
                if not isinstance(test_case, ParseTestCase) or test_case.qa_config is not None:
                    continue
                if not parse_evaluator:
                    continue
                now = datetime.now()
                blank_result = InferenceResult(
                    request=InferenceRequest(
                        example_id=test_id,
                        source_file_path=str(getattr(test_case, "file_path", "") or ""),
                        product_type=ProductType.PARSE,
                    ),
                    pipeline_name=pipeline_name or self.output_dir.name,
                    product_type=ProductType.PARSE,
                    raw_output={"note": "no inference result; scored as blank output"},
                    output=ParseOutput(
                        example_id=test_id,
                        pipeline_name=pipeline_name or self.output_dir.name,
                        markdown="",
                    ),
                    started_at=now,
                    completed_at=now,
                    latency_in_ms=0,
                )
                if not parse_evaluator.can_evaluate(blank_result, test_case):
                    continue
                non_qa_evaluations.append((blank_result, test_case, parse_evaluator, False))
                synthesized += 1
            if synthesized:
                print(
                    f"⚠️  {synthesized} test case(s) had no inference result; scoring them as blank output (0.0)",
                    flush=True,
                )

        # Count QA test cases for progress indication
        qa_test_cases = len(qa_evaluation_tasks)
        total_to_evaluate = len(non_qa_evaluations) + qa_test_cases

        # Plain-text progress logging for CI/non-TTY environments.
        # The log_progress closure is called unconditionally at each evaluation
        # site; it no-ops when Rich progress bars are active.
        eval_done = 0

        if not use_rich:
            print("=== Evaluation Plan ===")
            print(f"  Result files found: {len(result_files)} | Skipped: {skipped}")
            print(
                f"  Documents to evaluate: {total_to_evaluate} ({len(non_qa_evaluations)} standard, {qa_test_cases} QA)"
            )
            print("=======================")

        def log_progress(test_id: str, status: str = "") -> None:
            """Log evaluation progress as plain text (no-op when Rich is active)."""
            nonlocal eval_done
            eval_done += 1
            if use_rich:
                return
            status_suffix = f": {status}" if status else ""
            print(
                f"  [{eval_done}/{total_to_evaluate}] {test_id}{status_suffix}",
                flush=True,
            )

        # Create progress bars if using Rich
        progress: Progress | None = None
        qa_task_id: int | None = None
        total_task_id: int | None = None

        if use_rich and console:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(
                    bar_width=None,
                    style="bright_blue",
                    complete_style="green",
                    finished_style="green",
                ),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("[cyan]{task.completed}/{task.total}"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
                expand=True,
            )
            if qa_test_cases > 0:
                qa_task_id = progress.add_task(
                    "[yellow]QA Evaluation (LLM calls)[/yellow]",
                    total=qa_test_cases,
                )
            total_task_id = progress.add_task(
                "[bold green]Total Evaluation[/bold green]",
                total=len(result_files),
            )
            progress.start()

        try:
            # Separate multi_task evaluations from parallelizable evaluations
            # Multi_task requires instance methods (_evaluate_multi_task, _evaluators)
            # that cannot be pickled and sent to worker processes
            multi_task_evaluations = [
                (inf, tc, ev, mode) for inf, tc, ev, mode in non_qa_evaluations if mode == "multi_task"
            ]
            parallelizable_evaluations = [
                (inf, tc, ev, mode) for inf, tc, ev, mode in non_qa_evaluations if mode != "multi_task"
            ]

            # Process multi_task evaluations in main process (cannot be parallelized)
            for inf_result, tc, _, _ in multi_task_evaluations:
                eval_result = self._evaluate_single(inf_result, tc, None, "multi_task")
                evaluation_results.append(eval_result)
                if eval_result.success:
                    successful += 1
                    log_progress(tc.test_id, "OK")
                elif _is_skipped_result(eval_result):
                    skipped += 1
                    log_progress(tc.test_id, "skipped (no layout data)")
                else:
                    failed += 1
                    log_progress(tc.test_id, "FAILED")
                if progress and total_task_id is not None:
                    progress.update(total_task_id, advance=1)  # type: ignore[arg-type]

            # Process non-QA evaluations in parallel using ProcessPoolExecutor
            # Default to CPU count, but cap at 8 for CI environments
            num_workers = max_workers or min(os.cpu_count() or 4, 8)

            if parallelizable_evaluations:
                # Prepare tasks for ProcessPoolExecutor
                # We need to serialize data since processes don't share memory
                worker_tasks: list[
                    tuple[
                        dict,
                        dict,
                        str,
                        str | bool,
                        str | None,
                        str,
                        bool,
                        bool,
                        bool,
                    ]
                ] = []
                for inf_result, tc, _eval_obj, mode in parallelizable_evaluations:
                    # Serialize inference result and test case to dicts
                    inf_dict = inf_result.model_dump()
                    tc_dict = tc.model_dump()

                    # Determine test case type
                    if isinstance(tc, ExtractTestCase):
                        tc_type = "extract"
                    elif isinstance(tc, LayoutDetectionTestCase):
                        tc_type = "layout_detection"
                    elif isinstance(tc, ParseTestCase):
                        tc_type = "parse"
                    else:
                        raise ValueError(f"Unknown test case type: {type(tc).__name__}")

                    # Determine evaluator type
                    if mode is True:  # cross_eval
                        eval_type = "layout_detection"
                    else:
                        eval_type = inf_result.product_type.value

                    worker_tasks.append(
                        (
                            inf_dict,
                            tc_dict,
                            tc_type,
                            mode,
                            eval_type,
                            self.layout_ontology,
                            self.enable_teds,
                            self.skip_rules,
                            self.verified_only,
                        )
                    )

                # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all tasks
                    futures = [executor.submit(_evaluate_single_worker, *task) for task in worker_tasks]

                    # Per-worker timeout: 8 minutes per evaluation
                    worker_timeout = 8 * 60

                    # Collect results as they complete
                    completed = 0
                    for future in as_completed(futures):
                        try:
                            result_dict = future.result(timeout=worker_timeout)
                            eval_result = EvaluationResult.model_validate(result_dict)
                            evaluation_results.append(eval_result)

                            if eval_result.success:
                                successful += 1
                                log_progress(eval_result.test_id, "OK")
                            elif _is_skipped_result(eval_result):
                                skipped += 1
                                log_progress(eval_result.test_id, "skipped (no layout data)")
                            else:
                                failed += 1
                                log_progress(eval_result.test_id, "FAILED")
                        except FuturesTimeoutError:
                            failed += 1
                            log_progress("unknown", f"FAILED (worker timed out after {worker_timeout}s)")
                        except Exception:
                            # Worker process error
                            failed += 1
                            log_progress("unknown", "FAILED (worker error)")

                        # Update progress (can't do this in worker due to separate processes)
                        completed += 1
                        if progress and total_task_id is not None:
                            progress.update(total_task_id, completed=completed)  # type: ignore[arg-type]

            # Process QA evaluations concurrently
            if qa_evaluation_tasks:
                qa_results, qa_success, qa_failed = asyncio.run(
                    self._run_qa_evaluations_async(
                        qa_evaluation_tasks,
                        progress,
                        qa_task_id,
                        total_task_id,
                        log_progress,
                    )
                )
                evaluation_results.extend(qa_results)
                successful += qa_success
                failed += qa_failed
        finally:
            # Stop progress display
            if progress:
                progress.stop()

        # Stamp tags from test cases onto evaluation results
        for result in evaluation_results:
            tc = test_cases_dict.get(result.test_id)  # type: ignore[assignment]
            if tc is not None:
                result.tags = list(tc.tags)

        # Penalize inference failures: synthesize zero-scored EvaluationResults
        # for test cases the pipeline tried but failed on (no .result.json).
        # Without this, errored cases would be silently dropped from the
        # aggregation denominator, masking pipeline reliability problems.
        if test_cases_dict:
            inference_errors = self._load_inference_errors(self.output_dir, pipeline_name=pipeline_name)
            evaluated_test_ids = {r.test_id for r in evaluation_results}
            synthesized_failures = self._synthesize_failure_results(
                inference_errors=inference_errors,
                evaluated_test_ids=evaluated_test_ids,
                test_cases_dict=test_cases_dict,
                successful_results=evaluation_results,
                pipeline_name=pipeline_name,
                product_type=product_type,
            )
            if synthesized_failures:
                evaluation_results.extend(synthesized_failures)
                failed += len(synthesized_failures)
                if verbose:
                    print(f"  Penalized {len(synthesized_failures)} inference failures with zero-scored results")

            # Penalize missing predictions: pipelines that never emitted a
            # ``*.result.json`` for a given test case AND have no entry in
            # ``_errors.json`` are otherwise silently dropped from the
            # denominator. Mirror the parse-eval semantics by injecting zero
            # scores for every such (pipeline × test_case) pair.
            missing = self._synthesize_missing_attempts(
                output_dir=self.output_dir,
                evaluation_results=evaluation_results,
                test_cases_dict=test_cases_dict,
                pipeline_name=pipeline_name,
                product_type=product_type,
            )
            if missing:
                evaluation_results.extend(missing)
                failed += len(missing)
                if verbose:
                    print(
                        f"  Penalized {len(missing)} missing predictions "
                        f"(pipeline produced no result.json) with zero-scored results"
                    )

        # Aggregate metrics
        aggregate_metrics = self._aggregate_metrics(evaluation_results)
        aggregate_diagnostic_metrics = self._aggregate_metrics(
            evaluation_results,
            metric_list_name="diagnostic_metrics",
        )

        # Aggregate operational stats
        aggregate_stats = self._aggregate_stats(evaluation_results)

        # Compute confusion matrix for layout detection evaluations
        confusion_matrix = None
        if product_type == "layout_detection" and test_cases_dict:
            layout_evaluator = self._evaluators.get("layout_detection")
            if isinstance(layout_evaluator, LayoutDetectionEvaluator):
                # Collect inference results into dict for confusion matrix computation
                inference_results_dict: dict[str, InferenceResult] = {}
                for result_file in result_files:
                    inference_result = self._load_inference_result(result_file)
                    if inference_result:
                        inference_results_dict[inference_result.request.example_id] = inference_result

                # Compute confusion matrix
                try:
                    confusion_matrix = layout_evaluator.compute_confusion_matrix(
                        inference_results=inference_results_dict,
                        test_cases=test_cases_dict,
                        iou_threshold=0.5,
                    )
                except Exception as e:
                    print(f"Warning: Failed to compute confusion matrix: {e}", file=sys.stderr)

        # Aggregate per-tag metrics
        tag_metrics = self._aggregate_tag_metrics(evaluation_results)

        return EvaluationSummary(
            total_examples=len(evaluation_results),
            successful=successful,
            failed=failed,
            skipped=skipped,
            aggregate_metrics=aggregate_metrics,
            aggregate_diagnostic_metrics=aggregate_diagnostic_metrics,
            per_example_results=evaluation_results,
            confusion_matrix=confusion_matrix,
            tag_metrics=tag_metrics,
            completed_at=None,  # Will be set by caller
            aggregate_stats=aggregate_stats,
        )

    def _find_pipeline_dirs(
        self,
        output_dir: Path,
        pipeline_name: str | None = None,
    ) -> list[Path]:
        """Discover top-level pipeline subdirectories under ``output_dir``.

        A directory is considered a pipeline directory when it contains either a
        ``_metadata.json`` (preferred indicator written by the inference runner)
        or at least one ``*.result.json`` file. The result is sorted by name and
        optionally filtered to a single pipeline.
        """
        if not output_dir.is_dir():
            return []
        candidates: list[Path] = []
        for child in sorted(output_dir.iterdir()):
            if not child.is_dir():
                continue
            if pipeline_name is not None and child.name != pipeline_name:
                continue
            if (child / "_metadata.json").exists():
                candidates.append(child)
                continue
            try:
                next(child.rglob("*.result.json"))
            except StopIteration:
                continue
            candidates.append(child)
        return candidates

    def _load_inference_errors(
        self,
        output_dir: Path,
        *,
        pipeline_name: str | None,
    ) -> list[dict[str, Any]]:
        """Load inference-time error entries from ``_errors.json`` files.

        Pipeline runners write a per-pipeline ``_errors.json`` at the pipeline's
        output root with one entry per test case that failed inference (no
        ``.result.json`` was produced). This method aggregates entries across
        every ``_errors.json`` under ``output_dir`` and tags each with the
        pipeline directory the file came from so downstream filtering by
        pipeline still works.
        """
        errors: list[dict[str, Any]] = []
        for errors_path in output_dir.rglob("_errors.json"):
            # `_errors.json` is always written at the pipeline's output root, so
            # the owning pipeline is the parent directory name. Compare it
            # exactly rather than as a substring: registry names are prefixes of
            # one another (`llamaextract_agentic` vs `llamaextract_agentic_plus`),
            # and a substring test would charge one pipeline's inference
            # failures against another's score.
            if pipeline_name is not None and errors_path.parent.name != pipeline_name:
                continue
            try:
                raw = json.loads(errors_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(raw, list):
                continue
            pipeline_dir_name = errors_path.parent.name
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                # Tag with the pipeline directory name so synthesis can recover
                # pipeline_name when the runner-level filter is unset.
                tagged = dict(entry)
                tagged.setdefault("_pipeline_dir", pipeline_dir_name)
                errors.append(tagged)
        return errors

    def _synthesize_failure_results(
        self,
        *,
        inference_errors: list[dict[str, Any]],
        evaluated_test_ids: set[str],
        test_cases_dict: dict[str, TestCase],
        successful_results: list[EvaluationResult],
        pipeline_name: str | None,
        product_type: str | None,
    ) -> list[EvaluationResult]:
        """Synthesize zero-scored results for inference failures (penalize).

        For every entry in ``inference_errors`` whose ``example_id`` matches a
        test case but has no successful evaluation result, emit an
        ``EvaluationResult`` with ``success=True`` and every metric set to
        ``0.0`` so the aggregator includes the failure in macro averages.
        Without this, pipelines that error out silently on a subset of cases
        get the same score as if those cases had never existed.

        Metric-name set is the union of metric names that appeared on
        successful results. Diagnostic metrics get the same treatment so any
        per-pipeline diagnostic averages also get penalized.
        """
        if not inference_errors:
            return []

        metric_names: set[str] = set()
        diagnostic_metric_names: set[str] = set()
        for r in successful_results:
            if not r.success:
                continue
            for m in r.metrics:
                if _eligible_for_failure_synthesis(m.metric_name):
                    metric_names.add(m.metric_name)
            for m in r.diagnostic_metrics:
                if _eligible_for_failure_synthesis(m.metric_name):
                    diagnostic_metric_names.add(m.metric_name)

        if not metric_names and not diagnostic_metric_names:
            # Total failure: nothing succeeded, so there are no observed metric
            # names to project zeros onto. Fall back to the always-emitted
            # headline metrics for the product type so a pipeline that fails on
            # every document scores 0.0 instead of vanishing from the report.
            metric_names = set(_FALLBACK_METRIC_NAMES.get(product_type or "", ()))
            if not metric_names:
                return []

        synthesized: list[EvaluationResult] = []
        for entry in inference_errors:
            example_id = entry.get("example_id")
            if not isinstance(example_id, str) or not example_id:
                continue
            test_case = self._match_example_id_to_test_case(example_id, test_cases_dict)
            if test_case is None:
                continue
            if test_case.test_id in evaluated_test_ids:
                continue
            error_msg = str(entry.get("error", "inference_failed"))
            failure_metadata = {"inference_failed": True, "error_type": entry.get("error_type")}
            pooled_denom = self._failure_pooled_denominator(test_case)
            synthesized.append(
                EvaluationResult(
                    test_id=test_case.test_id,
                    example_id=example_id,
                    pipeline_name=pipeline_name or entry.get("_pipeline_dir") or "",
                    product_type=product_type or "",
                    success=True,
                    metrics=[
                        MetricValue(
                            metric_name=name,
                            value=0.0,
                            metadata=_build_failure_metric_metadata(name, failure_metadata, pooled_denom),
                        )
                        for name in sorted(metric_names)
                    ],
                    diagnostic_metrics=[
                        MetricValue(
                            metric_name=name,
                            value=0.0,
                            metadata=_build_failure_metric_metadata(name, failure_metadata, pooled_denom),
                        )
                        for name in sorted(diagnostic_metric_names)
                    ],
                    error=f"Inference failed: {error_msg}",
                    job_id=entry.get("job_id"),
                    tags=list(test_case.tags),
                )
            )
        return synthesized

    def _failure_pooled_denominator(self, test_case: TestCase) -> int:
        """Return the per-rule denominator used to project an inference-failed
        result as ``0/N`` in pooled (micro) aggregations.

        Returns 0 for product types that don't have a natural per-rule pooling
        — in that case the failure still contributes to macro as 0.0 (existing
        behaviour) but not to micro.
        """
        if isinstance(test_case, ExtractTestCase):
            return compute_failure_field_denominator(test_case.get_extract_field_rules())
        return 0

    def _synthesize_missing_attempts(
        self,
        *,
        output_dir: Path,
        evaluation_results: list[EvaluationResult],
        test_cases_dict: dict[str, TestCase],
        pipeline_name: str | None,
        product_type: str | None,
    ) -> list[EvaluationResult]:
        """Synthesize zero-scored results for (pipeline × test_case) pairs that
        produced no prediction at all (no ``*.result.json`` and no entry in any
        ``_errors.json``). Mirrors the parse-eval semantics: missing predictions
        are penalized as zeros so pipelines that silently skip a subset of
        documents don't appear better than pipelines that emit (wrong) outputs.

        Only pipelines that have at least one successful result for this
        product_type are zero-filled; otherwise an unrelated pipeline (wrong
        product type, never run) would be spuriously injected.
        """
        if not test_cases_dict:
            return []
        pipeline_dirs = self._find_pipeline_dirs(output_dir, pipeline_name)
        if not pipeline_dirs:
            return []

        accounted: set[tuple[str, str]] = {(r.pipeline_name, r.test_id) for r in evaluation_results}

        per_pipeline_metrics: dict[str, set[str]] = {}
        per_pipeline_diag_metrics: dict[str, set[str]] = {}
        for r in evaluation_results:
            if not r.success:
                continue
            per_pipeline_metrics.setdefault(r.pipeline_name, set()).update(m.metric_name for m in r.metrics)
            per_pipeline_diag_metrics.setdefault(r.pipeline_name, set()).update(
                m.metric_name for m in r.diagnostic_metrics
            )

        synthesized: list[EvaluationResult] = []
        failure_metadata = {"missing_prediction": True}
        for pipeline_dir in pipeline_dirs:
            pipe = pipeline_dir.name
            metric_names = per_pipeline_metrics.get(pipe, set())
            diag_names = per_pipeline_diag_metrics.get(pipe, set())
            if not metric_names and not diag_names:
                # Pipeline produced nothing usable for this product_type. When
                # the user named this pipeline explicitly we know it was meant
                # to run here, so zero-fill it from the product's headline
                # metrics (total-failure case). Without an explicit name we
                # cannot tell "failed completely" from "is a different product
                # type that lives in the same output dir", so we skip instead
                # of inventing zeros for an unrelated pipeline.
                if pipeline_name is None:
                    continue
                metric_names = set(_FALLBACK_METRIC_NAMES.get(product_type or "", ()))
                if not metric_names:
                    continue
            for tc in test_cases_dict.values():
                if (pipe, tc.test_id) in accounted:
                    continue
                synthesized.append(
                    EvaluationResult(
                        test_id=tc.test_id,
                        example_id=tc.test_id,
                        pipeline_name=pipe,
                        product_type=product_type or "",
                        success=True,
                        metrics=[
                            MetricValue(metric_name=name, value=0.0, metadata=dict(failure_metadata))
                            for name in sorted(metric_names)
                        ],
                        diagnostic_metrics=[
                            MetricValue(metric_name=name, value=0.0, metadata=dict(failure_metadata))
                            for name in sorted(diag_names)
                        ],
                        error="No prediction emitted",
                        tags=list(tc.tags),
                    )
                )
        return synthesized

    def _match_example_id_to_test_case(
        self,
        example_id: str,
        test_cases_dict: dict[str, TestCase],
    ) -> TestCase | None:
        """Best-effort match of an ``_errors.json`` example_id to a test case.

        Mirrors the matching used for successful results: tries direct test_id
        match first, then NFC-normalized comparisons, then suffix match
        (``group/<example_id>``) for the common layout where example_id is the
        bare stem.
        """
        if example_id in test_cases_dict:
            return test_cases_dict[example_id]
        example_id_nfc = unicodedata.normalize("NFC", example_id)
        suffix = f"/{example_id_nfc}"
        for test_id, test_case in test_cases_dict.items():
            test_id_nfc = unicodedata.normalize("NFC", test_id)
            if test_id_nfc == example_id_nfc or test_id_nfc.endswith(suffix):
                return test_case
        return None

    def _aggregate_metrics(
        self,
        evaluation_results: list[EvaluationResult],
        *,
        metric_list_name: str = "metrics",
    ) -> dict[str, float]:
        """
        Aggregate metrics across all evaluation results.

        :param evaluation_results: List of individual evaluation results
        :param metric_list_name: EvaluationResult metric list attribute to aggregate
        :return: Dictionary of aggregated metric values
        """
        if not evaluation_results:
            return {}

        # Collect all metric values by metric name
        metric_values: dict[str, list[float]] = {}
        # Also collect counts from metadata (for rules, etc.)
        metric_counts: dict[str, list[tuple[int, int]]] = {}  # (passed, total) pairs
        metric_verified_counts: dict[str, list[tuple[int, int]]] = {}  # (verified_passed, verified_total) pairs
        metric_prf_counts: dict[str, list[tuple[int, int, int]]] = {}  # (tp, fp, fn) triples
        metric_score_sums: dict[str, list[tuple[float, float]]] = {}  # (score_sum, score_count)
        weighted_metric_values: dict[str, list[tuple[float, float]]] = {}  # (weighted_value, weight)
        # Track scores where tables were predicted (for _predicted aggregates)
        # Applies to any metric with "tables_predicted" metadata (TEDS, GriTS, etc.)
        predicted_values: dict[str, list[float]] = {}
        metric_count_sums: dict[str, list[int]] = {}  # count totals

        # Count-as-zero: examples that errored out (genuine failures, not
        # legitimate skips or harness errors) should drag the score down, not
        # vanish from the denominator. Without this, a provider that can't
        # produce the required output for most of a group (e.g. a markdown tool
        # against bbox layout-detection cases) is scored only over its
        # surviving examples, badly inflating the headline. Failures are
        # counted per product type so that e.g. a layout-detection failure
        # only pads layout metrics, never parse/QA metrics observed in the
        # same group.
        failed_by_product: dict[str, int] = {}
        for r in evaluation_results:
            if r.success or _is_skipped_result(r) or _is_infra_failure(r):
                continue
            failed_by_product[r.product_type] = failed_by_product.get(r.product_type, 0) + 1

        # Which product types produced each metric (drives padding scope).
        metric_product_types: dict[str, set[str]] = {}

        for result in evaluation_results:
            if not result.success:
                continue
            result_metrics = result.diagnostic_metrics if metric_list_name == "diagnostic_metrics" else result.metrics
            for metric in result_metrics:
                if metric.metric_name not in metric_values:
                    metric_values[metric.metric_name] = []
                metric_values[metric.metric_name].append(metric.value)
                metric_product_types.setdefault(metric.metric_name, set()).add(result.product_type)

                # Track scores where tables were predicted and expected
                if metric.metadata and metric.metadata.get("tables_predicted", False):
                    key = f"{metric.metric_name}_predicted"
                    if key not in predicted_values:
                        predicted_values[key] = []
                    predicted_values[key].append(metric.value)

                # Extract counts from metadata if available
                if metric.metadata and "passed" in metric.metadata and "total" in metric.metadata:
                    if metric.metric_name not in metric_counts:
                        metric_counts[metric.metric_name] = []
                    passed = metric.metadata.get("passed", 0)
                    total = metric.metadata.get("total", 0)
                    if isinstance(passed, int) and isinstance(total, int):
                        metric_counts[metric.metric_name].append((passed, total))

                if metric.metadata and "verified_passed" in metric.metadata and "verified_total" in metric.metadata:
                    if metric.metric_name not in metric_verified_counts:
                        metric_verified_counts[metric.metric_name] = []
                    verified_passed = metric.metadata.get("verified_passed", 0)
                    verified_total = metric.metadata.get("verified_total", 0)
                    if isinstance(verified_passed, int) and isinstance(verified_total, int):
                        metric_verified_counts[metric.metric_name].append((verified_passed, verified_total))

                if metric.metadata and "count" in metric.metadata:
                    count = metric.metadata.get("count")
                    if isinstance(count, int):
                        if metric.metric_name not in metric_count_sums:
                            metric_count_sums[metric.metric_name] = []
                        metric_count_sums[metric.metric_name].append(count)

                if metric.metadata and {"tp", "fp", "fn"}.issubset(metric.metadata):
                    tp = metric.metadata.get("tp")
                    fp = metric.metadata.get("fp")
                    fn = metric.metadata.get("fn")
                    if isinstance(tp, int) and isinstance(fp, int) and isinstance(fn, int):
                        if metric.metric_name not in metric_prf_counts:
                            metric_prf_counts[metric.metric_name] = []
                        metric_prf_counts[metric.metric_name].append((tp, fp, fn))

                if metric.metadata and "score_sum" in metric.metadata and "score_count" in metric.metadata:
                    score_sum = metric.metadata.get("score_sum")
                    score_count = metric.metadata.get("score_count")
                    if (
                        isinstance(score_sum, (int, float))
                        and not isinstance(score_sum, bool)
                        and isinstance(score_count, (int, float))
                        and not isinstance(score_count, bool)
                        and score_count > 0
                    ):
                        metric_score_sums.setdefault(metric.metric_name, []).append(
                            (float(score_sum), float(score_count))
                        )

                if metric.metadata and metric.metric_name == "parse_field_text_similarity":
                    string_rule_count = metric.metadata.get("string_rule_count")
                    if (
                        isinstance(string_rule_count, (int, float))
                        and not isinstance(string_rule_count, bool)
                        and string_rule_count > 0
                    ):
                        weighted_metric_values.setdefault(metric.metric_name, []).append(
                            (metric.value * float(string_rule_count), float(string_rule_count))
                        )

        # Pad score metrics with a 0 per genuine failure of the same product
        # type, so the average is taken over all attempted examples, not just
        # the survivors. Padding is restricted to metrics that look like
        # higher-is-better scores: observed values all in [0, 1], not a known
        # diagnostic count or lower-is-better rate (_NON_SCORE_METRICS), and
        # not tracked as a count total (metric_count_sums). This is a no-op
        # for groups with no failures. Note the micro_* aggregates below
        # remain survivor-only by design: failures carry no rule totals or
        # tp/fp/fn counts to pool, so only avg_* (macro) reflects failures.
        if failed_by_product:
            for metric_name, values in metric_values.items():
                if metric_name in metric_count_sums or metric_name in _NON_SCORE_METRICS:
                    continue
                if not values or not all(0.0 <= v <= 1.0 for v in values):
                    continue
                pad = sum(failed_by_product.get(pt, 0) for pt in metric_product_types.get(metric_name, ()))
                if pad:
                    values.extend([0.0] * pad)

        # Compute averages
        aggregate: dict[str, float] = {}
        for metric_name, values in metric_values.items():
            if values:
                aggregate[f"avg_{metric_name}"] = sum(values) / len(values)
                aggregate[f"min_{metric_name}"] = min(values)
                aggregate[f"max_{metric_name}"] = max(values)

        # Aggregate counts for metrics that have them
        for metric_name, count_pairs in metric_counts.items():
            total_passed = sum(passed for passed, _ in count_pairs)
            total_rules = sum(total for _, total in count_pairs)
            if total_rules > 0:
                aggregate[f"total_{metric_name}_passed"] = float(total_passed)
                aggregate[f"total_{metric_name}_evaluated"] = float(total_rules)
                aggregate[f"micro_{metric_name}"] = total_passed / total_rules

        for metric_name, count_pairs in metric_verified_counts.items():
            total_passed = sum(passed for passed, _ in count_pairs)
            total_rules = sum(total for _, total in count_pairs)
            if total_rules > 0:
                aggregate[f"total_verified_{metric_name}_passed"] = float(total_passed)
                aggregate[f"total_verified_{metric_name}_evaluated"] = float(total_rules)
                aggregate[f"micro_verified_{metric_name}"] = total_passed / total_rules

        # Aggregate precision/recall/F1-style metrics from TP/FP/FN metadata.
        # Keep avg_* as the existing document-weighted macro average and expose
        # pooled counters through explicit micro_* metrics.
        for metric_name, count_triples in metric_prf_counts.items():
            tp = sum(item[0] for item in count_triples)
            fp = sum(item[1] for item in count_triples)
            fn = sum(item[2] for item in count_triples)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if metric_name == "precision" or metric_name.endswith("_precision"):
                aggregate[f"micro_{metric_name}"] = precision
            elif metric_name == "recall" or (
                metric_name.endswith("_recall") and metric_name != "record_grounded_recall"
            ):
                aggregate[f"micro_{metric_name}"] = recall
            elif metric_name == "f1" or metric_name.endswith("_f1"):
                aggregate[f"micro_{metric_name}"] = f1
            elif metric_name.endswith("accuracy"):
                total = tp + fp + fn
                if total > 0:
                    aggregate[f"micro_{metric_name}"] = tp / total
            aggregate[f"total_{metric_name}_tp"] = float(tp)
            aggregate[f"total_{metric_name}_fp"] = float(fp)
            aggregate[f"total_{metric_name}_fn"] = float(fn)
            if metric_name.endswith("pass_rate") and metric_name not in metric_counts:
                total = tp + fp + fn
                if total > 0:
                    aggregate[f"micro_{metric_name}"] = tp / total

        for metric_name, score_pairs in metric_score_sums.items():
            score_sum = sum(item[0] for item in score_pairs)
            score_count = sum(item[1] for item in score_pairs)
            if score_count > 0:
                aggregate[f"micro_{metric_name}"] = score_sum / score_count

        for metric_name, weighted_values in weighted_metric_values.items():
            weighted_sum = sum(item[0] for item in weighted_values)
            weight_sum = sum(item[1] for item in weighted_values)
            if weight_sum > 0:
                aggregate[f"micro_{metric_name}"] = weighted_sum / weight_sum

        # Add _predicted aggregates (only docs where tables were predicted)
        for key, values in predicted_values.items():
            if values:
                aggregate[f"avg_{key}"] = sum(values) / len(values)
                aggregate[f"min_{key}"] = min(values)
                aggregate[f"max_{key}"] = max(values)

        # Aggregate explicit count totals (e.g., unmatched elements)
        for metric_name, counts in metric_count_sums.items():
            aggregate[f"total_{metric_name}"] = float(sum(counts))

        return aggregate

    def _aggregate_tag_metrics(self, evaluation_results: list[EvaluationResult]) -> dict[str, dict[str, float]]:
        """
        Aggregate metrics grouped by tag.

        Groups results by tag, then calls _aggregate_metrics for each group.
        Adds example_count to each tag's metrics.

        :param evaluation_results: List of individual evaluation results
        :return: Dict keyed by tag name, each value containing aggregated metrics
        """
        from collections import defaultdict

        tag_groups: dict[str, list[EvaluationResult]] = defaultdict(list)
        for result in evaluation_results:
            for tag in result.tags:
                tag_groups[tag].append(result)

        tag_metrics: dict[str, dict[str, float]] = {}
        for tag, results in sorted(tag_groups.items()):
            metrics = self._aggregate_metrics(results)
            metrics["example_count"] = float(len(results))
            tag_metrics[tag] = metrics

        return tag_metrics

    def _aggregate_stats(self, evaluation_results: list[EvaluationResult]) -> dict[str, dict[str, Any]]:
        """
        Aggregate operational stats across all evaluation results.

        Collects values by stat name from successful results and computes
        total, avg, min, max, p50, p95, p99, count for each.

        :param evaluation_results: List of individual evaluation results
        :return: Dict keyed by stat name, each value containing aggregates + unit
        """
        # Collect values and units by stat name
        stat_values: dict[str, list[float]] = {}
        stat_units: dict[str, str] = {}
        for r in evaluation_results:
            if not r.success:
                continue
            for s in r.stats:
                stat_values.setdefault(s.name, []).append(s.value)
                stat_units[s.name] = s.unit

        aggregate: dict[str, dict[str, Any]] = {}
        for name, values in stat_values.items():
            values_sorted = sorted(values)
            n = len(values_sorted)

            def percentile(p: int, n: int = n, values_sorted: list[float] = values_sorted) -> float:
                idx = int(n * p / 100)
                return values_sorted[min(idx, n - 1)]

            aggregate[name] = {
                "total": sum(values),
                "avg": sum(values) / n,
                "min": min(values),
                "max": max(values),
                "p50": percentile(50),
                "p90": percentile(90),
                "p95": percentile(95),
                "p99": percentile(99),
                "count": n,
                "unit": stat_units[name],
            }

        return aggregate

    def _has_mixed_rules(self, test_case: TestCase) -> bool:
        """
        Check if test case has both layout and non-layout rules.

        :param test_case: Test case to check
        :return: True if test case has mixed rule types
        """
        # Get test_rules from the test case
        rules = []
        if isinstance(test_case, (ParseTestCase, LayoutDetectionTestCase)):
            rules = list(test_case.test_rules or [])

        if not rules:
            return False

        has_layout = any(get_rule_type(r) == "layout" for r in rules)
        has_non_layout = any(get_rule_type(r) is not None and get_rule_type(r) != "layout" for r in rules)

        return has_layout and has_non_layout

    def _is_llamaparse_output(self, inference_result: InferenceResult) -> bool:
        """
        Check if inference result exposes parse-capable normalized output.

        This is used to determine if multi-task evaluation can be performed
        even when product_type is LAYOUT_DETECTION.

        :param inference_result: Inference result to check
        :return: True if output is LlamaParse format with pages and markdown
        """
        from extract_bench.schemas.layout_detection_output import LayoutOutput
        from extract_bench.schemas.parse_output import ParseOutput

        if isinstance(inference_result.output, ParseOutput):
            if inference_result.output.layout_pages:
                return True
            return len(inference_result.output.pages) > 0

        # Layout detection outputs can still carry full document markdown
        # (e.g., normalized from LlamaParse layout runs).
        if isinstance(inference_result.output, LayoutOutput):
            if inference_result.output.markdown.strip():
                return True
        return False

    def _create_parse_output_from_raw(self, inference_result: InferenceResult) -> ParseOutput | None:
        """
        Create a ParseOutput from normalized inference output.

        Used in multi-task evaluation to create a synthetic PARSE output when
        the original product_type was LAYOUT_DETECTION and markdown is present.

        :param inference_result: Inference result
        :return: ParseOutput or None if conversion fails
        """
        from extract_bench.schemas.layout_detection_output import LayoutOutput
        from extract_bench.schemas.parse_output import PageIR, ParseOutput

        if isinstance(inference_result.output, ParseOutput):
            return inference_result.output

        # For layout runs that still provide markdown, synthesize minimal
        # ParseOutput so parse/order rules can be evaluated in multi-task mode.
        if isinstance(inference_result.output, LayoutOutput):
            markdown = inference_result.output.markdown
            if isinstance(markdown, str) and markdown.strip():
                return ParseOutput(
                    example_id=inference_result.request.example_id,
                    pipeline_name=inference_result.pipeline_name,
                    pages=[PageIR(page_index=0, markdown=markdown)],
                    markdown=markdown,
                )
        return None

    def _evaluate_multi_task(
        self,
        inference_result: InferenceResult,
        test_case: LayoutDetectionTestCase | ParseTestCase,
    ) -> EvaluationResult:
        """
        Evaluate mixed rule types by splitting rules and running appropriate evaluators.

        For test cases with mixed rules (table, order, layout, etc.):
        1. Split rules into layout vs non-layout
        2. Evaluate non-layout rules with ParseEvaluator
        3. Evaluate layout rules with LayoutDetectionEvaluator (cross-eval from PARSE)
        4. Merge metrics into single result

        :param inference_result: The inference result to evaluate
        :param test_case: Test case with mixed rule types
        :return: Combined evaluation result with metrics from both evaluators
        """

        # Get all rules from the test case
        all_rules = test_case.test_rules or []

        # Split rules by type
        layout_rules = [r for r in all_rules if get_rule_type(r) == "layout"]
        parse_rules = [r for r in all_rules if get_rule_type(r) != "layout"]

        all_metrics: list[MetricValue] = []
        errors: list[str] = []

        # Evaluate parse rules (table, order, present, absent, etc.)
        if parse_rules:
            temp_parse_test_case = ParseTestCase(
                test_id=test_case.test_id,
                group=test_case.group,
                file_path=test_case.file_path,
                test_rules=parse_rules,  # type: ignore[arg-type]
                expected_markdown=None,
            )

            # Create a synthetic PARSE inference result if needed
            # This allows ParseEvaluator to work even when the original
            # product_type was LAYOUT_DETECTION (auto-detected from test cases)
            parse_inference_result = inference_result
            if inference_result.product_type != ProductType.PARSE:
                parse_output = self._create_parse_output_from_raw(inference_result)
                if parse_output:
                    parse_inference_result = InferenceResult(
                        request=inference_result.request,
                        pipeline_name=inference_result.pipeline_name,
                        product_type=ProductType.PARSE,
                        raw_output=inference_result.raw_output,
                        output=parse_output,
                        started_at=inference_result.started_at,
                        completed_at=inference_result.completed_at,
                        latency_in_ms=inference_result.latency_in_ms,
                    )

            parse_evaluator = self._evaluators.get("parse")
            can_eval = (
                parse_evaluator.can_evaluate(parse_inference_result, temp_parse_test_case) if parse_evaluator else False
            )
            if parse_evaluator and can_eval:
                try:
                    parse_result = parse_evaluator.evaluate(parse_inference_result, temp_parse_test_case)
                    all_metrics.extend(parse_result.metrics)
                except Exception as e:
                    errors.append(f"Parse evaluation error: {e}")

        # Evaluate layout rules (cross-evaluation from PARSE output)
        if layout_rules:
            metadata = test_case.metadata if isinstance(test_case, LayoutDetectionTestCase) else None
            # For multi-page documents, layout rules may span multiple pages
            # Create test case with all layout rules (page_index=0 as default)
            temp_layout_test_case = LayoutDetectionTestCase(
                test_id=test_case.test_id,
                group=test_case.group,
                file_path=test_case.file_path,
                test_rules=layout_rules,
                source_dataset=metadata.get("source_dataset") if metadata else None,
                # Not used for multi-page; GT filtering is done by
                # get_layout_annotations.
                page_index=0,
                metadata=metadata,
            )

            adapter = create_layout_adapter_for_result(inference_result)
            layout_output = adapter.to_layout_output(inference_result)

            if layout_output.predictions:
                layout_evaluator = self._evaluators.get("layout_detection")
                if layout_evaluator:
                    try:
                        # Create synthetic inference result with layout output
                        layout_inference_result = InferenceResult(
                            request=inference_result.request,
                            pipeline_name=inference_result.pipeline_name,
                            product_type=ProductType.LAYOUT_DETECTION,
                            raw_output=inference_result.raw_output,
                            output=layout_output,
                            started_at=inference_result.started_at,
                            completed_at=inference_result.completed_at,
                            latency_in_ms=inference_result.latency_in_ms,
                        )
                        layout_result = layout_evaluator.evaluate(layout_inference_result, temp_layout_test_case)
                        all_metrics.extend(layout_result.metrics)
                    except Exception as e:
                        errors.append(f"Layout evaluation error: {e}")
            else:
                errors.append("Could not extract layout from PARSE output")

        stats = build_operational_stats(inference_result)

        return EvaluationResult(
            test_id=test_case.test_id,
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            product_type=inference_result.product_type.value,
            success=len(errors) == 0,
            metrics=all_metrics,
            error="; ".join(errors) if errors else None,
            stats=stats,
        )

    async def _evaluate_qa_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        evaluator: QAEvaluator,
        inference_result: InferenceResult,
        test_case: ParseTestCase,
        progress: Progress | None,
        qa_task_id: int | None,
        total_task_id: int | None,
        log_progress: Callable[[str, str], None] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a QA test case with semaphore-based concurrency control.

        :param semaphore: Semaphore for concurrency control
        :param evaluator: QA evaluator instance
        :param inference_result: The inference result to evaluate
        :param test_case: The test case with qa_config
        :param progress: Rich progress bar (optional)
        :param qa_task_id: QA progress task ID (optional)
        :param total_task_id: Total progress task ID (optional)
        :param log_progress: Plain-text progress callback (optional)
        :return: Evaluation result
        """
        async with semaphore:
            # Update progress description
            if progress and qa_task_id is not None:
                progress.update(
                    qa_task_id,  # type: ignore[arg-type]
                    description=f"[yellow]QA Evaluation: {test_case.test_id}[/yellow]",
                )

            # Run evaluation in thread (LLM calls are synchronous)
            try:
                eval_result = await asyncio.to_thread(evaluator.evaluate, inference_result, test_case)
            except Exception as e:
                # Handle evaluation errors
                eval_result = EvaluationResult(
                    test_id=test_case.test_id,
                    example_id=inference_result.request.example_id,
                    pipeline_name=inference_result.pipeline_name,
                    product_type=inference_result.product_type.value,
                    success=False,
                    error=f"Evaluation error: {str(e)}",
                )

            # Update progress after evaluation
            if log_progress:
                status = "OK" if eval_result.success else "FAILED"
                log_progress(test_case.test_id, f"QA {status}")
            if progress:
                if qa_task_id is not None:
                    progress.update(qa_task_id, advance=1)  # type: ignore[arg-type]
                if total_task_id is not None:
                    progress.update(total_task_id, advance=1)  # type: ignore[arg-type]

            return eval_result

    async def _run_qa_evaluations_async(
        self,
        qa_evaluation_tasks: list[tuple[InferenceResult, ParseTestCase, QAEvaluator]],
        progress: Progress | None,
        qa_task_id: int | None,
        total_task_id: int | None,
        log_progress: Callable[[str, str], None] | None = None,
    ) -> tuple[list[EvaluationResult], int, int]:
        """
        Run QA evaluations concurrently with semaphore-based concurrency control.

        :param qa_evaluation_tasks: List of (inference_result, test_case, evaluator) tuples
        :param progress: Rich progress bar (optional)
        :param qa_task_id: QA progress task ID (optional)
        :param total_task_id: Total progress task ID (optional)
        :param log_progress: Plain-text progress callback (optional)
        :return: Tuple of (results list, success count, failed count)
        """
        # Create semaphore for QA concurrency control (fixed at 20)
        max_concurrent_qa = 20
        semaphore = asyncio.Semaphore(max_concurrent_qa)

        # Create async tasks for QA evaluations
        qa_tasks = [
            self._evaluate_qa_with_semaphore(
                semaphore,
                evaluator,
                inference_result,
                test_case,
                progress,
                qa_task_id,
                total_task_id,
                log_progress,
            )
            for inference_result, test_case, evaluator in qa_evaluation_tasks
        ]

        # Run QA evaluations concurrently
        qa_results = await asyncio.gather(*qa_tasks, return_exceptions=True)

        # Process results
        results: list[EvaluationResult] = []
        success_count = 0
        failed_count = 0

        for result in qa_results:
            if isinstance(result, Exception):
                failed_count += 1
                # Create error result - we don't have test_case info here
                # This shouldn't happen, but handle it gracefully
                results.append(
                    EvaluationResult(
                        test_id="unknown",
                        example_id="unknown",
                        pipeline_name="unknown",
                        product_type="parse",
                        success=False,
                        error=f"Task execution error: {str(result)}",
                    )
                )
            else:
                results.append(result)  # type: ignore[arg-type]
                if result.success:  # type: ignore[union-attr]
                    success_count += 1
                else:
                    failed_count += 1

        return results, success_count, failed_count
