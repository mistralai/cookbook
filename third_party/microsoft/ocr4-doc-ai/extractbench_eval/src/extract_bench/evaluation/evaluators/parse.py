"""Evaluator for PARSE product type."""

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from extract_bench.evaluation.evaluators.base import BaseEvaluator
from extract_bench.evaluation.metrics.field_grounding.parse_adapter import (
    compute_parse_field_grounding_metrics,
)
from extract_bench.evaluation.metrics.field_grounding.rule_filters import (
    filter_extract_field_rules,
)
from extract_bench.evaluation.metrics.parse.grits_metric import (
    GriTSMetric,
)
from extract_bench.evaluation.metrics.parse.header_accuracy_metric import (
    HeaderAccuracyMetric,
    HeaderAccuracyMetricGenerous,
)
from extract_bench.evaluation.metrics.parse.rule_based_judge_metric import (
    RuleBasedJudgeMetric as RuleBasedMetric,
)
from extract_bench.evaluation.metrics.parse.structural_consistency_metric import (
    StructuralConsistencyMetric,
)
from extract_bench.evaluation.metrics.parse.table_extraction import (
    ExtractedTable,
    extract_html_tables,
    extract_table_pairs,
)
from extract_bench.evaluation.metrics.parse.table_parsing import (
    merge_preceding_titles_into_tables,
)
from extract_bench.evaluation.metrics.parse.table_record_match_metric import (
    TableRecordMatchMetric,
)
from extract_bench.evaluation.metrics.parse.table_splitting import (
    split_ambiguous_merged_pred,
)
from extract_bench.evaluation.metrics.parse.table_title_stripping import (
    strip_title_rows,
)
from extract_bench.evaluation.metrics.parse.teds_metric import (
    TEDS_CONTENT,
    TEDSMetric,
)
from extract_bench.evaluation.metrics.parse.text_similarity_metric import (
    TextSimilarityMetric,
)
from extract_bench.evaluation.stats import build_operational_stats
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue
from extract_bench.schemas.parse_output import ParseOutput
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.schema import ExtractTestCase, ParseTestCase, TestCase


def _has_html_tables(content: str) -> bool:
    """Check if content contains HTML tables."""
    return "<table" in content.lower()


logger = logging.getLogger(__name__)


def _has_extract_field_bboxes(test_case: ExtractTestCase) -> bool:
    return any(rule.bboxes for rule in test_case.get_extract_field_rules())


# ---------------------------------------------------------------------------
# Module-level helpers for parallel table metric computation
# (must be top-level functions so ProcessPoolExecutor can pickle them)
# ---------------------------------------------------------------------------


def _compute_teds_standalone(expected: str, actual: str, variants: set[str] | None = None) -> list[MetricValue]:
    """Compute TEDS metrics in a worker process."""
    return TEDSMetric(variants=variants).compute(expected=expected, actual=actual)


def _compute_grits_standalone(
    expected_tables: list[ExtractedTable],
    actual_tables: list[ExtractedTable],
) -> list[MetricValue]:
    """Compute GriTS metrics in a worker process."""
    return GriTSMetric().compute(expected_tables, actual_tables)


def _compute_table_metrics_parallel(
    expected: str,
    actual: str,
    expected_tables: list[ExtractedTable],
    actual_tables: list[ExtractedTable],
    teds_variants: set[str] | None = None,
) -> tuple[list[MetricValue], list[MetricValue]]:
    """Run TEDS and GriTS in parallel via separate processes.

    TEDS still operates on raw markdown (unchanged). GriTS receives the
    pre-extracted ExtractedTable lists from the shared stage.
    """
    with ProcessPoolExecutor(max_workers=2) as pool:
        teds_future = pool.submit(_compute_teds_standalone, expected, actual, teds_variants)
        grits_future = pool.submit(_compute_grits_standalone, expected_tables, actual_tables)
        return teds_future.result(), grits_future.result()


class ParseEvaluator(BaseEvaluator):
    """
    Evaluator for the PARSE product type.

    Supports four evaluation modes:
    1. Rule-based: Execute test rules against markdown output
    2. Ground truth: Compare markdown against expected_markdown using text similarity
    3. TEDS: Compare HTML tables using Tree Edit Distance based Similarity
    4. GriTS: Compare HTML tables using Grid Table Similarity (topology + content)
    """

    def __init__(
        self,
        enable_rule_based: bool = True,
        enable_text_similarity: bool = False,
        enable_teds: bool = False,
        enable_grits: bool = True,
        enable_header_accuracy: bool = False,
        enable_structural_consistency: bool = True,
        enable_table_record_match: bool = True,
        enable_table_composite: bool = False,
        teds_variants: set[str] | None = None,
    ):
        """
        Initialize the ParseEvaluator.

        :param enable_rule_based: Enable rule-based metric evaluation (default: True)
        :param enable_text_similarity: Enable text similarity metric (default: False).
            Disabled by default because exact/fuzzy text matching is not what we should
            optimize for - we care more about semantic match. This metric uses Levenshtein
            distance which measures character-level differences rather than meaning.
        :param enable_teds: Enable TEDS metric evaluation (default: False)
        :param enable_grits: Enable GriTS metric evaluation (default: True)
        :param enable_header_accuracy: Enable header accuracy metric (default: False)
        :param enable_structural_consistency: Enable structural consistency metric (default: True)
        :param teds_variants: Set of TEDS variant names to compute. Defaults to
            {TEDS_CONTENT} (standard TEDS only). Use ALL_TEDS_VARIANTS for all.
        """
        self._enable_rule_based = enable_rule_based
        self._enable_text_similarity = enable_text_similarity
        self._enable_teds = enable_teds
        self._enable_grits = enable_grits
        self._enable_header_accuracy = enable_header_accuracy
        self._enable_structural_consistency = enable_structural_consistency
        self._enable_table_record_match = enable_table_record_match
        self._enable_table_composite = enable_table_composite
        self._rule_metric = RuleBasedMetric()
        self._text_similarity_metric = TextSimilarityMetric()
        self._teds_metric = TEDSMetric(variants=teds_variants if teds_variants is not None else {TEDS_CONTENT})
        self._grits_metric = GriTSMetric()
        self._header_accuracy_metric = HeaderAccuracyMetric()
        self._header_accuracy_generous_metric = HeaderAccuracyMetricGenerous()
        self._structural_consistency_metric = StructuralConsistencyMetric()
        self._table_record_match_metric = TableRecordMatchMetric()
        # Reference implementation for comparison — remove before deploying.
        # Set to None to disable, or swap GriTSMetric() above with
        # ReferenceGriTSMetric() to use the reference as the primary.
        self._ref_grits_metric = None  # ReferenceGriTSMetric()

    def can_evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> bool:
        """
        Check if this evaluator can evaluate the given inference result and test case.

        Requires:
        - ProductType.PARSE
        - inference_result.output is a ParseOutput instance
        - test_case is a ParseTestCase with either test_rules or expected_markdown,
          or an ExtractTestCase with extract_field bbox rules.
        """
        if inference_result.product_type != ProductType.PARSE:
            return False

        if not isinstance(inference_result.output, ParseOutput):
            return False

        if isinstance(test_case, ExtractTestCase):
            return _has_extract_field_bboxes(test_case)

        if not isinstance(test_case, ParseTestCase):
            return False

        # Exclude QA test cases (handled by QAEvaluator)
        if test_case.qa_config is not None:
            return False

        # Need either test rules or expected markdown
        has_test_rules = test_case.test_rules is not None and len(test_case.test_rules) > 0
        has_expected_markdown = test_case.expected_markdown is not None

        return has_test_rules or has_expected_markdown

    def evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> EvaluationResult:
        """
        Evaluate a PARSE inference result against a test case.

        :param inference_result: The inference result to evaluate
        :param test_case: The test case with test rules or expected markdown
        :return: Evaluation result with metrics
        :raises ValueError: If test case is invalid or missing required data
        """
        if not self.can_evaluate(inference_result, test_case):
            raise ValueError("Cannot evaluate: missing test_rules or expected_markdown, or invalid product type")

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("Inference result output is not ParseOutput")

        if isinstance(test_case, ExtractTestCase):
            return self._evaluate_extract_field_grounding(inference_result, test_case)

        if not isinstance(test_case, ParseTestCase):
            raise ValueError("Test case must be ParseTestCase or ExtractTestCase for PARSE evaluation")

        metrics: list[MetricValue] = []

        # Rule-based evaluation
        if self._enable_rule_based:
            if not test_case.test_rules:
                logger.debug(
                    f"Skipping rule-based metric: test_rules not provided "
                    f"(test_id: {test_case.test_id}, "
                    f"example_id: {inference_result.request.example_id})"
                )
            else:
                # Get markdown content for the appropriate page(s)
                # For now, use document-level markdown
                # TODO: Support per-page rule execution
                markdown_content = inference_result.output.markdown

                # Execute rules
                rule_result = self._rule_metric.compute(
                    expected=test_case.test_rules,  # type: ignore[arg-type]
                    actual=markdown_content,
                    page=None,  # Document-level for now
                    raw_output=inference_result.raw_output,
                    parse_output=inference_result.output,
                )
                metrics.append(rule_result)
                if "judge_pass_rate" in rule_result.metadata:
                    metrics.append(
                        MetricValue(
                            metric_name="rule_pass_rate_judge",
                            value=rule_result.metadata["judge_pass_rate"],
                            metadata={
                                "passed": rule_result.metadata["judge_passed"],
                                "total": rule_result.metadata["total"],
                            },
                        )
                    )

                # Add per-rule-type breakdown if available
                if rule_result.metadata and "rule_results" in rule_result.metadata:
                    rule_results = rule_result.metadata["rule_results"]
                    # Group by rule type
                    rule_types: dict[str, list[dict[str, Any]]] = {}
                    for result in rule_results:
                        rule_type = result.get("type", "unknown")
                        if rule_type not in rule_types:
                            rule_types[rule_type] = []
                        rule_types[rule_type].append(result)

                    # Calculate per-type pass rates using graduated scores
                    per_type_avg: dict[str, float] = {}
                    for rule_type, type_results in rule_types.items():
                        total = len(type_results)
                        score_sum = sum(r.get("score", 1.0 if r.get("passed", False) else 0.0) for r in type_results)
                        pass_rate = score_sum / total if total > 0 else 0.0
                        per_type_avg[rule_type] = pass_rate
                        metrics.append(
                            MetricValue(
                                metric_name=f"rule_{rule_type}_pass_rate",
                                value=pass_rate,
                                metadata={
                                    "score_sum": score_sum,
                                    "total": total,
                                    "rule_type": rule_type,
                                },
                            )
                        )

                    # Per-angle breakdown for rotate_check rules
                    if "rotate_check" in rule_types:
                        angle_groups: dict[str, list[dict[str, Any]]] = {}
                        for r in rule_types["rotate_check"]:
                            angle = r.get("expected_angle")
                            if angle is not None:
                                key = f"{int(angle)}deg"
                                if key not in angle_groups:
                                    angle_groups[key] = []
                                angle_groups[key].append(r)

                        angle_pass_rates: dict[str, float] = {}
                        for angle_key, angle_results in angle_groups.items():
                            angle_total = len(angle_results)
                            angle_score_sum = sum(
                                r.get("score", 1.0 if r.get("passed", False) else 0.0) for r in angle_results
                            )
                            angle_pr = angle_score_sum / angle_total if angle_total > 0 else 0.0
                            angle_pass_rates[angle_key] = angle_pr
                            metrics.append(
                                MetricValue(
                                    metric_name=f"rule_rotate_check_{angle_key}_pass_rate",
                                    value=angle_pr,
                                    metadata={
                                        "score_sum": angle_score_sum,
                                        "total": angle_total,
                                        "angle": angle_key,
                                    },
                                )
                            )

                        # Normalized rotate pass rate: 0deg weighted 10x
                        pr_0 = angle_pass_rates.get("0deg", 0.0)
                        pr_90 = angle_pass_rates.get("90deg", 0.0)
                        pr_180 = angle_pass_rates.get("180deg", 0.0)
                        pr_270 = angle_pass_rates.get("270deg", 0.0)
                        has_any = any(k in angle_pass_rates for k in ("0deg", "90deg", "180deg", "270deg"))
                        if has_any:
                            normalized_rotate = (pr_0 * 10 + pr_90 + pr_180 + pr_270) / 13
                            metrics.append(
                                MetricValue(
                                    metric_name="rule_rotate_check_normalized_pass_rate",
                                    value=normalized_rotate,
                                    metadata={
                                        "0deg_pass_rate": pr_0,
                                        "90deg_pass_rate": pr_90,
                                        "180deg_pass_rate": pr_180,
                                        "270deg_pass_rate": pr_270,
                                        "formula": "(0deg*10 + 90deg + 180deg + 270deg) / 13",
                                    },
                                )
                            )

                    # Normalized category scores: avg of per-type averages
                    # to reduce impact of docs with many rules of one type.
                    # Text styling: bold, strikeout, sup, sub pairs.
                    # A pair is included only if the positive rule exists for this doc.
                    _TEXT_STYLING_PAIRS = [
                        ("is_bold", "is_not_bold"),
                        ("is_strikeout", "is_not_strikeout"),
                        ("is_sup", "is_not_sup"),
                        ("is_sub", "is_not_sub"),
                    ]
                    _TEXT_STYLING_POS_TYPES: set[str] = set()
                    _TEXT_STYLING_NEG_TYPES: set[str] = set()
                    for pos, neg in _TEXT_STYLING_PAIRS:
                        if pos in per_type_avg:
                            _TEXT_STYLING_POS_TYPES.add(pos)
                        if neg in per_type_avg:
                            _TEXT_STYLING_NEG_TYPES.add(neg)
                    _TEXT_STYLING_TYPES = _TEXT_STYLING_POS_TYPES | _TEXT_STYLING_NEG_TYPES
                    _TEXT_CORRECTNESS_TYPES = {
                        "missing_word_percent",
                        "unexpected_word_percent",
                        "too_many_word_occurence_percent",
                        "missing_sentence_percent",
                        "unexpected_sentence_percent",
                        "too_many_sentence_occurence_percent",
                        "extra_content",
                        "bag_of_digit_percent",
                    }
                    _ORDER_TYPES = {"order"}
                    _TITLE_TYPES = {"is_title", "title_hierarchy_percent"}
                    _CODE_BLOCK_TYPES = {"is_code_block"}
                    _LATEX_TYPES = {"is_latex"}

                    _NORMALIZED_CATEGORIES: dict[str, set[str]] = {
                        "normalized_text_styling": _TEXT_STYLING_TYPES,
                        "normalized_text_correctness": _TEXT_CORRECTNESS_TYPES,
                        "normalized_order": _ORDER_TYPES,
                        "normalized_title_accuracy": _TITLE_TYPES,
                        "normalized_code_block": _CODE_BLOCK_TYPES,
                        "normalized_latex": _LATEX_TYPES,
                    }

                    _cat_values: dict[str, float] = {}
                    for metric_name, type_set in _NORMALIZED_CATEGORIES.items():
                        if metric_name == "normalized_text_styling":
                            # Combine positive and negative pass rates using a weighted
                            # harmonic mean (F_β-score) with β=0.5 so that negative-rule
                            # failures (false styling) are penalised more heavily than
                            # missed styling.
                            pos_rules = [r for r in rule_results if r.get("type") in _TEXT_STYLING_POS_TYPES]
                            neg_rules = [r for r in rule_results if r.get("type") in _TEXT_STYLING_NEG_TYPES]
                            if pos_rules or neg_rules:

                                def _rule_score(r: dict[str, object]) -> float:
                                    s = r.get("score")
                                    if isinstance(s, (int, float)):
                                        return float(s)
                                    return 1.0 if r.get("passed", False) else 0.0

                                pos_score = (
                                    sum(_rule_score(r) for r in pos_rules) / len(pos_rules) if pos_rules else 1.0
                                )
                                neg_score = (
                                    sum(_rule_score(r) for r in neg_rules) / len(neg_rules) if neg_rules else 1.0
                                )
                                beta = 0.5
                                if pos_score + neg_score > 0:
                                    cat_value = (
                                        (1 + beta**2) * pos_score * neg_score / (beta**2 * pos_score + neg_score)
                                    )
                                else:
                                    cat_value = 0.0
                                _cat_values[metric_name] = cat_value
                                metrics.append(
                                    MetricValue(
                                        metric_name=metric_name,
                                        value=cat_value,
                                        metadata={
                                            "num_pos_rules": len(pos_rules),
                                            "num_neg_rules": len(neg_rules),
                                            "pos_score": pos_score,
                                            "neg_score": neg_score,
                                            "included_types": sorted(type_set & set(per_type_avg)),
                                            "per_type_scores": {
                                                t: per_type_avg[t] for t in type_set if t in per_type_avg
                                            },
                                        },
                                    )
                                )
                        else:
                            cat_scores = [per_type_avg[t] for t in type_set if t in per_type_avg]
                            if cat_scores:
                                cat_value = sum(cat_scores) / len(cat_scores)
                                _cat_values[metric_name] = cat_value
                                metrics.append(
                                    MetricValue(
                                        metric_name=metric_name,
                                        value=cat_value,
                                        metadata={
                                            "num_rule_types": len(cat_scores),
                                            "per_type_scores": {
                                                t: per_type_avg[t] for t in type_set if t in per_type_avg
                                            },
                                        },
                                    )
                                )

                    # Combined weighted metric across all normalized categories.
                    # Full-weight (1.0): text_correctness, text_styling, order, title_accuracy
                    # Reduced-weight (1/5): latex, code_block
                    # Denominator = 4 + 1/5 + 1/5 = 4.4
                    _COMBINED_WEIGHTS: dict[str, float] = {
                        "normalized_text_correctness": 1.0,
                        "normalized_text_styling": 1.0,
                        "normalized_order": 1.0,
                        "normalized_title_accuracy": 1.0,
                        "normalized_latex": 1.0 / 5.0,
                        "normalized_code_block": 1.0 / 5.0,
                    }
                    weighted_sum = 0.0
                    weight_sum = 0.0
                    present_categories: dict[str, float] = {}
                    for cat_name, weight in _COMBINED_WEIGHTS.items():
                        if cat_name in _cat_values:
                            weighted_sum += _cat_values[cat_name] * weight
                            weight_sum += weight
                            present_categories[cat_name] = _cat_values[cat_name]
                    if weight_sum > 0:
                        combined_value = weighted_sum / weight_sum
                        metrics.append(
                            MetricValue(
                                metric_name="normalized_text_score",
                                value=combined_value,
                                metadata={
                                    "weights": {k: v for k, v in _COMBINED_WEIGHTS.items() if k in present_categories},
                                    "category_scores": present_categories,
                                    "weight_sum": weight_sum,
                                },
                            )
                        )

                    # Content Faithfulness: is the right content there, in the right order?
                    # Correctness (hallucination/omission) at full weight, order at half weight.
                    _FAITHFULNESS_WEIGHTS: dict[str, float] = {
                        "normalized_text_correctness": 1.0,
                        "normalized_order": 0.5,
                    }
                    faith_weighted_sum = 0.0
                    faith_weight_sum = 0.0
                    faith_categories: dict[str, float] = {}
                    for cat_name, weight in _FAITHFULNESS_WEIGHTS.items():
                        if cat_name in _cat_values:
                            faith_weighted_sum += _cat_values[cat_name] * weight
                            faith_weight_sum += weight
                            faith_categories[cat_name] = _cat_values[cat_name]
                    if faith_weight_sum > 0:
                        faith_value = faith_weighted_sum / faith_weight_sum
                        metrics.append(
                            MetricValue(
                                metric_name="content_faithfulness",
                                value=faith_value,
                                metadata={
                                    "weights": {
                                        k: v for k, v in _FAITHFULNESS_WEIGHTS.items() if k in faith_categories
                                    },
                                    "category_scores": faith_categories,
                                    "weight_sum": faith_weight_sum,
                                },
                            )
                        )

                    # Semantic Formatting: is the meaningful markup preserved?
                    # Styling and titles at full weight, latex and code blocks at 1/5.
                    _FORMATTING_WEIGHTS: dict[str, float] = {
                        "normalized_text_styling": 1.0,
                        "normalized_title_accuracy": 1.0,
                        "normalized_latex": 1.0 / 5.0,
                        "normalized_code_block": 1.0 / 5.0,
                    }
                    fmt_weighted_sum = 0.0
                    fmt_weight_sum = 0.0
                    fmt_categories: dict[str, float] = {}
                    for cat_name, weight in _FORMATTING_WEIGHTS.items():
                        if cat_name in _cat_values:
                            fmt_weighted_sum += _cat_values[cat_name] * weight
                            fmt_weight_sum += weight
                            fmt_categories[cat_name] = _cat_values[cat_name]
                    if fmt_weight_sum > 0:
                        fmt_value = fmt_weighted_sum / fmt_weight_sum
                        metrics.append(
                            MetricValue(
                                metric_name="semantic_formatting",
                                value=fmt_value,
                                metadata={
                                    "weights": {k: v for k, v in _FORMATTING_WEIGHTS.items() if k in fmt_categories},
                                    "category_scores": fmt_categories,
                                    "weight_sum": fmt_weight_sum,
                                },
                            )
                        )

        # Ground truth evaluation
        if test_case.expected_markdown:
            actual_markdown = inference_result.output.markdown

            # Text similarity metric
            if self._enable_text_similarity:
                similarity_result = self._text_similarity_metric.compute(
                    expected=test_case.expected_markdown,
                    actual=actual_markdown,
                )
                metrics.append(similarity_result)

            # Table similarity metrics (TEDS and GriTS)
            # Normalize predicted tables: merge preceding titles into tables
            # when GT has full-width colspan title rows
            actual_for_tables = merge_preceding_titles_into_tables(test_case.expected_markdown, actual_markdown)

            # Check for HTML tables once (used by both TEDS and GriTS)
            has_expected_tables = _has_html_tables(test_case.expected_markdown)
            has_actual_tables = _has_html_tables(actual_for_tables)

            if has_expected_tables:
                if has_actual_tables:
                    # Both sides have tables — compute table metrics
                    metrics.extend(
                        self._compute_table_similarity_metrics(
                            test_case.expected_markdown,
                            actual_for_tables,
                            allow_splitting_ambiguous_merged_tables=test_case.allow_splitting_ambiguous_merged_tables,
                            trm_unsupported=test_case.trm_unsupported,
                            max_top_title_rows=test_case.max_top_title_rows,
                        )
                    )
                else:
                    # Expected has tables but actual doesn't — score 0.0
                    if self._enable_teds:
                        for variant in sorted(self._teds_metric.variants):
                            metrics.append(
                                MetricValue(
                                    metric_name=variant,
                                    value=0.0,
                                    metadata={
                                        "tables_predicted": False,
                                        "tables_found_expected": 1,
                                        "tables_found_actual": 0,
                                    },
                                )
                            )
                        logger.debug(
                            f"TEDS=0.0: no tables in actual "
                            f"(test_id: {test_case.test_id}, "
                            f"example_id: {inference_result.request.example_id})"
                        )
                    if self._enable_grits:
                        no_table_meta = {
                            "tables_predicted": False,
                            "tables_found_expected": 1,
                            "tables_found_actual": 0,
                        }
                        metrics.append(
                            MetricValue(
                                metric_name="grits_con",
                                value=0.0,
                                metadata=no_table_meta,
                            )
                        )
                        # Also emit 0.0 for reference metrics so they
                        # aggregate over the same denominator
                        if self._ref_grits_metric is not None:
                            metrics.append(
                                MetricValue(
                                    metric_name="ref_grits_top",
                                    value=0.0,
                                    metadata=no_table_meta,
                                )
                            )
                            metrics.append(
                                MetricValue(
                                    metric_name="ref_grits_con",
                                    value=0.0,
                                    metadata=no_table_meta,
                                )
                            )
                        logger.debug(
                            f"GriTS=0.0: no tables in actual "
                            f"(test_id: {test_case.test_id}, "
                            f"example_id: {inference_result.request.example_id})"
                        )
                    if self._enable_header_accuracy:
                        no_table_header_meta = {
                            "tables_predicted": False,
                            "tables_found_expected": 1,
                            "tables_found_actual": 0,
                        }
                        metrics.append(
                            MetricValue(
                                metric_name="header_composite_v3",
                                value=0.0,
                                metadata=no_table_header_meta,
                            )
                        )
                        metrics.append(
                            MetricValue(
                                metric_name="exp_header_composite_v3_generous",
                                value=0.0,
                                metadata=no_table_header_meta,
                            )
                        )
                    if self._enable_table_composite:
                        # Emit table_composite_v3=0 when all three components are present
                        if self._enable_grits and self._enable_header_accuracy and self._enable_structural_consistency:
                            no_table_composite_meta = {
                                "tables_predicted": False,
                                "tables_found_expected": 1,
                                "tables_found_actual": 0,
                            }
                            metrics.append(
                                MetricValue(
                                    metric_name="table_composite_v3",
                                    value=0.0,
                                    metadata=no_table_composite_meta,
                                )
                            )
                            metrics.append(
                                MetricValue(
                                    metric_name="exp_table_composite_v3_generous",
                                    value=0.0,
                                    metadata=no_table_composite_meta,
                                )
                            )
                            metrics.append(
                                MetricValue(
                                    metric_name="exp_table_composite_v3_generous_harmonic",
                                    value=0.0,
                                    metadata=no_table_composite_meta,
                                )
                            )
                    if self._enable_table_record_match:
                        metrics.extend(
                            self._table_record_match_metric.compute(
                                expected=test_case.expected_markdown,
                                actual=actual_for_tables,
                            )
                        )
                    if self._enable_grits and self._enable_table_record_match:
                        metrics.extend(
                            self._compute_grits_trm_composite(
                                existing_metrics=metrics,
                                trm_unsupported=test_case.trm_unsupported,
                            )
                        )
            else:
                if self._enable_teds:
                    logger.debug(
                        f"Skipping TEDS: no tables in expected "
                        f"(test_id: {test_case.test_id}, "
                        f"example_id: {inference_result.request.example_id})"
                    )
                if self._enable_grits:
                    logger.debug(
                        f"Skipping GriTS: no tables in expected "
                        f"(test_id: {test_case.test_id}, "
                        f"example_id: {inference_result.request.example_id})"
                    )
        else:
            # expected_markdown is missing, log if metrics that require it are enabled
            if self._enable_text_similarity:
                logger.debug(
                    f"Skipping text similarity metric: expected_markdown not provided "
                    f"(test_id: {test_case.test_id}, "
                    f"example_id: {inference_result.request.example_id})"
                )
            if self._enable_teds:
                logger.debug(
                    f"Skipping TEDS metric: expected_markdown not provided "
                    f"(test_id: {test_case.test_id}, "
                    f"example_id: {inference_result.request.example_id})"
                )
            if self._enable_grits:
                logger.debug(
                    f"Skipping GriTS metric: expected_markdown not provided "
                    f"(test_id: {test_case.test_id}, "
                    f"example_id: {inference_result.request.example_id})"
                )

        stats = build_operational_stats(inference_result)

        return EvaluationResult(
            test_id=test_case.test_id,
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            product_type=inference_result.product_type.value,
            success=True,
            metrics=metrics,
            stats=stats,
        )

    def _evaluate_extract_field_grounding(
        self,
        inference_result: InferenceResult,
        test_case: ExtractTestCase,
    ) -> EvaluationResult:
        """Evaluate parse output against extract_field rules, emitting parse_field_* metrics."""
        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("Inference result output is not ParseOutput")

        all_extract_field_rules = test_case.get_extract_field_rules()
        extract_field_rules = filter_extract_field_rules(
            all_extract_field_rules,
            require_bboxes=True,
        )
        metrics = compute_parse_field_grounding_metrics(
            inference_result=inference_result,
            field_rules=extract_field_rules,
            data_schema=test_case.data_schema,
        )
        stats = build_operational_stats(inference_result)
        return EvaluationResult(
            test_id=test_case.test_id,
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            product_type=inference_result.product_type.value,
            success=True,
            metrics=metrics,
            stats=stats,
        )

    # Type alias for alignment maps: {gt_row/col: pred_row/col}
    TableAlignment = dict[int, int]

    @staticmethod
    def _extract_table_pairs_from_grits(
        grits_results: list[MetricValue],
        expected: str,
        actual: str,
    ) -> tuple[list[tuple[str, str]], list[tuple[dict[int, int], dict[int, int]]]] | None:
        """Extract matched table pairs and alignment from GriTS results metadata.

        Returns (pairs, alignments) where:
          pairs = [(gt_html, pred_html), ...]
          alignments = [(row_map, col_map), ...]
        Or None if matching cannot be recovered.
        """
        expected_tables = extract_html_tables(expected)
        actual_tables = extract_html_tables(actual)
        if not expected_tables or not actual_tables:
            return None

        # Find per_table_details from any GriTS result
        details = None
        for r in grits_results:
            if r.metadata and "per_table_details" in r.metadata:
                details = r.metadata["per_table_details"]
                break
        if details is None:
            return None

        pairs: list[tuple[str, str]] = []
        alignments: list[tuple[dict[int, int], dict[int, int]]] = []
        for entry in details:
            gi = entry.get("gt_table_index")
            pi = entry.get("pred_table_index")
            row_align = entry.get("_con_row_alignment", {})
            col_align = entry.get("_con_col_alignment", {})
            if gi is None:
                continue
            if pi is not None and pi < len(actual_tables) and gi < len(expected_tables):
                pairs.append((expected_tables[gi], actual_tables[pi]))
                alignments.append((row_align, col_align))
            elif gi < len(expected_tables):
                # Unmatched GT table
                pairs.append((expected_tables[gi], ""))
                alignments.append(({}, {}))

        return (pairs, alignments) if pairs else None

    @staticmethod
    def _compute_grits_trm_composite(
        existing_metrics: list[MetricValue],
        *,
        trm_unsupported: bool,
    ) -> list[MetricValue]:
        """Emit grits_trm_composite = 0.5*grits_con + 0.5*trm, or grits_con
        alone when trm_unsupported is True or TRM is missing."""
        grits_con: float | None = None
        trm: float | None = None
        for r in existing_metrics:
            if r.metric_name == "grits_con":
                grits_con = r.value
            elif r.metric_name == "table_record_match":
                trm = r.value
        if grits_con is None:
            return []

        if trm_unsupported or trm is None:
            reason = "trm_unsupported" if trm_unsupported else "trm_missing"
            return [
                MetricValue(
                    metric_name="grits_trm_composite",
                    value=grits_con,
                    metadata={
                        "fallback": "grits_only",
                        "reason": reason,
                        "grits_con": grits_con,
                        "trm": trm,
                    },
                    details=[
                        f"{grits_con:.3f} = grits_con (fallback: {reason}; "
                        f"raw table_record_match shown separately may differ)",
                    ],
                )
            ]

        value = 0.5 * grits_con + 0.5 * trm
        return [
            MetricValue(
                metric_name="grits_trm_composite",
                value=value,
                metadata={
                    "grits_con": grits_con,
                    "trm": trm,
                    "fallback": None,
                },
                details=[
                    f"{value:.3f} = 0.5 × grits_con({grits_con:.3f}) + 0.5 × trm({trm:.3f})",
                ],
            )
        ]

    def _compute_table_similarity_metrics(
        self,
        expected: str,
        actual: str,
        *,
        allow_splitting_ambiguous_merged_tables: bool = False,
        trm_unsupported: bool = False,
        max_top_title_rows: int = 1,
    ) -> list[MetricValue]:
        """Compute enabled table similarity metrics.

        Runs TEDS and GriTS in parallel (separate processes) when both are
        enabled, since they are independent CPU-bound computations.
        Falls back to sequential execution if parallel dispatch fails or
        only one metric is enabled.
        """
        grits_results: list[MetricValue] = []

        # Shared table extraction stage. Run once per (expected, actual) so
        # that GriTS and TRM provably consume the same set of tables, paired
        # the same way. GT parse failures raise (dataset bug); pred parse
        # failures are dropped silently.
        # GriTS must run before TableRecordMatch — TRM consumes GriTS's pairing.
        expected_tables, actual_tables, table_counts = extract_table_pairs(expected, actual)

        # Lifted ambiguous-merged-table splitter. Runs once per doc before
        # GriTS/TEDS/TRM dispatch so GriTS sees the split sub-tables rather
        # than the merged blob. TEDS still sees the merged markdown because
        # it operates on raw markdown, not on the extracted table list —
        # that asymmetry is intentional.
        if allow_splitting_ambiguous_merged_tables:
            actual_tables, _ = split_ambiguous_merged_pred(expected_tables, actual_tables)

        # Title-row stripping: detect leading <td> title rows and top <th>
        # spanning titles, physically remove them from each table's grid,
        # and attach precomputed header hints. Runs once per doc, after
        # splitting and before any metric (GriTS/TEDS/TRM) consumes the
        # tables, so all metrics see the same trimmed grid.
        expected_tables = [strip_title_rows(et, max_top_title_rows=max_top_title_rows) for et in expected_tables]
        actual_tables = [strip_title_rows(et, max_top_title_rows=max_top_title_rows) for et in actual_tables]

        count_metrics: list[MetricValue] = [
            MetricValue(metric_name="tables_expected", value=float(table_counts.expected)),
            MetricValue(metric_name="tables_actual", value=float(table_counts.actual)),
            MetricValue(metric_name="tables_unparseable_pred", value=float(table_counts.unparseable_pred)),
        ]

        def _pairing_count_metrics(pairing: list[tuple[int, int | None]]) -> list[MetricValue]:
            paired_pred = {p for _, p in pairing if p is not None}
            tables_paired = len(paired_pred)
            unmatched_expected = sum(1 for _, p in pairing if p is None)
            unmatched_pred = max(0, len(actual_tables) - tables_paired)
            return [
                MetricValue(metric_name="tables_paired", value=float(tables_paired)),
                MetricValue(metric_name="tables_unmatched_expected", value=float(unmatched_expected)),
                MetricValue(metric_name="tables_unmatched_pred", value=float(unmatched_pred)),
            ]

        def _extract_pairing(grits_metrics: list[MetricValue]) -> list[tuple[int, int | None]]:
            for r in grits_metrics:
                if r.metadata and "pairing" in r.metadata:
                    return list(r.metadata["pairing"])
            return [(i, None) for i in range(len(expected_tables))]

        if self._enable_teds and self._enable_grits and self._ref_grits_metric is None:
            try:
                teds_results, grits_results = _compute_table_metrics_parallel(
                    expected,
                    actual,
                    expected_tables,
                    actual_tables,
                    teds_variants=self._teds_metric.variants,
                )
                results: list[MetricValue] = list(teds_results)
                for r in grits_results:
                    if r.metadata is None:
                        r.metadata = {}
                    r.metadata["tables_predicted"] = True
                results.extend(grits_results)
                results.extend(self._compute_header_and_consistency_metrics(expected, actual, grits_results))
                if self._enable_table_composite:
                    results.extend(self._compute_table_composite(results))
                pairing = _extract_pairing(grits_results)
                if self._enable_table_record_match:
                    results.extend(
                        self._table_record_match_metric.compute_extracted(
                            expected_tables,
                            actual_tables,
                            pairing=pairing,
                        )
                    )
                if self._enable_grits and self._enable_table_record_match:
                    results.extend(
                        self._compute_grits_trm_composite(
                            existing_metrics=results,
                            trm_unsupported=trm_unsupported,
                        )
                    )
                results.extend(count_metrics)
                results.extend(_pairing_count_metrics(pairing))
                return results
            except Exception as exc:
                logger.warning(
                    "Parallel table metric computation failed (%s), falling back to sequential",
                    exc,
                )

        # Sequential fallback (or only one metric enabled, or ref_grits active)
        results: list[MetricValue] = []  # type: ignore[no-redef]
        if self._enable_teds:
            results.extend(self._teds_metric.compute(expected=expected, actual=actual))
        if self._enable_grits:
            grits_results = self._grits_metric.compute(expected_tables, actual_tables)
            for r in grits_results:
                if r.metadata is None:
                    r.metadata = {}
                r.metadata["tables_predicted"] = True
            results.extend(grits_results)
            # Reference implementation for comparison
            if self._ref_grits_metric is not None:
                ref_results = self._ref_grits_metric.compute(expected=expected, actual=actual)
                for r in ref_results:
                    if r.metadata is None:
                        r.metadata = {}
                    r.metadata["tables_predicted"] = True
                results.extend(ref_results)
        results.extend(self._compute_header_and_consistency_metrics(expected, actual, grits_results))
        if self._enable_table_composite:
            results.extend(self._compute_table_composite(results))
        pairing = _extract_pairing(grits_results)
        if self._enable_table_record_match:
            results.extend(
                self._table_record_match_metric.compute_extracted(
                    expected_tables,
                    actual_tables,
                    pairing=pairing,
                )
            )
        if self._enable_grits and self._enable_table_record_match:
            results.extend(
                self._compute_grits_trm_composite(
                    existing_metrics=results,
                    trm_unsupported=trm_unsupported,
                )
            )
        results.extend(count_metrics)
        results.extend(_pairing_count_metrics(pairing))
        return results

    @staticmethod
    def _compute_table_composite(
        all_results: list[MetricValue],
    ) -> list[MetricValue]:
        """Compute table_composite_v3 as product of grits_con and header_composite_v3."""
        metric_map: dict[str, float] = {}
        for r in all_results:
            if r.metric_name in (
                "grits_con",
                "header_composite_v3",
                "exp_header_composite_v3_generous",
            ):
                metric_map[r.metric_name] = r.value

        out: list[MetricValue] = []

        # --- base composite (existing) ---
        grits_con = metric_map.get("grits_con")
        header_comp = metric_map.get("header_composite_v3")

        if grits_con is not None and header_comp is not None:
            composite = grits_con * header_comp
            harmonic = (
                (2 * grits_con * header_comp) / (grits_con + header_comp) if (grits_con + header_comp) > 0 else 0.0
            )
            out.append(
                MetricValue(
                    metric_name="table_composite_v3",
                    value=composite,
                    metadata={
                        "grits_con": grits_con,
                        "header_composite_v3": header_comp,
                        "tables_predicted": True,
                    },
                    details=[
                        f"{composite:.3f} = grits_con({grits_con:.3f}) × header_composite_v3({header_comp:.3f})",
                    ],
                )
            )
            out.append(
                MetricValue(
                    metric_name="table_composite_v3_harmonic",
                    value=harmonic,
                    metadata={
                        "grits_con": grits_con,
                        "header_composite_v3": header_comp,
                        "tables_predicted": True,
                    },
                    details=[
                        f"{harmonic:.3f} = harmonic_mean("
                        f"grits_con({grits_con:.3f}), header_composite_v3({header_comp:.3f}))",
                    ],
                )
            )

        # --- generous composite (new) ---
        header_gen = metric_map.get("exp_header_composite_v3_generous")

        if grits_con is not None and header_gen is not None:
            composite_gen = grits_con * header_gen
            harmonic_gen = (
                (2 * grits_con * header_gen) / (grits_con + header_gen) if (grits_con + header_gen) > 0 else 0.0
            )
            out.append(
                MetricValue(
                    metric_name="exp_table_composite_v3_generous",
                    value=composite_gen,
                    metadata={
                        "grits_con": grits_con,
                        "exp_header_composite_v3_generous": header_gen,
                        "tables_predicted": True,
                    },
                    details=[
                        f"{composite_gen:.3f} = grits_con({grits_con:.3f})"
                        f" × exp_header_composite_v3_generous({header_gen:.3f})",
                    ],
                )
            )
            out.append(
                MetricValue(
                    metric_name="exp_table_composite_v3_generous_harmonic",
                    value=harmonic_gen,
                    metadata={
                        "grits_con": grits_con,
                        "exp_header_composite_v3_generous": header_gen,
                        "tables_predicted": True,
                    },
                    details=[
                        f"{harmonic_gen:.3f} = harmonic_mean("
                        f"grits_con({grits_con:.3f}), exp_header_composite_v3_generous({header_gen:.3f}))",
                    ],
                )
            )

        return out

    def _compute_header_and_consistency_metrics(
        self,
        expected: str,
        actual: str,
        grits_results: list[MetricValue] | None = None,
    ) -> list[MetricValue]:
        """Compute header accuracy and structural consistency metrics."""
        results: list[MetricValue] = []
        if self._enable_header_accuracy:
            # Try to reuse GriTS table matching for header accuracy
            table_pairs: list[tuple[str, str]] | None = None
            table_alignments: list[tuple[dict[int, int], dict[int, int]]] | None = None
            if grits_results:
                extracted = self._extract_table_pairs_from_grits(grits_results, expected, actual)
                if extracted is not None:
                    table_pairs, table_alignments = extracted
            header_results = self._header_accuracy_metric.compute(
                expected=expected,
                actual=actual,
                table_pairs=table_pairs,
                table_alignments=table_alignments,
            )
            results.extend(header_results)
            results.extend(
                self._header_accuracy_generous_metric.compute(
                    expected=expected,
                    actual=actual,
                    table_pairs=table_pairs,
                    table_alignments=table_alignments,
                )
            )
        if self._enable_structural_consistency:
            consistency_results = self._structural_consistency_metric.compute(
                expected=expected,
                actual=actual,
            )
            results.extend(consistency_results)
        return results
