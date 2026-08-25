"""Evaluator for QA (question-answering) product type."""

import logging

from anls_star import anls_score

from extract_bench.evaluation.evaluators.base import BaseEvaluator
from extract_bench.evaluation.metrics.qa.answer_comparison import (
    AnswerComparisonMetric,
)
from extract_bench.evaluation.qa.llm_service import QALLMService
from extract_bench.evaluation.stats import build_operational_stats
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue
from extract_bench.schemas.parse_output import ParseOutput
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.schema import ParseTestCase, TestCase

logger = logging.getLogger(__name__)


def _split_comma_list(text: str) -> str | list[str]:
    """Split comma-delimited text into a list, or return as-is if no commas."""
    if "," in text:
        return [s.strip() for s in text.split(",")]
    return text


class QAEvaluator(BaseEvaluator):
    """
    Evaluator for question-answering evaluation.

    Uses parse markdown output as context to answer questions via LLM,
    then compares predicted answers with expected answers.
    """

    def __init__(
        self,
        llm_service: QALLMService | None = None,
        enable_qa: bool = True,
    ):
        """
        Initialize the QA evaluator.

        :param llm_service: Optional QALLMService instance (creates default if None)
        :param enable_qa: Enable QA evaluation (default: True)
        """
        self._enable_qa = enable_qa
        self._llm_service = llm_service
        self._answer_metric = AnswerComparisonMetric()

    def can_evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> bool:
        """
        Check if this evaluator can evaluate the given inference result and test case.

        Requires:
        - ProductType.PARSE
        - inference_result.output is a ParseOutput instance
        - test_case is a ParseTestCase with qa_config (not None)
        """
        if not self._enable_qa:
            return False

        if inference_result.product_type != ProductType.PARSE:
            return False

        if not isinstance(inference_result.output, ParseOutput):
            return False

        if not isinstance(test_case, ParseTestCase):
            return False

        # Need qa_config to be present
        return test_case.qa_config is not None

    def evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> EvaluationResult:
        """
        Evaluate a QA inference result against a test case.

        :param inference_result: The inference result to evaluate
        :param test_case: The test case with qa_config
        :return: Evaluation result with metrics
        :raises ValueError: If test case is invalid or missing required data
        """
        if not self.can_evaluate(inference_result, test_case):
            raise ValueError("Cannot evaluate: missing qa_config or invalid product type")

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("Inference result output is not ParseOutput")

        if not isinstance(test_case, ParseTestCase):
            raise ValueError("Test case must be ParseTestCase for QA evaluation")

        if not test_case.qa_config:
            raise ValueError("Test case must have qa_config for QA evaluation")

        qa_config = test_case.qa_config
        metrics: list[MetricValue] = []

        try:
            # Get markdown content from parse output
            markdown_content = inference_result.output.markdown

            # Build stats list once for all return paths
            _stats = build_operational_stats(inference_result)

            if not markdown_content:
                return EvaluationResult(
                    test_id=test_case.test_id,
                    example_id=inference_result.request.example_id,
                    pipeline_name=inference_result.pipeline_name,
                    product_type=inference_result.product_type.value,
                    success=False,
                    error="Empty markdown content in parse output",
                    stats=_stats,
                )

            # Call LLM to get predicted answer
            try:
                # Extract options and unit from metadata
                options = ""
                unit = ""
                if qa_config.metadata:
                    options = str(qa_config.metadata.get("options", ""))
                    unit = str(qa_config.metadata.get("unit", ""))

                if self._llm_service is None:
                    self._llm_service = QALLMService()
                predicted_answer = self._llm_service.answer_question(
                    markdown=markdown_content,
                    question=qa_config.question,
                    question_type=qa_config.question_type,
                    options=options,
                    unit=unit,
                )
            except Exception as e:
                logger.error(f"Failed to get answer from LLM for test {test_case.test_id}: {e}")
                return EvaluationResult(
                    test_id=test_case.test_id,
                    example_id=inference_result.request.example_id,
                    pipeline_name=inference_result.pipeline_name,
                    product_type=inference_result.product_type.value,
                    success=False,
                    error=f"LLM API error: {str(e)}",
                    stats=_stats,
                )

            # Compare predicted vs expected answer
            comparison_result = self._answer_metric.compare(
                predicted=predicted_answer,
                expected=qa_config.answer,
                question_type=qa_config.question_type,
                metadata=qa_config.metadata,
            )

            metrics.append(comparison_result)

            # Emit ANLS* as a second metric for free_text questions
            if qa_config.question_type == "free_text":
                # Use list inputs for comma-delimited answers so ANLS* is order-insensitive
                gt_val = _split_comma_list(qa_config.answer)
                pred_val = _split_comma_list(predicted_answer)
                raw_score = anls_score(gt_val, pred_val)
                score = float(raw_score if isinstance(raw_score, (int, float)) else raw_score[0])
                metrics.append(
                    MetricValue(
                        metric_name="qa_anls_star",
                        value=score,
                        metadata={
                            "predicted": predicted_answer,
                            "expected": qa_config.answer,
                        },
                    )
                )

            return EvaluationResult(
                test_id=test_case.test_id,
                example_id=inference_result.request.example_id,
                pipeline_name=inference_result.pipeline_name,
                product_type=inference_result.product_type.value,
                success=True,
                metrics=metrics,
                stats=_stats,
            )

        except Exception as e:
            logger.error(
                f"Error during QA evaluation for test {test_case.test_id}: {e}",
                exc_info=True,
            )
            return EvaluationResult(
                test_id=test_case.test_id,
                example_id=inference_result.request.example_id,
                pipeline_name=inference_result.pipeline_name,
                product_type=inference_result.product_type.value,
                success=False,
                error=f"Evaluation error: {str(e)}",
                stats=_stats,
            )
