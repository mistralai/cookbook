"""Base evaluator interface for product-specific evaluation."""

from abc import ABC, abstractmethod

from extract_bench.schemas.evaluation import EvaluationResult
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.test_cases.schema import TestCase


class BaseEvaluator(ABC):
    """Abstract base class for product-specific evaluators."""

    @abstractmethod
    def evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> EvaluationResult:
        """
        Evaluate an inference result against a test case.

        :param inference_result: The inference result to evaluate
        :param test_case: The test case with expected output/ground truth
        :return: Evaluation result with metrics
        :raises ValueError: If evaluation cannot be performed (e.g., missing expected_output)
        """
        raise NotImplementedError

    @abstractmethod
    def can_evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> bool:
        """
        Check if this evaluator can evaluate the given inference result and test case.

        :param inference_result: The inference result to evaluate
        :param test_case: The test case to evaluate against
        :return: True if this evaluator can handle this case
        """
        raise NotImplementedError
