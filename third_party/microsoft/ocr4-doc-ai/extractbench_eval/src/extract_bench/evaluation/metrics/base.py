"""Base metric interface for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any

from extract_bench.schemas.evaluation import MetricValue


class Metric(ABC):
    """
    Abstract base class for evaluation metrics.

    Metrics compute scores by comparing expected (ground truth) values
    with actual (predicted) values from inference results.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this metric."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, expected: Any, actual: Any, **kwargs: Any) -> MetricValue:
        """
        Compute the metric score by comparing expected vs actual values.

        :param expected: Expected/ground truth value
        :param actual: Actual/predicted value from inference
        :param kwargs: Additional configuration options for the metric
        :return: MetricValue with the computed score and metadata
        """
        raise NotImplementedError
