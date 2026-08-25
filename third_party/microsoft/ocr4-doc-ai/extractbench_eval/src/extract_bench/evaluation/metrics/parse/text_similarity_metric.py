"""Text similarity metric for comparing markdown against ground truth."""

from typing import Any

from extract_bench.evaluation.metrics.base import Metric
from extract_bench.schemas.evaluation import MetricValue


class TextSimilarityMetric(Metric):
    """Metric for comparing markdown text against ground truth using Levenshtein distance."""

    @property
    def name(self) -> str:
        """Return the name of this metric."""
        return "text_similarity"

    def compute(self, expected: str, actual: str, **kwargs: Any) -> MetricValue:
        """
        Compute text similarity between expected and actual markdown.

        :param expected: Expected markdown (ground truth)
        :param actual: Actual markdown (from inference)
        :param kwargs: Additional parameters (not used)
        :return: MetricValue with similarity score (0.0 to 1.0)
        """
        from autoevals.string import Levenshtein

        if not expected and not actual:
            return MetricValue(
                metric_name=self.name,
                value=1.0,
                metadata={"note": "Both empty"},
            )

        if not expected or not actual:
            return MetricValue(
                metric_name=self.name,
                value=0.0,
                metadata={"note": "One is empty"},
            )

        # Use Levenshtein distance from autoevals
        levenshtein = Levenshtein()
        result = levenshtein(expected, actual)

        # Convert to similarity score (0.0 to 1.0)
        # Levenshtein returns a score where higher is better
        # We normalize it to 0-1 range
        max_len = max(len(expected), len(actual))
        if max_len == 0:
            similarity = 1.0
        else:
            # Levenshtein score is typically normalized, but we ensure 0-1 range
            similarity = max(0.0, min(1.0, result.score / 100.0))

        return MetricValue(
            metric_name=self.name,
            value=similarity,
            metadata={
                "levenshtein_score": result.score,
                "expected_length": len(expected),
                "actual_length": len(actual),
            },
        )
