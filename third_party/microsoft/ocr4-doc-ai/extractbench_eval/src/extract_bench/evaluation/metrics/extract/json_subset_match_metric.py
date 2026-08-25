"""JSON subset match metric as a Metric class implementation."""

from typing import Any

from extract_bench.evaluation.metrics.base import Metric
from extract_bench.evaluation.metrics.extract.json_subset_match import (
    json_subset_match_score,
)
from extract_bench.schemas.evaluation import MetricValue


class JsonSubsetMatchMetric(Metric):
    """
    Metric that computes similarity between expected and actual JSON structures.

    Uses json_subset_match_score to compare JSON objects, only evaluating
    keys present in the expected structure (subset matching).
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        cosine_similarity: bool = False,
        normalize_dates: bool = True,
        weighted: bool = True,
    ):
        """
        Initialize the JSON subset match metric.

        :param case_sensitive: Whether string comparison should be case-sensitive
        :param cosine_similarity: Use embedding similarity for strings (requires OpenAI API key)
        :param normalize_dates: Normalize date strings before comparison
        :param weighted: If True (default), weight fields by their number of leaf nodes.
                         If False, use simple averaging (each field/element counts equally).
        """
        self._case_sensitive = case_sensitive
        self._cosine_similarity = cosine_similarity
        self._normalize_dates = normalize_dates
        self._weighted = weighted

    @property
    def name(self) -> str:
        """Return the name of this metric."""
        return "accuracy"

    def compute(self, expected: Any, actual: Any, **kwargs: Any) -> MetricValue:
        """
        Compute JSON subset match score.

        :param expected: Expected JSON structure
        :param actual: Actual JSON structure to compare
        :param kwargs: Additional options (can override instance defaults)
        :return: MetricValue with score and metadata
        """
        # Allow kwargs to override instance defaults
        case_sensitive = kwargs.get("case_sensitive", self._case_sensitive)
        cosine_similarity = kwargs.get("cosine_similarity", self._cosine_similarity)
        normalize_dates = kwargs.get("normalize_dates", self._normalize_dates)
        weighted = kwargs.get("weighted", self._weighted)
        identity_keys_by_path = kwargs.get("identity_keys_by_path")
        data_schema = kwargs.get("data_schema")

        score = json_subset_match_score(
            expected=expected,
            actual=actual,
            case_sensitive=case_sensitive,
            cosine_similarity=cosine_similarity,
            normalize_dates=normalize_dates,
            weighted=weighted,
            identity_keys_by_path=identity_keys_by_path,
            data_schema=data_schema,
        )

        return MetricValue(
            metric_name=self.name,
            value=score,
            metadata={
                "case_sensitive": case_sensitive,
                "cosine_similarity": cosine_similarity,
                "normalize_dates": normalize_dates,
                "weighted": weighted,
                "identity_paired_paths": sorted(".".join(p) for p in (identity_keys_by_path or {})),
            },
        )
