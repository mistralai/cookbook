"""Base classes and shared data structures for LLM normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LabelMatch:
    """Result of comparing an expected label to an actual label via LLM."""

    expected: str
    actual: str
    is_match: bool
    confidence: float
    reasoning: str
    strategy: str  # "structured" or "judge"


@dataclass
class ValueMatch:
    """Result of comparing an expected value to an actual value via LLM."""

    expected: str
    actual: str
    is_match: bool
    normalized_expected: str
    normalized_actual: str
    reasoning: str
    strategy: str  # "structured" or "judge"


@dataclass
class JudgmentResult:
    """Result of a direct LLM-as-judge semantic equivalence check."""

    expected: str
    actual: str
    is_equivalent: bool
    confidence: float
    reasoning: str


@dataclass
class NormalizationResult:
    """Aggregated result from a normalization run."""

    label_matches: list[LabelMatch] = field(default_factory=list)
    value_matches: list[ValueMatch] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    api_calls: int = 0


class BaseNormalizer(ABC):
    """Abstract base class for LLM normalization strategies."""

    @abstractmethod
    def normalize_labels(
        self,
        expected_labels: list[str],
        actual_labels: list[str],
        context: str = "",
    ) -> list[LabelMatch]:
        """Determine semantic equivalence between pairs of expected/actual labels.

        :param expected_labels: Ground truth column/row headers.
        :param actual_labels: Model-produced headers (same length as expected_labels).
        :param context: Optional context (test ID, chart description) for the LLM.
        :return: List of LabelMatch results, one per pair.
        """

    @abstractmethod
    def normalize_values(
        self,
        expected_values: list[str],
        actual_values: list[str],
        context: str = "",
    ) -> list[ValueMatch]:
        """Normalize and compare pairs of expected/actual cell values.

        :param expected_values: Ground truth cell values.
        :param actual_values: Model-produced values (same length as expected_values).
        :param context: Optional context for the LLM.
        :return: List of ValueMatch results, one per pair.
        """

    @abstractmethod
    def normalize_data_point_labels(
        self,
        missing_labels: list[str],
        table_headers: list[str],
        context: str = "",
    ) -> list[LabelMatch]:
        """Assess whether missing data-point labels could match table headers.

        Used by ChartDataPointRule when fuzzy matching fails to associate a
        label with any row/column header.

        :param missing_labels: Labels that fuzzy matching could not find.
        :param table_headers: All headers present in the actual table.
        :param context: Failure explanation or test ID for the LLM.
        :return: List of LabelMatch results, one per missing label.
        """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the strategy identifier (e.g. 'structured' or 'judge')."""

    @property
    @abstractmethod
    def total_cost_usd(self) -> float:
        """Return cumulative USD cost of all API calls made by this normalizer."""

    @property
    @abstractmethod
    def total_latency_ms(self) -> float:
        """Return cumulative latency in milliseconds."""

    @property
    @abstractmethod
    def total_api_calls(self) -> int:
        """Return cumulative count of API calls."""
