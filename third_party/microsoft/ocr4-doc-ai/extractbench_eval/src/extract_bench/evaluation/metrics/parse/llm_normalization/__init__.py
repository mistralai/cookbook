"""LLM normalization for chart evaluation metrics.

Uses Claude LLM-as-judge to improve chart benchmark accuracy by handling
semantic equivalence that deterministic fuzzy matching misses.

Controlled by env var ``EXTRACT_BENCH_LLM_NORMALIZATION``:
  - ``"judge"`` -- Claude LLM-as-judge normalization (default)
  - ``"off"``   -- no LLM normalization
"""

from extract_bench.evaluation.metrics.parse.llm_normalization.base import (
    BaseNormalizer,
    JudgmentResult,
    LabelMatch,
    NormalizationResult,
    ValueMatch,
)
from extract_bench.evaluation.metrics.parse.llm_normalization.config import (
    NormalizationMode,
    get_normalization_mode,
)

__all__ = [
    "BaseNormalizer",
    "JudgmentResult",
    "LabelMatch",
    "NormalizationMode",
    "NormalizationResult",
    "ValueMatch",
    "get_normalization_mode",
    "get_normalizer",
]


def get_normalizer(
    mode: NormalizationMode | None = None,
) -> BaseNormalizer | None:
    """Factory function that returns the appropriate normalizer for the given mode.

    :param mode: Normalization mode. If None, reads from env var.
    :return: A normalizer instance, or None if mode is OFF.
    """
    if mode is None:
        mode = get_normalization_mode()

    if mode == NormalizationMode.JUDGE:
        from extract_bench.evaluation.metrics.parse.llm_normalization.strategy_judge import JudgeNormalizer

        return JudgeNormalizer()

    return None
