"""Judge variant of the rule-based metric with LLM normalization.

Extends RuleBasedMetric by post-processing the rule results with LLM
normalization. Appends ``_judge`` counterparts to ``rule_results`` in the
returned metadata so the evaluator's per-type loop auto-generates
``rule_*_judge_pass_rate`` metrics. Also stores ``judge_pass_rate`` in
metadata so the evaluator can emit ``rule_pass_rate_judge``.

Controlled by env var ``EXTRACT_BENCH_LLM_NORMALIZATION`` (off by default).
When OFF this class is identical in behaviour to its parent.
"""

from __future__ import annotations

from typing import Any

from extract_bench.evaluation.metrics.parse.llm_normalization.postprocess import (
    maybe_apply_llm_normalization,
)
from extract_bench.evaluation.metrics.parse.rule_based_metric import RuleBasedMetric
from extract_bench.schemas.evaluation import MetricValue

_CHART_TYPES = {"chart_data_point", "chart_data_array_labels"}


class RuleBasedJudgeMetric(RuleBasedMetric):
    """RuleBasedMetric extended with LLM-normalization of chart rule results."""

    def compute(
        self,
        expected: list[dict[str, Any]] | None,  # type: ignore[override]
        actual: str,
        page: int | None = None,
        **kwargs: Any,
    ) -> MetricValue:
        result = super().compute(expected, actual, page, **kwargs)  # type: ignore[arg-type]

        rule_results = result.metadata.get("rule_results")
        total = result.metadata.get("total", 0)
        if not rule_results or not total:
            return result

        # ``normalized`` is the same length as ``rule_results``; zip pairs them.
        normalized, llm_metadata = maybe_apply_llm_normalization(rule_results)
        if llm_metadata is None:
            return result  # normalization OFF — identical to base class

        # Snapshot normalized scores before extending rule_results —
        # in BOTH mode, normalized may be the same list object as rule_results.
        result.metadata["judge_pass_rate"] = sum(r["score"] for r in normalized) / total
        result.metadata["judge_passed"] = sum(1 for r in normalized if r.get("passed"))

        for r, nr in zip(list(rule_results), normalized, strict=False):
            if r.get("type", "") in _CHART_TYPES:
                rule_results.append({**nr, "type": r["type"] + "_judge"})
        return result
