"""Post-processing LLM normalization for chart evaluation metrics.

Applies LLM normalization AFTER rules have already been evaluated, by parsing
the explanation strings from chart rule failures and re-evaluating them with
an LLM normalizer. This approach requires ZERO modifications to existing
test rules — it works entirely on the rule_results dicts produced by the
standard evaluation pipeline.

Controlled by env var ``EXTRACT_BENCH_LLM_NORMALIZATION`` (off by default).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from extract_bench.evaluation.metrics.parse.llm_normalization.base import (
    BaseNormalizer,
)
from extract_bench.evaluation.metrics.parse.llm_normalization.config import (
    NormalizationMode,
    get_normalization_mode,
)

logger = logging.getLogger(__name__)

# Chart rule types eligible for LLM normalization
_CHART_RULE_TYPES = {
    "chart_data_point",
    "chart_data_array_labels",
    "chart_data_array_data",
}


# ---------------------------------------------------------------------------
# Regex parsers (adapted from scripts/llm_normalization_prototype.py)
# ---------------------------------------------------------------------------


def _parse_label_pairs(explanation: str) -> list[tuple[str, str, float]]:
    """Extract (expected, actual, fuzzy_score) label pairs from explanation.

    Matches patterns like: 'Entity' vs 'Year' (22%)
    """
    pairs = []
    for m in re.finditer(r"'([^']+)'\s+vs\s+'([^']+)'\s+\((\d+)%\)", explanation):
        pairs.append((m.group(1), m.group(2), int(m.group(3)) / 100.0))
    return pairs


def _parse_value_pairs(
    explanation: str,
) -> list[tuple[list[str], list[str], float]]:
    """Extract (expected_vals, actual_vals, row_score) from data explanation rows.

    Matches patterns like:
        Row 1: 99% | Expected: ['Nigeria', '38.75'] | Actual: ['Nigeria', '39%']
    """
    rows = []
    for m in re.finditer(
        r"Row \d+: (\d+)% \| Expected: \[([^\]]+)\] \| Actual: \[([^\]]+)\]",
        explanation,
    ):
        score = int(m.group(1)) / 100.0
        exp = [v.strip().strip("'\"") for v in m.group(2).split(",")]
        act = [v.strip().strip("'\"") for v in m.group(3).split(",")]
        rows.append((exp, act, score))
    return rows


def _parse_data_point_labels(explanation: str) -> list[str]:
    """Extract missing label names from data point failure explanation.

    Matches patterns like: missing labels: ['number of major events']
    """
    labels: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"missing labels: \[([^\]]+)\]", explanation):
        for label_m in re.finditer(r"'([^']+)'", m.group(1)):
            label = label_m.group(1)
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def _extract_score_parts(explanation: str) -> tuple[float | None, float | None]:
    """Extract (achieved, total) from score notation like '(3.61/4.00)'."""
    m = re.search(r"\(([\d.]+)/([\d.]+)\)", explanation)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def _has_dimension_mismatch(explanation: str) -> bool:
    """Check if explanation contains a dimension mismatch (structural issue)."""
    return bool(re.search(r"Dimension mismatch:", explanation))


# ---------------------------------------------------------------------------
# Per-rule-type normalization
# ---------------------------------------------------------------------------


def _normalize_labels_rule(
    result: dict[str, Any],
    normalizer: BaseNormalizer,
) -> dict[str, Any] | None:
    """Try to upgrade a failed chart_data_array_labels rule via LLM."""
    explanation = result.get("explanation", "")
    pairs = _parse_label_pairs(explanation)
    if not pairs:
        return None

    expected = [p[0] for p in pairs]
    actual = [p[1] for p in pairs]

    matches = normalizer.normalize_labels(expected, actual, context=explanation[:300])

    upgraded = sum(1 for m in matches if m.is_match)
    if upgraded == 0:
        return None

    # Recalculate score: each upgraded label pair was contributing its fuzzy
    # score to achieved. Upgrading it to 1.0 adds (1.0 - fuzzy_score).
    achieved, total = _extract_score_parts(explanation)
    if achieved is not None and total is not None and total > 0:
        improvement = sum(1.0 - pairs[i][2] for i, m in enumerate(matches) if m.is_match and i < len(pairs))
        new_achieved = min(achieved + improvement, total)
        new_score = new_achieved / total
    else:
        new_score = 1.0 if upgraded == len(pairs) else result["score"]

    reasons = "; ".join(f"'{m.expected}'~'{m.actual}' ({m.reasoning})" for m in matches if m.is_match)

    return {
        **result,
        "score": new_score,
        "passed": new_score >= 1.0 - 1e-9,
        "explanation": f"{explanation} [LLM normalized {upgraded}/{len(pairs)} labels: {reasons}]",
        "llm_normalized": True,
    }


def _normalize_data_rule(
    result: dict[str, Any],
    normalizer: BaseNormalizer,
) -> dict[str, Any] | None:
    """Try to upgrade a failed chart_data_array_data rule via LLM."""
    explanation = result.get("explanation", "")

    if _has_dimension_mismatch(explanation):
        return None  # Can't fix structural mismatches with LLM

    rows = _parse_value_pairs(explanation)
    if not rows:
        return None

    # Flatten mismatched values across all failing rows
    all_exp: list[str] = []
    all_act: list[str] = []
    for exp_vals, act_vals, _row_score in rows:
        for e, a in zip(exp_vals, act_vals, strict=False):
            if e != a:
                all_exp.append(e)
                all_act.append(a)

    if not all_exp:
        return None

    matches = normalizer.normalize_values(all_exp, all_act, context=explanation[:300])

    upgraded = sum(1 for m in matches if m.is_match)
    if upgraded == 0:
        return None

    # Each upgraded value adds ~1 matched cell to the achieved count
    achieved, total = _extract_score_parts(explanation)
    if achieved is not None and total is not None and total > 0:
        new_score = min((achieved + upgraded) / total, 1.0)
    else:
        new_score = 1.0 if upgraded == len(all_exp) else result["score"]

    reasons = "; ".join(f"'{m.expected}'~'{m.actual}' ({m.reasoning})" for m in matches if m.is_match)

    return {
        **result,
        "score": new_score,
        "passed": new_score >= 1.0 - 1e-9,
        "explanation": (f"{explanation} [LLM normalized {upgraded}/{len(all_exp)} values: {reasons}]"),
        "llm_normalized": True,
    }


def _normalize_data_point_rule(
    result: dict[str, Any],
    normalizer: BaseNormalizer,
) -> dict[str, Any] | None:
    """Try to upgrade a failed chart_data_point rule via LLM."""
    explanation = result.get("explanation", "")

    missing = _parse_data_point_labels(explanation)
    if not missing:
        return None

    # We don't have direct access to table headers from the explanation,
    # but the normalizer can work with the explanation context alone.
    matches = normalizer.normalize_data_point_labels(missing, table_headers=[], context=explanation[:500])

    upgraded = sum(1 for m in matches if m.is_match)
    if upgraded == 0:
        return None

    all_matched = upgraded == len(missing)

    reasons = "; ".join(f"'{m.expected}' matched ({m.reasoning})" for m in matches if m.is_match)

    return {
        **result,
        "score": 1.0 if all_matched else result["score"],
        "passed": all_matched,
        "explanation": f"{explanation} [LLM matched {upgraded}/{len(missing)} labels: {reasons}]",
        "llm_normalized": True,
    }


# ---------------------------------------------------------------------------
# Core normalization pipeline
# ---------------------------------------------------------------------------

_RULE_HANDLERS = {
    "chart_data_array_labels": _normalize_labels_rule,
    "chart_data_array_data": _normalize_data_rule,
    "chart_data_point": _normalize_data_point_rule,
}


def _normalizer_stats(normalizer: BaseNormalizer) -> dict[str, Any]:
    """Extract cost/latency/api_calls stats from a normalizer."""
    return {
        "strategy": normalizer.strategy_name,
        "cost_usd": normalizer.total_cost_usd,
        "latency_ms": normalizer.total_latency_ms,
        "api_calls": normalizer.total_api_calls,
    }


def _apply_normalization(
    rule_results: list[dict[str, Any]],
    normalizer: BaseNormalizer,
) -> list[dict[str, Any]]:
    """Apply LLM normalization to chart rule failures, returning updated list."""
    updated: list[dict[str, Any]] = []
    for result in rule_results:
        rtype = result.get("type", "")

        # Only process failed chart rules
        if rtype not in _CHART_RULE_TYPES or result.get("passed", True):
            updated.append(result)
            continue

        handler = _RULE_HANDLERS.get(rtype)
        normalized = handler(result, normalizer) if handler else None
        updated.append(normalized if normalized is not None else result)

    return updated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def maybe_apply_llm_normalization(
    rule_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Post-process chart rule results with LLM normalization.

    Returns ``(updated_results, llm_metadata_or_None)``.
    When mode is OFF or initialization fails, returns results unchanged
    with ``None`` metadata.
    """
    # Lazy import to avoid circular dependency (__init__.py defines get_normalizer)
    from extract_bench.evaluation.metrics.parse.llm_normalization import (
        get_normalizer,
    )

    mode = get_normalization_mode()
    if mode == NormalizationMode.OFF:
        return rule_results, None

    try:
        normalizer_or_list = get_normalizer(mode)
    except Exception:
        logger.warning(
            "Failed to initialize LLM normalizer (mode=%s), skipping",
            mode.value,
            exc_info=True,
        )
        return rule_results, None

    if normalizer_or_list is None:
        return rule_results, None

    n_failures = sum(1 for r in rule_results if r.get("type", "") in _CHART_RULE_TYPES and not r.get("passed", True))
    logger.info(
        "LLM normalization mode=%s, post-processing %d chart failures out of %d rules",
        mode.value,
        n_failures,
        len(rule_results),
    )

    if not isinstance(normalizer_or_list, BaseNormalizer):
        return rule_results, None

    normalizer = normalizer_or_list

    logger.info("Post-processing with normalizer: %s", normalizer.strategy_name)
    updated = _apply_normalization(rule_results, normalizer)

    return updated, {"mode": mode.value, **_normalizer_stats(normalizer)}
