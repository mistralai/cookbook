"""Strategy 2: LLM-as-Judge normalization using Claude Haiku.

Uses Anthropic's Claude to:
  1. Judge semantic equivalence of label pairs (column headers).
  2. Normalize mismatched numeric values to canonical form.
  3. Assess missing data-point labels for common abbreviations/synonyms.
  4. Directly judge whether two items are semantically equivalent in context.

Based on the working prototype at ``scripts/llm_normalization_prototype.py``,
with improved prompts, structured parsing, error handling, and cost tracking.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import anthropic

from extract_bench.evaluation.metrics.parse.llm_normalization.base import (
    BaseNormalizer,
    JudgmentResult,
    LabelMatch,
    ValueMatch,
)
from extract_bench.evaluation.metrics.parse.llm_normalization.config import (
    JUDGE_MODEL,
    LABEL_CONFIDENCE_THRESHOLD,
    MAX_API_CALLS_PER_NORMALIZER,
    VALUE_BATCH_SIZE,
    get_anthropic_api_key,
)

logger = logging.getLogger(__name__)

# Claude Haiku pricing (USD per million tokens)
_INPUT_PRICE_PER_M = 0.25
_OUTPUT_PRICE_PER_M = 1.25


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes wrap around JSON."""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _safe_json_loads(text: str, label: str = "") -> list[dict[str, Any]] | None:
    """Parse JSON, returning None on failure instead of raising."""
    cleaned = _strip_code_fences(text)
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        logger.warning("LLM response for %s is not a JSON array: %s", label, type(result))
        return None
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse LLM JSON response for %s: %s — %s",
            label,
            exc,
            cleaned[:200],
        )
        return None


class JudgeNormalizer(BaseNormalizer):
    """LLM-as-judge normalizer backed by Claude Haiku.

    All methods are synchronous.  API calls use ``temperature=0`` for
    deterministic results.  Costs are tracked cumulatively and exposed via
    the ``total_cost_usd`` / ``total_latency_ms`` / ``total_api_calls``
    properties.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = anthropic.Anthropic(api_key=api_key or get_anthropic_api_key())
        self._total_cost_usd: float = 0.0
        self._total_latency_ms: float = 0.0
        self._total_api_calls: int = 0

    # ------------------------------------------------------------------
    # BaseNormalizer abstract properties
    # ------------------------------------------------------------------

    @property
    def strategy_name(self) -> str:
        return "judge"

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def total_latency_ms(self) -> float:
        return self._total_latency_ms

    @property
    def total_api_calls(self) -> int:
        return self._total_api_calls

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> tuple[str | None, float]:
        """Send a single request to Claude Haiku and track costs.

        Returns ``(response_text, latency_ms)``.  Returns ``(None, latency)``
        on API errors so callers can degrade gracefully.
        """
        if self._total_api_calls >= MAX_API_CALLS_PER_NORMALIZER:
            logger.warning(
                "API call limit reached (%d). Skipping further calls.",
                MAX_API_CALLS_PER_NORMALIZER,
            )
            return None, 0.0

        t0 = time.monotonic()
        try:
            response = self._client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            latency_ms = (time.monotonic() - t0) * 1000
            self._total_latency_ms += latency_ms
            self._total_api_calls += 1
            logger.warning("Anthropic API call failed", exc_info=True)
            return None, latency_ms

        latency_ms = (time.monotonic() - t0) * 1000

        usage = response.usage
        cost = (usage.input_tokens * _INPUT_PRICE_PER_M + usage.output_tokens * _OUTPUT_PRICE_PER_M) / 1_000_000

        self._total_cost_usd += cost
        self._total_latency_ms += latency_ms
        self._total_api_calls += 1

        text = response.content[0].text.strip()  # type: ignore[union-attr]
        return text, latency_ms

    # ------------------------------------------------------------------
    # 1. Label normalization
    # ------------------------------------------------------------------

    def normalize_labels(
        self,
        expected_labels: list[str],
        actual_labels: list[str],
        context: str = "",
    ) -> list[LabelMatch]:
        """Judge semantic equivalence of expected vs actual column headers.

        Pairs are formed positionally (``expected_labels[i]`` vs
        ``actual_labels[i]``).  If the lists differ in length the shorter
        one is padded with empty strings.
        """
        n = max(len(expected_labels), len(actual_labels))
        if n == 0:
            return []

        # Pad to equal length
        exp = list(expected_labels) + [""] * (n - len(expected_labels))
        act = list(actual_labels) + [""] * (n - len(actual_labels))

        pairs_desc = "\n".join(f'  {i + 1}. Expected: "{exp[i]}"  ->  Actual: "{act[i]}"' for i in range(n))

        ctx_line = f"\nContext: {context}\n" if context else ""

        prompt = f"""You are evaluating whether column headers from a chart-to-table conversion \
are semantically equivalent.
{ctx_line}
A benchmark test expected certain column headers in a table extracted from a chart image. The \
model produced different headers. Fuzzy string matching scored them low, but they might be \
semantically equivalent.

Label pairs to evaluate:
{pairs_desc}

For each pair, determine if the expected and actual labels refer to the same concept. Consider:
- Synonyms: "Year" ~ "Date", "Entity" ~ "Category" (both can mean the row identifier)
- Abbreviations: "CHF" = "Swiss francs" = "francs"
- Unit variations: "Value" ~ "Percentage" when the column contains percentages
- Formatting differences: "Nasdaq100" ~ "Nasdaq 100" (spacing/punctuation)
- Translations or locale variants: "Couts (CHF par mois)" ~ "Couts (en francs par mois)"
- Generic placeholders: "Entity" and "Value" are generic names that often map to whatever the \
actual column describes

Reject pairs that are genuinely different concepts:
- "Germany" vs "U.S." = different countries
- "Oppose" vs "Favor" = opposite meanings
- "Budget deficit" vs "Economy" = different topics

Respond with a JSON array. Each element:
{{"pair": <1-based index>, "match": true/false, "confidence": 0.0-1.0, \
"reasoning": "brief explanation"}}

Return ONLY the JSON array, no surrounding text."""

        text, _ = self._call_llm(prompt)
        results = _safe_json_loads(text, "normalize_labels") if text is not None else None

        matches: list[LabelMatch] = []
        if results is None:
            # Return non-match for every pair on parse/API failure
            reason = "LLM API call failed" if text is None else "LLM response could not be parsed"
            for i in range(n):
                matches.append(
                    LabelMatch(
                        expected=exp[i],
                        actual=act[i],
                        is_match=False,
                        confidence=0.0,
                        reasoning=reason,
                        strategy="judge",
                    )
                )
            return matches

        # Index the results by pair number for robustness
        by_pair: dict[int, dict[str, Any]] = {}
        for item in results:
            idx = item.get("pair", 0) - 1
            if 0 <= idx < n:
                by_pair[idx] = item

        for i in range(n):
            item = by_pair.get(i, {})
            conf = float(item.get("confidence", 0.0))
            is_match = bool(item.get("match", False)) and conf >= LABEL_CONFIDENCE_THRESHOLD
            matches.append(
                LabelMatch(
                    expected=exp[i],
                    actual=act[i],
                    is_match=is_match,
                    confidence=conf,
                    reasoning=str(item.get("reasoning", "")),
                    strategy="judge",
                )
            )

        return matches

    # ------------------------------------------------------------------
    # 2. Value normalization
    # ------------------------------------------------------------------

    def normalize_values(
        self,
        expected_values: list[str],
        actual_values: list[str],
        context: str = "",
    ) -> list[ValueMatch]:
        """Normalize mismatched value pairs to canonical numeric form.

        Pairs that are already exact string matches are returned
        immediately without an API call.  Remaining pairs are batched
        into groups of ``VALUE_BATCH_SIZE`` for efficiency.
        """
        n = max(len(expected_values), len(actual_values))
        if n == 0:
            return []

        exp = list(expected_values) + [""] * (n - len(expected_values))
        act = list(actual_values) + [""] * (n - len(actual_values))

        # Pre-populate results; only send mismatched pairs to the LLM
        all_matches: list[ValueMatch] = []
        mismatched_indices: list[int] = []
        for i in range(n):
            if exp[i] == act[i]:
                all_matches.append(
                    ValueMatch(
                        expected=exp[i],
                        actual=act[i],
                        is_match=True,
                        normalized_expected=exp[i],
                        normalized_actual=act[i],
                        reasoning="exact string match",
                        strategy="judge",
                    )
                )
            else:
                all_matches.append(
                    ValueMatch(
                        expected=exp[i],
                        actual=act[i],
                        is_match=False,
                        normalized_expected="",
                        normalized_actual="",
                        reasoning="",
                        strategy="judge",
                    )
                )
                mismatched_indices.append(i)

        if not mismatched_indices:
            return all_matches

        # Batch mismatched pairs
        for batch_start in range(0, len(mismatched_indices), VALUE_BATCH_SIZE):
            batch_indices = mismatched_indices[batch_start : batch_start + VALUE_BATCH_SIZE]

            pairs_desc = "\n".join(
                f'  {j + 1}. Expected: "{exp[idx]}"  ->  Actual: "{act[idx]}"' for j, idx in enumerate(batch_indices)
            )

            ctx_line = f"\nContext: {context}\n" if context else ""

            prompt = f"""You are comparing numeric values from a chart-to-table \
conversion benchmark.
{ctx_line}
The expected values come from ground truth annotations. The actual values come from a model \
reading a chart image. Values may differ in format but represent the same quantity.

Value pairs:
{pairs_desc}

For each pair, determine if they represent the same value. Consider:
- Percentage format: "38.75" ~ "39%" (rounded from chart reading)
- Precision: "41.35" ~ "41.5%" (chart visual precision is limited)
- Currency/unit symbols: "$1.2M" = "1,200,000" = "1.2 million"
- Thousands separators: "1,234" = "1234"
- Rounding within +/-2% of the expected value is acceptable for chart readings
- Trailing zeros: "10.0" = "10" = "10.00"
- Negative sign variants: "-5" = "(5)" = "- 5"

Respond with a JSON array. Each element:
{{"pair": <1-based index>, "match": true/false, "normalized_expected": "canonical", \
"normalized_actual": "canonical", "reasoning": "brief"}}

Return ONLY the JSON array, no surrounding text."""

            text, _ = self._call_llm(prompt, max_tokens=2048)
            results = _safe_json_loads(text, "normalize_values") if text is not None else None

            if results is None:
                continue

            for item in results:
                j = item.get("pair", 0) - 1
                if 0 <= j < len(batch_indices):
                    idx = batch_indices[j]
                    all_matches[idx] = ValueMatch(
                        expected=exp[idx],
                        actual=act[idx],
                        is_match=bool(item.get("match", False)),
                        normalized_expected=str(item.get("normalized_expected", "")),
                        normalized_actual=str(item.get("normalized_actual", "")),
                        reasoning=str(item.get("reasoning", "")),
                        strategy="judge",
                    )

        return all_matches

    # ------------------------------------------------------------------
    # 3. Data-point label normalization
    # ------------------------------------------------------------------

    def normalize_data_point_labels(
        self,
        missing_labels: list[str],
        table_headers: list[str],
        context: str = "",
    ) -> list[LabelMatch]:
        """Assess whether missing data-point labels match any table header.

        Unlike ``normalize_labels`` which compares positional pairs, this
        method checks each missing label against the full set of table
        headers and asks the LLM whether a semantic match is plausible.
        """
        if not missing_labels:
            return []

        labels_desc = "\n".join(f'  {i + 1}. Missing label: "{label}"' for i, label in enumerate(missing_labels))

        headers_desc = ", ".join(f'"{h}"' for h in table_headers) if table_headers else "(table headers not available)"

        ctx_line = f"\nFailure context: {context[:500]}\n" if context else ""

        prompt = f"""You are evaluating a chart-to-table benchmark. A data point's value was found \
in the table, but some expected labels could not be matched to any row/column header via fuzzy \
string matching.
{ctx_line}
Table headers present: [{headers_desc}]

Missing labels:
{labels_desc}

Common reasons for label mismatch:
- The model abbreviated a long header ("Number of major events (1,000+ attendees)" -> \
"Major Events")
- The model used a synonym ("Sub-Saharan Africa" -> "SSA")
- The label appears as a chart title/subtitle rather than a table header
- Locale differences ("Couts (CHF par mois)" -> "Couts (en francs par mois)")

For each missing label, assess:
1. Is it likely a semantic match to one of the table headers? (match=true, and specify which \
header in reasoning)
2. Or is it genuinely absent from the table? (match=false)

Be conservative: only mark match=true if there is a clear semantic link to a specific header.

Respond with a JSON array:
{{"pair": <1-based index>, "match": true/false, "confidence": 0.0-1.0, \
"reasoning": "brief explanation"}}

Return ONLY the JSON array, no surrounding text."""

        text, _ = self._call_llm(prompt)
        results = _safe_json_loads(text, "normalize_data_point_labels") if text is not None else None

        matches: list[LabelMatch] = []
        n = len(missing_labels)

        if results is None:
            reason = "LLM API call failed" if text is None else "LLM response could not be parsed"
            for label in missing_labels:
                matches.append(
                    LabelMatch(
                        expected=label,
                        actual="",
                        is_match=False,
                        confidence=0.0,
                        reasoning=reason,
                        strategy="judge",
                    )
                )
            return matches

        by_pair: dict[int, dict[str, Any]] = {}
        for item in results:
            idx = item.get("pair", 0) - 1
            if 0 <= idx < n:
                by_pair[idx] = item

        for i in range(n):
            item = by_pair.get(i, {})
            conf = float(item.get("confidence", 0.0))
            is_match = bool(item.get("match", False)) and conf >= LABEL_CONFIDENCE_THRESHOLD
            matches.append(
                LabelMatch(
                    expected=missing_labels[i],
                    actual="",
                    is_match=is_match,
                    confidence=conf,
                    reasoning=str(item.get("reasoning", "")),
                    strategy="judge",
                )
            )

        return matches

    # ------------------------------------------------------------------
    # 4. Direct LLM-as-judge equivalence (novel addition)
    # ------------------------------------------------------------------

    def judge_equivalence(
        self,
        expected: str,
        actual: str,
        context: str = "",
    ) -> JudgmentResult:
        """Directly ask the LLM whether two items are semantically equivalent.

        Instead of normalizing both sides and comparing deterministically,
        this method poses a single yes/no question to the LLM with full
        context.  This is useful for ambiguous cases where normalization
        alone would be insufficient (e.g., column-swap detection, locale
        differences, or multi-word semantic equivalence).

        :param expected: The ground truth string (label, value, or cell).
        :param actual: The model-produced string.
        :param context: Additional context such as chart description,
            surrounding data, or test ID.
        :return: A ``JudgmentResult`` with the LLM's verdict.
        """
        ctx_line = f"\nAdditional context: {context[:800]}\n" if context else ""

        prompt = f"""You are a judge in a chart-to-table conversion benchmark. Your task is to \
determine whether two items are semantically equivalent in the context of chart data extraction.
{ctx_line}
Expected (ground truth): "{expected}"
Actual (model output):   "{actual}"

Consider all of the following when making your judgment:
- Are they the same concept expressed differently? (synonyms, abbreviations, locale variants)
- If numeric, do they represent the same quantity? (format, rounding, units)
- If labels, do they refer to the same data dimension? (headers, axis labels)
- Could the difference be explained by reasonable chart-reading imprecision?

Do NOT consider them equivalent if:
- They refer to genuinely different entities (e.g., different countries, opposite sentiments)
- The numeric difference exceeds 5% of the expected value
- They describe fundamentally different concepts

Respond with a single JSON object:
{{"is_equivalent": true/false, "confidence": 0.0-1.0, \
"reasoning": "concise explanation of your judgment"}}

Return ONLY the JSON object, no surrounding text."""

        text, _ = self._call_llm(prompt, max_tokens=256)
        if text is None:
            return JudgmentResult(
                expected=expected,
                actual=actual,
                is_equivalent=False,
                confidence=0.0,
                reasoning="LLM API call failed",
            )

        cleaned = _strip_code_fences(text)

        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse judge_equivalence response: %s", cleaned[:200])
            return JudgmentResult(
                expected=expected,
                actual=actual,
                is_equivalent=False,
                confidence=0.0,
                reasoning="LLM response could not be parsed",
            )

        conf = float(obj.get("confidence", 0.0))
        return JudgmentResult(
            expected=expected,
            actual=actual,
            is_equivalent=(bool(obj.get("is_equivalent", False)) and conf >= LABEL_CONFIDENCE_THRESHOLD),
            confidence=conf,
            reasoning=str(obj.get("reasoning", "")),
        )
