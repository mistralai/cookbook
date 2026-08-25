"""Shared Gemini API pricing helpers."""

from __future__ import annotations

# Gemini pricing: USD per million tokens (input, output).
# Thinking tokens are billed at the output token rate.
# Source: https://ai.google.dev/gemini-api/docs/pricing (2026-03-25)
GEMINI_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    # model-prefix: (input_per_M, output_per_M)
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.1-pro": (2.00, 12.00),
    # Legacy 1.5 series (<=128k prompt tier; Gemini 1.5 prices were tiered above 128k)
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-flash-8b": (0.0375, 0.15),
}

# Gemini context caching pricing: USD per million tokens / per million token-hours.
# Source: https://ai.google.dev/gemini-api/docs/pricing (2026-04-05)
# NOTE: gemini-3.5-flash cache-hit (0.15) follows the table's consistent 0.1x-input
# convention (every flash entry below: cache_hit == 0.1 x input rate); confirm against
# the published cache pricing if billing exactness on cached tokens matters.
GEMINI_CONTEXT_CACHE_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    # model-prefix: (cache_hit_per_M, storage_per_M_token_hour)
    "gemini-3.7-flash": (0.075, 0.50),
    "gemini-3.6-flash": (0.15, 1.00),
    "gemini-3.5-flash": (0.15, 1.00),
    "gemini-3-flash": (0.05, 1.00),
    "gemini-3.1-flash-lite": (0.025, 1.00),
    "gemini-3.5-flash-lite": (0.03, 1.00),
    "gemini-2.5-flash": (0.03, 1.00),
    "gemini-2.5-flash-lite": (0.01, 1.00),
    "gemini-2.5-pro": (0.125, 4.50),
    "gemini-3.1-pro": (0.20, 4.50),
}


def gemini_pricing_per_million(model: str) -> tuple[float, float]:
    """Return (input_rate, output_rate) in USD per million tokens."""

    return _longest_prefix_match(model, GEMINI_PRICING_PER_MILLION, default=(0.0, 0.0))


def gemini_context_cache_pricing_per_million(model: str) -> tuple[float, float]:
    """Return (cache_hit_rate, storage_rate) in USD per million tokens."""

    return _longest_prefix_match(model, GEMINI_CONTEXT_CACHE_PRICING_PER_MILLION, default=(0.0, 0.0))


def _longest_prefix_match(
    model: str,
    pricing: dict[str, tuple[float, float]],
    *,
    default: tuple[float, float],
) -> tuple[float, float]:
    matches = [(prefix, rates) for prefix, rates in pricing.items() if model.startswith(prefix)]
    return max(matches, key=lambda item: len(item[0]))[1] if matches else default
