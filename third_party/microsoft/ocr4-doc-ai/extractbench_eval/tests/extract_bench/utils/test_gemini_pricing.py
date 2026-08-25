from __future__ import annotations

from datetime import date

from extract_bench.utils.gemini_pricing import (
    gemini_context_cache_pricing_per_million,
    gemini_pricing_per_million,
)


def test_gemini_pricing_per_million_3_7_flash() -> None:
    assert date.today() <= date(2026, 12, 31), "Promotional pricing expired after 2026-12-31"
    assert gemini_pricing_per_million("gemini-3.7-flash") == (0.75, 3.75)


def test_gemini_pricing_per_million_3_6_flash() -> None:
    assert gemini_pricing_per_million("gemini-3.6-flash") == (1.50, 7.50)


def test_gemini_context_cache_pricing_3_7_flash() -> None:
    assert date.today() <= date(2026, 12, 31), "Promotional pricing expired after 2026-12-31"
    assert gemini_context_cache_pricing_per_million("gemini-3.7-flash") == (0.075, 0.50)


def test_gemini_context_cache_pricing_3_6_flash() -> None:
    assert gemini_context_cache_pricing_per_million("gemini-3.6-flash") == (0.15, 1.00)
