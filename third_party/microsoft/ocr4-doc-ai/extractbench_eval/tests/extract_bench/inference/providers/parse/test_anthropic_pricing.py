"""Unit tests for Anthropic provider pricing, including the Sonnet 5
introductory-rate transition."""

from __future__ import annotations

from datetime import date

import pytest

pytest.importorskip("PIL", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

from extract_bench.inference.providers.parse import anthropic
from extract_bench.inference.providers.parse.anthropic import AnthropicProvider


def _provider_for_model(model: str) -> AnthropicProvider:
    provider = object.__new__(AnthropicProvider)
    provider._model = model
    return provider


def test_sonnet_5_uses_introductory_pricing_through_august_2026(monkeypatch: pytest.MonkeyPatch) -> None:
    class IntroDate(date):
        @classmethod
        def today(cls) -> date:
            return cls(2026, 8, 31)

    monkeypatch.setattr(anthropic, "date", IntroDate)

    assert _provider_for_model("claude-sonnet-5")._get_pricing() == (2.00, 10.00)


def test_sonnet_5_uses_standard_pricing_after_intro_period(monkeypatch: pytest.MonkeyPatch) -> None:
    class StandardDate(date):
        @classmethod
        def today(cls) -> date:
            return cls(2026, 9, 1)

    monkeypatch.setattr(anthropic, "date", StandardDate)

    assert _provider_for_model("claude-sonnet-5")._get_pricing() == (3.00, 15.00)
