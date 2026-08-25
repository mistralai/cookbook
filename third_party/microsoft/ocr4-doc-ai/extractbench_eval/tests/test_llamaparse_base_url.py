"""Tests for the configurable LlamaParse API base URL.

The base URL is resolved with the precedence:
``base_config["base_url"]`` -> ``LLAMA_CLOUD_BASE_URL`` env var ->
``use_europe`` -> default prod (``None``).

An explicit override must win over EU selection, and must be
forwarded into the underlying llama-cloud SDK client constructor. A
custom deployment may accept any/empty key, so an empty key must not
crash provider initialization when an explicit base URL is set.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PIL", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

import extract_bench.inference.providers.parse.llamaparse as llamaparse_module
from extract_bench.inference.providers.base import ProviderPermanentError
from extract_bench.inference.providers.parse.llamaparse import LlamaParseProvider


@pytest.fixture(autouse=True)
def _clean_llama_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from any ambient LlamaCloud env vars."""
    for var in (
        "LLAMA_CLOUD_API_KEY",
        "LLAMA_CLOUD_BASE_URL",
        "LLAMA_CLOUD_EU_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    # Pretend the V2 SDK is installed so __init__ proceeds past the guard.
    monkeypatch.setattr(llamaparse_module, "_HAS_V2_SDK", True)


def _make_provider(base_config: dict) -> LlamaParseProvider:
    return LlamaParseProvider("llamaparse", base_config)


def test_base_url_from_base_config_wins() -> None:
    provider = _make_provider({"api_key": "k", "base_url": "http://localhost:8000"})
    assert provider._base_url == "http://localhost:8000"


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMA_CLOUD_BASE_URL", "http://localhost:9000")
    provider = _make_provider({"api_key": "k"})
    assert provider._base_url == "http://localhost:9000"


def test_base_config_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMA_CLOUD_BASE_URL", "http://from-env:9000")
    provider = _make_provider({"api_key": "k", "base_url": "http://from-config:8000"})
    assert provider._base_url == "http://from-config:8000"


def test_no_override_keeps_europe_behavior() -> None:
    provider = _make_provider({"api_key": "k", "use_europe": True})
    assert provider._base_url == "https://api.cloud.eu.llamaindex.ai"


def test_explicit_base_url_beats_use_europe() -> None:
    provider = _make_provider({"api_key": "k", "use_europe": True, "base_url": "http://localhost:8000"})
    assert provider._base_url == "http://localhost:8000"


def test_empty_key_does_not_crash_with_explicit_base_url() -> None:
    # Custom deployment with no key configured.
    provider = _make_provider({"base_url": "http://localhost:8000"})
    assert provider._base_url == "http://localhost:8000"
    assert provider._api_key == ""


def test_base_url_not_forwarded_to_sdk_config() -> None:
    provider = _make_provider({"api_key": "k", "base_url": "http://localhost:8000"})
    assert "base_url" not in provider._sdk_config


def test_no_override_keeps_default_prod_behavior() -> None:
    provider = _make_provider({"api_key": "k"})
    assert provider._base_url is None


def test_override_forwarded_to_sdk_client() -> None:
    """A non-None base_url must reach the llama-cloud client constructor."""
    provider = _make_provider({"api_key": "k", "base_url": "http://localhost:8000"})

    captured: dict = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        # Short-circuit the rest of _parse_pdf; we only care about init kwargs.
        raise RuntimeError("stop after client init")

    with patch.object(llamaparse_module, "LlamaCloud", MagicMock(side_effect=fake_client)):
        # _parse_pdf wraps the RuntimeError raised after client init.
        with pytest.raises(ProviderPermanentError):
            provider._parse_pdf("dummy.pdf")

    assert captured.get("base_url") == "http://localhost:8000"
    assert captured.get("api_key") == "k"
