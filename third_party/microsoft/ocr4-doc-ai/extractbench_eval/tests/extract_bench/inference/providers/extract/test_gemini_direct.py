from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

from extract_bench.inference.providers.base import ProviderConfigError
from extract_bench.inference.providers.extract import gemini_direct
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType


def _pipeline(model: str = "gemini-3.6-flash", thinking_level: str | None = "medium") -> PipelineSpec:
    return PipelineSpec(
        pipeline_name=f"test_{model.replace('.', '_')}",
        provider_name="gemini_extract",
        product_type=ProductType.EXTRACT,
        config={"model": model, "additional_properties_false": True, "thinking_level": thinking_level},
    )


def _request(source_file_path: str) -> InferenceRequest:
    return InferenceRequest(
        example_id="example-1",
        source_file_path=source_file_path,
        product_type=ProductType.EXTRACT,
        schema_override={
            "type": "object",
            "properties": {"invoice_number": {"type": "string"}},
        },
    )


def _response(
    data: dict[str, Any] | None = None,
    finish_reason: str = "STOP",
) -> SimpleNamespace:
    payload = json.dumps(data if data is not None else {"invoice_number": "INV-001"})
    candidate = SimpleNamespace(
        finish_reason=SimpleNamespace(name=finish_reason),
        content=SimpleNamespace(parts=[SimpleNamespace(text=payload)]),
    )
    usage = SimpleNamespace(
        prompt_token_count=1000,
        candidates_token_count=200,
        thoughts_token_count=100,
        total_token_count=1300,
    )
    return SimpleNamespace(
        candidates=[candidate],
        text=payload,
        usage_metadata=usage,
    )


class _FakeModels:
    def __init__(self, response: Any | None = None, error: Exception | None = None):
        self.response = response or _response()
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, models: _FakeModels):
        self.models = models


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    models: _FakeModels,
) -> None:
    fake_client = _FakeClient(models)

    from google import genai

    monkeypatch.setattr(genai, "Client", lambda **kwargs: fake_client)


def test_gemini_direct_init_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ProviderConfigError, match="Gemini API key is required"):
        gemini_direct.GeminiDirectExtractProvider("gemini_extract")


def test_gemini_direct_pricing_for_models() -> None:
    assert gemini_direct.GeminiDirectExtractProvider._pricing_for_model("gemini-3.6-flash") == (1.50, 7.50)
    assert gemini_direct.GeminiDirectExtractProvider._pricing_for_model("gemini-3.7-flash") == (0.75, 3.75)
    assert gemini_direct.GeminiDirectExtractProvider._pricing_for_model("gemini-3.5-flash") == (1.50, 9.00)
    assert gemini_direct.GeminiDirectExtractProvider._pricing_for_model("gemini-3-flash") == (0.50, 3.00)


def test_gemini_direct_returns_normalized_extract_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "test-key")
    fake_models = _FakeModels()
    _patch_client(monkeypatch, fake_models)

    img_path = tmp_path / "doc.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

    provider = gemini_direct.GeminiDirectExtractProvider(
        "gemini_extract",
        base_config={"model": "gemini-3.6-flash", "thinking_level": "medium"},
    )
    pipeline = _pipeline(model="gemini-3.6-flash", thinking_level="medium")
    request = _request(str(img_path))

    raw_result = provider.run_inference(pipeline, request)
    normalized = provider.normalize(raw_result)

    assert normalized.output.task_type == "extract"
    assert normalized.output.example_id == "example-1"
    assert normalized.output.extracted_data == {"invoice_number": "INV-001"}

    # Cost calculation: 1000 input tokens * $1.50/M + (200 output + 100 thinking) * $7.50/M
    # = 0.0015 + 0.00225 = 0.00375
    assert raw_result.raw_output["cost_usd"] == pytest.approx(0.00375)
    assert len(fake_models.calls) == 1
    call = fake_models.calls[0]
    assert call["model"] == "gemini-3.6-flash"
    thinking_level = call["config"].thinking_config.thinking_level
    assert getattr(thinking_level, "value", str(thinking_level)).upper() == "MEDIUM"


def test_gemini_direct_schema_preparation() -> None:
    provider = gemini_direct.GeminiDirectExtractProvider(
        "gemini_extract",
        base_config={"api_key": "test-key", "additional_properties_false": True},
    )
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array"}},
        "repeated_structure": {"line_items": {"type": "array"}},
    }
    prepared = provider._prepare_schema(schema)
    assert "repeated_structure" not in prepared
    assert "line_items" in prepared["properties"]
    assert prepared["additionalProperties"] is False
