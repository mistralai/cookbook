from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

from extract_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract import anthropic_direct
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="anthropic_haiku_4_5_extract_oneshot_structured_output_file",
        provider_name="anthropic_extract",
        product_type=ProductType.EXTRACT,
        config={"model": "claude-haiku-4-5-20251001", "additional_properties_false": True},
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
    text: str = '{"invoice_number": "INV-001"}',
    *,
    parsed_output: Any | None = None,
    stop_reason: str = "end_turn",
) -> SimpleNamespace:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        stop_reason=stop_reason,
    )
    if parsed_output is not None:
        response.parsed_output = parsed_output
    return response


class _FakeMessages:
    def __init__(self, response: Any | None = None, error: Exception | None = None):
        self.response = response or _response()
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, messages: _FakeMessages, beta_messages: _FakeMessages):
        self.messages = messages
        self.beta = SimpleNamespace(messages=beta_messages)


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    messages: _FakeMessages,
    beta_messages: _FakeMessages | None = None,
) -> None:
    fake_client = _FakeClient(messages, beta_messages or _FakeMessages())

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key: fake_client)


def test_anthropic_direct_init_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ProviderConfigError, match="ANTHROPIC_API_KEY"):
        anthropic_direct.AnthropicDirectExtractProvider("anthropic_extract")


def test_anthropic_direct_returns_normalized_extract_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "invoice.png"
    source.write_bytes(b"fake-image")
    messages = _FakeMessages()
    _patch_client(monkeypatch, messages)

    provider = anthropic_direct.AnthropicDirectExtractProvider("anthropic_extract", {"api_key": "test-key"})
    raw = provider.run_inference(_pipeline(), _request(str(source)))
    result = provider.normalize(raw)

    assert result.output.extracted_data == {"invoice_number": "INV-001"}
    assert result.output.field_citations == []
    assert raw.raw_output["model"] == "claude-haiku-4-5-20251001"
    assert raw.raw_output["num_pages"] == 1
    assert messages.calls[0]["model"] == "claude-haiku-4-5-20251001"
    assert messages.calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_direct_uses_parsed_output_when_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "invoice.png"
    source.write_bytes(b"fake-image")
    messages = _FakeMessages(response=_response("ignored", parsed_output={"invoice_number": "INV-002"}))
    _patch_client(monkeypatch, messages)

    provider = anthropic_direct.AnthropicDirectExtractProvider("anthropic_extract", {"api_key": "test-key"})
    raw = provider.run_inference(_pipeline(), _request(str(source)))

    assert raw.raw_output["data"] == {"invoice_number": "INV-002"}


def test_anthropic_direct_usage_fields_and_cost_math(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "invoice.png"
    source.write_bytes(b"fake-image")
    messages = _FakeMessages()
    _patch_client(monkeypatch, messages)

    provider = anthropic_direct.AnthropicDirectExtractProvider("anthropic_extract", {"api_key": "test-key"})
    raw = provider.run_inference(_pipeline(), _request(str(source)))

    assert raw.raw_output["usage"] == {
        "input_tokens": 1000,
        "output_tokens": 200,
        "total_tokens": 1200,
    }
    assert raw.raw_output["cost_usd"] == pytest.approx((1000 * 1.00 + 200 * 5.00) / 1_000_000)
    assert raw.raw_output["cost_per_page_usd"] == pytest.approx(raw.raw_output["cost_usd"])


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_anthropic_direct_refusal_and_max_tokens_become_permanent_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stop_reason: str,
) -> None:
    source = tmp_path / "invoice.png"
    source.write_bytes(b"fake-image")
    messages = _FakeMessages(response=_response(stop_reason=stop_reason))
    _patch_client(monkeypatch, messages)

    provider = anthropic_direct.AnthropicDirectExtractProvider("anthropic_extract", {"api_key": "test-key"})

    with pytest.raises(ProviderPermanentError, match=stop_reason):
        provider.run_inference(_pipeline(), _request(str(source)))


def test_anthropic_direct_invalid_json_becomes_permanent_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "invoice.png"
    source.write_bytes(b"fake-image")
    _patch_client(monkeypatch, _FakeMessages(response=_response("not-json")))

    provider = anthropic_direct.AnthropicDirectExtractProvider("anthropic_extract", {"api_key": "test-key"})

    with pytest.raises(ProviderPermanentError, match="non-JSON"):
        provider.run_inference(_pipeline(), _request(str(source)))


def test_anthropic_direct_network_errors_become_transient(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "invoice.png"
    source.write_bytes(b"fake-image")
    _patch_client(monkeypatch, _FakeMessages(error=RuntimeError("connection timeout")))

    provider = anthropic_direct.AnthropicDirectExtractProvider("anthropic_extract", {"api_key": "test-key"})

    with pytest.raises(ProviderTransientError, match="Transient error"):
        provider.run_inference(_pipeline(), _request(str(source)))


def test_anthropic_direct_pdfs_use_beta_messages_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"not-a-real-pdf")
    messages = _FakeMessages()
    beta_messages = _FakeMessages()
    _patch_client(monkeypatch, messages, beta_messages)

    provider = anthropic_direct.AnthropicDirectExtractProvider("anthropic_extract", {"api_key": "test-key"})
    raw = provider.run_inference(_pipeline(), _request(str(source)))

    assert raw.raw_output["data"] == {"invoice_number": "INV-001"}
    assert messages.calls == []
    assert beta_messages.calls[0]["betas"] == ["pdfs-2024-09-25"]


def test_anthropic_direct_provider_model_pricing() -> None:
    assert anthropic_direct.AnthropicDirectExtractProvider._pricing_for_model("claude-haiku-4-5-20251001") == (
        1.00,
        5.00,
    )
    assert anthropic_direct.AnthropicDirectExtractProvider._pricing_for_model("claude-sonnet-4-6") == (3.00, 15.00)
    assert anthropic_direct.AnthropicDirectExtractProvider._pricing_for_model("claude-opus-4-8") == (5.00, 25.00)
    assert anthropic_direct.AnthropicDirectExtractProvider._pricing_for_model("unknown") == (0.0, 0.0)


def test_split_union_type_enums_rewrites_nullable_enum_into_any_of() -> None:
    # Anthropic 400s on {"type": ["string", "null"], "enum": [...]}; the anyOf
    # encoding is the only accepted form that still permits null.
    schema = {
        "type": "object",
        "properties": {
            "holdings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "put_call": {
                            "type": ["string", "null"],
                            "enum": ["Put", "Call", None],
                            "description": "Option class.",
                        },
                        "shares_or_principal_type": {"type": "string", "enum": ["SH", "PRN"]},
                    },
                },
            }
        },
    }

    out = anthropic_direct.split_union_type_enums(schema)

    put_call = out["properties"]["holdings"]["items"]["properties"]["put_call"]
    assert put_call == {
        "description": "Option class.",
        "anyOf": [{"type": "string", "enum": ["Put", "Call"]}, {"type": "null"}],
    }
    # Single-typed enums are already valid and must be left untouched.
    assert out["properties"]["holdings"]["items"]["properties"]["shares_or_principal_type"] == {
        "type": "string",
        "enum": ["SH", "PRN"],
    }
    # Input is not mutated.
    assert schema["properties"]["holdings"]["items"]["properties"]["put_call"]["type"] == ["string", "null"]


def test_split_union_type_enums_keeps_null_when_enum_omits_it() -> None:
    # The union declared null, so null stays reachable even though the author
    # left it out of the enum list — otherwise absent fields force a made-up value.
    out = anthropic_direct.split_union_type_enums({"type": ["string", "null"], "enum": ["Put", "Call"]})
    assert out == {"anyOf": [{"type": "string", "enum": ["Put", "Call"]}, {"type": "null"}]}


def test_split_union_type_enums_collapses_single_branch() -> None:
    # Only one declared type survives — emit it inline rather than a 1-branch anyOf.
    out = anthropic_direct.split_union_type_enums({"type": ["string", "integer"], "enum": ["SH", "PRN"]})
    assert out == {"type": "string", "enum": ["SH", "PRN"]}


class _FakeParseSource:
    def __init__(self, parsed: Any):
        self.parsed = parsed
        self.requests: list[InferenceRequest] = []

    def parse(self, request: InferenceRequest) -> Any:
        self.requests.append(request)
        return self.parsed


def test_anthropic_direct_parsed_text_mode_sends_text_and_totals_parse_cost(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from extract_bench.inference.providers.extract.parsed_text_source import ParsedDocumentText

    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"not-a-real-pdf")
    messages = _FakeMessages()
    beta_messages = _FakeMessages()
    _patch_client(monkeypatch, messages, beta_messages)
    fake_parse = _FakeParseSource(
        ParsedDocumentText(
            text="<!-- Page 1 -->\n\n# Invoice INV-001",
            num_pages=3,
            parse_cost_usd=0.0375,
            metadata={"type": "llamaparse", "tier": "agentic"},
        )
    )
    monkeypatch.setattr(anthropic_direct, "create_parse_text_source", lambda cfg: fake_parse)

    provider = anthropic_direct.AnthropicDirectExtractProvider(
        "anthropic_extract",
        {"api_key": "test-key", "input_mode": "parsed_text", "parse_source": {"tier": "agentic"}},
    )
    raw = provider.run_inference(_pipeline(), _request(str(source)))

    # Text input — no PDF beta even though the source file is a PDF.
    assert beta_messages.calls == []
    sent_content = messages.calls[0]["messages"][0]["content"]
    assert sent_content[0] == {"type": "text", "text": "<!-- Page 1 -->\n\n# Invoice INV-001"}

    extract_cost = (1000 * 1.00 + 200 * 5.00) / 1_000_000
    assert raw.raw_output["extract_cost_usd"] == pytest.approx(extract_cost)
    assert raw.raw_output["parse_cost_usd"] == pytest.approx(0.0375)
    assert raw.raw_output["cost_usd"] == pytest.approx(extract_cost + 0.0375)
    assert raw.raw_output["num_pages"] == 3
    assert raw.raw_output["parsed_text"] == "<!-- Page 1 -->\n\n# Invoice INV-001"
    assert raw.raw_output["_config"]["input_mode"] == "parsed_text"
