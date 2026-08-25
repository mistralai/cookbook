"""Providers for OpenAI Responses-API document extraction.

For PDFs, the source is uploaded via OpenAI's Files API (purpose="user_data")
and referenced as ``input_file`` in a Responses-API call. For raw images
(.png, .jpg/.jpeg, .gif, .webp) the bytes are base64-encoded into a data URL
and sent as ``input_image`` content (the Responses API's image content type
does not accept PDFs, and ``input_file`` does not render images). In both
cases the call demands structured JSON output via ``text.format = json_schema``.
No rasterization or local text extraction — the model receives the document
directly.

Two-stage mode: with ``input_mode="parsed_text"`` the document is first parsed
by a configurable parse stage (``parse_source`` config, default LlamaParse
agentic — see ``parsed_text_source.py``) and the model receives the per-page
parsed markdown as ``input_text`` instead of the file. Reported ``cost_usd``
includes the parse stage, with ``parse_cost_usd`` / ``extract_cost_usd``
breakdown keys.

Schema-agnostic by design — works with any JSON schema. One optional schema
transform is applied before the call:

- ``add_additional_properties_false`` (default ON): recursively sets
  ``additionalProperties: false`` on every object schema. Closing the schema
  empirically helps the model populate optional list fields instead of
  returning null.

Cost is tracked from the response ``usage`` object (input_tokens,
output_tokens, cached input tokens when reported). Default pricing matches the
client's tier; overridable via pipeline config.
"""

from __future__ import annotations

import base64
import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from pypdf import PdfReader

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.parsed_text_source import (
    ParsedDocumentText,
    ParseTextSource,
    create_parse_text_source,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

# Image extensions the Responses API accepts directly as ``input_image``
# (sent as a base64 data URL). PDFs continue to go through the Files API as
# ``input_file`` — the Responses ``input_image`` content type does not accept
# PDFs, and the Files API + ``input_file`` path does not handle raw images.
_IMAGE_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are extracting structured data from a document according to the "
    "provided JSON schema. Return only the JSON that matches the schema. Use "
    "null for fields not present in the document. When the schema includes a "
    "list field, populate every relevant row visible in the document — do not "
    "return an empty list when rows are present."
)

DEFAULT_USER_INSTRUCTION = (
    "Extract every field from the attached document according to the schema. "
    "Return JSON only. Use null for fields not present in the document. "
    "Whenever the schema declares a list field, enumerate every row visible in "
    "the document — do not collapse rows or return an empty list when rows are "
    "present."
)

# OpenAI pricing: USD per million tokens (input, output).
# Uses longest-prefix matching to support dated model ids.
_OPENAI_RESPONSES_EXTRACT_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


# ---------------------------------------------------------------------------
# Schema transforms
# ---------------------------------------------------------------------------


def add_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively set ``additionalProperties: false`` on every object schema.

    Walks ``properties``, ``items``, ``anyOf`` / ``oneOf`` / ``allOf``, and any
    ``$defs`` / ``definitions`` blocks. Returns a new schema; does not mutate
    the input."""
    out = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                for v in (node.get("properties") or {}).values():
                    walk(v)
            if "items" in node:
                walk(node["items"])
            for key in ("anyOf", "oneOf", "allOf"):
                for branch in node.get(key, []) or []:
                    walk(branch)
            for key in ("$defs", "definitions"):
                for d in (node.get(key) or {}).values():
                    walk(d)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(out)
    return out


# ---------------------------------------------------------------------------
# Provider base
# ---------------------------------------------------------------------------


@register_provider("openai_extract")
class OpenAIResponsesExtractProvider(Provider):
    """OpenAI Responses-API extract provider."""

    DEFAULT_MODEL = "gpt-5.4"

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        self._api_key = self.base_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ProviderConfigError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key in base_config."
            )

        self._model: str = self.base_config.get("model", self.DEFAULT_MODEL)
        self._schema_name: str = self.base_config.get("schema_name", "extraction")
        self._strict: bool = bool(self.base_config.get("strict", False))
        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))
        self._system_prompt: str = self.base_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self._user_instruction: str = self.base_config.get("user_instruction", DEFAULT_USER_INSTRUCTION)

        self._input_mode: str = self.base_config.get("input_mode", "file")
        if self._input_mode not in ("file", "parsed_text"):
            raise ProviderConfigError(f"input_mode must be 'file' or 'parsed_text', got {self._input_mode!r}")
        self._parse_text_source: ParseTextSource | None = (
            create_parse_text_source(self.base_config.get("parse_source"))
            if self._input_mode == "parsed_text"
            else None
        )

        # Optional pipeline-level schema override (a JSON Schema dict). When
        # set, it replaces request.schema_override at inference time — useful
        # for benchmarking a single hand-crafted schema (e.g. a flattened
        # variant) against a dataset whose test cases ship a different one.
        # The schema is inlined into pipeline config (rather than loaded from
        # a path) so the dashboard registry stays host-independent.
        config_schema = self.base_config.get("schema_override")
        self._schema_override_from_config: dict[str, Any] | None = (
            config_schema if isinstance(config_schema, dict) else None
        )

        default_input_price_per_1m, default_output_price_per_1m = self._pricing_for_model(self._model)
        self._input_price_per_1m: float = float(self.base_config.get("input_price_per_1m", default_input_price_per_1m))
        self._output_price_per_1m: float = float(
            self.base_config.get("output_price_per_1m", default_output_price_per_1m)
        )

        self._client = OpenAI(api_key=self._api_key)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _pricing_for_model(model: str) -> tuple[float, float]:
        matches = [(p, r) for p, r in _OPENAI_RESPONSES_EXTRACT_PRICING_PER_M.items() if model.startswith(p)]
        return max(matches, key=lambda x: len(x[0]))[1] if matches else (0.0, 0.0)

    def _prepare_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if self._additional_properties_false:
            return add_additional_properties_false(schema)
        return schema

    @staticmethod
    def _page_count(source_path: Path) -> int:
        if source_path.suffix.lower() in _IMAGE_EXTENSIONS:
            return 1
        try:
            return len(PdfReader(str(source_path)).pages)
        except Exception:
            return 0

    def _build_user_content(self, source_path: Path) -> tuple[list[dict[str, Any]], str | None]:
        """Build the user-message content array for the source file.

        Returns ``(content, uploaded_file_id)``. ``uploaded_file_id`` is set only
        when we uploaded via the Files API (PDFs) and must be deleted after the
        call. For images we base64-encode the bytes into a data URL — the
        Responses ``input_image`` content type does not accept Files-API file
        ids for raw images reliably, and ``input_file`` does not render images.
        """
        ext = source_path.suffix.lower()
        if ext in _IMAGE_EXTENSIONS:
            mime = _IMAGE_EXTENSIONS[ext]
            with open(source_path, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode("utf-8")
            content: list[dict[str, Any]] = [
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{b64}",
                    "detail": "high",
                },
                {"type": "input_text", "text": self._user_instruction},
            ]
            return content, None

        uploaded = self._client.files.create(file=open(source_path, "rb"), purpose="user_data")
        content = [
            {"type": "input_file", "file_id": uploaded.id},
            {"type": "input_text", "text": self._user_instruction},
        ]
        return content, uploaded.id

    def _call_api(
        self,
        schema: dict[str, Any],
        *,
        source_path: Path | None = None,
        document_text: str | None = None,
    ) -> dict[str, Any]:
        if document_text is not None:
            user_content: list[dict[str, Any]] = [
                {"type": "input_text", "text": document_text},
                {"type": "input_text", "text": self._user_instruction},
            ]
            uploaded_file_id: str | None = None
        elif source_path is not None:
            user_content, uploaded_file_id = self._build_user_content(source_path)
        else:
            raise ProviderPermanentError("_call_api requires source_path or document_text")
        # The Responses SDK types ``input`` against a wide TypedDict union;
        # constructing the user content via a helper drops mypy's ability to
        # narrow it. The shape is correct at runtime, so silence the overload
        # mismatch here rather than duplicating both branches at the call site.
        try:
            resp = self._client.responses.create(  # type: ignore[call-overload]
                model=self._model,
                input=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_content},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": self._schema_name,
                        "schema": schema,
                        "strict": self._strict,
                    }
                },
            )
        finally:
            if uploaded_file_id is not None:
                try:
                    self._client.files.delete(uploaded_file_id)
                except Exception:
                    # Best-effort cleanup — don't fail the call on it.
                    pass

        # Parse the structured output.
        raw_text = resp.output_text
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ProviderPermanentError(
                f"Model returned non-JSON output despite structured-output request: {e}"
            ) from e

        usage = resp.usage
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None) or 0
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None) or 0
        cached_input_tokens = None
        reasoning_tokens = None
        in_details = getattr(usage, "input_tokens_details", None)
        if in_details is not None:
            cached_input_tokens = getattr(in_details, "cached_tokens", None)
        out_details = getattr(usage, "output_tokens_details", None)
        if out_details is not None:
            reasoning_tokens = getattr(out_details, "reasoning_tokens", None)

        return {
            "data": data,
            "model": self._model,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_input_tokens": cached_input_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": getattr(usage, "total_tokens", None),
            },
            "_config": {
                "additional_properties_false": self._additional_properties_false,
                "strict": self._strict,
                "input_price_per_1m": self._input_price_per_1m,
                "output_price_per_1m": self._output_price_per_1m,
            },
        }

    # -- Provider interface -------------------------------------------------

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"{type(self).__name__} only supports EXTRACT product type, got {request.product_type}"
            )
        effective_schema = self._schema_override_from_config or request.schema_override
        if not effective_schema:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override or via "
                "the schema_override pipeline config."
            )

        started_at = datetime.now()
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        schema = self._prepare_schema(effective_schema)

        parsed_doc: ParsedDocumentText | None = None
        try:
            if self._parse_text_source is not None:
                parsed_doc = self._parse_text_source.parse(request)
                raw_output = self._call_api(schema, document_text=parsed_doc.text)
            else:
                raw_output = self._call_api(schema, source_path=file_path)
        except (ProviderPermanentError, ProviderTransientError, ProviderConfigError):
            raise
        except Exception as e:
            error_str = str(e).lower()
            transient_keywords = (
                "timeout",
                "network",
                "connection",
                "503",
                "502",
                "504",
                "429",
                "rate limit",
            )
            if any(k in error_str for k in transient_keywords):
                raise ProviderTransientError(f"Transient error during extraction: {e}") from e
            raise ProviderPermanentError(f"Error during extraction: {e}") from e

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        # Cost tracking — keys read by evaluation/stats.py.
        usage = raw_output["usage"]
        in_tok = usage["input_tokens"] or 0
        out_tok = usage["output_tokens"] or 0
        extract_cost_usd = (
            in_tok / 1_000_000 * self._input_price_per_1m + out_tok / 1_000_000 * self._output_price_per_1m
        )
        cost_usd = extract_cost_usd
        num_pages = parsed_doc.num_pages if parsed_doc and parsed_doc.num_pages > 0 else self._page_count(file_path)
        if parsed_doc is not None:
            cost_usd += parsed_doc.parse_cost_usd
            raw_output["extract_cost_usd"] = extract_cost_usd
            raw_output["parse_cost_usd"] = parsed_doc.parse_cost_usd
            raw_output["parse_metadata"] = parsed_doc.metadata
            raw_output["parsed_text"] = parsed_doc.text
            raw_output["_config"]["input_mode"] = self._input_mode
            raw_output["_config"]["parse_source"] = self.base_config.get("parse_source") or {}
        raw_output["num_pages"] = num_pages
        raw_output["cost_usd"] = cost_usd
        if num_pages > 0:
            raw_output["cost_per_page_usd"] = cost_usd / num_pages

        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=latency_ms,
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"{type(self).__name__} only supports EXTRACT product type, got {raw_result.product_type}"
            )

        extracted_data = raw_result.raw_output.get("data") or {}
        output = ExtractOutput(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data,
            field_citations=[],
        )

        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
