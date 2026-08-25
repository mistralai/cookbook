"""Direct Gemini document extraction provider."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.direct_model_utils import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_INSTRUCTION,
    IMAGE_EXTENSIONS,
    add_additional_properties_false,
    normalize_extract_result,
    page_count,
    pricing_for_model,
    promote_repeated_structure,
)
from extract_bench.inference.providers.extract.parsed_text_source import (
    ParsedDocumentText,
    ParseTextSource,
    create_parse_text_source,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType

_GEMINI_EXTRACT_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.7-flash": (0.75, 3.75),
}


@register_provider("gemini_extract")
class GeminiDirectExtractProvider(Provider):
    """Gemini direct extract provider using native JSON schema output."""

    DEFAULT_MODEL = "gemini-3-flash-preview"

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        self._api_key = (
            self.base_config.get("api_key")
            or os.getenv("GOOGLE_GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not self._api_key:
            raise ProviderConfigError(
                "Gemini API key is required. Set GOOGLE_GEMINI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY "
                "or pass api_key in base_config."
            )

        self._model: str = self.base_config.get("model", self.DEFAULT_MODEL)
        self._schema_name: str = self.base_config.get("schema_name", "extraction")
        self._strict: bool = bool(self.base_config.get("strict", False))
        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))
        self._system_prompt: str = self.base_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self._user_instruction: str = self.base_config.get("user_instruction", DEFAULT_USER_INSTRUCTION)
        self._max_cost_usd: float | None = self.base_config.get("max_cost_usd")
        max_tokens_cfg = self.base_config.get("max_tokens", 65536)
        self._max_tokens: int | None = int(max_tokens_cfg) if max_tokens_cfg is not None else None
        self._thinking_level: str | None = self.base_config.get("thinking_level")

        self._input_mode: str = self.base_config.get("input_mode", "file")
        if self._input_mode not in ("file", "parsed_text"):
            raise ProviderConfigError(f"input_mode must be 'file' or 'parsed_text', got {self._input_mode!r}")
        self._parse_text_source: ParseTextSource | None = (
            create_parse_text_source(self.base_config.get("parse_source"))
            if self._input_mode == "parsed_text"
            else None
        )
        default_input_price_per_1m, default_output_price_per_1m = self._pricing_for_model(self._model)
        self._input_price_per_1m: float = float(self.base_config.get("input_price_per_1m", default_input_price_per_1m))
        self._output_price_per_1m: float = float(
            self.base_config.get("output_price_per_1m", default_output_price_per_1m)
        )

        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ProviderConfigError("google-genai package not installed. Run: pip install google-genai") from e

        self._client = genai.Client(api_key=self._api_key)
        self._types = types

    @staticmethod
    def _pricing_for_model(model: str) -> tuple[float, float]:
        return pricing_for_model(model, _GEMINI_EXTRACT_PRICING_PER_M)

    def _prepare_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        schema = promote_repeated_structure(schema)
        if self._additional_properties_false:
            return add_additional_properties_false(schema)
        return schema

    @staticmethod
    def _mime_type(source_path: Path) -> str:
        ext = source_path.suffix.lower()
        if ext == ".pdf":
            return "application/pdf"
        if ext in IMAGE_EXTENSIONS:
            return IMAGE_EXTENSIONS[ext]
        raise ProviderPermanentError(
            f"GeminiDirectExtractProvider supports PDFs and {set(IMAGE_EXTENSIONS)}, got {source_path.suffix}"
        )

    @staticmethod
    def _finish_reason(response: Any) -> str | None:
        for candidate in getattr(response, "candidates", None) or []:
            reason = getattr(candidate, "finish_reason", None)
            if reason is None:
                continue
            return getattr(reason, "name", None) or str(reason)
        return None

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            return text

        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text:
                    return part_text
        return ""

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
        return {
            "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(meta, "candidates_token_count", 0) or 0,
            "thinking_tokens": getattr(meta, "thoughts_token_count", 0) or 0,
            "total_tokens": getattr(meta, "total_token_count", 0) or 0,
        }

    def _call_api(
        self,
        schema: dict[str, Any],
        *,
        source_path: Path | None = None,
        document_text: str | None = None,
    ) -> dict[str, Any]:
        types = self._types
        if document_text is not None:
            parts = [
                types.Part.from_text(text=document_text),
                types.Part.from_text(text=self._user_instruction),
            ]
        elif source_path is not None:
            mime_type = self._mime_type(source_path)
            parts = [
                types.Part.from_bytes(data=source_path.read_bytes(), mime_type=mime_type),
                types.Part.from_text(text=self._user_instruction),
            ]
        else:
            raise ProviderPermanentError("_call_api requires source_path or document_text")

        config_kwargs: dict[str, Any] = {
            "temperature": 0,
            "system_instruction": self._system_prompt,
            "response_mime_type": "application/json",
            "response_json_schema": schema,
        }
        if self._max_tokens is not None:
            config_kwargs["max_output_tokens"] = self._max_tokens
        if self._thinking_level is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self._thinking_level)

        response = self._client.models.generate_content(
            model=self._model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(**config_kwargs),
        )

        finish_reason = self._finish_reason(response)
        if finish_reason == "MAX_TOKENS":
            raise ProviderPermanentError(
                "Gemini hit the provider/model output-token ceiling before completing the JSON response."
            )

        raw_text = self._extract_text(response)
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ProviderPermanentError(
                f"Model returned non-JSON output despite structured-output request (finish_reason={finish_reason}): {e}"
            ) from e

        return {
            "data": data,
            "model": self._model,
            "usage": self._extract_usage(response),
            "_config": {
                "additional_properties_false": self._additional_properties_false,
                "strict": self._strict,
                "schema_name": self._schema_name,
                "max_tokens": self._max_tokens,
                "thinking_level": self._thinking_level,
                "max_cost_usd": self._max_cost_usd,
                "input_price_per_1m": self._input_price_per_1m,
                "output_price_per_1m": self._output_price_per_1m,
            },
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"{type(self).__name__} only supports EXTRACT product type, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override."
            )

        started_at = datetime.now()
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        schema = self._prepare_schema(request.schema_override)

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
                "rate_limit",
                "resource_exhausted",
            )
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during Gemini extraction: {e}") from e
            raise ProviderPermanentError(f"Error during Gemini extraction: {e}") from e

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        usage = raw_output["usage"]
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        thinking_tokens = int(usage.get("thinking_tokens", 0) or 0)
        extract_cost_usd = (
            input_tokens / 1_000_000 * self._input_price_per_1m
            + (output_tokens + thinking_tokens) / 1_000_000 * self._output_price_per_1m
        )
        cost_usd = extract_cost_usd
        num_pages = parsed_doc.num_pages if parsed_doc and parsed_doc.num_pages > 0 else page_count(file_path)
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
        raw_output["cost_exceeded_budget"] = self._max_cost_usd is not None and cost_usd > self._max_cost_usd
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
        return normalize_extract_result(raw_result)
