"""DeepSeek OpenAI-compatible text extraction provider.

DeepSeek V4 is exposed through an OpenAI-compatible chat completions API and is
text-only for these model endpoints, so this provider uses the shared parse-text
stage first and asks DeepSeek for JSON output guided by the extract schema.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.direct_model_utils import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_INSTRUCTION,
    add_additional_properties_false,
    normalize_extract_result,
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

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_FIREWORKS_DEEPSEEK_V4_FLASH_MODEL = "accounts/fireworks/models/deepseek-v4-flash"
_FIREWORKS_DEEPSEEK_V4_PRO_MODEL = "accounts/fireworks/models/deepseek-v4-pro"

# USD per million tokens: input, cached input, output.
_DEEPSEEK_EXTRACT_PRICING_PER_M: dict[str, tuple[float, float, float]] = {
    "deepseek-v4-flash": (0.14, 0.0028, 0.28),
    "deepseek-v4-pro": (0.435, 0.003625, 0.87),
    _FIREWORKS_DEEPSEEK_V4_FLASH_MODEL: (0.21, 0.045, 0.42),
    _FIREWORKS_DEEPSEEK_V4_PRO_MODEL: (1.74, 0.15, 3.48),
}


@register_provider("deepseek_extract")
class DeepSeekExtractProvider(Provider):
    """Structured extraction through DeepSeek chat completions."""

    DEFAULT_MODEL = "deepseek-v4-flash"
    DEFAULT_MAX_TOKENS = None

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        self._model: str = self.base_config.get("model", self.DEFAULT_MODEL)
        self._base_url: str = self.base_config.get(
            "base_url",
            _FIREWORKS_BASE_URL if self._model.startswith("accounts/fireworks/models/") else _DEEPSEEK_BASE_URL,
        )
        self._uses_fireworks = "fireworks.ai" in self._base_url
        self._api_key = self.base_config.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
        if not self._api_key and self._uses_fireworks:
            self._api_key = os.getenv("FIREWORKS_API_KEY")
        if not self._api_key:
            if self._uses_fireworks:
                raise ProviderConfigError(
                    "The registered deepseek pipeline runs DeepSeek on Fireworks: set "
                    "FIREWORKS_API_KEY (or DEEPSEEK_API_KEY) to a Fireworks API key. "
                    "A deepseek.com key only works with a deepseek.com base_url."
                )
            raise ProviderConfigError("DeepSeek API key is required for deepseek_extract.")

        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))
        self._system_prompt: str = self.base_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self._user_instruction: str = self.base_config.get("user_instruction", DEFAULT_USER_INSTRUCTION)
        self._max_cost_usd: float | None = self.base_config.get("max_cost_usd")
        self._thinking_type: str | None = self.base_config.get("thinking_type", "disabled")
        self._reasoning_effort: str | None = self.base_config.get("reasoning_effort")
        max_tokens_cfg = self.base_config.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        self._max_tokens: int | None = int(max_tokens_cfg) if max_tokens_cfg is not None else None

        self._input_mode: str = self.base_config.get("input_mode", "parsed_text")
        if self._input_mode != "parsed_text":
            raise ProviderConfigError("deepseek_extract supports input_mode='parsed_text' only.")
        self._parse_text_source: ParseTextSource = create_parse_text_source(self.base_config.get("parse_source"))

        default_input, default_cached_input, default_output = self._pricing_for_model(self._model)
        self._input_price_per_1m: float = float(self.base_config.get("input_price_per_1m", default_input))
        self._cached_input_price_per_1m: float = float(
            self.base_config.get("cached_input_price_per_1m", default_cached_input)
        )
        self._output_price_per_1m: float = float(self.base_config.get("output_price_per_1m", default_output))

        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    @staticmethod
    def _pricing_for_model(model: str) -> tuple[float, float, float]:
        matches = [
            (prefix, rates) for prefix, rates in _DEEPSEEK_EXTRACT_PRICING_PER_M.items() if model.startswith(prefix)
        ]
        return max(matches, key=lambda item: len(item[0]))[1] if matches else (0.0, 0.0, 0.0)

    def _prepare_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        schema = promote_repeated_structure(schema)
        if self._additional_properties_false:
            return add_additional_properties_false(schema)
        return schema

    def _build_prompt(self, document_text: str, schema: dict[str, Any]) -> str:
        return (
            "Extract every field from the parsed document text according to the JSON schema.\n\n"
            "Document text:\n"
            f"{document_text}\n\n"
            "JSON schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Return only one valid JSON object. Do not wrap it in markdown fences.\n\n"
            f"{self._user_instruction}"
        )

    def _call_api(self, schema: dict[str, Any], document_text: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._build_prompt(document_text, schema)},
            ],
            "response_format": {"type": "json_object"},
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if self._thinking_type and not self._uses_fireworks:
            kwargs["extra_body"] = {"thinking": {"type": self._thinking_type}}
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise ProviderPermanentError(
                "DeepSeek hit max_tokens before completing the JSON response. Increase max_tokens."
            )

        content = getattr(choice.message, "content", "") or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ProviderPermanentError(f"DeepSeek returned non-JSON output despite response_format: {e}") from e

        usage = self._extract_usage(response)
        return {
            "data": data,
            "model": self._model,
            "usage": usage,
            "_config": {
                "provider": "deepseek",
                "base_url": self._base_url,
                "input_mode": self._input_mode,
                "parse_source": self.base_config.get("parse_source") or {},
                "additional_properties_false": self._additional_properties_false,
                "thinking_type": self._thinking_type,
                "reasoning_effort": self._reasoning_effort,
                "max_tokens": self._max_tokens,
                "max_cost_usd": self._max_cost_usd,
                **self._pricing_snapshot(),
            },
        }

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
        output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
        total_tokens = getattr(usage, "total_tokens", None) or input_tokens + output_tokens
        cached_input_tokens = getattr(usage, "prompt_cache_hit_tokens", None) or 0

        for details_name in ("prompt_tokens_details", "input_tokens_details"):
            details = getattr(usage, details_name, None)
            if details is None:
                continue
            cached_input_tokens = (
                getattr(details, "cached_tokens", None)
                or getattr(details, "cached_input_tokens", None)
                or cached_input_tokens
            )

        return {
            "input_tokens": int(input_tokens),
            "cached_input_tokens": int(cached_input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
        }

    def _estimate_extract_cost_usd(self, usage: dict[str, int]) -> float:
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
        return (
            uncached_input_tokens / 1_000_000 * self._input_price_per_1m
            + cached_input_tokens / 1_000_000 * self._cached_input_price_per_1m
            + output_tokens / 1_000_000 * self._output_price_per_1m
        )

    def _pricing_snapshot(self) -> dict[str, Any]:
        return {
            "pricing_basis": "deepseek_api_standard",
            "input_price_per_1m": self._input_price_per_1m,
            "cached_input_price_per_1m": self._cached_input_price_per_1m,
            "output_price_per_1m": self._output_price_per_1m,
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

        try:
            parsed_doc: ParsedDocumentText = self._parse_text_source.parse(request)
            raw_output = self._call_api(schema, parsed_doc.text)
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
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during DeepSeek extraction: {e}") from e
            raise ProviderPermanentError(f"Error during DeepSeek extraction: {e}") from e

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        extract_cost_usd = self._estimate_extract_cost_usd(raw_output["usage"])
        cost_usd = extract_cost_usd + parsed_doc.parse_cost_usd
        raw_output["extract_cost_usd"] = extract_cost_usd
        raw_output["parse_cost_usd"] = parsed_doc.parse_cost_usd
        raw_output["cost_usd"] = cost_usd
        raw_output["cost_exceeded_budget"] = self._max_cost_usd is not None and cost_usd > self._max_cost_usd
        raw_output["num_pages"] = parsed_doc.num_pages
        raw_output["cost_per_page_usd"] = cost_usd / parsed_doc.num_pages if parsed_doc.num_pages > 0 else None
        raw_output["parse_metadata"] = parsed_doc.metadata
        raw_output["parsed_text"] = parsed_doc.text

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
