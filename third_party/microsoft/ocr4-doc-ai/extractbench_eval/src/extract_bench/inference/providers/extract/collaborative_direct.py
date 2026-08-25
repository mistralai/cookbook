"""Collaborative direct extraction provider.

Runs a first direct-model extraction, then asks a second direct-model
extraction to verify/fix the first model's JSON draft against the same source
document and schema.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import Any

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.direct_model_utils import (
    DEFAULT_USER_INSTRUCTION,
    normalize_extract_result,
)
from extract_bench.inference.providers.registry import create_provider, register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType

DEFAULT_REVIEW_USER_INSTRUCTION_TEMPLATE = (
    "{base_instruction}\n\n"
    "Another extraction model produced the following JSON draft. Treat it as a "
    "non-authoritative reference, not as ground truth. Use the source document "
    "as the source of truth. Keep values from the draft only when they are "
    "supported by the document. Review and fix the entire JSON output required "
    "by the schema, including top-level fields, nested objects, arrays, array "
    "items, scalar values, enum/category values, evidence fields, and required "
    "null or empty values. Fix any missing, incorrect, duplicated, misplaced, "
    "malformed, incomplete, or schema-incompatible content. Do not mention the "
    "draft, changes, uncertainty, or reasoning. Return only JSON that matches "
    "the schema.\n\n"
    "Draft JSON:\n"
    "{draft_json}"
)

_STAGE_META_KEYS = {"provider_name", "pipeline_name", "config"}


@register_provider("collaborative_extract")
class CollaborativeDirectExtractProvider(Provider):
    """Two-stage direct extraction wrapper with simple fallback semantics."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._first_stage = self._parse_stage_spec("first_stage")
        self._second_stage = self._parse_stage_spec("second_stage")
        self._review_user_instruction_template = str(
            self.base_config.get("review_user_instruction_template") or DEFAULT_REVIEW_USER_INSTRUCTION_TEMPLATE
        )

    def _parse_stage_spec(self, key: str) -> PipelineSpec:
        stage = self.base_config.get(key)
        if not isinstance(stage, dict):
            raise ProviderConfigError(f"collaborative_extract requires a {key} config object")

        provider_name = stage.get("provider_name")
        if not isinstance(provider_name, str) or not provider_name:
            raise ProviderConfigError(f"{key}.provider_name must be a non-empty string")

        pipeline_name = stage.get("pipeline_name")
        if not isinstance(pipeline_name, str) or not pipeline_name:
            pipeline_name = f"collaborative_{key}_{provider_name}"

        config = stage.get("config")
        if config is None:
            config = {name: copy.deepcopy(value) for name, value in stage.items() if name not in _STAGE_META_KEYS}
        if not isinstance(config, dict):
            raise ProviderConfigError(f"{key}.config must be an object when provided")

        return PipelineSpec(
            pipeline_name=pipeline_name,
            provider_name=provider_name,
            product_type=ProductType.EXTRACT,
            config=copy.deepcopy(config),
        )

    def _review_stage_spec(self, first_data: Any) -> PipelineSpec:
        draft_json = json.dumps(first_data, ensure_ascii=False, indent=2, sort_keys=True)
        base_instruction = str(self._second_stage.config.get("user_instruction") or DEFAULT_USER_INSTRUCTION)
        user_instruction = self._review_user_instruction_template.format(
            base_instruction=base_instruction,
            draft_json=draft_json,
        )
        config = {**self._second_stage.config, "user_instruction": user_instruction}
        return self._second_stage.model_copy(update={"config": config})

    @staticmethod
    def _run_stage(stage: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        provider = create_provider(stage)
        return provider.run_inference(stage, request)

    @staticmethod
    def _error_payload(error: Exception) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        if isinstance(error, ProviderError):
            if error.job_id is not None:
                payload["job_id"] = error.job_id
            if error.debug_payload is not None:
                payload["debug_payload"] = error.debug_payload
        return payload

    @staticmethod
    def _stage_payload(label: str, raw_result: RawInferenceResult | None, error: Exception | None) -> dict[str, Any]:
        if raw_result is None:
            return {
                "label": label,
                "status": "failed",
                "error": CollaborativeDirectExtractProvider._error_payload(error) if error else None,
            }

        raw_output = raw_result.raw_output
        return {
            "label": label,
            "status": "succeeded",
            "pipeline_name": raw_result.pipeline_name,
            "provider_name": raw_result.pipeline.provider_name,
            "model": raw_output.get("model"),
            "latency_in_ms": raw_result.latency_in_ms,
            "num_pages": raw_output.get("num_pages"),
            "cost_usd": raw_output.get("cost_usd"),
            "cost_per_page_usd": raw_output.get("cost_per_page_usd"),
            "usage": raw_output.get("usage"),
            "raw_output": raw_output,
        }

    @staticmethod
    def _token_count(raw_output: dict[str, Any], *keys: str) -> int:
        usage = raw_output.get("usage")
        for key in keys:
            value = raw_output.get(key)
            if value is None and isinstance(usage, dict):
                value = usage.get(key)
            if isinstance(value, int | float):
                return int(value)
        return 0

    @classmethod
    def _aggregate_usage(cls, raw_results: list[RawInferenceResult]) -> dict[str, Any]:
        input_tokens = 0
        output_tokens = 0
        thinking_tokens = 0
        total_tokens = 0
        by_stage: list[dict[str, Any]] = []

        for raw_result in raw_results:
            raw_output = raw_result.raw_output
            stage_input = cls._token_count(raw_output, "input_tokens")
            stage_output = cls._token_count(raw_output, "output_tokens")
            stage_thinking = cls._token_count(raw_output, "thinking_tokens", "reasoning_tokens")
            stage_total = cls._token_count(raw_output, "total_tokens")
            if stage_total == 0:
                stage_total = stage_input + stage_output + stage_thinking

            input_tokens += stage_input
            output_tokens += stage_output
            thinking_tokens += stage_thinking
            total_tokens += stage_total
            by_stage.append(
                {
                    "label": raw_result.raw_output.get("_collaboration_stage"),
                    "pipeline_name": raw_result.pipeline_name,
                    "provider_name": raw_result.pipeline.provider_name,
                    "model": raw_output.get("model"),
                    "input_tokens": stage_input,
                    "output_tokens": stage_output,
                    "thinking_tokens": stage_thinking,
                    "total_tokens": stage_total,
                }
            )

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total_tokens,
            "num_api_calls": len(raw_results),
            "by_stage": by_stage,
        }

    @staticmethod
    def _failure_to_raise(first_error: Exception | None, second_error: Exception | None) -> ProviderError:
        errors = [error for error in (first_error, second_error) if error is not None]
        message = "Both collaborative extraction stages failed: " + "; ".join(
            f"{type(error).__name__}: {error}" for error in errors
        )
        if any(isinstance(error, ProviderConfigError) for error in errors):
            return ProviderConfigError(message)
        if any(isinstance(error, ProviderRateLimitError) for error in errors):
            return ProviderRateLimitError(message)
        if any(isinstance(error, ProviderTransientError) for error in errors):
            return ProviderTransientError(message)
        return ProviderPermanentError(message)

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

        first_raw: RawInferenceResult | None = None
        first_error: Exception | None = None
        second_raw: RawInferenceResult | None = None
        second_error: Exception | None = None

        try:
            first_raw = self._run_stage(self._first_stage, request)
            first_raw.raw_output["_collaboration_stage"] = "first"
        except Exception as exc:
            first_error = exc

        try:
            if first_raw is not None:
                second_stage = self._review_stage_spec(first_raw.raw_output.get("data") or {})
            else:
                second_stage = self._second_stage
            second_raw = self._run_stage(second_stage, request)
            second_raw.raw_output["_collaboration_stage"] = "second"
        except Exception as exc:
            second_error = exc

        if second_raw is not None:
            selected_raw = second_raw
            selection_reason = "second_stage_success"
        elif first_raw is not None:
            selected_raw = first_raw
            selection_reason = "second_stage_failed_returning_first_stage"
        else:
            raise self._failure_to_raise(first_error, second_error)

        completed_at = datetime.now()
        successful_stages = [raw for raw in (first_raw, second_raw) if raw is not None]
        usage = self._aggregate_usage(successful_stages)
        cost_usd = sum(float(raw.raw_output.get("cost_usd") or 0.0) for raw in successful_stages)
        num_pages = int(selected_raw.raw_output.get("num_pages") or 0)

        raw_output = {
            "data": selected_raw.raw_output.get("data") or {},
            "model": selected_raw.raw_output.get("model"),
            "selected_stage": selected_raw.raw_output.get("_collaboration_stage"),
            "selection_reason": selection_reason,
            "num_pages": num_pages,
            "cost_usd": cost_usd,
            "usage": usage,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "thinking_tokens": usage["thinking_tokens"],
            "total_tokens": usage["total_tokens"],
            "num_api_calls": usage["num_api_calls"],
            "collaboration": {
                "first_stage": self._stage_payload("first", first_raw, first_error),
                "second_stage": self._stage_payload("second", second_raw, second_error),
            },
            "_config": {
                "first_stage": {
                    "pipeline_name": self._first_stage.pipeline_name,
                    "provider_name": self._first_stage.provider_name,
                    "config": self._first_stage.config,
                },
                "second_stage": {
                    "pipeline_name": self._second_stage.pipeline_name,
                    "provider_name": self._second_stage.provider_name,
                    "config": self._second_stage.config,
                },
            },
        }
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
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        return normalize_extract_result(raw_result)
