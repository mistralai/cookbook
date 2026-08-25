"""Direct Anthropic document extraction provider."""

from __future__ import annotations

import base64
import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

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
    force_all_properties_required,
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

_ANTHROPIC_EXTRACT_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
}


def _matches_json_type(value: Any, json_type: str) -> bool:
    """Return whether ``value`` is an instance of the JSON Schema ``json_type``."""
    if value is None:
        return json_type == "null"
    if isinstance(value, bool):
        return json_type == "boolean"
    if isinstance(value, int):
        return json_type in ("integer", "number")
    if isinstance(value, float):
        return json_type == "number"
    if isinstance(value, str):
        return json_type == "string"
    return False


def split_union_type_enums(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite union-typed ``enum`` nodes into an equivalent ``anyOf``.

    Anthropic's structured-output grammar compiler validates each ``enum`` value
    against a *single* declared type, so the nullable-enum shape the extraction
    schemas use is rejected outright::

        {"type": ["string", "null"], "enum": ["Put", "Call", null]}
        -> 400 Enum value 'Put' does not match declared type '['string', 'null']'

    That is valid JSON Schema, so the fix is an encoding change, not a schema
    relaxation: one ``anyOf`` branch per declared type, each carrying only the
    enum values of that type. ``null`` stays allowed whenever the union declared
    it — dropping it would force the model to invent an enum value for fields
    that are legitimately absent.
    """
    out = copy.deepcopy(schema)

    def rewrite(node: dict[str, Any]) -> None:
        enum = node.get("enum")
        types = node.get("type")
        if not isinstance(enum, list) or not isinstance(types, list) or "anyOf" in node:
            return
        branches: list[dict[str, Any]] = []
        for json_type in types:
            if json_type == "null":
                branches.append({"type": "null"})
                continue
            values = [v for v in enum if _matches_json_type(v, json_type)]
            if values:
                branches.append({"type": json_type, "enum": values})
        if not branches:
            return
        for key in ("type", "enum"):
            node.pop(key, None)
        if len(branches) == 1:
            node.update(branches[0])
        else:
            node["anyOf"] = branches

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            rewrite(node)
            for value in (node.get("properties") or {}).values():
                walk(value)
            if "items" in node:
                walk(node["items"])
            for key in ("anyOf", "oneOf", "allOf"):
                for branch in node.get(key, []) or []:
                    walk(branch)
            for key in ("$defs", "definitions"):
                for definition in (node.get(key) or {}).values():
                    walk(definition)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(out)
    return out


@register_provider("anthropic_extract")
class AnthropicDirectExtractProvider(Provider):
    """Anthropic direct extract provider using native JSON schema output."""

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        self._api_key = self.base_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ProviderConfigError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key in base_config."
            )

        self._model: str = self.base_config.get("model", self.DEFAULT_MODEL)
        self._schema_name: str = self.base_config.get("schema_name", "extraction")
        self._strict: bool = bool(self.base_config.get("strict", False))
        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))
        self._all_properties_required: bool = bool(self.base_config.get("all_properties_required", True))
        self._system_prompt: str = self.base_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self._user_instruction: str = self.base_config.get("user_instruction", DEFAULT_USER_INSTRUCTION)
        self._max_tokens: int = int(self.base_config.get("max_tokens", 8192))
        self._thinking = self.base_config.get("thinking")
        self._effort = self.base_config.get("effort")

        self._input_mode: str = self.base_config.get("input_mode", "file")
        if self._input_mode not in ("file", "parsed_text"):
            raise ProviderConfigError(f"input_mode must be 'file' or 'parsed_text', got {self._input_mode!r}")
        self._parse_text_source: ParseTextSource | None = (
            create_parse_text_source(self.base_config.get("parse_source"))
            if self._input_mode == "parsed_text"
            else None
        )
        # Opus 4.7+, Opus 5 and Fable 5 reject temperature at non-default values (400).
        self._supports_temperature = not self._model.startswith(
            ("claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-fable-5")
        )
        default_input_price_per_1m, default_output_price_per_1m = self._pricing_for_model(self._model)
        self._input_price_per_1m: float = float(self.base_config.get("input_price_per_1m", default_input_price_per_1m))
        self._output_price_per_1m: float = float(
            self.base_config.get("output_price_per_1m", default_output_price_per_1m)
        )

        try:
            import anthropic
        except ImportError as e:
            raise ProviderConfigError("anthropic package not installed. Run: pip install anthropic") from e

        self._client = anthropic.Anthropic(api_key=self._api_key)

    @staticmethod
    def _pricing_for_model(model: str) -> tuple[float, float]:
        return pricing_for_model(model, _ANTHROPIC_EXTRACT_PRICING_PER_M)

    def _prepare_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        schema = promote_repeated_structure(schema)
        if self._all_properties_required:
            schema = force_all_properties_required(schema)
        if self._additional_properties_false:
            schema = add_additional_properties_false(schema)
        return split_union_type_enums(schema)

    def _output_config(self, schema: dict[str, Any]) -> dict[str, Any]:
        config: dict[str, Any] = {
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        }
        if self._effort:
            config["effort"] = self._effort
        return config

    def _extra_kwargs(self) -> dict[str, Any]:
        extra_kwargs: dict[str, Any] = {}
        if self._thinking:
            extra_kwargs["thinking"] = self._thinking
        elif self._supports_temperature:
            extra_kwargs["temperature"] = 0
        return extra_kwargs

    @staticmethod
    def _text_from_response(response: Any) -> str:
        for block in getattr(response, "content", None) or []:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    return str(block.get("text") or "")
            elif getattr(block, "type", None) == "text":
                return getattr(block, "text", "") or ""
        return ""

    @staticmethod
    def _usage_from_response(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _message_content(self, source_path: Path) -> tuple[list[dict[str, Any]], bool]:
        ext = source_path.suffix.lower()
        data = base64.standard_b64encode(source_path.read_bytes()).decode("utf-8")
        if ext == ".pdf":
            return (
                [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": data,
                        },
                    },
                    {"type": "text", "text": self._user_instruction},
                ],
                True,
            )
        if ext in IMAGE_EXTENSIONS:
            return (
                [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": IMAGE_EXTENSIONS[ext],
                            "data": data,
                        },
                    },
                    {"type": "text", "text": self._user_instruction},
                ],
                False,
            )
        raise ProviderPermanentError(
            f"AnthropicDirectExtractProvider supports PDFs and {set(IMAGE_EXTENSIONS)}, got {source_path.suffix}"
        )

    def _call_api(
        self,
        schema: dict[str, Any],
        *,
        source_path: Path | None = None,
        document_text: str | None = None,
    ) -> dict[str, Any]:
        if document_text is not None:
            content: list[dict[str, Any]] = [
                {"type": "text", "text": document_text},
                {"type": "text", "text": self._user_instruction},
            ]
            use_pdf_beta = False
        elif source_path is not None:
            content, use_pdf_beta = self._message_content(source_path)
        else:
            raise ProviderPermanentError("_call_api requires source_path or document_text")
        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": self._system_prompt,
            "timeout": httpx.Timeout(3600.0, connect=5.0),
            "messages": [{"role": "user", "content": content}],
            "output_config": self._output_config(schema),
            **self._extra_kwargs(),
        }
        if use_pdf_beta:
            response = self._client.beta.messages.create(betas=["pdfs-2024-09-25"], **kwargs)
        else:
            response = self._client.messages.create(**kwargs)

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason in {"refusal", "max_tokens"}:
            raise ProviderPermanentError(f"Anthropic extraction stopped before schema output: {stop_reason}")

        parsed_output = getattr(response, "parsed_output", None)
        if parsed_output is not None:
            data = parsed_output
        else:
            raw_text = self._text_from_response(response)
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as e:
                raise ProviderPermanentError(
                    f"Model returned non-JSON output despite structured-output request: {e}"
                ) from e

        return {
            "data": data,
            "model": self._model,
            "usage": self._usage_from_response(response),
            "_config": {
                "additional_properties_false": self._additional_properties_false,
                "all_properties_required": self._all_properties_required,
                "strict": self._strict,
                "schema_name": self._schema_name,
                "max_tokens": self._max_tokens,
                "thinking": self._thinking,
                "effort": self._effort,
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
            schema_size_keywords = (
                "too many optional parameters",
                "too many parameters with union",
            )
            if any(keyword in error_str for keyword in schema_size_keywords):
                raise ProviderPermanentError(
                    "Anthropic structured-output rejected the schema due to grammar-compiler "
                    "limits (max 24 optional and 16 union-typed properties per object). "
                    "This schema is too large for Haiku direct extract; consider migrating to "
                    f"tool-use input schemas. Original error: {e}"
                ) from e
            transient_keywords = (
                "timeout",
                "network",
                "connection",
                "503",
                "502",
                "504",
                "529",
                "429",
                "rate limit",
                "rate_limit",
            )
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during Anthropic extraction: {e}") from e
            raise ProviderPermanentError(f"Error during Anthropic extraction: {e}") from e

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        usage = raw_output["usage"]
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        extract_cost_usd = (
            input_tokens / 1_000_000 * self._input_price_per_1m + output_tokens / 1_000_000 * self._output_price_per_1m
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
