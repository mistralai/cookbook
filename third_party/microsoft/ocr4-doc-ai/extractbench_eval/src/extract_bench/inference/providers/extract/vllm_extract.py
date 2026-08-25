"""vLLM vision-model one-shot extraction provider.

Talks to an OpenAI-compatible vLLM server that hosts a vision-language model
(e.g. the Qwen3.6-35B-A3B-FP8 self-hosted deployment). The document is rasterized to
one image per page and sent directly to the model — no upstream parse stage —
mirroring the ``*_oneshot_structured_output_file`` cloud-API pipelines but for a
self-hosted vLLM endpoint.

Structured output is requested via the OpenAI ``response_format`` json_schema
form, which vLLM turns into guided decoding (xgrammar). The extract schema is
also inlined into the prompt so the model sees the field names/descriptions.
"""

from __future__ import annotations

import base64
import io
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
    IMAGE_EXTENSIONS,
    add_additional_properties_false,
    normalize_extract_result,
    page_count,
    promote_repeated_structure,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType


@register_provider("vllm_extract")
class VLLMExtractProvider(Provider):
    """One-shot structured extraction through an OpenAI-compatible vLLM VLM.

    Configuration options:
        - server_url (str, required): vLLM server base URL (self-hosted deployment).
        - model (str, required): Served model name (e.g. "qwen3.6-35b-a3b-fp8").
        - api_key_env (str, default "VLLM_API_KEY"): env var for the bearer key.
        - dpi (int, default 150): rasterization DPI for PDF pages.
        - max_pages (int | None, default None): cap on pages sent (None = all).
        - max_tokens (int, default 32768): output-token ceiling.
        - temperature (float, default 0.0): sampling temperature.
        - structured_output (bool, default True): use response_format json_schema
          guided decoding; when False fall back to json_object mode.
        - additional_properties_false (bool, default True): close every object.
        - schema_name (str, default "extraction"): json_schema name field.
        - strict (bool, default False): json_schema strict flag.
        - timeout_s (float, default 900): request timeout.
        - input_price_per_1m / output_price_per_1m (float, default 0.0):
          self-hosted, so cost defaults to 0; override to attribute compute.
        - max_cost_usd (float | None): optional per-doc budget flag.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        # There is no default endpoint: the deployment is yours. A pipeline can
        # name the env var that carries its URL (`endpoint_env_var`) so several
        # self-hosted models can share this provider without hardcoding hosts.
        server_url = self.base_config.get("server_url")
        endpoint_env_var = self.base_config.get("endpoint_env_var")
        if not server_url and endpoint_env_var:
            server_url = os.environ.get(str(endpoint_env_var), "")
        if not server_url:
            raise ProviderConfigError(
                "vllm_extract requires 'server_url' in config"
                + (f" or the {endpoint_env_var} environment variable." if endpoint_env_var else ".")
            )
        self._server_url: str = str(server_url).rstrip("/")

        self._model: str = self.base_config.get("model", "")
        if not self._model:
            raise ProviderConfigError("vllm_extract requires 'model' in config.")

        api_key_env = self.base_config.get("api_key_env", "VLLM_API_KEY")
        # vLLM tolerates any bearer when --api-key is unset; use a dummy fallback.
        self._api_key: str = os.environ.get(api_key_env, "") or "dummy"

        self._dpi: int = int(self.base_config.get("dpi", 150))
        max_pages_cfg = self.base_config.get("max_pages")
        self._max_pages: int | None = int(max_pages_cfg) if max_pages_cfg is not None else None
        self._max_tokens: int = int(self.base_config.get("max_tokens", 32768))
        self._temperature: float = float(self.base_config.get("temperature", 0.0))
        self._structured_output: bool = bool(self.base_config.get("structured_output", True))
        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))
        self._schema_name: str = self.base_config.get("schema_name", "extraction")
        self._strict: bool = bool(self.base_config.get("strict", False))
        self._system_prompt: str = self.base_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self._user_instruction: str = self.base_config.get("user_instruction", DEFAULT_USER_INSTRUCTION)
        self._timeout_s: float = float(self.base_config.get("timeout_s", self.base_config.get("timeout", 900.0)))
        self._max_cost_usd: float | None = self.base_config.get("max_cost_usd")

        self._input_price_per_1m: float = float(self.base_config.get("input_price_per_1m", 0.0))
        self._output_price_per_1m: float = float(self.base_config.get("output_price_per_1m", 0.0))

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=f"{self._server_url}/v1",
            timeout=self._timeout_s,
            # A timed-out generation continues on the remote vLLM server.
            # SDK-level retries create duplicate orphaned generations that
            # consume every GPU slot, so leave retry policy to the benchmark.
            max_retries=0,
        )

    # -- internals ----------------------------------------------------------

    def _prepare_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        schema = promote_repeated_structure(schema)
        if self._additional_properties_false:
            return add_additional_properties_false(schema)
        return schema

    def _render_page_images(self, source_path: Path) -> list[str]:
        """Return a list of base64-encoded PNGs, one per (capped) page."""
        ext = source_path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return [base64.b64encode(source_path.read_bytes()).decode("utf-8")]
        if ext != ".pdf":
            raise ProviderPermanentError(
                f"vllm_extract supports PDFs and {set(IMAGE_EXTENSIONS)}, got {source_path.suffix}"
            )
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ProviderPermanentError("pdf2image is required for vllm_extract.") from e

        try:
            images = convert_from_path(str(source_path), dpi=self._dpi)
        except Exception as e:
            raise ProviderPermanentError(f"Error converting PDF to images: {e}") from e
        if not images:
            raise ProviderPermanentError(f"No pages found in PDF: {source_path}")
        if self._max_pages is not None:
            images = images[: self._max_pages]

        out: list[str] = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return out

    def _build_user_content(self, page_images: list[str], schema: dict[str, Any]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}} for b64 in page_images
        ]
        text = (
            "Extract every field from the attached document page image(s) according to the JSON schema below.\n\n"
            "JSON schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Return only one valid JSON object matching the schema. Do not wrap it in markdown fences.\n\n"
            f"{self._user_instruction}"
        )
        content.append({"type": "text", "text": text})
        return content

    def _response_format(self, schema: dict[str, Any]) -> dict[str, Any]:
        if not self._structured_output:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self._schema_name,
                "schema": schema,
                "strict": self._strict,
            },
        }

    def _call_api(self, schema: dict[str, Any], page_images: list[str]) -> dict[str, Any]:
        # Build the request as a dict[str, Any] and splat it in: the OpenAI SDK's
        # typed create() overloads reject our list/dict message+response_format
        # shapes under strict mypy, and passing **kwargs sidesteps that (same
        # approach as deepseek_extract / glm_extract).
        kwargs: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._build_user_content(page_images, schema)},
            ],
            "response_format": self._response_format(schema),
        }
        response = self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise ProviderPermanentError(
                "vLLM model hit max_tokens before completing the JSON response. Increase max_tokens."
            )

        content = getattr(choice.message, "content", "") or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ProviderPermanentError(
                f"vLLM model returned non-JSON output despite structured-output request "
                f"(finish_reason={finish_reason}): {e}"
            ) from e

        return {
            "data": data,
            "model": self._model,
            "usage": self._extract_usage(response),
            "_config": {
                "provider": "vllm_extract",
                "server_url": self._server_url,
                "dpi": self._dpi,
                "max_pages": self._max_pages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "structured_output": self._structured_output,
                "additional_properties_false": self._additional_properties_false,
                "strict": self._strict,
                "max_cost_usd": self._max_cost_usd,
                "input_price_per_1m": self._input_price_per_1m,
                "output_price_per_1m": self._output_price_per_1m,
            },
        }

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
        output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
        total_tokens = getattr(usage, "total_tokens", None) or input_tokens + output_tokens
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
        }

    # -- Provider interface -------------------------------------------------

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
            page_images = self._render_page_images(file_path)
            raw_output = self._call_api(schema, page_images)
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
            )
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during vLLM extraction: {e}") from e
            raise ProviderPermanentError(f"Error during vLLM extraction: {e}") from e

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        usage = raw_output["usage"]
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        extract_cost_usd = (
            input_tokens / 1_000_000 * self._input_price_per_1m + output_tokens / 1_000_000 * self._output_price_per_1m
        )
        cost_usd = extract_cost_usd
        num_pages = page_count(file_path)

        raw_output["extract_cost_usd"] = extract_cost_usd
        raw_output["cost_usd"] = cost_usd
        raw_output["cost_exceeded_budget"] = self._max_cost_usd is not None and cost_usd > self._max_cost_usd
        raw_output["num_pages"] = num_pages
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
