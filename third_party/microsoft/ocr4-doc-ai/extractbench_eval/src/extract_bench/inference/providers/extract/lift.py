"""Provider for the lift (datalab-to/lift) extract self-hosted server.

``lift`` is datalab's structured-extraction VLM (9B, Qwen 3.5-based). Given the
document plus a JSON schema it emits a schema-constrained JSON object (vLLM
guided decoding). This provider posts the source file + schema to the
``lift_sdk_server.py`` ``/extract`` endpoint, which runs the official
``lift-pdf`` SDK against the vLLM backend (all pages in one request) and returns
the parsed ``extraction`` dict.

Unlike the Datalab cloud extract provider, lift returns no per-field citation
block-ids / bounding boxes — it is pure schema-shaped JSON extraction — so
``field_citations`` is always empty. The bench scores the extracted values
(``extract_unified_value_f1``); evidence-bbox metrics are N/A for this model.
"""

import asyncio
import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
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


@register_provider("lift_extract")
class LiftExtractProvider(Provider):
    """
    Provider for the lift extract self-hosted SDK server.

    Configuration options:
        - ``server_url`` (str, required): the lift SDK ``/extract`` endpoint URL
          (e.g. ``https://<your-lift-deployment>.modal.run``).
        - ``timeout`` (int, default 900): per-request timeout in seconds. lift
          sends every page in one request and retries internally, so multi-page
          docs (plus a possible backend cold start) can take a while.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("LIFT_ENDPOINT_URL")
        if not server_url:
            raise ProviderConfigError(
                "lift_extract provider requires 'server_url' in config or the "
                "LIFT_ENDPOINT_URL environment variable (a self-hosted lift SDK "
                "/extract endpoint). Example: https://<your-lift-deployment>.modal.run"
            )
        self._server_url: str = str(server_url)
        self._timeout = int(self.base_config.get("timeout", 900))

    async def _extract_async(self, file_bytes: bytes, filename: str, schema: dict[str, Any]) -> dict[str, Any]:
        file_b64 = base64.b64encode(file_bytes).decode()
        payload: dict[str, Any] = {
            "file_base64": file_b64,
            "filename": filename,
            "schema": schema,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._server_url.rstrip("/"),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    # 422 here is NOT a client error: serverless HTTP ingress
                    # intermittently drops the request body (FastAPI then reports
                    # json_invalid → 422) when the serving container is saturated
                    # with concurrent in-flight requests. It succeeds on retry, so
                    # treat it as transient alongside the usual 408/429/5xx.
                    if resp.status in (408, 422, 429, 502, 503, 504):
                        raise ProviderTransientError(f"HTTP {resp.status}: {error_text[:200]}")
                    raise ProviderPermanentError(f"HTTP {resp.status}: {error_text[:200]}")

                result: dict[str, Any] = await resp.json()

        if result.get("status") == "error":
            raise ProviderPermanentError(f"lift SDK error: {result.get('error', 'unknown')}")

        # error=True means lift exhausted its own retries (repeat tokens / vLLM
        # errors); extraction=None means the output was not parseable JSON.
        # Either way the doc produced no usable extraction — surface as a failure
        # so the run does not silently report <100% success as success.
        if result.get("error"):
            raise ProviderPermanentError("lift returned error=True (generation failed after retries)")
        if result.get("extraction") is None:
            raise ProviderPermanentError("lift returned no parseable extraction (extraction=None)")

        result["_config"] = {
            "server_url": self._server_url,
            "timeout": self._timeout,
        }
        return result

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"LiftExtractProvider only supports EXTRACT product type, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override"
            )

        started_at = datetime.now()

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        try:
            raw_output = asyncio.run(
                self._extract_async(
                    file_bytes=file_path.read_bytes(),
                    filename=file_path.name,
                    schema=request.schema_override,
                )
            )

            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

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

        except (ProviderPermanentError, ProviderTransientError):
            raise
        except TimeoutError as e:
            raise ProviderTransientError(f"Request timed out after {self._timeout}s") from e
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"LiftExtractProvider only supports EXTRACT product type, got {raw_result.product_type}"
            )

        extraction = raw_result.raw_output.get("extraction")
        extracted_data = extraction if isinstance(extraction, (dict, list)) else {}

        output = ExtractOutput(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data,
            field_citations=[],  # lift emits no per-field citations / bboxes
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
