"""Provider for Docling parse via a custom HTTP endpoint."""

import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from docling_core.types.doc.document import DoclingDocument

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.parse._docling_common import _build_docling_layout_pages
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.parse_output import (
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType


@register_provider("docling_parse")
class DoclingParseProvider(Provider):
    """
    Provider for Docling PDF parsing via a custom HTTP endpoint.

    This provider sends PDFs to a Docling endpoint and returns markdown
    with tables formatted as HTML.
    """

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Docling parse provider.

        Args:
            provider_name: Name of the provider
            base_config: Optional configuration with:
                - `api_key`: Optional bearer token for the endpoint
                - `hf_token`: Deprecated alias for `api_key`
                - `endpoint_url`: Endpoint URL (required)
                - `timeout`: Request timeout in seconds (default: 120)
        """
        super().__init__(provider_name, base_config)

        # Optional bearer token; keep `hf_token` / `HF_TOKEN` as a deprecated
        # fallback for backwards compatibility with the previous HF deployment.
        self._api_key = (
            self.base_config.get("api_key")
            or self.base_config.get("hf_token")
            or os.getenv("DOCLING_PARSE_API_KEY")
            or os.getenv("HF_TOKEN")
            or ""
        )

        # Get endpoint URL (from config or env var)
        self._endpoint_url = self.base_config.get("endpoint_url") or os.getenv("DOCLING_PARSE_ENDPOINT_URL")
        if not self._endpoint_url:
            raise ProviderConfigError(
                "Docling endpoint URL is required. "
                "Set DOCLING_PARSE_ENDPOINT_URL environment variable or "
                "pass endpoint_url in pipeline config."
            )

        # Get timeout (default 120 seconds - PDF processing can be slow)
        self._timeout = self.base_config.get("timeout", 120)

    def _call_endpoint(self, pdf_bytes: bytes) -> dict[str, Any]:
        """
        Call the Docling endpoint with PDF bytes.

        Args:
            pdf_bytes: Raw PDF file bytes

        Returns:
            Raw JSON response from endpoint

        Raises:
            ProviderError: For any API errors
        """
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # Encode PDF as base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        payload = {
            "inputs": {
                "pdf_base64": pdf_base64,
            }
        }

        try:
            response = requests.post(
                self._endpoint_url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            result_json = response.json()
            if isinstance(result_json, list):
                if not result_json:
                    raise ProviderPermanentError("Endpoint returned an empty list response.")
                first_result = result_json[0]
                if not isinstance(first_result, dict):
                    raise ProviderPermanentError("Endpoint returned a list response with a non-dict payload.")
                result = first_result
            elif isinstance(result_json, dict):
                result = result_json
            else:
                raise ProviderPermanentError(
                    f"Endpoint returned unsupported response type: {type(result_json).__name__}"
                )
            return result

        except requests.exceptions.Timeout as e:
            raise ProviderTransientError(f"Request timed out: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ProviderTransientError(f"Connection error: {e}") from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            if status_code == 429:
                raise ProviderRateLimitError(f"Rate limit exceeded: {e}") from e
            elif status_code and 500 <= status_code < 600:
                raise ProviderTransientError(f"Server error ({status_code}): {e}") from e
            elif status_code and 400 <= status_code < 500:
                raise ProviderPermanentError(f"Client error ({status_code}): {e}") from e
            else:
                raise ProviderPermanentError(f"HTTP error: {e}") from e
        except (ProviderPermanentError, ProviderTransientError, ProviderRateLimitError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error calling endpoint: {e}") from e

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        Args:
            pipeline: Pipeline specification
            request: Inference request

        Returns:
            Raw inference result

        Raises:
            ProviderError: For any provider-related failures
        """
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DoclingParseProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        # Check if file exists
        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")

        try:
            # Read PDF bytes
            pdf_bytes = source_path.read_bytes()

            # Call endpoint
            raw_output = self._call_endpoint(pdf_bytes)

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

        except (ProviderPermanentError, ProviderTransientError, ProviderRateLimitError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce ParseOutput.

        Args:
            raw_result: Raw inference result from run_inference()

        Returns:
            Inference result with ParseOutput

        Raises:
            ProviderError: For any normalization failures
        """
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DoclingParseProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        # Extract pages from response
        # Response format:
        # {
        #   "pages": [{"page": 1, "markdown": "..."}, ...],
        #   "markdown": "...",
        #   "docling_document": {...},
        # }
        raw_pages = raw_result.raw_output.get("pages", [])
        full_markdown = raw_result.raw_output.get("markdown", "")
        raw_docling_document = raw_result.raw_output.get("docling_document")

        # Convert to PageIR list (0-indexed)
        pages: list[PageIR] = []
        for page_data in raw_pages:
            # Docling uses 1-indexed pages, we use 0-indexed
            page_number = page_data.get("page", 1)
            page_index = page_number - 1 if page_number > 0 else 0
            markdown = page_data.get("markdown", "")

            pages.append(PageIR(page_index=page_index, markdown=markdown))

        # Sort by page index
        pages.sort(key=lambda p: p.page_index)

        # If we have pages but no full markdown, concatenate
        if pages and not full_markdown:
            full_markdown = "\n\n".join(p.markdown for p in pages)

        layout_pages: list[ParseLayoutPageIR] = []
        if raw_docling_document is not None:
            try:
                docling_document = DoclingDocument.model_validate(raw_docling_document)
            except Exception as e:
                raise ProviderPermanentError(f"Failed to validate docling_document payload: {e}") from e

            layout_pages = _build_docling_layout_pages(
                doc=docling_document,
                raw_pages=[page for page in raw_pages if isinstance(page, dict)],
            )

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=full_markdown,
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
