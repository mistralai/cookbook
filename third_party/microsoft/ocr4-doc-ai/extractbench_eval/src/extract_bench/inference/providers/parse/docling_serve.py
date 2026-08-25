"""Provider for Docling via the official docling-serve HTTP API."""

import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from docling_core.transforms.serializer.html import HTMLTableSerializer
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling_core.types.doc.base import ImageRefMode
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

_MD_PAGE_BREAK_PLACEHOLDER = "<!-- page-break -->"


@register_provider("docling_serve")
class DoclingServeProvider(Provider):
    """
    Provider for Docling PDF parsing via the official docling-serve HTTP API.

    This provider sends PDFs to the docling-serve HTTP API endpoint and returns markdown
    with tables formatted as HTML. It was tested with docling-serve v1.17.0.
    """

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Docling Serve provider.

        Args:
            provider_name: Name of the provider
            base_config: Optional configuration with:
                - `api_key`: Optional bearer token for the endpoint
                - `endpoint_url`: Endpoint URL (required)
                - `timeout`: Request timeout in seconds (default: 120)
        """
        super().__init__(provider_name, base_config)

        self._api_key = self.base_config.get("api_key") or os.getenv("DOCLING_SERVE_API_KEY") or ""

        # Get endpoint URL (from config or env var)
        self._endpoint_url = self.base_config.get("endpoint_url") or os.getenv("DOCLING_SERVE_ENDPOINT_URL")
        self._endpoint_url = self._endpoint_url.rstrip("/")
        if not self._endpoint_url:
            raise ProviderConfigError(
                "Docling Serve endpoint URL is required. "
                "Set DOCLING_SERVE_ENDPOINT_URL environment variable or "
                "pass endpoint_url in pipeline config."
            )

        # Get timeout (default 120 seconds - PDF processing can be slow)
        self._timeout = self.base_config.get("timeout", 120)

    def _call_endpoint(self, pdf_bytes: bytes, filename: str) -> dict[str, Any]:
        """
        Call the Docling endpoint with PDF bytes.

        Args:
            pdf_bytes: Raw PDF file bytes
            filename: Name of the PDF file

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
            "sources": [
                {
                    "base64_string": pdf_base64,
                    "filename": filename,
                    "kind": "file",
                }
            ],
            "options": {
                "to_formats": ["json"],
                "pipeline": "standard",
                "include_images": False,
                "image_export_mode": "placeholder",
            },
        }

        try:
            response = requests.post(
                f"{self._endpoint_url}/v1/convert/source",
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
            if status_code == 422:
                raise ProviderPermanentError(
                    "Docling Serve returned 422. Ensure docling-serve >= 1.0 "
                    f"(older versions expect 'file_sources' instead of 'sources'): {e}"
                ) from e
            elif status_code == 429:
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
                f"DoclingServeProvider only supports PARSE product type, got {request.product_type}"
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
            raw_output = self._call_endpoint(pdf_bytes, source_path.name)

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
                f"DoclingServeProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        # Response format:
        # {
        #   "document": {
        #     "json_content": {...},
        #   }
        # }
        full_markdown = ""
        raw_docling_document = raw_result.raw_output.get("document", {}).get("json_content")
        pages: list[PageIR] = []

        layout_pages: list[ParseLayoutPageIR] = []
        if raw_docling_document is not None:
            try:
                docling_document = DoclingDocument.model_validate(raw_docling_document)
            except Exception as e:
                raise ProviderPermanentError(f"Failed to validate docling_document payload: {e}") from e

            doc_serializer = MarkdownDocSerializer(doc=docling_document)
            doc_serializer.table_serializer = HTMLTableSerializer()

            full_markdown = doc_serializer.serialize(
                page_break_placeholder=_MD_PAGE_BREAK_PLACEHOLDER, image_mode=ImageRefMode.PLACEHOLDER
            ).text
            raw_pages_md = full_markdown.split(_MD_PAGE_BREAK_PLACEHOLDER)
            raw_pages_dicts = []

            for page_index, markdown in enumerate(raw_pages_md):
                pages.append(PageIR(page_index=page_index, markdown=markdown))
                raw_pages_dicts.append({"page": page_index + 1, "markdown": markdown})

            layout_pages = _build_docling_layout_pages(
                doc=docling_document,
                raw_pages=raw_pages_dicts,
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
