"""Provider for Reducto PARSE."""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from reducto import Reducto

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
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

# Reducto block type -> Canonical17 label string
REDUCTO_LABEL_MAP: dict[str, str] = {
    "Title": "Title",
    "Section Header": "Section-header",
    "Text": "Text",
    "Table": "Table",
    "Figure": "Picture",
    "List Item": "List-item",
    "Header": "Page-header",
    "Footer": "Page-footer",
    "Page Number": "Page-footer",
    "Key Value": "Key-Value Region",
    "Comment": "Footnote",
    # "Signature" is skipped (no canonical equivalent)
}

# Virtual page dimensions for normalized coordinate conversion.
# Since Reducto bbox is already [0,1], these scale factors cancel out
# during evaluation (pixel_coord / page_dim == original_normalized_value).
_VIRTUAL_PAGE_DIM = 1000.0


@register_provider("reducto")
class ReductoProvider(Provider):
    """
    Provider for Reducto PARSE.

    This provider uses the Reducto API for parsing tasks.
    """

    CREDIT_RATE_USD = 0.015  # $0.015 per credit

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `api_key`: Reducto API key (defaults to REDUCTO_API_KEY env var)
            - `ocr_system`: OCR system to use - "standard" or "legacy"
              (default: "standard")
            - `agentic`: Whether to use agentic enhancements (default: True)
            - `agentic_scopes`: List of agentic scopes - ["text"] or ["text", "table"]
              (default: ["text"])
            - `table_output_format`: Table output format - "html", "md", "json", "jsonbbox", "csv" or "dynamic"
              (default: "html")
            - `formatting_include`: List of Reducto formatting include flags to preserve
              additional semantic annotations such as change tracking/highlights/comments
              (default: [])
            - `advanced_chart_agent`: Enable advanced chart agent for figure agentic scope
              to convert charts/graphs to tabular format (default: False)
        """
        super().__init__(provider_name, base_config)

        # Get API key
        self._api_key = self.base_config.get("api_key") or os.getenv("REDUCTO_API_KEY")
        if not self._api_key:
            raise ProviderConfigError(
                "Reducto API key is required. Set REDUCTO_API_KEY environment variable or pass api_key in base_config."
            )

        # Get configuration with defaults
        self._ocr_system = self.base_config.get("ocr_system", "standard")
        self._agentic = self.base_config.get("agentic", True)
        self._agentic_scopes = self.base_config.get("agentic_scopes", ["text"])
        self._table_output_format = self.base_config.get("table_output_format", "html")
        self._formatting_include = self.base_config.get("formatting_include", [])
        self._advanced_chart_agent = self.base_config.get("advanced_chart_agent", False)

    def _is_pdf_file(self, file_path: str) -> bool:
        """
        Check if a file is a PDF by reading its header.

        :param file_path: Path to the file
        :return: True if the file is a PDF, False otherwise
        """
        try:
            with open(file_path, "rb") as f:
                header = f.read(4)
                # PDF files start with %PDF
                return header == b"%PDF"
        except Exception:
            # If we can't read the file, assume it's not a PDF
            return False

    def _get_page_count(self, file_path: str) -> int:
        """
        Get the page count for a file. For PDFs, reads the actual page count.
        For images, returns 1.

        :param file_path: Path to the file
        :return: Number of pages (1 for images, actual count for PDFs)
        """
        if self._is_pdf_file(file_path):
            try:
                reader = PdfReader(file_path)
                return len(reader.pages)
            except Exception:
                # If PDF reading fails, fall back to 1
                return 1
        else:
            # For images and other non-PDF files, assume 1 page
            return 1

    async def _parse_pdf_async(self, pdf_path: str) -> dict[str, Any]:
        """
        Parse a PDF using Reducto API (async).

        :param pdf_path: Path to the PDF file
        :return: Raw API response as dictionary
        :raises ProviderError: For any API errors
        """
        try:
            # Get page count (works for both PDFs and images)
            num_pages = self._get_page_count(pdf_path)

            # Initialize Reducto client
            client = Reducto(api_key=self._api_key)

            # Upload the file
            upload = await asyncio.to_thread(client.upload, file=Path(pdf_path))

            # Configure parse options
            enhance_config: dict[str, Any] = {}
            if self._agentic:
                agentic_list = []
                for scope in self._agentic_scopes:
                    scope_config: dict[str, Any] = {"scope": scope}
                    if scope == "figure" and self._advanced_chart_agent:
                        scope_config["advanced_chart_agent"] = True
                    agentic_list.append(scope_config)
                enhance_config["agentic"] = agentic_list

            formatting_config = {
                "table_output_format": self._table_output_format,
            }
            if self._formatting_include:
                formatting_config["include"] = self._formatting_include

            settings_config = {
                "ocr_system": self._ocr_system,
                # Don't specify page_range - process all pages
            }

            # Parse the document (run in executor since SDK is synchronous)
            # Build kwargs dynamically — only include enhance if non-empty,
            # since the SDK uses `omit` sentinel and passing None causes 422.
            parse_kwargs: dict[str, Any] = {
                "input": upload,
                "formatting": formatting_config,
                "settings": settings_config,
            }
            if enhance_config:
                parse_kwargs["enhance"] = enhance_config

            result = await asyncio.to_thread(
                client.parse.run,  # type: ignore[arg-type]
                **parse_kwargs,
            )

            # Capture the original Reducto API response as-is using model_dump()
            # According to https://docs.reducto.ai/parsing/response-format
            # The response has: job_id, duration, pdf_url, studio_link, usage, result
            try:
                # Try Pydantic v2 first
                if hasattr(result, "model_dump"):
                    raw_response = result.model_dump()
                # Try Pydantic v1
                elif hasattr(result, "dict"):
                    raw_response = result.dict()
                else:
                    # Fallback: manually extract if not a Pydantic model
                    raw_response = {}
                    for attr in ["job_id", "duration", "pdf_url", "studio_link", "usage", "result"]:
                        if hasattr(result, attr):
                            value = getattr(result, attr)
                            if not callable(value):
                                raw_response[attr] = value
            except Exception:
                # If model_dump fails, fall back to manual extraction
                raw_response = {}
                for attr in ["job_id", "duration", "pdf_url", "studio_link", "usage", "result"]:
                    if hasattr(result, attr):
                        value = getattr(result, attr)
                        if not callable(value):
                            raw_response[attr] = value

            # Also store the configuration used for reference
            raw_response["_config"] = {
                "ocr_system": self._ocr_system,
                "agentic": self._agentic,
                "agentic_scopes": self._agentic_scopes,
                "table_output_format": self._table_output_format,
                "formatting_include": self._formatting_include,
                "advanced_chart_agent": self._advanced_chart_agent,
                "total_pages": num_pages,
            }

            # Extract cost from API usage response
            usage = raw_response.get("usage") or {}
            credits = usage.get("credits")
            usage_pages = usage.get("num_pages") or num_pages
            if credits is not None and credits > 0:
                cost_usd = credits * self.CREDIT_RATE_USD
                raw_response["credits_used"] = credits
                raw_response["cost_usd"] = cost_usd
                raw_response["num_pages"] = usage_pages
                if usage_pages > 0:
                    raw_response["cost_per_page_usd"] = cost_usd / usage_pages

            return raw_response

        except Exception as e:
            # Check if it's a transient error (network, timeout, etc.)
            error_str = str(e).lower()
            transient_keywords = ["timeout", "network", "connection", "503", "502", "504"]
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during parsing: {e}") from e
            else:
                raise ProviderPermanentError(f"Error during parsing: {e}") from e

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        :param pipeline: Pipeline specification
        :param request: Inference request
        :return: Raw inference result
        :raises ProviderError: For any provider-related failures
        """
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"ReductoProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        # Check if file exists
        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"PDF file not found: {pdf_path}")

        try:
            # Run async parsing
            raw_output = asyncio.run(self._parse_pdf_async(str(pdf_path)))

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

        except ProviderPermanentError:
            # Re-raise provider errors as-is
            raise
        except ProviderTransientError:
            # Re-raise provider errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce ParseOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"ReductoProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        # Convert to ParseOutput
        # Reducto response structure: result.chunks[] with blocks[] that have bbox.page
        # According to docs: https://docs.reducto.ai/parsing/response-format
        # Similar to run_reducto.py, we use the first chunk's content
        result_obj = raw_result.raw_output.get("result", {})
        chunks = result_obj.get("chunks", [])

        # Handle URL-based results for large documents (>~6MB response)
        # When result.type == "url", chunks are not inline — fetch from URL
        if result_obj.get("type") == "url" and not chunks:
            import requests

            result_url = result_obj.get("url", "")
            if result_url:
                try:
                    resp = requests.get(result_url, timeout=120)
                    resp.raise_for_status()
                    chunks = resp.json()
                except Exception as e:
                    raise ProviderPermanentError(f"Failed to fetch URL-based result from Reducto: {e}") from e

        # Extract content from first chunk
        # Similar to run_reducto.py: result.result.chunks[0].content
        markdown = ""
        if chunks and len(chunks) > 0:
            markdown = chunks[0].get("content", "")

        # Build layout_pages from block-level bboxes for layout cross-evaluation
        layout_pages = _build_layout_pages(chunks)

        # Populate document-level markdown, leave pages empty
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],  # Leave pages empty
            layout_pages=layout_pages,
            markdown=markdown,
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


def _build_layout_pages(chunks: list[dict[str, Any]]) -> list[ParseLayoutPageIR]:
    """Build layout_pages from Reducto chunks/blocks for layout cross-evaluation.

    Groups blocks by page number and converts each block's normalized [0,1] bbox
    into a LayoutSegmentIR with canonical label mapping.
    """
    from collections import defaultdict

    # Group blocks by page
    pages_blocks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        for block in chunk.get("blocks", []):
            bbox_data = block.get("bbox", {})
            page_num = bbox_data.get("page", 1)
            pages_blocks[page_num].append(block)

    layout_pages: list[ParseLayoutPageIR] = []
    for page_num in sorted(pages_blocks.keys()):
        blocks = pages_blocks[page_num]
        items: list[LayoutItemIR] = []

        for block in blocks:
            block_type = block.get("type", "")
            canonical_label = REDUCTO_LABEL_MAP.get(block_type)
            if canonical_label is None:
                continue  # Skip unmapped types (e.g., Signature)

            bbox_data = block.get("bbox", {})
            left = float(bbox_data.get("left", 0.0))
            top = float(bbox_data.get("top", 0.0))
            width = float(bbox_data.get("width", 0.0))
            height = float(bbox_data.get("height", 0.0))

            # Parse confidence
            conf_raw = block.get("confidence")
            try:
                confidence = float(conf_raw) if conf_raw is not None else 1.0
            except (TypeError, ValueError):
                confidence = 1.0

            seg = LayoutSegmentIR(
                x=left,
                y=top,
                w=width,
                h=height,
                confidence=confidence,
                label=canonical_label,
            )

            content = block.get("content", "")
            norm_label = canonical_label.strip().lower()
            if norm_label == "table":
                item_type = "table"
            elif norm_label == "picture":
                item_type = "image"
            else:
                item_type = "text"

            items.append(
                LayoutItemIR(
                    type=item_type,
                    value=content,
                    bbox=seg,
                    layout_segments=[seg],
                )
            )

        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_num,
                width=_VIRTUAL_PAGE_DIM,
                height=_VIRTUAL_PAGE_DIM,
                items=items,
            )
        )

    return layout_pages
