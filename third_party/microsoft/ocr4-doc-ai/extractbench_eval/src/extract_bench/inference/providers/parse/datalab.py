"""Provider for Datalab PARSE."""

import asyncio
import dataclasses
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from datalab_sdk import AsyncDatalabClient
from datalab_sdk.models import ConvertOptions
from pypdf import PdfReader

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

IMAGE_SOURCE_SUFFIXES = {".png", ".jpg", ".jpeg", ".jfif"}

# Datalab JSON block_type -> Canonical17 label
DATALAB_LABEL_MAP: dict[str, str] = {
    "Text": "Text",
    "SectionHeader": "Section-header",
    "Table": "Table",
    "Figure": "Picture",
    "Picture": "Picture",
    "ListGroup": "List-item",
    "PageHeader": "Page-header",
    "PageFooter": "Page-footer",
    "Caption": "Caption",
    "Footnote": "Footnote",
    "Formula": "Formula",
    "Equation": "Formula",
    "Code": "Code",
    "Form": "Form",
    "Handwriting": "Text",
    "TableOfContents": "Document Index",
}


def _build_layout_pages(json_data: dict[str, Any]) -> list[ParseLayoutPageIR]:
    """Build layout_pages from Datalab JSON output for layout cross-evaluation.

    Datalab JSON structure:
      {"children": [<page>, ...], "metadata": {...}}
    Each page:
      {"block_type": "Page", "bbox": [0, 0, w, h], "children": [<block>, ...]}
    Each block:
      {"block_type": "Text", "bbox": [x1, y1, x2, y2], "html": "...", "children": [...]}
    """
    pages = json_data.get("children", [])
    layout_pages: list[ParseLayoutPageIR] = []

    for page_idx, page in enumerate(pages):
        if page.get("block_type") != "Page":
            continue

        page_bbox = page.get("bbox", [0, 0, 1, 1])
        page_w = float(page_bbox[2]) if len(page_bbox) >= 3 else 1.0
        page_h = float(page_bbox[3]) if len(page_bbox) >= 4 else 1.0
        if page_w <= 0:
            page_w = 1.0
        if page_h <= 0:
            page_h = 1.0

        items: list[LayoutItemIR] = []

        for block in page.get("children", []):
            block_type = block.get("block_type", "")
            canonical_label = DATALAB_LABEL_MAP.get(block_type)
            if canonical_label is None:
                continue

            bbox = block.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

            # Normalize pixel coords to [0,1] xywh
            nx = x1 / page_w
            ny = y1 / page_h
            nw = (x2 - x1) / page_w
            nh = (y2 - y1) / page_h

            seg = LayoutSegmentIR(
                x=nx,
                y=ny,
                w=nw,
                h=nh,
                confidence=1.0,
                label=canonical_label,
            )

            content = block.get("html", "") or ""
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
                page_number=page_idx + 1,
                width=page_w,
                height=page_h,
                items=items,
            )
        )

    return layout_pages


def _is_pdf_file(file_path: str | Path) -> bool:
    try:
        with Path(file_path).open("rb") as f:
            return f.read(4) == b"%PDF"
    except OSError:
        return False


def _get_source_page_count(source_path: str | Path) -> int | None:
    path = Path(source_path)
    if _is_pdf_file(path):
        try:
            return len(PdfReader(path).pages)
        except Exception:
            return None

    if path.suffix.lower() in IMAGE_SOURCE_SUFFIXES:
        return 1

    # Datalab supports non-PDF document inputs too. Prefer the API-reported
    # page_count when available.
    return None


@register_provider("datalab")
class DatalabProvider(Provider):
    """
    Provider for Datalab PARSE.

    This provider uses the Datalab API (powered by Marker/Surya) for parsing tasks.
    Uses the /api/v1/convert endpoint via datalab-python-sdk.
    """

    COST_PER_PAGE_USD_BY_MODE = {
        "fast": 0.004,
        "balanced": 0.004,
        "accurate": 0.01,
    }

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `api_key`: Datalab API key (defaults to DATALAB_API_KEY env var)
            - `output_format`: Output format - "markdown", "html", "json", or "chunks"
              (default: "html"). SDK default is "markdown".
              Use "html,json" for both parse eval (html) and layout eval (json bboxes).
            - `max_pages`: Maximum number of pages to parse (default: 25)
            - `mode`: Processing mode - "fast", "balanced", or "accurate"
              (default: "balanced"). SDK default is "fast".
            - `skip_cache` / `invalidate_cache`: Skip server-side caching (default: False)
            - `extras`: Comma-separated extra features, e.g. "chart_understanding,table_row_bboxes"
        """
        super().__init__(provider_name, base_config)

        # Get API key
        self._api_key = self.base_config.get("api_key") or os.getenv("DATALAB_API_KEY")
        if not self._api_key:
            raise ProviderConfigError(
                "Datalab API key is required. Set DATALAB_API_KEY environment variable or pass api_key in base_config."
            )

        # Get configuration with defaults
        self._output_format = self.base_config.get("output_format", "html")
        self._max_pages = self.base_config.get("max_pages", 25)
        self._mode = self.base_config.get("mode", "balanced")
        if self._mode not in self.COST_PER_PAGE_USD_BY_MODE:
            raise ProviderConfigError(
                f"Unsupported Datalab parse mode: {self._mode!r}. "
                f"Expected one of {sorted(self.COST_PER_PAGE_USD_BY_MODE)}."
            )
        self._skip_cache = self.base_config.get("skip_cache", self.base_config.get("invalidate_cache", False))
        self._extras = self.base_config.get("extras", None)

    async def _parse_file_async(self, source_path: str) -> dict[str, Any]:
        """
        Parse a document or image using Datalab API (async).

        :param source_path: Path to the source file
        :return: Raw API response as dictionary
        :raises ProviderError: For any API errors
        """
        try:
            num_pages = _get_source_page_count(source_path)

            # Create convert options
            options = ConvertOptions(
                output_format=self._output_format,
                max_pages=self._max_pages,
                mode=self._mode,
                skip_cache=self._skip_cache,
            )
            if self._extras and hasattr(options, "extras"):
                options.extras = self._extras

            # Parse the source file asynchronously
            async with AsyncDatalabClient(api_key=self._api_key) as client:
                result = await client.convert(source_path, options=options)

            # Use dataclasses.asdict() for clean serialization
            raw_response = dataclasses.asdict(result)

            # Store the configuration used for reference
            raw_response["_config"] = {
                "output_format": self._output_format,
                "max_pages": self._max_pages,
                "mode": self._mode,
                "total_pages": num_pages,
            }

            # Cost tracking
            page_count = raw_response.get("page_count") or num_pages or 1
            cost_usd = page_count * self.COST_PER_PAGE_USD_BY_MODE[self._mode]
            raw_response["cost_usd"] = cost_usd
            raw_response["cost_per_page_usd"] = cost_usd / max(page_count, 1)

            return raw_response

        except (ProviderTransientError, ProviderPermanentError):
            raise
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
                f"DatalabProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        # Check if file exists
        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")

        try:
            # Run async parsing
            raw_output = asyncio.run(self._parse_file_async(str(source_path)))

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
            raise
        except ProviderTransientError:
            raise
        except Exception as e:
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
                f"DatalabProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        # Extract content based on the output format
        markdown = ""
        output_format = raw_result.raw_output.get("_config", {}).get("output_format", "html")

        if "markdown" in output_format:
            markdown = raw_result.raw_output.get("markdown", "") or ""
        elif "html" in output_format:
            markdown = raw_result.raw_output.get("html", "") or ""
        elif "json" in output_format:
            # JSON-only: fall back to markdown if available
            markdown = raw_result.raw_output.get("markdown", "") or ""
        elif "chunks" in output_format:
            markdown = raw_result.raw_output.get("markdown", "") or ""

        # Build layout_pages from JSON if available
        layout_pages: list[ParseLayoutPageIR] = []
        json_data = raw_result.raw_output.get("json")
        if json_data and isinstance(json_data, dict):
            layout_pages = _build_layout_pages(json_data)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
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
