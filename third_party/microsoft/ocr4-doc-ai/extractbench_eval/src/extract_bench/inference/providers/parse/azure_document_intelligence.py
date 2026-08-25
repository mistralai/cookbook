"""Provider for Azure Document Intelligence PARSE."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.core.credentials import AzureKeyCredential

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

# Azure DI paragraph role -> Canonical17 label string
AZURE_DI_LABEL_MAP: dict[str, str] = {
    "title": "Title",
    "sectionHeading": "Section-header",
    "pageHeader": "Page-header",
    "pageFooter": "Page-footer",
    "footnote": "Footnote",
    "pageNumber": "Page-footer",
}

# Default label for paragraphs without a recognized role
_DEFAULT_PARAGRAPH_LABEL = "Text"

# Virtual page dimensions for normalized coordinate conversion.
# Azure DI polygons are normalized to [0,1] via page width/height, so these cancel out.
_VIRTUAL_PAGE_DIM = 1000.0


@register_provider("azure_document_intelligence")
class AzureDocumentIntelligenceProvider(Provider):
    """
    Provider for Azure Document Intelligence PARSE.

    This provider uses Azure AI Document Intelligence for parsing tasks.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `api_key`: Azure Document Intelligence API key
              (defaults to AZURE_DOCUMENT_INTELLIGENCE_KEY env var)
            - `endpoint`: Azure Document Intelligence endpoint URL
              (defaults to AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT env var)
            - `model_id`: Model to use for analysis (default: "prebuilt-layout")
              Options: "prebuilt-read", "prebuilt-layout", "prebuilt-document"
            - `output_content_format`: Output format - "text" or "markdown"
              (default: "markdown")
        """
        super().__init__(provider_name, base_config)

        # Get API key and endpoint
        self._api_key = self.base_config.get("api_key") or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        self._endpoint = self.base_config.get("endpoint") or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

        if not self._api_key:
            raise ProviderConfigError(
                "Azure Document Intelligence API key is required. "
                "Set AZURE_DOCUMENT_INTELLIGENCE_KEY environment variable "
                "or pass api_key in base_config."
            )

        if not self._endpoint:
            raise ProviderConfigError(
                "Azure Document Intelligence endpoint is required. "
                "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT environment variable "
                "or pass endpoint in base_config."
            )

        # Get configuration with defaults
        self._model_id = self.base_config.get("model_id", "prebuilt-layout")
        self._output_content_format = self.base_config.get("output_content_format", "markdown")

        # Initialize client
        self._client = DocumentIntelligenceClient(
            endpoint=self._endpoint,
            credential=AzureKeyCredential(self._api_key),
        )

    def _parse_pdf(self, pdf_path: str) -> dict[str, Any]:
        """
        Parse a PDF using Azure Document Intelligence API.

        :param pdf_path: Path to the PDF file
        :return: Raw API response as dictionary
        :raises ProviderError: For any API errors
        """
        try:
            # Read PDF file
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            # Analyze the document
            poller = self._client.begin_analyze_document(  # type: ignore[call-overload]
                self._model_id,
                body=pdf_bytes,
                output_content_format=self._output_content_format,
            )

            # Wait for completion and get result
            result: AnalyzeResult = poller.result()

            # Convert result to dictionary for raw storage
            raw_response = self._convert_result_to_dict(result)

            # Store configuration for reference
            raw_response["_config"] = {
                "model_id": self._model_id,
                "output_content_format": self._output_content_format,
            }

            return raw_response

        except FileNotFoundError as e:
            raise ProviderPermanentError(f"PDF file not found: {pdf_path}") from e
        except Exception as e:
            # Check if it's a transient error
            error_str = str(e).lower()
            transient_keywords = [
                "timeout",
                "network",
                "connection",
                "503",
                "502",
                "504",
                "429",
                "throttl",
                "rate limit",
            ]
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during parsing: {e}") from e
            else:
                raise ProviderPermanentError(f"Error during parsing: {e}") from e

    def _convert_result_to_dict(self, result: AnalyzeResult) -> dict[str, Any]:
        """
        Convert Azure Document Intelligence AnalyzeResult to dictionary.

        :param result: AnalyzeResult from Azure API
        :return: Dictionary representation of the result
        """
        response: dict[str, Any] = {}

        # Extract content (full document text/markdown)
        if result.content:
            response["content"] = result.content

        # Extract pages with their content
        if result.pages:
            pages_data = []
            for page in result.pages:
                page_dict: dict[str, Any] = {
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "unit": page.unit,
                }

                # Extract lines if available
                if page.lines:
                    page_dict["lines"] = [
                        {
                            "content": line.content,
                            "polygon": line.polygon if line.polygon else None,
                        }
                        for line in page.lines
                    ]

                # Extract words if available
                if page.words:
                    page_dict["word_count"] = len(page.words)

                pages_data.append(page_dict)

            response["pages"] = pages_data

        # Extract tables if available
        if result.tables:
            tables_data = []
            for table in result.tables:
                table_dict: dict[str, Any] = {
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "cells": [],
                }

                if table.cells:
                    for cell in table.cells:
                        cell_dict = {
                            "row_index": cell.row_index,
                            "column_index": cell.column_index,
                            "content": cell.content,
                            "row_span": cell.row_span,
                            "column_span": cell.column_span,
                        }
                        table_dict["cells"].append(cell_dict)

                tables_data.append(table_dict)

            response["tables"] = tables_data

        # Extract paragraphs if available (with bounding regions for layout)
        if result.paragraphs:
            paragraphs_data = []
            for para in result.paragraphs:
                para_dict: dict[str, Any] = {
                    "content": para.content,
                    "role": para.role if para.role else None,
                }
                if para.bounding_regions:
                    para_dict["bounding_regions"] = [
                        {
                            "page_number": br.page_number,
                            "polygon": list(br.polygon) if br.polygon else None,
                        }
                        for br in para.bounding_regions
                    ]
                paragraphs_data.append(para_dict)
            response["paragraphs"] = paragraphs_data

        # Extract tables if available (with bounding regions for layout)
        if result.tables:
            for i, table in enumerate(result.tables):
                if table.bounding_regions and i < len(response.get("tables", [])):
                    response["tables"][i]["bounding_regions"] = [
                        {
                            "page_number": br.page_number,
                            "polygon": list(br.polygon) if br.polygon else None,
                        }
                        for br in table.bounding_regions
                    ]

        # Extract figures if available (with bounding regions for layout)
        if result.figures:
            response["figures"] = [
                {
                    "caption": fig.caption.content if fig.caption else None,
                    "bounding_regions": [
                        {
                            "page_number": br.page_number,
                            "polygon": list(br.polygon) if br.polygon else None,
                        }
                        for br in fig.bounding_regions
                    ]
                    if fig.bounding_regions
                    else [],
                }
                for fig in result.figures
            ]

        # Extract key-value pairs if available
        if result.key_value_pairs:
            response["key_value_pairs"] = [
                {
                    "key": kvp.key.content if kvp.key else None,
                    "value": kvp.value.content if kvp.value else None,
                }
                for kvp in result.key_value_pairs
            ]

        return response

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
                f"AzureDocumentIntelligenceProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        # Check if file exists
        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"PDF file not found: {pdf_path}")

        try:
            # Run parsing
            raw_output = self._parse_pdf(str(pdf_path))

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

        except (ProviderPermanentError, ProviderTransientError, ProviderConfigError):
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
                f"AzureDocumentIntelligenceProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        # Extract the main content (markdown or text)
        content = raw_result.raw_output.get("content", "")

        # Build page-level data if available
        pages: list[PageIR] = []
        raw_pages = raw_result.raw_output.get("pages", [])

        if raw_pages:
            # Azure returns page boundaries, we can try to split content by pages
            # For now, we'll create page entries with line-based content
            for page_data in raw_pages:
                page_num = page_data.get("page_number", 1)
                page_index = page_num - 1  # Convert to 0-indexed

                # Reconstruct page content from lines if available
                page_content = ""
                if "lines" in page_data:
                    page_content = "\n".join(line.get("content", "") for line in page_data.get("lines", []))

                pages.append(PageIR(page_index=page_index, markdown=page_content))

        # Build layout_pages for layout cross-evaluation
        layout_pages = _build_layout_pages(raw_result.raw_output)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=content,
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


def _polygon_to_normalized_bbox(
    polygon: list[float],
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    """Convert Azure DI polygon (8 floats, 4 corner points in page units) to normalized [0,1] xywh.

    The polygon contains [x1,y1, x2,y2, x3,y3, x4,y4] in the page's coordinate
    system (typically inches). We take min/max to get axis-aligned bbox, then
    normalize by page dimensions.
    """
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    # Normalize to [0, 1]
    nx = x_min / page_width if page_width > 0 else 0.0
    ny = y_min / page_height if page_height > 0 else 0.0
    nw = (x_max - x_min) / page_width if page_width > 0 else 0.0
    nh = (y_max - y_min) / page_height if page_height > 0 else 0.0

    return (nx, ny, nw, nh)


def _build_layout_pages(raw_output: dict[str, Any]) -> list[ParseLayoutPageIR]:
    """Build layout_pages from Azure DI paragraphs/tables/figures for layout cross-evaluation.

    Groups elements by page using bounding_regions and converts Azure DI polygon
    coordinates (in page units) into normalized [0,1] LayoutSegmentIR entries.
    """
    from collections import defaultdict

    # Build page dimension lookup from pages data
    page_dims: dict[int, tuple[float, float]] = {}
    for page_data in raw_output.get("pages", []):
        page_num = page_data.get("page_number", 1)
        width = float(page_data.get("width", 1.0))
        height = float(page_data.get("height", 1.0))
        page_dims[page_num] = (width, height)

    # Collect all layout elements grouped by page: (canonical_label, nx, ny, nw, nh, content)
    pages_items: dict[int, list[tuple[str, float, float, float, float, str, float]]] = defaultdict(list)

    # Process paragraphs
    for para in raw_output.get("paragraphs", []):
        role = para.get("role")
        canonical_label = AZURE_DI_LABEL_MAP.get(role, _DEFAULT_PARAGRAPH_LABEL) if role else _DEFAULT_PARAGRAPH_LABEL
        content = para.get("content", "")

        for br in para.get("bounding_regions", []):
            page_num = br.get("page_number", 1)
            polygon = br.get("polygon")
            if not polygon or len(polygon) < 8:
                continue
            pw, ph = page_dims.get(page_num, (1.0, 1.0))
            nx, ny, nw, nh = _polygon_to_normalized_bbox(polygon, pw, ph)
            pages_items[page_num].append((canonical_label, nx, ny, nw, nh, content, 1.0))

    # Process tables
    for table in raw_output.get("tables", []):
        # Build table content from cells for attribution
        cells = table.get("cells", [])
        content = " ".join(c.get("content", "") for c in cells if c.get("content"))

        for br in table.get("bounding_regions", []):
            page_num = br.get("page_number", 1)
            polygon = br.get("polygon")
            if not polygon or len(polygon) < 8:
                continue
            pw, ph = page_dims.get(page_num, (1.0, 1.0))
            nx, ny, nw, nh = _polygon_to_normalized_bbox(polygon, pw, ph)
            pages_items[page_num].append(("Table", nx, ny, nw, nh, content, 1.0))

    # Process figures
    for fig in raw_output.get("figures", []):
        caption = fig.get("caption") or ""
        for br in fig.get("bounding_regions", []):
            page_num = br.get("page_number", 1)
            polygon = br.get("polygon")
            if not polygon or len(polygon) < 8:
                continue
            pw, ph = page_dims.get(page_num, (1.0, 1.0))
            nx, ny, nw, nh = _polygon_to_normalized_bbox(polygon, pw, ph)
            pages_items[page_num].append(("Picture", nx, ny, nw, nh, caption, 1.0))

    # Build ParseLayoutPageIR list
    layout_pages: list[ParseLayoutPageIR] = []
    for page_num in sorted(pages_items.keys()):
        items_data = pages_items[page_num]
        items: list[LayoutItemIR] = []

        for canonical_label, nx, ny, nw, nh, content, confidence in items_data:
            seg = LayoutSegmentIR(
                x=nx,
                y=ny,
                w=nw,
                h=nh,
                confidence=confidence,
                label=canonical_label,
            )

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
