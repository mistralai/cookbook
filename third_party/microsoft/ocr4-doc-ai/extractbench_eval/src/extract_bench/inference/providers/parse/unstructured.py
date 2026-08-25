"""Provider for Unstructured PARSE."""

import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

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

# ---------------------------------------------------------------------------
# Label mapping: Unstructured element types → Canonical17 labels
# ---------------------------------------------------------------------------
UNSTRUCTURED_LABEL_MAP: dict[str, str | None] = {
    "Title": "Title",
    "NarrativeText": "Text",
    "UncategorizedText": "Text",
    "ListItem": "List-item",
    "Table": "Table",
    "Image": "Picture",
    "FigureCaption": "Caption",
    "Formula": "Formula",
    "Header": "Page-header",
    "Footer": "Page-footer",
    "Address": "Text",
    "EmailAddress": "Text",
    "CodeSnippet": "Text",
    "PageNumber": None,  # skip
    "PageBreak": None,  # skip
    "CompositeElement": None,  # skip (chunking artifact)
}

_VIRTUAL_PAGE_DIM = 1000.0


@register_provider("unstructured")
class UnstructuredProvider(Provider):
    """
    Provider for Unstructured PARSE.

    Uses the Unstructured API for document parsing and extraction.
    """

    COST_PER_PAGE_USD = 0.03  # $0.03 per page

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `api_key`: Unstructured API key (defaults to UNSTRUCTURED_API_KEY env var)
            - `server_url`: Optional custom API endpoint URL
            - `strategy`: Processing strategy - "fast", "hi_res", or "auto" (default: "hi_res")
            - `languages`: List of languages in the document (default: ["eng"])
            - `pdf_infer_table_structure`: Whether to infer table structure (default: True)
            - `skip_infer_table_types`: List of doc types to skip table inference (default: [])
            - `coordinates`: Whether to return element coordinates (default: False)
            - `include_page_breaks`: Whether to include page breaks (default: True)
            - `split_pdf_concurrency_level`: Concurrency for PDF splitting (default: 5)
            - `hi_res_model_name`: Model name for hi_res strategy (default: None)
        """
        super().__init__(provider_name, base_config)

        # Get API key
        self._api_key = self.base_config.get("api_key") or os.getenv("UNSTRUCTURED_API_KEY")
        if not self._api_key:
            raise ProviderConfigError(
                "Unstructured API key is required. "
                "Set UNSTRUCTURED_API_KEY environment variable or pass api_key in base_config."
            )

        # Get optional server URL
        self._server_url = self.base_config.get("server_url") or os.getenv("UNSTRUCTURED_API_URL")

        # Configuration options
        self._strategy = self.base_config.get("strategy", "hi_res")
        self._languages = self.base_config.get("languages", ["eng"])
        self._pdf_infer_table_structure = self.base_config.get("pdf_infer_table_structure", True)
        self._skip_infer_table_types = self.base_config.get("skip_infer_table_types", [])
        self._coordinates = self.base_config.get("coordinates", False)
        self._include_page_breaks = self.base_config.get("include_page_breaks", True)
        self._split_pdf_concurrency_level = self.base_config.get("split_pdf_concurrency_level", 5)
        self._hi_res_model_name = self.base_config.get("hi_res_model_name")

    async def _parse_document_async(self, file_path: str) -> dict[str, Any]:
        """
        Parse a document using Unstructured API (async).

        :param file_path: Path to the document file
        :return: Raw API response as dictionary
        :raises ProviderError: For any API errors
        """
        try:
            from unstructured_client import UnstructuredClient
            from unstructured_client.models import operations, shared

            # Initialize client
            client_kwargs: dict[str, Any] = {"api_key_auth": self._api_key}
            if self._server_url:
                client_kwargs["server_url"] = self._server_url

            client = UnstructuredClient(**client_kwargs)

            # Read file
            with open(file_path, "rb") as f:
                file_content = f.read()

            file_name = Path(file_path).name

            # Build partition parameters
            partition_params: dict[str, Any] = {
                "files": shared.Files(
                    content=file_content,
                    file_name=file_name,
                ),
                "strategy": self._strategy,
                "languages": self._languages,
                "coordinates": self._coordinates,
                "include_page_breaks": self._include_page_breaks,
                "split_pdf_concurrency_level": self._split_pdf_concurrency_level,
            }

            # Add optional parameters
            if self._hi_res_model_name:
                partition_params["hi_res_model_name"] = self._hi_res_model_name

            # Handle table inference settings
            if self._skip_infer_table_types:
                partition_params["skip_infer_table_types"] = self._skip_infer_table_types

            # Create request
            req = operations.PartitionRequest(partition_parameters=shared.PartitionParameters(**partition_params))

            # Execute partition request
            res = client.general.partition(request=req)

            # Convert elements to list of dicts
            elements = []
            if res.elements:
                for element in res.elements:
                    if hasattr(element, "model_dump"):
                        elements.append(element.model_dump())
                    elif hasattr(element, "dict"):
                        elements.append(element.dict())
                    elif isinstance(element, dict):
                        elements.append(element)
                    else:
                        # Try to convert to dict
                        if hasattr(element, "__iter__"):
                            elements.append(dict(element))
                        else:
                            elements.append({"text": str(element)})

            # Count unique pages from element metadata
            page_numbers: set[int] = set()
            for el in elements:
                pn = (el.get("metadata") or {}).get("page_number")
                if isinstance(pn, int) and pn > 0:
                    page_numbers.add(pn)
            num_pages = len(page_numbers)

            raw_response: dict[str, Any] = {
                "elements": elements,
                "_config": {
                    "strategy": self._strategy,
                    "languages": self._languages,
                    "coordinates": self._coordinates,
                    "include_page_breaks": self._include_page_breaks,
                    "split_pdf_concurrency_level": self._split_pdf_concurrency_level,
                    "hi_res_model_name": self._hi_res_model_name,
                },
            }

            if num_pages > 0:
                cost_usd = num_pages * self.COST_PER_PAGE_USD
                raw_response["num_pages"] = num_pages
                raw_response["cost_usd"] = cost_usd
                raw_response["cost_per_page_usd"] = cost_usd / num_pages

            return raw_response

        except ImportError as e:
            raise ProviderConfigError(
                "unstructured-client package not installed. Run: pip install unstructured-client"
            ) from e
        except Exception as e:
            error_str = str(e).lower()
            transient_keywords = [
                "timeout",
                "network",
                "connection",
                "503",
                "502",
                "504",
                "429",
                "rate limit",
            ]
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during parsing: {e}") from e
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
                f"UnstructuredProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        try:
            raw_output = self.run_async_from_sync(self._parse_document_async(str(file_path)))

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

    def _elements_to_markdown(self, elements: list[dict[str, Any]]) -> str:
        """
        Convert Unstructured elements to markdown format.

        :param elements: List of element dictionaries from Unstructured API
        :return: Markdown string
        """
        markdown_parts: list[str] = []
        current_page: int | None = None

        for element in elements:
            element_type = element.get("type", "")
            text = element.get("text", "")
            metadata = element.get("metadata", {})
            page_number = metadata.get("page_number")

            # Track page breaks
            if page_number is not None and page_number != current_page:
                if current_page is not None:
                    markdown_parts.append("")  # Add blank line between pages
                current_page = page_number

            # Skip empty elements
            if not text.strip():
                # Handle page breaks specifically
                if element_type == "PageBreak":
                    markdown_parts.append("\n---\n")
                continue

            # Convert based on element type
            if element_type == "Title":
                markdown_parts.append(f"# {text}")
            elif element_type == "Header":
                markdown_parts.append(f"## {text}")
            elif element_type == "NarrativeText":
                markdown_parts.append(text)
            elif element_type == "ListItem":
                markdown_parts.append(f"- {text}")
            elif element_type == "Table":
                # Tables may have HTML in text_as_html
                html_content = metadata.get("text_as_html", "")
                if html_content:
                    markdown_parts.append(html_content)
                else:
                    markdown_parts.append(text)
            elif element_type == "FigureCaption":
                markdown_parts.append(f"*{text}*")
            elif element_type == "Image":
                # Images may have captions
                caption = metadata.get("image_caption", "")
                if caption:
                    markdown_parts.append(f"![{caption}]({caption})")
                else:
                    markdown_parts.append(f"[Image: {text}]" if text else "[Image]")
            elif element_type == "Formula":
                markdown_parts.append(f"$${text}$$")
            elif element_type == "CodeSnippet":
                markdown_parts.append(f"```\n{text}\n```")
            elif element_type == "Address":
                markdown_parts.append(text)
            elif element_type == "EmailAddress":
                markdown_parts.append(f"<{text}>")
            elif element_type == "PageBreak":
                markdown_parts.append("\n---\n")
            else:
                # Default: just add the text
                markdown_parts.append(text)

        return "\n\n".join(markdown_parts)

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce ParseOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"UnstructuredProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        # Extract elements
        elements = raw_result.raw_output.get("elements", [])

        # Convert elements to markdown
        markdown = self._elements_to_markdown(elements)

        # Build layout_pages for layout cross-evaluation
        layout_pages = _build_layout_pages(raw_result.raw_output)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],  # Unstructured doesn't provide per-page split by default
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


def _build_layout_pages(raw_output: dict[str, Any]) -> list[ParseLayoutPageIR]:
    """Build layout_pages from Unstructured elements for layout cross-evaluation.

    Extracts bounding box coordinates from ``metadata.coordinates.points`` (PixelSpace,
    absolute pixels, top-left origin) and normalises them to [0,1] using the per-element
    ``layout_width`` / ``layout_height`` values.
    """
    elements = raw_output.get("elements", [])
    if not elements:
        return []

    # Group elements by page
    pages_items: dict[int, list[tuple[str, float, float, float, float, str, float]]] = defaultdict(list)

    for el in elements:
        el_type = el.get("type", "")
        canonical = UNSTRUCTURED_LABEL_MAP.get(el_type)
        if canonical is None:
            continue  # skip PageNumber, PageBreak, CompositeElement, unknown types

        metadata = el.get("metadata") or {}
        page_number = metadata.get("page_number", 1)
        if not isinstance(page_number, int) or page_number < 1:
            page_number = 1

        coords = metadata.get("coordinates")
        if not coords:
            continue

        points = coords.get("points")
        layout_width = coords.get("layout_width")
        layout_height = coords.get("layout_height")
        if not points or not layout_width or not layout_height:
            continue

        layout_width = float(layout_width)
        layout_height = float(layout_height)
        if layout_width <= 0 or layout_height <= 0:
            continue

        # points is [[x,y], [x,y], [x,y], [x,y]] — extract axis-aligned bbox
        xs = [float(p[0]) for p in points if len(p) >= 2]
        ys = [float(p[1]) for p in points if len(p) >= 2]
        if not xs or not ys:
            continue

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Normalize to [0, 1]
        nx = x_min / layout_width
        ny = y_min / layout_height
        nw = (x_max - x_min) / layout_width
        nh = (y_max - y_min) / layout_height

        # Extract text content
        text = el.get("text", "")
        if canonical == "Table":
            # Prefer HTML table representation for attribution
            text = metadata.get("text_as_html", "") or text

        # Use detection_class_prob if available, else default to 1.0
        confidence = float(metadata.get("detection_class_prob", 1.0))

        pages_items[page_number].append((canonical, nx, ny, nw, nh, text, confidence))

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
