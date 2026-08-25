"""Provider for Extend AI PARSE using the official Python SDK.

Based on Extend AI documentation: https://docs.extend.ai/product/parsing/parse
SDK: pip install extend-ai
"""

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from extend_ai import Extend
from extend_ai.core.api_error import ApiError
from extend_ai.types import FileFromId, ParseConfig, ParseConfigChunkingStrategy
from pypdf import PdfReader

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
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

# Extend block type -> Canonical17 label string
EXTEND_LABEL_MAP: dict[str, str] = {
    "heading": "Section-header",
    "section_heading": "Section-header",
    "text": "Text",
    "table": "Table",
    "figure": "Picture",
    "header": "Page-header",
    "footer": "Page-footer",
    "key_value": "Key-Value Region",
    "page_number": "Page-footer",
    "formula": "Formula",
}

# Virtual page dimensions for normalized coordinate conversion.
# Extend bboxes are converted to [0,1] using PDF page dims, so these cancel out.
_VIRTUAL_PAGE_DIM = 1000.0


@register_provider("extend_parse")
class ExtendParseProvider(Provider):
    """
    Provider for Extend AI document parsing using the official SDK.

    This provider uses the extend-ai Python SDK for parsing tasks.
    SDK Documentation: https://docs.extend.ai/developers/sd-ks

    Workflow:
    1. Upload file via client.file.upload()
    2. Call client.parse() with configuration options
    3. Return markdown content from parsed result
    """

    # Extend meters parsing in credits; one credit is $0.0125.
    CREDIT_RATE_USD = 0.0125

    # Credits billed per page, keyed by parse engine. The default engine (no
    # explicit `engine` in the pipeline config) bills at the parse_performance rate.
    ENGINE_CREDITS_PER_PAGE: dict[str, float] = {
        "parse_performance": 2.0,
        "parse_light": 0.5,
    }
    DEFAULT_CREDITS_PER_PAGE = 2.0

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `api_key`: Extend AI API key (defaults to EXTEND_API_KEY env var)
            - `base_url`: Optional base URL for different deployments
              (default: https://api.extend.ai, alternatives: https://api.us2.extend.app,
               https://api.eu1.extend.ai)
            - `timeout`: Request timeout in seconds (default: 300)
            - `chunking_strategy`: "page", "section", or "document" (default: "page")
            - `target`: Output format - "markdown" or "spatial" (default: "markdown")
        """
        super().__init__(provider_name, base_config)

        # Get API key
        api_key = self.base_config.get("api_key") or os.getenv("EXTEND_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "Extend AI API key is required. Set EXTEND_API_KEY environment variable or pass api_key in base_config."
            )

        # Configuration
        timeout = self.base_config.get("timeout", 300)

        # Initialize the Extend client
        client_kwargs: dict[str, Any] = {
            "token": api_key,
            "timeout": float(timeout),
        }

        # Optional base URL for different deployments (US2, EU1, etc.)
        base_url = self.base_config.get("base_url")
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = Extend(**client_kwargs)

        # Thread lock for file uploads
        self._upload_lock = threading.Lock()

    def _handle_api_error(self, e: ApiError, context: str) -> None:
        """Convert SDK ApiError to appropriate ProviderError."""
        status_code = getattr(e, "status_code", None)
        error_body = getattr(e, "body", str(e))

        if status_code == 429:
            raise ProviderRateLimitError(f"Rate limit exceeded during {context}: {error_body}")
        elif status_code in (502, 503, 504):
            raise ProviderTransientError(f"Transient error during {context}: {status_code} - {error_body}")
        elif status_code and status_code >= 400:
            raise ProviderPermanentError(f"Error during {context}: {status_code} - {error_body}")
        else:
            raise ProviderPermanentError(f"API error during {context}: {error_body}")

    def _is_pdf_file(self, file_path: str) -> bool:
        """
        Check if a file is a PDF by reading its header.

        :param file_path: Path to the file
        :return: True if the file is a PDF, False otherwise
        """
        try:
            with open(file_path, "rb") as f:
                header = f.read(4)
                return header == b"%PDF"
        except Exception:
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
                return 1
        else:
            return 1

    def _credits_per_page(self, pipeline_config: dict[str, Any]) -> float | None:
        """
        Resolve how many credits per page the configured engine bills.

        :param pipeline_config: Pipeline configuration options
        :return: Credits per page, or None if the engine's rate is unknown
        :raises ProviderConfigError: If an explicit override is negative
        """
        override = pipeline_config.get("credits_per_page", self.base_config.get("credits_per_page"))
        if override is not None:
            credits = float(override)
            if credits < 0:
                raise ProviderConfigError("credits_per_page must be non-negative")
            return credits

        engine = pipeline_config.get("engine")
        if engine is None:
            return self.DEFAULT_CREDITS_PER_PAGE
        # Unknown engine: omit cost rather than report a rate we cannot vouch for.
        return self.ENGINE_CREDITS_PER_PAGE.get(engine)

    def _upload_file(self, file_path: str) -> str:
        """
        Upload a file to Extend AI.

        :param file_path: Path to the file to upload
        :return: File ID from Extend AI
        :raises ProviderError: For any upload errors
        """
        try:
            with open(file_path, "rb") as f:
                upload_response = self._client.files.upload(file=f)

            # Extract file ID from response
            if hasattr(upload_response, "id"):
                return str(upload_response.id)
            elif hasattr(upload_response, "file") and hasattr(upload_response.file, "id"):
                return str(upload_response.file.id)
            elif isinstance(upload_response, dict):
                file_data = upload_response.get("file", upload_response)
                file_id = file_data.get("id") or file_data.get("fileId")
                if file_id:
                    return str(file_id)

            raise ProviderPermanentError(f"No file ID in upload response: {upload_response}")

        except ApiError as e:
            self._handle_api_error(e, "file upload")
            raise
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "timed out", "connection", "network", "readtimeout"]):
                raise ProviderTransientError(f"Transient error during file upload: {e}") from e
            raise ProviderPermanentError(f"Unexpected error during file upload: {e}") from e

    def _build_parse_config(self, pipeline_config: dict[str, Any]) -> dict[str, Any]:
        """
        Build the parse config from pipeline configuration.

        :param pipeline_config: Pipeline configuration options
        :return: Parse configuration dict
        """
        config: dict[str, Any] = {}

        # Target format: "markdown" or "spatial"
        if "target" in pipeline_config:
            config["target"] = pipeline_config["target"]

        # Chunking strategy: "page", "section", or "document"
        if "chunking_strategy" in pipeline_config:
            config["chunking_strategy"] = ParseConfigChunkingStrategy(type=pipeline_config["chunking_strategy"])

        # Block options for fine-grained control
        if "block_options" in pipeline_config:
            config["block_options"] = pipeline_config["block_options"]

        # Advanced options (OCR enhancements, page filtering)
        if "advanced_options" in pipeline_config:
            config["advanced_options"] = pipeline_config["advanced_options"]

        # Engine selection (e.g. "parse_performance")
        if "engine" in pipeline_config:
            config["engine"] = pipeline_config["engine"]

        # Engine version (e.g. "2.0.0-beta")
        if "engineVersion" in pipeline_config:
            config["engineVersion"] = pipeline_config["engineVersion"]

        return config

    def _parse_document(
        self,
        file_path: str,
        pipeline_config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Parse a document using Extend AI.

        :param file_path: Path to the document file
        :param pipeline_config: Pipeline configuration options
        :return: Raw API response with parsed content
        :raises ProviderError: For any parsing errors
        """
        # Get page count and page dimensions (for bbox normalization)
        num_pages = self._get_page_count(file_path)
        page_dims = _get_pdf_page_dims(file_path)

        # Step 1: Upload file
        file_id = self._upload_file(file_path)

        # Step 2: Build parse config
        parse_config = self._build_parse_config(pipeline_config)

        # Step 3: Call parse API
        try:
            # The Extend SDK parse method
            parse_response = self._client.parse(
                file=FileFromId(id=file_id),
                config=ParseConfig(**parse_config) if parse_config else None,
            )

            # Convert response to dict
            if hasattr(parse_response, "model_dump"):
                result = parse_response.model_dump()
            elif hasattr(parse_response, "dict"):
                result = parse_response.dict()
            elif isinstance(parse_response, dict):
                result = parse_response
            else:
                # Try to extract attributes manually
                result = {}
                for attr in [
                    "id",
                    "status",
                    "chunks",
                    "content",
                    "markdown",
                    "pages",
                    "error",
                    "fileId",
                ]:
                    if hasattr(parse_response, attr):
                        value = getattr(parse_response, attr)
                        if not callable(value):
                            result[attr] = value

            # Add metadata
            result["_extend_metadata"] = {
                "file_id": file_id,
                "num_pages": num_pages,
                "page_dims": page_dims,
                "config": parse_config,
            }

            # Operational stats: page count feeds per-page latency, credits feed cost.
            result["num_pages"] = num_pages
            credits_per_page = self._credits_per_page(pipeline_config)
            if credits_per_page is not None:
                cost_per_page_usd = credits_per_page * self.CREDIT_RATE_USD
                result["credits_used"] = credits_per_page * num_pages
                result["cost_per_page_usd"] = cost_per_page_usd
                result["cost_usd"] = cost_per_page_usd * num_pages

            return result

        except ApiError as e:
            self._handle_api_error(e, "document parsing")
            raise
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "timed out", "connection", "network", "readtimeout"]):
                raise ProviderTransientError(f"Transient error during parsing: {e}") from e
            raise ProviderPermanentError(f"Unexpected error during parsing: {e}") from e

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
                f"ExtendParseProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        # Check if file exists
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        try:
            # Run parsing with pipeline config options
            raw_output = self._parse_document(
                file_path=str(file_path),
                pipeline_config=pipeline.config,
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

        except (ProviderPermanentError, ProviderTransientError, ProviderRateLimitError):
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
                f"ExtendParseProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        raw_output = raw_result.raw_output

        # SDK 1.x wraps content under raw_output["output"]; legacy responses had it at the top level.
        # Source the chunk-bearing payload from whichever shape applies.
        payload = raw_output.get("output") if isinstance(raw_output.get("output"), dict) else raw_output

        # Extract markdown content from response
        # Extend API can return content in different formats depending on config
        markdown = ""

        # Try different response formats
        # 1. Direct markdown field
        if "markdown" in payload:
            markdown = payload["markdown"]
        # 2. Content field
        elif "content" in payload:
            content = payload["content"]
            if isinstance(content, str):
                markdown = content
            elif isinstance(content, dict):
                markdown = content.get("markdown", "") or content.get("text", "")
        # 3. Chunks array (similar to Reducto)
        elif "chunks" in payload:
            chunks = payload["chunks"]
            if chunks and isinstance(chunks, list):
                # Concatenate all chunk contents
                chunk_contents = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        chunk_content = chunk.get("content", "") or chunk.get("markdown", "")
                        if chunk_content:
                            chunk_contents.append(chunk_content)
                    elif isinstance(chunk, str):
                        chunk_contents.append(chunk)
                markdown = "\n\n".join(chunk_contents)
        # 4. Pages array
        elif "pages" in payload:
            pages = payload["pages"]
            if pages and isinstance(pages, list):
                page_contents = []
                for page in pages:
                    if isinstance(page, dict):
                        page_content = page.get("markdown", "") or page.get("content", "")
                        if page_content:
                            page_contents.append(page_content)
                    elif isinstance(page, str):
                        page_contents.append(page)
                markdown = "\n\n".join(page_contents)

        # Get job ID if available
        job_id = raw_output.get("id") or raw_output.get("job_id")

        # Build layout_pages from chunk blocks for layout cross-evaluation
        metadata = raw_output.get("_extend_metadata", {})
        page_dims = metadata.get("page_dims", {})
        chunks = payload.get("chunks", [])
        layout_pages = _build_layout_pages(chunks, page_dims)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],  # Leave pages empty for now
            layout_pages=layout_pages,
            markdown=markdown,
            job_id=str(job_id) if job_id else None,
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


def _get_pdf_page_dims(file_path: str) -> dict[int, tuple[float, float]]:
    """Read per-page dimensions (width, height) in PDF points from a PDF file.

    Returns a dict mapping 1-indexed page number to (width, height).
    Returns empty dict for non-PDF files or on error.
    """
    try:
        with open(file_path, "rb") as f:
            if f.read(4) != b"%PDF":
                return {}
        reader = PdfReader(file_path)
        dims: dict[int, tuple[float, float]] = {}
        for i, page in enumerate(reader.pages):
            box = page.mediabox
            dims[i + 1] = (float(box.width), float(box.height))
        return dims
    except Exception:
        return {}


def _build_layout_pages(
    chunks: list[dict[str, Any]],
    page_dims: dict[int, tuple[float, float]] | dict[str, Any],
) -> list[ParseLayoutPageIR]:
    """Build layout_pages from Extend chunk blocks for layout cross-evaluation.

    Iterates through chunks and their blocks, normalizes bboxes to [0,1]
    using page dimensions, and groups by page number.

    The Extend API returns bounding box coordinates in its own pixel coordinate
    system (reported in each block's ``metadata.page.width/height``).  We use
    those pixel dimensions for normalization.  The ``page_dims`` argument (PDF
    point dimensions) is only used as a fallback when block-level metadata is
    absent.
    """
    from collections import defaultdict

    # Normalize page_dims keys to int (JSON serialization may stringify them).
    # These are PDF-point dims used only as a last-resort fallback.
    norm_dims: dict[int, tuple[float, float]] = {}
    for k, v in page_dims.items():
        try:
            page_key = int(k)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                norm_dims[page_key] = (float(v[0]), float(v[1]))
        except (TypeError, ValueError):
            continue

    pages_items: dict[int, list[LayoutItemIR]] = defaultdict(list)
    pages_headers: dict[int, list[str]] = defaultdict(list)
    pages_footers: dict[int, list[str]] = defaultdict(list)

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        blocks = chunk.get("blocks", [])
        if not isinstance(blocks, list):
            continue

        for block in blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")
            canonical_label = EXTEND_LABEL_MAP.get(block_type)
            if canonical_label is None:
                continue

            bbox = block.get("boundingBox") or block.get("bounding_box") or {}
            if not isinstance(bbox, dict):
                continue

            left = float(bbox.get("left", 0.0))
            top = float(bbox.get("top", 0.0))
            right = float(bbox.get("right", 0.0))
            bottom = float(bbox.get("bottom", 0.0))

            # Extract page number and pixel dimensions from block metadata
            block_meta = block.get("metadata", {}) or {}
            block_page_meta = block_meta.get("page", {}) or {}
            page_num = block_page_meta.get("number") or block.get("page") or block.get("pageNumber") or 1
            if isinstance(page_num, str):
                try:
                    page_num = int(page_num)
                except ValueError:
                    page_num = 1

            # Use pixel dimensions from the API's block metadata (the coordinate
            # system the bbox values are expressed in).  Fall back to PDF-point
            # dims only when the API does not report per-block page dimensions.
            pixel_w = float(block_page_meta.get("width", 0))
            pixel_h = float(block_page_meta.get("height", 0))
            if pixel_w > 0 and pixel_h > 0:
                pw, ph = pixel_w, pixel_h
            else:
                pw, ph = norm_dims.get(page_num, (0, 0))

            if pw > 0 and ph > 0:
                x_norm = left / pw
                y_norm = top / ph
                w_norm = (right - left) / pw
                h_norm = (bottom - top) / ph
            else:
                # Fallback: store raw values (adapter will handle as-is)
                x_norm = left
                y_norm = top
                w_norm = right - left
                h_norm = bottom - top

            confidence = float(block.get("confidence", 1.0))

            seg = LayoutSegmentIR(
                x=x_norm,
                y=y_norm,
                w=w_norm,
                h=h_norm,
                confidence=confidence,
                label=canonical_label,
            )

            content = block.get("content", "") or block.get("text", "")
            norm_label = canonical_label.strip().lower()
            if norm_label == "table":
                item_type = "table"
            elif norm_label == "picture":
                item_type = "image"
            else:
                item_type = "text"

            pages_items[page_num].append(
                LayoutItemIR(
                    type=item_type,
                    value=content,
                    bbox=seg,
                    layout_segments=[seg],
                )
            )

            section_content = f"<page_number>{content}</page_number>" if block_type == "page_number" else content
            if canonical_label == "Page-header" and content:
                pages_headers[page_num].append(section_content)
            elif canonical_label == "Page-footer" and content:
                pages_footers[page_num].append(section_content)

    layout_pages: list[ParseLayoutPageIR] = []
    for page_num in sorted(pages_items.keys()):
        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_num,
                width=_VIRTUAL_PAGE_DIM,
                height=_VIRTUAL_PAGE_DIM,
                items=pages_items[page_num],
                page_header_markdown="\n\n".join(pages_headers.get(page_num, [])),
                page_footer_markdown="\n\n".join(pages_footers.get(page_num, [])),
            )
        )

    return layout_pages
