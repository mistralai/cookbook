"""Provider for AWS Textract document parsing."""

import os
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

# Textract LAYOUT_* BlockType -> Canonical17 label string
TEXTRACT_LABEL_MAP: dict[str, str] = {
    "LAYOUT_TITLE": "Title",
    "LAYOUT_SECTION_HEADER": "Section-header",
    "LAYOUT_TEXT": "Text",
    "LAYOUT_TABLE": "Table",
    "LAYOUT_FIGURE": "Picture",
    "LAYOUT_LIST": "List-item",
    "LAYOUT_HEADER": "Page-header",
    "LAYOUT_FOOTER": "Page-footer",
    "LAYOUT_PAGE_NUMBER": "Page-footer",
    "LAYOUT_KEY_VALUE": "Key-Value Region",
}

# Virtual page dimensions for normalized coordinate conversion.
# Textract BoundingBox is already [0,1], so these cancel out during evaluation.
_VIRTUAL_PAGE_DIM = 1000.0


@register_provider("textract")
class TextractProvider(Provider):
    """
    Provider for AWS Textract document parsing.

    Extracts text, tables, and forms from PDFs and images using AWS Textract.
    Tables are converted to HTML to preserve their visual structure.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `aws_access_key_id`: AWS access key (or use AWS_ACCESS_KEY_ID env var)
            - `aws_secret_access_key`: AWS secret key (or use AWS_SECRET_ACCESS_KEY env var)
            - `aws_region`: AWS region (default: "us-east-1", or use AWS_REGION env var)
            - `output_tables_as_html`: Whether to output tables as HTML (default: True)
            - `detect_tables`: Whether to detect tables (default: True)
            - `detect_forms`: Whether to detect forms/key-value pairs (default: False)
        """
        super().__init__(provider_name, base_config)

        # Get AWS credentials from config or environment
        self._aws_access_key_id = self.base_config.get("aws_access_key_id", os.environ.get("AWS_ACCESS_KEY_ID"))
        self._aws_secret_access_key = self.base_config.get(
            "aws_secret_access_key", os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
        self._aws_region = self.base_config.get("aws_region", os.environ.get("AWS_REGION", "us-east-1"))

        # Configuration options
        self._output_tables_as_html = self.base_config.get("output_tables_as_html", True)
        self._detect_tables = self.base_config.get("detect_tables", True)
        self._detect_forms = self.base_config.get("detect_forms", False)

        # Validate credentials
        if not self._aws_access_key_id or not self._aws_secret_access_key:
            raise ProviderConfigError(
                "AWS credentials not configured. Set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY environment variables or provide them in config."
            )

        # Initialize boto3 client
        try:
            import boto3
        except ImportError as e:
            raise ProviderConfigError("boto3 package not installed. Run: pip install boto3") from e

        self._textract_client = boto3.client(
            "textract",
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
            region_name=self._aws_region,
        )

    # Textract synchronous API limits
    _MAX_DIMENSION = 10000  # Max 10,000 pixels in any dimension
    _MAX_BYTES = 10 * 1024 * 1024  # Max 10 MB
    _TARGET_BYTES = 9 * 1024 * 1024  # Target 9 MB to leave margin

    def _resize_image_for_textract(self, image: Any) -> bytes:
        """
        Resize and compress an image to fit within Textract's limits.

        Textract synchronous API limits:
        - Max dimension: 10,000 pixels
        - Max file size: 10 MB

        :param image: PIL Image object
        :return: PNG bytes that fit within Textract limits
        """
        import io

        from PIL import Image

        # Step 1: Resize if dimensions exceed limit
        width, height = image.size
        if width > self._MAX_DIMENSION or height > self._MAX_DIMENSION:
            scale = min(self._MAX_DIMENSION / width, self._MAX_DIMENSION / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Step 2: Try PNG first
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG", optimize=True)
        img_bytes = img_buffer.getvalue()

        # Step 3: If still too large, progressively reduce size
        scale = 0.9
        while len(img_bytes) > self._TARGET_BYTES and scale > 0.3:
            new_width = int(image.size[0] * scale)
            new_height = int(image.size[1] * scale)
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            img_buffer = io.BytesIO()
            resized.save(img_buffer, format="PNG", optimize=True)
            img_bytes = img_buffer.getvalue()

            if len(img_bytes) <= self._TARGET_BYTES:
                break
            scale *= 0.9

        return img_bytes

    def _analyze_document(self, file_path: str) -> dict[str, Any]:
        """
        Analyze a document using AWS Textract.

        :param file_path: Path to the PDF or image file
        :return: Raw Textract API response
        :raises ProviderError: For any API errors
        """
        try:
            from botocore.exceptions import ClientError
        except ImportError as e:
            raise ProviderConfigError("botocore package not installed. Run: pip install boto3") from e

        # Read the file and check if it needs resizing (for images)
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif"}:
            # For images, load and resize if needed
            from PIL import Image

            with Image.open(file_path) as img:
                document_bytes = self._resize_image_for_textract(img)
        else:
            # For other formats (shouldn't happen), read as-is
            with open(file_path, "rb") as f:
                document_bytes = f.read()

        # Determine which features to analyze
        feature_types = ["LAYOUT"]
        if self._detect_tables:
            feature_types.append("TABLES")
        if self._detect_forms:
            feature_types.append("FORMS")

        try:
            if feature_types:
                response = self._textract_client.analyze_document(
                    Document={"Bytes": document_bytes},
                    FeatureTypes=feature_types,
                )
            else:
                # Just detect text without tables/forms
                response = self._textract_client.detect_document_text(Document={"Bytes": document_bytes})
            return response  # type: ignore[no-any-return]

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            # Categorize errors
            if error_code in ("ThrottlingException", "ProvisionedThroughputExceededException"):
                raise ProviderTransientError(f"Rate limit exceeded: {error_message}") from e
            elif error_code in ("InvalidParameterException", "UnsupportedDocumentException"):
                raise ProviderPermanentError(f"Invalid document: {error_message}") from e
            elif error_code in ("AccessDeniedException", "InvalidS3ObjectException"):
                raise ProviderConfigError(f"AWS access error: {error_message}") from e
            else:
                raise ProviderTransientError(f"AWS Textract error: {error_message}") from e
        except Exception as e:
            raise ProviderTransientError(f"Unexpected error calling Textract: {e}") from e

    def _analyze_multipage_document(self, file_path: str) -> dict[str, Any]:
        """
        Analyze a multi-page document using AWS Textract async API.

        For PDFs, Textract requires using S3 + async operations for multi-page.
        This method handles single-page PDFs and images via synchronous API,
        and falls back to page-by-page processing for multi-page PDFs.

        :param file_path: Path to the document file
        :return: Combined Textract response
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        # For images, use direct synchronous API
        if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif"}:
            return self._analyze_document(file_path)

        # For PDFs, convert each page to image and process
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ProviderConfigError("pdf2image package not installed. Run: pip install pdf2image") from e

        try:
            images = convert_from_path(file_path, dpi=300)
        except Exception as e:
            raise ProviderPermanentError(f"Failed to convert PDF to images: {e}") from e

        all_blocks: list[dict[str, Any]] = []
        current_page = 0

        for page_num, image in enumerate(images):
            # Convert PIL image to bytes, resizing if needed for Textract limits
            img_bytes = self._resize_image_for_textract(image)

            # Analyze this page
            feature_types = ["LAYOUT"]
            if self._detect_tables:
                feature_types.append("TABLES")
            if self._detect_forms:
                feature_types.append("FORMS")

            try:
                from botocore.exceptions import ClientError

                if feature_types:
                    response = self._textract_client.analyze_document(
                        Document={"Bytes": img_bytes},
                        FeatureTypes=feature_types,
                    )
                else:
                    response = self._textract_client.detect_document_text(Document={"Bytes": img_bytes})

                # Add page number to blocks and accumulate
                for block in response.get("Blocks", []):
                    block["Page"] = page_num + 1
                    all_blocks.append(block)

                current_page = page_num + 1

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                error_message = e.response.get("Error", {}).get("Message", str(e))

                if error_code in ("ThrottlingException", "ProvisionedThroughputExceededException"):
                    raise ProviderTransientError(f"Rate limit exceeded: {error_message}") from e
                elif error_code in ("InvalidParameterException", "UnsupportedDocumentException"):
                    raise ProviderPermanentError(f"Invalid document: {error_message}") from e
                else:
                    raise ProviderTransientError(f"AWS Textract error: {error_message}") from e

        return {
            "Blocks": all_blocks,
            "DocumentMetadata": {"Pages": current_page},
        }

    def _convert_to_markdown(self, textract_response: dict[str, Any]) -> dict[str, Any]:
        """
        Convert Textract response to markdown format with HTML tables.

        Uses the amazon-textract-textractor library to properly parse
        and convert tables to HTML while preserving their visual structure.

        :param textract_response: Raw Textract API response
        :return: Dict with pages and markdown content
        """
        try:
            from textractor.parsers import response_parser
        except ImportError as e:
            raise ProviderConfigError(
                "amazon-textract-textractor package not installed. Run: pip install amazon-textract-textractor"
            ) from e

        # Parse the response using textractor
        document = response_parser.parse(textract_response)

        # Get number of pages
        num_pages = textract_response.get("DocumentMetadata", {}).get("Pages", 1)

        pages_content: dict[int, list[str]] = {i: [] for i in range(1, num_pages + 1)}

        # Process each page — interleave lines and tables by y-position
        for page in document.pages:
            page_num = page.page_num

            # Collect all elements with their y-positions for reading order
            elements: list[tuple[float, str]] = []

            for line in page.lines:
                # Skip lines that are part of tables
                if not self._is_in_table(line, page):
                    y_pos = line.bbox.y if hasattr(line, "bbox") and line.bbox else 0.0
                    elements.append((y_pos, line.text))

            if self._detect_tables and self._output_tables_as_html:
                for table in page.tables:
                    y_pos = table.bbox.y if hasattr(table, "bbox") and table.bbox else 0.0
                    # Use textractor's built-in to_html() which handles colspan/rowspan
                    html_table = table.to_html() if hasattr(table, "to_html") else ""
                    if html_table:
                        elements.append((y_pos, html_table))

            # Sort by y-position to reconstruct reading order
            elements.sort(key=lambda x: x[0])
            pages_content[page_num] = [elem[1] for elem in elements]

        # Build page-level markdown
        pages_data = []
        for page_num in range(1, num_pages + 1):
            content = pages_content.get(page_num, [])
            markdown = "\n\n".join(content)
            pages_data.append(
                {
                    "page_index": page_num - 1,
                    "markdown": markdown,
                }
            )

        # Build full document markdown
        full_markdown = "\n\n".join(page["markdown"] for page in pages_data if page["markdown"])  # type: ignore[misc]

        return {
            "pages": pages_data,
            "markdown": full_markdown,
            "num_pages": num_pages,
        }

    def _is_in_table(self, line: Any, page: Any) -> bool:
        """
        Check if a line is contained within any table on the page.

        :param line: A textractor Line object
        :param page: A textractor Page object
        :return: True if line is within a table
        """
        if not hasattr(page, "tables") or not page.tables:
            return False

        line_bbox = line.bbox if hasattr(line, "bbox") else None
        if not line_bbox:
            return False

        for table in page.tables:
            table_bbox = table.bbox if hasattr(table, "bbox") else None
            if table_bbox and self._bbox_contains(table_bbox, line_bbox):
                return True
        return False

    def _bbox_contains(self, outer: Any, inner: Any) -> bool:
        """
        Check if outer bounding box contains inner bounding box.

        :param outer: Outer bounding box
        :param inner: Inner bounding box
        :return: True if outer contains inner
        """
        try:
            return (  # type: ignore[no-any-return]
                outer.x <= inner.x
                and outer.y <= inner.y
                and (outer.x + outer.width) >= (inner.x + inner.width)
                and (outer.y + outer.height) >= (inner.y + inner.height)
            )
        except AttributeError:
            return False

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
                f"TextractProvider only supports PARSE product type, got {request.product_type}"
            )

        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")

        # Check file extension
        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
        if source_path.suffix.lower() not in supported_extensions:
            raise ProviderPermanentError(
                f"TextractProvider only supports {supported_extensions}, got {source_path.suffix}"
            )

        # Apply config overrides from pipeline
        config = pipeline.config or {}
        if "output_tables_as_html" in config:
            self._output_tables_as_html = config["output_tables_as_html"]
        if "detect_tables" in config:
            self._detect_tables = config["detect_tables"]
        if "detect_forms" in config:
            self._detect_forms = config["detect_forms"]

        started_at = datetime.now()

        try:
            # Analyze the document
            textract_response = self._analyze_multipage_document(str(source_path))

            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

            return RawInferenceResult(
                request=request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=request.product_type,
                raw_output={
                    "textract_response": textract_response,
                    "config": {
                        "output_tables_as_html": self._output_tables_as_html,
                        "detect_tables": self._detect_tables,
                        "detect_forms": self._detect_forms,
                    },
                },
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
                f"TextractProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        # Extract config from raw output
        config = raw_result.raw_output.get("config", {})
        self._output_tables_as_html = config.get("output_tables_as_html", True)
        self._detect_tables = config.get("detect_tables", True)

        # Convert Textract response to markdown
        textract_response = raw_result.raw_output.get("textract_response", {})
        markdown_result = self._convert_to_markdown(textract_response)

        # Build page-level output
        pages: list[PageIR] = []
        for page_data in markdown_result.get("pages", []):
            pages.append(
                PageIR(
                    page_index=page_data["page_index"],
                    markdown=page_data["markdown"],
                )
            )

        # Build layout_pages for layout cross-evaluation
        blocks = textract_response.get("Blocks", [])
        layout_pages = _build_layout_pages(blocks)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=markdown_result.get("markdown", ""),
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


def _build_layout_pages(blocks: list[dict[str, Any]]) -> list[ParseLayoutPageIR]:
    """Build layout_pages from Textract LAYOUT_* blocks for layout cross-evaluation.

    Groups LAYOUT_* blocks by page and converts each block's normalized [0,1]
    BoundingBox into a LayoutSegmentIR with canonical label mapping.
    Text content is extracted by traversing child LINE blocks.
    """
    from collections import defaultdict

    # Build block ID index for child traversal
    block_index: dict[str, dict[str, Any]] = {}
    for block in blocks:
        block_id = block.get("Id")
        if block_id:
            block_index[block_id] = block

    # Group LAYOUT_* blocks by page
    pages_blocks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        block_type = block.get("BlockType", "")
        if block_type in TEXTRACT_LABEL_MAP:
            page_num = block.get("Page", 1)
            pages_blocks[page_num].append(block)

    layout_pages: list[ParseLayoutPageIR] = []
    for page_num in sorted(pages_blocks.keys()):
        page_blocks = pages_blocks[page_num]
        items: list[LayoutItemIR] = []

        for block in page_blocks:
            block_type = block.get("BlockType", "")
            canonical_label = TEXTRACT_LABEL_MAP.get(block_type)
            if canonical_label is None:
                continue

            # Extract bbox (normalized [0,1] xywh)
            bbox = block.get("Geometry", {}).get("BoundingBox", {})
            left = float(bbox.get("Left", 0.0))
            top = float(bbox.get("Top", 0.0))
            width = float(bbox.get("Width", 0.0))
            height = float(bbox.get("Height", 0.0))

            confidence = float(block.get("Confidence", 100.0)) / 100.0

            seg = LayoutSegmentIR(
                x=left,
                y=top,
                w=width,
                h=height,
                confidence=confidence,
                label=canonical_label,
            )

            # Extract text from child LINE blocks
            content = _get_block_text(block, block_index)

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


def _get_block_text(block: dict[str, Any], block_index: dict[str, dict[str, Any]]) -> str:
    """Extract text from a LAYOUT block by traversing child LINE blocks."""
    relationships = block.get("Relationships", [])
    lines: list[str] = []
    for rel in relationships:
        if rel.get("Type") != "CHILD":
            continue
        for child_id in rel.get("Ids", []):
            child = block_index.get(child_id)
            if child and child.get("BlockType") == "LINE":
                text = child.get("Text", "")
                if text:
                    lines.append(text)
    return "\n".join(lines)
