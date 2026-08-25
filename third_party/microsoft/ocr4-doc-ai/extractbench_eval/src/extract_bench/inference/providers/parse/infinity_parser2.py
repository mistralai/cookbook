"""Provider for Infinity-Parser2 PARSE via infinity_parser2 SDK with vLLM server."""

import json
import logging
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from pdf2image import convert_from_path
from PIL import Image as PILImage

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "infly/Infinity-Parser2-Flash"

# Infinity-Parser2 category → Canonical17 label mapping
INFINITY_CATEGORY_MAP: dict[str, str] = {
    "header": "Page-header",
    "title": "Section-header",
    "text": "Text",
    "figure": "Picture",
    "table": "Table",
    "formula": "Formula",
    "figure_caption": "Caption",
    "table_caption": "Caption",
    "formula_caption": "Caption",
    "figure_footnote": "Footnote",
    "table_footnote": "Footnote",
    "page_footnote": "Footnote",
    "footer": "Page-footer",
}


@register_provider("infinity_parser2")
class InfinityParser2Provider(Provider):
    """
    Provider for Infinity-Parser2 via the infinity_parser2 SDK.

    Infinity-Parser2 is a document understanding model that converts PDFs
    and images to structured markdown/JSON. This provider uses the
    ``vllm-server`` backend which communicates with a running vLLM OpenAI-
    compatible server over HTTP. This avoids thread-safety issues in the
    ``vllm-engine`` backend when running concurrent requests.

    Configuration options:
        - model_name (str, default="infly/Infinity-Parser2-Flash"): Model name (must match server)
        - api_url (str, default="http://localhost:8000/v1/chat/completions"): vLLM server endpoint
        - api_key (str, default="EMPTY"): API key for the server
        - timeout (int, default=300): Request timeout in seconds
        - task_type (str, default="doc2json"): Parse task type
        - output_format (str, default="json"): Output format (json returns per-element layout with bboxes)
        - batch_size (int, default=4): Batch size for processing
        - max_new_tokens (int, default=None): Override max tokens for generation
        - temperature (float, default=0.0): Sampling temperature
        - deep_parsing_mode (bool, default=True): Parse figure content.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        self._model_name = self.base_config.get("model_name", DEFAULT_MODEL_NAME)
        self._api_url = self.base_config.get("api_url", "http://localhost:8000/v1/chat/completions")
        self._api_key = self.base_config.get("api_key", "EMPTY")
        self._timeout = self.base_config.get("timeout", 300)
        self._task_type = self.base_config.get("task_type", "doc2json")
        self._output_format = self.base_config.get("output_format", "json")
        self._batch_size = self.base_config.get("batch_size", 4)
        self._max_new_tokens = self.base_config.get("max_new_tokens")
        self._temperature = self.base_config.get("temperature", 0.0)
        self._deep_parsing_mode = self.base_config.get("deep_parsing_mode", True)

        try:
            from infinity_parser2 import InfinityParser2
        except ImportError as e:
            traceback.print_exc()
            raise ProviderConfigError("import infinity_parser2 failed") from e

        kwargs: dict[str, Any] = {
            "model_name": self._model_name,
            "backend": "vllm-server",
            "api_url": self._api_url,
            "api_key": self._api_key,
            "timeout": self._timeout,
        }

        self._parser = InfinityParser2(**kwargs)

    def _parse_document(self, file_path: str) -> dict[str, Any]:
        """
        Parse a document using InfinityParser2.

        :param file_path: Path to the PDF or image file
        :return: Raw parsing result
        """
        try:
            parse_kwargs: dict[str, Any] = {
                "task_type": self._task_type,
                "batch_size": self._batch_size,
            }

            if self._output_format:
                parse_kwargs["output_format"] = self._output_format

            if self._max_new_tokens is not None:
                parse_kwargs["max_new_tokens"] = self._max_new_tokens

            if "temperature" in self.base_config:
                parse_kwargs["temperature"] = self._temperature

            pil_image, page_width, page_height = load_image(file_path)
            result = self._parser.parse(pil_image, **parse_kwargs)

            if self._deep_parsing_mode:
                result = self._apply_deep_parsing(result, pil_image)

            return {
                "result": result,
                "_config": {
                    "model_name": self._model_name,
                    "backend": "vllm-server",
                    "api_url": self._api_url,
                    "task_type": self._task_type,
                    "output_format": self._output_format,
                    "batch_size": self._batch_size,
                    "page_width": page_width,
                    "page_height": page_height,
                },
            }

        except Exception as e:
            error_str = str(e).lower()
            transient_keywords = ["timeout", "network", "connection", "cuda", "out of memory", "oom"]
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Error during parsing (GPU/memory): {e}") from e
            raise ProviderPermanentError(f"Error parsing document: {e}") from e

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"InfinityParser2Provider only supports PARSE product type, got {request.product_type}"
            )

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"Source file not found: {file_path}")

        started_at = datetime.now()

        try:
            raw_output = self._parse_document(str(file_path))
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

    def _build_layout_segment(self, bbox: list, label: str) -> dict:
        """Build a LayoutSegmentIR from a bbox."""
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            x, y, w, h = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
        else:
            x, y, w, h = 0.0, 0.0, 0.0, 0.0

        return {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "confidence": 1.0,
            "label": label,
            "start_index": None,
            "end_index": None,
        }

    def _reassemble_text(self, label: str, text: str) -> str:
        """Reassemble text content based on label."""
        if not text:
            return ""

        if label == "Section-header":
            return f"# {text.lstrip('# ')}"
        elif label == "Formula":
            stripped = re.sub(r"^[\s$\(\)\[\]]+|[\s$\(\)\[\]]+$", "", text)
            return f"$${stripped}$$"
        elif label == "Picture":
            text = _convert_nonstandard_table(text)
            return text
        elif label == "Table":
            return _convert_table_header(text)
        else:
            return text

    def _build_layout_item(self, elem: dict, label: str) -> dict:
        """Build a single LayoutItemIR from an infinity-parser2 JSON element."""
        bbox = elem.get("bbox", [0, 0, 0, 0])
        text = elem.get("text", "")

        layout_seg = self._build_layout_segment(bbox, label)
        text = self._reassemble_text(label, text)

        return {
            "type": label,
            "md": text,
            "html": text if label == "Table" else "",
            "value": text,
            "bbox": layout_seg,
            "layout_segments": [layout_seg],
        }

    def _apply_deep_parsing(
        self,
        result: str,
        pil_image: PILImage.Image,
    ) -> str:
        """Apply deep parsing on figure elements, re-parsing cropped figure images as markdown tables.

        Extracts all ``figure`` elements from the parsed JSON, crops each figure region from
        ``pil_image``, re-parses the cropped images with a custom table-extraction prompt,
        and overwrites ``elem["text"]`` in place before serializing back to JSON.

        Returns the (possibly modified) JSON string.
        """
        try:
            elements: list[dict] = json.loads(result)
            if not isinstance(elements, list):
                return result

            figure_elements = [elem for elem in elements if elem.get("category", "").strip().lower() == "figure"]
            if not figure_elements:
                return result

            pil_figure_images = [
                pil_image.crop(
                    (
                        max(0, int(elem["bbox"][0])),
                        max(0, int(elem["bbox"][1])),
                        min(pil_image.width, int(elem["bbox"][2])),
                        min(pil_image.height, int(elem["bbox"][3])),
                    )
                )
                for elem in figure_elements
            ]

            deep_parse_kwargs = {
                "task_type": "custom",
                "custom_prompt": "please convert the image to a markdown table",
                "max_new_tokens": 2048,
            }
            deep_results = [self._parser.parse(img, **deep_parse_kwargs) for img in pil_figure_images]
            for elem, deep_result in zip(figure_elements, deep_results):
                elem["text"] = deep_result

            return json.dumps(elements)

        except Exception:
            logger.exception("Deep parsing pass failed; returning shallow parse result")
            return result

    def _normalize(self, raw_result: RawInferenceResult) -> ParseOutput:
        """Normalize JSON layout result into ParseOutput with pages, layout_pages, and markdown."""
        result_str = raw_result.raw_output.get("result", "")
        if not result_str:
            raise ProviderPermanentError(f"Empty result from InfinityParser2 for {raw_result.pipeline_name}")

        page_width = raw_result.raw_output["_config"]["page_width"]
        page_height = raw_result.raw_output["_config"]["page_height"]

        # Load elements
        try:
            elements: list[dict] = json.loads(result_str)
            if not isinstance(elements, list):
                elements = []
        except json.JSONDecodeError:
            elements = []

        # Group elements by page
        pages_dict: dict[int, list[dict]] = {}
        for elem in elements:
            page_num = elem.get("page", 1)
            if page_num not in pages_dict:
                pages_dict[page_num] = []
            pages_dict[page_num].append(elem)

        if not pages_dict:
            pages_dict = {1: []}

        if len(pages_dict) != 1:
            raise ProviderPermanentError(
                f"Infinity-Parser2 provider only supports single-page documents; "
                f"got {len(pages_dict)} pages for example {raw_result.request.example_id}"
            )

        # Get layout pages and markdown
        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []
        markdown_parts: list[str] = []

        for page_num in sorted(pages_dict.keys()):
            page_elements = pages_dict[page_num]

            header_items: list[dict] = []
            footer_items: list[dict] = []
            regular_items: list[dict] = []

            for elem in page_elements:
                raw_cat = elem.get("category", "text").strip().lower()
                norm_cat = INFINITY_CATEGORY_MAP.get(raw_cat, "Text")
                item = self._build_layout_item(elem, norm_cat)
                if norm_cat == "Page-header":
                    header_items.append(item)
                elif norm_cat == "Page-footer":
                    footer_items.append(item)
                else:
                    regular_items.append(item)

            page_items = header_items + regular_items + footer_items
            page_md_parts = [item.get("md", "") for item in page_items if item.get("md")]
            page_md = "\n\n".join(page_md_parts)

            header_md = " ".join(c.get("value", "") for c in header_items)
            footer_md = " ".join(c.get("value", "") for c in footer_items)

            layout_pages.append(
                ParseLayoutPageIR(
                    page_number=page_num,
                    width=page_width,
                    height=page_height,
                    md=page_md,
                    text=page_md,
                    page_header_markdown=header_md,
                    page_footer_markdown=footer_md,
                    printed_page_number="",
                    original_orientation_angle=0,
                    items=page_items,
                )
            )

            pages.append(PageIR(page_index=page_num - 1, markdown=page_md))
            if page_md:
                markdown_parts.append(page_md)

        full_markdown = "\n\n".join(markdown_parts)

        return ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=full_markdown,
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"InfinityParser2Provider only supports PARSE product type, got {raw_result.product_type}"
            )

        output = self._normalize(raw_result)

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


def load_image(file_path: str) -> tuple[PILImage.Image, float, float]:
    """Load a PDF or image file as a PIL Image and return its dimensions.

    - PDF: converts the first page to RGB image at 300 DPI.
    - Image: opens and converts to RGB.

    Returns:
        Tuple of (PIL Image, width, height) where width and height are in pixels.
    """
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        images = convert_from_path(str(path), dpi=300, first_page=1, last_page=1)
        if not images:
            raise ProviderPermanentError(f"Failed to render PDF page: {file_path}")
        pil_image = images[0].convert("RGB")
    else:
        pil_image = PILImage.open(str(path)).convert("RGB")

    width, height = pil_image.size
    return pil_image, float(width), float(height)


# =============================================================================
# Postprocess for chart2table
# =============================================================================


def _is_valid_md_table(table_text: str) -> bool:
    """Check if a markdown table is valid (non-empty)."""
    if not table_text or not table_text.strip():
        return False

    if not all(ch in table_text for ch in ["|", "-", "\n"]):
        return False

    stripped = table_text[table_text.find("|") : table_text.rfind("|") + 1]
    stripped = re.sub(r"^\s*\|[\s\-:|]+\|\s*$", "", stripped, flags=re.MULTILINE)
    if not stripped.replace(" ", "").replace("\n", "").replace("|", ""):
        return False

    return True


def _is_nonstandard_table(text: str) -> bool:
    """Check if text is a non-standard markdown table (no leading '|', contains '&' separators)."""
    if not text:
        return False
    stripped = text.strip()
    if stripped.startswith("|"):
        return False
    return "&" in text


def _find_column_number(text: str) -> int:
    """Find the column number from a nonstandard table.

    Split the text by '&' and count '|' in each segment. The header row always
    has the most pipes (full cells). Column count = max(pipe_counts) + 1.
    """
    if "&" not in text:
        return 0
    raw_segments = text.split("&")
    segments = [s.strip() for s in raw_segments if s.strip()]
    if not segments:
        return 0
    pipe_counts = [s.count("|") for s in segments]
    return max(pipe_counts) + 1


def _find_all_separator_indices(text: str, col_num: int) -> list[int]:
    """Identify which '&' characters are row-group separators based on pipe counts."""
    if col_num == 0:
        return []
    expected_pipes = col_num - 1
    sep_positions = []
    prev_sep = -1
    i = 0
    while i < len(text):
        amp = text.find("&", i)
        if amp == -1:
            break

        # Count pipes between prev_sep+1 and amp-1
        pipe_count = 0
        for j in range(prev_sep + 1, amp):
            if text[j] == "|":
                pipe_count += 1

        if pipe_count == expected_pipes:
            sep_positions.append(amp)
            prev_sep = amp

        i = amp + 1
    return sep_positions


def _convert_nonstandard_table(text: str) -> str:
    """Convert a non-standard markdown table (with '&' row-group separators) to proper markdown table format."""
    if not _is_nonstandard_table(text):
        return text

    col_num = _find_column_number(text)
    if col_num == 0:
        return text

    sep_indices = _find_all_separator_indices(text, col_num)
    if not sep_indices:
        return text

    segments = []
    prev = 0
    for idx in sep_indices:
        segments.append(text[prev:idx].strip())
        prev = idx + 1
    segments.append(text[prev:].strip())

    header = segments[0]
    if not header.startswith("|"):
        header = "| " + header
    if not header.rstrip().endswith("|"):
        header = header.rstrip() + " |"

    separator = "| " + " | ".join(["---"] * col_num) + " |"
    normalized_lines = [header, separator]

    for seg in segments[1:]:
        if not seg:
            continue
        cells = [c.strip() for c in seg.split("|") if c.strip()]
        padded = cells + [""] * max(0, col_num - len(cells))
        row = "| " + " | ".join(padded[:col_num]).rstrip() + " |"
        normalized_lines.append(row)

    return "\n".join(normalized_lines)


# =============================================================================
# Postprocess for HTML table header
# =============================================================================


def _is_year_cell(text: str) -> bool:
    """Return True if text looks like a date/year (yyyy, yyyymm, yyyymmdd, etc.)."""
    text = text.strip()
    return bool(re.fullmatch(r"(19|20)\d{2,4}([-/]?\d{2}([-/]?\d{2})?)?", text))


def _is_gender_cell(text: str) -> bool:
    """Return True if text looks like gender."""
    text = text.strip().lower()
    return text in ("male", "female", "non-binary", "other", "undisclosed")


def _is_pure_text_cell(text: str) -> bool:
    """Return True if text contains no digits at all."""
    text = text.strip()
    return bool(text) and any(c.isalpha() for c in text)


def _is_pure_number_cell(text: str) -> bool:
    """Return True if text looks like a pure numeric value.

    Accepts numbers with commas, decimals, dollar sign, percent sign,
    plus/minus sign, and parentheses (for negative numbers).
    """
    text = text.strip()
    if not text:
        return False
    # Allow: digits, comma, dot, minus, plus, $, %, parentheses
    allowed = set("0123456789,.-+()$% ")
    return all(c in allowed for c in text)


def _determine_header_row_count(rows: list) -> int:
    """Determine how many top rows are header rows (year/gender/value rules + rowspan fallback)."""
    if not rows:
        return 0

    def non_empty_cells(row):
        return [td.get_text(strip=True) for td in row.find_all("td", recursive=False) if td.get_text(strip=True)]

    def stats(row_list):
        """Return (pure_text_count, pure_number_count, total) for a list of rows."""
        text_count = number_count = total = 0
        for row in row_list:
            for cell in non_empty_cells(row):
                total += 1
                if _is_pure_text_cell(cell):
                    text_count += 1
                elif _is_pure_number_cell(cell):
                    number_count += 1
        return text_count, number_count, total

    # Rule 1: Year
    for i, row in enumerate(rows):
        if i > 3:
            break
        cells = non_empty_cells(row)
        if not cells:
            continue
        year_count = sum(1 for c in cells if _is_year_cell(c))
        if year_count / len(cells) >= 0.5:
            return i + 1

    # Rule 2: Gender
    for i, row in enumerate(rows):
        if i > 3:
            break
        cells = non_empty_cells(row)
        if not cells:
            continue
        gender_count = sum(1 for c in cells if _is_gender_cell(c))
        if gender_count / len(cells) >= 0.5:
            return i + 1

    # Rule 3: Value (pure-text header region followed by pure-number data region)
    best_i = -1
    best_score = -1.0
    for i in range(3):
        header_rows = rows[: i + 1]
        data_rows = rows[i + 1 :]
        if not header_rows or not data_rows:
            continue
        header_text, header_num, header_total = stats(header_rows)
        data_text, data_num, data_total = stats(data_rows)
        if header_total == 0 or data_total == 0:
            continue
        if header_text / header_total >= 0.5 and data_num / data_total >= 0.5:
            score = header_text / header_total + data_num / data_total
            if score > best_score:
                best_score = score
                best_i = i
    if best_i >= 0:
        return best_i + 1

    # Rule 4: Fallback — max rowspan in row 0
    first_row = rows[0]
    max_rowspan = 1
    for td in first_row.find_all("td", recursive=False):
        rowspan = int(td.get("rowspan", 1))
        if rowspan > max_rowspan:
            max_rowspan = rowspan
    return max_rowspan


def _convert_table_header(html: str) -> str:
    """Convert <td> tags in HTML table header rows to <th> for TEDS/GriTS evaluation."""
    if not html or "<table" not in html.lower():
        return html

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    for table in tables:
        rows = table.find_all("tr", recursive=False)
        if not rows:
            continue

        header_row_count = _determine_header_row_count(rows)

        for i, row in enumerate(rows):
            if i >= header_row_count:
                break
            tds = row.find_all("td", recursive=False)
            for td in tds:
                new_th = soup.new_tag("th")
                for key, value in td.attrs.items():
                    new_th[key] = value
                new_th.string = td.get_text()
                td.replace_with(new_th)

    return str(soup)
