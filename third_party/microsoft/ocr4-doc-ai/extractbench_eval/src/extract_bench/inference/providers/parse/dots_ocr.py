"""Provider for dots.ocr parse via a self-hosted OpenAI-compatible API.

Supports two prompt modes:
- ``prompt_parse_markdown``: Returns clean markdown (parse-only, no layout data).
- ``prompt_layout_all_en_v1_5``: Returns structured JSON with bboxes, categories,
  and text.  Markdown is reassembled from the layout elements and ``layout_pages``
  is populated so the same pipeline can be cross-evaluated for layout detection.
"""

import base64
import io
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

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

# Default model name served by the self-hosted vLLM deployment
SERVED_MODEL_NAME = "dots-ocr-1.5"

# ---------------------------------------------------------------------------
# Prompt definitions
# ---------------------------------------------------------------------------

# Markdown-oriented prompt (no layout data)
PROMPT_PARSE_MARKDOWN = (
    "Parse this document image and output its content as clean markdown.\n"
    "- Preserve document structure (headings, paragraphs, lists, tables)\n"
    "- Convert tables to HTML format (<table>, <tr>, <th>, <td>) "
    "with colspan/rowspan for merged cells\n"
    "- Format formulas as LaTeX\n"
    "- Describe images/figures briefly in square brackets "
    "like [Figure: description]\n"
    "- Maintain reading order\n"
    "- Output the original text with no translation\n"
    "- Do not add commentary - only output the parsed content\n"
)

# dots.ocr 1.5 layout+text prompt (Core11 categories, structured JSON output)
PROMPT_LAYOUT_ALL_EN_V1_5 = (
    "Please output the layout information from the PDF image, "
    "including each layout element's bbox, its category, and the "
    "corresponding text content within the bbox.\n"
    "\n"
    "1. Bbox format: [x1, y1, x2, y2]\n"
    "\n"
    "2. Layout Categories: The possible categories are "
    "['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', "
    "'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n"
    "\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: If the picture is a chart or graph, extract all data points "
    "and format as an HTML table with flat combined column headers "
    "(e.g., 'Revenue 2023' not nested header rows). Include axis labels "
    "as column/row headers. For non-chart pictures, the text field should "
    "be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n"
    "\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, "
    "with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n"
    "\n"
    "5. Final Output: The entire output must be a single JSON object.\n"
)

PROMPT_CONFIGS: dict[str, str] = {
    "prompt_parse_markdown": PROMPT_PARSE_MARKDOWN,
    "prompt_layout_all_en_v1_5": PROMPT_LAYOUT_ALL_EN_V1_5,
}

# Prompt modes that return structured JSON with layout elements
_LAYOUT_PROMPT_MODES = {"prompt_layout_all_en_v1_5"}


# ---------------------------------------------------------------------------
# dots.ocr layout response schema
# ---------------------------------------------------------------------------


class DotsOcrLayoutItem(BaseModel):
    """Single layout element returned by the dots.ocr layout prompt."""

    bbox: list[float] = Field(..., min_length=4, max_length=4)
    category: str
    text: str = ""


@register_provider("dots_ocr_parse")
class DotsOcrParseProvider(Provider):
    """
    Unified parse provider for dots.ocr deployed on your own infrastructure.

    When ``prompt_mode`` is a layout prompt (e.g. ``prompt_layout_all_en_v1_5``),
    the model returns structured JSON with bounding boxes, categories, and text.
    The provider reassembles markdown from the layout elements and populates
    ``ParseOutput.layout_pages`` so the same pipeline can be cross-evaluated
    against layout detection datasets (following the LlamaParse pattern).

    When ``prompt_mode`` is ``prompt_parse_markdown`` (default), the model
    returns clean markdown and no layout data is produced.

    Configuration options:
        - endpoint_url (str, required): server URL (or DOTS_OCR_ENDPOINT_URL env var)
        - model (str, default: "dots-ocr-1.5"): Served model name
        - prompt_mode (str, default: "prompt_parse_markdown"): Prompt selection
        - timeout (int, default: 180): Request timeout in seconds
        - max_tokens (int, default: 32768): Max tokens per response
        - dpi (int, default: 150): DPI for PDF to image conversion
        - temperature (float, default: 0.1): Sampling temperature
        - top_p (float, default: 0.9): Top-p sampling
        - prompt_override (str, optional): Custom prompt text (overrides prompt_mode)
    """

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(provider_name, base_config)

        endpoint_url = self.base_config.get("endpoint_url") or os.getenv("DOTS_OCR_ENDPOINT_URL")
        if not endpoint_url:
            raise ProviderConfigError(
                "endpoint_url is required for dots_ocr_parse provider. "
                "Set DOTS_OCR_ENDPOINT_URL or pass endpoint_url in config."
            )

        self._client = OpenAI(
            base_url=endpoint_url,
            api_key=os.getenv("DOTS_OCR_API_KEY", "not-needed"),
        )

        self._model = self.base_config.get("model", SERVED_MODEL_NAME)
        self._timeout = self.base_config.get("timeout", 180)
        self._max_tokens = self.base_config.get("max_tokens", 16384)
        self._dpi = self.base_config.get("dpi", 150)
        self._temperature = self.base_config.get("temperature", 0.1)
        self._top_p = self.base_config.get("top_p", 0.9)

        self._prompt_mode = self.base_config.get("prompt_mode", "prompt_parse_markdown")
        prompt_override = self.base_config.get("prompt_override")
        if prompt_override:
            self._prompt = prompt_override
        else:
            prompt = PROMPT_CONFIGS.get(self._prompt_mode)
            if not prompt:
                raise ProviderConfigError(
                    f"Unknown prompt_mode '{self._prompt_mode}'. Available: {sorted(PROMPT_CONFIGS.keys())}"
                )
            self._prompt = prompt

        self._is_layout_mode = self._prompt_mode in _LAYOUT_PROMPT_MODES

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _image_to_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _pdf_to_images(self, pdf_path: str) -> list[Image.Image]:
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ProviderConfigError("pdf2image package not installed. Run: pip install pdf2image") from e
        try:
            return convert_from_path(pdf_path, dpi=self._dpi)
        except Exception as e:
            raise ProviderPermanentError(f"Failed to convert PDF to images: {e}") from e

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _call_endpoint(self, image: Image.Image) -> str:
        """Call dots.ocr via OpenAI-compatible API and return raw response text."""
        img_base64 = self._image_to_base64(image)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                            },
                        ],
                    },
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "connection" in error_msg:
                raise ProviderTransientError(f"API call failed: {e}") from e
            raise ProviderPermanentError(f"API call failed: {e}") from e

        content = response.choices[0].message.content
        if not content:
            raise ProviderPermanentError("Empty response from model")
        return content

    # ------------------------------------------------------------------
    # HTML sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_html_attributes(text: str) -> str:
        """Quote unquoted HTML attributes so tables are valid XML."""

        def _quote_attrs(match: re.Match) -> str:
            tag_text = match.group(0)
            tag_text = re.sub(
                r'(\w+)=([^\s"\'<>=]+)',
                r'\1="\2"',
                tag_text,
            )
            return tag_text

        return re.sub(r"<[^>]+>", _quote_attrs, text)

    # ------------------------------------------------------------------
    # Layout JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_layout_items(content: str) -> list[DotsOcrLayoutItem]:
        """Parse dots.ocr layout response into typed items.

        The model is fine-tuned to return a JSON array of
        ``{bbox, category, text}`` objects.  We try ``json.loads``
        first, then fall back to extracting a JSON array from
        markdown fences or raw brackets.
        """
        candidates: list[str] = [content]

        # Fallback: extract from ```json ... ``` fences
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if fence:
            candidates.append(fence.group(1))

        # Fallback: extract outermost [...] bracket
        bracket = re.search(r"\[[\s\S]*\]", content)
        if bracket:
            candidates.append(bracket.group(0))

        from pydantic import TypeAdapter

        adapter = TypeAdapter(list[DotsOcrLayoutItem])
        for candidate in candidates:
            try:
                return adapter.validate_json(candidate)
            except Exception:
                continue

        raise ProviderPermanentError(f"Could not parse layout items from response: {content[:500]}")

    # ------------------------------------------------------------------
    # Per-page inference
    # ------------------------------------------------------------------

    def _run_inference_pages(self, source_path: Path) -> dict[str, Any]:
        """Convert source file to images and run inference on each page."""
        if source_path.suffix.lower() == ".pdf":
            images = self._pdf_to_images(str(source_path))
        else:
            images = [Image.open(source_path)]

        pages = []
        for page_index, image in enumerate(images):
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")

            raw_text = self._call_endpoint(image)

            page_data: dict[str, Any] = {
                "page_index": page_index,
                "width": image.width,
                "height": image.height,
                "raw_response": raw_text,
            }

            if self._is_layout_mode:
                # Parse structured JSON → typed layout items + reassemble markdown
                try:
                    layout_items = self._parse_layout_items(raw_text)
                except ProviderPermanentError:
                    layout_items = []

                page_data["layout_items"] = [item.model_dump() for item in layout_items]
                page_data["markdown"] = _reassemble_markdown(layout_items)
            else:
                page_data["markdown"] = raw_text
                page_data["layout_items"] = []

            pages.append(page_data)

        return {
            "pages": pages,
            "num_pages": len(images),
            "model": self._model,
            "prompt_mode": self._prompt_mode,
            "config": {
                "dpi": self._dpi,
                "max_tokens": self._max_tokens,
                "timeout": self._timeout,
            },
        }

    # ------------------------------------------------------------------
    # run_inference
    # ------------------------------------------------------------------

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DotsOcrParseProvider only supports PARSE product type, got {request.product_type}"
            )

        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")

        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
        if source_path.suffix.lower() not in supported_extensions:
            raise ProviderPermanentError(
                f"DotsOcrParseProvider supports {supported_extensions}, got {source_path.suffix}"
            )

        started_at = datetime.now()
        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                raw_output = self._run_inference_pages(source_path)

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

            except ProviderTransientError as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = 15 * (2**attempt)
                    print(
                        f"[dots.ocr] Transient error on {request.example_id}: {e}. "
                        f"Retrying in {delay}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                    continue

            except (ProviderPermanentError, ProviderConfigError) as e:
                last_error = e
                break

            except Exception as e:
                last_error = e
                break

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        error_msg = str(last_error)
        if isinstance(last_error, TimeoutError):
            error_msg = f"Request timed out after {self._timeout} seconds"

        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={
                "pages": [],
                "_error": error_msg,
                "_error_type": type(last_error).__name__ if last_error else "Unknown",
                "model": self._model,
                "config": {
                    "dpi": self._dpi,
                    "max_tokens": self._max_tokens,
                    "timeout": self._timeout,
                },
            },
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # normalize
    # ------------------------------------------------------------------

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DotsOcrParseProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []
        page_markdowns: list[str] = []

        for page_data in raw_result.raw_output.get("pages", []):
            page_index = page_data.get("page_index", 0)
            markdown = page_data.get("markdown", "")
            img_width = page_data.get("width", 0)
            img_height = page_data.get("height", 0)

            if markdown:
                markdown = self._sanitize_html_attributes(markdown)

            pages.append(PageIR(page_index=page_index, markdown=markdown))
            page_markdowns.append(markdown)

            # Build layout_pages from structured layout items (if available)
            layout_items = page_data.get("layout_items", [])
            if layout_items and img_width > 0 and img_height > 0:
                layout_page = _build_layout_page(
                    layout_items=layout_items,
                    page_number=page_index + 1,
                    img_width=img_width,
                    img_height=img_height,
                    page_markdown=markdown,
                )
                layout_pages.append(layout_page)

        pages.sort(key=lambda p: p.page_index)
        full_markdown = "\n\n".join(page_markdowns)

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


# ======================================================================
# Module-level helpers
# ======================================================================


def _reassemble_markdown(layout_items: list[DotsOcrLayoutItem]) -> str:
    """Reassemble page markdown from layout element text fields."""
    parts: list[str] = []
    for item in layout_items:
        label = item.category.strip().lower()
        if not item.text:
            continue

        if label in ("title", "section-header"):
            parts.append(f"## {item.text}")
        elif label == "table":
            parts.append(item.text)  # Already HTML
        elif label == "formula":
            parts.append(f"$${item.text}$$")
        else:
            parts.append(item.text)

    return "\n\n".join(parts)


def _build_layout_page(
    *,
    layout_items: list[dict[str, Any]],
    page_number: int,
    img_width: int,
    img_height: int,
    page_markdown: str,
) -> ParseLayoutPageIR:
    """Convert dots.ocr layout items into a ParseLayoutPageIR for cross-eval."""
    from pydantic import TypeAdapter

    adapter = TypeAdapter(list[DotsOcrLayoutItem])
    typed_items = adapter.validate_python(layout_items)

    items: list[LayoutItemIR] = []
    for li in typed_items:
        x1, y1, x2, y2 = li.bbox

        seg = LayoutSegmentIR(
            x=x1 / img_width,
            y=y1 / img_height,
            w=(x2 - x1) / img_width,
            h=(y2 - y1) / img_height,
            confidence=1.0,
            label=li.category,
        )

        norm_label = li.category.strip().lower()
        if norm_label == "table":
            item_type = "table"
        elif norm_label == "picture":
            item_type = "image"
        else:
            item_type = "text"

        items.append(
            LayoutItemIR(
                type=item_type,
                value=li.text,
                bbox=seg,
                layout_segments=[seg],
            )
        )

    return ParseLayoutPageIR(
        page_number=page_number,
        width=float(img_width),
        height=float(img_height),
        md=page_markdown,
        items=items,
    )
