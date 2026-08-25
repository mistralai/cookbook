"""Provider for Qwen3.5 self-hosted vLLM servers.

Qwen3.5 is a unified multimodal model family with built-in vision via early
fusion. This provider supports two prompt modes:

- "parse" (default): Pure markdown output, with md-table-to-HTML conversion
  for GriTS/TEDS evaluation. No layout data.
- "layout": Structured JSON with bboxes + categories + text per region.
  Produces both reassembled markdown and layout_pages for layout detection
  cross-evaluation.
"""

import asyncio
import base64
import io
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from pydantic import BaseModel

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

DEFAULT_SERVED_MODEL_NAME = "qwen3.5-4b"

# --- Parse mode prompt ---
PROMPT_PARSE = (
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

# --- Layout mode prompt ---
PROMPT_LAYOUT = (
    "Please output the layout information from the PDF image, "
    "including each layout element's bbox, its category, and the "
    "corresponding text content within the bbox.\n"
    "\n"
    "1. Bbox format: [x1, y1, x2, y2] using normalized 0-1000 coordinates.\n"
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
    "5. Final Output: Return ONLY a JSON array. Each element MUST be:\n"
    '{"bbox": [x1, y1, x2, y2], "category": "<category>", "text": "<content>"}\n'
    "No markdown fences, no prose outside the JSON.\n"
)


class LayoutItem(BaseModel):
    """Single layout element from the model response.

    Different Qwen model versions use different field names:
    - 4B: bbox_2d, category, text
    - 3.5-35B: bbox_2d, label (no text)
    - 3.6-35B: bbox, category, text
    We normalize all variants here.
    """

    model_config = {"extra": "ignore"}

    bbox: list[float] | None = None
    bbox_2d: list[float] | None = None
    category: str = "Text"
    label: str | None = None
    text: str = ""

    def model_post_init(self, __context: Any) -> None:
        # Some models use "label" instead of "category"
        if self.label is not None and self.category == "Text":
            self.category = self.label

    @property
    def coords(self) -> list[float]:
        return self.bbox_2d or self.bbox or [0, 0, 0, 0]


@register_provider("qwen3_5")
class Qwen35Provider(Provider):
    """
    Provider for Qwen3.5 self-hosted vLLM servers.

    Configuration options:
        - server_url (str, required): server URL
        - model (str, default="qwen3.5-4b"): Served model name
        - prompt_mode (str, default="parse"): "parse" or "layout"
        - timeout (int, default=600): Request timeout in seconds
        - dpi (int, default=150): DPI for PDF to image conversion
        - max_tokens (int, default=16384): Max tokens per response
        - temperature (float, default=0.1): Sampling temperature
        - api_key_env (str, default="VLLM_API_KEY"): Env var for API key
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("QWEN35_SERVER_URL")
        if not server_url:
            raise ProviderConfigError("Qwen3.5 provider requires 'server_url' in config.")
        self._server_url: str = str(server_url)

        self._model = self.base_config.get("model", DEFAULT_SERVED_MODEL_NAME)
        self._prompt_mode = self.base_config.get("prompt_mode", "parse")
        self._timeout = self.base_config.get("timeout", 600)
        self._dpi = self.base_config.get("dpi", 150)
        self._max_tokens = self.base_config.get("max_tokens", 16384)
        self._temperature = self.base_config.get("temperature", 0.1)

        api_key_env = self.base_config.get("api_key_env", "VLLM_API_KEY")
        self._api_key = os.environ.get(api_key_env, "")

        self._prompt = PROMPT_LAYOUT if self._prompt_mode == "layout" else PROMPT_PARSE

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _pdf_to_image_with_size(self, pdf_path: Path) -> tuple[bytes, int, int]:
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path, dpi=self._dpi)
            if not images:
                raise ProviderPermanentError(f"No pages found in PDF: {pdf_path}")
            img = images[0]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue(), img.width, img.height
        except ImportError as e:
            raise ProviderPermanentError("pdf2image is required.") from e
        except ProviderPermanentError:
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Error converting PDF to image: {e}") from e

    def _read_image_with_size(self, file_path: Path) -> tuple[bytes, int, int]:
        from PIL import Image

        try:
            img = Image.open(file_path)
            w, h = img.size
            return file_path.read_bytes(), w, h
        except Exception as e:
            raise ProviderPermanentError(f"Error reading image file: {e}") from e

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    async def _call_api(self, session: aiohttp.ClientSession, image_b64: str) -> str:
        api_url = f"{self._server_url.rstrip('/')}/v1/chat/completions"

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": self._prompt},
                    ],
                }
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with session.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                if resp.status in (408, 502, 503, 504):
                    raise ProviderTransientError(f"HTTP {resp.status}: {error_text[:200]}")
                raise ProviderPermanentError(f"HTTP {resp.status}: {error_text[:200]}")

            result = await resp.json()
            try:
                content = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                raise ProviderPermanentError(f"Invalid response format: {e}") from e

            if not content:
                raise ProviderPermanentError("Empty content response from API")
            return str(content)

    # ------------------------------------------------------------------
    # JSON parsing (layout mode only)
    # ------------------------------------------------------------------

    @staticmethod
    def _repair_json(text: str) -> str:
        """Fix common LLM JSON errors: missing commas between objects."""
        return re.sub(r"\}\s*\n\s*\{", "},\n{", text)

    @staticmethod
    def _parse_layout_items(content: str) -> list[LayoutItem]:
        """Parse model response into typed layout items."""
        import json as json_mod

        from pydantic import TypeAdapter

        candidates: list[str] = [content]

        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if fence:
            candidates.append(fence.group(1))

        bracket = re.search(r"\[[\s\S]*\]", content)
        if bracket:
            candidates.append(bracket.group(0))

        adapter = TypeAdapter(list[LayoutItem])

        for candidate in candidates:
            try:
                return adapter.validate_json(candidate)
            except Exception:
                pass

        for candidate in candidates:
            repaired = Qwen35Provider._repair_json(candidate)
            try:
                return adapter.validate_json(repaired)
            except Exception:
                try:
                    parsed = json_mod.loads(repaired)
                    return adapter.validate_python(parsed)
                except Exception:
                    continue

        raise ProviderPermanentError(f"Could not parse layout items from response: {content[:500]}")

    # ------------------------------------------------------------------
    # run_inference
    # ------------------------------------------------------------------

    async def _run_inference_async(self, image_bytes: bytes, img_width: int, img_height: int) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_bytes).decode()

        async with aiohttp.ClientSession() as session:
            raw_content = await self._call_api(session, image_b64)

        if self._prompt_mode == "layout":
            try:
                items = self._parse_layout_items(raw_content)
                layout_items = [{"bbox": item.coords, "category": item.category, "text": item.text} for item in items]
            except ProviderPermanentError:
                layout_items = []

            return {
                "pages": [
                    {
                        "page_index": 0,
                        "width": img_width,
                        "height": img_height,
                        "raw_response": raw_content,
                        "layout_items": layout_items,
                    }
                ],
                "prompt_mode": "layout",
                "_config": {
                    "server_url": self._server_url,
                    "model": self._model,
                    "dpi": self._dpi,
                },
            }
        else:
            return {
                "markdown": raw_content,
                "prompt_mode": "parse",
                "_config": {
                    "server_url": self._server_url,
                    "model": self._model,
                    "dpi": self._dpi,
                },
            }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"Qwen35Provider only supports PARSE, got {request.product_type}")

        started_at = datetime.now()

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"Source file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            image_bytes, img_w, img_h = self._pdf_to_image_with_size(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
            image_bytes, img_w, img_h = self._read_image_with_size(file_path)
        else:
            raise ProviderPermanentError(
                f"Unsupported file type: {suffix}. Supported: .pdf, .png, .jpg, .jpeg, .webp, .tiff, .bmp"
            )

        try:
            raw_output = asyncio.run(self._run_inference_async(image_bytes, img_w, img_h))
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

        except (ProviderPermanentError, ProviderTransientError):
            raise

        except Exception as e:
            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

            error_msg = str(e)
            if isinstance(e, asyncio.TimeoutError):
                error_msg = f"Request timed out after {self._timeout} seconds"

            return RawInferenceResult(
                request=request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=request.product_type,
                raw_output={
                    "markdown": "" if self._prompt_mode == "parse" else None,
                    "pages": [] if self._prompt_mode == "layout" else None,
                    "_error": error_msg,
                    "_error_type": type(e).__name__,
                    "_config": {
                        "server_url": self._server_url,
                        "model": self._model,
                        "dpi": self._dpi,
                    },
                },
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=latency_ms,
            )

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_html_attributes(text: str) -> str:
        def _quote_attrs(match: re.Match) -> str:
            tag_text = match.group(0)
            return re.sub(r'(\w+)=([^\s"\'<>=]+)', r'\1="\2"', tag_text)

        return re.sub(r"<[^>]+>", _quote_attrs, text)

    @staticmethod
    def _convert_md_tables_to_html(content: str) -> str:
        """Convert markdown pipe tables to HTML <table> elements."""
        import markdown2

        lines = content.split("\n")
        result_parts: list[str] = []
        table_lines: list[str] = []
        in_table = False

        for line in lines:
            is_table_line = "|" in line and line.strip().startswith("|")
            if is_table_line:
                if not in_table:
                    in_table = True
                    table_lines = [line]
                else:
                    table_lines.append(line)
            else:
                if in_table:
                    if len(table_lines) >= 2:
                        table_md = "\n".join(table_lines)
                        html = markdown2.markdown(table_md, extras=["tables"]).strip()
                        if "<table>" in html.lower():
                            result_parts.append(html)
                        else:
                            result_parts.extend(table_lines)
                    else:
                        result_parts.extend(table_lines)
                    table_lines = []
                    in_table = False
                result_parts.append(line)

        if in_table and len(table_lines) >= 2:
            table_md = "\n".join(table_lines)
            html = markdown2.markdown(table_md, extras=["tables"]).strip()
            if "<table>" in html.lower():
                result_parts.append(html)
            else:
                result_parts.extend(table_lines)
        elif in_table:
            result_parts.extend(table_lines)

        return "\n".join(result_parts)

    # ------------------------------------------------------------------
    # normalize
    # ------------------------------------------------------------------

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"Qwen35Provider only supports PARSE, got {raw_result.product_type}")

        prompt_mode = raw_result.raw_output.get("prompt_mode", "parse")

        if prompt_mode == "layout":
            # Layout mode: reassemble markdown + build layout_pages
            pages: list[PageIR] = []
            layout_pages: list[ParseLayoutPageIR] = []
            page_markdowns: list[str] = []

            for page_data in raw_result.raw_output.get("pages", []):
                page_index = page_data.get("page_index", 0)
                img_width = page_data.get("width", 0)
                img_height = page_data.get("height", 0)
                layout_items_raw = page_data.get("layout_items", [])

                markdown = _reassemble_markdown(layout_items_raw)
                if markdown:
                    markdown = self._sanitize_html_attributes(markdown)

                pages.append(PageIR(page_index=page_index, markdown=markdown))
                page_markdowns.append(markdown)

                if layout_items_raw and img_width > 0 and img_height > 0:
                    layout_page = _build_layout_page(
                        layout_items=layout_items_raw,
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
        else:
            # Parse mode: direct markdown with md-table-to-HTML conversion
            markdown = raw_result.raw_output.get("markdown", "")
            if markdown:
                markdown = self._convert_md_tables_to_html(markdown)
                markdown = self._sanitize_html_attributes(markdown)

            output = ParseOutput(
                task_type="parse",
                example_id=raw_result.request.example_id,
                pipeline_name=raw_result.pipeline_name,
                pages=[],
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


# ======================================================================
# Module-level helpers (layout mode)
# ======================================================================


def _reassemble_markdown(layout_items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in layout_items:
        label = item.get("category", "").strip().lower()
        text = item.get("text", "")
        if not text:
            continue

        if label == "title":
            parts.append(f"# {text}")
        elif label in ("section-header", "section_header"):
            parts.append(f"## {text}")
        elif label == "table":
            parts.append(text)
        elif label == "formula":
            parts.append(f"$${text}$$")
        else:
            parts.append(text)

    return "\n\n".join(parts)


def _build_layout_page(
    *,
    layout_items: list[dict[str, Any]],
    page_number: int,
    img_width: int,
    img_height: int,
    page_markdown: str,
) -> ParseLayoutPageIR:
    items: list[LayoutItemIR] = []
    for li in layout_items:
        bbox = li.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        nx = x1 / 1000.0
        ny = y1 / 1000.0
        nw = (x2 - x1) / 1000.0
        nh = (y2 - y1) / 1000.0

        category = li.get("category", "Text")
        text = li.get("text", "")

        seg = LayoutSegmentIR(x=nx, y=ny, w=nw, h=nh, confidence=1.0, label=category)

        norm_label = category.strip().lower()
        if norm_label == "table":
            item_type = "table"
        elif norm_label == "picture":
            item_type = "image"
        else:
            item_type = "text"

        items.append(LayoutItemIR(type=item_type, value=text, bbox=seg, layout_segments=[seg]))

    return ParseLayoutPageIR(
        page_number=page_number,
        width=float(img_width),
        height=float(img_height),
        md=page_markdown,
        items=items,
    )
