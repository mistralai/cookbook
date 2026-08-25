# ruff: noqa: E501 — prompt strings are verbatim from chandra.prompts, cannot be line-wrapped
"""Provider for Chandra OCR 2 self-hosted servers.

Chandra OCR 2 (datalab-to/chandra-ocr-2, 5B) is a Qwen 3.5-based multimodal VLM
that outputs structured HTML with layout bounding boxes. It handles layout detection
internally via the OCR_LAYOUT prompt.

The model outputs HTML natively — tables are <table> elements, so no pipe-table-to-HTML
conversion is needed. We strip <div data-bbox> layout wrappers and sanitize attributes
for XML-based metric parsers.

This provider supports two API formats:
- "openai": OpenAI-compatible vLLM API (for chandra2_server.py)
- "simple": JSON API with image_base64 (for chandra2_sdk_server.py)
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

# Model name registered in vLLM (from chandra.settings.VLLM_MODEL_NAME)
SERVED_MODEL_NAME = "chandra"

# Prompts — exact copies from chandra.prompts
ALLOWED_TAGS = [
    "math",
    "br",
    "i",
    "b",
    "u",
    "del",
    "sup",
    "sub",
    "table",
    "tr",
    "td",
    "p",
    "th",
    "div",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "ul",
    "ol",
    "li",
    "input",
    "a",
    "span",
    "img",
    "hr",
    "tbody",
    "small",
    "caption",
    "strong",
    "thead",
    "big",
    "code",
    "chem",
]
ALLOWED_ATTRIBUTES = [
    "class",
    "colspan",
    "rowspan",
    "display",
    "checked",
    "type",
    "border",
    "value",
    "style",
    "href",
    "alt",
    "align",
    "data-bbox",
    "data-label",
]

_PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()

TASK_PROMPTS = {
    "ocr_layout": f"""
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page

{_PROMPT_ENDING}
""".strip(),
    "ocr": f"""
OCR this image to HTML.

{_PROMPT_ENDING}
""".strip(),
}


# Chandra OCR 2 label → Canonical17 mapping (from ChandraLayoutDetLabelAdapter)
CHANDRA2_LABEL_MAP: dict[str, str] = {
    "Caption": "Caption",
    "Footnote": "Footnote",
    "Equation-Block": "Formula",
    "List-Group": "List-item",
    "Page-Header": "Page-header",
    "Page-Footer": "Page-footer",
    "Image": "Picture",
    "Section-Header": "Section-header",
    "Table": "Table",
    "Text": "Text",
    "Complex-Block": "Text",
    "Code-Block": "Code",
    "Form": "Form",
    "Table-Of-Contents": "Document Index",
    "Figure": "Picture",
    "Chemical-Block": "Text",
    "Diagram": "Picture",
    "Bibliography": "Text",
}


@register_provider("chandra2")
class Chandra2Provider(Provider):
    """
    Provider for Chandra OCR 2 self-hosted servers.

    Configuration options:
        - server_url (str, required): server URL
        - api_format (str, default="openai"): "openai" or "simple"
        - task (str, default="ocr_layout"): Task prompt — "ocr_layout" or "ocr"
        - timeout (int, default=600): Request timeout in seconds
        - dpi (int, default=192): DPI for PDF to image conversion (matches chandra.settings.IMAGE_DPI)
        - api_key_env (str, default="VLLM_API_KEY"): Env var for API key (openai format only)
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("CHANDRA2_SERVER_URL")
        if not server_url:
            raise ProviderConfigError(
                "Chandra2 provider requires 'server_url' in config. "
                "Example: https://<your-chandra2-deployment>.modal.run"
            )
        self._server_url: str = str(server_url)

        self._api_format = self.base_config.get("api_format", "openai")
        if self._api_format not in ("openai", "simple"):
            raise ProviderConfigError(f"Invalid api_format '{self._api_format}'. Must be 'openai' or 'simple'.")

        self._task = self.base_config.get("task", "ocr_layout")
        if self._task not in TASK_PROMPTS:
            raise ProviderConfigError(f"Invalid task '{self._task}'. Must be one of: {list(TASK_PROMPTS.keys())}")

        self._timeout = self.base_config.get("timeout", 600)
        self._dpi = self.base_config.get("dpi", 192)

        # API key for authenticated vLLM endpoints
        api_key_env = self.base_config.get("api_key_env", "VLLM_API_KEY")
        self._api_key = os.environ.get(api_key_env, "")

    def _pdf_to_image(self, pdf_path: Path) -> bytes:
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path, dpi=self._dpi)
            if not images:
                raise ProviderPermanentError(f"No pages found in PDF: {pdf_path}")
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            return buf.getvalue()
        except ImportError as e:
            raise ProviderPermanentError("pdf2image is required. Install with: pip install pdf2image") from e
        except Exception as e:
            if "pdf2image" in str(e).lower():
                raise
            raise ProviderPermanentError(f"Error converting PDF to image: {e}") from e

    def _read_image(self, file_path: Path) -> bytes:
        try:
            return file_path.read_bytes()
        except Exception as e:
            raise ProviderPermanentError(f"Error reading image file: {e}") from e

    async def _call_openai_api(self, session: aiohttp.ClientSession, image_b64: str) -> str:
        api_url = f"{self._server_url.rstrip('/')}/v1/chat/completions"

        payload = {
            "model": SERVED_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {
                            "type": "text",
                            "text": TASK_PROMPTS.get(self._task, TASK_PROMPTS["ocr_layout"]),
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "top_p": 0.1,
            "max_tokens": 12384,
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

    async def _call_simple_api(self, session: aiohttp.ClientSession, image_b64: str) -> dict[str, str]:
        api_url = self._server_url.rstrip("/")

        payload: dict[str, str] = {
            "image_base64": image_b64,
            "prompt_type": self._task,
        }

        async with session.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                if resp.status in (408, 502, 503, 504):
                    raise ProviderTransientError(f"HTTP {resp.status}: {error_text[:200]}")
                raise ProviderPermanentError(f"HTTP {resp.status}: {error_text[:200]}")

            result = await resp.json()
            if result.get("status") == "error":
                raise ProviderPermanentError(result.get("error", "Unknown error from API"))

            markdown = result.get("markdown", "")
            html = result.get("html", "")
            raw_html = result.get("raw_html", "")
            if not markdown and not html:
                raise ProviderPermanentError("Empty response from API")
            return {"markdown": markdown, "html": html, "raw_html": raw_html}

    async def _run_inference_async(self, image_bytes: bytes) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_bytes).decode()

        async with aiohttp.ClientSession() as session:
            if self._api_format == "simple":
                result = await self._call_simple_api(session, image_b64)
                return {
                    "markdown": result["markdown"],
                    "html": result["html"],
                    "raw_html": result.get("raw_html", ""),
                    "_source": "sdk",
                    "_config": {
                        "server_url": self._server_url,
                        "api_format": self._api_format,
                        "task": self._task,
                        "dpi": self._dpi,
                    },
                }
            else:
                raw_html = await self._call_openai_api(session, image_b64)
                return {
                    "markdown": raw_html,
                    "html": "",
                    "_source": "vllm",
                    "_config": {
                        "server_url": self._server_url,
                        "api_format": self._api_format,
                        "task": self._task,
                        "dpi": self._dpi,
                    },
                }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"Chandra2Provider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"Source file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            image_bytes = self._pdf_to_image(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
            image_bytes = self._read_image(file_path)
        else:
            raise ProviderPermanentError(
                f"Unsupported file type: {suffix}. Supported: .pdf, .png, .jpg, .jpeg, .webp, .tiff, .bmp"
            )

        try:
            raw_output = asyncio.run(self._run_inference_async(image_bytes))
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
                    "markdown": "",
                    "_error": error_msg,
                    "_error_type": type(e).__name__,
                    "_config": {
                        "server_url": self._server_url,
                        "api_format": self._api_format,
                        "task": self._task,
                        "dpi": self._dpi,
                    },
                },
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=latency_ms,
            )

    @staticmethod
    def _sanitize_html_attributes(markdown: str) -> str:
        """Quote unquoted HTML attributes for XML-based metric parsers."""

        def _quote_attrs(match: re.Match) -> str:
            tag_text = match.group(0)
            tag_text = re.sub(
                r'(\w+)=([^\s"\'<>=]+)',
                r'\1="\2"',
                tag_text,
            )
            return tag_text

        return re.sub(r"<[^>]+>", _quote_attrs, markdown)

    @staticmethod
    def _build_layout_pages(raw_html: str) -> list[ParseLayoutPageIR]:
        """Extract layout bboxes from raw Chandra OCR 2 HTML output.

        Parses <div data-bbox="x0 y0 x1 y1" data-label="Label"> elements.
        Bboxes are normalized 0-1000 in the model output; we convert to [0,1].
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw_html, "html.parser")
        top_level_divs = soup.find_all("div", recursive=False)

        if not top_level_divs:
            return []

        items: list[LayoutItemIR] = []
        for div in top_level_divs:
            label_raw = str(div.get("data-label", ""))
            bbox_str = str(div.get("data-bbox", ""))

            if not label_raw or not bbox_str:
                continue
            if label_raw == "Blank-Page":
                continue

            try:
                parts = bbox_str.strip().split()
                if len(parts) != 4:
                    continue
                x0, y0, x1, y1 = [float(p) for p in parts]
            except (ValueError, TypeError):
                continue

            # Convert from 0-1000 to normalized [0,1]
            nx = x0 / 1000.0
            ny = y0 / 1000.0
            nw = max(0, (x1 - x0) / 1000.0)
            nh = max(0, (y1 - y0) / 1000.0)

            canonical_label = CHANDRA2_LABEL_MAP.get(label_raw, "Text")

            seg = LayoutSegmentIR(
                x=nx,
                y=ny,
                w=nw,
                h=nh,
                confidence=1.0,
                label=canonical_label,
            )

            # Determine item type from label
            label_lower = label_raw.lower()
            if label_lower == "table":
                item_type = "table"
            elif label_lower in ("image", "figure", "diagram"):
                item_type = "image"
            else:
                item_type = "text"

            # Extract inner content for attribution
            content = str(div.decode_contents()).strip()

            items.append(
                LayoutItemIR(
                    type=item_type,
                    value=content,
                    bbox=seg,
                    layout_segments=[seg],
                )
            )

        if not items:
            return []

        return [
            ParseLayoutPageIR(
                page_number=1,
                width=1000.0,
                height=1000.0,
                items=items,
            )
        ]

    @staticmethod
    def _strip_layout_divs(raw_html: str) -> str:
        """Strip <div data-bbox data-label> layout wrappers from raw model output.

        Chandra OCR 2 outputs structured HTML like:
            <div data-bbox="..." data-label="Text"><p>content</p></div>
            <div data-bbox="..." data-label="Table"><table>...</table></div>

        This extracts the inner content of each div, skipping headers/footers/blanks,
        and concatenates them. The result has HTML tables intact.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw_html, "html.parser")
        top_level_divs = soup.find_all("div", recursive=False)

        # If no top-level divs, the output isn't structured — return as-is
        if not top_level_divs:
            return raw_html

        parts = []
        for div in top_level_divs:
            label = str(div.get("data-label", ""))
            if label in ("Page-Header", "Page-Footer", "Blank-Page"):
                continue
            content = str(div.decode_contents())
            if content.strip():
                parts.append(content)

        return "\n".join(parts) if parts else raw_html

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"Chandra2Provider only supports PARSE product type, got {raw_result.product_type}"
            )

        source = raw_result.raw_output.get("_source", "vllm")
        raw_markdown = raw_result.raw_output.get("markdown", "")

        # Build layout_pages from raw HTML (before stripping divs).
        # For vLLM: raw_markdown IS the structured HTML with <div data-bbox>.
        # For SDK: use raw_output["raw_html"] if available, else raw_markdown.
        layout_html = raw_markdown
        if source == "sdk":
            layout_html = raw_result.raw_output.get("raw_html", raw_markdown)
        layout_pages = self._build_layout_pages(layout_html) if layout_html else []

        # Now produce the clean markdown for parse evaluation
        markdown = raw_markdown
        if markdown:
            if source == "vllm":
                # vLLM returns raw structured HTML with <div data-bbox> wrappers.
                # Strip the layout divs to get clean HTML content with tables intact.
                markdown = self._strip_layout_divs(markdown)

            # SDK returns processed markdown with HTML tables preserved (via
            # chandra's Markdownify which keeps <table> elements as-is).
            # Both paths: sanitize HTML attributes for XML parsers.
            markdown = self._sanitize_html_attributes(markdown)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            markdown=markdown,
            layout_pages=layout_pages,
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
