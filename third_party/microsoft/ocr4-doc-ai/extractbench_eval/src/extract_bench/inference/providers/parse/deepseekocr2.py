"""Provider for DeepSeek-OCR-2 self-hosted server.

DeepSeek-OCR-2 (deepseek-ai/DeepSeek-OCR-2) is a MoE vision-language model
that handles layout detection + OCR in a single pass via the <|grounding|> token.

API format: POST /predict with {"image_base64": "..."} → {"markdown": "...", "status": "success"}
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


@register_provider("deepseekocr2")
class DeepSeekOCR2Provider(Provider):
    """
    Provider for DeepSeek-OCR-2 self-hosted server.

    Configuration options:
        - server_url (str, required): self-hosted server predict endpoint URL
        - timeout (int, default=600): Request timeout in seconds
        - dpi (int, default=150): DPI for PDF to image conversion
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("DEEPSEEKOCR2_SERVER_URL")
        if not server_url:
            raise ProviderConfigError("DeepSeekOCR2 provider requires 'server_url' in config.")
        self._server_url: str = server_url

        self._timeout = self.base_config.get("timeout", 600)
        self._dpi = self.base_config.get("dpi", 150)

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
            raise ProviderPermanentError("pdf2image is required.") from e
        except Exception as e:
            if "pdf2image" in str(e).lower():
                raise
            raise ProviderPermanentError(f"Error converting PDF to image: {e}") from e

    def _read_image(self, file_path: Path) -> bytes:
        try:
            return file_path.read_bytes()
        except Exception as e:
            raise ProviderPermanentError(f"Error reading image file: {e}") from e

    async def _call_api(self, session: aiohttp.ClientSession, image_b64: str) -> dict[str, Any]:
        api_url = self._server_url.rstrip("/")

        payload: dict[str, str] = {"image_base64": image_b64}

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

            result: dict[str, Any] = await resp.json()
            if result.get("status") == "error":
                raise ProviderPermanentError(result.get("error", "Unknown error from API"))

            content: str = result.get("markdown", "")
            if not content:
                raise ProviderPermanentError("Empty markdown response from API")
            return result

    async def _run_inference_async(self, image_bytes: bytes) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_bytes).decode()

        async with aiohttp.ClientSession() as session:
            result = await self._call_api(session, image_b64)

        return {
            "markdown": result.get("markdown", ""),
            "grounding_items": result.get("grounding_items", []),
            "image_width": result.get("image_width"),
            "image_height": result.get("image_height"),
            "_config": {
                "server_url": self._server_url,
                "dpi": self._dpi,
            },
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DeepSeekOCR2Provider only supports PARSE product type, got {request.product_type}"
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
                        "dpi": self._dpi,
                    },
                },
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=latency_ms,
            )

    @staticmethod
    def _close_unclosed_table_tags(content: str) -> str:
        """Auto-close unclosed HTML table tags from truncated model output."""
        opens = content.count("<table>")
        closes = content.count("</table>")
        if opens > closes:
            # Close any unclosed row/cell tags, then close the table
            if content.rstrip().endswith(">"):
                pass  # last tag is already closed
            else:
                # Truncated mid-cell — close the cell and row
                content += "</td></tr>"
            content += "</table>" * (opens - closes)
        return content

    @staticmethod
    def _promote_first_row_to_thead(content: str) -> str:
        """Wrap the first <tr> of each <table> in <thead> and convert <td> to <th>.

        The grounding model outputs all cells as <td>, never using <th>/<thead>.
        This heuristic promotes the first row to a header row, matching how
        markdown2 handles pipe tables and improving header metric scores.
        """

        def _promote_table(match: re.Match[str]) -> str:
            table_html: str = match.group(0)
            # Find the first <tr>...</tr>
            first_tr = re.search(r"<tr>(.*?)</tr>", table_html, re.DOTALL)
            if not first_tr:
                return table_html
            first_tr_full: str = first_tr.group(0)
            first_tr_inner: str = first_tr.group(1)
            # Convert <td> to <th> in the first row
            header_inner = first_tr_inner.replace("<td>", "<th>").replace("</td>", "</th>")
            # Also handle <td with attributes
            header_inner = re.sub(r"<td(\s)", r"<th\1", header_inner)
            header_inner = re.sub(r"</td>", "</th>", header_inner)
            thead = f"<thead><tr>{header_inner}</tr></thead>"
            # Replace first <tr> with <thead> block
            table_html = table_html.replace(first_tr_full, thead, 1)
            return table_html

        return re.sub(r"<table>.*?</table>", _promote_table, content, flags=re.DOTALL)

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
    def _convert_md_tables_to_html(content: str) -> str:
        """Convert markdown pipe tables to HTML for GriTS/TEDS metrics."""
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

    # Label mapping: DeepSeek-OCR-2 grounding labels → Canonical17-compatible
    LABEL_MAP: dict[str, str] = {
        "image": "Picture",
        "title": "Title",
        "table": "Table",
        "figure": "Picture",
        "caption": "Caption",
        "footnote": "Footnote",
        "header": "Page-header",
        "footer": "Page-footer",
    }

    @staticmethod
    def _build_layout_pages(
        grounding_items: list[dict[str, Any]],
        image_width: int,
        image_height: int,
        markdown: str,
    ) -> list[ParseLayoutPageIR]:
        """Convert grounding items (0-999 grid bboxes) to ParseLayoutPageIR."""
        if not grounding_items or not image_width or not image_height:
            return []

        items: list[LayoutItemIR] = []
        for gi in grounding_items:
            bbox = gi.get("bbox", [])
            label_raw = gi.get("label", "text")
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            # Convert from 0-999 grid to normalized [0,1]
            nx = x1 / 999.0
            ny = y1 / 999.0
            nw = (x2 - x1) / 999.0
            nh = (y2 - y1) / 999.0

            label = DeepSeekOCR2Provider.LABEL_MAP.get(label_raw.lower(), "Text")

            seg = LayoutSegmentIR(
                x=nx,
                y=ny,
                w=nw,
                h=nh,
                confidence=1.0,
                label=label,
            )

            norm_label = label_raw.lower()
            if norm_label == "table":
                item_type = "table"
            elif norm_label in ("image", "figure"):
                item_type = "image"
            else:
                item_type = "text"

            items.append(
                LayoutItemIR(
                    type=item_type,
                    value="",
                    bbox=seg,
                    layout_segments=[seg],
                )
            )

        if not items:
            return []

        return [
            ParseLayoutPageIR(
                page_number=1,
                width=float(image_width),
                height=float(image_height),
                md=markdown,
                items=items,
            )
        ]

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DeepSeekOCR2Provider only supports PARSE product type, got {raw_result.product_type}"
            )

        markdown = raw_result.raw_output.get("markdown", "")
        if markdown:
            # Auto-close unclosed HTML table tags (model truncates at max_tokens)
            markdown = self._close_unclosed_table_tags(markdown)
            markdown = self._convert_md_tables_to_html(markdown)
            # Promote first row to <thead>/<th> (model outputs all <td>)
            markdown = self._promote_first_row_to_thead(markdown)
            markdown = self._sanitize_html_attributes(markdown)

        # Build layout pages from grounding items (if available)
        grounding_items = raw_result.raw_output.get("grounding_items", [])
        image_width = raw_result.raw_output.get("image_width", 0)
        image_height = raw_result.raw_output.get("image_height", 0)
        layout_pages = self._build_layout_pages(grounding_items, image_width, image_height, markdown)

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
