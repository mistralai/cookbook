"""Provider for Unlimited-OCR server.

Unlimited-OCR (baidu/Unlimited-OCR) is a DeepSeek-OCR successor: a
vision-language model that does layout detection + OCR in a single pass, emitting
DeepSeek-OCR grounding tags (<|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>,
coords on a 0-999 grid) interleaved with markdown. Same output format as
DeepSeek-OCR-2, so the normalization mirrors that provider.

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
from extract_bench.inference.providers.parse._layout_utils import build_layout_pages
from extract_bench.inference.providers.parse.mistral_ocr import _convert_pipe_tables_to_html
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.parse_output import ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType


@register_provider("unlimitedocr")
class UnlimitedOCRProvider(Provider):
    """
    Provider for Unlimited-OCR server.

    Configuration options:
        - server_url (str, required): Server predict endpoint URL. Falls back to
          the UNLIMITEDOCR_SERVER_URL environment variable.
        - timeout (int, default=1200): Request timeout in seconds
        - dpi (int, default=300): DPI for PDF to image conversion
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("UNLIMITEDOCR_SERVER_URL")
        if not server_url:
            raise ProviderConfigError("UnlimitedOCR provider requires 'server_url' in config.")
        self._server_url: str = server_url

        # Match the model's reference config: PDF_DPI=300, REQUEST_TIMEOUT=1200.
        self._timeout = self.base_config.get("timeout", 1200)
        self._dpi = self.base_config.get("dpi", 300)

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
                f"UnlimitedOCRProvider only supports PARSE product type, got {request.product_type}"
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

    # Grounding label aliases -> the names the shared _layout_utils.LABEL_MAP
    # understands (title/table/caption/footnote already match; these three differ).
    _LABEL_ALIASES: dict[str, str] = {
        "header": "page-header",
        "footer": "page-footer",
        "image": "picture",
    }

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"UnlimitedOCRProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        markdown = raw_result.raw_output.get("markdown", "")
        if markdown:
            # Auto-close unclosed HTML table tags (model truncates at max_tokens)
            markdown = self._close_unclosed_table_tags(markdown)
            # Convert any markdown pipe tables to HTML (shared helper)
            markdown = _convert_pipe_tables_to_html(markdown)
            # Promote first row to <thead>/<th> (model outputs all <td>)
            markdown = self._promote_first_row_to_thead(markdown)
            markdown = self._sanitize_html_attributes(markdown)

        # Build layout pages from grounding items via the shared builder. The
        # grounding bboxes are [x1,y1,x2,y2] on a 0-999 grid (build_layout_pages
        # divides by 1000); aliasing maps grounding labels onto its LABEL_MAP.
        grounding_items = raw_result.raw_output.get("grounding_items", [])
        image_width = raw_result.raw_output.get("image_width", 0)
        image_height = raw_result.raw_output.get("image_height", 0)
        layout_items = [
            {
                "label": self._LABEL_ALIASES.get(str(gi.get("label", "")).lower(), gi.get("label", "")),
                "bbox": gi.get("bbox", []),
            }
            for gi in grounding_items
        ]
        layout_pages = build_layout_pages(
            items=layout_items,
            image_width=image_width,
            image_height=image_height,
            markdown=markdown,
            page_number=1,
        )

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
