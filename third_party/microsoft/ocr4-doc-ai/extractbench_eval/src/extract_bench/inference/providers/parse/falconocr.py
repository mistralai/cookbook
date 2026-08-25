"""Provider for Falcon-OCR server.

Falcon-OCR (tiiuae/Falcon-OCR) is a 300M early-fusion document OCR VLM
with built-in layout-aware OCR via `generate_with_layout`. The server
exposes a simple JSON endpoint at /predict that accepts a base64 image
and returns assembled markdown plus per-region layout metadata.
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
from extract_bench.schemas.layout_ontology import CanonicalLabel
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

# Falcon-OCR uses PP-DocLayoutV3 internally, so the raw region labels match
# the PP-DocLayoutV3 label set.
_FALCONOCR_LABEL_TO_CANONICAL: dict[str, tuple[str, dict[str, str]]] = {
    "doc_title": (CanonicalLabel.TITLE.value, {"title_level": "document"}),
    "paragraph_title": (CanonicalLabel.SECTION_HEADER.value, {"title_level": "paragraph"}),
    "text": (CanonicalLabel.TEXT.value, {}),
    "vertical_text": (CanonicalLabel.TEXT.value, {"text_role": "vertical"}),
    "number": (CanonicalLabel.TEXT.value, {"text_role": "page_number"}),
    "abstract": (CanonicalLabel.TEXT.value, {"text_role": "abstract"}),
    "content": (CanonicalLabel.TEXT.value, {"text_role": "body"}),
    "reference": (CanonicalLabel.TEXT.value, {"text_role": "references"}),
    "aside_text": (CanonicalLabel.TEXT.value, {"text_role": "sidebar"}),
    "reference_content": (CanonicalLabel.TEXT.value, {"text_role": "references"}),
    "formula_number": (CanonicalLabel.TEXT.value, {"text_role": "formula_number"}),
    "header": (CanonicalLabel.PAGE_HEADER.value, {"furniture": "page-header"}),
    "header_image": (CanonicalLabel.PAGE_HEADER.value, {"furniture": "page-header"}),
    "footer": (CanonicalLabel.PAGE_FOOTER.value, {"furniture": "page-footer"}),
    "footer_image": (CanonicalLabel.PAGE_FOOTER.value, {"furniture": "page-footer"}),
    "footnote": (CanonicalLabel.FOOTNOTE.value, {}),
    "vision_footnote": (CanonicalLabel.FOOTNOTE.value, {"footnote_of": "picture"}),
    "image": (CanonicalLabel.PICTURE.value, {"picture_type": "image"}),
    "chart": (CanonicalLabel.PICTURE.value, {"picture_type": "chart"}),
    "seal": (CanonicalLabel.PICTURE.value, {"picture_type": "seal"}),
    "figure_title": (CanonicalLabel.CAPTION.value, {"caption_of": "picture"}),
    "table": (CanonicalLabel.TABLE.value, {}),
    "formula": (CanonicalLabel.FORMULA.value, {}),
    "display_formula": (CanonicalLabel.FORMULA.value, {"formula_style": "display"}),
    "inline_formula": (CanonicalLabel.FORMULA.value, {"formula_style": "inline"}),
    "algorithm": (CanonicalLabel.CODE.value, {}),
}


def _regions_to_layout_items(regions: list[dict[str, Any]]) -> list[LayoutItemIR]:
    """Map Falcon-OCR `generate_with_layout` regions to LayoutItemIR.

    Each region is `{category, bbox: [x0,y0,x1,y1], score, text}` where text
    already has markdown formatting baked in by the model.
    """
    items: list[LayoutItemIR] = []
    for region in regions:
        label_raw = str(region.get("category", "")).strip().lower()
        mapping = _FALCONOCR_LABEL_TO_CANONICAL.get(label_raw)
        if mapping is None:
            continue
        canonical, _attrs = mapping

        bbox = region.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue

        try:
            score = float(region.get("score", 1.0))
        except (TypeError, ValueError):
            score = 1.0
        score = max(0.0, min(1.0, score))

        seg = LayoutSegmentIR(
            x=x1,
            y=y1,
            w=max(0.0, x2 - x1),
            h=max(0.0, y2 - y1),
            confidence=score,
            label=canonical,
        )

        text = region.get("text") or ""
        item_md = ""
        item_html = ""
        item_value = ""
        norm = canonical.strip().lower()
        if text and norm != "picture":
            if norm == "table":
                item_html = str(text)
                item_type = "table"
            else:
                item_md = str(text)
                item_value = str(text)
                item_type = "text"
        elif norm == "picture":
            item_type = "image"
        else:
            item_type = "text"

        items.append(
            LayoutItemIR(
                type=item_type,
                md=item_md,
                html=item_html,
                value=item_value,
                bbox=seg,
                layout_segments=[seg],
            )
        )
    return items


@register_provider("falconocr")
class FalconOcrProvider(Provider):
    """Provider for Falcon-OCR server.

    Configuration options:
        - server_url (str): server URL root (no /predict). Falls back to
          the ``FALCONOCR_SERVER_URL`` environment variable.
        - task (str, default="ocr"): "ocr" (layout-aware) or a generate()
          category like "plain", "text", "table", "formula".
        - timeout (int, default=600): Request timeout in seconds.
        - dpi (int, default=200): DPI for PDF-to-image conversion.
        - max_new_tokens (int, default=4096): Generation budget.
        - temperature (float, default=0.0): Sampling temperature.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("FALCONOCR_SERVER_URL")
        if not server_url:
            raise ProviderConfigError(
                "FalconOCR provider requires 'server_url' in config or FALCONOCR_SERVER_URL in the environment."
            )
        self._server_url: str = str(server_url).rstrip("/")
        self._task: str = str(self.base_config.get("task", "ocr"))
        self._timeout = int(self.base_config.get("timeout", 600))
        self._dpi = int(self.base_config.get("dpi", 200))
        self._max_new_tokens = int(self.base_config.get("max_new_tokens", 4096))
        self._temperature = float(self.base_config.get("temperature", 0.0))

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

    async def _call_api(self, session: aiohttp.ClientSession, image_b64: str) -> dict[str, Any]:
        api_url = f"{self._server_url}/predict"
        payload = {
            "image_base64": image_b64,
            "task": self._task,
            "max_new_tokens": self._max_new_tokens,
            "temperature": self._temperature,
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
            result: dict[str, Any] = await resp.json()

        if result.get("status") != "success":
            raise ProviderPermanentError(
                f"Server returned status={result.get('status')}: {str(result.get('error'))[:200]}"
            )
        return result

    async def _run_inference_async(self, image_bytes: bytes) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_bytes).decode()

        async with aiohttp.ClientSession() as session:
            response = await self._call_api(session, image_b64)

        return {
            "markdown": response.get("markdown", ""),
            "regions": response.get("regions", []),
            "image_width": response.get("image_width"),
            "image_height": response.get("image_height"),
            "_task_used": response.get("task"),
            "_config": {
                "server_url": self._server_url,
                "task": self._task,
                "dpi": self._dpi,
                "max_new_tokens": self._max_new_tokens,
                "temperature": self._temperature,
            },
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"FalconOcrProvider only supports PARSE product type, got {request.product_type}"
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
                    "_config": {"server_url": self._server_url, "dpi": self._dpi},
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
    def _convert_md_tables_to_html(content: str) -> str:
        """Convert markdown pipe tables to HTML <table> elements.

        Falcon-OCR's table category emits HTML <table> directly, but mixed
        outputs (e.g. plain task on a doc with tables) may include pipe
        tables. GriTS/TEDS metrics only parse HTML, so we convert.
        """
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

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"FalconOcrProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        markdown = raw_result.raw_output.get("markdown", "")
        if markdown:
            markdown = self._convert_md_tables_to_html(markdown)
            markdown = self._sanitize_html_attributes(markdown)

        regions = raw_result.raw_output.get("regions") or []
        image_width = int(raw_result.raw_output.get("image_width") or 1)
        image_height = int(raw_result.raw_output.get("image_height") or 1)
        image_width = max(image_width, 1)
        image_height = max(image_height, 1)

        items = _regions_to_layout_items(regions)
        layout_pages: list[ParseLayoutPageIR] = []
        if items:
            layout_pages.append(
                ParseLayoutPageIR(
                    page_number=1,
                    width=float(image_width),
                    height=float(image_height),
                    items=items,
                )
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
