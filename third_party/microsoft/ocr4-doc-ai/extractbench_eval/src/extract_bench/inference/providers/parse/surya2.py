"""Provider for a self-hosted Surya OCR 2 SDK server.

Surya OCR 2 (datalab-to/surya-ocr-2, 650M VLM, Qwen 3.5-style) does full-page
OCR via the official surya-ocr SDK, returning reading-ordered blocks with HTML
content and pixel-space polygons. The SDK server assembles page-level HTML
(tables preserved as <table>) and returns per-block layout, so this provider
only consumes the "simple" JSON API.

We sanitize HTML attributes for XML-based metric parsers and build layout_pages
from the per-block polygons + labels (mapped to the canonical layout vocabulary).
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

# Surya OCR 2 layout label → bench Canonical17. The bench layout evaluator
# resolves each predicted label against CanonicalLabel (case-insensitively) and
# then projects down to the GT's ontology (e.g. Basic7: Section-header→Section,
# List-item/Caption/Footnote/Formula/Code/Document Index→Text). Every target
# string below is therefore a valid Canonical17 value — anything unrecognized
# would be silently dropped by the evaluator, so the default falls back to "Text".
#
# Surya's SDK emits the *post-relabel* camelCase labels
# (surya.recognition.LAYOUT_PRED_RELABEL values) on each block. We also accept
# the raw, pre-relabel prompt labels (hyphenated) defensively, in case a future
# surya version surfaces them before relabeling — both forms resolve identically.
SURYA2_LABEL_MAP: dict[str, str] = {
    # Post-relabel labels (the form the surya SDK actually emits)
    "Text": "Text",
    "SectionHeader": "Section-header",
    "Table": "Table",
    "Equation": "Formula",
    "PageHeader": "Page-header",
    "PageFooter": "Page-footer",
    "ListGroup": "List-item",
    "Caption": "Caption",
    "Footnote": "Footnote",
    "Picture": "Picture",
    "Code": "Code",
    "Form": "Form",
    "TableOfContents": "Document Index",
    "Figure": "Picture",
    "ChemicalBlock": "Text",
    "Diagram": "Picture",
    "Bibliography": "Text",
    "BlankPage": "Text",
    # Raw prompt labels (pre-relabel) — composed through LAYOUT_PRED_RELABEL.
    "Section-Header": "Section-header",
    "Equation-Block": "Formula",
    "Page-Header": "Page-header",
    "Page-Footer": "Page-footer",
    "List-Group": "List-item",
    "Image": "Picture",
    "Complex-Block": "Picture",
    "Code-Block": "Code",
    "Table-Of-Contents": "Document Index",
    "Chemical-Block": "Text",
    "Blank-Page": "Text",
}


@register_provider("surya2")
class Surya2Provider(Provider):
    """
    Provider for a self-hosted Surya OCR 2 SDK server.

    Configuration options:
        - server_url (str, required): SDK server /predict URL. Falls back to the
          SURYA2_SERVER_URL environment variable.
        - timeout (int, default=600): Request timeout in seconds
        - dpi (int, default=192): DPI for PDF→image (matches surya IMAGE_DPI_HIGHRES)
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("SURYA2_SERVER_URL")
        if not server_url:
            raise ProviderConfigError(
                "Surya2 provider requires 'server_url' in config or the "
                "SURYA2_SERVER_URL environment variable (the SDK server /predict URL)."
            )
        self._server_url: str = str(server_url)
        self._timeout = self.base_config.get("timeout", 600)
        self._dpi = self.base_config.get("dpi", 192)

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

    async def _call_simple_api(self, session: aiohttp.ClientSession, image_b64: str) -> dict[str, Any]:
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

            markdown = result.get("markdown", "")
            if not markdown and not result.get("blocks"):
                raise ProviderPermanentError("Empty response from API")
            return result

    async def _run_inference_async(self, image_bytes: bytes) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_bytes).decode()
        async with aiohttp.ClientSession() as session:
            result = await self._call_simple_api(session, image_b64)
            return {
                "markdown": result.get("markdown", ""),
                "html": result.get("html", ""),
                "blocks": result.get("blocks", []),
                "image_width": result.get("image_width", 0),
                "image_height": result.get("image_height", 0),
                "_config": {
                    "server_url": self._server_url,
                    "dpi": self._dpi,
                },
            }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"Surya2Provider only supports PARSE product type, got {request.product_type}")

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
            return re.sub(r'(\w+)=([^\s"\'<>=]+)', r'\1="\2"', tag_text)

        return re.sub(r"<[^>]+>", _quote_attrs, markdown)

    def _build_layout_pages(self, blocks: list[dict[str, Any]], width: float, height: float) -> list[ParseLayoutPageIR]:
        """Build layout pages from Surya OCR 2 per-block polygons (pixel coords)."""
        if not blocks or width <= 0 or height <= 0:
            return []

        items: list[LayoutItemIR] = []
        for block in blocks:
            bbox = block.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            raw_label = str(block.get("label", "Text"))
            canonical_label = SURYA2_LABEL_MAP.get(raw_label, "Text")

            x0, y0, x1, y1 = (float(v) for v in bbox)
            nx = x0 / width
            ny = y0 / height
            nw = max(0.0, (x1 - x0) / width)
            nh = max(0.0, (y1 - y0) / height)

            conf = block.get("confidence")
            seg = LayoutSegmentIR(
                x=nx,
                y=ny,
                w=nw,
                h=nh,
                confidence=float(conf) if conf is not None else 1.0,
                label=canonical_label,
            )

            label_lower = raw_label.lower()
            if label_lower == "table":
                item_type = "table"
            elif label_lower in ("picture", "figure", "diagram", "image"):
                item_type = "image"
            else:
                item_type = "text"

            items.append(
                LayoutItemIR(
                    type=item_type,
                    value=str(block.get("html", "")).strip(),
                    bbox=seg,
                    layout_segments=[seg],
                )
            )

        if not items:
            return []

        return [
            ParseLayoutPageIR(
                page_number=1,
                width=float(width),
                height=float(height),
                items=items,
            )
        ]

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"Surya2Provider only supports PARSE product type, got {raw_result.product_type}"
            )

        markdown = raw_result.raw_output.get("markdown", "")
        if markdown:
            markdown = self._sanitize_html_attributes(markdown)

        blocks = raw_result.raw_output.get("blocks", []) or []
        width = float(raw_result.raw_output.get("image_width", 0) or 0)
        height = float(raw_result.raw_output.get("image_height", 0) or 0)
        layout_pages = self._build_layout_pages(blocks, width, height)

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
