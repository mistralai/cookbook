"""Provider for MinerU-Diffusion-V1 self-hosted server.

MinerU-Diffusion-V1 (opendatalab/MinerU-Diffusion-V1-0320-2.5B) is a
diffusion-decoding document-OCR VLM. The server runs the official two-stage
parse: layout detection on a 1036x1036 page, then per-region recognition
(table -> OTSL, formula -> LaTeX, else text), merged into markdown with OTSL
converted to HTML <table> server-side.

API format: POST {server_url} with {"image_base64": "..."} ->
    {"markdown": "...", "blocks": [...], "image_width", "image_height",
     "num_blocks", "latency_s", "status": "success"}

Each block is {"type": str, "bbox": [x1,y1,x2,y2] normalized [0,1], "angle",
"content": raw region text, "rendered": markdown/HTML form}. `blocks` carries
EVERY detected region (incl. image + paratext) so layout detection is scorable,
while `markdown` already drops images/paratext per the official pipeline.
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


@register_provider("mineru_diffusion")
class MinerUDiffusionProvider(Provider):
    """Provider for a self-hosted MinerU-Diffusion-V1 server.

    Config:
        - server_url (str, required): POST /predict endpoint. May also be
          supplied via the ``MINERU_DIFFUSION_SERVER_URL`` environment variable.
        - timeout (int, default=900): request timeout seconds (two-stage parse is
          many sequential diffusion calls, so allow generous time)
        - dpi (int, default=150): PDF -> image render DPI
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("MINERU_DIFFUSION_SERVER_URL")
        if not server_url:
            raise ProviderConfigError(
                "MinerUDiffusion provider requires 'server_url' in config or "
                "MINERU_DIFFUSION_SERVER_URL in the environment."
            )
        self._server_url: str = str(server_url)
        self._timeout = self.base_config.get("timeout", 900)
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
        except ProviderPermanentError:
            raise
        except Exception as e:
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
            return result

    async def _run_inference_async(self, image_bytes: bytes) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_bytes).decode()
        async with aiohttp.ClientSession() as session:
            result = await self._call_api(session, image_b64)
        return {
            "markdown": result.get("markdown", ""),
            "blocks": result.get("blocks", []),
            "image_width": result.get("image_width"),
            "image_height": result.get("image_height"),
            "_config": {"server_url": self._server_url, "dpi": self._dpi},
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"MinerUDiffusionProvider only supports PARSE product type, got {request.product_type}"
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

    # -----------------------------------------------------------------------
    # Normalization helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _sanitize_html_attributes(markdown: str) -> str:
        """Quote unquoted HTML attribute values so XML-based table parsers accept
        the server's HTML tables."""

        def _quote_attrs(match: re.Match) -> str:
            tag = match.group(0)
            return re.sub(r'(\w+)=([^\s"\'<>=]+)', r'\1="\2"', tag)

        return re.sub(r"<[^>]+>", _quote_attrs, markdown)

    # MinerU-Diffusion layout labels -> Canonical17 (everything emitted MUST be a
    # valid Canonical17 string or the layout evaluator silently drops it).
    LABEL_MAP: dict[str, str] = {
        "text": "Text",
        "title": "Section-header",  # Title + Section-header both project to Basic7 "Section"
        "doc_title": "Title",
        "paragraph_title": "Section-header",
        "table": "Table",
        "table_caption": "Caption",
        "table_footnote": "Footnote",
        "image": "Picture",
        "figure": "Picture",
        "chart": "Picture",
        "image_caption": "Caption",
        "figure_caption": "Caption",
        "image_footnote": "Footnote",
        "equation": "Formula",
        "equation_block": "Formula",
        "formula": "Formula",
        "formula_caption": "Caption",
        "code": "Code",
        "code_caption": "Caption",
        "header": "Page-header",
        "page_header": "Page-header",
        "footer": "Page-footer",
        "page_footer": "Page-footer",
        "page_number": "Page-footer",
        "page_footnote": "Footnote",
        "footnote": "Footnote",
        "aside_text": "Text",
        "list": "List-item",
        "list_item": "List-item",
        "index": "Document Index",
        "unknown": "Text",
    }

    @staticmethod
    def _build_layout_pages(
        blocks: list[dict[str, Any]],
        image_width: int,
        image_height: int,
        markdown: str,
    ) -> list[ParseLayoutPageIR]:
        if not blocks or not image_width or not image_height:
            return []

        items: list[LayoutItemIR] = []
        for blk in blocks:
            bbox = blk.get("bbox", [])
            raw_label = (blk.get("type") or "text").lower()
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            x1 = max(0.0, min(1.0, float(x1)))
            y1 = max(0.0, min(1.0, float(y1)))
            x2 = max(0.0, min(1.0, float(x2)))
            y2 = max(0.0, min(1.0, float(y2)))

            nx, ny = x1, y1
            nw = max(0.0, x2 - x1)
            nh = max(0.0, y2 - y1)

            label = MinerUDiffusionProvider.LABEL_MAP.get(raw_label, "Text")

            seg = LayoutSegmentIR(x=nx, y=ny, w=nw, h=nh, confidence=1.0, label=label)

            if raw_label == "table":
                item_type = "table"
            elif raw_label in ("image", "figure", "chart"):
                item_type = "image"
            else:
                item_type = "text"

            # Prefer the rendered (HTML/markdown) form for attribution text.
            value = str(blk.get("rendered") or blk.get("content") or "")

            items.append(
                LayoutItemIR(
                    type=item_type,
                    value=value,
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
                f"MinerUDiffusionProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        markdown = raw_result.raw_output.get("markdown", "")
        if markdown:
            # Tables are already HTML (OTSL converted server-side); just make the
            # attributes XML-safe for the table evaluators.
            markdown = self._sanitize_html_attributes(markdown)

        blocks = raw_result.raw_output.get("blocks", [])
        image_width = raw_result.raw_output.get("image_width", 0)
        image_height = raw_result.raw_output.get("image_height", 0)
        layout_pages = self._build_layout_pages(blocks, image_width, image_height, markdown)

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
