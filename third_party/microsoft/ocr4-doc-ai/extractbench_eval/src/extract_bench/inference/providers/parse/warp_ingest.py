"""Provider for Warp-Ingest PARSE."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.parse_output import PageIR, ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

_DEFAULT_PARSE_OPTIONS: dict[str, Any] = {
    "apply_ocr": False,
    "render_format": "all",
}
_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".jfif"}

ParsePayloadFn = Callable[..., dict[str, Any]]
RenderPagesFn = Callable[[dict[str, Any]], list[tuple[int, str]]]

# warp_ingest.ingestor imports fail at load time unless these NLTK corpora are
# present (punkt/punkt_tab back the sentence tokenizer, stopwords back the line
# parser). The package ships no post-install hook, so ensure them ourselves —
# once per process, only downloading what is missing — so warp_ingest works in
# fresh CI runners without a separate setup step.
_NLTK_CORPORA: dict[str, str] = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
}
_nltk_lock = threading.Lock()
_nltk_ready = False


def ensure_nltk_corpora() -> None:
    """Download the NLTK corpora warp_ingest needs, once, if not already present."""
    global _nltk_ready
    if _nltk_ready:
        return
    with _nltk_lock:
        if _nltk_ready:
            return
        try:
            import nltk
        except ImportError:
            # warp_ingest itself depends on nltk; let its own import raise the
            # clearer "not installed" error downstream.
            _nltk_ready = True
            return
        for resource, path in _NLTK_CORPORA.items():
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource, quiet=True)
        _nltk_ready = True


def _load_markdown_exporter() -> tuple[ParsePayloadFn, RenderPagesFn]:
    ensure_nltk_corpora()
    try:
        from warp_ingest.ingestor.markdown_exporter import (
            parse_to_markdown_payload,
            render_pages,
        )
    except ImportError as exc:
        raise ProviderConfigError(
            "warp-ingest[ocr]>=2.0.1 not installed. Run: pip install 'warp-ingest[ocr]>=2.0.1'"
        ) from exc

    return parse_to_markdown_payload, render_pages


@register_provider("warp_ingest")
class WarpIngestProvider(Provider):
    """Provider for Warp-Ingest (local, no API key)."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        configured_parse_options = self.base_config.get("parse_options")
        if configured_parse_options is None:
            configured_parse_options = {}
        if not isinstance(configured_parse_options, dict):
            raise ProviderConfigError("warp_ingest parse_options must be a dictionary")
        self._parse_options = {**_DEFAULT_PARSE_OPTIONS, **configured_parse_options}
        self._include_native_tables = bool(self.base_config.get("include_native_tables", True))

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"WarpIngestProvider only supports PARSE, got {request.product_type}")

        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"Input file not found: {pdf_path}")
        if pdf_path.suffix.lower() not in {".pdf", *_SUPPORTED_IMAGE_SUFFIXES}:
            raise ProviderPermanentError(
                "WarpIngestProvider only supports .pdf and image files "
                f"({', '.join(sorted(_SUPPORTED_IMAGE_SUFFIXES))}), got {pdf_path.suffix}"
            )

        parse_to_markdown_payload = _load_markdown_exporter()[0]

        started_at = datetime.now()
        temp_pdf_path: Path | None = None
        try:
            input_path = pdf_path
            if pdf_path.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES:
                temp_pdf_path = self._convert_image_to_pdf(pdf_path)
                input_path = temp_pdf_path

            raw_output = parse_to_markdown_payload(
                str(input_path),
                parse_options=self._parse_options,
                include_native_tables=self._include_native_tables,
            )
        except Exception as exc:
            raise ProviderPermanentError(f"Warp-Ingest parse error: {exc}") from exc
        finally:
            if temp_pdf_path is not None:
                temp_pdf_path.unlink(missing_ok=True)
        completed_at = datetime.now()

        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    @staticmethod
    def _convert_image_to_pdf(image_path: Path) -> Path:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ProviderConfigError("Pillow is required to run Warp-Ingest on image inputs") from exc

        with Image.open(image_path) as image:
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                rgb_image = Image.new("RGB", rgba.size, "white")
                rgb_image.paste(rgba, mask=rgba.getchannel("A"))
            else:
                rgb_image = image.convert("RGB")

            with NamedTemporaryFile(prefix="parsebench_warp_image_", suffix=".pdf", delete=False) as temp_pdf:
                temp_path = Path(temp_pdf.name)
            rgb_image.save(temp_path, "PDF", resolution=72.0)
            return temp_path

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"WarpIngestProvider only supports PARSE, got {raw_result.product_type}")

        render_pages = _load_markdown_exporter()[1]

        try:
            rendered = render_pages(raw_result.raw_output)
        except Exception as exc:
            raise ProviderPermanentError(f"Warp-Ingest normalize error: {exc}") from exc

        pages = [PageIR(page_index=page_index, markdown=markdown) for page_index, markdown in rendered]
        full_text = "\n\n".join(markdown for _, markdown in rendered)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            markdown=full_text,
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
