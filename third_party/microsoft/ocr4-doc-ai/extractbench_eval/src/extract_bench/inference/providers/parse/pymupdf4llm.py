"""Provider for PyMuPDF4LLM PARSE."""

from datetime import datetime
from pathlib import Path
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


@register_provider("pymupdf4llm")
class PyMuPDF4LLMProvider(Provider):
    """Provider for PyMuPDF4LLM (markdown). AGPL — runtime dep only."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

    def _extract(self, pdf_path: str) -> dict[str, Any]:
        try:
            import pymupdf4llm
        except ImportError as e:
            raise ProviderConfigError("pymupdf4llm not installed. Run: pip install pymupdf4llm") from e

        try:
            page_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, show_progress=False, use_ocr=False)
        except Exception as e:
            raise ProviderPermanentError(f"PyMuPDF4LLM error: {e}") from e

        pages = []
        for i, chunk in enumerate(page_chunks):
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            pages.append({"page_index": i, "text": text})

        return {"pages": pages, "num_pages": len(pages)}

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"PyMuPDF4LLMProvider only supports PARSE, got {request.product_type}")

        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"File not found: {pdf_path}")

        started_at = datetime.now()
        try:
            raw_output = self._extract(str(pdf_path))
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
        except (ProviderPermanentError, ProviderConfigError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error: {e}") from e

    @staticmethod
    def _convert_md_tables_to_html(content: str) -> str:
        import markdown2

        lines = content.split("\n")
        result_parts: list[str] = []
        table_lines: list[str] = []
        in_table = False

        def _flush() -> None:
            nonlocal table_lines
            if len(table_lines) >= 2:
                html = markdown2.markdown("\n".join(table_lines), extras=["tables"]).strip()
                if "<table>" in html.lower():
                    result_parts.append(html)
                else:
                    result_parts.extend(table_lines)
            else:
                result_parts.extend(table_lines)
            table_lines = []

        for line in lines:
            if "|" in line and line.strip().startswith("|"):
                in_table = True
                table_lines.append(line)
            else:
                if in_table:
                    _flush()
                    in_table = False
                result_parts.append(line)
        if in_table:
            _flush()
        return "\n".join(result_parts)

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        pages: list[PageIR] = []
        page_texts: list[str] = []
        for page_data in raw_result.raw_output.get("pages", []):
            page_index = page_data.get("page_index", 0)
            text = self._convert_md_tables_to_html(page_data.get("text", "") or "")
            pages.append(PageIR(page_index=page_index, markdown=text))
            page_texts.append(text)

        full_text = "\n\n".join(page_texts)
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
