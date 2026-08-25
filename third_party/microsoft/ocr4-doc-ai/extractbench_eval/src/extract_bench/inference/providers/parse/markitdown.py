"""Provider for MarkItDown PARSE."""

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

_markitdown = None


@register_provider("markitdown")
class MarkItDownProvider(Provider):
    """Provider for Microsoft MarkItDown (markdown). MIT."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

    def _extract(self, pdf_path: str) -> dict[str, Any]:
        global _markitdown
        try:
            from markitdown import MarkItDown
        except ImportError as e:
            raise ProviderConfigError("markitdown not installed. Run: pip install 'markitdown[pdf]'") from e

        if _markitdown is None:
            _markitdown = MarkItDown()

        try:
            result = _markitdown.convert(pdf_path)
        except Exception as e:
            raise ProviderPermanentError(f"MarkItDown error: {e}") from e

        text = result.text_content or ""
        # MarkItDown returns the whole document; we don't have per-page splits.
        return {"pages": [{"page_index": 0, "text": text}], "num_pages": 1, "text": text}

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"MarkItDownProvider only supports PARSE, got {request.product_type}")

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
            text = self._convert_md_tables_to_html(page_data.get("text", "") or "")
            pages.append(PageIR(page_index=page_data.get("page_index", 0), markdown=text))
            page_texts.append(text)

        full_text = raw_result.raw_output.get("text") or "\n\n".join(page_texts)
        full_text = self._convert_md_tables_to_html(full_text)
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
