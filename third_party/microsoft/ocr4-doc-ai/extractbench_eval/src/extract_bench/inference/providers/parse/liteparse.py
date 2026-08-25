"""Provider for LiteParse PARSE."""

import json as _json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)

# Hardcoded path to the workspace `lit` release binary. ParseBench lives at
# <workspace>/ParseBench/src/extract_bench/inference/providers/parse/liteparse.py,
# so parents[6] is the workspace root.
_LIT_BIN = Path(__file__).resolve().parents[6] / "target" / "release" / "lit"

# Kwargs accepted in base_config that map directly to CLI flags.
_CLI_FLAG_MAP: dict[str, str] = {
    "ocr_server_url": "--ocr-server-url",
    "ocr_language": "--ocr-language",
    "tessdata_path": "--tessdata-path",
    "dpi": "--dpi",
    "num_workers": "--num-workers",
    "image_mode": "--image-mode",
}
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.parse_output import PageIR, ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType


@register_provider("liteparse")
class LiteParseProvider(Provider):
    """
    Provider for LiteParse PARSE.

    Uses the local in-process LiteParse Python bindings (no API key required).
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration. Recognized keys:
            - `output_format`: "markdown" | "text" | "json" (default: "markdown")
            - `ocr_enabled`: bool
            - `ocr_server_url`: str
            - `ocr_language`: str
            - `dpi`: float
        """
        super().__init__(provider_name, base_config)
        self._output_format = self.base_config.get("output_format", "markdown")
        self._ocr_enabled = self.base_config.get("ocr_enabled", True)
        self._preserve_small_text = self.base_config.get("preserve_very_small_text", False)
        self._flag_kwargs = {k: v for k, v in self.base_config.items() if k in _CLI_FLAG_MAP and v is not None}

    def _build_cli_args(self, pdf_path: str, fmt: str) -> list[str]:
        args: list[str] = [str(_LIT_BIN), "parse", pdf_path, "--format", fmt, "--quiet"]
        # Ground truth uses plain text (no markdown link syntax), so disable
        # hyperlink extraction for benchmark parity.
        args.append("--no-links")
        if not self._ocr_enabled:
            args.append("--no-ocr")
        if self._preserve_small_text:
            args.append("--preserve-small-text")
        for key, flag in _CLI_FLAG_MAP.items():
            if key in self._flag_kwargs:
                args.extend([flag, str(self._flag_kwargs[key])])
        return args

    def _invoke_cli(self, pdf_path: str, fmt: str) -> str:
        try:
            proc = subprocess.run(
                self._build_cli_args(pdf_path, fmt),
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as e:
            raise ProviderConfigError(
                f"liteparse CLI binary not found at {_LIT_BIN}. "
                f"Build it with `cargo build --release --bin lit` from the workspace root."
            ) from e
        if proc.returncode != 0:
            raise ProviderPermanentError(
                f"lit parse --format {fmt} failed (exit {proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc.stdout

    def _parse(self, pdf_path: str) -> dict[str, Any]:
        json_stdout = self._invoke_cli(pdf_path, "json")
        try:
            parsed = _json.loads(json_stdout)
        except _json.JSONDecodeError as e:
            raise ProviderPermanentError(f"Failed to parse lit JSON output: {e}") from e

        pages_raw = parsed.get("pages", []) or []
        pages = [
            {
                "page_index": (p.get("page", 1) - 1) if isinstance(p.get("page"), int) and p["page"] > 0 else 0,
                "text": p.get("text", "") or "",
            }
            for p in pages_raw
        ]

        if self._output_format == "json":
            full_text = ""
        elif self._output_format == "markdown":
            full_text = self._invoke_cli(pdf_path, "markdown")
        else:
            full_text = "\n".join(p["text"] for p in pages)

        return {
            "pages": pages,
            "num_pages": len(pages),
            "text": full_text,
            "output_format": self._output_format,
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"LiteParseProvider only supports PARSE product type, got {request.product_type}"
            )

        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"File not found: {pdf_path}")

        started_at = datetime.now()
        try:
            raw_output = self._parse(str(pdf_path))
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
        except (ProviderPermanentError, ProviderConfigError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    @staticmethod
    def _convert_md_tables_to_html(content: str) -> str:
        """Convert markdown pipe tables to HTML so GriTS / TRM can score them.

        ParseBench's table metrics only see ``<table>`` blocks; pipe tables in
        markdown are invisible to them. We convert here at the boundary rather
        than changing liteparse's emitter.
        """
        import markdown2

        lines = content.split("\n")
        result_parts: list[str] = []
        table_lines: list[str] = []
        in_table = False

        def _flush() -> None:
            nonlocal table_lines
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

        for line in lines:
            is_table_line = "|" in line and line.strip().startswith("|")
            if is_table_line:
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
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"LiteParseProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        convert_tables = self._output_format == "markdown"

        pages: list[PageIR] = []
        page_texts: list[str] = []
        for page_data in raw_result.raw_output.get("pages", []):
            page_index = page_data.get("page_index", 0)
            text = page_data.get("text", "") or ""
            if convert_tables:
                text = self._convert_md_tables_to_html(text)
            pages.append(PageIR(page_index=page_index, markdown=text))
            page_texts.append(text)

        full_text = raw_result.raw_output.get("text") or "\n\n".join(page_texts)
        if convert_tables and full_text:
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
