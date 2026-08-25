"""Provider for Granite Vision self-hosted servers.

ibm-granite/granite-4.0-3b-vision (~4.25B) is an enterprise-grade document
extraction model supporting task tags:
- "<tables_html>" -- extract tables as HTML
- "<chart2csv>" -- extract chart data as CSV
- "<chart2summary>" -- describe chart content
- Free-form text prompts for general OCR

This provider supports two API formats:
- "openai": OpenAI-compatible vLLM API (for granite_vision_server.py)
- "simple": JSON API with image_base64 (for granite_vision_pipeline_server.py)
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
from extract_bench.schemas.parse_output import ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

# Default model name registered in vLLM (4.0). Override per pipeline via config.
SERVED_MODEL_NAME = "granite-vision"

# Task-specific prompts / tags for Granite Vision
TASK_PROMPTS = {
    "ocr": "Convert the text in this image to markdown.",
    "tables_html": "<tables_html>",
    "tables_json": "<tables_json>",
    "tables_otsl": "<tables_otsl>",
    "chart2csv": "<chart2csv>",
    "chart2code": "<chart2code>",
    "chart2summary": "<chart2summary>",
}


@register_provider("granite_vision")
class GraniteVisionProvider(Provider):
    """
    Provider for Granite Vision self-hosted servers.

    Configuration options:
        - server_url (str, required): server URL
        - api_format (str, default="openai"): "openai" or "simple"
        - task (str, default="ocr"): Task prompt -- "ocr", "tables_html", etc.
        - timeout (int, default=600): Request timeout in seconds
        - dpi (int, default=150): DPI for PDF to image conversion
        - api_key_env (str, default="VLLM_API_KEY"): Env var for API key
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("GRANITE_VISION_SERVER_URL")
        if not server_url:
            raise ProviderConfigError(
                "GraniteVision provider requires 'server_url' in config. "
                "Example: https://<your-granite-vision-deployment>.modal.run"
            )
        self._server_url: str = str(server_url)

        self._api_format = self.base_config.get("api_format", "openai")
        if self._api_format not in ("openai", "simple"):
            raise ProviderConfigError(f"Invalid api_format '{self._api_format}'. Must be 'openai' or 'simple'.")

        # `task` accepts a single string OR a list of strings. With a list,
        # the provider runs each task tag once and concatenates the outputs --
        # that lets one pipeline cover datasets that mix tables, charts, and
        # text without a separate pipeline per task tag.
        task_cfg: str | list[str] = self.base_config.get("task", "ocr")
        if isinstance(task_cfg, str):
            tasks: list[str] = [task_cfg]
        elif isinstance(task_cfg, list) and all(isinstance(t, str) for t in task_cfg):
            tasks = list(task_cfg)
        else:
            raise ProviderConfigError(
                f"task must be a string or list of strings, got {type(task_cfg).__name__}: {task_cfg!r}"
            )
        if not tasks:
            raise ProviderConfigError("task list cannot be empty")
        for t in tasks:
            if t not in TASK_PROMPTS:
                raise ProviderConfigError(f"Invalid task '{t}'. Must be one of: {list(TASK_PROMPTS.keys())}")
        self._tasks: list[str] = tasks
        self._task = task_cfg

        self._timeout = self.base_config.get("timeout", 600)
        self._dpi = self.base_config.get("dpi", 150)
        self._served_model_name: str = str(self.base_config.get("served_model_name", SERVED_MODEL_NAME))

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

    async def _call_openai_api(self, session: aiohttp.ClientSession, image_b64: str, task: str) -> str:
        api_url = f"{self._server_url.rstrip('/')}/v1/chat/completions"

        payload = {
            "model": self._served_model_name,
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
                            "text": TASK_PROMPTS[task],
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 4096,
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

    async def _call_simple_api(self, session: aiohttp.ClientSession, image_b64: str) -> str:
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

            result = await resp.json()
            if result.get("status") == "error":
                raise ProviderPermanentError(result.get("error", "Unknown error from API"))

            content = result.get("markdown", "")
            if not content:
                raise ProviderPermanentError("Empty markdown response from API")
            return str(content)

    async def _run_inference_async(self, image_bytes: bytes) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_bytes).decode()

        async with aiohttp.ClientSession() as session:
            if self._api_format == "simple":
                # Pipeline server does its own per-region task routing.
                markdown = await self._call_simple_api(session, image_b64)
            elif len(self._tasks) == 1:
                markdown = await self._call_openai_api(session, image_b64, self._tasks[0])
            else:
                # Run each task tag in parallel and concatenate. Granite Vision
                # task tags are mutually-exclusive output formats, so a list
                # like ["tables_html", "chart2csv"] means: extract tables AND
                # extract chart data, glue them together.
                results = await asyncio.gather(
                    *[self._call_openai_api(session, image_b64, t) for t in self._tasks],
                    return_exceptions=True,
                )
                parts: list[str] = []
                for r in results:
                    if isinstance(r, Exception):
                        continue
                    if r:
                        parts.append(str(r))
                if not parts:
                    raise ProviderPermanentError(f"All tasks ({self._tasks}) returned empty or errored")
                markdown = "\n\n".join(parts)

        return {
            "markdown": markdown,
            "_config": {
                "server_url": self._server_url,
                "api_format": self._api_format,
                "task": self._task,
                "dpi": self._dpi,
                "served_model_name": self._served_model_name,
            },
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"GraniteVisionProvider only supports PARSE product type, got {request.product_type}"
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
                        "served_model_name": self._served_model_name,
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
    def _convert_csv_to_html(content: str) -> str:
        """Convert CSV blocks to HTML <table> elements.

        Granite Vision's <chart2csv> tag outputs CSV data, often inside a
        ```csv code fence. The chart_data_point and TEDS/GriTS metrics expect
        HTML <table> markup to locate cells by row/column.
        """
        import csv
        import io

        def csv_block_to_html(csv_text: str) -> str | None:
            csv_text = csv_text.strip()
            if not csv_text:
                return None
            try:
                rows = list(csv.reader(io.StringIO(csv_text)))
            except csv.Error:
                return None
            rows = [r for r in rows if any(c.strip() for c in r)]
            if len(rows) < 2 or max(len(r) for r in rows) < 2:
                return None

            def esc(s: str) -> str:
                return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            header, *body = rows
            ncols = max(len(r) for r in rows)
            header = header + [""] * (ncols - len(header))
            parts = ["<table>", "<thead>", "<tr>"]
            parts.extend(f"<th>{esc(c)}</th>" for c in header)
            parts.extend(["</tr>", "</thead>", "<tbody>"])
            for row in body:
                row = row + [""] * (ncols - len(row))
                parts.append("<tr>")
                parts.extend(f"<td>{esc(c)}</td>" for c in row)
                parts.append("</tr>")
            parts.extend(["</tbody>", "</table>"])
            return "".join(parts)

        fenced = re.compile(r"```\s*csv\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

        def _fence_replace(m: re.Match) -> str:
            html = csv_block_to_html(m.group(1))
            return html if html is not None else m.group(0)

        out = fenced.sub(_fence_replace, content)

        if out == content and "<table" not in out.lower() and "|" not in out:
            stripped = out.strip()
            if "," in stripped and stripped.count("\n") >= 1:
                html = csv_block_to_html(stripped)
                if html is not None:
                    out = html

        return out

    @staticmethod
    def _convert_md_tables_to_html(content: str) -> str:
        """Convert markdown pipe tables to HTML <table> elements.

        Granite Vision's <tables_html> tag outputs HTML tables directly,
        but the OCR prompt may produce markdown pipe tables for inline tables.
        This converts them to HTML for GriTS/TEDS metric evaluation.
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

        # Handle trailing table
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
                f"GraniteVisionProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        markdown = raw_result.raw_output.get("markdown", "")
        if markdown:
            # <chart2csv> output is CSV (often fenced) -- convert to HTML
            # tables so chart_data_point / TEDS / GriTS can locate cells.
            markdown = self._convert_csv_to_html(markdown)
            # Convert any markdown pipe tables to HTML so GriTS/TEDS can score them
            markdown = self._convert_md_tables_to_html(markdown)
            markdown = self._sanitize_html_attributes(markdown)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            markdown=markdown,
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
