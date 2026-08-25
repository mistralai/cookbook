"""Provider for NVIDIA Nemotron-3-Nano-Omni 30B-A3B Reasoning vLLM server.

Nemotron-3-Nano-Omni is a Mamba2-Transformer hybrid MoE (31B total / 3B active)
with native video + audio + image + text input. It is served OpenAI-compatible
via vLLM and used here strictly for PARSE (document -> markdown).

The model defaults to reasoning mode, which is unhelpful for OCR-style tasks
(it produces long internal thoughts before the answer). Thinking is toggled via
`chat_template_kwargs.enable_thinking` per the model card's "Instruct Mode"
best practices.

Model card: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
"""

import asyncio
import base64
import io
import logging
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
from extract_bench.inference.providers.parse._layout_utils import (
    SYSTEM_PROMPT_LAYOUT,
    USER_PROMPT_LAYOUT,
    build_layout_pages,
    items_to_markdown,
    parse_layout_blocks,
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

logger = logging.getLogger(__name__)

DEFAULT_SERVED_MODEL_NAME = "nemotron-omni-30b"

# Byte-identical to the standard parse prompt used by openai.py /
# anthropic.py / gemma4.py — keeps Nemotron head-to-head comparable.
SYSTEM_PROMPT_PARSE = (
    "You are a document parser. Your task is to convert "
    "document images to clean, well-structured markdown."
    "\n\nGuidelines:\n"
    "- Preserve the document structure "
    "(headings, paragraphs, lists, tables)\n"
    "- Convert tables to HTML format "
    "(<table>, <tr>, <th>, <td>)\n"
    "- For existing tables in the document: use colspan "
    "and rowspan attributes to preserve merged cells "
    "and hierarchical headers\n"
    "- For charts/graphs being converted to tables: use "
    "flat combined column headers (e.g., "
    '"Primary 2015" not separate rows) so each data '
    "cell's row contains all its labels\n"
    "- Describe images/figures briefly in square brackets "
    "like [Figure: description]\n"
    "- Preserve any code blocks with appropriate syntax "
    "highlighting\n"
    "- Maintain reading order (left-to-right, "
    "top-to-bottom for Western documents)\n"
    "- Do not add commentary or explanations "
    "- only output the parsed content"
)

USER_PROMPT_PARSE = (
    "Parse this document page and output its content as "
    "clean markdown. Use HTML tables for any tabular "
    "data. For charts/graphs, use flat combined column "
    "headers. Output ONLY the parsed content, "
    "no explanations."
)


@register_provider("nemotron_omni")
class NemotronOmniProvider(Provider):
    """Provider for a self-hosted Nemotron-Omni-30B vLLM server.

    Configuration options:
        - server_url (str, required): vLLM server URL. May also be supplied via
          the ``NEMOTRON_OMNI_SERVER_URL`` environment variable.
        - model (str, default="nemotron-omni-30b"): Served model name
        - prompt_mode (str, default="parse"): "parse" or "layout"
        - timeout (int, default=900): Request timeout in seconds (30B is slow)
        - dpi (int, default=150): DPI for PDF → image conversion
        - max_tokens (int, default=8192): Max tokens per response
        - temperature (float, default=0.2): Sampling temperature
        - top_k (int | None, default=1): Top-k sampling. None to omit.
        - top_p (float | None, default=None): Top-p sampling. None to omit.
        - enable_thinking (bool, default=False): Toggle reasoning mode
        - reasoning_budget (int | None, default=None): Tokens for the thinking
            block (card recommends 16384 in thinking mode)
        - thinking_token_budget (int | None, default=None): Total budget cap
            forwarded via extra_body when thinking is enabled
        - api_key_env (str, default="VLLM_API_KEY"): Env var for API key
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("NEMOTRON_OMNI_SERVER_URL")
        if not server_url:
            raise ProviderConfigError(
                "NemotronOmni provider requires 'server_url' in config or NEMOTRON_OMNI_SERVER_URL in the environment."
            )
        self._server_url: str = str(server_url)

        self._model = self.base_config.get("model", DEFAULT_SERVED_MODEL_NAME)
        self._prompt_mode = self.base_config.get("prompt_mode", "parse")
        self._timeout = self.base_config.get("timeout", 900)
        self._dpi = self.base_config.get("dpi", 150)
        self._max_tokens = self.base_config.get("max_tokens", 8192)
        self._temperature = self.base_config.get("temperature", 0.2)
        self._top_k = self.base_config.get("top_k", 1)
        self._top_p = self.base_config.get("top_p", None)
        self._enable_thinking = self.base_config.get("enable_thinking", False)
        self._reasoning_budget = self.base_config.get("reasoning_budget", None)
        self._thinking_token_budget = self.base_config.get("thinking_token_budget", None)

        if self._prompt_mode == "layout":
            self._system_prompt = SYSTEM_PROMPT_LAYOUT
            self._user_prompt = USER_PROMPT_LAYOUT
        else:
            self._system_prompt = SYSTEM_PROMPT_PARSE
            self._user_prompt = USER_PROMPT_PARSE

        api_key_env = self.base_config.get("api_key_env", "VLLM_API_KEY")
        self._api_key = os.environ.get(api_key_env, "")

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _pdf_to_image_with_size(self, pdf_path: Path) -> tuple[bytes, int, int]:
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path, dpi=self._dpi)
            if not images:
                raise ProviderPermanentError(f"No pages found in PDF: {pdf_path}")
            img = images[0]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue(), img.width, img.height
        except ImportError as e:
            raise ProviderPermanentError("pdf2image is required.") from e
        except ProviderPermanentError:
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Error converting PDF to image: {e}") from e

    def _read_image_with_size(self, file_path: Path) -> tuple[bytes, int, int]:
        from PIL import Image

        try:
            img = Image.open(file_path)
            w, h = img.size
            return file_path.read_bytes(), w, h
        except Exception as e:
            raise ProviderPermanentError(f"Error reading image file: {e}") from e

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    async def _call_api(self, session: aiohttp.ClientSession, image_b64: str) -> str:
        api_url = f"{self._server_url.rstrip('/')}/v1/chat/completions"

        chat_template_kwargs: dict[str, Any] = {"enable_thinking": self._enable_thinking}
        if self._enable_thinking and self._reasoning_budget is not None:
            chat_template_kwargs["reasoning_budget"] = self._reasoning_budget

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": self._user_prompt},
                    ],
                },
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
            # Nemotron-3 vLLM extras; vLLM's OpenAI server accepts top-level
            # extras for top_k/top_p and chat_template_kwargs.
            "chat_template_kwargs": chat_template_kwargs,
        }
        if self._top_k is not None:
            payload["top_k"] = self._top_k
        if self._top_p is not None:
            payload["top_p"] = self._top_p
        if self._enable_thinking and self._thinking_token_budget is not None:
            payload["thinking_token_budget"] = self._thinking_token_budget

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

    # ------------------------------------------------------------------
    # run_inference
    # ------------------------------------------------------------------

    async def _run_inference_async(self, image_bytes: bytes, img_width: int, img_height: int) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_bytes).decode()

        async with aiohttp.ClientSession() as session:
            raw_content = await self._call_api(session, image_b64)

        result: dict[str, Any] = {
            "prompt_mode": self._prompt_mode,
            "_config": {
                "server_url": self._server_url,
                "model": self._model,
                "dpi": self._dpi,
            },
        }
        if self._prompt_mode == "layout":
            items = parse_layout_blocks(raw_content)
            result["raw_content"] = raw_content
            result["layout_items"] = items
            result["image_width"] = img_width
            result["image_height"] = img_height
        else:
            result["markdown"] = raw_content
        return result

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"NemotronOmniProvider only supports PARSE, got {request.product_type}")

        started_at = datetime.now()

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"Source file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            image_bytes, img_w, img_h = self._pdf_to_image_with_size(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
            image_bytes, img_w, img_h = self._read_image_with_size(file_path)
        else:
            raise ProviderPermanentError(
                f"Unsupported file type: {suffix}. Supported: .pdf, .png, .jpg, .jpeg, .webp, .tiff, .bmp"
            )

        try:
            raw_output = asyncio.run(self._run_inference_async(image_bytes, img_w, img_h))
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
                        "model": self._model,
                        "dpi": self._dpi,
                    },
                },
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=latency_ms,
            )

    # ------------------------------------------------------------------
    # HTML helpers (same as Gemma4 — pipe-table → HTML for GriTS/TEDS)
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_html_attributes(text: str) -> str:
        def _quote_attrs(match: re.Match) -> str:
            tag_text = match.group(0)
            return re.sub(r'(\w+)=([^\s"\'<>=]+)', r'\1="\2"', tag_text)

        return re.sub(r"<[^>]+>", _quote_attrs, text)

    @staticmethod
    def _convert_md_tables_to_html(content: str) -> str:
        """Convert markdown pipe tables to HTML <table> elements."""
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

    # ------------------------------------------------------------------
    # normalize
    # ------------------------------------------------------------------

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"NemotronOmniProvider only supports PARSE, got {raw_result.product_type}")

        prompt_mode = raw_result.raw_output.get("prompt_mode", "parse")

        if prompt_mode == "layout":
            layout_items = raw_result.raw_output.get("layout_items", [])
            img_w = raw_result.raw_output.get("image_width", 0)
            img_h = raw_result.raw_output.get("image_height", 0)

            markdown = items_to_markdown(layout_items)
            if markdown:
                markdown = self._sanitize_html_attributes(markdown)

            layout_pages = build_layout_pages(
                items=layout_items,
                image_width=img_w,
                image_height=img_h,
                markdown=markdown,
                page_number=1,
            )

            output = ParseOutput(
                task_type="parse",
                example_id=raw_result.request.example_id,
                pipeline_name=raw_result.pipeline_name,
                pages=[],
                layout_pages=layout_pages,
                markdown=markdown,
            )
        else:
            markdown = raw_result.raw_output.get("markdown", "")
            if markdown:
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
