"""Provider for Gemini 3 Flash vision-based PARSE."""

import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.parse._layout_utils import (
    build_layout_pages,
    items_to_markdown,
    parse_layout_blocks,
    resolve_layout_prompts,
    split_pdf_to_pages,
    swap_gemini_bbox,
)
from extract_bench.inference.providers.parse.google_agentic_vision import (
    GoogleAgenticVisionRunner,
    build_layout_pages_from_agentic_items,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.parse_output import (
    PageIR,
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

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
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

USER_PROMPT = (
    "Parse this document page and output its content as "
    "clean markdown. Use HTML tables for any tabular "
    "data. For charts/graphs, use flat combined column "
    "headers. Output ONLY the parsed content, "
    "no explanations."
)

# Gemini pricing: USD per million tokens (input, output)
# Thinking tokens are billed at the output token rate.
# Source: https://ai.google.dev/gemini-api/docs/pricing (2026-03-25)
_GEMINI_PRICING_PER_M: dict[str, tuple[float, float]] = {
    # model-prefix: (input_per_M, output_per_M)
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.1-pro": (2.00, 12.00),
}

# Gemini context caching pricing: USD per million tokens / per million token-hours.
# Source: https://ai.google.dev/gemini-api/docs/pricing (2026-04-05)
_GEMINI_CONTEXT_CACHE_PRICING_PER_M: dict[str, tuple[float, float]] = {
    # model-prefix: (cache_hit_per_M, storage_per_M_token_hour)
    "gemini-3.7-flash": (0.075, 0.50),
    "gemini-3.6-flash": (0.15, 1.00),
    "gemini-3.5-flash": (0.15, 1.00),
    "gemini-3-flash": (0.05, 1.00),
    "gemini-3.1-flash-lite": (0.025, 1.00),
    "gemini-3.5-flash-lite": (0.03, 1.00),
    "gemini-2.5-flash": (0.03, 1.00),
    "gemini-2.5-flash-lite": (0.01, 1.00),
    "gemini-2.5-pro": (0.125, 4.50),
    "gemini-3.1-pro": (0.20, 4.50),
}


@register_provider("google")
class GoogleProvider(Provider):
    """
    Provider for Google Gemini vision-based document parsing.

    Renders PDF pages to images and uses Gemini's vision
    capabilities to parse document content to markdown.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `model`: Gemini model to use (default: "gemini-3-flash-preview")
            - `dpi`: DPI for PDF to image conversion (default: 150)
            - `max_tokens`: Max tokens per response (default: 8192)
            - `timeout`: Request timeout in seconds (default: 120)
            - `thinking_level`: Thinking level for Gemini 3 models
              ("minimal", "low", "medium", "high"). If not set, uses
              model default.
            - `mode`: "image" (default) to send page screenshots, or "file" to send raw PDF
        """
        super().__init__(provider_name, base_config)

        # Get API key from environment
        self._api_key = os.environ.get("GOOGLE_GEMINI_API_KEY")
        if not self._api_key:
            raise ProviderConfigError("GOOGLE_GEMINI_API_KEY environment variable not set")

        # Configuration
        self._model = self.base_config.get("model", "gemini-3-flash-preview")
        self._dpi = self.base_config.get("dpi", 150)
        self._max_tokens = self.base_config.get("max_tokens", 8192)
        self._timeout = self.base_config.get("timeout", 120)
        self._thinking_level = self.base_config.get("thinking_level", None)
        self._mode = self.base_config.get("mode", "image")  # "image" or "file"
        self._enable_explicit_context_cache = bool(self.base_config.get("enable_explicit_context_cache", False))
        self._context_cache_ttl_seconds = int(self.base_config.get("context_cache_ttl_seconds", 900))
        self._min_cacheable_tokens = int(self.base_config.get("min_cacheable_tokens", 1024))

        if self._mode not in (
            "image",
            "file",
            "parse_with_layout",
            "parse_with_layout_file",
            "parse_with_layout_agentic_vision",
        ):
            raise ProviderConfigError(
                f"Invalid mode '{self._mode}'. "
                "Must be 'image', 'file', 'parse_with_layout', 'parse_with_layout_file', "
                "or 'parse_with_layout_agentic_vision'."
            )

        # Grid the layout-mode bboxes are on: 1000 (the 0-1000 prompt, the
        # default) or None (absolute pixels of the sent image, for models
        # that ground well but re-normalize unreliably).
        self._bbox_scale = self.base_config.get("bbox_scale", 1000)
        self._layout_system_prompt, self._layout_user_prompt = resolve_layout_prompts(
            self._bbox_scale,
            self._mode,
            gemini=True,
            pixel_disallowed_modes=("parse_with_layout_file", "parse_with_layout_agentic_vision"),
        )

        # Initialize Gemini client
        try:
            from google import genai
            from google.genai import types

            self._client = genai.Client(api_key=self._api_key)
            self._types = types
        except ImportError as e:
            raise ProviderConfigError("google-genai package not installed. Run: pip install google-genai") from e

    # Gemini API limits
    MAX_IMAGE_DIMENSION = 8000  # pixels
    MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB (raw bytes, no base64 overhead)

    def _get_pricing(self) -> tuple[float, float]:
        """Return (input_rate, output_rate) in USD per million tokens.

        Uses longest-prefix matching to avoid ambiguity when one model
        prefix is a substring of another (e.g. "gemini-2.5-flash" vs
        "gemini-2.5-flash-lite").
        """
        matches = [(p, r) for p, r in _GEMINI_PRICING_PER_M.items() if self._model.startswith(p)]
        return max(matches, key=lambda x: len(x[0]))[1] if matches else (0.0, 0.0)

    def _get_context_cache_pricing(self) -> tuple[float, float]:
        """Return (cache_hit_rate, storage_rate) in USD per million tokens."""
        matches = [(p, r) for p, r in _GEMINI_CONTEXT_CACHE_PRICING_PER_M.items() if self._model.startswith(p)]
        return max(matches, key=lambda x: len(x[0]))[1] if matches else (0.0, 0.0)

    def _usage_cost_breakdown(self, usage: dict[str, int]) -> dict[str, float]:
        """Compute cost breakdown for one Gemini API call."""
        input_rate, output_rate = self._get_pricing()
        cache_hit_rate, _ = self._get_context_cache_pricing()

        input_tokens = int(usage.get("input_tokens", 0) or 0)
        cached_content_tokens = min(input_tokens, int(usage.get("cached_content_tokens", 0) or 0))
        tool_use_prompt_tokens = int(usage.get("tool_use_prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        thinking_tokens = int(usage.get("thinking_tokens", 0) or 0)

        non_cached_input_tokens = max(input_tokens - cached_content_tokens - tool_use_prompt_tokens, 0)
        input_cost_usd = non_cached_input_tokens * input_rate / 1_000_000
        tool_use_prompt_cost_usd = tool_use_prompt_tokens * input_rate / 1_000_000
        cached_input_cost_usd = cached_content_tokens * cache_hit_rate / 1_000_000
        output_and_thinking_cost_usd = (output_tokens + thinking_tokens) * output_rate / 1_000_000
        cost_usd = input_cost_usd + tool_use_prompt_cost_usd + cached_input_cost_usd + output_and_thinking_cost_usd

        return {
            "input_cost_usd": input_cost_usd,
            "tool_use_prompt_cost_usd": tool_use_prompt_cost_usd,
            "cached_input_cost_usd": cached_input_cost_usd,
            "output_and_thinking_cost_usd": output_and_thinking_cost_usd,
            "cost_usd": cost_usd,
        }

    def _compute_usage_cost_summary(
        self,
        usages: list[dict[str, int]],
        *,
        num_pages: int,
        cache_storage_cost_usd: float = 0.0,
    ) -> dict[str, float | int]:
        """Aggregate token and cost accounting across all Gemini calls for one document."""
        total_input = sum(int(usage.get("input_tokens", 0) or 0) for usage in usages)
        total_tool_use_prompt = sum(int(usage.get("tool_use_prompt_tokens", 0) or 0) for usage in usages)
        total_cached_content = sum(int(usage.get("cached_content_tokens", 0) or 0) for usage in usages)
        total_output = sum(int(usage.get("output_tokens", 0) or 0) for usage in usages)
        total_thinking = sum(int(usage.get("thinking_tokens", 0) or 0) for usage in usages)
        total_tokens = sum(int(usage.get("total_tokens", 0) or 0) for usage in usages)

        per_call_breakdowns = [self._usage_cost_breakdown(usage) for usage in usages]
        input_cost_usd = sum(breakdown["input_cost_usd"] for breakdown in per_call_breakdowns)
        tool_use_prompt_cost_usd = sum(breakdown["tool_use_prompt_cost_usd"] for breakdown in per_call_breakdowns)
        cached_input_cost_usd = sum(breakdown["cached_input_cost_usd"] for breakdown in per_call_breakdowns)
        output_and_thinking_cost_usd = sum(
            breakdown["output_and_thinking_cost_usd"] for breakdown in per_call_breakdowns
        )
        cost_usd = (
            input_cost_usd
            + tool_use_prompt_cost_usd
            + cached_input_cost_usd
            + output_and_thinking_cost_usd
            + cache_storage_cost_usd
        )

        return {
            "input_tokens": total_input,
            "tool_use_prompt_tokens": total_tool_use_prompt,
            "cached_content_tokens": total_cached_content,
            "output_tokens": total_output,
            "thinking_tokens": total_thinking,
            "total_tokens": total_tokens,
            "num_api_calls": len(usages),
            "cost_usd": cost_usd,
            "cost_per_page_usd": cost_usd / num_pages if num_pages > 0 else 0.0,
            "input_cost_usd": input_cost_usd,
            "tool_use_prompt_cost_usd": tool_use_prompt_cost_usd,
            "cached_input_cost_usd": cached_input_cost_usd,
            "output_and_thinking_cost_usd": output_and_thinking_cost_usd,
            "cache_storage_cost_usd": cache_storage_cost_usd,
            "input_tokens_per_page": total_input / num_pages if num_pages > 0 else 0.0,
            "tool_use_prompt_tokens_per_page": total_tool_use_prompt / num_pages if num_pages > 0 else 0.0,
            "cached_content_tokens_per_page": total_cached_content / num_pages if num_pages > 0 else 0.0,
            "output_tokens_per_page": total_output / num_pages if num_pages > 0 else 0.0,
        }

    def _annotate_api_calls_with_costs(self, api_calls: list[dict[str, Any]]) -> None:
        """Populate cost fields for serialized Agentic Vision API calls."""
        for call in api_calls:
            if not isinstance(call, dict):
                continue
            usage = call.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}
            breakdown = self._usage_cost_breakdown(usage)
            call["cost_usd"] = breakdown["cost_usd"]
            call["cost_breakdown_usd"] = {
                "input_cost_usd": breakdown["input_cost_usd"],
                "tool_use_prompt_cost_usd": breakdown["tool_use_prompt_cost_usd"],
                "cached_input_cost_usd": breakdown["cached_input_cost_usd"],
                "output_and_thinking_cost_usd": breakdown["output_and_thinking_cost_usd"],
            }

    def _build_agentic_vision_runner(self, expected_page_calls: int) -> GoogleAgenticVisionRunner:
        """Build the shared Agentic Vision runner for one document."""
        input_rate, _ = self._get_pricing()
        cache_hit_rate, storage_rate = self._get_context_cache_pricing()
        return GoogleAgenticVisionRunner(
            client=self._client,
            types_module=self._types,
            model=self._model,
            max_output_tokens=self._max_tokens,
            thinking_level=self._thinking_level,
            enable_explicit_context_cache=self._enable_explicit_context_cache,
            context_cache_ttl_seconds=self._context_cache_ttl_seconds,
            min_cacheable_tokens=self._min_cacheable_tokens,
            input_cost_per_million=input_rate,
            cache_hit_cost_per_million=cache_hit_rate,
            cache_storage_cost_per_million_token_hour=storage_rate,
            expected_page_calls=expected_page_calls,
        )

    @staticmethod
    def _convert_layout_items_to_agentic_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert x-first layout items to Gemini-native y-first bbox ordering."""
        converted: list[dict[str, Any]] = []
        for item in items:
            bbox = item.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            converted.append(
                {
                    "bbox_2d": [y1, x1, y2, x2],
                    "label": item.get("label", "Text"),
                    "text": item.get("text", ""),
                }
            )
        return converted

    @staticmethod
    def _extract_usage(response) -> dict[str, int]:  # type: ignore[no-untyped-def]
        """Extract token counts from a Gemini API response."""
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
        input_tok = getattr(meta, "prompt_token_count", 0) or 0
        output_tok = getattr(meta, "candidates_token_count", 0) or 0
        thinking_tok = getattr(meta, "thoughts_token_count", 0) or 0
        total_tok = getattr(meta, "total_token_count", 0) or 0
        return {
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "thinking_tokens": thinking_tok,
            "total_tokens": total_tok,
        }

    def _prepare_image_for_api(self, image: Image.Image) -> Image.Image:
        """
        Resize image if it exceeds Gemini API dimension limits.

        :param image: PIL Image to prepare
        :return: Resized image if needed, otherwise original
        """
        width, height = image.size
        max_dim = max(width, height)

        if max_dim <= self.MAX_IMAGE_DIMENSION:
            return image

        # Calculate scale factor to fit within limits
        scale = self.MAX_IMAGE_DIMENSION / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """
        Convert PIL Image to JPEG bytes, respecting Gemini API limits.

        Handles:
        - Images with dimensions exceeding 8000 pixels (resizes proportionally)
        - Images exceeding 20MB (reduces quality iteratively)
        """
        # Resize if dimensions exceed limit
        image = self._prepare_image_for_api(image)

        # Convert to RGB if necessary (e.g., RGBA images)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        # Try encoding with decreasing quality until under size limit
        quality = 85
        min_quality = 20

        while quality >= min_quality:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            data = buffer.getvalue()

            if len(data) <= self.MAX_IMAGE_SIZE_BYTES:
                return data

            quality -= 10

        # If still too large after quality reduction, resize the image
        while True:
            width, height = image.size
            new_width = int(width * 0.8)
            new_height = int(height * 0.8)

            if new_width < 100 or new_height < 100:
                # Give up - image is too complex to fit in limits
                break

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=min_quality)
            data = buffer.getvalue()

            if len(data) <= self.MAX_IMAGE_SIZE_BYTES:
                return data

        # Final fallback - return what we have
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=min_quality)
        return buffer.getvalue()

    def _pdf_to_images(self, pdf_path: str) -> list[Image.Image]:
        """
        Convert PDF pages to images.

        :param pdf_path: Path to the PDF file
        :return: List of PIL Images, one per page
        """
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ProviderConfigError("pdf2image package not installed. Run: pip install pdf2image") from e

        try:
            images = convert_from_path(pdf_path, dpi=self._dpi)
            return images
        except Exception as e:
            raise ProviderPermanentError(f"Failed to convert PDF to images: {e}") from e

    @staticmethod
    def _extract_text(response) -> str | None:  # type: ignore[no-untyped-def]
        """Extract text from a Gemini response, or None if empty."""
        if not response.candidates:
            return None
        content = response.candidates[0].content
        if content is None or content.parts is None:
            return None
        text = content.parts[0].text
        return text if text else None

    @staticmethod
    def _failure_reason(response) -> str:  # type: ignore[no-untyped-def]
        """Return a human-readable reason why a Gemini response had no text."""
        if not response.candidates:
            block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
            if block_reason:
                return f"no candidates (prompt blocked: {block_reason})"
            return "no candidates returned"
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason:
            return f"finish_reason={finish_reason}"
        content = getattr(candidate, "content", None)
        if content is None:
            return "candidate has no content"
        if content.parts is None:
            return "candidate content has no parts"
        return "empty text in response"

    def _parse_image(self, image: Image.Image) -> tuple[str, dict[str, int]]:
        """
        Send image to Gemini Flash and get markdown response.

        Retries once if the response is empty.

        :param image: PIL Image to parse
        :return: Tuple of (markdown content, usage dict)
        """
        img_bytes = self._image_to_bytes(image)
        types = self._types

        try:
            image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            text_part = types.Part.from_text(text=USER_PROMPT)

            gen_config = types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=self._max_tokens,
                system_instruction=SYSTEM_PROMPT,
            )
            if self._thinking_level is not None:
                gen_config.thinking_config = types.ThinkingConfig(
                    thinking_level=self._thinking_level,
                )

            contents = [
                types.Content(
                    role="user",
                    parts=[image_part, text_part],
                )
            ]

            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=gen_config,
            )
            usage = self._extract_usage(response)
            text = self._extract_text(response)

            if text is None:
                reason1 = self._failure_reason(response)
                # Single retry on empty response
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=gen_config,
                )
                usage = self._extract_usage(response)
                text = self._extract_text(response)

            if text is None:
                reason2 = self._failure_reason(response)
                return f"[No output after 2 attempts: 1st={reason1}, 2nd={reason2}]", usage
            return text, usage

        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "connection", "network"]):
                raise ProviderTransientError(f"Transient error calling Gemini API: {e}") from e
            if any(kw in error_str for kw in ["rate_limit", "rate limit", "429", "resource_exhausted"]):
                raise ProviderTransientError(f"Rate limited: {e}") from e
            raise ProviderPermanentError(f"Error calling Gemini API: {e}") from e

    def _parse_image_with_layout(self, image: Image.Image) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
        """Send image to Gemini with layout prompt and get annotated response.

        :param image: PIL Image to parse
        :return: Tuple of (parsed layout items, raw content, usage dict)
        """
        img_bytes = self._image_to_bytes(image)
        types = self._types

        try:
            image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            text_part = types.Part.from_text(text=self._layout_user_prompt)

            gen_config = types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=self._max_tokens,
                system_instruction=self._layout_system_prompt,
            )
            if self._thinking_level is not None:
                gen_config.thinking_config = types.ThinkingConfig(
                    thinking_level=self._thinking_level,
                )

            contents = [
                types.Content(
                    role="user",
                    parts=[image_part, text_part],
                )
            ]

            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=gen_config,
            )
            usage = self._extract_usage(response)
            text = self._extract_text(response)

            if text is None:
                reason1 = self._failure_reason(response)
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=gen_config,
                )
                usage = self._extract_usage(response)
                text = self._extract_text(response)

            if text is None:
                reason2 = self._failure_reason(response)
                return [], f"[No output after 2 attempts: 1st={reason1}, 2nd={reason2}]", usage

            items = swap_gemini_bbox(parse_layout_blocks(text))
            return items, text, usage

        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "connection", "network"]):
                raise ProviderTransientError(f"Transient error calling Gemini API: {e}") from e
            if any(kw in error_str for kw in ["rate_limit", "rate limit", "429", "resource_exhausted"]):
                raise ProviderTransientError(f"Rate limited: {e}") from e
            raise ProviderPermanentError(f"Error calling Gemini API: {e}") from e

    def _parse_pdf_file(self, pdf_path: str) -> tuple[str, dict[str, int]]:
        """
        Send raw PDF file to Gemini using inline data.

        Uses Gemini's document understanding capability to process
        the PDF directly without converting to images. Retries once
        if the response is empty.

        :param pdf_path: Path to the PDF file
        :return: Tuple of (markdown content, usage dict)
        """
        types = self._types

        try:
            # Read PDF file
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()

            # Send PDF as inline data
            pdf_part = types.Part.from_bytes(data=pdf_data, mime_type="application/pdf")
            text_part = types.Part.from_text(text=USER_PROMPT)

            gen_config = types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=self._max_tokens,
                system_instruction=SYSTEM_PROMPT,
            )
            if self._thinking_level is not None:
                gen_config.thinking_config = types.ThinkingConfig(
                    thinking_level=self._thinking_level,
                )

            contents = [
                types.Content(
                    role="user",
                    parts=[pdf_part, text_part],
                )
            ]

            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=gen_config,
            )
            usage = self._extract_usage(response)
            text = self._extract_text(response)

            if text is None:
                reason1 = self._failure_reason(response)
                # Single retry on empty response
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=gen_config,
                )
                usage = self._extract_usage(response)
                text = self._extract_text(response)

            if text is None:
                reason2 = self._failure_reason(response)
                return f"[No output after 2 attempts: 1st={reason1}, 2nd={reason2}]", usage
            return text, usage

        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "connection", "network"]):
                raise ProviderTransientError(f"Transient error calling Gemini API: {e}") from e
            if any(kw in error_str for kw in ["rate_limit", "rate limit", "429", "resource_exhausted"]):
                raise ProviderTransientError(f"Rate limited: {e}") from e
            raise ProviderPermanentError(f"Error calling Gemini API: {e}") from e

    def _parse_pdf_page_with_layout(self, pdf_bytes: bytes) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
        """Send a single-page PDF to Gemini with layout prompt.

        :param pdf_bytes: Raw bytes of a single-page PDF
        :return: Tuple of (parsed layout items, raw content, usage dict)
        """
        types = self._types

        try:
            pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            text_part = types.Part.from_text(text=self._layout_user_prompt)

            gen_config = types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=self._max_tokens,
                system_instruction=self._layout_system_prompt,
            )
            if self._thinking_level is not None:
                gen_config.thinking_config = types.ThinkingConfig(
                    thinking_level=self._thinking_level,
                )

            contents = [
                types.Content(
                    role="user",
                    parts=[pdf_part, text_part],
                )
            ]

            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=gen_config,
            )
            usage = self._extract_usage(response)
            text = self._extract_text(response)

            if text is None:
                reason1 = self._failure_reason(response)
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=gen_config,
                )
                usage = self._extract_usage(response)
                text = self._extract_text(response)

            if text is None:
                reason2 = self._failure_reason(response)
                text = f"[No output after 2 attempts: 1st={reason1}, 2nd={reason2}]"

            items = swap_gemini_bbox(parse_layout_blocks(text))
            return items, text, usage

        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "connection", "network"]):
                raise ProviderTransientError(f"Transient error calling Gemini API: {e}") from e
            if any(kw in error_str for kw in ["rate_limit", "rate limit", "429", "resource_exhausted"]):
                raise ProviderTransientError(f"Rate limited: {e}") from e
            raise ProviderPermanentError(f"Error calling Gemini API: {e}") from e

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        :param pipeline: Pipeline specification
        :param request: Inference request
        :return: Raw inference result
        """
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"GoogleProvider only supports PARSE product type, got {request.product_type}")

        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")

        # Check file extension
        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg"}
        if source_path.suffix.lower() not in supported_extensions:
            raise ProviderPermanentError(f"GoogleProvider supports {supported_extensions}, got {source_path.suffix}")

        started_at = datetime.now()

        try:
            page_usages: list[dict[str, int]] = []

            if self._mode == "file":
                if source_path.suffix.lower() == ".pdf":
                    # File mode: send raw PDF to API
                    markdown, usage = self._parse_pdf_file(str(source_path))
                    page_usages.append(usage)
                    # In file mode, we get one response for the entire document
                    # We don't have page-level info, so we treat it as a single "page"
                    pages = [
                        {
                            "page_index": 0,
                            "markdown": markdown,
                            "width": None,
                            "height": None,
                        }
                    ]
                    num_pages = 1  # We don't know actual page count in file mode
                else:
                    # Non-PDF: fall back to image-based parsing
                    image = Image.open(source_path)
                    markdown, usage = self._parse_image(image)
                    page_usages.append(usage)
                    pages = [
                        {
                            "page_index": 0,
                            "markdown": markdown,
                            "width": image.width,
                            "height": image.height,
                        }
                    ]
                    num_pages = 1
            elif self._mode == "parse_with_layout_file":
                if source_path.suffix.lower() == ".pdf":
                    # Split PDF into single-page PDFs, send each with layout prompt
                    layout_pdf_pages = split_pdf_to_pages(str(source_path))
                    pages = []
                    for page_index, (pdf_bytes, w, h) in enumerate(layout_pdf_pages):
                        items, raw_content, usage = self._parse_pdf_page_with_layout(pdf_bytes)
                        page_usages.append(usage)
                        pages.append(
                            {
                                "page_index": page_index,
                                "items": items,
                                "raw_content": raw_content,
                                "width": w,
                                "height": h,
                            }
                        )
                    num_pages = len(layout_pdf_pages)
                else:
                    # Non-PDF: fall back to image-based layout parsing
                    image = Image.open(source_path)
                    items, raw_content, usage = self._parse_image_with_layout(image)
                    page_usages.append(usage)
                    pages = [
                        {
                            "page_index": 0,
                            "items": items,
                            "raw_content": raw_content,
                            "width": image.width,
                            "height": image.height,
                        }
                    ]
                    num_pages = 1
            elif self._mode == "parse_with_layout_agentic_vision":
                if source_path.suffix.lower() == ".pdf":
                    images = self._pdf_to_images(str(source_path))
                else:
                    images = [Image.open(source_path)]

                num_pages = len(images)
                runner = self._build_agentic_vision_runner(expected_page_calls=num_pages)
                pages = []

                for page_index, image in enumerate(images):  # type: ignore[assignment]
                    img_bytes = self._image_to_bytes(image)

                    try:
                        page_result = runner.parse_page(
                            page_index=page_index,
                            image=image,
                            image_bytes=img_bytes,
                            image_mime_type="image/jpeg",
                        )
                        self._annotate_api_calls_with_costs(page_result.api_calls)
                        page_usages.extend(
                            call.get("usage", {}) for call in page_result.api_calls if isinstance(call, dict)
                        )
                        pages.append(
                            {
                                "page_index": page_result.page_index,
                                "items": page_result.items,
                                "markdown": page_result.markdown,
                                "raw_content": page_result.raw_content,
                                "width": page_result.width,
                                "height": page_result.height,
                                "image_mime_type": page_result.image_mime_type,
                                "thought_summaries": page_result.thought_summaries,
                                "thought_signatures": page_result.thought_signatures,
                                "generated_code": page_result.generated_code,
                                "code_execution_results": page_result.code_execution_results,
                                "api_calls": page_result.api_calls,
                            }
                        )
                    except (ProviderPermanentError, ProviderTransientError) as exc:
                        debug_payload = exc.debug_payload if isinstance(exc.debug_payload, dict) else None
                        if debug_payload is not None:
                            maybe_calls = debug_payload.get("api_calls", [])
                            if isinstance(maybe_calls, list):
                                failed_api_calls = [call for call in maybe_calls if isinstance(call, dict)]
                                self._annotate_api_calls_with_costs(failed_api_calls)
                                page_usages.extend(call.get("usage", {}) for call in failed_api_calls)
                                debug_payload["api_calls"] = failed_api_calls
                        raise
            else:
                # Image mode (both "image" and "parse_with_layout"):
                # convert PDF to images and process each page
                if source_path.suffix.lower() == ".pdf":
                    images = self._pdf_to_images(str(source_path))
                else:
                    images = [Image.open(source_path)]

                pages = []
                for page_index, image in enumerate(images):  # type: ignore[assignment]
                    if self._mode == "parse_with_layout":
                        items, raw_content, usage = self._parse_image_with_layout(image)
                        page_usages.append(usage)
                        pages.append(
                            {
                                "page_index": page_index,
                                "items": items,
                                "raw_content": raw_content,
                                "width": image.width,
                                "height": image.height,
                            }
                        )
                    else:
                        markdown, usage = self._parse_image(image)
                        page_usages.append(usage)
                        pages.append(
                            {
                                "page_index": page_index,
                                "markdown": markdown,
                                "width": image.width,
                                "height": image.height,
                            }
                        )
                num_pages = len(images)

            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

            config_info: dict[str, Any] = {
                "dpi": self._dpi,
                "max_tokens": self._max_tokens,
                "mode": self._mode,
            }
            if self._thinking_level is not None:
                config_info["thinking_level"] = self._thinking_level
            if self._mode == "parse_with_layout_agentic_vision":
                config_info["enable_explicit_context_cache"] = self._enable_explicit_context_cache
                config_info["context_cache_ttl_seconds"] = self._context_cache_ttl_seconds
                config_info["min_cacheable_tokens"] = self._min_cacheable_tokens

            if self._mode == "parse_with_layout_agentic_vision":
                cache_info = runner.cache_info if "runner" in locals() else None
                cache_storage_cost_usd = cache_info.storage_cost_usd if cache_info is not None else 0.0
                usage_summary = self._compute_usage_cost_summary(
                    page_usages,
                    num_pages=num_pages,
                    cache_storage_cost_usd=cache_storage_cost_usd,
                )
            else:
                total_input = sum(u["input_tokens"] for u in page_usages)
                total_output = sum(u["output_tokens"] for u in page_usages)
                total_thinking = sum(u["thinking_tokens"] for u in page_usages)
                total_all = sum(u["total_tokens"] for u in page_usages)

                input_rate, output_rate = self._get_pricing()
                cost = (total_input * input_rate + (total_output + total_thinking) * output_rate) / 1_000_000
                usage_summary = {
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "thinking_tokens": total_thinking,
                    "total_tokens": total_all,
                    "cost_usd": cost,
                    "cost_per_page_usd": cost / num_pages if num_pages > 0 else 0.0,
                    "input_tokens_per_page": total_input / num_pages if num_pages > 0 else 0.0,
                    "output_tokens_per_page": total_output / num_pages if num_pages > 0 else 0.0,
                }

            raw_output = {
                "pages": pages,
                "num_pages": num_pages,
                "model": self._model,
                "mode": self._mode,
                "bbox_scale": self._bbox_scale,
                "config": config_info,
                **usage_summary,
            }
            if self._mode == "parse_with_layout_agentic_vision":
                cache_info = runner.cache_info if "runner" in locals() else None
                raw_output["cache_error"] = runner.cache_error if "runner" in locals() else None
                raw_output["explicit_context_cache"] = (
                    {
                        "name": cache_info.name,
                        "display_name": cache_info.display_name,
                        "token_count": cache_info.token_count,
                        "ttl_seconds": cache_info.ttl_seconds,
                        "storage_cost_usd": cache_info.storage_cost_usd,
                        "created": cache_info.created,
                    }
                    if cache_info is not None
                    else None
                )

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

        except (ProviderPermanentError, ProviderTransientError, ProviderConfigError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce ParseOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        """
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"GoogleProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        mode = raw_result.raw_output.get("mode", "image")

        # Build page-level output
        pages: list[PageIR] = []
        page_markdowns: list[str] = []
        layout_pages: list[ParseLayoutPageIR] = []

        for page_data in raw_result.raw_output.get("pages", []):
            page_index = page_data.get("page_index", 0)

            if mode in ("parse_with_layout", "parse_with_layout_file"):
                items = page_data.get("items", [])
                image_width = page_data.get("width", 0)
                image_height = page_data.get("height", 0)
                markdown = items_to_markdown(items)
                layout_pages.extend(
                    build_layout_pages(
                        items,
                        image_width,
                        image_height,
                        markdown,
                        page_number=page_index + 1,
                        bbox_scale=raw_result.raw_output.get("bbox_scale", 1000),
                    )
                )
            elif mode == "parse_with_layout_agentic_vision":
                items = page_data.get("items", [])
                image_width = page_data.get("width", 0)
                image_height = page_data.get("height", 0)
                markdown, page_layout_pages = build_layout_pages_from_agentic_items(
                    items,
                    image_width,
                    image_height,
                    page_number=page_index + 1,
                )
                layout_pages.extend(page_layout_pages)
            else:
                markdown = page_data.get("markdown", "")

            pages.append(PageIR(page_index=page_index, markdown=markdown))
            page_markdowns.append(markdown)

        # Sort by page index and concatenate
        pages.sort(key=lambda p: p.page_index)
        full_markdown = "\n\n".join(page_markdowns)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            markdown=full_markdown,
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
