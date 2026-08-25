"""Provider for OpenAI vision-based PARSE."""

import base64
import io
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
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

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


# OpenAI pricing: USD per million tokens (input, output)
# Reasoning tokens billed at output rate.
# Source: https://developers.openai.com/api/docs/pricing (2026-03-25)
_OPENAI_PRICING_PER_M: dict[str, tuple[float, float]] = {
    # model-prefix: (input_per_M, output_per_M)
    "gpt-5-mini": (0.75, 4.50),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6-luna": (1.00, 6.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
}


@register_provider("openai")
class OpenAIProvider(Provider):
    """
    Provider for OpenAI GPT-5 Mini vision-based document parsing.

    Renders PDF pages to images and uses GPT-5 Mini's vision
    capabilities to parse document content to markdown.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `model`: OpenAI model to use (default: "gpt-5-mini")
            - `dpi`: DPI for PDF to image conversion (default: 150)
            - `max_tokens`: Max tokens per response (default: 8192)
            - `timeout`: Request timeout in seconds (default: 120)
            - `reasoning_effort`: Reasoning effort for OpenAI reasoning models
              ("minimal", "low", "medium", "high"). If not set, uses model default.
            - `mode`: "image" (default) to send page screenshots, or "file" to send raw PDF
        """
        super().__init__(provider_name, base_config)

        # Get API key from environment
        self._api_key = os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ProviderConfigError("OPENAI_API_KEY environment variable not set")

        # Configuration
        self._model = self.base_config.get("model", "gpt-5-mini")
        self._dpi = self.base_config.get("dpi", 150)
        self._max_tokens = self.base_config.get("max_tokens", 8192)
        self._timeout = self.base_config.get("timeout", 120)
        self._reasoning_effort = self.base_config.get("reasoning_effort", None)
        self._mode = self.base_config.get("mode", "image")  # "image", "file", or "parse_with_layout"

        if self._mode not in ("image", "file", "parse_with_layout", "parse_with_layout_file"):
            raise ProviderConfigError(
                f"Invalid mode '{self._mode}'. "
                "Must be 'image', 'file', 'parse_with_layout', or 'parse_with_layout_file'."
            )

        # Grid the layout-mode bboxes are on: 1000 (the 0-1000 prompt, the
        # default) or None (absolute pixels of the sent image, for models
        # that ground well but re-normalize unreliably).
        self._bbox_scale = self.base_config.get("bbox_scale", 1000)
        self._layout_system_prompt, self._layout_user_prompt = resolve_layout_prompts(self._bbox_scale, self._mode)

        # Initialize OpenAI client
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        except ImportError as e:
            raise ProviderConfigError("openai package not installed. Run: pip install openai") from e

    # OpenAI API limits (conservative values that work across models)
    MAX_IMAGE_DIMENSION = 8000  # pixels
    # API limit is 20MB for base64 data; base64 adds ~33% overhead, so raw limit is 20MB * 3/4
    MAX_IMAGE_SIZE_BYTES = int(20 * 1024 * 1024 * 3 / 4)  # ~15 MB raw -> ~20 MB base64

    def _get_pricing(self) -> tuple[float, float]:
        """Return (input_rate, output_rate) in USD per million tokens.

        Uses longest-prefix matching to avoid ambiguity when one model
        prefix is a substring of another.
        """
        matches = [(p, r) for p, r in _OPENAI_PRICING_PER_M.items() if self._model.startswith(p)]
        return max(matches, key=lambda x: len(x[0]))[1] if matches else (0.0, 0.0)

    @staticmethod
    def _extract_usage(response) -> dict[str, int]:  # type: ignore[no-untyped-def]
        """Extract token counts from an OpenAI API response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
        input_tok = getattr(usage, "prompt_tokens", 0) or 0
        output_tok = getattr(usage, "completion_tokens", 0) or 0
        total_tok = getattr(usage, "total_tokens", 0) or 0
        # Reasoning tokens (o-series models)
        details = getattr(usage, "completion_tokens_details", None)
        thinking_tok = getattr(details, "reasoning_tokens", 0) or 0 if details else 0
        return {
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "thinking_tokens": thinking_tok,
            "total_tokens": total_tok,
        }

    def _prepare_image_for_api(self, image: Image.Image) -> Image.Image:
        """
        Resize image if it exceeds OpenAI API dimension limits.

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

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string, respecting OpenAI API limits.

        Handles:
        - Images with dimensions exceeding limits (resizes proportionally)
        - Images exceeding size limit after encoding (reduces quality iteratively)
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
            buffer.seek(0)
            data = buffer.getvalue()

            if len(data) <= self.MAX_IMAGE_SIZE_BYTES:
                return base64.standard_b64encode(data).decode("utf-8")

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
            buffer.seek(0)
            data = buffer.getvalue()

            if len(data) <= self.MAX_IMAGE_SIZE_BYTES:
                return base64.standard_b64encode(data).decode("utf-8")

        # Final fallback - return what we have
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=min_quality)
        buffer.seek(0)
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

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

    def _parse_image(self, image: Image.Image) -> tuple[str, dict[str, int]]:
        """
        Send image to GPT-5 Mini and get markdown response.

        :param image: PIL Image to parse
        :return: Tuple of (markdown content, usage dict)
        """
        img_base64 = self._image_to_base64(image)

        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_completion_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": USER_PROMPT,
                            },
                        ],
                    },
                ],
            }
            if self._reasoning_effort is not None:
                kwargs["reasoning_effort"] = self._reasoning_effort
            response = self._client.chat.completions.create(**kwargs)

            usage = self._extract_usage(response)

            # Extract text from response
            content = response.choices[0].message.content if response.choices else ""
            return (content or ""), usage

        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "connection", "network"]):
                raise ProviderTransientError(f"Transient error calling OpenAI API: {e}") from e
            if any(kw in error_str for kw in ["rate_limit", "rate limit", "429"]):
                raise ProviderTransientError(f"Rate limited: {e}") from e
            raise ProviderPermanentError(f"Error calling OpenAI API: {e}") from e

    def _parse_image_with_layout(self, image: Image.Image) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
        """Send image to OpenAI with layout prompt and get annotated response.

        :param image: PIL Image to parse
        :return: Tuple of (parsed layout items, raw content, usage dict)
        """
        img_base64 = self._image_to_base64(image)

        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_completion_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": self._layout_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": self._layout_user_prompt,
                            },
                        ],
                    },
                ],
            }
            if self._reasoning_effort is not None:
                kwargs["reasoning_effort"] = self._reasoning_effort
            response = self._client.chat.completions.create(**kwargs)

            usage = self._extract_usage(response)
            content = response.choices[0].message.content if response.choices else ""
            text = content or ""

            items = parse_layout_blocks(text)
            return items, text, usage

        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "connection", "network"]):
                raise ProviderTransientError(f"Transient error calling OpenAI API: {e}") from e
            if any(kw in error_str for kw in ["rate_limit", "rate limit", "429"]):
                raise ProviderTransientError(f"Rate limited: {e}") from e
            raise ProviderPermanentError(f"Error calling OpenAI API: {e}") from e

    def _parse_pdf_file(self, pdf_path: str) -> tuple[str, dict[str, int]]:
        """
        Send raw PDF file to OpenAI using base64 encoding.

        Uses OpenAI's file input support to send the PDF directly
        without converting to images.

        :param pdf_path: Path to the PDF file
        :return: Tuple of (markdown content, usage dict)
        """
        try:
            # Read PDF file and encode as base64
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()

            pdf_base64 = base64.standard_b64encode(pdf_data).decode("utf-8")

            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_completion_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "file",
                                "file": {
                                    "filename": Path(pdf_path).name,
                                    "file_data": f"data:application/pdf;base64,{pdf_base64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": USER_PROMPT,
                            },
                        ],
                    },
                ],
            }
            if self._reasoning_effort is not None:
                kwargs["reasoning_effort"] = self._reasoning_effort
            response = self._client.chat.completions.create(**kwargs)

            usage = self._extract_usage(response)

            # Extract text from response
            content = response.choices[0].message.content if response.choices else ""
            return (content or ""), usage

        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "connection", "network"]):
                raise ProviderTransientError(f"Transient error calling OpenAI API: {e}") from e
            if any(kw in error_str for kw in ["rate_limit", "rate limit", "429"]):
                raise ProviderTransientError(f"Rate limited: {e}") from e
            raise ProviderPermanentError(f"Error calling OpenAI API: {e}") from e

    def _parse_pdf_page_with_layout(self, pdf_bytes: bytes) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
        """Send a single-page PDF to OpenAI with layout prompt.

        :param pdf_bytes: Raw bytes of a single-page PDF
        :return: Tuple of (parsed layout items, raw content, usage dict)
        """
        pdf_base64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_completion_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": self._layout_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "file",
                                "file": {
                                    "filename": "page.pdf",
                                    "file_data": f"data:application/pdf;base64,{pdf_base64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": self._layout_user_prompt,
                            },
                        ],
                    },
                ],
            }
            if self._reasoning_effort is not None:
                kwargs["reasoning_effort"] = self._reasoning_effort
            response = self._client.chat.completions.create(**kwargs)

            usage = self._extract_usage(response)
            content = response.choices[0].message.content if response.choices else ""
            text = content or ""

            items = parse_layout_blocks(text)
            return items, text, usage

        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "connection", "network"]):
                raise ProviderTransientError(f"Transient error calling OpenAI API: {e}") from e
            if any(kw in error_str for kw in ["rate_limit", "rate limit", "429"]):
                raise ProviderTransientError(f"Rate limited: {e}") from e
            # GPT-5.6 intermittently returns a 401 "insufficient permissions"
            # that clears on retry; treat it as transient so the runner retries.
            is_gpt56_401_blip = (
                self._model.startswith("gpt-5.6-") and "insufficient permissions for this operation" in error_str
            )
            if is_gpt56_401_blip:
                raise ProviderTransientError(f"Transient OpenAI 401 (retryable): {e}") from e
            raise ProviderPermanentError(f"Error calling OpenAI API: {e}") from e

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        :param pipeline: Pipeline specification
        :param request: Inference request
        :return: Raw inference result
        """
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"OpenAIProvider only supports PARSE product type, got {request.product_type}")

        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")

        # Check file extension
        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg"}
        if source_path.suffix.lower() not in supported_extensions:
            raise ProviderPermanentError(f"OpenAIProvider supports {supported_extensions}, got {source_path.suffix}")

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
                    pdf_pages = split_pdf_to_pages(str(source_path))
                    pages = []
                    for page_index, (pdf_bytes, w, h) in enumerate(pdf_pages):
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
                    num_pages = len(pdf_pages)
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
            else:
                # Image mode (both "image" and "parse_with_layout"):
                # convert PDF to images and process each page
                if source_path.suffix.lower() == ".pdf":
                    images = self._pdf_to_images(str(source_path))
                else:
                    images = [Image.open(source_path)]

                # Parse each page
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

            # Aggregate token usage across pages
            total_input = sum(u["input_tokens"] for u in page_usages)
            total_output = sum(u["output_tokens"] for u in page_usages)
            total_thinking = sum(u["thinking_tokens"] for u in page_usages)
            total_all = sum(u["total_tokens"] for u in page_usages)

            # Compute cost
            input_rate, output_rate = self._get_pricing()
            cost = (total_input * input_rate + (total_output + total_thinking) * output_rate) / 1_000_000

            config_info: dict[str, Any] = {
                "dpi": self._dpi,
                "max_tokens": self._max_tokens,
                "mode": self._mode,
            }
            if self._reasoning_effort is not None:
                config_info["reasoning_effort"] = self._reasoning_effort

            raw_output = {
                "pages": pages,
                "num_pages": num_pages,
                "model": self._model,
                "mode": self._mode,
                "bbox_scale": self._bbox_scale,
                "config": config_info,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "thinking_tokens": total_thinking,
                "total_tokens": total_all,
                "cost_usd": cost,
                "cost_per_page_usd": cost / num_pages if num_pages > 0 else 0.0,
                "input_tokens_per_page": total_input / num_pages if num_pages > 0 else 0.0,
                "output_tokens_per_page": total_output / num_pages if num_pages > 0 else 0.0,
            }

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
                f"OpenAIProvider only supports PARSE product type, got {raw_result.product_type}"
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
            else:
                markdown = page_data.get("markdown", "")

            pages.append(PageIR(page_index=page_index, markdown=markdown))
            page_markdowns.append(markdown)

        # Sort by page index and concatenate in sorted order
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
