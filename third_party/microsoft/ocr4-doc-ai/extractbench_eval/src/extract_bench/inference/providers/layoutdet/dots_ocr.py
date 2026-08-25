"""Provider for dots.ocr layout detection via a self-hosted OpenAI-compatible API."""

import base64
import io
import json
import os
import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from openai import OpenAI
from PIL import Image

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.layout_detection_output import (
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
    LayoutTableContent,
    LayoutTextContent,
)
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

DEFAULT_PROMPT_MODE = "prompt_layout_all_en"

# dots.ocr 1.0 prompt (generic, no explicit layout categories)
PROMPT_LAYOUT_ALL_EN = (
    "Extract all the text in the image and return it in "
    "structured JSON format, including the bounding box "
    "for each text block. The text blocks should include "
    "general text, tables, and forms."
    "\n\n"
    "- Output must be valid JSON.\n"
    "- Do NOT use markdown format.\n"
    "- The bounding box must be in the format of "
    "[x1, y1, x2, y2], with (x1, y1) being the "
    "top-left corner and (x2, y2) being the "
    "bottom-right corner.\n"
    "- If there is text inside a table cell, extract it, "
    "and output a list of rows and columns for "
    "the table.\n"
    "- If there is a figure or diagram in the image, "
    "describe it briefly in one sentence."
)

# dots.ocr 1.5 prompt (Core11 layout categories, structured output)
PROMPT_LAYOUT_ALL_EN_V1_5 = (
    "Please output the layout information from the PDF image, "
    "including each layout element's bbox, its category, and the "
    "corresponding text content within the bbox.\n"
    "\n"
    "1. Bbox format: [x1, y1, x2, y2]\n"
    "\n"
    "2. Layout Categories: The possible categories are "
    "['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', "
    "'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n"
    "\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n"
    "\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, "
    "with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n"
    "\n"
    "5. Final Output: The entire output must be a single JSON object.\n"
)

PROMPT_LAYOUT_ONLY_EN_V1_5 = (
    "Please output the layout information from this PDF image, "
    "including each layout's bbox and its category. The bbox should be "
    "in the format [x1, y1, x2, y2]. The layout categories for the PDF "
    "document include ['Caption', 'Footnote', 'Formula', 'List-item', "
    "'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', "
    "'Text', 'Title']. Do not output the corresponding text. "
    "The layout result should be in JSON format."
)

PROMPT_DESCRIPTIONS = {
    "prompt_layout_all_en": "Layout + OCR JSON (bboxes + text + tables + figures)",
    "prompt_layout_all_en_v1_5": ("Layout + OCR JSON (Core11 categories, HTML tables, LaTeX formulas)"),
    "prompt_layout_only_en": "Layout JSON only (bboxes + classes, no OCR text)",
    "prompt_layout_only_en_v1_5": "Layout JSON only (Core11 categories, no OCR text)",
    "prompt_ocr": "Text-only OCR (markdown/plain text output)",
    "prompt_grounding_ocr": "OCR for a specified bounding box region",
}

PROMPT_ENV_VARS = {
    "prompt_layout_all_en": "DOTS_OCR_PROMPT_LAYOUT_ALL_EN",
    "prompt_layout_all_en_v1_5": "DOTS_OCR_PROMPT_LAYOUT_ALL_EN_V1_5",
    "prompt_layout_only_en": "DOTS_OCR_PROMPT_LAYOUT_ONLY_EN",
    "prompt_layout_only_en_v1_5": "DOTS_OCR_PROMPT_LAYOUT_ONLY_EN_V1_5",
    "prompt_ocr": "DOTS_OCR_PROMPT_OCR",
    "prompt_grounding_ocr": "DOTS_OCR_PROMPT_GROUNDING_OCR",
}

PROMPT_CONFIGS = {
    "prompt_layout_all_en": PROMPT_LAYOUT_ALL_EN,
    "prompt_layout_all_en_v1_5": PROMPT_LAYOUT_ALL_EN_V1_5,
    "prompt_layout_only_en": os.getenv("DOTS_OCR_PROMPT_LAYOUT_ONLY_EN"),
    "prompt_layout_only_en_v1_5": PROMPT_LAYOUT_ONLY_EN_V1_5,
    "prompt_ocr": os.getenv("DOTS_OCR_PROMPT_OCR"),
    "prompt_grounding_ocr": os.getenv("DOTS_OCR_PROMPT_GROUNDING_OCR"),
}


@register_provider("dots_ocr_layout")
class DotsOcrLayoutProvider(Provider):
    """
    Layout detection using dots.ocr via a self-hosted OpenAI-compatible API.

    Dots.ocr returns JSON containing layout elements with bboxes, categories, and
    OCR text. This provider extracts layout elements and maps them to canonical,
    core, and basic ontologies.
    """

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(provider_name, base_config)

        endpoint_url = self.base_config.get("endpoint_url") or os.getenv("DOTS_OCR_ENDPOINT_URL")
        if not endpoint_url:
            raise ProviderConfigError(
                "endpoint_url is required for dots_ocr_layout provider. "
                "Set DOTS_OCR_ENDPOINT_URL or pass endpoint_url in config."
            )

        self._client = OpenAI(
            base_url=endpoint_url,
            api_key=os.getenv("DOTS_OCR_API_KEY", "not-needed"),
        )

        self._timeout = self.base_config.get("timeout", 180)
        self._prompt_mode = self.base_config.get("prompt_mode", DEFAULT_PROMPT_MODE)
        self._prompt_override = self.base_config.get("prompt_override")
        self._prompt, self._prompt_description = _resolve_prompt(self._prompt_mode, self._prompt_override)
        self._max_tokens = self.base_config.get("max_tokens", 8192)
        self._temperature = self.base_config.get("temperature", 0.0)
        self._bbox_scale = self.base_config.get("bbox_scale")
        self._dpi = self.base_config.get("dpi", 150)
        self._page_index = self.base_config.get("page_index", 0)

    def _pdf_page_to_image(self, pdf_path: str, page_index: int) -> Image.Image:
        """Render a single PDF page to a PIL Image."""
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ProviderConfigError("pdf2image package not installed. Run: pip install pdf2image") from e

        try:
            images = convert_from_path(
                pdf_path,
                dpi=self._dpi,
                first_page=page_index + 1,
                last_page=page_index + 1,
            )
        except Exception as e:
            raise ProviderPermanentError(f"Failed to convert PDF page {page_index} to image: {e}") from e

        if not images:
            raise ProviderPermanentError(f"PDF has no page at index {page_index}: {pdf_path}")

        return images[0]

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _call_endpoint(self, image: Image.Image) -> tuple[list[dict[str, Any]], str]:
        """
        Call dots.ocr via OpenAI API and return parsed predictions.

        :param image: PIL Image to analyze
        :return: Tuple of (parsed predictions list, raw response content)
        :raises ProviderError: For API errors
        """
        img_base64 = self._image_to_base64(image)

        try:
            response = self._client.chat.completions.create(
                model=self.base_config.get("model", "dots-ocr"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                            },
                        ],
                    },
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "connection" in error_msg:
                raise ProviderTransientError(f"API call failed: {e}") from e
            raise ProviderPermanentError(f"API call failed: {e}") from e

        content = response.choices[0].message.content
        if not content:
            raise ProviderPermanentError("Empty response from model")

        payload = _extract_json(content)
        items = _extract_layout_items(payload)

        return items, content

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run layout detection inference on an image.

        :param pipeline: Pipeline specification
        :param request: Inference request (source_file_path should be an image)
        :return: Raw inference result
        :raises ProviderError: For any provider-related failures
        """
        if request.product_type != ProductType.LAYOUT_DETECTION:
            raise ProviderPermanentError(
                f"{self.__class__.__name__} only supports LAYOUT_DETECTION product type, got {request.product_type}"
            )

        started_at = datetime.now()

        source_path = request.source_file_path
        page_index = (
            request.config_override.get("page_index", self._page_index) if request.config_override else self._page_index
        )

        try:
            if source_path.lower().endswith(".pdf"):
                image = self._pdf_page_to_image(source_path, page_index)
            else:
                image = Image.open(source_path)
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")
        except ProviderPermanentError:
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Failed to load image: {e}") from e

        image_width, image_height = image.size

        items, raw_content = self._call_endpoint(image)
        normalized_items = _normalize_items(
            items,
            image_width=image_width,
            image_height=image_height,
            bbox_scale=self._bbox_scale,
        )

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        # page_index is 0-indexed; evaluator uses 1-indexed page numbers
        page_number = page_index + 1

        raw_output = {
            "response": normalized_items,
            "raw_content": raw_content,
            "image_width": image_width,
            "image_height": image_height,
            "page_number": page_number,
            "prompt_mode": self._prompt_mode,
            "prompt_description": self._prompt_description,
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

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce LayoutOutput.
        """
        if raw_result.product_type != ProductType.LAYOUT_DETECTION:
            raise ProviderPermanentError(
                f"{self.__class__.__name__} only supports LAYOUT_DETECTION product type, got {raw_result.product_type}"
            )

        image_width = raw_result.raw_output.get("image_width", 0)
        image_height = raw_result.raw_output.get("image_height", 0)
        default_page = raw_result.raw_output.get("page_number", 1)

        response = raw_result.raw_output.get("response", [])

        raw_predictions: list[LayoutPrediction] = []

        for item in response:
            label_str = item.get("label", "")
            bbox = item.get("bbox", [0, 0, 0, 0])
            score = item.get("score", 1.0)
            page = item.get("page") or default_page

            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 1.0
            score = max(0.0, min(1.0, score))

            # Build content from text returned by the model
            content = _build_content(label_str, item.get("text"))

            raw_predictions.append(
                LayoutPrediction(
                    bbox=bbox,
                    score=score,
                    label=label_str,
                    page=page,
                    content=content,
                    provider_metadata={"text": item.get("text")},
                )
            )
        output = LayoutOutput(
            task_type="layout_detection",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            model=LayoutDetectionModel.DOTS_OCR,
            image_width=max(int(image_width), 1),
            image_height=max(int(image_height), 1),
            predictions=raw_predictions,
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


def _build_content(label: str, text: str | None) -> LayoutTextContent | LayoutTableContent | None:
    """Build LayoutContent from model output text based on element label."""
    if not text:
        return None
    normalized = label.strip().lower()
    if normalized == "table":
        return LayoutTableContent(html=text)
    if normalized == "picture":
        return None
    return LayoutTextContent(text=text)


def _resolve_prompt(prompt_mode: str, prompt_override: str | None) -> tuple[str, str]:
    if prompt_override:
        return prompt_override, "custom override"

    prompt = PROMPT_CONFIGS.get(prompt_mode)
    if not prompt:
        env_var = PROMPT_ENV_VARS.get(prompt_mode, "DOTS_OCR_PROMPT")
        raise ProviderConfigError(f"Prompt for '{prompt_mode}' not configured. Set {env_var} or pass prompt_override.")

    description = PROMPT_DESCRIPTIONS.get(prompt_mode, "")
    return prompt, description


def _extract_json(content: str) -> dict | list:
    """Extract JSON object or array from LLM response."""
    try:
        result = json.loads(content)
        if isinstance(result, (dict, list)):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, (dict, list)):
                return result
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, (dict, list)):
                return result
        except json.JSONDecodeError:
            pass

    match = re.search(r"\[[\s\S]*\]", content)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, (dict, list)):
                return result
        except json.JSONDecodeError:
            pass

    raise ProviderPermanentError(f"Could not extract JSON from response: {content[:500]}")


def _extract_layout_items(payload: dict | list) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        raise ProviderPermanentError("dots.ocr response JSON is not an object or array")

    items = _extract_items_from_container(payload)
    if items:
        return items

    pages = payload.get("pages") or payload.get("page_results") or payload.get("results")
    if isinstance(pages, list):
        collected: list[dict[str, Any]] = []
        for idx, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            page_items = _extract_items_from_container(page)
            if not page_items and _looks_like_item(page):
                page_items = [page]
            page_num = page.get("page") or page.get("page_num") or page.get("page_index")
            if page_num is None:
                page_num = idx + 1
            for item in page_items:
                if "page" not in item:
                    item["page"] = page_num
            collected.extend(page_items)
        if collected:
            return collected

    if _looks_like_item(payload):
        return [payload]

    raise ProviderPermanentError("dots.ocr response JSON did not contain recognizable layout items")


def _extract_items_from_container(container: dict[str, Any]) -> list[dict[str, Any]]:
    for key in (
        "cells",
        "layout",
        "elements",
        "blocks",
        "items",
        "regions",
        "predictions",
        "detections",
        "layout_elements",
        "text_blocks",
    ):
        value = container.get(key)
        if isinstance(value, list):
            return value
    return []


def _looks_like_item(item: dict[str, Any]) -> bool:
    return any(key in item for key in ("bbox", "bounding_box", "box", "label", "category"))


def _normalize_items(
    items: Iterable[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    bbox_scale: float | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        label = _extract_label(item)
        if not label:
            continue

        bbox = _extract_bbox(item)
        if bbox is None:
            continue

        if bbox_scale:
            bbox = _scale_bbox(bbox, image_width, image_height, bbox_scale)

        score = item.get("score", item.get("confidence", 1.0))
        try:
            score = float(score)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            score = 1.0
        score = max(0.0, min(1.0, score))

        page = item.get("page") or item.get("page_num") or item.get("page_index")

        normalized.append(
            {
                "bbox": bbox,
                "label": str(label),
                "score": score,
                "page": page,
                "text": item.get("text"),
            }
        )

    return normalized


def _extract_label(item: dict[str, Any]) -> str | None:
    for key in ("category", "label", "type", "class", "category_type"):
        value = item.get(key)
        if value:
            return str(value)
    return None


def _extract_bbox(item: dict[str, Any]) -> list[float] | None:
    for key in ("bbox", "bounding_box", "box", "bbox_2d", "bbox2d", "coordinates"):
        value = item.get(key)
        if value is None:
            continue
        bbox = _coerce_bbox(value)
        if bbox is not None:
            return bbox
    return None


def _coerce_bbox(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(value.keys()):
            return [
                float(value["x1"]),
                float(value["y1"]),
                float(value["x2"]),
                float(value["y2"]),
            ]
        if {"left", "top", "right", "bottom"}.issubset(value.keys()):
            return [
                float(value["left"]),
                float(value["top"]),
                float(value["right"]),
                float(value["bottom"]),
            ]
        if {"x", "y", "w", "h"}.issubset(value.keys()):
            x = float(value["x"])
            y = float(value["y"])
            w = float(value["w"])
            h = float(value["h"])
            return [x, y, x + w, y + h]

    if isinstance(value, (list, tuple)):
        if len(value) == 4:
            return [float(v) for v in value]
        if len(value) == 2 and all(isinstance(v, (list, tuple)) for v in value):
            flat = [float(v) for pair in value for v in pair]
            if len(flat) == 4:
                return flat
        if len(value) == 8:
            xs = [float(value[i]) for i in range(0, 8, 2)]
            ys = [float(value[i]) for i in range(1, 8, 2)]
            return [min(xs), min(ys), max(xs), max(ys)]

    return None


def _scale_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
    bbox_scale: float,
) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        x1 * image_width / bbox_scale,
        y1 * image_height / bbox_scale,
        x2 * image_width / bbox_scale,
        y2 * image_height / bbox_scale,
    ]
