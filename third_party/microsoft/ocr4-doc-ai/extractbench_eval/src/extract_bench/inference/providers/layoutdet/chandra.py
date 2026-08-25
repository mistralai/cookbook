"""Provider for Chandra OCR layout detection via a self-hosted OpenAI-compatible API."""

import base64
import io
import json
import logging
import os
import re
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
    CHANDRA_STR_TO_LABEL,
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
)
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

logger = logging.getLogger(__name__)

# Allowed HTML tags for Chandra output
ALLOWED_TAGS = [
    "math",
    "br",
    "i",
    "b",
    "u",
    "del",
    "sup",
    "sub",
    "table",
    "tr",
    "td",
    "p",
    "th",
    "div",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "ul",
    "ol",
    "li",
    "input",
    "a",
    "span",
    "img",
    "hr",
    "tbody",
    "small",
    "caption",
    "strong",
    "thead",
    "big",
    "code",
]

ALLOWED_ATTRIBUTES = [
    "class",
    "colspan",
    "rowspan",
    "display",
    "checked",
    "type",
    "border",
    "value",
    "style",
    "href",
    "alt",
    "align",
]

PROMPT_ENDING = (
    f"Only use these tags {ALLOWED_TAGS}, "
    f"and these attributes {ALLOWED_ATTRIBUTES}."
    "\n\nGuidelines:\n"
    "* Inline math: Surround math with <math>...</math> "
    "tags. Math expressions should be rendered in "
    "KaTeX-compatible LaTeX. Use display for block "
    "math.\n"
    "* Tables: Use colspan and rowspan attributes to "
    "match table structure.\n"
    "* Formatting: Maintain consistent formatting with "
    "the image, including spacing, indentation, "
    "subscripts/superscripts, and special characters.\n"
    "* Images: Include a description of any images in "
    "the alt attribute of an <img> tag. Do not fill out "
    "the src property.\n"
    "* Forms: Mark checkboxes and radio buttons "
    "properly.\n"
    "* Text: join lines together properly into paragraphs "
    "using <p>...</p> tags.  Use <br> tags for line "
    "breaks within paragraphs, but only when absolutely "
    "necessary to maintain meaning.\n"
    "* Use the simplest possible HTML structure that "
    "accurately represents the content of the block.\n"
    "* Make sure the text is accurate and easy for a "
    "human to read and interpret.  Reading order should "
    "be correct and natural."
)


@register_provider("chandra_layout")
class ChandraLayoutProvider(Provider):
    """
    Layout detection using Chandra OCR via a self-hosted OpenAI-compatible API.

    This provider sends images to the Chandra model using the ocr_layout prompt
    and parses the HTML response containing layout blocks with bounding boxes.

    Response format (HTML):
        <div data-bbox="[x0, y0, x1, y1]" data-label="Text">content...</div>
        <div data-bbox="[x0, y0, x1, y1]" data-label="Table">...</div>
        ...

    Coordinates are normalized to [0-1024] and converted to pixel coords.
    """

    model_type = LayoutDetectionModel.CHANDRA

    # Chandra uses 0-1024 normalized bbox coordinates
    BBOX_SCALE = 1024

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """Initialize the Chandra layout detection provider."""
        super().__init__(provider_name, base_config)

        # Self-hosted vLLM endpoint: config `endpoint_url` or CHANDRA_ENDPOINT_URL env var.
        endpoint_url = self.base_config.get("endpoint_url") or os.getenv("CHANDRA_ENDPOINT_URL")
        if not endpoint_url:
            raise ProviderConfigError(
                "Chandra layout detection requires a self-hosted vLLM server: set "
                "`endpoint_url` in the pipeline config or the CHANDRA_ENDPOINT_URL "
                "environment variable (an OpenAI-compatible /v1 base URL, e.g. "
                "https://<your-chandra-deployment>.modal.run/v1)"
            )
        self._client = OpenAI(
            base_url=endpoint_url,
            api_key="not-needed",
        )

        # Get timeout (default 180 seconds for VLM - Chandra can be slow)
        self._timeout = self.base_config.get("timeout", 180)

        # Build the prompt with bbox_scale
        self._prompt = self._build_ocr_layout_prompt()

    def _build_ocr_layout_prompt(self) -> str:
        """Build the OCR layout prompt with the correct bbox_scale."""
        return (
            "OCR this image to HTML, arranged as layout "
            "blocks.  Each layout block should be a div "
            "with the data-bbox attribute representing the "
            "bounding box of the block in "
            "[x0, y0, x1, y1] format.  "
            f"Bboxes are normalized 0-{self.BBOX_SCALE}. "
            "The data-label attribute is the label for "
            "the block."
            "\n\nUse the following labels:\n"
            "- Caption\n"
            "- Footnote\n"
            "- Equation-Block\n"
            "- List-Group\n"
            "- Page-Header\n"
            "- Page-Footer\n"
            "- Image\n"
            "- Section-Header\n"
            "- Table\n"
            "- Text\n"
            "- Complex-Block\n"
            "- Code-Block\n"
            "- Form\n"
            "- Table-Of-Contents\n"
            "- Figure"
            f"\n\n{PROMPT_ENDING}"
        ).strip()

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_html_layout_blocks(self, html_content: str) -> list[dict[str, Any]]:
        """
        Parse HTML content to extract layout blocks with data-bbox and data-label.

        :param html_content: Raw HTML response from Chandra
        :return: List of dicts with 'bbox', 'label' keys
        """
        predictions: list[dict[str, Any]] = []

        # Pattern to match divs with data-bbox and data-label attributes
        # Handles both attribute orders: data-bbox first or data-label first
        pattern = r'<div[^>]*data-bbox=["\'](\[[^\]]+\])["\'][^>]*data-label=["\']([^"\']+)["\']'
        pattern_alt = r'<div[^>]*data-label=["\']([^"\']+)["\'][^>]*data-bbox=["\'](\[[^\]]+\])["\']'

        # Find all matches with data-bbox first
        for match in re.finditer(pattern, html_content):
            bbox_str = match.group(1)
            label = match.group(2)
            try:
                bbox = json.loads(bbox_str)
                if isinstance(bbox, list) and len(bbox) == 4:
                    predictions.append(
                        {
                            "bbox": bbox,
                            "label": label,
                        }
                    )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse bbox: {bbox_str}")
                continue

        # Find all matches with data-label first
        for match in re.finditer(pattern_alt, html_content):
            label = match.group(1)
            bbox_str = match.group(2)
            try:
                bbox = json.loads(bbox_str)
                if isinstance(bbox, list) and len(bbox) == 4:
                    predictions.append(
                        {
                            "bbox": bbox,
                            "label": label,
                        }
                    )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse bbox: {bbox_str}")
                continue

        return predictions

    def _normalize_bbox(
        self,
        bbox_normalized: list[float],
        image_width: int,
        image_height: int,
    ) -> list[float]:
        """
        Convert [0-1024] normalized coords to pixel coords.

        :param bbox_normalized: Bounding box in [0-1024] normalized coords
        :param image_width: Actual image width in pixels
        :param image_height: Actual image height in pixels
        :return: Bounding box in pixel coordinates [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox_normalized

        # Clamp to valid range
        x1 = max(0, min(self.BBOX_SCALE, x1))
        y1 = max(0, min(self.BBOX_SCALE, y1))
        x2 = max(0, min(self.BBOX_SCALE, x2))
        y2 = max(0, min(self.BBOX_SCALE, y2))

        return [
            x1 * image_width / self.BBOX_SCALE,
            y1 * image_height / self.BBOX_SCALE,
            x2 * image_width / self.BBOX_SCALE,
            y2 * image_height / self.BBOX_SCALE,
        ]

    def _call_endpoint(self, image: Image.Image) -> tuple[list[dict], str]:
        """
        Call Chandra via OpenAI API and return parsed predictions.

        :param image: PIL Image to analyze
        :return: Tuple of (parsed predictions list, raw response content)
        :raises ProviderError: For API errors
        """
        img_base64 = self._image_to_base64(image)

        try:
            response = self._client.chat.completions.create(
                model="chandra",  # Model name configured in vLLM
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
                max_tokens=8192,
                temperature=0.1,
                top_p=0.1,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "connection" in error_msg:
                raise ProviderTransientError(f"API call failed: {e}") from e
            raise ProviderPermanentError(f"API call failed: {e}") from e

        content = response.choices[0].message.content
        if not content:
            raise ProviderPermanentError("Empty response from model")

        # Parse HTML to extract layout blocks
        predictions = self._parse_html_layout_blocks(content)

        return predictions, content

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

        # Load the image
        try:
            image: Image.Image = Image.open(request.source_file_path)
            # Ensure image is in RGB mode
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")
        except Exception as e:
            raise ProviderPermanentError(f"Failed to load image: {e}") from e

        # Get image dimensions
        image_width, image_height = image.size

        # Call the endpoint
        predictions, raw_content = self._call_endpoint(image)

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        # Store in raw output for normalization
        raw_output = {
            "response": predictions,
            "raw_content": raw_content,
            "image_width": image_width,
            "image_height": image_height,
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

        Converts normalized [0-1024] coordinates to pixel coordinates and
        maps string labels to canonical labels.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        if raw_result.product_type != ProductType.LAYOUT_DETECTION:
            raise ProviderPermanentError(
                f"{self.__class__.__name__} only supports LAYOUT_DETECTION product type, got {raw_result.product_type}"
            )

        # Get image dimensions
        image_width = raw_result.raw_output.get("image_width", 0)
        image_height = raw_result.raw_output.get("image_height", 0)

        # Parse the response into predictions
        response = raw_result.raw_output.get("response", [])

        raw_predictions: list[LayoutPrediction] = []

        for item in response:
            label_str = item.get("label", "")
            bbox_normalized = item.get("bbox", [0, 0, 0, 0])

            # Convert string label to enum
            label_enum = CHANDRA_STR_TO_LABEL.get(label_str)
            if label_enum is None:
                # Unknown label, skip
                logger.warning(f"Unknown Chandra label: {label_str}")
                continue

            # Chandra doesn't output confidence scores, use 1.0
            score = 1.0

            # Convert normalized coords to pixel coords
            bbox_pixels = self._normalize_bbox(bbox_normalized, image_width, image_height)

            # Create raw prediction
            raw_predictions.append(
                LayoutPrediction(
                    bbox=bbox_pixels,
                    score=score,
                    label=str(int(label_enum)),
                    provider_metadata={"label_name": label_enum.name},
                )
            )

        output = LayoutOutput(
            task_type="layout_detection",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            model=self.model_type,
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
