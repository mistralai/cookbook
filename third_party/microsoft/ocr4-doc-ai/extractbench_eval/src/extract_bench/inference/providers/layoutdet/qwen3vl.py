"""Provider for Qwen3-VL layout detection via a self-hosted OpenAI-compatible API."""

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
    QWEN3VL_STR_TO_LABEL,
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


@register_provider("qwen3vl_layout")
class Qwen3VLLayoutProvider(Provider):
    """
    Layout detection using Qwen3-VL-8B via a self-hosted OpenAI-compatible API.

    This provider sends images to the Qwen3-VL model and parses the JSON
    response containing layout predictions with normalized coordinates.

    Response format:
        [
            {"label": "text", "bbox_2d": [x1, y1, x2, y2], "score": 0.95},
            ...
        ]

    Coordinates are normalized to [0-1000] and converted to pixel coords.
    """

    model_type = LayoutDetectionModel.QWEN3_VL_8B

    # Image pixel constraints (from Qwen3-VL reference)
    MIN_PIXELS = 512 * 32 * 32  # 524,288
    MAX_PIXELS = 2048 * 32 * 32  # 2,097,152

    SYSTEM_PROMPT = """You are a document layout detector.
Output ONLY valid JSON (no markdown / html, no prose).
Use bbox_2d with normalized coordinates in [0, 1000] as [x1, y1, x2, y2]."""

    USER_PROMPT = """<image>
Locate every instance that belongs to the following document layout categories:
"caption", "footnote", "formula", "list_item", "page_footer", "page_header",
"picture", "section_header", "table", "text", "title".

Report bbox coordinates in JSON format.

Return ONLY a JSON array. Each element MUST be:
{
  "label": one of ["caption","footnote","formula",
    "list_item","page_footer","page_header","picture",
    "section_header","table","text","title"],
  "bbox_2d": [x1, y1, x2, y2],
  "score": number between 0.0 and 1.0
}

Rules:
- bbox_2d uses normalized 0-1000 coordinates [x1,y1,x2,y2]. (No pixel coords.)
- Detect DocLayNet-style BLOCKS (regions), not word/line boxes.
- Prefer a single box per logical region. Merge adjacent
  lines into one text block when they form a paragraph.
- Avoid duplicates: if two boxes overlap heavily
  (IoU > 0.7) and have the same label, keep only the
  one with the higher score.
- Output in approximate reading order (top-to-bottom, left-to-right).
- If no instances exist, return []."""

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """Initialize the Qwen3VL layout detection provider."""
        super().__init__(provider_name, base_config)

        # Self-hosted vLLM endpoint: config `endpoint_url` or QWEN3VL_ENDPOINT_URL env var.
        endpoint_url = self.base_config.get("endpoint_url") or os.getenv("QWEN3VL_ENDPOINT_URL")
        if not endpoint_url:
            raise ProviderConfigError(
                "Qwen3-VL layout detection requires a self-hosted vLLM server: set "
                "`endpoint_url` in the pipeline config or the QWEN3VL_ENDPOINT_URL "
                "environment variable (an OpenAI-compatible /v1 base URL, e.g. "
                "https://<your-qwen3vl-deployment>.modal.run/v1)"
            )
        self._client = OpenAI(
            base_url=endpoint_url,
            api_key="not-needed",
        )

        # Get timeout (default 120 seconds for VLM)
        self._timeout = self.base_config.get("timeout", 120)

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _extract_json(self, content: str) -> list[dict]:
        """
        Extract JSON array from LLM response, handling markdown fences.

        :param content: Raw response content from the model
        :return: Parsed JSON array
        :raises ValueError: If JSON cannot be extracted
        """
        # Try direct parse first
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Try to find array in content
        match = re.search(r"\[[\s\S]*\]", content)
        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract JSON from response: {content[:500]}")

    def _normalize_bbox(
        self,
        bbox_normalized: list[float],
        image_width: int,
        image_height: int,
    ) -> list[float]:
        """
        Convert [0-1000] normalized coords to pixel coords.

        :param bbox_normalized: Bounding box in [0-1000] normalized coords
        :param image_width: Actual image width in pixels
        :param image_height: Actual image height in pixels
        :return: Bounding box in pixel coordinates [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox_normalized

        # Clamp to valid range
        x1 = max(0, min(1000, x1))
        y1 = max(0, min(1000, y1))
        x2 = max(0, min(1000, x2))
        y2 = max(0, min(1000, y2))

        return [
            x1 * image_width / 1000,
            y1 * image_height / 1000,
            x2 * image_width / 1000,
            y2 * image_height / 1000,
        ]

    def _call_endpoint(self, image: Image.Image) -> tuple[list[dict], str]:
        """
        Call Qwen3VL via OpenAI API and return parsed predictions.

        :param image: PIL Image to analyze
        :return: Tuple of (parsed predictions list, raw response content)
        :raises ProviderError: For API errors
        """
        img_base64 = self._image_to_base64(image)

        try:
            response = self._client.chat.completions.create(  # type: ignore[call-overload]
                model=None,  # Not needed for this server
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "min_pixels": self.MIN_PIXELS,
                                "max_pixels": self.MAX_PIXELS,
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                            },
                            {"type": "text", "text": self.USER_PROMPT},
                        ],
                    },
                ],
                max_tokens=12384,
                temperature=0.7,
                extra_body={
                    "top_k": 20,
                    "top_p": 0.8,
                    "repetition_penalty": 1.05,
                },
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "connection" in error_msg:
                raise ProviderTransientError(f"API call failed: {e}") from e
            raise ProviderPermanentError(f"API call failed: {e}") from e

        content = response.choices[0].message.content
        if not content:
            raise ProviderPermanentError("Empty response from model")

        try:
            predictions = self._extract_json(content)
        except ValueError as e:
            raise ProviderPermanentError(str(e)) from e

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
            image = Image.open(request.source_file_path)
            # Ensure image is in RGB mode
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")  # type: ignore[assignment]
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

        Converts normalized [0-1000] coordinates to pixel coordinates and
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
            bbox_normalized = item.get("bbox_2d", [0, 0, 0, 0])
            score = item.get("score", 1.0)

            # Convert string label to enum
            label_enum = QWEN3VL_STR_TO_LABEL.get(label_str.lower())
            if label_enum is None:
                # Unknown label, skip
                continue

            # Clamp score to valid range
            score = max(0.0, min(1.0, float(score)))

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
