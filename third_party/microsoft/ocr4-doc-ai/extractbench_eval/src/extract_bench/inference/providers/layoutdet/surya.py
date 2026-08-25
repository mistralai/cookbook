"""Provider for Surya OCR layout detection via a self-hosted HTTP API."""

import base64
import io
import logging
import os
from datetime import datetime
from typing import Any

import requests
from PIL import Image

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.layout_detection_output import (
    SURYA_STR_TO_LABEL,
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


@register_provider("surya_layout")
class SuryaLayoutProvider(Provider):
    """
    Layout detection using Surya OCR via a self-hosted HTTP API.

    This provider sends images to the Surya layout detection model
    deployed on your own infrastructure and parses the JSON response.

    Response format from the endpoint:
        {
            "predictions": [
                {
                    "bbox": [x1, y1, x2, y2],
                    "label": "Text",
                    "score": 0.95,
                    "position": 0
                },
                ...
            ],
            "image_width": 612,
            "image_height": 792
        }

    Coordinates are already in pixel coordinates.
    """

    model_type = LayoutDetectionModel.SURYA_LAYOUT

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """Initialize the Surya layout detection provider."""
        super().__init__(provider_name, base_config)

        # Self-hosted endpoint: config `endpoint_url` or SURYA_ENDPOINT_URL env var.
        self.endpoint_url = self.base_config.get("endpoint_url") or os.getenv("SURYA_ENDPOINT_URL")
        if not self.endpoint_url:
            raise ProviderConfigError(
                "Surya layout detection requires a self-hosted server: set `endpoint_url` "
                "in the pipeline config or the SURYA_ENDPOINT_URL environment variable "
                "(e.g. https://<your-surya-deployment>.modal.run)"
            )

        # Get timeout (default 120 seconds)
        self._timeout = self.base_config.get("timeout", 120)

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _call_endpoint(self, image: Image.Image) -> dict[str, Any]:
        """
        Call the Surya endpoint with base64 image.

        :param image: PIL Image to analyze
        :return: Parsed JSON response
        :raises ProviderError: For API errors
        """
        img_base64 = self._image_to_base64(image)

        try:
            response = requests.post(
                f"{self.endpoint_url}/predict",
                json={"image": img_base64},
                headers={"Content-Type": "application/json"},
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout as e:
            raise ProviderTransientError(f"Request timed out: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ProviderTransientError(f"Connection error: {e}") from e
        except Exception as e:
            raise ProviderPermanentError(f"Request failed: {e}") from e

        # Handle HTTP errors
        if response.status_code == 429:
            raise ProviderTransientError("Rate limited (429)")
        if response.status_code >= 500:
            raise ProviderTransientError(f"Server error ({response.status_code}): {response.text[:500]}")
        if response.status_code >= 400:
            raise ProviderPermanentError(f"Client error ({response.status_code}): {response.text[:500]}")

        try:
            result: dict[str, Any] = response.json()
        except Exception as e:
            raise ProviderPermanentError(f"Failed to parse JSON response: {e}") from e

        # Check for error in response
        if "error" in result:
            raise ProviderPermanentError(f"API error: {result['error']}")

        return result

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

        # Call the endpoint
        result = self._call_endpoint(image)

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        # Store in raw output for normalization
        raw_output = {
            "response": result.get("predictions", []),
            "image_width": result.get("image_width", image.size[0]),
            "image_height": result.get("image_height", image.size[1]),
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

        Maps string labels to canonical labels using the adapter.

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
            bbox = item.get("bbox", [0, 0, 0, 0])
            score = item.get("score", 1.0)
            position = item.get("position", 0)

            # Convert string label to enum
            label_enum = SURYA_STR_TO_LABEL.get(label_str)
            if label_enum is None:
                # Unknown label, skip
                logger.warning(f"Unknown Surya label: {label_str}")
                continue

            # Clamp score to valid range
            score = max(0.0, min(1.0, float(score)))

            # Create raw prediction (bbox is already in pixel coordinates)
            raw_predictions.append(
                LayoutPrediction(
                    bbox=bbox,
                    score=score,
                    label=str(int(label_enum)),
                    provider_metadata={
                        "label_name": label_enum.name,
                        "position": position,
                    },
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
