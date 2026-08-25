"""Provider for Layout-V3 BYOC (Bring Your Own Cloud) deployments."""

import io
import os
from datetime import datetime
from typing import Any

import requests
from PIL import Image

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.layout_detection_output import (
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
    LayoutV3Label,
)
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType


class LayoutV3BYOCProvider(Provider):
    """
    Base provider for Layout-V3 BYOC deployments.

    Uses multipart form data instead of raw image bytes (HuggingFace style).
    Response format is identical to the HuggingFace endpoint.
    """

    endpoint_url: str = ""
    model_type = LayoutDetectionModel.LAYOUT_V3

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        super().__init__(provider_name, base_config)

        # Allow endpoint_url override from config
        if base_config and "endpoint_url" in base_config:
            self.endpoint_url = base_config["endpoint_url"]

        if not self.endpoint_url:
            raise ProviderConfigError(
                f"endpoint_url is required for {self.__class__.__name__}. Set via config or environment variable."
            )

        self._timeout = self.base_config.get("timeout", 120)

    def _call_endpoint(self, image: Image.Image) -> dict[str, Any]:
        """Call BYOC endpoint with multipart form data."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        files = {"file": ("image.png", buffer, "image/png")}

        try:
            response = requests.post(
                f"{self.endpoint_url}/predict",
                files=files,
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

        except requests.exceptions.Timeout as e:
            raise ProviderTransientError(f"Request timed out: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ProviderTransientError(f"Connection error: {e}") from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            if status_code == 429:
                raise ProviderRateLimitError(f"Rate limit exceeded: {e}") from e
            elif status_code and 500 <= status_code < 600:
                raise ProviderTransientError(f"Server error ({status_code}): {e}") from e
            elif status_code and 400 <= status_code < 500:
                raise ProviderPermanentError(f"Client error ({status_code}): {e}") from e
            else:
                raise ProviderPermanentError(f"HTTP error: {e}") from e

    def _parse_response(self, response: dict[str, Any]) -> list[LayoutPrediction]:
        """Parse Layout-V3 response (same format as HF endpoint)."""
        predictions: list[LayoutPrediction] = []

        boxes = response.get("pred_boxes", [])
        classes = response.get("pred_classes", [])
        labels = response.get("pred_labels", [])
        scores = response.get("scores", [])
        figure_classifications = response.get("figure_classifications", {})

        for idx, (bbox, class_id, label_str, score) in enumerate(zip(boxes, classes, labels, scores, strict=False)):
            try:
                label = LayoutV3Label(class_id)
            except ValueError:
                continue

            predictions.append(
                LayoutPrediction(
                    bbox=bbox,
                    score=score,
                    label=str(int(label)),
                    provider_metadata={
                        "label_name": label.name,
                        "label_str": label_str,
                        "figure_classification": figure_classifications.get(str(idx)),
                    },
                )
            )

        return predictions

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """Run layout detection inference."""
        if request.product_type != ProductType.LAYOUT_DETECTION:
            raise ProviderPermanentError(f"{self.__class__.__name__} only supports LAYOUT_DETECTION")

        started_at = datetime.now()

        try:
            image = Image.open(request.source_file_path)
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")  # type: ignore[assignment]
        except Exception as e:
            raise ProviderPermanentError(f"Failed to load image: {e}") from e

        image_width, image_height = image.size
        raw_response = self._call_endpoint(image)

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        raw_output = {
            "response": raw_response,
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
        """Normalize raw inference result (identical to LayoutV3Provider)."""
        if raw_result.product_type != ProductType.LAYOUT_DETECTION:
            raise ProviderPermanentError(f"{self.__class__.__name__} only supports LAYOUT_DETECTION")

        response = raw_result.raw_output.get("response", {})
        raw_predictions = self._parse_response(response)

        output = LayoutOutput(
            task_type="layout_detection",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            model=self.model_type,
            image_width=max(int(raw_result.raw_output.get("image_width", 1)), 1),
            image_height=max(int(raw_result.raw_output.get("image_height", 1)), 1),
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


@register_provider("layout_v3_byoc_cpu")
class LayoutV3BYOCCPUProvider(LayoutV3BYOCProvider):
    """Layout-V3 BYOC provider for CPU deployments."""

    endpoint_url = os.getenv("LAYOUT_V3_BYOC_CPU_URL", "http://localhost:8001")


@register_provider("layout_v3_byoc_gpu")
class LayoutV3BYOCGPUProvider(LayoutV3BYOCProvider):
    """Layout-V3 BYOC provider for GPU deployments."""

    endpoint_url = os.getenv("LAYOUT_V3_BYOC_GPU_URL", "http://localhost:8002")
