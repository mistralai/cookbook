"""Provider for Layout-V3 layout detection with figure classification."""

from typing import Any

from extract_bench.inference.providers.base import ProviderPermanentError
from extract_bench.inference.providers.layoutdet.base import HFLayoutDetProvider
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.layout_detection_output import (
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
    LayoutV3Label,
)
from extract_bench.schemas.pipeline_io import InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType


@register_provider("layout_v3")
class LayoutV3Provider(HFLayoutDetProvider):
    """
    Provider for Layout-V3 layout detection model.

    This provider uses the Layout-V3 model served on HuggingFace
    inference endpoints for detecting document layout regions.

    Layout-V3 uses RT-DETRv2 with ResNet-50 backbone and automatically
    classifies detected Picture regions into 16 figure categories.

    Response format:
        {
            "pred_boxes": [[x1, y1, x2, y2], ...],
            "pred_classes": [class_id, ...],
            "pred_labels": ["Picture", "Text", ...],
            "scores": [score, ...],
            "figure_classifications": {
                "0": {
                    "figure_class": "bar_chart",
                    "figure_class_id": 0,
                    "figure_score": 0.89,
                    "top_3": [...]
                },
                ...
            }
        }
    """

    endpoint_env_var = "LAYOUT_V3_ENDPOINT_URL"
    model_type = LayoutDetectionModel.LAYOUT_V3

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """Initialize the Layout-V3 layout detection provider."""
        # Endpoint resolution (config `endpoint_url` -> env var) lives in the base class.
        super().__init__(provider_name, base_config)

    def _parse_response(self, response: dict[str, Any]) -> list[LayoutPrediction]:
        """
        Parse Layout-V3 response into layout predictions.

        :param response: Raw JSON response with pred_boxes, pred_classes,
                        pred_labels, scores, and figure_classifications
        :return: List of unified LayoutPrediction objects
        """
        predictions: list[LayoutPrediction] = []

        boxes = response.get("pred_boxes", [])
        classes = response.get("pred_classes", [])
        labels = response.get("pred_labels", [])
        scores = response.get("scores", [])
        figure_classifications = response.get("figure_classifications", {})

        for idx, (bbox, class_id, label_str, score) in enumerate(zip(boxes, classes, labels, scores, strict=False)):
            # Convert class_id to LayoutV3Label enum
            try:
                label = LayoutV3Label(class_id)
            except ValueError:
                # Unknown label, skip
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

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce LayoutOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        if raw_result.product_type != ProductType.LAYOUT_DETECTION:
            raise ProviderPermanentError(
                f"{self.__class__.__name__} only supports LAYOUT_DETECTION product type, got {raw_result.product_type}"
            )

        # Parse the response into raw predictions
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
