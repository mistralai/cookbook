"""Provider for YOLO-DocLayNet layout detection."""

from typing import Any

from extract_bench.inference.providers.base import ProviderPermanentError
from extract_bench.inference.providers.layoutdet.base import HFLayoutDetProvider
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.layout_detection_output import (
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
    YoloLabel,
)
from extract_bench.schemas.pipeline_io import InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType


@register_provider("yolo_layout")
class YoloLayoutProvider(HFLayoutDetProvider):
    """
    Provider for YOLO-DocLayNet layout detection model.

    This provider uses the YOLO model trained on DocLayNet served on HuggingFace
    inference endpoints for detecting document layout regions.

    Response format:
        {
            "pred_boxes": [[x1, y1, x2, y2], ...],
            "pred_classes": [class_id, ...],
            "scores": [score, ...]
        }
    """

    endpoint_env_var = "YOLO_LAYOUT_ENDPOINT_URL"
    model_type = LayoutDetectionModel.YOLO_DOCLAYNET

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """Initialize the YOLO-DocLayNet layout detection provider."""
        super().__init__(provider_name, base_config)

    def _parse_response(self, response: dict[str, Any]) -> list[LayoutPrediction]:
        """
        Parse YOLO-DocLayNet response into layout predictions.

        :param response: Raw JSON response with pred_boxes, pred_classes, scores
        :return: List of unified LayoutPrediction objects
        """
        predictions: list[LayoutPrediction] = []

        boxes = response.get("pred_boxes", [])
        classes = response.get("pred_classes", [])
        scores = response.get("scores", [])

        for bbox, class_id, score in zip(boxes, classes, scores, strict=False):
            # Convert class_id to YoloLabel enum
            # Model outputs 0-indexed labels (0-10) that match YoloLabel directly
            label = YoloLabel(class_id)
            predictions.append(
                LayoutPrediction(
                    bbox=bbox,
                    score=score,
                    label=str(int(label)),
                    provider_metadata={"label_name": label.name},
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
