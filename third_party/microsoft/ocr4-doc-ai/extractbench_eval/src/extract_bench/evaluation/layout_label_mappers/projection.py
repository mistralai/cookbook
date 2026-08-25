"""Projection helpers that map unified layout predictions to evaluation labels."""

from __future__ import annotations

from typing import Literal

from extract_bench.evaluation.layout_adapters.base import normalize_bbox_xyxy
from extract_bench.evaluation.layout_label_mappers.registry import (
    build_mapping_context,
    resolve_layout_label_mapper,
)
from extract_bench.schemas.layout_detection_output import LayoutOutput
from extract_bench.schemas.layout_ontology import CANONICAL_TO_CORE
from extract_bench.schemas.pipeline_io import InferenceResult


def project_layout_predictions(
    inference_result: InferenceResult,
    layout_output: LayoutOutput,
    *,
    evaluation_view: Literal["core", "canonical"] = "core",
    target_ontology: str = "basic",
    page_filter: int | None = None,
) -> list[dict[str, float | str | list[float] | int | None]]:
    """Project unified layout predictions to evaluator-ready class labels."""
    if layout_output.image_width <= 0 or layout_output.image_height <= 0:
        return []

    context = build_mapping_context(inference_result, layout_output)
    mapper = resolve_layout_label_mapper(context)

    projected: list[dict[str, float | str | list[float] | int | None]] = []
    for prediction in layout_output.predictions:
        if page_filter is not None and prediction.page != page_filter:
            continue
        if not mapper.should_include_prediction(prediction, context):
            continue

        canonical = mapper.to_canonical(prediction.label, prediction, context)
        label_for_view = canonical
        if evaluation_view == "core":
            core_class = CANONICAL_TO_CORE.get(canonical)
            if core_class is None:
                continue
            label_for_view = core_class

        class_name = mapper.to_target_ontology(label_for_view, target_ontology)
        raw_order_index = prediction.provider_metadata.get("order_index")
        order_index = raw_order_index if isinstance(raw_order_index, int) else None
        projected.append(
            {
                "bbox": normalize_bbox_xyxy(
                    prediction.bbox,
                    width=layout_output.image_width,
                    height=layout_output.image_height,
                ),
                "class_name": class_name,
                "score": prediction.score,
                "page": prediction.page,
                "order_index": order_index,
            }
        )

    return projected
