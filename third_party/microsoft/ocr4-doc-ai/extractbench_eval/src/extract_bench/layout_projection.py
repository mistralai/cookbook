"""Runtime projection utilities for unified layout outputs."""

from __future__ import annotations

from collections.abc import Callable

from extract_bench.inference.providers.layoutdet.adapters import (
    ChandraLayoutDetLabelAdapter,
    ChunkrLayoutDetLabelAdapter,
    DoclingLayoutDetLabelAdapter,
    DotsOcrLayoutDetLabelAdapter,
    LayoutV3LabelAdapter,
    PPLayoutDetLabelAdapter,
    Qwen3VLLayoutDetLabelAdapter,
    SuryaLayoutDetLabelAdapter,
    YoloLayoutDetLabelAdapter,
)
from extract_bench.layout_label_mapping import (
    UnknownRawLayoutLabelError,
    detect_llamaparse_label_version,
    map_docling_raw_label_to_canonical,
    map_llamaparse_raw_label_to_canonical,
)
from extract_bench.schemas.layout_detection_output import (
    CanonicalLayoutPrediction,
    CoreLayoutPrediction,
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
)
from extract_bench.schemas.layout_ontology import CANONICAL_TO_CORE


def _parse_int_label(raw_label: str) -> int:
    """Parse an integer-ish raw label string into an int index."""
    try:
        return int(raw_label)
    except ValueError as exc:
        raise UnknownRawLayoutLabelError(f"Expected integer layout label, got '{raw_label}'") from exc


def _build_canonical(  # type: ignore[no-untyped-def]
    prediction: LayoutPrediction,
    canonical_class,
    mapped_attributes: dict[str, str],
) -> CanonicalLayoutPrediction:
    attributes = dict(mapped_attributes)
    attributes.update(prediction.attributes)
    return CanonicalLayoutPrediction(
        bbox=prediction.bbox,
        score=prediction.score,
        canonical_class=canonical_class,
        attributes=attributes,
        original_label=prediction.label,
        page=prediction.page,
    )


def _map_via_int_adapter(
    prediction: LayoutPrediction,
    adapter_to_canonical: Callable[[int, float, list[float]], CanonicalLayoutPrediction | None],
    model: LayoutDetectionModel,
) -> CanonicalLayoutPrediction:
    label_int = _parse_int_label(prediction.label)
    mapped = adapter_to_canonical(label_int, prediction.score, prediction.bbox)
    if mapped is None:
        raise UnknownRawLayoutLabelError(f"Unknown raw layout label '{prediction.label}' for model '{model.value}'")
    return _build_canonical(prediction, mapped.canonical_class, mapped.attributes)


def _map_via_str_adapter(
    prediction: LayoutPrediction,
    adapter_to_canonical: Callable[[str, float, list[float]], CanonicalLayoutPrediction | None],
    model: LayoutDetectionModel,
) -> CanonicalLayoutPrediction:
    mapped = adapter_to_canonical(prediction.label, prediction.score, prediction.bbox)
    if mapped is None:
        raise UnknownRawLayoutLabelError(f"Unknown raw layout label '{prediction.label}' for model '{model.value}'")
    return _build_canonical(prediction, mapped.canonical_class, mapped.attributes)


def project_to_canonical_predictions(
    layout_output: LayoutOutput,
    *,
    page_filter: int | None = None,
) -> list[CanonicalLayoutPrediction]:
    """Project unified raw predictions to canonical labels at runtime."""
    model = layout_output.model
    predictions = layout_output.predictions

    if page_filter is not None:
        predictions = [pred for pred in predictions if pred.page == page_filter]

    yolo_adapter = YoloLayoutDetLabelAdapter()
    docling_adapter = DoclingLayoutDetLabelAdapter()
    pp_adapter = PPLayoutDetLabelAdapter()
    qwen_adapter = Qwen3VLLayoutDetLabelAdapter()
    surya_adapter = SuryaLayoutDetLabelAdapter()
    chandra_adapter = ChandraLayoutDetLabelAdapter()
    layout_v3_adapter = LayoutV3LabelAdapter()
    chunkr_adapter = ChunkrLayoutDetLabelAdapter()
    dots_adapter = DotsOcrLayoutDetLabelAdapter()

    canonical_predictions: list[CanonicalLayoutPrediction] = []

    if model == LayoutDetectionModel.DOCLING_PARSE_LAYOUT:
        for pred in predictions:
            canonical_class, attrs = map_docling_raw_label_to_canonical(pred.label)
            canonical_predictions.append(_build_canonical(pred, canonical_class, attrs))
        return canonical_predictions

    if model == LayoutDetectionModel.LLAMAPARSE:
        labels = [pred.label for pred in predictions if pred.label]
        label_version = detect_llamaparse_label_version(labels)
        for pred in predictions:
            canonical_class, attrs = map_llamaparse_raw_label_to_canonical(
                pred.label,
                label_version=label_version,
            )
            canonical_predictions.append(_build_canonical(pred, canonical_class, attrs))
        return canonical_predictions

    for pred in predictions:
        if model == LayoutDetectionModel.YOLO_DOCLAYNET:
            canonical_predictions.append(_map_via_int_adapter(pred, yolo_adapter.to_canonical, model))
        elif model in {
            LayoutDetectionModel.DOCLING_LAYOUT_OLD,
            LayoutDetectionModel.DOCLING_LAYOUT_HERON_101,
            LayoutDetectionModel.DOCLING_LAYOUT_HERON,
        }:
            canonical_predictions.append(_map_via_int_adapter(pred, docling_adapter.to_canonical, model))
        elif model == LayoutDetectionModel.PPDOCLAYOUT_PLUS_L:
            canonical_predictions.append(_map_via_int_adapter(pred, pp_adapter.to_canonical, model))
        elif model == LayoutDetectionModel.QWEN3_VL_8B:
            canonical_predictions.append(_map_via_int_adapter(pred, qwen_adapter.to_canonical, model))
        elif model == LayoutDetectionModel.SURYA_LAYOUT:
            canonical_predictions.append(_map_via_int_adapter(pred, surya_adapter.to_canonical, model))
        elif model == LayoutDetectionModel.CHANDRA:
            canonical_predictions.append(_map_via_int_adapter(pred, chandra_adapter.to_canonical, model))
        elif model == LayoutDetectionModel.LAYOUT_V3:
            canonical_predictions.append(_map_via_int_adapter(pred, layout_v3_adapter.to_canonical, model))
        elif model == LayoutDetectionModel.CHUNKR:
            canonical_predictions.append(_map_via_str_adapter(pred, chunkr_adapter.to_canonical, model))
        elif model == LayoutDetectionModel.DOTS_OCR:
            canonical_predictions.append(_map_via_str_adapter(pred, dots_adapter.to_canonical, model))
        else:
            raise UnknownRawLayoutLabelError(f"No canonical mapping available for layout model '{model.value}'")

    return canonical_predictions


def project_to_core_predictions(
    layout_output: LayoutOutput,
    *,
    page_filter: int | None = None,
) -> list[CoreLayoutPrediction]:
    """Project unified raw predictions to core labels at runtime."""
    canonical_predictions = project_to_canonical_predictions(
        layout_output,
        page_filter=page_filter,
    )

    core_predictions: list[CoreLayoutPrediction] = []
    for canonical in canonical_predictions:
        core_class = CANONICAL_TO_CORE.get(canonical.canonical_class)
        if core_class is None:
            continue
        core_predictions.append(
            CoreLayoutPrediction(
                bbox=canonical.bbox,
                score=canonical.score,
                core_class=core_class,
                attributes=canonical.attributes,
                original_label=canonical.original_label,
                page=canonical.page,
            )
        )

    return core_predictions
