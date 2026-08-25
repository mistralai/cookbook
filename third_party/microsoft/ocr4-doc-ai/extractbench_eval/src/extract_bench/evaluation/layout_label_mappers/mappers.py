"""Concrete layout label mappers."""

from __future__ import annotations

from typing import Any

from extract_bench.evaluation.layout_label_mappers.base import (
    LayoutLabelMapper,
    MappingContext,
)
from extract_bench.evaluation.layout_label_mappers.registry import register_layout_label_mapper
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
    LayoutDetectionModel,
    LayoutPrediction,
)
from extract_bench.schemas.layout_ontology import CanonicalLabel


def _parse_int_label(label: str, model: LayoutDetectionModel) -> int:
    try:
        return int(label)
    except ValueError as exc:
        raise UnknownRawLayoutLabelError(
            f"Expected integer layout label for model '{model.value}', got '{label}'"
        ) from exc


@register_layout_label_mapper("__default__", priority=-100)
class CanonicalPassthroughMapper(LayoutLabelMapper):
    """Fallback mapper for already-canonical labels."""

    def to_canonical(
        self,
        label: str,
        prediction: LayoutPrediction,
        context: MappingContext,
    ) -> CanonicalLabel:
        del prediction, context
        try:
            return CanonicalLabel(label)
        except ValueError as exc:
            raise UnknownRawLayoutLabelError(f"Unknown raw layout label '{label}' and no mapper was resolved") from exc


@register_layout_label_mapper(
    "llamaparse",
    "model:llamaparse",
    priority=100,
)
class LlamaParseRawLabelMapper(LayoutLabelMapper):
    """Mapper for LlamaParse raw labels from `layoutAwareBbox[*].label`."""

    def _resolve_label_version(self, context: MappingContext) -> str:
        if context.raw_label_version:
            return context.raw_label_version
        labels = [pred.label for pred in context.layout_output.predictions if pred.label]
        return detect_llamaparse_label_version(labels)

    def should_include_prediction(
        self,
        prediction: LayoutPrediction,
        context: MappingContext,
    ) -> bool:
        version = self._resolve_label_version(context)
        # Preserve historical parity with prior evaluator behavior.
        return not (version == "v2" and prediction.label == "heading")

    def to_canonical(
        self,
        label: str,
        prediction: LayoutPrediction,
        context: MappingContext,
    ) -> CanonicalLabel:
        del prediction
        version = self._resolve_label_version(context)
        canonical, _attrs = map_llamaparse_raw_label_to_canonical(label, label_version=version)
        return canonical


@register_layout_label_mapper("docling_parse", "model:docling_parse_layout", priority=95)
class DoclingParseLabelMapper(LayoutLabelMapper):
    """Mapper for raw Docling labels emitted from the native DoclingDocument payload."""

    def to_canonical(
        self,
        label: str,
        prediction: LayoutPrediction,
        context: MappingContext,
    ) -> CanonicalLabel:
        del prediction, context
        canonical, _attrs = map_docling_raw_label_to_canonical(label)
        return canonical


@register_layout_label_mapper("oi_parser", "model:oi_parser_layout", priority=90)
class OIParserLabelMapper(LayoutLabelMapper):
    """Mapper for oi-parser layout labels.

    oi-parser emits Canonical17 labels in lowercase-hyphenated form (e.g.
    ``page-header``); normalize the casing back to the Canonical17 enum values.
    """

    # Canonical17 values keyed by their lowercased form for case-insensitive lookup.
    _BY_LOWER: dict[str, CanonicalLabel] = {label.value.lower(): label for label in CanonicalLabel}

    def to_canonical(
        self,
        label: str,
        prediction: LayoutPrediction,
        context: MappingContext,
    ) -> CanonicalLabel:
        del prediction, context
        canonical = self._BY_LOWER.get(label.strip().lower())
        if canonical is None:
            raise UnknownRawLayoutLabelError(f"Unknown oi-parser layout label '{label}'")
        return canonical


@register_layout_label_mapper(
    "model:yolo_doclaynet",
    "model:docling_layout_old",
    "model:docling_layout_heron_101",
    "model:docling_layout_heron",
    "model:ppdoclayout_plus_l",
    "model:qwen3_vl_8b",
    "model:gemini_layout",
    "model:openai_layout",
    "model:anthropic_layout",
    "model:gemma4_layout",
    "model:surya_layout",
    "model:chandra",
    "model:layout_v3",
    priority=90,
)
class IndexedLayoutModelMapper(LayoutLabelMapper):
    """Mapper for integer-index model outputs."""

    _adapters: dict[LayoutDetectionModel, Any] = {
        LayoutDetectionModel.YOLO_DOCLAYNET: YoloLayoutDetLabelAdapter(),
        LayoutDetectionModel.DOCLING_LAYOUT_OLD: DoclingLayoutDetLabelAdapter(),
        LayoutDetectionModel.DOCLING_LAYOUT_HERON_101: DoclingLayoutDetLabelAdapter(),
        LayoutDetectionModel.DOCLING_LAYOUT_HERON: DoclingLayoutDetLabelAdapter(),
        LayoutDetectionModel.PPDOCLAYOUT_PLUS_L: PPLayoutDetLabelAdapter(),
        LayoutDetectionModel.QWEN3_VL_8B: Qwen3VLLayoutDetLabelAdapter(),
        LayoutDetectionModel.GEMINI_LAYOUT: Qwen3VLLayoutDetLabelAdapter(),
        LayoutDetectionModel.OPENAI_LAYOUT: Qwen3VLLayoutDetLabelAdapter(),
        LayoutDetectionModel.ANTHROPIC_LAYOUT: Qwen3VLLayoutDetLabelAdapter(),
        LayoutDetectionModel.GEMMA4_LAYOUT: Qwen3VLLayoutDetLabelAdapter(),
        LayoutDetectionModel.SURYA_LAYOUT: SuryaLayoutDetLabelAdapter(),
        LayoutDetectionModel.CHANDRA: ChandraLayoutDetLabelAdapter(),
        LayoutDetectionModel.LAYOUT_V3: LayoutV3LabelAdapter(),
    }

    def to_canonical(
        self,
        label: str,
        prediction: LayoutPrediction,
        context: MappingContext,
    ) -> CanonicalLabel:
        adapter = self._adapters.get(context.model)
        if adapter is None:
            raise UnknownRawLayoutLabelError(f"No indexed label adapter for model '{context.model.value}'")

        label_int = _parse_int_label(label, context.model)
        mapped = None
        if context.model == LayoutDetectionModel.LAYOUT_V3 and hasattr(adapter, "to_canonical_with_figure_class"):
            figure_metadata = prediction.provider_metadata.get("figure_classification")
            figure_class = None
            figure_score = None
            if isinstance(figure_metadata, dict):
                figure_class = figure_metadata.get("figure_class")
                figure_score_value = figure_metadata.get("figure_score")
                if isinstance(figure_score_value, (int, float)):
                    figure_score = float(figure_score_value)
            mapped = adapter.to_canonical_with_figure_class(
                label_int,
                prediction.score,
                prediction.bbox,
                figure_class=figure_class,
                figure_score=figure_score,
            )
        else:
            mapped = adapter.to_canonical(label_int, prediction.score, prediction.bbox)

        if mapped is None:
            raise UnknownRawLayoutLabelError(f"Unknown raw layout label '{label}' for model '{context.model.value}'")
        return mapped.canonical_class  # type: ignore[no-any-return]


@register_layout_label_mapper("chunkr", "model:chunkr", priority=90)
class ChunkrLabelMapper(LayoutLabelMapper):
    """Mapper for Chunkr string labels."""

    _adapter = ChunkrLayoutDetLabelAdapter()

    def to_canonical(
        self,
        label: str,
        prediction: LayoutPrediction,
        context: MappingContext,
    ) -> CanonicalLabel:
        del context
        mapped = self._adapter.to_canonical(label, prediction.score, prediction.bbox)
        if mapped is None:
            raise UnknownRawLayoutLabelError(f"Unknown Chunkr raw layout label '{label}'")
        return mapped.canonical_class


@register_layout_label_mapper("dots_ocr_layout", "model:dots_ocr", priority=90)
class DotsOcrLabelMapper(LayoutLabelMapper):
    """Mapper for dots.ocr string labels."""

    _adapter = DotsOcrLayoutDetLabelAdapter()

    def to_canonical(
        self,
        label: str,
        prediction: LayoutPrediction,
        context: MappingContext,
    ) -> CanonicalLabel:
        del context
        mapped = self._adapter.to_canonical(label, prediction.score, prediction.bbox)
        if mapped is None:
            raise UnknownRawLayoutLabelError(f"Unknown dots.ocr raw layout label '{label}'")
        return mapped.canonical_class
