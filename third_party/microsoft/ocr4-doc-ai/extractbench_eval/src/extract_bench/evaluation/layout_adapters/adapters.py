"""Concrete layout adapters and registry bindings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

from extract_bench.evaluation.layout_adapters.base import LayoutAdapter
from extract_bench.evaluation.layout_adapters.registry import register_layout_adapter
from extract_bench.evaluation.metrics.attribution.core import (
    PredBlock,
    parse_pred_blocks,
)
from extract_bench.evaluation.metrics.attribution.text_utils import (
    extract_text_from_html,
    normalize_attribution_text,
    tokenize,
)
from extract_bench.inference.layout_extraction import (
    extract_all_layouts_from_llamaparse_output,
)
from extract_bench.inference.providers.layoutdet.adapters import ChunkrLayoutDetLabelAdapter
from extract_bench.inference.providers.parse.geometry import polygon_to_literal_xywh_r
from extract_bench.inference.providers.parse.llamaparse_v2_normalization import (
    build_pages_from_cli2_raw_payload,
    build_pages_from_sdk_response_payload,
    layout_pages_to_legacy_pages_payload,
)
from extract_bench.layout_label_mapping import (
    UnknownRawLayoutLabelError,
)
from extract_bench.schemas.layout_detection_output import (
    QWEN3VL_STR_TO_LABEL,
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
    LayoutTableContent,
    LayoutTextContent,
)
from extract_bench.schemas.parse_output import ParseOutput
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.test_cases.schema import TestCase


@dataclass(frozen=True)
class _GranularSegment:
    x: float
    y: float
    w: float
    h: float
    r: float | None = None


@dataclass(frozen=True)
class _GranularTextUnit:
    text: str
    bbox: _GranularSegment
    order_index: int


@dataclass(frozen=True)
class _GranularPage:
    page_number: int
    lines: list[_GranularTextUnit]
    words: list[_GranularTextUnit]


@register_layout_adapter("__default__", priority=-100)
class NormalizedLayoutOutputAdapter(LayoutAdapter):
    """Adapter for providers that already emit `LayoutOutput`."""

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if not isinstance(inference_result.output, LayoutOutput):
            raise ValueError("Inference output is not LayoutOutput and no provider adapter matched.")

        if page_filter is None:
            return inference_result.output

        predictions = [
            prediction for prediction in inference_result.output.predictions if prediction.page == page_filter
        ]
        return inference_result.output.model_copy(update={"predictions": predictions})


@register_layout_adapter(
    "llamaparse",
    "llamaparse_local_cli2",
    priority=100,
)
class LlamaParseLayoutAdapter(LayoutAdapter):
    """Adapter for LlamaParse-family outputs with output-first + legacy fallback support."""

    def __init__(self) -> None:
        self._pages_payload: list[dict[str, Any]] | None = None
        self._attribution_pages_payload: list[dict[str, Any]] | None = None

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if isinstance(inference_result.output, ParseOutput):
            if len(inference_result.output.layout_pages) > 0 or len(inference_result.output.grounded_pages) > 0:
                return True

        if (
            isinstance(inference_result.output, LayoutOutput)
            and inference_result.output.model == LayoutDetectionModel.LLAMAPARSE
        ):
            return True

        raw_output = inference_result.raw_output
        if not isinstance(raw_output, dict):
            return False
        pages = raw_output.get("pages")
        if not isinstance(pages, list) or not pages:
            return False
        first_page = pages[0]
        if not isinstance(first_page, dict):
            return False
        items = first_page.get("items")
        return isinstance(items, list)

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        pages = _resolve_llamaparse_pages(inference_result)
        self._attribution_pages_payload = _resolve_llamaparse_pages(
            inference_result,
            include_bbox_segment_fallback=False,
        )
        raw_output = inference_result.raw_output if isinstance(inference_result.raw_output, dict) else {}

        if pages:
            self._pages_payload = pages
            extraction_input: dict[str, Any] = {"pages": pages}
            raw_image_width = raw_output.get("image_width")
            raw_image_height = raw_output.get("image_height")
            if isinstance(raw_image_width, (int, float)) and isinstance(raw_image_height, (int, float)):
                extraction_input["image_width"] = raw_image_width
                extraction_input["image_height"] = raw_image_height

            layout_output = extract_all_layouts_from_llamaparse_output(
                raw_output=extraction_input,
                example_id=inference_result.request.example_id,
                pipeline_name=inference_result.pipeline_name,
            )
            if page_filter is None:
                return layout_output

            predictions = [prediction for prediction in layout_output.predictions if prediction.page == page_filter]
            return layout_output.model_copy(update={"predictions": predictions})

        self._pages_payload = None
        if (
            isinstance(inference_result.output, LayoutOutput)
            and inference_result.output.model == LayoutDetectionModel.LLAMAPARSE
        ):
            if page_filter is None:
                return inference_result.output
            predictions = [
                prediction for prediction in inference_result.output.predictions if prediction.page == page_filter
            ]
            return inference_result.output.model_copy(update={"predictions": predictions})

        raise ValueError("LlamaParse adapter requires ParseOutput.layout_pages or raw_output.pages")

    def to_attribution_blocks(
        self,
        layout_output: LayoutOutput,
        *,
        page_number: int,
        test_case: TestCase | None = None,
    ) -> list[PredBlock]:
        del test_case
        if self._pages_payload is None:
            return super().to_attribution_blocks(
                layout_output,
                page_number=page_number,
                test_case=None,
            )

        attribution_pages = self._attribution_pages_payload or self._pages_payload
        raw_page = _find_page_payload(attribution_pages, page_number)
        if raw_page is None:
            return super().to_attribution_blocks(
                layout_output,
                page_number=page_number,
                test_case=None,
            )

        items = raw_page.get("items")
        if not isinstance(items, list):
            return super().to_attribution_blocks(
                layout_output,
                page_number=page_number,
                test_case=None,
            )

        page_md = raw_page.get("md", "") or raw_page.get("text", "") or ""
        page_width = float(raw_page.get("width") or layout_output.image_width or 1)
        page_height = float(raw_page.get("height") or layout_output.image_height or 1)
        return parse_pred_blocks(
            items,
            page_md,
            page_width,
            page_height,
            require_layout_aware_segments=True,
        )

    def to_granular_pages(self, inference_result: InferenceResult) -> list[_GranularPage]:
        if isinstance(inference_result.output, ParseOutput) and inference_result.output.grounded_pages:
            return _build_llamaparse_granular_pages_from_payload(inference_result.output.grounded_pages)

        raw_output = inference_result.raw_output if isinstance(inference_result.raw_output, dict) else {}
        grounded_pages = raw_output.get("v2_grounded_items", raw_output.get("grounded_items"))
        return _build_llamaparse_granular_pages_from_payload(grounded_pages)


@register_layout_adapter("warp_ingest", priority=100)
class WarpIngestLayoutAdapter(LayoutAdapter):
    """Adapter for generic layout predictions emitted by Warp-Ingest."""

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        try:
            from extract_bench.inference.providers.parse.warp_ingest import ensure_nltk_corpora

            ensure_nltk_corpora()
            from warp_ingest.ingestor.markdown_exporter import render_layout_predictions
        except ImportError as exc:
            raise ValueError("warp-ingest>=2.0.1 is required for Warp-Ingest layout evaluation") from exc

        payload = inference_result.raw_output if isinstance(inference_result.raw_output, dict) else {}
        rendered = render_layout_predictions(payload, page_filter=page_filter)
        predictions = [
            self._build_prediction(prediction)
            for prediction in rendered.get("predictions", [])
            if isinstance(prediction, dict)
        ]

        return LayoutOutput(
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.UNSTRUCTURED_LAYOUT,
            image_width=self._positive_int(rendered.get("image_width"), 612),
            image_height=self._positive_int(rendered.get("image_height"), 792),
            predictions=predictions,
        )

    @staticmethod
    def _positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(round(float(value)))
        except (TypeError, ValueError):
            return default
        return max(1, parsed)

    @staticmethod
    def _build_prediction(prediction: dict[str, Any]) -> LayoutPrediction:
        content_payload = prediction.get("content")
        content: LayoutTextContent | LayoutTableContent | None = None
        if isinstance(content_payload, dict) and content_payload.get("type") == "table":
            content = LayoutTableContent(html=str(content_payload.get("html") or ""))
        elif isinstance(content_payload, dict):
            content = LayoutTextContent(text=str(content_payload.get("text") or ""))

        provider_metadata = {"order_index": prediction.get("order_index")}
        return LayoutPrediction(
            bbox=[float(value) for value in prediction.get("bbox", [])],
            score=float(prediction.get("score", 1.0)),
            label=str(prediction.get("label", "")),
            page=prediction.get("page"),
            content=content,
            provider_metadata=provider_metadata,
        )


def _build_llamaparse_granular_pages_from_payload(grounded_pages: Any) -> list[_GranularPage]:
    if not isinstance(grounded_pages, list):
        return []

    pages: list[_GranularPage] = []
    for page_payload in grounded_pages:
        if not isinstance(page_payload, dict):
            continue
        if page_payload.get("success") is False:
            continue

        page_number = page_payload.get("page_number")
        page_width = page_payload.get("page_width")
        page_height = page_payload.get("page_height")
        if not isinstance(page_number, int):
            continue
        if not isinstance(page_width, (int, float)) or page_width <= 0:
            continue
        if not isinstance(page_height, (int, float)) or page_height <= 0:
            continue

        raw_items = page_payload.get("items")
        if not isinstance(raw_items, list):
            continue

        line_units: list[_GranularTextUnit] = []
        word_units: list[_GranularTextUnit] = []
        for order_index, line_context in enumerate(_iter_llamaparse_line_contexts(raw_items)):
            line_text = line_context["text"]
            line_bbox = line_context["bbox"]
            if not line_text or line_bbox is None:
                continue

            normalized_line_bbox = _normalize_grounded_bbox(
                line_bbox,
                page_width=float(page_width),
                page_height=float(page_height),
            )
            if normalized_line_bbox is None:
                continue

            line_units.append(
                _GranularTextUnit(
                    text=line_text,
                    bbox=normalized_line_bbox,
                    order_index=order_index,
                )
            )
            word_units.extend(
                _build_llamaparse_word_units(
                    line_context,
                    page_width=float(page_width),
                    page_height=float(page_height),
                    order_index=order_index,
                )
            )

        deduped_lines = _dedupe_granular_units(line_units)
        deduped_words = _dedupe_granular_units(word_units)
        if not deduped_lines and not deduped_words:
            continue

        pages.append(
            _GranularPage(
                page_number=page_number,
                lines=deduped_lines,
                words=deduped_words,
            )
        )

    return pages


def _iter_llamaparse_line_contexts(raw_nodes: list[Any]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for raw_node in raw_nodes:
        contexts.extend(_collect_llamaparse_line_contexts(raw_node))
    return contexts


def _collect_llamaparse_line_contexts(raw_node: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_node, dict):
        return []

    contexts: list[dict[str, Any]] = []
    grounding = raw_node.get("grounding")
    if isinstance(grounding, dict):
        source_text = _resolve_llamaparse_grounding_source_text(raw_node, grounding)
        raw_lines = grounding.get("lines")
        if isinstance(raw_lines, list):
            contexts.extend(_build_llamaparse_line_context_entries(source_text, raw_lines))

        raw_rows = grounding.get("rows")
        source_rows = raw_node.get("rows")
        if isinstance(raw_rows, list) and isinstance(source_rows, list):
            contexts.extend(_build_llamaparse_table_line_context_entries(source_rows, raw_rows))

    child_items = raw_node.get("items")
    if isinstance(child_items, list):
        for child in child_items:
            contexts.extend(_collect_llamaparse_line_contexts(child))

    return contexts


def _build_llamaparse_line_context_entries(source_text: str, raw_lines: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            continue

        line_span = _coerce_span(raw_line.get("span"))
        line_bbox = raw_line.get("bbox")
        if line_span is None or not isinstance(line_bbox, dict):
            continue

        line_text = _normalize_llamaparse_grounded_text(_slice_span_text(source_text, line_span))
        if not line_text:
            continue

        entries.append(
            {
                "text": line_text,
                "bbox": line_bbox,
                "line_span": line_span,
                "raw_words": raw_line.get("words") if isinstance(raw_line.get("words"), list) else [],
                "source_text": source_text,
            }
        )

    return entries


def _build_llamaparse_table_line_context_entries(
    source_rows: list[Any],
    raw_rows: list[Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for source_row, grounding_row in zip(source_rows, raw_rows, strict=False):
        if not isinstance(source_row, list) or not isinstance(grounding_row, list):
            continue
        for source_cell, grounding_cell in zip(source_row, grounding_row, strict=False):
            if not isinstance(grounding_cell, dict):
                continue

            cell_text = _coerce_llamaparse_cell_text(source_cell)
            if not cell_text:
                continue

            cell_lines = grounding_cell.get("lines")
            if isinstance(cell_lines, list):
                entries.extend(_build_llamaparse_line_context_entries(cell_text, cell_lines))

    return entries


def _resolve_llamaparse_grounding_source_text(raw_node: dict[str, Any], grounding: dict[str, Any]) -> str:
    source_name = grounding.get("source")
    if source_name == "caption":
        source_text = raw_node.get("caption")
    elif source_name == "value":
        source_text = raw_node.get("value")
    else:
        source_text = raw_node.get("md")

    if isinstance(source_text, str) and source_text:
        return source_text

    for candidate_key in ("value", "md", "caption", "html"):
        candidate = raw_node.get(candidate_key)
        if isinstance(candidate, str) and candidate:
            return candidate

    return ""


def _build_llamaparse_word_units(
    line_context: dict[str, Any],
    *,
    page_width: float,
    page_height: float,
    order_index: int,
) -> list[_GranularTextUnit]:
    source_text = str(line_context.get("source_text") or "")
    line_span = _coerce_span(line_context.get("line_span"))
    raw_words = line_context.get("raw_words")
    if not source_text or line_span is None or not isinstance(raw_words, list):
        return []

    units: list[_GranularTextUnit] = []
    for token_start, token_end in _iter_token_spans(source_text, line_span):
        matching_word_boxes: list[dict[str, Any]] = []
        for raw_word in raw_words:
            if not isinstance(raw_word, dict):
                continue
            word_span = _coerce_span(raw_word.get("span"))
            word_bbox = raw_word.get("bbox")
            if word_span is None or not isinstance(word_bbox, dict):
                continue
            if word_span[1] <= token_start or word_span[0] >= token_end:
                continue
            matching_word_boxes.append(word_bbox)

        if not matching_word_boxes:
            continue

        word_text = _normalize_llamaparse_grounded_text(_slice_span_text(source_text, (token_start, token_end)))
        if not word_text:
            continue

        merged_bbox = _merge_llamaparse_bboxes(matching_word_boxes)
        normalized_bbox = _normalize_grounded_bbox(
            merged_bbox,
            page_width=page_width,
            page_height=page_height,
        )
        if normalized_bbox is None:
            continue

        units.append(
            _GranularTextUnit(
                text=word_text,
                bbox=normalized_bbox,
                order_index=order_index,
            )
        )

    return units


def _coerce_span(raw_span: Any) -> tuple[int, int] | None:
    if not isinstance(raw_span, list | tuple) or len(raw_span) != 2:
        return None
    try:
        start = int(raw_span[0])
        end = int(raw_span[1])
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return (start, end)


def _slice_span_text(source_text: str, span: tuple[int, int]) -> str:
    start = max(span[0], 0)
    source_bytes = source_text.encode("utf-8")
    end = min(span[1], len(source_bytes))
    if end <= start:
        return ""
    return source_bytes[start:end].decode("utf-8", errors="ignore")


def _normalize_llamaparse_grounded_text(text: str) -> str:
    normalized = text.replace("<br/>", "\n").replace("<br />", "\n")
    if "<" in normalized and ">" in normalized:
        normalized = extract_text_from_html(normalized)
    return normalized.strip()


def _coerce_llamaparse_cell_text(source_cell: Any) -> str:
    if isinstance(source_cell, str):
        return source_cell
    if isinstance(source_cell, dict):
        for key in ("value", "md", "text", "html"):
            value = source_cell.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _iter_token_spans(source_text: str, line_span: tuple[int, int]) -> list[tuple[int, int]]:
    line_text = _slice_span_text(source_text, line_span)
    return [
        (
            line_span[0] + len(line_text[: match.start()].encode("utf-8")),
            line_span[0] + len(line_text[: match.end()].encode("utf-8")),
        )
        for match in re.finditer(r"\S+", line_text, flags=re.UNICODE)
    ]


def _merge_llamaparse_bboxes(raw_bboxes: list[dict[str, Any]]) -> dict[str, float]:
    x1 = min(float(bbox.get("x", 0.0)) for bbox in raw_bboxes)
    y1 = min(float(bbox.get("y", 0.0)) for bbox in raw_bboxes)
    x2 = max(float(bbox.get("x", 0.0)) + float(bbox.get("w", 0.0)) for bbox in raw_bboxes)
    y2 = max(float(bbox.get("y", 0.0)) + float(bbox.get("h", 0.0)) for bbox in raw_bboxes)
    merged = {"x": x1, "y": y1, "w": max(0.0, x2 - x1), "h": max(0.0, y2 - y1)}
    r_values = [bbox.get("r") for bbox in raw_bboxes]
    if r_values and all(isinstance(value, int | float) for value in r_values):
        first = float(cast(int | float, r_values[0]))
        if all(abs(float(cast(int | float, value)) - first) <= 1e-6 for value in r_values):
            merged["r"] = first
    return merged


def _dedupe_granular_units(units: list[_GranularTextUnit]) -> list[_GranularTextUnit]:
    deduped: list[_GranularTextUnit] = []
    seen: set[tuple[str, float, float, float, float]] = set()
    for unit in units:
        key = (
            unit.text,
            round(unit.bbox.x, 6),
            round(unit.bbox.y, 6),
            round(unit.bbox.w, 6),
            round(unit.bbox.h, 6),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(unit)
    return deduped


def _normalize_grounded_bbox(
    bbox_payload: Any,
    *,
    page_width: float,
    page_height: float,
) -> _GranularSegment | None:
    if not isinstance(bbox_payload, dict):
        return None

    x = bbox_payload.get("x")
    y = bbox_payload.get("y")
    w = bbox_payload.get("w")
    h = bbox_payload.get("h")
    if not all(isinstance(value, (int, float)) for value in (x, y, w, h)):
        return None
    x_num = float(cast(int | float, x))
    y_num = float(cast(int | float, y))
    w_num = float(cast(int | float, w))
    h_num = float(cast(int | float, h))
    raw_r = bbox_payload.get("r")

    return _GranularSegment(
        x=x_num / page_width,
        y=y_num / page_height,
        w=w_num / page_width,
        h=h_num / page_height,
        r=float(raw_r) if isinstance(raw_r, int | float) else None,
    )


def _build_textract_granular_pages(textract_response: dict[str, Any]) -> list[_GranularPage]:
    blocks = textract_response.get("Blocks")
    if not isinstance(blocks, list):
        return []

    pages: dict[int, _GranularPage] = {}
    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue

        block_type = block.get("BlockType")
        if block_type not in {"LINE", "WORD"}:
            continue

        page_number = int(block.get("Page", 1))
        geometry = block.get("Geometry", {})
        bbox = geometry.get("BoundingBox", {}) if isinstance(geometry, dict) else {}
        left = float(bbox.get("Left", 0.0))
        top = float(bbox.get("Top", 0.0))
        width = float(bbox.get("Width", 0.0))
        height = float(bbox.get("Height", 0.0))
        text = str(block.get("Text", "") or "")
        if not text:
            continue

        unit = _GranularTextUnit(
            text=text,
            bbox=_GranularSegment(x=left, y=top, w=width, h=height),
            order_index=block_index,
        )

        page = pages.setdefault(page_number, _GranularPage(page_number=page_number, lines=[], words=[]))
        if block_type == "LINE":
            page.lines.append(unit)
        else:
            page.words.append(unit)

    return [pages[page_number] for page_number in sorted(pages)]


def _build_azure_di_granular_pages(raw_output: dict[str, Any]) -> list[_GranularPage]:
    raw_pages = raw_output.get("pages")
    if not isinstance(raw_pages, list):
        return []

    granular_pages: list[_GranularPage] = []
    for page_data in raw_pages:
        if not isinstance(page_data, dict):
            continue

        page_number = int(page_data.get("page_number", 1))
        page_width = float(page_data.get("width", 1.0))
        page_height = float(page_data.get("height", 1.0))

        line_units = _build_azure_di_granular_units(
            page_data.get("lines"),
            page_width=page_width,
            page_height=page_height,
            text_key="content",
        )
        word_units = _build_azure_di_granular_units(
            page_data.get("words"),
            page_width=page_width,
            page_height=page_height,
            text_key="content",
        )

        if not line_units and not word_units:
            continue

        granular_pages.append(
            _GranularPage(
                page_number=page_number,
                lines=line_units,
                words=word_units,
            )
        )

    return granular_pages


def _build_azure_di_granular_units(
    raw_units: Any,
    *,
    page_width: float,
    page_height: float,
    text_key: str,
) -> list[_GranularTextUnit]:
    if not isinstance(raw_units, list):
        return []

    units: list[_GranularTextUnit] = []
    for index, raw_unit in enumerate(raw_units):
        if not isinstance(raw_unit, dict):
            continue

        polygon = raw_unit.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 8:
            continue

        text = str(raw_unit.get(text_key, "") or "")
        if not text:
            continue

        box = polygon_to_literal_xywh_r(
            polygon,
            page_width=page_width,
            page_height=page_height,
        )
        if box is None:
            continue
        units.append(
            _GranularTextUnit(
                text=text,
                bbox=_GranularSegment(x=box.x, y=box.y, w=box.w, h=box.h, r=box.r),
                order_index=index,
            )
        )

    return units


@register_layout_adapter("chunkr", priority=90)
class ChunkrLayoutAdapter(LayoutAdapter):
    """Adapter for Chunkr raw parse output (`output.chunks[].segments[]`)."""

    _label_adapter = ChunkrLayoutDetLabelAdapter()

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        raw_output = inference_result.raw_output
        if not isinstance(raw_output, dict):
            return False
        output = raw_output.get("output")
        if not isinstance(output, dict):
            return False
        return isinstance(output.get("chunks"), list)

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        raw_output = inference_result.raw_output
        if not isinstance(raw_output, dict):
            raise ValueError("Chunkr adapter requires dict raw_output")

        chunks = raw_output.get("output", {}).get("chunks", [])
        if not isinstance(chunks, list):
            raise ValueError("Chunkr adapter requires raw_output.output.chunks")

        inferred_page_number = _infer_page_number_from_example_id(inference_result.request.example_id)
        predictions: list[LayoutPrediction] = []
        output_width = 0
        output_height = 0

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            segments = chunk.get("segments")
            if not isinstance(segments, list):
                continue

            for segment in segments:
                if not isinstance(segment, dict):
                    continue

                page_number = int(segment.get("page_number", 1))
                # Chunkr single-page inference artifacts frequently report page_number=1,
                # while benchmark example IDs keep original doc page (e.g., "..._page136_...").
                # Use inferred page for this case so cross-evaluation page filtering works.
                if inferred_page_number is not None and page_number == 1:
                    page_number = inferred_page_number
                if page_filter is not None and page_number != page_filter:
                    continue

                segment_label = segment.get("segment_type")
                if not isinstance(segment_label, str):
                    continue

                bbox_data = segment.get("bbox") or {}
                left = float(bbox_data.get("left", 0.0))
                top = float(bbox_data.get("top", 0.0))
                width = float(bbox_data.get("width", 0.0))
                height = float(bbox_data.get("height", 0.0))
                bbox_xyxy = [left, top, left + width, top + height]

                if self._label_adapter.to_canonical(segment_label) is None:
                    raise UnknownRawLayoutLabelError(f"Unknown Chunkr raw layout label '{segment_label}'")

                if output_width == 0:
                    output_width = int(segment.get("page_width", 0))
                    output_height = int(segment.get("page_height", 0))

                html = segment.get("html")
                text = segment.get("content") or segment.get("text")
                content = None
                is_table_segment = segment_label.strip().lower() == "table"
                if is_table_segment:
                    if isinstance(html, str) and html:
                        content = LayoutTableContent(html=html)
                    elif isinstance(text, str) and text:
                        content = LayoutTextContent(text=text)  # type: ignore[assignment]
                else:
                    if isinstance(text, str) and text:
                        content = LayoutTextContent(text=text)  # type: ignore[assignment]
                    elif isinstance(html, str) and html:
                        # Fallback when provider omits plain text but includes HTML.
                        content = LayoutTextContent(text=html)  # type: ignore[assignment]

                predictions.append(
                    LayoutPrediction(
                        bbox=bbox_xyxy,
                        score=float(segment.get("confidence", 1.0)),
                        label=segment_label,
                        page=page_number,
                        content=content,
                        provider_metadata={
                            "segment_id": segment.get("segment_id"),
                            "order_index": len(predictions),
                        },
                    )
                )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.CHUNKR,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("dots_ocr_parse", priority=90)
class DotsOcrLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from dots.ocr ParseOutput.layout_pages.

    This enables cross-evaluation: a single dots.ocr PARSE pipeline can be
    evaluated against both parse and layout detection datasets, following the
    same pattern as LlamaParse's ``ours_agentic`` pipeline.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Distinguish from LlamaParse by checking raw_output for dots.ocr markers
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            prompt_mode = raw_output.get("prompt_mode", "")
            return isinstance(prompt_mode, str) and prompt_mode.startswith("prompt_layout")
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        # Handle synthetic LayoutOutput results (e.g. from cross-eval runner)
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("DotsOcrLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("DotsOcrLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_dots_ocr_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.DOTS_OCR,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


def _build_docling_parse_content(item_type: str, text: str) -> LayoutTextContent | LayoutTableContent | None:
    """Build content object for Docling parse-derived layout items."""
    if not text:
        return None
    if item_type == "table":
        return LayoutTableContent(html=text)
    return LayoutTextContent(text=text)


@register_layout_adapter("docling_parse", priority=90)
class DoclingParseLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Docling ParseOutput.layout_pages."""

    def __init__(self) -> None:
        self._current_layout_pages: list[Any] | None = None

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        return isinstance(raw_output, dict) and isinstance(raw_output.get("docling_document"), dict)

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("DoclingParseLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("DoclingParseLayoutAdapter requires non-empty layout_pages")

        selected_pages = [lp for lp in layout_pages if page_filter is None or lp.page_number == page_filter]
        reference_page = selected_pages[0] if selected_pages else layout_pages[0]
        output_width = int(reference_page.width or 1)
        output_height = int(reference_page.height or 1)
        self._current_layout_pages = layout_pages

        predictions: list[LayoutPrediction] = []
        markdown_parts: list[str] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)
            if lp.md:
                markdown_parts.append(lp.md)

            for item_idx, item in enumerate(lp.items):
                segments = item.layout_segments or ([item.bbox] if item.bbox is not None else [])
                for segment_idx, seg in enumerate(segments):
                    label = seg.label or item.type or "text"
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=_build_docling_parse_content(item.type, item.value),
                            provider_metadata={
                                "order_index": len(predictions),
                                "item_index": item_idx,
                                "segment_index": segment_idx,
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.DOCLING_PARSE_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
            markdown="\n\n".join(markdown_parts),
        )

    def to_attribution_blocks(
        self,
        layout_output: LayoutOutput,
        *,
        page_number: int,
        test_case: TestCase | None = None,
    ) -> list[PredBlock]:
        del test_case
        if self._current_layout_pages is None:
            return super().to_attribution_blocks(layout_output, page_number=page_number, test_case=None)

        layout_pages = self._current_layout_pages
        page = next((lp for lp in layout_pages if lp.page_number == page_number), None)
        if page is None:
            return []

        blocks: list[PredBlock] = []
        for item_index, item in enumerate(page.items):
            segments = item.layout_segments
            if not segments:
                continue

            for seg in segments:
                label = seg.label or item.type or "unknown"
                block_type = item.type or "text"
                if (item.type or "").strip().lower() == "table":
                    raw_text = extract_text_from_html(item.value)
                else:
                    raw_text = item.value or ""
                    if (
                        isinstance(seg.start_index, int)
                        and isinstance(seg.end_index, int)
                        and seg.end_index >= seg.start_index
                    ):
                        raw_text = raw_text[seg.start_index : seg.end_index + 1]

                normalized_text = normalize_attribution_text(raw_text)
                blocks.append(
                    PredBlock(
                        bbox_xyxy=[seg.x, seg.y, seg.x + seg.w, seg.y + seg.h],
                        block_type=block_type,
                        label=label,
                        text=raw_text,
                        normalized_text=normalized_text,
                        tokens=tokenize(normalized_text),
                        order_index=item_index,
                    )
                )

        return blocks


@register_layout_adapter("qwen3vl_layout", priority=90)
class Qwen3VLLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Qwen3-VL ParseOutput.layout_pages.

    Enables cross-evaluation: the ``qwen3vl_layout`` PARSE pipeline can be
    evaluated against layout detection datasets using the bboxes from
    the structured JSON output.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            return "items" in raw_output and "raw_content" in raw_output
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("Qwen3VLLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("Qwen3VLLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    str_label = seg.label or item.type or "Text"

                    # Convert string label to integer label for Qwen3VL evaluator
                    # Map canonical-style "Page-header" → "page_header" for lookup
                    lookup_key = str_label.lower().replace("-", "_")
                    qwen_enum = QWEN3VL_STR_TO_LABEL.get(lookup_key)
                    int_label = str(int(qwen_enum)) if qwen_enum is not None else str_label

                    # Convert normalized [0,1] xywh -> pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_dots_ocr_content(str_label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=int_label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.QWEN3_VL_8B,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


def _parse_with_layout_to_layout_output(
    inference_result: InferenceResult,
    *,
    model: LayoutDetectionModel,
    page_filter: int | None = None,
) -> LayoutOutput:
    """Shared conversion for LLM parse_with_layout adapters (Google/OpenAI/Anthropic)."""
    # Handle LayoutOutput (e.g. from multi-task re-evaluation)
    if isinstance(inference_result.output, LayoutOutput):
        if page_filter is None:
            return inference_result.output
        filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
        return inference_result.output.model_copy(update={"predictions": filtered})

    if not isinstance(inference_result.output, ParseOutput):
        out_type = type(inference_result.output).__name__
        raise ValueError(f"parse_with_layout adapter requires ParseOutput or LayoutOutput, got {out_type}")

    layout_pages = inference_result.output.layout_pages
    if not layout_pages:
        raise ValueError("parse_with_layout adapter requires non-empty layout_pages")

    first_page = layout_pages[0]
    output_width = int(first_page.width or 1)
    output_height = int(first_page.height or 1)

    predictions: list[LayoutPrediction] = []

    for lp in layout_pages:
        page_number = lp.page_number
        if page_filter is not None and page_number != page_filter:
            continue

        page_w = float(lp.width or output_width)
        page_h = float(lp.height or output_height)

        for item in lp.items:
            for seg in item.layout_segments:
                str_label = seg.label or item.type or "Text"

                lookup_key = str_label.lower().replace("-", "_")
                qwen_enum = QWEN3VL_STR_TO_LABEL.get(lookup_key)
                int_label = str(int(qwen_enum)) if qwen_enum is not None else str_label

                # Convert normalized [0,1] xywh -> pixel xyxy
                x1 = seg.x * page_w
                y1 = seg.y * page_h
                x2 = (seg.x + seg.w) * page_w
                y2 = (seg.y + seg.h) * page_h

                content = _build_dots_ocr_content(str_label, item.value)

                predictions.append(
                    LayoutPrediction(
                        bbox=[x1, y1, x2, y2],
                        score=float(seg.confidence or 1.0),
                        label=int_label,
                        page=page_number,
                        content=content,
                        provider_metadata={
                            "order_index": len(predictions),
                        },
                    )
                )

    return LayoutOutput(
        task_type="layout_detection",
        example_id=inference_result.request.example_id,
        pipeline_name=inference_result.pipeline_name,
        model=model,
        image_width=max(output_width, 1),
        image_height=max(output_height, 1),
        predictions=predictions,
    )


@register_layout_adapter("google", priority=90)
class GoogleLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Google Gemini ParseOutput.layout_pages.

    Enables cross-evaluation: the ``gemini_*_parse_with_layout`` PARSE pipelines
    can be evaluated against layout detection datasets using the bboxes from
    the div-wrapped output.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            model = raw_output.get("model", "")
            return (
                raw_output.get("mode") == "parse_with_layout" and isinstance(model, str) and model.startswith("gemini")
            )
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        return _parse_with_layout_to_layout_output(
            inference_result,
            model=LayoutDetectionModel.GEMINI_LAYOUT,
            page_filter=page_filter,
        )


@register_layout_adapter("gemma4", priority=90)
class Gemma4LayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Gemma 4 ParseOutput.layout_pages.

    Enables cross-evaluation: the ``gemma4_*_vllm_with_layout`` PARSE pipelines
    can be evaluated against layout detection datasets using the bboxes from
    the structured layout output.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            prompt_mode = raw_output.get("prompt_mode", "")
            config = raw_output.get("_config", {})
            model = config.get("model", "") if isinstance(config, dict) else ""
            return prompt_mode == "layout" and isinstance(model, str) and model.startswith("gemma")
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        return _parse_with_layout_to_layout_output(
            inference_result,
            model=LayoutDetectionModel.GEMMA4_LAYOUT,
            page_filter=page_filter,
        )


@register_layout_adapter("openai", priority=90)
class OpenAILayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from OpenAI ParseOutput.layout_pages.

    Enables cross-evaluation: the ``openai_*_parse_with_layout`` PARSE pipelines
    can be evaluated against layout detection datasets using the bboxes from
    the div-wrapped output.
    """

    # OpenAI model prefixes (gpt-*, o3-*, o4-*, etc.)
    _OPENAI_PREFIXES = ("gpt", "o3", "o4")

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            model = raw_output.get("model", "")
            return (
                raw_output.get("mode") == "parse_with_layout"
                and isinstance(model, str)
                and any(model.startswith(p) for p in cls._OPENAI_PREFIXES)
            )
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        return _parse_with_layout_to_layout_output(
            inference_result,
            model=LayoutDetectionModel.OPENAI_LAYOUT,
            page_filter=page_filter,
        )


@register_layout_adapter("anthropic_haiku", "anthropic", priority=90)
class AnthropicLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Anthropic ParseOutput.layout_pages.

    Enables cross-evaluation: Anthropic ``parse_with_layout`` PARSE pipelines
    can be evaluated against layout detection datasets using the bboxes from
    the div-wrapped output.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            model = raw_output.get("model", "")
            return (
                raw_output.get("mode") == "parse_with_layout" and isinstance(model, str) and model.startswith("claude")
            )
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        return _parse_with_layout_to_layout_output(
            inference_result,
            model=LayoutDetectionModel.ANTHROPIC_LAYOUT,
            page_filter=page_filter,
        )


@register_layout_adapter("reducto", priority=90)
class ReductoLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Reducto ParseOutput.layout_pages.

    Enables cross-evaluation: the ``reducto`` PARSE pipeline can be evaluated
    against layout detection datasets using the block-level bboxes from the
    Reducto API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Reducto by checking raw_output for Reducto-specific markers
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and "ocr_system" in config
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        # Handle synthetic LayoutOutput results (e.g. from cross-eval runner)
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("ReductoLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("ReductoLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.REDUCTO_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("oi_parser", priority=90)
class OIParserLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from oi-parser ParseOutput.layout_pages.

    Enables cross-evaluation: the ``oi_parser`` PARSE pipeline is scored against the
    layout-detection dataset using the block bboxes the API returns in
    ``output.layout_pages`` (absolute-pixel xywh + a raw label per item; label casing
    is canonicalized to Canonical17 by ``OIParserLabelMapper``).
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        return isinstance(inference_result.output, ParseOutput) and bool(inference_result.output.layout_pages)

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        # Handle synthetic LayoutOutput results (e.g. from cross-eval re-runs).
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("OIParserLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            # Doc produced markdown but no layout segments (e.g. a huge table whose
            # layout-detect response was truncated). Emit an EMPTY LayoutOutput so the
            # doc is still COUNTED (scores ~0 on layout) rather than erroring and being
            # dropped — dropping would inflate the average (anti-cheat concern).
            return LayoutOutput(
                task_type="layout_detection",
                example_id=inference_result.request.example_id,
                pipeline_name=inference_result.pipeline_name,
                model=LayoutDetectionModel.OI_PARSER_LAYOUT,
                image_width=1,
                image_height=1,
                predictions=[],
            )

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            for item in lp.items:
                for seg in item.layout_segments:
                    # Emit the raw API label; OIParserLabelMapper canonicalizes casing.
                    label = seg.label or item.type or "Text"

                    # oi-parser emits ABSOLUTE pixel xywh (in the page's width/height
                    # pixel space), so map straight to pixel xyxy — no [0,1] scaling.
                    x1 = seg.x
                    y1 = seg.y
                    x2 = seg.x + seg.w
                    y2 = seg.y + seg.h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.OI_PARSER_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("databricks_ai_parse", priority=90)
class DatabricksAiParseLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Databricks ai_parse_document
    ParseOutput.layout_pages (normalized [0,1] xywh + Canonical17 labels)."""

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("DatabricksAiParseLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("DatabricksAiParseLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []
        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence) if seg.confidence is not None else 1.0,
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.DATABRICKS_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("textract", priority=89)
class TextractLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Textract ParseOutput.layout_pages.

    Enables cross-evaluation: the ``textract`` PARSE pipeline can be evaluated
    against layout detection datasets using the LAYOUT_* block bboxes from the
    Textract API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Textract by checking raw_output for textract_response key
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            return "textract_response" in raw_output
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("TextractLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("TextractLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.TEXTRACT_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )

    def to_granular_pages(self, inference_result: InferenceResult) -> list[_GranularPage]:
        raw_output = inference_result.raw_output
        if not isinstance(raw_output, dict):
            return []
        textract_response = raw_output.get("textract_response")
        if not isinstance(textract_response, dict):
            return []
        return _build_textract_granular_pages(textract_response)


@register_layout_adapter("landingai", priority=89)
class LandingAILayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from LandingAI ParseOutput.layout_pages.

    Enables cross-evaluation: the ``landingai`` PARSE pipeline can be evaluated
    against layout detection datasets using the chunk-level bboxes from the
    LandingAI ADE API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify LandingAI by checking raw_output for grounding + chunks keys
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            return "grounding" in raw_output and "chunks" in raw_output
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("LandingAILayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("LandingAILayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.LANDINGAI_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("extend_parse", priority=89)
class ExtendLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Extend ParseOutput.layout_pages.

    Enables cross-evaluation: the ``extend_parse`` PARSE pipeline can be evaluated
    against layout detection datasets using the block-level bboxes from the
    Extend AI API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Extend by checking raw_output for _extend_metadata key
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            return "_extend_metadata" in raw_output
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("ExtendLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("ExtendLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.EXTEND_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("azure_document_intelligence", priority=89)
class AzureDILayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Azure DI ParseOutput.layout_pages.

    Enables cross-evaluation: the ``azure_document_intelligence`` PARSE pipeline
    can be evaluated against layout detection datasets using the paragraph/table/figure
    bboxes from the Azure Document Intelligence API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Azure DI by checking raw_output for _config with model_id key
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and "model_id" in config
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("AzureDILayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("AzureDILayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)
                    attributes = dict(item.attributes)
                    if label in {"Checkbox-Selected", "Checkbox-Unselected"}:
                        attributes["scope"] = "mark"

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            attributes=attributes,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.AZURE_DI_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )

    def to_granular_pages(self, inference_result: InferenceResult) -> list[_GranularPage]:
        raw_output = inference_result.raw_output
        if not isinstance(raw_output, dict):
            return []
        return _build_azure_di_granular_pages(raw_output)


@register_layout_adapter("google_docai", priority=89)
class GoogleDocAILayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Google DocAI ParseOutput.layout_pages.

    Enables cross-evaluation: the ``google_docai`` PARSE pipeline can be evaluated
    against layout detection datasets using the paragraph/table bboxes from the
    Google Document AI API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Google DocAI by checking raw_output for mode key and _config
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return (
                isinstance(config, dict)
                and "processor_id" in config
                and raw_output.get("mode")
                in (
                    "ocr",
                    "layout_parser",
                )
            )
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("GoogleDocAILayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("GoogleDocAILayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.GOOGLE_DOCAI_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("unstructured", priority=89)
class UnstructuredLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Unstructured ParseOutput.layout_pages.

    Enables cross-evaluation: the ``unstructured`` PARSE pipeline (hi_res strategy)
    can be evaluated against layout detection datasets using the element-level bboxes
    from the Unstructured API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Unstructured by checking raw_output for _config with strategy key
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and "strategy" in config
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("UnstructuredLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("UnstructuredLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.UNSTRUCTURED_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


def _build_vendor_content(label: str, text: str) -> LayoutTextContent | LayoutTableContent | None:
    """Build content object from vendor layout element."""
    if not text:
        return None
    normalized = label.strip().lower()
    if normalized == "table":
        return LayoutTableContent(html=text)
    if normalized == "picture":
        return None
    return LayoutTextContent(text=text)


def _build_dots_ocr_content(label: str, text: str) -> LayoutTextContent | LayoutTableContent | None:
    """Build content object from dots.ocr layout element."""
    if not text:
        return None
    normalized = label.strip().lower()
    if normalized == "table":
        return LayoutTableContent(html=text)
    if normalized == "picture":
        return None
    return LayoutTextContent(text=text)


@register_layout_adapter("deepseekocr2", priority=90)
class DeepSeekOCR2LayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from DeepSeek-OCR-2 ParseOutput.layout_pages.

    Enables cross-evaluation: the ``deepseekocr2_vllm`` PARSE pipeline can be
    evaluated against layout detection datasets using the grounding bboxes from
    the model output.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and "deepseek" in str(config.get("server_url", "")).lower()
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("DeepSeekOCR2LayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("DeepSeekOCR2LayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.DEEPSEEK_OCR2_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("chandra2", priority=90)
class Chandra2LayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Chandra OCR 2 ParseOutput.layout_pages.

    Enables cross-evaluation: the ``chandra2_vllm`` / ``chandra2_sdk`` PARSE pipelines
    can be evaluated against layout detection datasets using the native bboxes from
    the model output. Chandra OCR 2 has 19 fine-grained labels mapping to Canonical17.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and "chandra2" in str(config.get("server_url", "")).lower()
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("Chandra2LayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("Chandra2LayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.CHANDRA2_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("qfocr", priority=90)
class QfOcrLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Qianfan-OCR ParseOutput.layout_pages.

    Enables cross-evaluation: the ``qfocr_vllm_thinking`` PARSE pipeline can be
    evaluated against layout detection datasets using the Layout-as-Thought bboxes
    parsed from the model's ``<think>`` block.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and config.get("thinking") is True
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("QfOcrLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("QfOcrLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.QFOCR_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


def _infer_page_number_from_example_id(example_id: str) -> int | None:
    match = re.search(r"_page(\d+)(?:_|$)", example_id)
    if not match:
        return None
    page_token = int(match.group(1))
    # Dataset IDs are mixed:
    # - most use 1-indexed page tokens (e.g. page136 -> page 136)
    # - some use page0 for first page.
    return page_token if page_token > 0 else 1


def _resolve_llamaparse_pages(
    inference_result: InferenceResult,
    *,
    include_bbox_segment_fallback: bool = True,
) -> list[dict[str, Any]]:
    raw_output = inference_result.raw_output
    if isinstance(raw_output, dict):
        raw_pages = raw_output.get("pages")
        if isinstance(raw_pages, list):
            return [page for page in raw_pages if isinstance(page, dict)]

        # CLI2 local provider stores items under v2_items instead of pages.
        # Normalize into the legacy page format so parse_pred_blocks can
        # access layoutAwareBbox segments for per-cell attribution.
        if "v2_items" in raw_output:
            try:
                return build_pages_from_cli2_raw_payload(
                    raw_payload=raw_output,
                    output_tables_as_markdown=False,
                )
            except (ValueError, TypeError):
                pass

        # V2 SDK API responses have items/text/metadata expansions but no
        # pre-normalized pages list.  Normalize them so parse_pred_blocks
        # can access layoutAwareBbox segments for per-cell attribution.
        if "items" in raw_output and "job" in raw_output:
            try:
                return build_pages_from_sdk_response_payload(
                    raw_payload=raw_output,
                    output_tables_as_markdown=False,
                )
            except (ValueError, TypeError):
                pass

    if isinstance(inference_result.output, ParseOutput):
        if len(inference_result.output.layout_pages) > 0:
            return layout_pages_to_legacy_pages_payload(
                inference_result.output.layout_pages,
                include_bbox_segment_fallback=include_bbox_segment_fallback,
            )

    return []


def _find_page_payload(
    pages: list[dict[str, Any]],
    page_number: int,
) -> dict[str, Any] | None:
    for page_index, page in enumerate(pages):
        page_raw = page.get("page")
        page_value = page_raw if isinstance(page_raw, int) and page_raw > 0 else page_index + 1
        if page_value == page_number:
            return page

    return None


@register_layout_adapter("datalab", priority=90)
class DatalabLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Datalab ParseOutput.layout_pages.

    Enables cross-evaluation: the ``datalab`` PARSE pipeline can be evaluated
    against layout detection datasets using block-level bboxes from the
    Datalab JSON output (powered by Marker/Surya).
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Datalab by checking raw_output for Datalab-specific markers
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            # Datalab v0.3.0 returns parse_quality_score in raw_output
            if "parse_quality_score" in raw_output:
                return True
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and "mode" in config and "ocr_system" not in config
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("DatalabLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("DatalabLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh -> pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.DATALAB_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("qwen3_5", priority=90)
class Qwen35LayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Qwen3.5 ParseOutput.layout_pages.

    Enables cross-evaluation: the ``qwen3_5_4b_vllm`` PARSE pipeline can be
    evaluated against layout detection datasets using the bboxes from the
    merged layout+content JSON output.

    Bboxes use normalized 0-1000 coordinates (divided by 1000 to [0,1] in the
    provider, then multiplied by page pixel dimensions here).
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            if isinstance(config, dict):
                model = config.get("model", "")
                return isinstance(model, str) and (model.startswith("qwen3.5") or model.startswith("qwen3.6"))
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("Qwen35LayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("Qwen35LayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh -> pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_dots_ocr_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.QWEN3_5_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("mineru25", priority=90)
class MinerU25LayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from MinerU 2.5 ParseOutput.layout_pages.

    Enables cross-evaluation: the ``mineru25_vllm`` PARSE pipeline can be
    evaluated against layout detection datasets using the native bboxes from
    the model's two-step extraction (already in normalized [0,1] coordinates).
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and "mineru25" in str(config.get("server_url", "")).lower()
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("MinerU25LayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("MinerU25LayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.MINERU25_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("nemotron_ocr_v2", priority=90)
class NemotronOcrV2LayoutAdapter(LayoutAdapter):
    """Expose nemotron_ocr_v2 predictions to granular layout metrics.

    The layout server returns per-page predictions in normalized [0,1] coords as
    ``{text, confidence, left, upper, right, lower}``. ``upper``/``lower`` may
    use either axis convention so we take min/max for the y bounds.

    The Nemotron SDK only returns line-level chunks (it ignores ``merge_level``
    in practice), so we split each chunk's text on whitespace and distribute
    the line bbox across words proportionally to character width — the standard
    fallback for line-only OCR systems benchmarked on word-level GT.
    """

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        raise NotImplementedError(
            "nemotron_ocr_v2 produces text-only predictions; region-level layout detection is not supported."
        )

    def to_granular_pages(self, inference_result: InferenceResult) -> list[_GranularPage]:
        raw_output = inference_result.raw_output
        if not isinstance(raw_output, dict):
            return []
        raw_pages = raw_output.get("pages")
        if not isinstance(raw_pages, list):
            return []

        granular_pages: list[_GranularPage] = []
        for page_index, page_data in enumerate(raw_pages):
            if not isinstance(page_data, dict):
                continue
            preds = page_data.get("predictions")
            if not isinstance(preds, list):
                continue

            line_units: list[_GranularTextUnit] = []
            word_units: list[_GranularTextUnit] = []
            line_order = 0
            word_order = 0
            for pred in preds:
                if not isinstance(pred, dict):
                    continue
                text = str(pred.get("text", "") or "")
                if not text.strip():
                    continue
                left = float(pred.get("left", 0.0))
                right = float(pred.get("right", 0.0))
                upper = float(pred.get("upper", 0.0))
                lower = float(pred.get("lower", 0.0))
                x_min, x_max = min(left, right), max(left, right)
                y_min, y_max = min(upper, lower), max(upper, lower)
                line_w = max(x_max - x_min, 0.0)
                line_h = max(y_max - y_min, 0.0)

                line_units.append(
                    _GranularTextUnit(
                        text=text,
                        bbox=_GranularSegment(x=x_min, y=y_min, w=line_w, h=line_h),
                        order_index=line_order,
                    )
                )
                line_order += 1

                tokens = text.split()
                if not tokens or line_w <= 0:
                    continue
                total_chars = sum(len(t) for t in tokens)
                if total_chars <= 0:
                    continue
                # gap chars: one space between consecutive tokens
                total_units = total_chars + max(len(tokens) - 1, 0)
                cursor = 0
                for tok in tokens:
                    tok_len = len(tok)
                    start_frac = cursor / total_units
                    end_frac = (cursor + tok_len) / total_units
                    word_units.append(
                        _GranularTextUnit(
                            text=tok,
                            bbox=_GranularSegment(
                                x=x_min + line_w * start_frac,
                                y=y_min,
                                w=line_w * (end_frac - start_frac),
                                h=line_h,
                            ),
                            order_index=word_order,
                        )
                    )
                    word_order += 1
                    cursor += tok_len + 1  # +1 for the gap

            if not line_units and not word_units:
                continue
            granular_pages.append(
                _GranularPage(
                    page_number=page_index + 1,
                    lines=line_units,
                    words=word_units,
                )
            )

        return granular_pages


# Ported from ParseBench PR run-llama/ParseBench#49 (the upstream KDL submission).
# In ParseBench the layout-detection eval runs through this adapter; in
# the internal bench the `layout_element_rule_pass_rate` metric instead reads
# `ParseOutput.layout_pages` directly (see metrics/parse/layout_detection.py),
# so the equivalent [0,1]->pixel + label fix already lives in the provider's
# normalize(). This adapter is what the attribution/analysis path uses, kept in
# sync with the provider (same SCALE, same Chart/Flowchart -> Picture mapping
# that upstream omitted).
_NANO_LAYOUT_LABEL_TO_CANONICAL = {
    "Chart": "Picture",
    "Flowchart": "Picture",
}


@register_layout_adapter("kdl_frontier_nano", priority=90)
class KdlFrontierNanoLayoutAdapter(LayoutAdapter):
    """Extract LayoutOutput from the kdl_frontier_nano provider's
    ParseOutput.layout_pages.

    The provider emits per-region elements with normalized [0,1] bboxes (no
    page pixel dims). Coordinates are scaled to a consistent SCALE so the
    layout metric's normalize_bbox_xyxy(image_width/height) recovers the
    original [0,1] space.
    """

    _SCALE = 1000

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        out = inference_result.output
        return isinstance(out, ParseOutput) and bool(out.layout_pages)

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})
        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("KdlFrontierNanoLayoutAdapter requires ParseOutput or LayoutOutput")

        scale = self._SCALE
        predictions: list[LayoutPrediction] = []
        for lp in inference_result.output.layout_pages:
            if page_filter is not None and lp.page_number != page_filter:
                continue
            for item in lp.items:
                segs = item.layout_segments or ([item.bbox] if item.bbox else [])
                for seg in segs:
                    if seg is None:
                        continue
                    raw_label = seg.label or item.type or "Text"
                    label = _NANO_LAYOUT_LABEL_TO_CANONICAL.get(str(raw_label), str(raw_label))
                    x1, y1 = seg.x * scale, seg.y * scale
                    x2, y2 = (seg.x + seg.w) * scale, (seg.y + seg.h) * scale
                    text = item.md or item.value or ""
                    content = _build_docling_parse_content("table" if str(label).lower() == "table" else "text", text)
                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=1.0,
                            label=str(label),
                            page=lp.page_number,
                            content=content,
                            provider_metadata={"order_index": len(predictions)},
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.KDL_FRONTIER_NANO_LAYOUT,
            image_width=scale,
            image_height=scale,
            predictions=predictions,
        )


@register_layout_adapter("pulse", priority=90)
class PulseLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Pulse ParseOutput.layout_pages.

    Enables cross-evaluation: the ``pulse`` PARSE pipeline can be evaluated
    against layout detection datasets using the bounding_boxes from the
    Pulse API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            return "bounding_boxes" in raw_output and "extraction_id" in raw_output
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("PulseLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("PulseLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.PULSE_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("infinity_parser2", priority=90)
class InfinityParser2LayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from InfinityParser2 ParseOutput.layout_pages.

    Enables cross-evaluation: the ``infinity_parser2`` PARSE pipeline can be
    evaluated against layout detection datasets using the native bboxes from
    the model output.

    InfinityParser2 stores bboxes in pixel coordinates (page_width x page_height),
    unlike Chandra2 which stores them in normalized [0,1] space. The adapter
    converts pixel bboxes to absolute coordinates before building LayoutOutput.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            if not isinstance(config, dict) or config.get("backend") != "vllm-server":
                return False
            model_name = config.get("model_name") or ""
            return isinstance(model_name, str) and model_name.startswith("infly/Infinity-Parser2")
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("InfinityParser2LayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("InfinityParser2LayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # InfinityParser2 stores bboxes in pixel coordinates (x, y, w, h).
                    # seg.x, seg.y are already pixel values — no normalization needed.
                    x1 = float(seg.x)
                    y1 = float(seg.y)
                    x2 = float(seg.x + seg.w)
                    y2 = float(seg.y + seg.h)

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.INFINITY_PARSER2_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )
