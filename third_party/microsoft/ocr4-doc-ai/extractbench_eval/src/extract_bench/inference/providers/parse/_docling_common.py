"""Common functionality for docling and docling_serve providers."""

from typing import Any

from docling_core.types.doc.document import DoclingDocument

from extract_bench.layout_label_mapping import (
    UnknownRawLayoutLabelError,
    map_docling_raw_label_to_canonical,
)
from extract_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    ParseLayoutPageIR,
)

_DOCLING_EXCLUDED_LAYOUT_LABELS = frozenset(
    {
        "empty_value",
        "field_heading",
        "field_hint",
        "field_item",
        "field_key",
        "field_region",
        "field_value",
        "marker",
    }
)
_DOCLING_TABLE_LABELS = frozenset({"document_index", "table"})
_DOCLING_IMAGE_LABELS = frozenset({"chart", "picture"})


def _normalize_docling_label(label: object) -> str | None:
    if label is None:
        return None
    value = getattr(label, "value", label)
    if not isinstance(value, str):
        return None
    return value.strip().lower()


def _should_include_docling_label(raw_label: str) -> bool:
    if raw_label in _DOCLING_EXCLUDED_LAYOUT_LABELS:
        return False
    try:
        map_docling_raw_label_to_canonical(raw_label)
    except UnknownRawLayoutLabelError:
        return False
    return True


def _docling_item_type(raw_label: str) -> str:
    if raw_label in _DOCLING_TABLE_LABELS:
        return "table"
    if raw_label in _DOCLING_IMAGE_LABELS:
        return "image"
    return "text"


def _extract_docling_item_value(item: Any, doc: DoclingDocument, raw_label: str) -> str:
    item_type = _docling_item_type(raw_label)
    if item_type == "image":
        return ""

    if item_type == "table" and hasattr(item, "export_to_html"):
        try:
            html = item.export_to_html(doc=doc, add_caption=True)
            if isinstance(html, str):
                return html
        except Exception:
            pass

    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text

    if hasattr(item, "export_to_markdown"):
        try:
            markdown = item.export_to_markdown()
            if isinstance(markdown, str):
                return markdown
        except Exception:
            pass

    return ""


def _normalize_docling_charspan(
    charspan: object,
    *,
    text_length: int,
    include_span: bool,
) -> tuple[int | None, int | None]:
    if not include_span or not isinstance(charspan, (list, tuple)) or len(charspan) != 2:
        return (None, None)

    start_raw, end_raw = charspan
    if not isinstance(start_raw, int) or not isinstance(end_raw, int):
        return (None, None)

    start = max(0, min(start_raw, text_length))
    end_exclusive = max(start, min(end_raw, text_length))
    if end_exclusive <= start:
        return (None, None)

    # Docling charspan behaves like a Python slice [start, end).
    return (start, end_exclusive - 1)


def _build_docling_segment(
    *,
    prov: Any,
    raw_label: str,
    page_width: float,
    page_height: float,
    include_span: bool,
    text_length: int,
) -> LayoutSegmentIR | None:
    bbox = getattr(prov, "bbox", None)
    if bbox is None or page_width <= 0 or page_height <= 0:
        return None

    bbox_top_left = bbox.to_top_left_origin(page_height=page_height)
    width = bbox_top_left.r - bbox_top_left.l
    height = bbox_top_left.b - bbox_top_left.t
    if width <= 0 or height <= 0:
        return None

    start_index, end_index = _normalize_docling_charspan(
        getattr(prov, "charspan", None),
        text_length=text_length,
        include_span=include_span,
    )

    return LayoutSegmentIR(
        x=bbox_top_left.l / page_width,
        y=bbox_top_left.t / page_height,
        w=width / page_width,
        h=height / page_height,
        confidence=1.0,
        label=raw_label,
        start_index=start_index,
        end_index=end_index,
    )


def _merge_segments(segments: list[LayoutSegmentIR]) -> LayoutSegmentIR | None:
    if not segments:
        return None

    x1 = min(segment.x for segment in segments)
    y1 = min(segment.y for segment in segments)
    x2 = max(segment.x + segment.w for segment in segments)
    y2 = max(segment.y + segment.h for segment in segments)
    return LayoutSegmentIR(
        x=x1,
        y=y1,
        w=x2 - x1,
        h=y2 - y1,
        confidence=1.0,
        label=segments[0].label,
    )


def _build_docling_layout_pages(
    *,
    doc: DoclingDocument,
    raw_pages: list[dict[str, Any]],
) -> list[ParseLayoutPageIR]:
    page_markdown_by_number: dict[int, str] = {}
    for page_data in raw_pages:
        page_number = page_data.get("page")
        if isinstance(page_number, int) and page_number > 0:
            page_markdown_by_number[page_number] = str(page_data.get("markdown", ""))

    layout_pages: list[ParseLayoutPageIR] = []
    for page_number in sorted(doc.pages.keys()):
        page = doc.pages[page_number]
        page_width = float(page.size.width)
        page_height = float(page.size.height)
        items: list[LayoutItemIR] = []

        for item, _level in doc.iterate_items(page_no=page_number):
            raw_label = _normalize_docling_label(getattr(item, "label", None))
            if raw_label is None or not _should_include_docling_label(raw_label):
                continue

            item_type = _docling_item_type(raw_label)
            item_value = _extract_docling_item_value(item, doc, raw_label)
            include_span = item_type == "text"

            page_provs = [
                prov for prov in getattr(item, "prov", []) or [] if getattr(prov, "page_no", None) == page_number
            ]
            segments = [
                segment
                for prov in page_provs
                if (
                    segment := _build_docling_segment(
                        prov=prov,
                        raw_label=raw_label,
                        page_width=page_width,
                        page_height=page_height,
                        include_span=include_span,
                        text_length=len(item_value),
                    )
                )
                is not None
            ]
            if not segments:
                continue

            merged_bbox = _merge_segments(segments)
            items.append(
                LayoutItemIR(
                    type=item_type,
                    value=item_value,
                    bbox=merged_bbox,
                    layout_segments=segments,
                )
            )

        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_number,
                width=page_width,
                height=page_height,
                md=page_markdown_by_number.get(page_number, ""),
                items=items,
            )
        )

    return layout_pages
