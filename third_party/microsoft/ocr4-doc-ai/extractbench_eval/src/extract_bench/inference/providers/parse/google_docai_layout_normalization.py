"""Google Document AI layout normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape
from typing import TYPE_CHECKING, Any, Literal

from extract_bench.inference.providers.base import ProviderPermanentError
from extract_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)

if TYPE_CHECKING:
    from google.cloud import documentai_v1beta3 as layout_documentai

    from extract_bench.schemas.pipeline_io import RawInferenceResult


_VIRTUAL_PAGE_DIM = 1000.0
_TOP_LEVEL_SUBTITLE_LEVEL = 2
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?(\d+)$", re.IGNORECASE)

_TEXT_TYPE_TO_CANONICAL_LABEL: dict[str, str] = {
    "paragraph": "Text",
    "subtitle": "Section-header",
    "heading-1": "Title",
    "heading-2": "Section-header",
    "heading-3": "Section-header",
    "heading-4": "Section-header",
    "heading-5": "Section-header",
    "header": "Page-header",
    "footer": "Page-footer",
}

_BLOCK_TYPE_TO_CANONICAL_LABEL: dict[str, str] = {
    "table": "Table",
    "list": "List-item",
    "image": "Picture",
}


@dataclass(slots=True)
class NormalizedBBox:
    """Normalized xywh bounding box."""

    x: float
    y: float
    w: float
    h: float


@dataclass(slots=True)
class LayoutTableCellNode:
    """Internal representation of one layout table cell."""

    row_span: int = 1
    col_span: int = 1
    blocks: list[LayoutNode] = field(default_factory=list)


@dataclass(slots=True)
class LayoutNode:
    """Canonical internal node extracted from DocAI layout blocks."""

    block_id: str
    page_start: int
    page_end: int
    kind: Literal["text", "table", "list", "image"]
    canonical_label: str
    bbox: NormalizedBBox | None
    text: str = ""
    text_type: str | None = None
    list_type: str | None = None
    children: list[LayoutNode] = field(default_factory=list)
    header_rows: list[list[LayoutTableCellNode]] = field(default_factory=list)
    body_rows: list[list[LayoutTableCellNode]] = field(default_factory=list)
    caption: str = ""
    image_text: str = ""
    image_mime_type: str = ""
    image_description: str = ""
    image_blob_asset_id: str = ""
    raw_subtype_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def item_type(self) -> str:
        if self.kind == "table":
            return "table"
        if self.kind == "image":
            return "image"
        return "text"


def normalize_layout_document(
    *,
    document: Any,
    raw_result: RawInferenceResult,
) -> ParseOutput:
    """Normalize a typed Layout Parser document into ParseOutput."""
    root_nodes = [_extract_layout_node(block) for block in document.document_layout.blocks]
    page_count = _determine_page_count(document, root_nodes)
    page_dimensions = _build_page_dimension_lookup(document, page_count)

    page_payloads = _build_page_payloads(root_nodes=root_nodes, page_count=page_count, page_dimensions=page_dimensions)
    if not page_payloads:
        raise ProviderPermanentError(
            "Layout Parser normalization requires native layout_pages from the DocAI payload. "
            "No OCR fallback is allowed for google_docai_layout."
        )
    if not any(item.bbox is not None for page_payload in page_payloads for item in page_payload.items):
        raise ProviderPermanentError(
            "Layout Parser normalization requires native layout_pages from the DocAI payload. "
            "No OCR fallback is allowed for google_docai_layout."
        )

    pages = [
        PageIR(
            page_index=page_payload.page_number - 1,
            markdown=page_payload.md,
        )
        for page_payload in page_payloads
    ]
    full_markdown = "\n\n---\n\n".join(page.markdown for page in pages if page.markdown)

    return ParseOutput(
        task_type="parse",
        example_id=raw_result.request.example_id,
        pipeline_name=raw_result.pipeline_name,
        pages=pages,
        layout_pages=page_payloads,
        markdown=full_markdown,
        job_id=None,
    )


def _extract_layout_node(block: Any) -> LayoutNode:
    page_span = block.page_span
    page_start = int(page_span.page_start or 1) if page_span else 1
    page_end = int(page_span.page_end or page_start) if page_span else page_start
    bbox = _bbox_from_bounding_poly(block.bounding_box)
    block_id = block.block_id or ""

    if block.text_block:
        text_type = block.text_block.type_.strip().lower() if block.text_block.type_ else None
        children = [_extract_layout_node(child) for child in block.text_block.blocks]
        return LayoutNode(
            block_id=block_id,
            page_start=page_start,
            page_end=page_end,
            kind="text",
            canonical_label=_canonical_label_for_text_type(text_type),
            bbox=bbox,
            text=block.text_block.text or "",
            text_type=text_type,
            children=children,
            raw_subtype_metadata={"text_type": text_type},
        )

    if block.table_block:
        return LayoutNode(
            block_id=block_id,
            page_start=page_start,
            page_end=page_end,
            kind="table",
            canonical_label="Table",
            bbox=bbox,
            header_rows=[_extract_table_row(row) for row in block.table_block.header_rows],
            body_rows=[_extract_table_row(row) for row in block.table_block.body_rows],
            caption=(block.table_block.caption or "").strip(),
            raw_subtype_metadata={"has_annotations": bool(block.table_block.annotations)},
        )

    if block.list_block:
        entries: list[LayoutNode] = []
        for entry in block.list_block.list_entries:
            for child in entry.blocks:
                entries.append(_extract_layout_node(child))
        list_type = block.list_block.type_.strip().lower() if block.list_block.type_ else None
        return LayoutNode(
            block_id=block_id,
            page_start=page_start,
            page_end=page_end,
            kind="list",
            canonical_label="List-item",
            bbox=bbox,
            list_type=list_type,
            children=entries,
            raw_subtype_metadata={"list_type": list_type},
        )

    if getattr(block, "image_block", None):
        image_block = block.image_block
        description = (getattr(image_block.annotations, "description", "") or "").strip()
        return LayoutNode(
            block_id=block_id,
            page_start=page_start,
            page_end=page_end,
            kind="image",
            canonical_label="Picture",
            bbox=bbox,
            image_text=(image_block.image_text or "").strip(),
            image_mime_type=(image_block.mime_type or "").strip(),
            image_description=description,
            image_blob_asset_id=(image_block.blob_asset_id or "").strip(),
            raw_subtype_metadata={
                "mime_type": image_block.mime_type or "",
                "blob_asset_id": image_block.blob_asset_id or "",
            },
        )

    raise ProviderPermanentError(f"Unsupported DocumentLayout block kind for block_id={block_id!r}")


def _extract_table_row(
    row: Any,
) -> list[LayoutTableCellNode]:
    return [
        LayoutTableCellNode(
            row_span=int(cell.row_span or 1),
            col_span=int(cell.col_span or 1),
            blocks=[_extract_layout_node(block) for block in cell.blocks],
        )
        for cell in row.cells
    ]


def _bbox_from_bounding_poly(poly: Any) -> NormalizedBBox | None:
    if poly is None:
        return None

    normalized_vertices = list(getattr(poly, "normalized_vertices", []) or [])
    if normalized_vertices:
        xs = [float(vertex.x) for vertex in normalized_vertices if getattr(vertex, "x", None) is not None]
        ys = [float(vertex.y) for vertex in normalized_vertices if getattr(vertex, "y", None) is not None]
        if xs and ys:
            x1 = min(xs)
            y1 = min(ys)
            x2 = max(xs)
            y2 = max(ys)
            if x2 > x1 and y2 > y1:
                return NormalizedBBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

    return None


def _canonical_label_for_text_type(text_type: str | None) -> str:
    if not text_type:
        return "Text"
    return _TEXT_TYPE_TO_CANONICAL_LABEL.get(text_type, "Text")


def _determine_page_count(document: layout_documentai.Document, root_nodes: list[LayoutNode]) -> int:
    if document.pages:
        return len(document.pages)

    page_end = 0
    for node in root_nodes:
        page_end = max(page_end, _max_page_end(node))
    return page_end


def _max_page_end(node: LayoutNode) -> int:
    page_end = node.page_end
    for child in node.children:
        page_end = max(page_end, _max_page_end(child))
    for row in (*node.header_rows, *node.body_rows):
        for cell in row:
            for block in cell.blocks:
                page_end = max(page_end, _max_page_end(block))
    return page_end


def _build_page_dimension_lookup(
    document: layout_documentai.Document,
    page_count: int,
) -> dict[int, tuple[float, float]]:
    lookup: dict[int, tuple[float, float]] = {}
    for page in document.pages:
        width = float(page.dimension.width) if page.dimension and page.dimension.width else _VIRTUAL_PAGE_DIM
        height = float(page.dimension.height) if page.dimension and page.dimension.height else _VIRTUAL_PAGE_DIM
        lookup[int(page.page_number or len(lookup) + 1)] = (width, height)

    for page_number in range(1, page_count + 1):
        lookup.setdefault(page_number, (_VIRTUAL_PAGE_DIM, _VIRTUAL_PAGE_DIM))
    return lookup


def _build_page_payloads(
    *,
    root_nodes: list[LayoutNode],
    page_count: int,
    page_dimensions: dict[int, tuple[float, float]],
) -> list[ParseLayoutPageIR]:
    page_payloads: list[ParseLayoutPageIR] = []

    for page_number in range(1, page_count + 1):
        items: list[LayoutItemIR] = []
        for node in root_nodes:
            items.extend(_build_items_for_page(node, page_number=page_number, ancestor_heading_level=None))

        if not items:
            continue

        page_md_parts = [_item_markdown(item) for item in items if _item_markdown(item)]
        page_markdown = "\n\n".join(page_md_parts).strip()
        header_markdown = "\n\n".join(
            _item_markdown(item)
            for item in items
            if item.bbox and item.bbox.label == "Page-header" and _item_markdown(item)
        ).strip()
        footer_markdown = "\n\n".join(
            _item_markdown(item)
            for item in items
            if item.bbox and item.bbox.label == "Page-footer" and _item_markdown(item)
        ).strip()
        printed_page_number = _extract_printed_page_number(items)
        page_width, page_height = page_dimensions[page_number]

        page_payloads.append(
            ParseLayoutPageIR(
                page_number=page_number,
                width=page_width,
                height=page_height,
                md=page_markdown,
                text=page_markdown,
                page_header_markdown=header_markdown,
                page_footer_markdown=footer_markdown,
                printed_page_number=printed_page_number,
                items=items,
            )
        )

    return page_payloads


def _build_items_for_page(
    node: LayoutNode,
    *,
    page_number: int,
    ancestor_heading_level: int | None,
) -> list[LayoutItemIR]:
    if page_number < node.page_start or page_number > node.page_end:
        return []

    if node.kind == "table":
        return [_table_item(node)]
    if node.kind == "image":
        return [_image_item(node)]
    if node.kind == "list":
        item = _list_item(node, page_number=page_number, ancestor_heading_level=ancestor_heading_level)
        return [item] if item.md else []

    item = _text_item(node, ancestor_heading_level=ancestor_heading_level)
    next_heading_level = _next_heading_level(node, ancestor_heading_level)
    items = [item] if item.md or item.value else []
    for child in node.children:
        items.extend(_build_items_for_page(child, page_number=page_number, ancestor_heading_level=next_heading_level))
    return items


def _text_item(node: LayoutNode, *, ancestor_heading_level: int | None) -> LayoutItemIR:
    markdown = _render_text_markdown(node, ancestor_heading_level=ancestor_heading_level)
    return LayoutItemIR(
        type="text",
        md=markdown,
        value=node.text.strip(),
        bbox=_segment_for_node(node),
        layout_segments=_segments_for_node(node),
    )


def _table_item(node: LayoutNode) -> LayoutItemIR:
    html = _render_table_html(node)
    return LayoutItemIR(
        type="table",
        md=html,
        html=html,
        value=html,
        bbox=_segment_for_node(node),
        layout_segments=_segments_for_node(node),
    )


def _image_item(node: LayoutNode) -> LayoutItemIR:
    markdown = _render_image_markdown(node)
    value = node.image_description or node.image_text or node.image_blob_asset_id
    return LayoutItemIR(
        type="image",
        md=markdown,
        value=value,
        bbox=_segment_for_node(node),
        layout_segments=_segments_for_node(node),
    )


def _list_item(
    node: LayoutNode,
    *,
    page_number: int,
    ancestor_heading_level: int | None,
) -> LayoutItemIR:
    markdown = _render_list_markdown(node, page_number=page_number, ancestor_heading_level=ancestor_heading_level)
    return LayoutItemIR(
        type="text",
        md=markdown,
        value=markdown,
        bbox=_segment_for_node(node),
        layout_segments=_segments_for_node(node),
    )


def _render_text_markdown(node: LayoutNode, *, ancestor_heading_level: int | None) -> str:
    text = node.text.strip()
    if not text:
        return ""

    if node.text_type and node.text_type.startswith("heading-"):
        try:
            heading_level = int(node.text_type.split("-", maxsplit=1)[1])
        except (IndexError, ValueError):
            heading_level = 4
        heading_level = max(1, min(heading_level, 6))
        return f"{'#' * heading_level} {text}"

    if node.text_type == "subtitle":
        if ancestor_heading_level is None:
            heading_level = _TOP_LEVEL_SUBTITLE_LEVEL
        else:
            heading_level = min(ancestor_heading_level + 1, 6)
        return f"{'#' * heading_level} {text}"

    return text


def _render_list_markdown(
    node: LayoutNode,
    *,
    page_number: int,
    ancestor_heading_level: int | None,
) -> str:
    marker = "1." if (node.list_type or "").lower() == "ordered" else "-"
    rendered_entries: list[str] = []
    for child in node.children:
        if page_number < child.page_start or page_number > child.page_end:
            continue
        child_text = _render_inline_text(child, ancestor_heading_level=ancestor_heading_level)
        if child_text:
            rendered_entries.append(f"{marker} {child_text}")
    return "\n".join(rendered_entries)


def _render_image_markdown(node: LayoutNode) -> str:
    description = node.image_description or node.image_text
    if description:
        return description
    if node.image_blob_asset_id:
        return f"[Image: {node.image_blob_asset_id}]"
    return ""


def _render_table_html(node: LayoutNode) -> str:
    parts = ["<table>"]
    if node.caption:
        parts.append(f"<caption>{escape(node.caption)}</caption>")

    if node.header_rows:
        parts.append("<thead>")
        for row in node.header_rows:
            parts.append("<tr>")
            for cell in row:
                parts.append(_render_table_cell(cell, tag="th"))
            parts.append("</tr>")
        parts.append("</thead>")

    if node.body_rows:
        parts.append("<tbody>")
        for row in node.body_rows:
            parts.append("<tr>")
            for cell in row:
                parts.append(_render_table_cell(cell, tag="td"))
            parts.append("</tr>")
        parts.append("</tbody>")

    parts.append("</table>")
    return "\n".join(parts)


def _render_table_cell(cell: LayoutTableCellNode, *, tag: str) -> str:
    attrs: list[str] = []
    if cell.col_span > 1:
        attrs.append(f' colspan="{cell.col_span}"')
    if cell.row_span > 1:
        attrs.append(f' rowspan="{cell.row_span}"')
    content = escape(_inline_text_from_nodes(cell.blocks))
    return f"<{tag}{''.join(attrs)}>{content}</{tag}>"


def _inline_text_from_nodes(nodes: list[LayoutNode]) -> str:
    rendered = [_render_inline_text(node, ancestor_heading_level=None) for node in nodes]
    return " ".join(part for part in rendered if part).strip()


def _render_inline_text(node: LayoutNode, *, ancestor_heading_level: int | None) -> str:
    if node.kind == "text":
        text = node.text.strip()
        child_text = " ".join(
            _render_inline_text(child, ancestor_heading_level=_next_heading_level(node, ancestor_heading_level))
            for child in node.children
        ).strip()
        return " ".join(part for part in [text, child_text] if part).strip()

    if node.kind == "image":
        return node.image_description or node.image_text or ""

    if node.kind == "list":
        return " ".join(
            _render_inline_text(child, ancestor_heading_level=ancestor_heading_level) for child in node.children
        ).strip()

    if node.kind == "table":
        return node.caption

    return ""


def _segment_for_node(node: LayoutNode) -> LayoutSegmentIR | None:
    if node.bbox is None:
        return None
    return LayoutSegmentIR(
        x=node.bbox.x,
        y=node.bbox.y,
        w=node.bbox.w,
        h=node.bbox.h,
        confidence=1.0,
        label=node.canonical_label,
    )


def _segments_for_node(node: LayoutNode) -> list[LayoutSegmentIR]:
    segment = _segment_for_node(node)
    return [segment] if segment else []


def _next_heading_level(node: LayoutNode, ancestor_heading_level: int | None) -> int | None:
    if node.text_type and node.text_type.startswith("heading-"):
        try:
            return int(node.text_type.split("-", maxsplit=1)[1])
        except (IndexError, ValueError):
            return ancestor_heading_level
    if node.text_type == "subtitle":
        if ancestor_heading_level is None:
            return _TOP_LEVEL_SUBTITLE_LEVEL
        return min(ancestor_heading_level + 1, 6)
    return ancestor_heading_level


def _item_markdown(item: LayoutItemIR) -> str:
    return item.md.strip() or item.html.strip() or item.value.strip()


def _extract_printed_page_number(items: list[LayoutItemIR]) -> str:
    for item in items:
        if item.bbox is None or item.bbox.label not in {"Page-header", "Page-footer"}:
            continue
        candidate = item.value.strip() or item.md.strip()
        if candidate and _PAGE_NUMBER_RE.match(candidate):
            return candidate
    return ""
