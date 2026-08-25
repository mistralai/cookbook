"""Shared normalization helpers for LlamaParse V2 SDK and local cli2 outputs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

from llama_cloud.types import (
    CodeItem,
    FooterItem,
    HeaderItem,
    HeadingItem,
    ImageItem,
    LinkItem,
    ListItem,
    TableItem,
    TextItem,
)
from llama_cloud.types.parsing_get_response import (
    ItemsPage,
    ItemsPageStructuredResultPage,
    Metadata,
    MetadataPage,
    ParsingGetResponse,
    Text,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from extract_bench.schemas.parse_output import (
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)

JsonItem = HeaderItem | FooterItem | TableItem | ListItem | CodeItem | HeadingItem | ImageItem | LinkItem | TextItem

logger = logging.getLogger(__name__)


class StructuredResultPage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_number: int
    items: list[JsonItem]
    page_width: float
    page_height: float
    success: Literal[True] = True


class FailedStructuredPage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_number: int
    error: str
    success: Literal[False] = False


class StructuredResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pages: list[StructuredResultPage | FailedStructuredPage]


class MarkdownPage(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    page_number: int
    markdown: str = Field(default="", alias="md")
    header: str = Field(default="", alias="pageHeaderMarkdown")
    footer: str = Field(default="", alias="pageFooterMarkdown")
    printed_page_number: str = Field(default="", alias="printedPageNumber")


class MarkdownResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pages: list[MarkdownPage]


def flatten_v2_items(
    items: Sequence[object],
    output_tables_as_markdown: bool,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str], list[str]]:
    """Flatten V2 container items and normalize bbox payloads."""
    page_items: list[dict[str, Any]] = []
    item_markdowns: list[str] = []
    item_texts: list[str] = []
    header_markdowns: list[str] = []
    footer_markdowns: list[str] = []

    for item in items:
        if not isinstance(
            item,
            (
                TextItem,
                HeadingItem,
                ListItem,
                CodeItem,
                TableItem,
                ImageItem,
                LinkItem,
                HeaderItem,
                FooterItem,
            ),
        ):
            continue

        if isinstance(item, (ListItem, HeaderItem, FooterItem)):
            if item.md:
                item_markdowns.append(item.md)
                if isinstance(item, HeaderItem):
                    header_markdowns.append(item.md)
                elif isinstance(item, FooterItem):
                    footer_markdowns.append(item.md)
            child_items, child_mds, child_texts, child_headers, child_footers = flatten_v2_items(
                item.items,
                output_tables_as_markdown,
            )
            page_items.extend(child_items)
            item_markdowns.extend(child_mds)
            item_texts.extend(child_texts)
            header_markdowns.extend(child_headers)
            footer_markdowns.extend(child_footers)
            continue

        item_data: dict[str, Any] = {"type": item.type}

        if isinstance(item, TableItem):
            table_content = item.md if output_tables_as_markdown else (item.html or item.md)
            if table_content:
                item_markdowns.append(table_content)
            # Preserve md on table item_data so that parse_pred_blocks can
            # slice per-segment text using startIndex/endIndex.
            if item.md:
                item_data["md"] = item.md
        elif isinstance(item, LinkItem):
            # Use plain display text instead of linkified markdown so that
            # "www.tdi.texas.gov" stays as-is rather than becoming
            # "[www.tdi.texas.gov](http://www.tdi.texas.gov)".
            if item.text:
                item_markdowns.append(item.text)
        elif item.md:
            item_markdowns.append(item.md)

        if isinstance(item, (TextItem, HeadingItem, CodeItem)):
            item_data["value"] = item.value
            item_texts.append(item.value)

        # Preserve md on ALL items that may have layoutAwareBbox segments so
        # parse_pred_blocks can slice text correctly using startIndex/endIndex
        # (which are computed relative to the md field including markdown
        # formatting, not the stripped value field).
        if item.md and item.bbox:
            item_data["md"] = item.md

        if item.bbox:
            first_bbox = item.bbox[0]
            item_data["bBox"] = {
                "x": first_bbox.x,
                "y": first_bbox.y,
                "w": first_bbox.w,
                "h": first_bbox.h,
                "confidence": first_bbox.confidence,
                "label": first_bbox.label,
            }
            item_data["layoutAwareBbox"] = [
                {
                    "x": bbox.x,
                    "y": bbox.y,
                    "w": bbox.w,
                    "h": bbox.h,
                    "confidence": bbox.confidence,
                    "label": bbox.label,
                    "startIndex": bbox.start_index,
                    "endIndex": bbox.end_index,
                }
                for bbox in item.bbox
            ]

        page_items.append(item_data)

    return page_items, item_markdowns, item_texts, header_markdowns, footer_markdowns


def _build_page(
    *,
    page_number: int,
    items: Sequence[object],
    output_tables_as_markdown: bool,
    page_width: float | None = None,
    page_height: float | None = None,
    include_items: bool = True,
    md_fallback: str = "",
    text_fallback: str = "",
    header: str = "",
    footer: str = "",
    printed_page_number: str = "",
    orientation: int | None = None,
) -> dict[str, Any]:
    page_data: dict[str, Any] = {"page": page_number}

    if page_width is not None:
        page_data["width"] = page_width
    if page_height is not None:
        page_data["height"] = page_height

    if include_items:
        (
            page_items,
            item_markdowns,
            item_texts,
            inferred_headers,
            inferred_footers,
        ) = flatten_v2_items(items, output_tables_as_markdown)
        page_data["items"] = page_items
        if item_markdowns:
            page_data["md"] = "\n\n".join(item_markdowns)
        if item_texts:
            page_data["text"] = "\n\n".join(item_texts)

        # Why: V2 SDK responses commonly expand only items/text/metadata
        # (no markdown expansion), so page header/footer markdown would
        # otherwise be dropped and is_header/is_footer rules fail.
        if not header and inferred_headers:
            page_data["pageHeaderMarkdown"] = "\n\n".join(inferred_headers)
            logger.debug(
                "Inferred pageHeaderMarkdown from HeaderItem(s): page=%s count=%s",
                page_number,
                len(inferred_headers),
            )
        if not footer and inferred_footers:
            page_data["pageFooterMarkdown"] = "\n\n".join(inferred_footers)
            logger.debug(
                "Inferred pageFooterMarkdown from FooterItem(s): page=%s count=%s",
                page_number,
                len(inferred_footers),
            )

    if "md" not in page_data and md_fallback:
        page_data["md"] = md_fallback
    if "text" not in page_data and text_fallback:
        page_data["text"] = text_fallback

    if header:
        page_data["pageHeaderMarkdown"] = header
    if footer:
        page_data["pageFooterMarkdown"] = footer
    if printed_page_number:
        page_data["printedPageNumber"] = printed_page_number
    if orientation is not None:
        page_data["original_orientation_angle"] = orientation

    return page_data


def build_pages_from_cli2_v2_sidecars(
    *,
    items_payload: Any,
    output_tables_as_markdown: bool,
    md_payload: Any | None = None,
    text_payload: Any | None = None,
    metadata_payload: Any | None = None,
) -> list[dict[str, Any]]:
    """Build bench pages from local cli2 V2 sidecar payloads."""
    try:
        structured = StructuredResult.model_validate(items_payload)
        markdown_pages = MarkdownResult.model_validate(md_payload).pages if md_payload is not None else []
        text_pages = Text.model_validate(text_payload).pages if text_payload is not None else []
        metadata_pages = Metadata.model_validate(metadata_payload).pages if metadata_payload is not None else []
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    items_by_page = {page.page_number: page for page in structured.pages}
    md_by_page = {page.page_number: page for page in markdown_pages}
    text_by_page = {page.page_number: page for page in text_pages}
    metadata_by_page = {page.page_number: page for page in metadata_pages}

    page_numbers = set(items_by_page) | set(md_by_page) | set(text_by_page) | set(metadata_by_page)
    pages: list[dict[str, Any]] = []

    for page_number in sorted(page_numbers):
        items_page = items_by_page.get(page_number)
        md_page = md_by_page.get(page_number)
        text_page = text_by_page.get(page_number)
        metadata_page = metadata_by_page.get(page_number)

        md_fallback = md_page.markdown if md_page else ""
        text_fallback = text_page.text if text_page else ""
        header = md_page.header if md_page else ""
        footer = md_page.footer if md_page else ""
        printed_page_number = md_page.printed_page_number if md_page else ""
        orientation = metadata_page.original_orientation_angle if metadata_page else None

        if items_page is None:
            pages.append(
                _build_page(
                    page_number=page_number,
                    items=[],
                    include_items=False,
                    output_tables_as_markdown=output_tables_as_markdown,
                    md_fallback=md_fallback,
                    text_fallback=text_fallback,
                    header=header,
                    footer=footer,
                    printed_page_number=printed_page_number,
                    orientation=orientation,
                )
            )
            continue

        if isinstance(items_page, FailedStructuredPage):
            failed_page: dict[str, Any] = {"page": page_number}
            if text_fallback:
                failed_page["text"] = text_fallback
            if md_fallback:
                failed_page["md"] = md_fallback
            if header:
                failed_page["pageHeaderMarkdown"] = header
            if footer:
                failed_page["pageFooterMarkdown"] = footer
            if printed_page_number:
                failed_page["printedPageNumber"] = printed_page_number
            if orientation is not None:
                failed_page["original_orientation_angle"] = orientation
            pages.append(failed_page)
            continue

        pages.append(
            _build_page(
                page_number=page_number,
                items=items_page.items,
                output_tables_as_markdown=output_tables_as_markdown,
                page_width=items_page.page_width,
                page_height=items_page.page_height,
                md_fallback=md_fallback,
                text_fallback=text_fallback,
                header=header,
                footer=footer,
                printed_page_number=printed_page_number,
                orientation=orientation,
            )
        )

    return pages


def build_pages_from_sdk_expansions(
    *,
    items_pages: Sequence[ItemsPage],
    text_by_page: dict[int, str] | None,
    metadata_by_page: dict[int, MetadataPage] | None,
    output_tables_as_markdown: bool,
    num_pages: int | None = None,
) -> list[dict[str, Any]]:
    """Build bench pages from SDK expansion payloads."""
    text_map = text_by_page or {}
    metadata_map = metadata_by_page or {}
    total_pages = num_pages if num_pages is not None else max(len(items_pages), len(text_map), 1)

    pages: list[dict[str, Any]] = []
    for page_number in range(1, total_pages + 1):
        items_page = items_pages[page_number - 1] if page_number - 1 < len(items_pages) else None
        text_fallback = text_map.get(page_number, "")
        metadata_page = metadata_map.get(page_number)
        orientation = metadata_page.original_orientation_angle if metadata_page else None

        if not isinstance(items_page, ItemsPageStructuredResultPage):
            page_data: dict[str, Any] = {"page": page_number}
            if text_fallback:
                page_data["text"] = text_fallback
            if orientation is not None:
                page_data["original_orientation_angle"] = orientation
            pages.append(page_data)
            continue

        pages.append(
            _build_page(
                page_number=page_number,
                items=items_page.items,
                output_tables_as_markdown=output_tables_as_markdown,
                page_width=items_page.page_width,
                page_height=items_page.page_height,
                text_fallback=text_fallback,
                orientation=orientation,
            )
        )

    return pages


def build_pages_from_sdk_response_payload(
    *,
    raw_payload: Any,
    output_tables_as_markdown: bool,
) -> list[dict[str, Any]]:
    """Build normalized pages from V2 SDK raw payload or legacy normalized payload."""
    if not isinstance(raw_payload, dict):
        return []

    pages_payload = raw_payload.get("pages")
    if isinstance(pages_payload, list) and _looks_like_normalized_pages(pages_payload):
        return pages_payload

    try:
        result = ParsingGetResponse.model_validate(raw_payload)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    items_pages: list[ItemsPage] = result.items.pages if result.items is not None else []

    text_by_page: dict[int, str] = {}
    if result.text is not None:
        for text_page in result.text.pages:
            text_by_page[text_page.page_number] = text_page.text

    metadata_by_page: dict[int, MetadataPage] = {}
    if result.metadata is not None:
        for metadata_page in result.metadata.pages:
            metadata_by_page[metadata_page.page_number] = metadata_page

    num_pages = max(len(items_pages), len(text_by_page), len(metadata_by_page), 1)
    return build_pages_from_sdk_expansions(
        items_pages=items_pages,
        text_by_page=text_by_page,
        metadata_by_page=metadata_by_page,
        output_tables_as_markdown=output_tables_as_markdown,
        num_pages=num_pages,
    )


def build_pages_from_cli2_raw_payload(
    *,
    raw_payload: Any,
    output_tables_as_markdown: bool,
) -> list[dict[str, Any]]:
    """Build normalized pages from cli2 raw payload (new raw or legacy raw)."""
    if not isinstance(raw_payload, dict):
        return []

    pages_payload = raw_payload.get("pages")
    if isinstance(pages_payload, list) and _looks_like_normalized_pages(pages_payload):
        return pages_payload

    items_payload = raw_payload.get("v2_items", raw_payload)
    md_payload = raw_payload.get("v2_md")
    text_payload = raw_payload.get("v2_txt")
    metadata_payload = raw_payload.get("v2_metadata")
    return build_pages_from_cli2_v2_sidecars(
        items_payload=items_payload,
        output_tables_as_markdown=output_tables_as_markdown,
        md_payload=md_payload,
        text_payload=text_payload,
        metadata_payload=metadata_payload,
    )


def build_pages_from_v1_raw_payload(raw_payload: Any) -> list[dict[str, Any]]:
    """Build normalized pages from V1 raw payload (supports legacy and new raw dumps)."""
    if not isinstance(raw_payload, dict):
        return []

    pages_payload = raw_payload.get("pages")
    if not isinstance(pages_payload, list):
        return []

    normalized_pages: list[dict[str, Any]] = []
    for page_index, page_data in enumerate(pages_payload):
        if not isinstance(page_data, dict):
            continue
        page_copy = dict(page_data)
        if "page" not in page_copy:
            page_number = page_copy.get("page_number")
            if isinstance(page_number, int) and page_number > 0:
                page_copy["page"] = page_number
            else:
                page_copy["page"] = page_index + 1
        normalized_pages.append(page_copy)
    return normalized_pages


def extract_job_id_from_raw_payload(raw_payload: Any) -> str | None:
    """Extract job id from raw payload across old and new payload variants."""
    if not isinstance(raw_payload, dict):
        return None

    direct_job_id = raw_payload.get("job_id")
    if isinstance(direct_job_id, str) and direct_job_id:
        return direct_job_id

    job = raw_payload.get("job")
    if isinstance(job, dict):
        job_id = job.get("id")
        if isinstance(job_id, str) and job_id:
            return job_id

    return None


def build_layout_pages_from_pages_payload(pages_payload: Any) -> list[ParseLayoutPageIR]:
    """Build typed ParseLayoutPageIR entries from normalized pages payload."""
    raw_pages = pages_payload if isinstance(pages_payload, list) else []
    layout_pages: list[ParseLayoutPageIR] = []

    for page_index, page_data in enumerate(raw_pages):
        if not isinstance(page_data, dict):
            continue
        page_candidate = dict(page_data)
        if "page" not in page_candidate and "page_number" not in page_candidate:
            page_candidate["page"] = page_index + 1
        try:
            layout_page = ParseLayoutPageIR.model_validate(page_candidate)
        except ValidationError:
            continue
        layout_pages.append(layout_page)

    layout_pages.sort(key=lambda page: page.page_number)
    return layout_pages


def layout_pages_to_legacy_pages_payload(
    layout_pages: Sequence[ParseLayoutPageIR],
) -> list[dict[str, Any]]:
    """Convert typed layout pages into legacy pages payload consumed by layout extractors."""
    legacy_pages: list[dict[str, Any]] = []
    for page in sorted(layout_pages, key=lambda p: p.page_number):
        page_data: dict[str, Any] = {
            "page": page.page_number,
            "items": [],
        }
        if page.width is not None:
            page_data["width"] = page.width
        if page.height is not None:
            page_data["height"] = page.height
        if page.md:
            page_data["md"] = page.md
        if page.text:
            page_data["text"] = page.text
        if page.page_header_markdown:
            page_data["pageHeaderMarkdown"] = page.page_header_markdown
        if page.page_footer_markdown:
            page_data["pageFooterMarkdown"] = page.page_footer_markdown
        if page.printed_page_number:
            page_data["printedPageNumber"] = page.printed_page_number
        if page.original_orientation_angle is not None:
            page_data["original_orientation_angle"] = page.original_orientation_angle

        items: list[dict[str, Any]] = []
        for item in page.items:
            item_data: dict[str, Any] = {"type": item.type}
            if item.md:
                item_data["md"] = item.md
            if item.html:
                item_data["html"] = item.html
            if item.value:
                item_data["value"] = item.value
            if item.bbox is not None:
                item_data["bBox"] = _segment_to_legacy_bbox(item.bbox, include_span=False)
            if item.layout_segments:
                item_data["layoutAwareBbox"] = [
                    _segment_to_legacy_bbox(segment, include_span=True) for segment in item.layout_segments
                ]
            elif item.bbox is not None:
                # Preserve legacy fallback behavior where a single bBox can act as segment.
                item_data["layoutAwareBbox"] = [_segment_to_legacy_bbox(item.bbox, include_span=True)]
            items.append(item_data)

        page_data["items"] = items
        legacy_pages.append(page_data)

    return legacy_pages


def build_parse_output_from_pages(
    *,
    pages_payload: Any,
    example_id: str,
    pipeline_name: str,
    job_id: str | None = None,
) -> ParseOutput:
    """Build ParseOutput from normalized page payloads."""
    layout_pages = build_layout_pages_from_pages_payload(pages_payload)
    page_irs: list[PageIR] = []
    for page in layout_pages:
        # Keep section metadata (header/footer/page number) in structured layout_pages only.
        # Do not inject tags into markdown body to avoid duplicated content.
        markdown = page.md or page.text

        page_irs.append(PageIR(page_index=page.page_number - 1, markdown=markdown))

    page_irs.sort(key=lambda page: page.page_index)
    full_markdown = "\n\n---\n\n".join(page.markdown for page in page_irs)

    return ParseOutput(
        task_type="parse",
        example_id=example_id,
        pipeline_name=pipeline_name,
        pages=page_irs,
        layout_pages=layout_pages,
        markdown=full_markdown,
        job_id=job_id,
    )


def _looks_like_normalized_pages(pages_payload: list[Any]) -> bool:
    for page in pages_payload:
        if isinstance(page, dict):
            if "page_number" in page:
                return False
            if "page" in page:
                return True
            if any(
                key in page
                for key in (
                    "md",
                    "text",
                    "width",
                    "height",
                    "pageHeaderMarkdown",
                    "pageFooterMarkdown",
                    "printedPageNumber",
                )
            ):
                return True
            return False
    return False


def _segment_to_legacy_bbox(
    segment: LayoutSegmentIR,
    *,
    include_span: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "x": segment.x,
        "y": segment.y,
        "w": segment.w,
        "h": segment.h,
        "confidence": segment.confidence,
        "label": segment.label,
    }
    if include_span:
        payload["startIndex"] = segment.start_index
        payload["endIndex"] = segment.end_index
    return payload
