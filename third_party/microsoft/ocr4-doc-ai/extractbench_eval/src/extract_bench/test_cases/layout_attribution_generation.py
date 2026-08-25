"""Generate layout-attribution benchmark annotations from normalized parse output."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from extract_bench.layout_label_mapping import (
    detect_llamaparse_label_version,
    map_llamaparse_raw_label_to_canonical,
)
from extract_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from extract_bench.test_cases.rule_ids import canonical_rule_signature, compute_rule_id
from extract_bench.test_cases.schema import LayoutTestRule

_TABLE_HTML_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)
_ITEM_TYPE_TO_LABEL = {
    "caption": "caption",
    "footer": "page-footer",
    "footnote": "footnote",
    "header": "page-header",
    "list-item": "list-item",
    "page-footer": "page-footer",
    "page-header": "page-header",
    "picture": "picture",
    "section-header": "section-header",
    "table": "table",
    "text": "text",
    "title": "title",
}


def compute_page_hash_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Compute the layout-attribution document hash from source asset bytes."""
    return hashlib.sha256(pdf_bytes).hexdigest()


def build_layout_attribution_test_case(
    *,
    parse_output: ParseOutput,
    page_hash: str,
    source_id: str,
    original_filename: str,
    doc_category: str | None,
    source_dataset: str,
    hash_len: int = 16,
    page_no: int = 1,
) -> dict[str, Any]:
    """Build a layout-attribution compatible JSON payload from `ParseOutput`."""
    if not parse_output.layout_pages:
        raise ValueError("Layout attribution generation requires at least one layout page in the parse output.")

    expected_markdown = _resolve_expected_markdown(parse_output)
    test_rules: list[dict[str, Any]] = []
    sorted_pages = sorted(parse_output.layout_pages, key=lambda page: page.page_number)
    for page in sorted_pages:
        test_rules.extend(
            build_layout_rules_for_page(
                page=page,
                hash_len=hash_len,
                page_number=page.page_number,
            )
        )

    metadata: dict[str, Any] = {
        "doc_category": doc_category,
        "original_filename": original_filename,
        "page_hash": page_hash,
        "page_count": len(sorted_pages),
    }
    if len(sorted_pages) == 1:
        metadata["page_no"] = page_no

    return {
        "expected_markdown": expected_markdown,
        "metadata": metadata,
        "ontology": "canonical",
        "page_index": 0,
        "source_dataset": source_dataset,
        "source_id": source_id,
        "source_ontology": "canonical",
        "test_rules": test_rules,
    }


def build_layout_rules_for_page(
    *,
    page: ParseLayoutPageIR,
    hash_len: int = 16,
    page_number: int | None = None,
) -> list[dict[str, Any]]:
    """Generate segment-level layout rules for a single page."""
    page_width, page_height = _resolve_page_dimensions(page)
    table_htmls = _extract_table_htmls(page.md or page.text)
    label_version = detect_llamaparse_label_version(_collect_raw_labels(page))
    rule_page_number = page_number if page_number is not None else page.page_number

    test_rules: list[dict[str, Any]] = []
    table_html_idx = 0
    ro_index = 0

    for item in page.items:
        segments = _segments_for_item(item)
        if not segments:
            continue

        table_content = None
        if item.type == "table":
            table_content, consumed_html = _build_table_content(
                item=item,
                table_htmls=table_htmls,
                table_html_idx=table_html_idx,
            )
            if consumed_html:
                table_html_idx += 1

        for segment in segments:
            raw_label = _resolve_raw_label(item=item, segment=segment)
            if raw_label is None:
                continue

            canonical_label, attributes = map_llamaparse_raw_label_to_canonical(
                raw_label,
                label_version=label_version,
            )
            rule_payload: dict[str, Any] = {
                "type": "layout",
                "page": rule_page_number,
                "bbox": _normalize_bbox(segment=segment, page_width=page_width, page_height=page_height),
                "canonical_class": canonical_label.value,
                "attributes": attributes,
                "source_label": raw_label,
                "ro_index": ro_index,
            }

            content = table_content if item.type == "table" else _build_text_content(item=item, segment=segment)
            if content is not None:
                rule_payload["content"] = content

            validated = LayoutTestRule.model_validate(rule_payload)
            test_rules.append(validated.model_dump(exclude_none=True))
            ro_index += 1

    _assign_deterministic_ids(test_rules, hash_len=hash_len)
    return test_rules


def _resolve_expected_markdown(parse_output: ParseOutput) -> str:
    if parse_output.markdown:
        return parse_output.markdown

    page_markdowns = [
        page.md or page.text for page in sorted(parse_output.layout_pages, key=lambda page: page.page_number)
    ]
    non_empty_markdowns = [markdown for markdown in page_markdowns if markdown]
    return "\n\n".join(non_empty_markdowns)


def _collect_raw_labels(page: ParseLayoutPageIR) -> list[str]:
    labels: list[str] = []
    for item in page.items:
        for segment in _segments_for_item(item):
            raw_label = _resolve_raw_label(item=item, segment=segment)
            if raw_label is not None:
                labels.append(raw_label)
    return labels


def _segments_for_item(item: LayoutItemIR) -> list[LayoutSegmentIR]:
    if item.layout_segments:
        return list(item.layout_segments)
    if item.bbox is not None:
        return [item.bbox]
    return []


def _resolve_raw_label(item: LayoutItemIR, segment: LayoutSegmentIR) -> str | None:
    if segment.label:
        return segment.label
    if item.bbox is not None and item.bbox.label:
        return item.bbox.label
    return _ITEM_TYPE_TO_LABEL.get(item.type.strip().lower())


def _resolve_page_dimensions(page: ParseLayoutPageIR) -> tuple[float, float]:
    width = page.width or 0.0
    height = page.height or 0.0
    if width > 0 and height > 0:
        return float(width), float(height)

    max_x = 0.0
    max_y = 0.0
    for item in page.items:
        for segment in _segments_for_item(item):
            max_x = max(max_x, float(segment.x + segment.w))
            max_y = max(max_y, float(segment.y + segment.h))
    if max_x <= 0 or max_y <= 0:
        raise ValueError("Unable to resolve page dimensions from layout page content.")
    return max_x, max_y


def _normalize_bbox(
    *,
    segment: LayoutSegmentIR,
    page_width: float,
    page_height: float,
) -> list[float]:
    return [
        segment.x / page_width,
        segment.y / page_height,
        segment.w / page_width,
        segment.h / page_height,
    ]


def _slice_text(item: LayoutItemIR, segment: LayoutSegmentIR) -> str:
    item_text = item.value or ""
    start = segment.start_index
    end = segment.end_index
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        return item_text[start : end + 1]
    return item_text


def _build_text_content(item: LayoutItemIR, segment: LayoutSegmentIR) -> dict[str, str] | None:
    text = _slice_text(item, segment).strip()
    if not text:
        return None
    return {"type": "text", "text": text}


def _build_table_content(
    *,
    item: LayoutItemIR,
    table_htmls: list[str],
    table_html_idx: int,
) -> tuple[dict[str, str] | None, bool]:
    if table_html_idx < len(table_htmls):
        return {"type": "table", "html": table_htmls[table_html_idx]}, True

    value = item.value.strip()
    if value:
        if _TABLE_HTML_RE.fullmatch(value):
            return {"type": "table", "html": value}, False
        return {"type": "text", "text": value}, False
    return None, False


def _extract_table_htmls(markdown: str) -> list[str]:
    return _TABLE_HTML_RE.findall(markdown)


def _assign_deterministic_ids(test_rules: list[dict[str, Any]], *, hash_len: int) -> None:
    indexed_rules: list[tuple[int, dict[str, Any], str]] = [
        (index, rule, canonical_rule_signature(rule)) for index, rule in enumerate(test_rules)
    ]

    for _, rule, _ in indexed_rules:
        rule["id"] = compute_rule_id(rule, hash_len)

    by_id: dict[str, list[tuple[int, dict[str, Any], str]]] = {}
    for entry in indexed_rules:
        _, rule, _ = entry
        rule_id = rule["id"]
        by_id.setdefault(rule_id, []).append(entry)

    for base_id, duplicates in by_id.items():
        if len(duplicates) <= 1:
            continue
        duplicates_sorted = sorted(
            duplicates,
            key=lambda entry: (entry[2], entry[0]),
        )
        for prefix_counter, entry in enumerate(duplicates_sorted):
            entry[1]["id"] = f"{prefix_counter:03d}-{base_id}"


def read_pdf_bytes(pdf_path: Path) -> bytes:
    """Read a page-level PDF object from disk."""
    return pdf_path.read_bytes()
