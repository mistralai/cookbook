"""Shared layout parsing utilities for LLM-based parse_with_layout providers.

These utilities are used by Google, OpenAI, and Anthropic providers that
produce layout-annotated output using <div data-bbox="..." data-label="...">
HTML wrappers with the Core11 label set.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from extract_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    ParseLayoutPageIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout-annotated prompts (Core11 label set)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_LAYOUT = (
    "You are a document parser. Your task is to convert "
    "document images to clean, well-structured markdown."
    "\n\nGuidelines:\n"
    "- Preserve the document structure "
    "(headings, paragraphs, lists, tables)\n"
    "- Convert tables to HTML format "
    "(<table>, <tr>, <th>, <td>)\n"
    "- For existing tables in the document: use colspan "
    "and rowspan attributes to preserve merged cells "
    "and hierarchical headers\n"
    "- For charts/graphs being converted to tables: use "
    "flat combined column headers (e.g., "
    '"Primary 2015" not separate rows) so each data '
    "cell's row contains all its labels\n"
    "- Describe images/figures briefly in square brackets "
    "like [Figure: description]\n"
    "- Preserve any code blocks with appropriate syntax "
    "highlighting\n"
    "- Maintain reading order (left-to-right, "
    "top-to-bottom for Western documents)\n"
    "- Do not add commentary or explanations "
    "- only output the parsed content"
    "\n\n"
    "Additionally, wrap each layout element in a <div> tag with:\n"
    '- data-bbox="[x1, y1, x2, y2]" — bounding box in normalized 0-1000 '
    "coordinates where x is horizontal (left edge = 0, right edge = 1000) "
    "and y is vertical (top = 0, bottom = 1000). "
    "x1,y1 is the top-left corner and x2,y2 is the bottom-right corner.\n"
    '- data-label="<category>" — one of: Caption, Footnote, Formula, '
    "List-item, Page-footer, Page-header, Picture, Section-header, "
    "Table, Text, Title\n\n"
    "Place elements in reading order. Every piece of content must be "
    "inside exactly one <div> wrapper."
)

USER_PROMPT_LAYOUT = (
    "Parse this document page and output its content as "
    "clean markdown, with each layout element wrapped in a "
    '<div data-bbox="[x1,y1,x2,y2]" data-label="Category"> tag. '
    "Use HTML tables for any tabular data. "
    "For charts/graphs, use flat combined column headers. "
    "Output ONLY the parsed content with div wrappers, "
    "no explanations."
)

# ---------------------------------------------------------------------------
# Gemini-specific layout prompts — use native [y_min, x_min, y_max, x_max]
# format to avoid intermittent coordinate inversion when asking for [x1,y1,x2,y2].
# Callers must convert with swap_gemini_bbox() after parse_layout_blocks().
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_LAYOUT_GEMINI = SYSTEM_PROMPT_LAYOUT.replace(
    '"[x1, y1, x2, y2]" — bounding box in normalized 0-1000 '
    "coordinates where x is horizontal (left edge = 0, right edge = 1000) "
    "and y is vertical (top = 0, bottom = 1000). "
    "x1,y1 is the top-left corner and x2,y2 is the bottom-right corner.",
    '"[y_min, x_min, y_max, x_max]" — bounding box in normalized 0-1000 '
    "coordinates where x is horizontal (left edge = 0, right edge = 1000) "
    "and y is vertical (top = 0, bottom = 1000). "
    "The order is [y_min, x_min, y_max, x_max].",
)

USER_PROMPT_LAYOUT_GEMINI = USER_PROMPT_LAYOUT.replace(
    "[x1,y1,x2,y2]",
    "[y_min,x_min,y_max,x_max]",
)


# ---------------------------------------------------------------------------
# Absolute-pixel prompt variants (bbox_scale=None in the provider config).
# Some models ground well but are unreliable at re-normalizing their bboxes
# into the 0-1000 range, which conflates grounding quality with coordinate
# compliance. These variants ask for the model's native pixel frame instead;
# build_layout_pages(bbox_scale=None) then normalizes by the image
# dimensions. Built by string replacement so the variants can never drift
# from the base prompts on anything except the coordinate bullet.
# ---------------------------------------------------------------------------

_DATA_BBOX_BULLET_NORM = (
    '"[x1, y1, x2, y2]" — bounding box in normalized 0-1000 '
    "coordinates where x is horizontal (left edge = 0, right edge = 1000) "
    "and y is vertical (top = 0, bottom = 1000). "
    "x1,y1 is the top-left corner and x2,y2 is the bottom-right corner."
)
_DATA_BBOX_BULLET_ABS = (
    '"[x1, y1, x2, y2]" — bounding box in ABSOLUTE PIXEL coordinates '
    "of the image, where x is horizontal (left edge = 0) and y is "
    "vertical (top = 0). "
    "x1,y1 is the top-left corner and x2,y2 is the bottom-right corner."
)

SYSTEM_PROMPT_LAYOUT_ABS = SYSTEM_PROMPT_LAYOUT.replace(_DATA_BBOX_BULLET_NORM, _DATA_BBOX_BULLET_ABS)
assert SYSTEM_PROMPT_LAYOUT_ABS != SYSTEM_PROMPT_LAYOUT, (
    "data-bbox bullet replacement did not match SYSTEM_PROMPT_LAYOUT"
)

USER_PROMPT_LAYOUT_ABS = USER_PROMPT_LAYOUT.replace(
    'data-label="Category"> tag. ',
    'data-label="Category"> tag (data-bbox in absolute pixel coordinates). ',
)
assert USER_PROMPT_LAYOUT_ABS != USER_PROMPT_LAYOUT, "user-prompt replacement did not match USER_PROMPT_LAYOUT"

_DATA_BBOX_BULLET_GEMINI_NORM = (
    '"[y_min, x_min, y_max, x_max]" — bounding box in normalized 0-1000 '
    "coordinates where x is horizontal (left edge = 0, right edge = 1000) "
    "and y is vertical (top = 0, bottom = 1000). "
    "The order is [y_min, x_min, y_max, x_max]."
)
_DATA_BBOX_BULLET_GEMINI_ABS = (
    '"[y_min, x_min, y_max, x_max]" — bounding box in ABSOLUTE PIXEL '
    "coordinates of the image, where x is horizontal (left edge = 0) "
    "and y is vertical (top = 0). "
    "The order is [y_min, x_min, y_max, x_max]."
)

SYSTEM_PROMPT_LAYOUT_GEMINI_ABS = SYSTEM_PROMPT_LAYOUT_GEMINI.replace(
    _DATA_BBOX_BULLET_GEMINI_NORM, _DATA_BBOX_BULLET_GEMINI_ABS
)
assert SYSTEM_PROMPT_LAYOUT_GEMINI_ABS != SYSTEM_PROMPT_LAYOUT_GEMINI, (
    "data-bbox bullet replacement did not match SYSTEM_PROMPT_LAYOUT_GEMINI"
)

USER_PROMPT_LAYOUT_GEMINI_ABS = USER_PROMPT_LAYOUT_GEMINI.replace(
    'data-label="Category"> tag. ',
    'data-label="Category"> tag (data-bbox in absolute pixel coordinates). ',
)
assert USER_PROMPT_LAYOUT_GEMINI_ABS != USER_PROMPT_LAYOUT_GEMINI, (
    "user-prompt replacement did not match USER_PROMPT_LAYOUT_GEMINI"
)


def resolve_layout_prompts(
    bbox_scale: float | None,
    mode: str | None,
    *,
    gemini: bool = False,
    pixel_disallowed_modes: tuple[str, ...] = ("parse_with_layout_file",),
    pixel_frame_supported: bool = False,
) -> tuple[str, str]:
    """Validate *bbox_scale* and return the (system, user) layout prompts.

    ``bbox_scale`` follows the same convention as the layoutdet providers:
    a number means the model's coordinates are on that grid (only 1000 is
    supported here, since the prompt text hardcodes the range), and ``None``
    means absolute pixel coordinates, normalized by the image dimensions.

    Shared by the div-layout providers so the scale validation, the
    scale/mode constraint, and the prompt-pair selection cannot drift
    between them.

    The pixel frame is normalized by the recorded page dimensions, so it
    assumes the model perceives the image at the sent resolution. Provider
    APIs downscale images past their native limits, and the model then
    reports pixels in that downscaled frame while ``normalize()`` divides by
    the dimensions we recorded — so a provider may only opt in
    (``pixel_frame_supported=True``) once it pre-resizes each page to the
    size the model actually perceives. Anthropic does this in
    ``_resize_to_perceived_size``; the other div-layout providers do not,
    and are rejected rather than allowed to report a silently shifted frame.

    Raises:
        ProviderConfigError: for an unsupported scale; for ``bbox_scale=None``
            on a provider that cannot pin the perceived frame; or for
            ``bbox_scale=None`` combined with a mode in
            *pixel_disallowed_modes* (absolute pixel coordinates are
            undefined when the model is sent the PDF file rather than a
            rendered image).
    """
    from extract_bench.inference.providers.base import ProviderConfigError

    if bbox_scale is not None and bbox_scale != 1000:
        raise ProviderConfigError(
            f"Unsupported bbox_scale {bbox_scale!r}. Use 1000 (normalized 0-1000, "
            "the default) or None (absolute pixel coordinates)."
        )
    if bbox_scale is None and not pixel_frame_supported:
        raise ProviderConfigError(
            "bbox_scale=None (pixel coordinates) is not supported by this provider. "
            "The provider API may downscale the page before the model sees it, so "
            "the model's pixel coordinates would be in a different frame than the "
            "recorded page dimensions. Use bbox_scale=1000 (the default), or add a "
            "perceived-size pre-resize to the provider first."
        )
    if bbox_scale is None and mode in pixel_disallowed_modes:
        raise ProviderConfigError(
            f"bbox_scale=None (pixel coordinates) is not supported with mode "
            f"'{mode}'; absolute pixel coordinates are undefined when the model "
            "is sent the PDF file."
        )
    if bbox_scale is None:
        if gemini:
            return SYSTEM_PROMPT_LAYOUT_GEMINI_ABS, USER_PROMPT_LAYOUT_GEMINI_ABS
        return SYSTEM_PROMPT_LAYOUT_ABS, USER_PROMPT_LAYOUT_ABS
    if gemini:
        return SYSTEM_PROMPT_LAYOUT_GEMINI, USER_PROMPT_LAYOUT_GEMINI
    return SYSTEM_PROMPT_LAYOUT, USER_PROMPT_LAYOUT


def swap_gemini_bbox(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Gemini native [y_min, x_min, y_max, x_max] to [x1, y1, x2, y2]."""
    for item in items:
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            y_min, x_min, y_max, x_max = bbox
            item["bbox"] = [x_min, y_min, x_max, y_max]
    return items


# Label mapping (case-insensitive raw label -> canonical label string)
LABEL_MAP: dict[str, str] = {
    "caption": "Caption",
    "footnote": "Footnote",
    "formula": "Formula",
    "list-item": "List-item",
    "list_item": "List-item",
    "page-footer": "Page-footer",
    "page_footer": "Page-footer",
    "page-header": "Page-header",
    "page_header": "Page-header",
    "picture": "Picture",
    "figure": "Picture",
    "section-header": "Section-header",
    "section_header": "Section-header",
    "table": "Table",
    "text": "Text",
    "title": "Title",
}


def split_pdf_to_pages(pdf_path: str) -> list[tuple[bytes, int, int]]:
    """Split a PDF into single-page PDF bytes.

    Returns a list of (pdf_bytes, width_px, height_px) tuples, one per page.
    Width/height are at 72 DPI (PDF points).
    """
    import fitz  # PyMuPDF

    src = fitz.open(pdf_path)
    results: list[tuple[bytes, int, int]] = []
    for page_num in range(len(src)):
        page = src[page_num]
        rect = page.rect
        # Create a single-page PDF in memory
        dst = fitz.open()
        dst.insert_pdf(src, from_page=page_num, to_page=page_num)
        pdf_bytes = dst.tobytes()
        dst.close()
        results.append((pdf_bytes, int(rect.width), int(rect.height)))
    src.close()
    return results


def parse_layout_blocks(content: str) -> list[dict[str, Any]]:
    """Parse <div data-bbox="..." data-label="...">content</div> blocks.

    Handles both attribute orderings. Returns list of dicts with
    'bbox' (list[float]), 'label' (str), and 'text' (str) keys.
    """
    blocks: list[dict[str, Any]] = []

    # Match opening div with both attribute orders
    pattern_bbox_first = re.compile(
        r'<div\s+[^>]*?data-bbox=["\'](\[[^\]]+\])["\'][^>]*?data-label=["\']([^"\']+)["\'][^>]*?>'
        r"([\s\S]*?)</div>",
        re.IGNORECASE,
    )
    pattern_label_first = re.compile(
        r'<div\s+[^>]*?data-label=["\']([^"\']+)["\'][^>]*?data-bbox=["\'](\[[^\]]+\])["\'][^>]*?>'
        r"([\s\S]*?)</div>",
        re.IGNORECASE,
    )

    # Collect all matches with their start positions, then sort by
    # position so mixed attribute orderings preserve document order.
    raw_matches: list[tuple[int, str, str, str]] = []  # (pos, bbox_str, label, text)

    for match in pattern_bbox_first.finditer(content):
        raw_matches.append((match.start(), match.group(1), match.group(2), match.group(3)))

    for match in pattern_label_first.finditer(content):
        raw_matches.append((match.start(), match.group(2), match.group(1), match.group(3)))

    raw_matches.sort(key=lambda m: m[0])

    seen_positions: set[int] = set()
    for pos, bbox_str, label, text in raw_matches:
        if pos in seen_positions:
            continue  # skip duplicate from overlapping patterns
        seen_positions.add(pos)
        try:
            bbox = json.loads(bbox_str)
            if isinstance(bbox, list) and len(bbox) == 4:
                blocks.append({"bbox": bbox, "label": label, "text": text.strip()})
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse bbox: {bbox_str}")

    return blocks


# A markdown ATX heading marker: optional indent, 1-6 hashes, then whitespace.
# The trailing whitespace is required (as in CommonMark), so literal text like
# "#1 Best Seller" is not mistaken for a marker.
_LEADING_HEADING_MARKER = re.compile(r"^[ \t]*(#{1,6})[ \t]+")


def _strip_heading_marker(text: str) -> str:
    """Strip a leading markdown heading marker, if any, from item text."""
    match = _LEADING_HEADING_MARKER.match(text)
    if not match:
        return text
    bare = text[match.end() :].strip()
    return bare if bare else text


def _strip_outer_formula_delimiters(text: str) -> str:
    """Strip one outer pair of ``$$``/``$`` delimiters when unambiguous.

    Only strips when the interior contains no further ``$``, so multi-formula
    items (``$a$ + $b$``) and text with interior dollar signs pass through
    verbatim rather than being corrupted.
    """
    stripped = text.strip()
    for delim in ("$$", "$"):
        if len(stripped) > 2 * len(delim) and stripped.startswith(delim) and stripped.endswith(delim):
            interior = stripped[len(delim) : -len(delim)]
            if "$" not in interior and interior.strip():
                return interior.strip()
    return text


def items_to_markdown(items: list[dict[str, Any]]) -> str:
    """Assemble clean markdown from parsed layout items.

    The layout prompt asks for clean markdown inside each div, so models
    frequently emit heading markers (``# Heading``) and formula delimiters
    (``$$...$$``) in the item text already. Any existing markers are stripped
    before the label-derived ones are applied, so they don't double up
    (``# # Heading``, nested ``$$`` blocks) — doubled markers defeat the
    heading/formula detection in downstream scoring. Title still always
    renders as H1 and Section-header as H2.
    """
    parts: list[str] = []
    for item in items:
        label = item.get("label", "").lower()
        text = item.get("text", "")
        if not text:
            continue
        if label == "title":
            parts.append(f"# {_strip_heading_marker(text)}")
        elif label in ("section-header", "section_header"):
            parts.append(f"## {_strip_heading_marker(text)}")
        elif label == "formula":
            parts.append(f"$$\n{_strip_outer_formula_delimiters(text)}\n$$")
        else:
            parts.append(text)
    return "\n\n".join(parts)


# Coordinates may overshoot their grid by rounding noise; values beyond
# scale * this tolerance indicate the model did not follow the coordinate
# instruction (see the compliance warning in ``build_layout_pages``).
_BBOX_ROUNDING_TOLERANCE = 1.05


def _is_finite_bbox(bbox: Any) -> bool:
    """True when *bbox* is four finite numeric (non-bool) values."""
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bbox)
    )


def build_layout_pages(
    items: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    markdown: str,
    page_number: int = 1,
    bbox_scale: float | None = 1000.0,
) -> list[ParseLayoutPageIR]:
    """Convert parsed layout blocks to ParseLayoutPageIR.

    Args:
        items: Parsed layout blocks from ``parse_layout_blocks``.
        image_width: Page image width in pixels.
        image_height: Page image height in pixels.
        markdown: Page markdown content.
        page_number: 1-indexed page number.
        bbox_scale: The grid the bboxes are on (same convention as the
            layoutdet providers): a number divides coordinates by that
            scale; ``None`` means absolute pixel coordinates (the
            ``bbox_scale=None`` pipelines, prompted with the ``*_ABS``
            variants), normalized by the image dimensions.
    """
    if not items or not image_width or not image_height:
        return []

    if bbox_scale is None:
        scale_x = float(image_width)
        scale_y = float(image_height)
    else:
        scale_x = scale_y = float(bbox_scale)

    compliance_warned = False
    layout_items: list[LayoutItemIR] = []
    for item in items:
        bbox = item.get("bbox", [])
        label_raw = item.get("label", "text")
        text = item.get("text", "")

        # Clamping a non-finite value would silently place it at the page
        # edge (min(scale, nan) returns scale), so skip boxes the clamp
        # cannot handle honestly.
        if not _is_finite_bbox(bbox):
            if len(bbox) == 4:
                logger.warning(f"Skipping layout item with non-numeric bbox: {bbox!r}")
            continue

        if (
            bbox_scale is not None
            and not compliance_warned
            and any(v > bbox_scale * _BBOX_ROUNDING_TOLERANCE for v in bbox)
        ):
            logger.warning(
                f"Page {page_number}: bbox coordinates exceed the 0-{bbox_scale:g} "
                "range; the model may be emitting absolute pixels (consider "
                "bbox_scale=None)"
            )
            compliance_warned = True

        x1, y1, x2, y2 = bbox

        # Clamp to the page so out-of-range values cannot produce segments
        # outside the unit square.
        x1 = max(0.0, min(scale_x, x1))
        x2 = max(0.0, min(scale_x, x2))
        y1 = max(0.0, min(scale_y, y1))
        y2 = max(0.0, min(scale_y, y2))

        # Convert [x1,y1,x2,y2] to normalized [0,1] COCO [x,y,w,h]
        nx = x1 / scale_x
        ny = y1 / scale_y
        nw = (x2 - x1) / scale_x
        nh = (y2 - y1) / scale_y

        label = LABEL_MAP.get(label_raw.lower(), "Text")
        seg = LayoutSegmentIR(x=nx, y=ny, w=nw, h=nh, confidence=1.0, label=label)

        norm_label = label_raw.lower()
        if norm_label == "table":
            item_type = "table"
        elif norm_label in ("picture", "figure"):
            item_type = "image"
        else:
            item_type = "text"

        layout_items.append(LayoutItemIR(type=item_type, value=text, bbox=seg, layout_segments=[seg]))

    if not layout_items:
        return []

    return [
        ParseLayoutPageIR(
            page_number=page_number,
            width=float(image_width),
            height=float(image_height),
            md=markdown,
            items=layout_items,
        )
    ]
