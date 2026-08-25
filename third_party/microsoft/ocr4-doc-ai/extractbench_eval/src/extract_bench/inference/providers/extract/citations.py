"""Helpers for normalizing provider field citation bboxes."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from extract_bench.schemas.extract_output import FieldCitation

logger = logging.getLogger(__name__)

_STRUCTURAL_KEYS = {
    "citation",
    "citations",
    "document_metadata",
    "field_metadata",
    "fields",
    "metadata",
    "page_metadata",
    "properties",
    "row_metadata",
}


def extract_extend_field_citations(raw_output: Mapping[str, Any]) -> list[FieldCitation]:
    """Extract citations from Extend AI processor-run metadata."""
    output = _as_mapping(_as_mapping(raw_output.get("processor_run")).get("output"))
    metadata = _as_mapping(output.get("metadata"))
    return _dedupe(_collect_field_map(metadata, source="extend"))


def extract_reducto_field_citations(raw_output: Mapping[str, Any]) -> list[FieldCitation]:
    """Extract citations from Reducto Extract result payloads."""
    result = raw_output.get("result")
    if isinstance(result, list) and result:
        result = result[0]
    return _dedupe(_collect_recursive(node=_as_mapping(result), source="reducto", path=[]))


def extract_landingai_field_citations(
    raw_output: Mapping[str, Any],
    *,
    pdf_path: str | Path | None = None,
    extracted_data: Mapping[str, Any] | None = None,
) -> list[FieldCitation]:
    """Extract citations from LandingAI extraction references and parse grounding.

    LandingAI's API ships a single ``chunkText`` grounding box for whole header
    sections, then references the same box from every cover-page scalar field
    (e.g., ``filing_manager_name``, ``report_calendar_quarter_end`` all share
    one UUID under ``parse_response.grounding``). When ``pdf_path`` and
    ``extracted_data`` are supplied, those coarse chunk bboxes are refined to a
    tight per-value bbox by text-searching for the extracted value within the
    chunk's clip rectangle (PyMuPDF ``page.search_for``). Cell-level groundings
    (``tableCell``) are already tight and pass through unchanged.
    """
    extraction_metadata = _as_mapping(raw_output.get("extraction_metadata"))
    parse_response = _as_mapping(raw_output.get("parse_response"))
    grounding = _as_mapping(parse_response.get("grounding"))
    if not extraction_metadata or not grounding:
        return []
    citations = _dedupe(
        _collect_landingai_metadata_references(
            node=extraction_metadata,
            grounding=grounding,
            path=[],
        )
    )
    if pdf_path is not None and extracted_data is not None:
        citations = _refine_chunk_citations_via_text_search(
            citations,
            pdf_path=Path(pdf_path),
            extracted_data=extracted_data,
        )
    return citations


def extract_llamaextract_field_citations(metadata: Any, *, source: str) -> list[FieldCitation]:
    """Extract citations from LlamaExtract metadata in known and fallback shapes."""
    metadata_map = _as_mapping(metadata)
    if not metadata_map:
        return []

    citations: list[FieldCitation] = []

    for key in ("field_metadata", "document_metadata", "fields"):
        citations.extend(_collect_field_map(_as_mapping(metadata_map.get(key)), source=source))

    for key in ("page_metadata", "row_metadata"):
        entries = metadata_map.get(key)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
            continue
        for entry in entries:
            entry_map = _as_mapping(entry)
            default_page = _extract_page(entry_map)
            default_dimensions = _extract_dimensions(entry_map)
            for field_key in ("field_metadata", "document_metadata", "fields"):
                citations.extend(
                    _collect_field_map(
                        _as_mapping(entry_map.get(field_key)),
                        source=source,
                        default_page=default_page,
                        default_dimensions=default_dimensions,
                    )
                )

    citations.extend(_collect_recursive(node=metadata_map, source=source, path=[]))
    return _dedupe(citations)


def _collect_field_map(
    field_map: Mapping[str, Any],
    *,
    source: str,
    default_page: int | None = None,
    default_dimensions: tuple[float, float] | None = None,
) -> list[FieldCitation]:
    citations: list[FieldCitation] = []
    for field_path, node in field_map.items():
        if field_path.startswith("_"):
            continue
        citations.extend(
            _collect_node_citations(
                field_path=field_path,
                node=node,
                source=source,
                default_page=default_page,
                default_dimensions=default_dimensions,
            )
        )
    return citations


def _collect_node_citations(
    *,
    field_path: str,
    node: Any,
    source: str,
    default_page: int | None,
    default_dimensions: tuple[float, float] | None,
) -> list[FieldCitation]:
    node_map = _as_mapping(node)
    if not node_map:
        return []

    page = _extract_page(node_map) or default_page
    dimensions = _extract_dimensions(node_map) or default_dimensions
    citations: list[FieldCitation] = []
    for citation in _iter_citation_entries(node_map):
        citations.extend(
            _normalize_citation(
                field_path=field_path,
                citation=citation,
                source=source,
                default_page=page,
                default_dimensions=dimensions,
                default_confidence=_extract_confidence(node_map),
            )
        )
    if not citations:
        # Confidence-only fallback: when a node has confidence but no bbox/citation
        # entries, emit a page-only FieldCitation so the metric can still read the
        # signal. If descendants later emit their own citations they keep distinct
        # field paths, so both coexist without double-counting — the consumer
        # picks the leaf via exact-path lookup and falls back to the parent only
        # when nothing leaf-level matches.
        confidence = _extract_confidence(node_map)
        if confidence is not None:
            citations.append(
                FieldCitation(
                    field_path=field_path,
                    page=page or 1,
                    bbox=None,
                    source=source,
                    confidence=confidence,
                    metadata=_compact_metadata(node_map),
                )
            )
    return citations


def _iter_citation_entries(node: Mapping[str, Any]) -> list[Any]:
    """Iterate citation entries supporting both plural `citations` and singular `citation` keys."""
    entries: list[Any] = []
    for key in ("citations", "citation"):
        for entry in _as_sequence(node.get(key)):
            entries.append(entry)
    return entries


def _collect_recursive(*, node: Any, source: str, path: list[str]) -> list[FieldCitation]:
    node_map = _as_mapping(node)
    if not node_map:
        return []

    citations: list[FieldCitation] = []
    explicit_path = _extract_field_path(node_map)
    field_path = explicit_path or _format_field_path(path)
    if field_path:
        for citation in _iter_citation_entries(node_map):
            citations.extend(
                _normalize_citation(
                    field_path=field_path,
                    citation=citation,
                    source=source,
                    default_page=_extract_page(node_map),
                    default_dimensions=_extract_dimensions(node_map),
                    default_confidence=_extract_confidence(node_map),
                )
            )
        if not citations:
            # See _collect_node_citations: parent-level confidence-only citations
            # coexist with any leaf citations from the descendant walk below.
            # Distinct field paths keep them from colliding in the consumer maps.
            confidence = _extract_confidence(node_map)
            if confidence is not None:
                citations.append(
                    FieldCitation(
                        field_path=field_path,
                        page=_extract_page(node_map) or 1,
                        bbox=None,
                        source=source,
                        confidence=confidence,
                        metadata=_compact_metadata(node_map),
                    )
                )

    for key, value in node_map.items():
        if key in ("citations", "citation"):
            continue
        next_path = path if key in _STRUCTURAL_KEYS else [*path, key]
        if isinstance(value, Mapping):
            citations.extend(_collect_recursive(node=value, source=source, path=next_path))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                item_path = next_path if key in _STRUCTURAL_KEYS else [*next_path, f"[{index}]"]
                citations.extend(_collect_recursive(node=item, source=source, path=item_path))

    return citations


def _collect_landingai_metadata_references(
    *,
    node: Any,
    grounding: Mapping[str, Any],
    path: list[str],
) -> list[FieldCitation]:
    node_map = _as_mapping(node)
    if node_map:
        field_path = _format_field_path(path)
        references = _as_sequence(node_map.get("references"))
        if field_path and references:
            citations: list[FieldCitation] = []
            reference_text = _extract_reference_text(node_map)
            for reference in references:
                if not isinstance(reference, str) or not reference:
                    continue
                grounding_entry = _as_mapping(grounding.get(reference))
                citation = _landingai_reference_to_citation(
                    field_path=field_path,
                    reference=reference,
                    reference_text=reference_text,
                    grounding_entry=grounding_entry,
                )
                if citation is not None:
                    citations.append(citation)
            return citations

        citations = []
        for key, value in node_map.items():
            if key in {"references", "value"}:
                continue
            citations.extend(
                _collect_landingai_metadata_references(
                    node=value,
                    grounding=grounding,
                    path=[*path, key],
                )
            )
        return citations

    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        citations = []
        for index, item in enumerate(node):
            citations.extend(
                _collect_landingai_metadata_references(
                    node=item,
                    grounding=grounding,
                    path=[*path, f"[{index}]"],
                )
            )
        return citations

    return []


def _landingai_reference_to_citation(
    *,
    field_path: str,
    reference: str,
    reference_text: str | None,
    grounding_entry: Mapping[str, Any],
) -> FieldCitation | None:
    box = _as_mapping(grounding_entry.get("box"))
    left = _coerce_float(box.get("left"))
    top = _coerce_float(box.get("top"))
    right = _coerce_float(box.get("right"))
    bottom = _coerce_float(box.get("bottom"))
    if left is None or top is None or right is None or bottom is None:
        return None

    bbox = _normalize_bbox([left, top, right - left, bottom - top], None)
    if bbox is None:
        return None

    page_index = _coerce_int(grounding_entry.get("page"))
    page = page_index + 1 if page_index is not None and page_index >= 0 else 1
    confidence = _extract_confidence(grounding_entry)
    metadata = {"reference": reference}
    grounding_type = grounding_entry.get("type")
    if isinstance(grounding_type, str):
        metadata["type"] = grounding_type

    return FieldCitation(
        field_path=field_path,
        page=page,
        bbox=bbox,
        reference_text=reference_text,
        confidence=confidence,
        source="landingai",
        metadata=metadata,
    )


def _format_field_path(path: list[str]) -> str:
    """Render path tokens so list-index tokens (`[N]`) attach to the prior key without a dot.

    GT field paths use bracket notation (`employees[0].basic_salary`). We collect tokens during
    the recursive walk and convert any leading-bracket tokens into bracket-joined segments so
    predictions match GT field path scope.
    """
    rendered = ""
    for token in path:
        if token.startswith("[") and token.endswith("]"):
            rendered += token
        elif rendered:
            rendered += "." + token
        else:
            rendered = token
    return rendered


def _normalize_citation(
    *,
    field_path: str,
    citation: Any,
    source: str,
    default_page: int | None,
    default_dimensions: tuple[float, float] | None,
    default_confidence: float | None = None,
) -> list[FieldCitation]:
    citation_map = _as_mapping(citation)
    if not citation_map:
        return []

    page = _extract_page(citation_map) or _extract_nested_bbox_page(citation_map) or default_page or 1
    dimensions = _extract_dimensions(citation_map) or default_dimensions
    polygon = _extract_polygon(citation_map)
    reference_text = _extract_reference_text(citation_map)
    confidence = _extract_confidence(citation_map)
    if confidence is None:
        confidence = default_confidence
    metadata = _compact_metadata(citation_map)

    plural_bboxes = _extract_bbox_list(citation_map)
    if plural_bboxes:
        normalized_polygon = _normalize_polygon(polygon, dimensions) if polygon is not None else None
        results: list[FieldCitation] = []
        for entry_bbox in plural_bboxes:
            normalized_bbox = _normalize_bbox(entry_bbox, dimensions)
            if normalized_bbox is None:
                continue
            results.append(
                FieldCitation(
                    field_path=field_path,
                    page=page,
                    bbox=normalized_bbox,
                    polygon=normalized_polygon,
                    reference_text=reference_text,
                    confidence=confidence,
                    source=source,
                    metadata=metadata,
                )
            )
        if results:
            return results
        # All entries failed to normalize — the bbox attempt was real but
        # garbage. Drop rather than emit a misleading page-only fallback.
        return []

    raw_bbox = _bbox_from_polygon(polygon) if polygon is not None else _extract_bbox(citation_map)
    normalized_bbox = _normalize_bbox(raw_bbox, dimensions)
    normalized_polygon = _normalize_polygon(polygon, dimensions) if polygon is not None else None
    if normalized_bbox is None and not _has_bbox_attempt(citation_map):
        # v0.2: page-only citation (e.g. Extend metadata with `{page: N}` and no
        # bbox/polygon keys at all). Distinct from a malformed citation where a
        # bbox key was present but failed to parse — those still get dropped.
        return [
            FieldCitation(
                field_path=field_path,
                page=page,
                bbox=None,
                polygon=normalized_polygon,
                reference_text=reference_text,
                confidence=confidence,
                source=source,
                metadata=metadata,
            )
        ]
    if normalized_bbox is None:
        return []
    return [
        FieldCitation(
            field_path=field_path,
            page=page,
            bbox=normalized_bbox,
            polygon=normalized_polygon,
            reference_text=reference_text,
            confidence=confidence,
            source=source,
            metadata=metadata,
        )
    ]


def _extract_bbox_list(node: Mapping[str, Any]) -> list[list[float]] | None:
    """Extract a plural list of bboxes if `bounding_boxes` is present.

    Each entry can be either a 4-element [x, y, w, h] sequence or a mapping with
    x/y/w/h or x1/y1/x2/y2 keys.
    """
    raw = node.get("bounding_boxes")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    if not raw:
        return None
    bboxes: list[list[float]] = []
    for entry in raw:
        bbox: list[float] | None = None
        if isinstance(entry, Mapping):
            bbox = _bbox_from_mapping(entry)
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
            bbox = _bbox_from_sequence(entry)
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes or None


def _extract_field_path(node: Mapping[str, Any]) -> str | None:
    for key in ("field_path", "fieldPath", "path", "field", "name", "key"):
        value = node.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_page(node: Mapping[str, Any]) -> int | None:
    for key in ("page", "page_number", "pageNumber"):
        value = _coerce_int(node.get(key))
        if value is not None and value >= 1:
            return value
    for key in ("page_index", "pageIndex"):
        value = _coerce_int(node.get(key))
        if value is not None and value >= 0:
            return value + 1
    return None


def _extract_nested_bbox_page(node: Mapping[str, Any]) -> int | None:
    for key in ("bbox", "bounding_box", "boundingBox", "box"):
        page = _extract_page(_as_mapping(node.get(key)))
        if page is not None:
            return page
    return None


def _extract_dimensions(node: Mapping[str, Any]) -> tuple[float, float] | None:
    width = _coerce_float(_first_present(node, ("page_width", "pageWidth", "width", "image_width", "imageWidth")))
    height = _coerce_float(_first_present(node, ("page_height", "pageHeight", "height", "image_height", "imageHeight")))
    if width is not None and height is not None and width > 0 and height > 0:
        return width, height

    for key in ("page_dimensions", "pageDimensions", "page_size", "pageSize", "dimensions", "image_size", "imageSize"):
        size = _as_mapping(node.get(key))
        width = _coerce_float(_first_present(size, ("width", "w")))
        height = _coerce_float(_first_present(size, ("height", "h")))
        if width is not None and height is not None and width > 0 and height > 0:
            return width, height
    return None


def _extract_bbox(node: Mapping[str, Any]) -> list[float] | None:
    for key in ("bbox", "bounding_box", "boundingBox", "box"):
        bbox = node.get(key)
        bbox_from_dict = _bbox_from_mapping(_as_mapping(bbox))
        if bbox_from_dict is not None:
            return bbox_from_dict
        bbox_from_sequence = _bbox_from_sequence(bbox)
        if bbox_from_sequence is not None:
            return bbox_from_sequence

    bbox_from_dict = _bbox_from_mapping(node)
    if bbox_from_dict is not None:
        return bbox_from_dict
    return None


def _has_bbox_attempt(node: Mapping[str, Any]) -> bool:
    """True when the citation map contains a non-null bbox/polygon-shaped key.

    Used to distinguish v0.2 page-only citations (no bbox metadata at all, e.g.
    Extend's `{page: 1}` shape) from malformed citations where a bbox key was
    provided but unparseable. Page-only citations are emitted; malformed are
    silently dropped, matching pre-v0.2 behavior on bad input.
    """
    bbox_keys = (
        "bbox",
        "bounding_box",
        "boundingBox",
        "box",
        "bounding_boxes",
        "polygon",
        "bounding_polygon",
        "boundingPolygon",
        "points",
        "vertices",
    )
    for key in bbox_keys:
        if key in node and node[key] is not None:
            return True
    # Inline x/y/w/h or x1/y1/x2/y2 also count as a bbox attempt.
    inline_keys = ("x", "y", "w", "h", "left", "top", "width", "height", "x1", "y1", "x2", "y2", "right", "bottom")
    return any(key in node and node[key] is not None for key in inline_keys)


def _bbox_from_mapping(node: Mapping[str, Any]) -> list[float] | None:
    if not node:
        return None

    x = _coerce_float(_first_present(node, ("x", "left")))
    y = _coerce_float(_first_present(node, ("y", "top")))
    width = _coerce_float(_first_present(node, ("w", "width")))
    height = _coerce_float(_first_present(node, ("h", "height")))
    if x is not None and y is not None and width is not None and height is not None:
        return [x, y, width, height]

    x1 = _coerce_float(_first_present(node, ("x1", "left")))
    y1 = _coerce_float(_first_present(node, ("y1", "top")))
    x2 = _coerce_float(_first_present(node, ("x2", "right")))
    y2 = _coerce_float(_first_present(node, ("y2", "bottom")))
    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
        return [x1, y1, x2 - x1, y2 - y1]
    return None


def _bbox_from_sequence(raw: Any) -> list[float] | None:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(raw) != 4:
        return None
    values = [_coerce_float(value) for value in raw]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values if value is not None]


def _extract_polygon(node: Mapping[str, Any]) -> list[list[float]] | None:
    for key in ("polygon", "bounding_polygon", "boundingPolygon", "points", "vertices"):
        polygon = _polygon_from_raw(node.get(key))
        if polygon is not None:
            return polygon
    return None


def _polygon_from_raw(raw: Any) -> list[list[float]] | None:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    if not raw:
        return None

    points: list[list[float]] = []
    if all(isinstance(point, Mapping) for point in raw):
        for point in raw:
            point_map = _as_mapping(point)
            x = _coerce_float(point_map.get("x"))
            y = _coerce_float(point_map.get("y"))
            if x is None or y is None:
                return None
            points.append([x, y])
    elif all(isinstance(point, Sequence) and not isinstance(point, (str, bytes, bytearray)) for point in raw):
        for point in raw:
            if len(point) < 2:
                return None
            x = _coerce_float(point[0])
            y = _coerce_float(point[1])
            if x is None or y is None:
                return None
            points.append([x, y])
    else:
        values = [_coerce_float(value) for value in raw]
        if len(values) % 2 != 0 or any(value is None for value in values):
            return None
        numeric_values = [float(value) for value in values if value is not None]
        points = [[numeric_values[index], numeric_values[index + 1]] for index in range(0, len(numeric_values), 2)]

    return points if len(points) >= 2 else None


def _bbox_from_polygon(polygon: list[list[float]] | None) -> list[float] | None:
    if not polygon:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    left = min(xs)
    top = min(ys)
    return [left, top, max(xs) - left, max(ys) - top]


def _normalize_bbox(raw_bbox: list[float] | None, dimensions: tuple[float, float] | None) -> list[float] | None:
    if raw_bbox is None or len(raw_bbox) != 4:
        return None
    x, y, width, height = raw_bbox
    if width <= 0 or height <= 0:
        return None

    if _looks_normalized(raw_bbox):
        normalized = raw_bbox
    elif dimensions is not None:
        page_width, page_height = dimensions
        normalized = [x / page_width, y / page_height, width / page_width, height / page_height]
    else:
        return None

    if not _looks_normalized(normalized):
        return None
    return [round(value, 8) for value in normalized]


def _normalize_polygon(
    polygon: list[list[float]] | None,
    dimensions: tuple[float, float] | None,
) -> list[list[float]] | None:
    if polygon is None:
        return None
    flat = [coordinate for point in polygon for coordinate in point]
    if all(0 <= value <= 1 for value in flat):
        return [[round(point[0], 8), round(point[1], 8)] for point in polygon]
    if dimensions is None:
        return None
    page_width, page_height = dimensions
    normalized = [[point[0] / page_width, point[1] / page_height] for point in polygon]
    if not all(0 <= value <= 1 for point in normalized for value in point):
        return None
    return [[round(point[0], 8), round(point[1], 8)] for point in normalized]


def _looks_normalized(bbox: list[float]) -> bool:
    x, y, width, height = bbox
    return (
        0 <= x <= 1
        and 0 <= y <= 1
        and 0 < width <= 1
        and 0 < height <= 1
        and x + width <= 1.000001
        and y + height <= 1.000001
    )


def _extract_reference_text(node: Mapping[str, Any]) -> str | None:
    value = _first_present(
        node, ("reference_text", "referenceText", "matching_text", "matchingText", "text", "content", "value")
    )
    if isinstance(value, str):
        return value
    return None


def _extract_confidence(node: Mapping[str, Any]) -> float | None:
    confidence = _coerce_float(_first_present(node, ("confidence", "extraction_confidence", "score", "probability")))
    if confidence is None or math.isnan(confidence) or math.isinf(confidence):
        return None
    return confidence


def _compact_metadata(node: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = {
        key: value
        for key, value in node.items()
        if key
        not in {
            "bbox",
            "bounding_box",
            "boundingBox",
            "box",
            "bounding_boxes",
            "polygon",
            "bounding_polygon",
            "boundingPolygon",
            "points",
            "vertices",
        }
    }
    return dict(metadata) if metadata else None


def _dedupe(citations: list[FieldCitation]) -> list[FieldCitation]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[FieldCitation] = []
    for citation in citations:
        key = (
            citation.field_path,
            citation.page,
            tuple(citation.bbox) if citation.bbox is not None else None,
            citation.reference_text,
            citation.source,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _first_present(node: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in node:
            return node[key]
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Chunk-bbox refinement (LandingAI cover-scalar fix)
#
# LandingAI's ``extraction_metadata.references`` for cover-page scalars all
# point to the same chunk-level grounding UUID — one rectangle covering the
# whole header section, reused by every leaf field. We refine those coarse
# bboxes by text-searching for the extracted value within the chunk
# rectangle, replacing the coarse bbox with a tight value-anchored bbox when
# found. Cell-level groundings (``tableCell``) are already tight; gating on
# ``metadata.type == "chunkText"`` keeps them untouched.
# ---------------------------------------------------------------------------

# Grounding types whose bboxes are eligible for value-text refinement. Add
# new entries here only after confirming the source bbox is genuinely coarse
# (a section/line block, not an already-tight cell).
_REFINABLE_GROUNDING_TYPES = frozenset({"chunkText"})


def _refine_chunk_citations_via_text_search(
    citations: list[FieldCitation],
    *,
    pdf_path: Path,
    extracted_data: Mapping[str, Any],
) -> list[FieldCitation]:
    """Refine coarse chunk-level citations to tight value-anchored bboxes.

    For each citation whose ``metadata.type`` is in
    :data:`_REFINABLE_GROUNDING_TYPES`, look up the extracted value at
    ``citation.field_path``, open the PDF page, clip to the citation's bbox,
    and run :func:`fitz.Page.search_for` for the value text. If found, replace
    the bbox with the search hit. Otherwise keep the original.

    Falls back gracefully (returns ``citations`` unchanged) when:

    * ``pdf_path`` does not exist or fails to open
    * The citation's grounding type is not refinable
    * The extracted value cannot be resolved or has no useful text form
    * No search hit lands inside the chunk rectangle
    """
    import fitz  # PyMuPDF; declared as a hard dep in pyproject.toml

    if not pdf_path.exists():
        logger.debug("PDF not found at %s; skipping chunk-citation refinement.", pdf_path)
        return citations

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to open PDF %s for chunk refinement: %s", pdf_path, exc)
        return citations

    refined: list[FieldCitation] = []
    try:
        for citation in citations:
            metadata = citation.metadata or {}
            grounding_type = metadata.get("type")
            if grounding_type not in _REFINABLE_GROUNDING_TYPES:
                refined.append(citation)
                continue
            value = _get_value_at_field_path(extracted_data, citation.field_path)
            queries = _value_search_candidates(value)
            if not queries:
                refined.append(citation)
                continue
            page_index = (citation.page or 1) - 1
            if page_index < 0 or page_index >= document.page_count:
                refined.append(citation)
                continue
            if citation.bbox is None or len(citation.bbox) != 4:
                refined.append(citation)
                continue
            page = document[page_index]
            page_width, page_height = page.rect.width, page.rect.height
            x, y, w, h = citation.bbox
            clip = fitz.Rect(
                x * page_width,
                y * page_height,
                (x + w) * page_width,
                (y + h) * page_height,
            )
            new_bbox = _search_for_value_bbox(page, queries, clip, page_width, page_height)
            if new_bbox is None:
                refined.append(citation)
                continue
            refined_metadata = dict(metadata)
            refined_metadata["refined_from"] = grounding_type
            refined.append(citation.model_copy(update={"bbox": new_bbox, "metadata": refined_metadata}))
    finally:
        document.close()

    return refined


def _search_for_value_bbox(
    page: Any,
    queries: list[str],
    clip: Any,
    page_width: float,
    page_height: float,
) -> list[float] | None:
    """Return a normalized ``[x, y, w, h]`` bbox for a query hit inside ``clip``.

    Queries are tried in order; the first query producing any hit wins.
    Within a query, :func:`fitz.Page.search_for` returns rectangles in
    reading order (top-to-bottom, left-to-right), so we pick ``rects[0]`` —
    the first occurrence of the value text inside the chunk. This matches
    the typical SEC-form layout convention where label and value sit on the
    same line in reading order, so the first-hit value bbox is the one a
    human annotator would have picked.

    We empirically verified this against gold annotations on the SEC-form
    documents in the corpus; alternative tie-breakers like "closest to chunk center"
    regress the metric on real cover pages where the same value appears once
    at the cover header and once again later in the chunk (e.g. the filer
    name on an NPORT cover repeats in a series-info row below).

    Caveat: when a value is so short and ambiguous that it legitimately
    appears multiple times inside the chunk (a bare ``"52"``), the first
    occurrence may not be the one the schema field refers to. Callers that
    are sensitive to this should ship a tighter chunk bbox or use cell-level
    groundings — both of which the LandingAI provider already does for
    ``tableCell`` references, so the failure mode is bounded to coarse
    ``chunkText`` references.
    """
    if page_width <= 0 or page_height <= 0:
        return None
    for query in queries:
        try:
            rects = page.search_for(query, clip=clip)
        except Exception:  # pragma: no cover - defensive against fitz quirks
            continue
        if not rects:
            continue
        chosen = rects[0]
        return [
            chosen.x0 / page_width,
            chosen.y0 / page_height,
            (chosen.x1 - chosen.x0) / page_width,
            (chosen.y1 - chosen.y0) / page_height,
        ]
    return None


def _value_search_candidates(value: Any) -> list[str]:
    """Build a deduped list of plausible PDF text variants for an extracted value.

    Returns an empty list when the value is not searchable (None, empty, bool).
    For ints/floats: includes the bare numeric, comma-grouped, and ``$``-prefixed
    forms. For ISO dates ``YYYY-MM-DD``: also tries ``MM-DD-YYYY`` and
    ``MM/DD/YYYY`` (common SEC-form rendering). For strings: returns the
    stripped value. Order is preserved (first occurrence wins) so callers
    that try queries in order still get the most-specific form first.
    """
    if value is None or isinstance(value, bool):
        return []
    candidates: list[str] = []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        candidates.append(stripped)
        if len(stripped) == 10 and stripped[4] == "-" and stripped[7] == "-":
            year, month, day = stripped[:4], stripped[5:7], stripped[8:10]
            if year.isdigit() and month.isdigit() and day.isdigit():
                candidates.append(f"{month}-{day}-{year}")
                candidates.append(f"{month}/{day}/{year}")
    elif isinstance(value, int):
        formatted_int = f"{value:,}"
        candidates.extend([str(value), formatted_int, f"${formatted_int}"])
    elif isinstance(value, float):
        candidates.append(f"{value:g}")
        if value.is_integer():
            as_int = int(value)
            candidates.extend([str(as_int), f"{as_int:,}", f"${as_int:,}"])
    else:
        return []
    # Order-preserving dedupe: a small integer like ``52`` produces ``"52"``,
    # ``"52"`` again from comma-grouped, and ``"$52"`` — three searches collapse
    # to two distinct queries.
    return list(dict.fromkeys(candidates))


def _get_value_at_field_path(data: Mapping[str, Any] | Any, field_path: str) -> Any:
    """Resolve dotted/bracketed field paths like ``holdings[0].issuer_name``.

    Returns ``None`` when any segment cannot be resolved or the path is
    malformed (e.g. unclosed brackets — the tokenizer signals this by
    returning an empty list, which we must not silently treat as "root
    value" since the caller asked for a real path).
    """
    if not field_path:
        return None
    tokens = _tokenize_field_path(field_path)
    if not tokens:
        return None
    node: Any = data
    for token in tokens:
        if isinstance(token, int):
            if not isinstance(node, list) or token < 0 or token >= len(node):
                return None
            node = node[token]
        else:
            if not isinstance(node, Mapping) or token not in node:
                return None
            node = node[token]
    return node


def _tokenize_field_path(field_path: str) -> list[Any]:
    """Split ``a.b[2].c`` into ``["a", "b", 2, "c"]``."""
    tokens: list[Any] = []
    current = ""
    i = 0
    while i < len(field_path):
        ch = field_path[i]
        if ch == ".":
            if current:
                tokens.append(current)
                current = ""
            i += 1
            continue
        if ch == "[":
            if current:
                tokens.append(current)
                current = ""
            close = field_path.find("]", i)
            if close == -1:
                return []
            try:
                tokens.append(int(field_path[i + 1 : close]))
            except ValueError:
                return []
            i = close + 1
            continue
        current += ch
        i += 1
    if current:
        tokens.append(current)
    return tokens


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        if parsed.is_integer():
            return int(parsed)
    return None
