"""Central utilities for layout label and ontology mapping."""

from __future__ import annotations

from extract_bench.schemas.layout_ontology import (
    CANONICAL_TO_BASIC,
    DEFAULT_LAYOUT_EVALUATION_ONTOLOGY,
    SUPPORTED_LAYOUT_EVALUATION_ONTOLOGIES,
    BasicLabel,
    CanonicalLabel,
)


class UnknownRawLayoutLabelError(ValueError):
    """Raised when a raw provider label cannot be mapped."""


SUPPORTED_EVALUATION_ONTOLOGIES = SUPPORTED_LAYOUT_EVALUATION_ONTOLOGIES
DEFAULT_EVALUATION_ONTOLOGY = DEFAULT_LAYOUT_EVALUATION_ONTOLOGY


# LlamaParse V3-only labels (not in V2)
LLAMAPARSE_V3_ONLY_LABELS = frozenset(
    {
        "caption",
        "list-item",
        "page-footer",
        "page-header",
        "picture",
        "section-header",
        "title",
        "document-index",
        "code",
        "checkbox-selected",
        "checkbox-unselected",
        "form",
        "key-value-region",
    }
)


# LlamaParse V2-only labels (not in V3)
LLAMAPARSE_V2_ONLY_LABELS = frozenset(
    {
        "doc_title",
        "paragraph_title",
        "image",
        "header",
        "footer",
        "reference",
        "algorithm",
        "figure_title",
        "number",
        "abstract",
        "content",
        "aside_text",
        "seal",
        "reference_content",
        "formula_number",
    }
)


# String label -> (CanonicalLabel, attributes)
# Mappings intentionally shared between inference-time and evaluation-time adaptation.
LLAMAPARSE_V2_RAW_TO_CANONICAL: dict[str, tuple[CanonicalLabel, dict[str, str]]] = {
    # Title variants
    "doc_title": (CanonicalLabel.TITLE, {"title_level": "document"}),
    "paragraph_title": (CanonicalLabel.SECTION_HEADER, {"title_level": "paragraph"}),
    "heading": (CanonicalLabel.SECTION_HEADER, {"title_level": "heading"}),
    # Text variants
    "text": (CanonicalLabel.TEXT, {}),
    "number": (CanonicalLabel.TEXT, {"text_role": "page_number"}),
    "abstract": (CanonicalLabel.TEXT, {"text_role": "abstract"}),
    "content": (CanonicalLabel.TEXT, {"text_role": "body"}),
    "reference": (CanonicalLabel.TEXT, {"text_role": "references"}),
    "aside_text": (CanonicalLabel.TEXT, {"text_role": "sidebar"}),
    "reference_content": (CanonicalLabel.TEXT, {"text_role": "references"}),
    "formula_number": (CanonicalLabel.TEXT, {"text_role": "formula_number"}),
    # Page furniture
    "header": (CanonicalLabel.PAGE_HEADER, {"furniture": "page-header"}),
    "footer": (CanonicalLabel.PAGE_FOOTER, {"furniture": "page-footer"}),
    "footnote": (CanonicalLabel.FOOTNOTE, {}),
    # Pictures
    "image": (CanonicalLabel.PICTURE, {"picture_type": "image"}),
    "chart": (CanonicalLabel.PICTURE, {"picture_type": "chart"}),
    "seal": (CanonicalLabel.PICTURE, {"picture_type": "seal"}),
    # Captions
    "figure_title": (CanonicalLabel.CAPTION, {"caption_of": "picture"}),
    # Other
    "table": (CanonicalLabel.TABLE, {}),
    "formula": (CanonicalLabel.FORMULA, {}),
    "algorithm": (CanonicalLabel.CODE, {}),
    # LlamaParse V2 SDK ``CodeItem`` emits ``type="code"`` even when
    # ``layoutAwareBbox`` labels look V2-only, so without this alias the V2-path
    # canonicalizer silently dropped code blocks.
    "code": (CanonicalLabel.CODE, {}),
}


LLAMAPARSE_V3_RAW_TO_CANONICAL: dict[str, tuple[CanonicalLabel, dict[str, str]]] = {
    # Core11 identity mappings (V3)
    "caption": (CanonicalLabel.CAPTION, {}),
    "footnote": (CanonicalLabel.FOOTNOTE, {}),
    "formula": (CanonicalLabel.FORMULA, {}),
    "list-item": (CanonicalLabel.LIST_ITEM, {}),
    "page-footer": (CanonicalLabel.PAGE_FOOTER, {}),
    "page-header": (CanonicalLabel.PAGE_HEADER, {}),
    "picture": (CanonicalLabel.PICTURE, {}),
    "section-header": (CanonicalLabel.SECTION_HEADER, {}),
    "table": (CanonicalLabel.TABLE, {}),
    "text": (CanonicalLabel.TEXT, {}),
    "title": (CanonicalLabel.TITLE, {}),
    # Extended Canonical17 labels (V3)
    "document-index": (CanonicalLabel.DOCUMENT_INDEX, {}),
    "code": (CanonicalLabel.CODE, {}),
    "checkbox-selected": (CanonicalLabel.CHECKBOX_SELECTED, {}),
    "checkbox-unselected": (CanonicalLabel.CHECKBOX_UNSELECTED, {}),
    "form": (CanonicalLabel.FORM, {}),
    "key-value-region": (CanonicalLabel.KEY_VALUE_REGION, {}),
    # Chart maps to Picture with attribute (V3)
    "chart": (CanonicalLabel.PICTURE, {"picture_type": "chart"}),
    # V2 fallback mappings (for mixed V2/V3 responses)
    "doc_title": (CanonicalLabel.TITLE, {"title_level": "document"}),
    "paragraph_title": (CanonicalLabel.SECTION_HEADER, {"title_level": "paragraph"}),
    "number": (CanonicalLabel.TEXT, {"text_role": "page_number"}),
    "abstract": (CanonicalLabel.TEXT, {"text_role": "abstract"}),
    "content": (CanonicalLabel.TEXT, {"text_role": "body"}),
    "reference": (CanonicalLabel.TEXT, {"text_role": "references"}),
    "aside_text": (CanonicalLabel.TEXT, {"text_role": "sidebar"}),
    "reference_content": (CanonicalLabel.TEXT, {"text_role": "references"}),
    "formula_number": (CanonicalLabel.TEXT, {"text_role": "formula_number"}),
    "header": (CanonicalLabel.PAGE_HEADER, {"furniture": "page-header"}),
    "footer": (CanonicalLabel.PAGE_FOOTER, {"furniture": "page-footer"}),
    "image": (CanonicalLabel.PICTURE, {"picture_type": "image"}),
    "seal": (CanonicalLabel.PICTURE, {"picture_type": "seal"}),
    "figure_title": (CanonicalLabel.CAPTION, {"caption_of": "picture"}),
    "algorithm": (CanonicalLabel.CODE, {}),
    "heading": (CanonicalLabel.SECTION_HEADER, {"title_level": "heading"}),
}


# Docling raw labels from the native ``DoclingDocument`` payload.
# ``reference`` is bibliography/reference-list text, while ``document_index``
# is the TOC/index-style structural region.
DOCLING_RAW_TO_CANONICAL: dict[str, tuple[CanonicalLabel, dict[str, str]]] = {
    # Canonical17 identity mappings
    "caption": (CanonicalLabel.CAPTION, {}),
    "chart": (CanonicalLabel.PICTURE, {"picture_type": "chart"}),
    "footnote": (CanonicalLabel.FOOTNOTE, {}),
    "formula": (CanonicalLabel.FORMULA, {}),
    "list_item": (CanonicalLabel.LIST_ITEM, {}),
    "page_footer": (CanonicalLabel.PAGE_FOOTER, {}),
    "page_header": (CanonicalLabel.PAGE_HEADER, {}),
    "picture": (CanonicalLabel.PICTURE, {}),
    "section_header": (CanonicalLabel.SECTION_HEADER, {}),
    "table": (CanonicalLabel.TABLE, {}),
    "text": (CanonicalLabel.TEXT, {}),
    "title": (CanonicalLabel.TITLE, {}),
    "document_index": (CanonicalLabel.DOCUMENT_INDEX, {}),
    "code": (CanonicalLabel.CODE, {}),
    "checkbox_selected": (CanonicalLabel.CHECKBOX_SELECTED, {}),
    "checkbox_unselected": (CanonicalLabel.CHECKBOX_UNSELECTED, {}),
    "form": (CanonicalLabel.FORM, {}),
    "key_value_region": (CanonicalLabel.KEY_VALUE_REGION, {}),
    # Docling parse-only textual extensions
    "paragraph": (CanonicalLabel.TEXT, {"text_role": "paragraph"}),
    "reference": (CanonicalLabel.TEXT, {"text_role": "references"}),
    "grading_scale": (CanonicalLabel.TEXT, {"text_role": "grading_scale"}),
    "handwritten_text": (CanonicalLabel.TEXT, {"text_role": "handwritten"}),
    "empty_value": (CanonicalLabel.TEXT, {"text_role": "empty_value"}),
    "field_region": (CanonicalLabel.KEY_VALUE_REGION, {"text_role": "field_region"}),
    "field_heading": (CanonicalLabel.TEXT, {"text_role": "field_heading"}),
    "field_item": (CanonicalLabel.TEXT, {"text_role": "field_item"}),
    "field_key": (CanonicalLabel.TEXT, {"text_role": "field_key"}),
    "field_value": (CanonicalLabel.TEXT, {"text_role": "field_value"}),
    "field_hint": (CanonicalLabel.TEXT, {"text_role": "field_hint"}),
    "marker": (CanonicalLabel.TEXT, {"text_role": "marker"}),
}


def detect_llamaparse_label_version(labels: list[str]) -> str:
    """Detect LlamaParse label version from observed raw labels."""
    label_set = set(labels)
    if label_set & LLAMAPARSE_V3_ONLY_LABELS:
        return "v3"
    if label_set & LLAMAPARSE_V2_ONLY_LABELS:
        return "v2"
    return "v2"


def normalize_evaluation_ontology(ontology: str | None) -> str:
    """Normalize ontology string and validate support."""
    if ontology is None:
        return DEFAULT_EVALUATION_ONTOLOGY
    normalized = str(ontology).strip().lower()
    if normalized.startswith("basic"):
        return "basic"
    if normalized.startswith("canonical"):
        return "canonical"
    if normalized.startswith("core"):
        return "core"
    raise ValueError(
        f"Unsupported layout ontology '{ontology}'. Supported values: {sorted(SUPPORTED_EVALUATION_ONTOLOGIES)}"
    )


def map_llamaparse_raw_label_to_canonical(
    raw_label: str,
    *,
    label_version: str,
) -> tuple[CanonicalLabel, dict[str, str]]:
    """Map a raw LlamaParse label to Canonical17 with strict unknown handling."""
    version = label_version.strip().lower()
    if version == "v3":
        mapping = LLAMAPARSE_V3_RAW_TO_CANONICAL
    elif version == "v2":
        mapping = LLAMAPARSE_V2_RAW_TO_CANONICAL
    else:
        raise ValueError(f"Unknown LlamaParse label version: {label_version}")

    mapped = mapping.get(raw_label)
    if mapped is None:
        raise UnknownRawLayoutLabelError(
            f"Unknown LlamaParse raw layout label '{raw_label}' for label_version='{version}'"
        )
    canonical, attributes = mapped
    return canonical, dict(attributes)


def map_docling_raw_label_to_canonical(
    raw_label: str,
) -> tuple[CanonicalLabel, dict[str, str]]:
    """Map a Docling raw layout/document label to Canonical17."""
    normalized = raw_label.strip().lower().replace("-", "_")
    mapped = DOCLING_RAW_TO_CANONICAL.get(normalized)
    if mapped is None:
        raise UnknownRawLayoutLabelError(f"Unknown Docling raw layout label '{raw_label}'")
    canonical, attributes = mapped
    return canonical, dict(attributes)


def map_canonical_label_to_target_ontology(
    canonical_label: CanonicalLabel,
    target_ontology: str | None,
) -> str:
    """Map canonical class to a target ontology class name."""
    normalized_ontology = normalize_evaluation_ontology(target_ontology)
    if normalized_ontology in {"canonical", "core"}:
        return canonical_label.value
    if normalized_ontology == "basic":
        mapped = CANONICAL_TO_BASIC.get(canonical_label)
        if mapped is None:
            raise ValueError(f"No Basic ontology mapping found for canonical label '{canonical_label.value}'")
        basic_label, _attrs = mapped
        return basic_label.value
    raise ValueError(f"Unsupported target ontology '{target_ontology}'")


def map_label_to_target_ontology(
    label: str | None,
    target_ontology: str | None,
) -> str | None:
    """Best-effort label mapping to a target ontology for evaluation."""
    if label is None:
        return None

    normalized_ontology = normalize_evaluation_ontology(target_ontology)
    if normalized_ontology in {"canonical", "core"}:
        return label

    if normalized_ontology == "basic":
        # "Formula" is still accepted as a legacy on-disk Basic label, but the
        # current Basic ontology collapses it into Text for evaluation.
        if label == BasicLabel.FORMULA.value:
            basic_label, _attrs = CANONICAL_TO_BASIC[CanonicalLabel.FORMULA]
            return basic_label.value

        # Already basic
        try:
            return BasicLabel(label).value
        except ValueError:
            pass

        # Canonical -> basic
        try:
            canonical_label = CanonicalLabel(label)
        except ValueError:
            # Preserve unknowns for backwards compatibility in non-raw paths.
            return label

        mapped = CANONICAL_TO_BASIC.get(canonical_label)
        if mapped is None:
            return label
        basic_label, _attrs = mapped
        return basic_label.value

    return label
