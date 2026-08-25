"""Label adapters for canonicalizing layout detection predictions.

This module implements the canonicalization logic defined in
layout_detection_class_label_canonicalization_proposal.md.

Each model has its own adapter that converts model-specific labels
to the Canonical17 schema with optional attributes, and then to
the Core11 (DocLayNet-compatible) schema for evaluation.
"""

from abc import ABC, abstractmethod

from extract_bench.layout_label_mapping import (
    LLAMAPARSE_V2_RAW_TO_CANONICAL,
    LLAMAPARSE_V3_RAW_TO_CANONICAL,
)
from extract_bench.schemas.layout_detection_output import (
    CanonicalLayoutPrediction,
    ChandraLabel,
    CoreLayoutPrediction,
    DoclingLabel,
    LayoutV3Label,
    PPDocLayoutLabel,
    Qwen3VLLabel,
    SuryaLabel,
)
from extract_bench.schemas.layout_ontology import (
    CANONICAL_TO_BASIC,
    CANONICAL_TO_CORE,
    CanonicalLabel,
)


def canonical_to_core(
    canonical_pred: CanonicalLayoutPrediction,
) -> CoreLayoutPrediction | None:
    """
    Convert a Canonical17 prediction to Core11 prediction.

    :param canonical_pred: Canonical17 prediction
    :return: CoreLayoutPrediction or None if no Core11 equivalent
    """
    core_class = CANONICAL_TO_CORE.get(canonical_pred.canonical_class)
    if core_class is None:
        return None

    return CoreLayoutPrediction(
        bbox=canonical_pred.bbox,
        score=canonical_pred.score,
        core_class=core_class,
        attributes=canonical_pred.attributes,
        original_label=canonical_pred.original_label,
    )


def canonical_to_basic(
    canonical_pred: CanonicalLayoutPrediction,
) -> tuple[str, dict[str, str]] | None:
    """
    Convert a Canonical17 prediction to a Basic label and merged attributes.

    Existing attributes take precedence over conversion attributes.
    """
    mapping = CANONICAL_TO_BASIC.get(canonical_pred.canonical_class)
    if mapping is None:
        return None

    basic_label, conversion_attrs = mapping
    merged_attrs = dict(conversion_attrs)
    merged_attrs.update(canonical_pred.attributes)
    return basic_label.value, merged_attrs


class BaseLabelAdapter(ABC):
    """Base class for label canonicalization adapters."""

    @abstractmethod
    def to_canonical(
        self,
        label: int,
        score: float,
        bbox: list[float],
    ) -> CanonicalLayoutPrediction | None:
        """
        Convert model-specific label to canonical prediction.

        :param label: Model-specific label as int
        :param score: Confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: CanonicalLayoutPrediction or None if label should be skipped
        """
        raise NotImplementedError

    def to_core(
        self,
        label: int,
        score: float,
        bbox: list[float],
    ) -> CoreLayoutPrediction | None:
        """
        Convert model-specific label to Core11 prediction.

        First converts to Canonical17, then maps to Core11.

        :param label: Model-specific label as int
        :param score: Confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: CoreLayoutPrediction or None if no Core11 equivalent
        """
        canonical = self.to_canonical(label, score, bbox)
        if canonical is None:
            return None
        return canonical_to_core(canonical)


class YoloLayoutDetLabelAdapter(BaseLabelAdapter):
    """Adapter for YOLO-DocLayNet labels -> Canonical17.

    YOLO outputs Core11 labels (DocLayNet) that map 1:1 to Canonical17 (identity mapping).
    """

    # YoloLabel -> (CanonicalLabel, attributes)
    # Identity mapping - YOLO labels are Core11 which map directly to Canonical17
    MAPPING: dict[int, tuple[CanonicalLabel, dict[str, str]]] = {
        0: (CanonicalLabel.CAPTION, {}),
        1: (CanonicalLabel.FOOTNOTE, {}),
        2: (CanonicalLabel.FORMULA, {}),
        3: (CanonicalLabel.LIST_ITEM, {}),
        4: (CanonicalLabel.PAGE_FOOTER, {}),
        5: (CanonicalLabel.PAGE_HEADER, {}),
        6: (CanonicalLabel.PICTURE, {}),
        7: (CanonicalLabel.SECTION_HEADER, {}),
        8: (CanonicalLabel.TABLE, {}),
        9: (CanonicalLabel.TEXT, {}),
        10: (CanonicalLabel.TITLE, {}),
    }

    def to_canonical(
        self,
        label: int,
        score: float,
        bbox: list[float],
    ) -> CanonicalLayoutPrediction | None:
        """
        Convert YOLO label to canonical prediction.

        :param label: YOLO label as int (0-10)
        :param score: Confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: CanonicalLayoutPrediction or None if unknown label
        """
        mapping = self.MAPPING.get(label)
        if mapping is None:
            # Unknown label, skip
            return None

        canonical_class, attributes = mapping

        return CanonicalLayoutPrediction(
            bbox=bbox,
            score=score,
            canonical_class=canonical_class,
            attributes=attributes,
            original_label=label,
        )


class DoclingLayoutDetLabelAdapter(BaseLabelAdapter):
    """Adapter for Docling RT-DETR labels -> Canonical17.

    Docling Heron outputs labels that map 1:1 to Canonical17 (identity mapping).
    See proposal: "Docling Heron → Canonical17: No mapping required."
    """

    # DoclingLabel -> (CanonicalLabel, attributes)
    # Identity mapping - Docling labels are already Canonical17
    MAPPING: dict[DoclingLabel, tuple[CanonicalLabel, dict[str, str]]] = {
        DoclingLabel.CAPTION: (CanonicalLabel.CAPTION, {}),
        DoclingLabel.FOOTNOTE: (CanonicalLabel.FOOTNOTE, {}),
        DoclingLabel.FORMULA: (CanonicalLabel.FORMULA, {}),
        DoclingLabel.LIST_ITEM: (CanonicalLabel.LIST_ITEM, {}),
        DoclingLabel.PAGE_FOOTER: (CanonicalLabel.PAGE_FOOTER, {}),
        DoclingLabel.PAGE_HEADER: (CanonicalLabel.PAGE_HEADER, {}),
        DoclingLabel.PICTURE: (CanonicalLabel.PICTURE, {}),
        DoclingLabel.SECTION_HEADER: (CanonicalLabel.SECTION_HEADER, {}),
        DoclingLabel.TABLE: (CanonicalLabel.TABLE, {}),
        DoclingLabel.TEXT: (CanonicalLabel.TEXT, {}),
        DoclingLabel.TITLE: (CanonicalLabel.TITLE, {}),
        DoclingLabel.DOCUMENT_INDEX: (CanonicalLabel.DOCUMENT_INDEX, {}),
        DoclingLabel.CODE: (CanonicalLabel.CODE, {}),
        DoclingLabel.CHECKBOX_SELECTED: (CanonicalLabel.CHECKBOX_SELECTED, {}),
        DoclingLabel.CHECKBOX_UNSELECTED: (CanonicalLabel.CHECKBOX_UNSELECTED, {}),
        DoclingLabel.FORM: (CanonicalLabel.FORM, {}),
        DoclingLabel.KEY_VALUE_REGION: (CanonicalLabel.KEY_VALUE_REGION, {}),
    }

    def to_canonical(
        self,
        label: int,
        score: float,
        bbox: list[float],
    ) -> CanonicalLayoutPrediction | None:
        """
        Convert Docling label to canonical prediction.

        :param label: Docling label as int (0-16)
        :param score: Confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: CanonicalLayoutPrediction or None if unknown label
        """
        try:
            docling_label = DoclingLabel(label)
        except ValueError:
            # Unknown label, skip
            return None

        mapping = self.MAPPING.get(docling_label)
        if mapping is None:
            return None

        canonical_class, attributes = mapping

        return CanonicalLayoutPrediction(
            bbox=bbox,
            score=score,
            canonical_class=canonical_class,
            attributes=attributes,
            original_label=label,
        )


class LayoutV3LabelAdapter(BaseLabelAdapter):
    """Adapter for Layout-V3 labels -> Canonical17 with figure classification.

    Layout-V3 outputs the same 17-class schema as Docling Heron.
    This adapter extends the mapping to support figure classification
    attributes for Picture detections.
    """

    # LayoutV3Label -> (CanonicalLabel, attributes)
    # Identity mapping - Layout-V3 labels are already Canonical17
    MAPPING: dict[LayoutV3Label, tuple[CanonicalLabel, dict[str, str]]] = {
        LayoutV3Label.CAPTION: (CanonicalLabel.CAPTION, {}),
        LayoutV3Label.FOOTNOTE: (CanonicalLabel.FOOTNOTE, {}),
        LayoutV3Label.FORMULA: (CanonicalLabel.FORMULA, {}),
        LayoutV3Label.LIST_ITEM: (CanonicalLabel.LIST_ITEM, {}),
        LayoutV3Label.PAGE_FOOTER: (CanonicalLabel.PAGE_FOOTER, {}),
        LayoutV3Label.PAGE_HEADER: (CanonicalLabel.PAGE_HEADER, {}),
        LayoutV3Label.PICTURE: (CanonicalLabel.PICTURE, {}),
        LayoutV3Label.SECTION_HEADER: (CanonicalLabel.SECTION_HEADER, {}),
        LayoutV3Label.TABLE: (CanonicalLabel.TABLE, {}),
        LayoutV3Label.TEXT: (CanonicalLabel.TEXT, {}),
        LayoutV3Label.TITLE: (CanonicalLabel.TITLE, {}),
        LayoutV3Label.DOCUMENT_INDEX: (CanonicalLabel.DOCUMENT_INDEX, {}),
        LayoutV3Label.CODE: (CanonicalLabel.CODE, {}),
        LayoutV3Label.CHECKBOX_SELECTED: (CanonicalLabel.CHECKBOX_SELECTED, {}),
        LayoutV3Label.CHECKBOX_UNSELECTED: (CanonicalLabel.CHECKBOX_UNSELECTED, {}),
        LayoutV3Label.FORM: (CanonicalLabel.FORM, {}),
        LayoutV3Label.KEY_VALUE_REGION: (CanonicalLabel.KEY_VALUE_REGION, {}),
    }

    def to_canonical_with_figure_class(
        self,
        label: int,
        score: float,
        bbox: list[float],
        figure_class: str | None = None,
        figure_score: float | None = None,
    ) -> CanonicalLayoutPrediction | None:
        """
        Convert Layout-V3 label to canonical prediction with figure classification.

        For Picture detections, figure classification is stored as attributes:
        - picture_type: The classified figure type (e.g., "bar_chart", "logo")
        - figure_score: The figure classification confidence score

        :param label: Layout-V3 label as int (0-16)
        :param score: Detection confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :param figure_class: Figure classification type (for Picture labels only)
        :param figure_score: Figure classification confidence (for Picture labels only)
        :return: CanonicalLayoutPrediction or None if unknown label
        """
        try:
            v3_label = LayoutV3Label(label)
        except ValueError:
            # Unknown label, skip
            return None

        mapping = self.MAPPING.get(v3_label)
        if mapping is None:
            return None

        canonical_class, base_attributes = mapping

        # Build attributes, adding figure classification if present
        attributes = dict(base_attributes)
        if figure_class is not None and canonical_class == CanonicalLabel.PICTURE:
            attributes["picture_type"] = figure_class
            if figure_score is not None:
                attributes["figure_score"] = str(round(figure_score, 4))

        return CanonicalLayoutPrediction(
            bbox=bbox,
            score=score,
            canonical_class=canonical_class,
            attributes=attributes,
            original_label=label,
        )

    def to_canonical(
        self,
        label: int,
        score: float,
        bbox: list[float],
    ) -> CanonicalLayoutPrediction | None:
        """
        Convert Layout-V3 label to canonical prediction (without figure class).

        :param label: Layout-V3 label as int (0-16)
        :param score: Confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: CanonicalLayoutPrediction or None if unknown label
        """
        return self.to_canonical_with_figure_class(label, score, bbox)


class PPLayoutDetLabelAdapter(BaseLabelAdapter):
    """Adapter for Paddle PP-DocLayout labels -> Canonical17 + attributes.

    See proposal section: "DocLayout / PP-DocLayout-style → Canonical17 + attributes"

    This adapter generates attributes for finer semantic information
    (e.g., text_role, picture_type, title_level).
    """

    # PPDocLayoutLabel -> (CanonicalLabel, attributes)
    MAPPING: dict[PPDocLayoutLabel, tuple[CanonicalLabel, dict[str, str]]] = {
        # Title variants
        PPDocLayoutLabel.DOC_TITLE: (CanonicalLabel.TITLE, {"title_level": "document"}),
        PPDocLayoutLabel.PARAGRAPH_TITLE: (
            CanonicalLabel.SECTION_HEADER,
            {"title_level": "paragraph"},
        ),
        # Text variants
        PPDocLayoutLabel.TEXT: (CanonicalLabel.TEXT, {}),
        PPDocLayoutLabel.NUMBER: (CanonicalLabel.TEXT, {"text_role": "page_number"}),
        PPDocLayoutLabel.ABSTRACT: (CanonicalLabel.TEXT, {"text_role": "abstract"}),
        PPDocLayoutLabel.CONTENT: (CanonicalLabel.TEXT, {"text_role": "body"}),
        PPDocLayoutLabel.REFERENCE: (CanonicalLabel.TEXT, {"text_role": "references"}),
        PPDocLayoutLabel.ASIDE_TEXT: (CanonicalLabel.TEXT, {"text_role": "sidebar"}),
        PPDocLayoutLabel.REFERENCE_CONTENT: (CanonicalLabel.TEXT, {"text_role": "references"}),
        PPDocLayoutLabel.FORMULA_NUMBER: (CanonicalLabel.TEXT, {"text_role": "formula_number"}),
        # Page furniture
        PPDocLayoutLabel.HEADER: (CanonicalLabel.PAGE_HEADER, {"furniture": "page-header"}),
        PPDocLayoutLabel.FOOTER: (CanonicalLabel.PAGE_FOOTER, {"furniture": "page-footer"}),
        PPDocLayoutLabel.FOOTNOTE: (CanonicalLabel.FOOTNOTE, {}),
        # Pictures
        PPDocLayoutLabel.IMAGE: (CanonicalLabel.PICTURE, {"picture_type": "image"}),
        PPDocLayoutLabel.CHART: (CanonicalLabel.PICTURE, {"picture_type": "chart"}),
        PPDocLayoutLabel.SEAL: (CanonicalLabel.PICTURE, {"picture_type": "seal"}),
        # Captions
        PPDocLayoutLabel.FIGURE_TITLE: (CanonicalLabel.CAPTION, {"caption_of": "picture"}),
        # Other
        PPDocLayoutLabel.TABLE: (CanonicalLabel.TABLE, {}),
        PPDocLayoutLabel.FORMULA: (CanonicalLabel.FORMULA, {}),
        PPDocLayoutLabel.ALGORITHM: (CanonicalLabel.CODE, {}),
    }

    def to_canonical(
        self,
        label: int,
        score: float,
        bbox: list[float],
    ) -> CanonicalLayoutPrediction | None:
        """
        Convert PP-DocLayout label to canonical prediction with attributes.

        :param label: PP-DocLayout label as int (0-19)
        :param score: Confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: CanonicalLayoutPrediction or None if unknown label
        """
        try:
            pp_label = PPDocLayoutLabel(label)
        except ValueError:
            # Unknown label, skip
            return None

        mapping = self.MAPPING.get(pp_label)
        if mapping is None:
            return None

        canonical_class, attributes = mapping

        return CanonicalLayoutPrediction(
            bbox=bbox,
            score=score,
            canonical_class=canonical_class,
            attributes=attributes,
            original_label=label,
        )


# =============================================================================
# Qwen3VL Adapter (Qwen3VLLabel -> Canonical17)
# =============================================================================


class Qwen3VLLayoutDetLabelAdapter(BaseLabelAdapter):
    """Adapter for Qwen3VL labels -> Canonical17.

    Qwen3VL outputs Core11 labels. Identity mapping to Canonical17.
    """

    # Qwen3VLLabel -> (CanonicalLabel, attributes)
    # Identity mapping for Core11 labels
    MAPPING: dict[Qwen3VLLabel, tuple[CanonicalLabel, dict[str, str]]] = {
        Qwen3VLLabel.CAPTION: (CanonicalLabel.CAPTION, {}),
        Qwen3VLLabel.FOOTNOTE: (CanonicalLabel.FOOTNOTE, {}),
        Qwen3VLLabel.FORMULA: (CanonicalLabel.FORMULA, {}),
        Qwen3VLLabel.LIST_ITEM: (CanonicalLabel.LIST_ITEM, {}),
        Qwen3VLLabel.PAGE_FOOTER: (CanonicalLabel.PAGE_FOOTER, {}),
        Qwen3VLLabel.PAGE_HEADER: (CanonicalLabel.PAGE_HEADER, {}),
        Qwen3VLLabel.PICTURE: (CanonicalLabel.PICTURE, {}),
        Qwen3VLLabel.SECTION_HEADER: (CanonicalLabel.SECTION_HEADER, {}),
        Qwen3VLLabel.TABLE: (CanonicalLabel.TABLE, {}),
        Qwen3VLLabel.TEXT: (CanonicalLabel.TEXT, {}),
        Qwen3VLLabel.TITLE: (CanonicalLabel.TITLE, {}),
    }

    def to_canonical(
        self,
        label: int,
        score: float,
        bbox: list[float],
    ) -> CanonicalLayoutPrediction | None:
        """
        Convert Qwen3VL label to canonical prediction.

        :param label: Qwen3VL label as int (0-10)
        :param score: Confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: CanonicalLayoutPrediction or None if unknown label
        """
        qwen_label = Qwen3VLLabel(label)
        mapping = self.MAPPING.get(qwen_label)
        if mapping is None:
            return None

        canonical_class, attributes = mapping

        return CanonicalLayoutPrediction(
            bbox=bbox,
            score=score,
            canonical_class=canonical_class,
            attributes=attributes,
            original_label=label,
        )


# =============================================================================
# dots.ocr Adapter (string labels -> Canonical17)
# =============================================================================


class DotsOcrLayoutDetLabelAdapter:
    """Adapter for dots.ocr layout labels -> Canonical17 + attributes.

    dots.ocr outputs Core11-style labels as strings (e.g., Caption, Text).
    This adapter normalizes labels and maps to Canonical17 with optional
    attributes for picture variants.
    """

    # Normalized string label -> (CanonicalLabel, attributes)
    MAPPING: dict[str, tuple[CanonicalLabel, dict[str, str]]] = {
        # Core11 direct mappings
        "caption": (CanonicalLabel.CAPTION, {}),
        "footnote": (CanonicalLabel.FOOTNOTE, {}),
        "formula": (CanonicalLabel.FORMULA, {}),
        "list-item": (CanonicalLabel.LIST_ITEM, {}),
        "listitem": (CanonicalLabel.LIST_ITEM, {}),
        "page-footer": (CanonicalLabel.PAGE_FOOTER, {}),
        "pagefooter": (CanonicalLabel.PAGE_FOOTER, {}),
        "page-header": (CanonicalLabel.PAGE_HEADER, {}),
        "pageheader": (CanonicalLabel.PAGE_HEADER, {}),
        "picture": (CanonicalLabel.PICTURE, {}),
        "section-header": (CanonicalLabel.SECTION_HEADER, {}),
        "sectionheader": (CanonicalLabel.SECTION_HEADER, {}),
        "table": (CanonicalLabel.TABLE, {}),
        "text": (CanonicalLabel.TEXT, {}),
        "title": (CanonicalLabel.TITLE, {}),
        # Picture variants
        "image": (CanonicalLabel.PICTURE, {"picture_type": "image"}),
        "figure": (CanonicalLabel.PICTURE, {"picture_type": "figure"}),
    }

    def to_canonical(
        self,
        label: str,
        score: float,
        bbox: list[float],
    ) -> CanonicalLayoutPrediction | None:
        """Convert dots.ocr string label to canonical prediction."""
        normalized = _normalize_dots_label(label)
        mapping = self.MAPPING.get(normalized)
        if mapping is None:
            return None

        canonical_class, attributes = mapping

        return CanonicalLayoutPrediction(
            bbox=bbox,
            score=score,
            canonical_class=canonical_class,
            attributes=attributes,
            original_label=label,
        )

    def to_core(
        self,
        label: str,
        score: float,
        bbox: list[float],
    ) -> CoreLayoutPrediction | None:
        """Convert dots.ocr label to Core11 prediction."""
        canonical = self.to_canonical(label, score, bbox)
        if canonical is None:
            return None
        return canonical_to_core(canonical)

    def to_basic(
        self,
        label: str,
        score: float,
        bbox: list[float],
    ) -> tuple[str, dict[str, str]] | None:
        """Convert dots.ocr label to Basic label and merged attributes."""
        canonical = self.to_canonical(label, score, bbox)
        if canonical is None:
            return None
        return canonical_to_basic(canonical)


def _normalize_dots_label(label: str) -> str:
    normalized = label.strip().lower()
    normalized = normalized.replace("_", "-").replace(" ", "-")
    return normalized


# =============================================================================
# Surya Adapter (SuryaLabel -> Canonical17)
# =============================================================================


class SuryaLayoutDetLabelAdapter(BaseLabelAdapter):
    """Adapter for Surya OCR layout labels -> Canonical17 + attributes.

    Surya outputs 16 layout classes. Mapping based on semantic equivalence
    to DocLayNet/Canonical17 labels.

    Surya labels: Caption, Footnote, Formula/Equation, List-item, Page-footer,
    Page-header, Picture, Figure, Section-header, Table, Form,
    Table-of-contents, Handwriting, Text, Text-inline-math, Code
    """

    # SuryaLabel -> (CanonicalLabel, attributes)
    MAPPING: dict[SuryaLabel, tuple[CanonicalLabel, dict[str, str]]] = {
        # Core11 identity mappings
        SuryaLabel.CAPTION: (CanonicalLabel.CAPTION, {}),
        SuryaLabel.FOOTNOTE: (CanonicalLabel.FOOTNOTE, {}),
        SuryaLabel.FORMULA: (CanonicalLabel.FORMULA, {}),  # Also "Equation" in v0.17+
        SuryaLabel.LIST_ITEM: (CanonicalLabel.LIST_ITEM, {}),
        SuryaLabel.PAGE_FOOTER: (CanonicalLabel.PAGE_FOOTER, {}),
        SuryaLabel.PAGE_HEADER: (CanonicalLabel.PAGE_HEADER, {}),
        SuryaLabel.PICTURE: (CanonicalLabel.PICTURE, {}),
        SuryaLabel.SECTION_HEADER: (CanonicalLabel.SECTION_HEADER, {}),
        SuryaLabel.TABLE: (CanonicalLabel.TABLE, {}),
        SuryaLabel.TEXT: (CanonicalLabel.TEXT, {}),
        # Figure -> Picture with attribute
        SuryaLabel.FIGURE: (CanonicalLabel.PICTURE, {"picture_type": "figure"}),
        # Form -> Form (Canonical17 extended, no Core11 equivalent)
        SuryaLabel.FORM: (CanonicalLabel.FORM, {}),
        # Table-of-contents -> Document Index
        SuryaLabel.TABLE_OF_CONTENTS: (CanonicalLabel.DOCUMENT_INDEX, {}),
        # Handwriting -> Text with attribute
        SuryaLabel.HANDWRITING: (CanonicalLabel.TEXT, {"text_role": "handwriting"}),
        # Text-inline-math -> Formula with attribute
        SuryaLabel.TEXT_INLINE_MATH: (CanonicalLabel.FORMULA, {"formula_type": "inline"}),
        # Code -> Code (new in Surya v0.17+)
        SuryaLabel.CODE: (CanonicalLabel.CODE, {}),
    }

    def to_canonical(
        self,
        label: int,
        score: float,
        bbox: list[float],
    ) -> CanonicalLayoutPrediction | None:
        """
        Convert Surya label to canonical prediction with attributes.

        :param label: Surya label as int (0-14)
        :param score: Confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: CanonicalLayoutPrediction or None if unknown label
        """
        try:
            surya_label = SuryaLabel(label)
        except ValueError:
            # Unknown label, skip
            return None

        mapping = self.MAPPING.get(surya_label)
        if mapping is None:
            return None

        canonical_class, attributes = mapping

        return CanonicalLayoutPrediction(
            bbox=bbox,
            score=score,
            canonical_class=canonical_class,
            attributes=attributes,
            original_label=label,
        )


# =============================================================================
# Chandra Adapter (ChandraLabel -> Canonical17)
# =============================================================================


class ChandraLayoutDetLabelAdapter(BaseLabelAdapter):
    """Adapter for Chandra OCR layout labels -> Canonical17 + attributes.

    Chandra outputs 15 layout classes via its ocr_layout prompt mode.
    Mapping based on semantic equivalence to DocLayNet/Canonical17 labels.

    Chandra labels: Caption, Footnote, Equation-Block, List-Group, Page-Header,
    Page-Footer, Image, Section-Header, Table, Text, Complex-Block, Code-Block,
    Form, Table-Of-Contents, Figure
    """

    # ChandraLabel -> (CanonicalLabel, attributes)
    MAPPING: dict[ChandraLabel, tuple[CanonicalLabel, dict[str, str]]] = {
        # Core11 identity mappings
        ChandraLabel.CAPTION: (CanonicalLabel.CAPTION, {}),
        ChandraLabel.FOOTNOTE: (CanonicalLabel.FOOTNOTE, {}),
        ChandraLabel.EQUATION_BLOCK: (CanonicalLabel.FORMULA, {}),
        ChandraLabel.LIST_GROUP: (CanonicalLabel.LIST_ITEM, {}),
        ChandraLabel.PAGE_HEADER: (CanonicalLabel.PAGE_HEADER, {}),
        ChandraLabel.PAGE_FOOTER: (CanonicalLabel.PAGE_FOOTER, {}),
        ChandraLabel.SECTION_HEADER: (CanonicalLabel.SECTION_HEADER, {}),
        ChandraLabel.TABLE: (CanonicalLabel.TABLE, {}),
        ChandraLabel.TEXT: (CanonicalLabel.TEXT, {}),
        # Image -> Picture with attribute
        ChandraLabel.IMAGE: (CanonicalLabel.PICTURE, {"picture_type": "image"}),
        # Figure -> Picture with attribute
        ChandraLabel.FIGURE: (CanonicalLabel.PICTURE, {"picture_type": "figure"}),
        # Complex-Block -> Text with attribute (generic complex content)
        ChandraLabel.COMPLEX_BLOCK: (CanonicalLabel.TEXT, {"text_role": "complex"}),
        # Code-Block -> Code
        ChandraLabel.CODE_BLOCK: (CanonicalLabel.CODE, {}),
        # Form -> Form (Canonical17 extended, no Core11 equivalent)
        ChandraLabel.FORM: (CanonicalLabel.FORM, {}),
        # Table-Of-Contents -> Document Index
        ChandraLabel.TABLE_OF_CONTENTS: (CanonicalLabel.DOCUMENT_INDEX, {}),
    }

    def to_canonical(
        self,
        label: int,
        score: float,
        bbox: list[float],
    ) -> CanonicalLayoutPrediction | None:
        """
        Convert Chandra label to canonical prediction with attributes.

        :param label: Chandra label as int (0-14)
        :param score: Confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: CanonicalLayoutPrediction or None if unknown label
        """
        try:
            chandra_label = ChandraLabel(label)
        except ValueError:
            # Unknown label, skip
            return None

        mapping = self.MAPPING.get(chandra_label)
        if mapping is None:
            return None

        canonical_class, attributes = mapping

        return CanonicalLayoutPrediction(
            bbox=bbox,
            score=score,
            canonical_class=canonical_class,
            attributes=attributes,
            original_label=label,
        )


# =============================================================================
# LlamaParse Adapter (V2 string labels -> Canonical17)
# =============================================================================


class LlamaParseLayoutDetLabelAdapter:
    """Adapter for LlamaParse V2 layout labels -> Canonical17 + attributes.

    LlamaParse uses the same V2 label schema as Paddle PP-DocLayout (20 classes),
    but returns string labels instead of integer indices.

    V2 labels:
    - 0: paragraph_title, 1: image, 2: text, 3: number, 4: abstract, 5: content,
    - 6: figure_title, 7: formula, 8: table, 9: reference, 10: doc_title,
    - 11: footnote, 12: header, 13: algorithm, 14: footer, 15: seal,
    - 16: chart, 17: formula_number, 18: aside_text, 19: reference_content
    """

    # Shared central mapping.
    MAPPING: dict[str, tuple[CanonicalLabel, dict[str, str]]] = LLAMAPARSE_V2_RAW_TO_CANONICAL

    def to_canonical(
        self,
        label: str,
        score: float,
        bbox: list[float],
    ) -> CanonicalLayoutPrediction | None:
        """
        Convert LlamaParse string label to canonical prediction with attributes.

        :param label: LlamaParse V2 label as string (e.g., "text", "table")
        :param score: Confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: CanonicalLayoutPrediction or None if unknown label
        """
        mapping = self.MAPPING.get(label)
        if mapping is None:
            # Unknown label, skip
            return None

        canonical_class, attributes = mapping

        return CanonicalLayoutPrediction(
            bbox=bbox,
            score=score,
            canonical_class=canonical_class,
            attributes=attributes,
            original_label=label,
        )

    def to_core(
        self,
        label: str,
        score: float,
        bbox: list[float],
    ) -> CoreLayoutPrediction | None:
        """
        Convert LlamaParse string label to Core11 prediction.

        First converts to Canonical17, then maps to Core11.

        :param label: LlamaParse V2 label as string
        :param score: Confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: CoreLayoutPrediction or None if no Core11 equivalent
        """
        canonical = self.to_canonical(label, score, bbox)
        if canonical is None:
            return None
        return canonical_to_core(canonical)

    def to_basic(
        self,
        label: str,
        score: float,
        bbox: list[float],
    ) -> tuple[str, dict[str, str]] | None:
        """
        Convert LlamaParse string label to Basic label and merged attributes.
        """
        canonical = self.to_canonical(label, score, bbox)
        if canonical is None:
            return None
        return canonical_to_basic(canonical)


class LlamaParseV3LayoutDetLabelAdapter:
    """Adapter for LlamaParse V3 layout labels -> Canonical17.

    V3 labels align closely with the Canonical17 schema, requiring
    simpler 1:1 identity mapping for most labels.

    V3 labels (18 classes):
    - caption, footnote, formula, list-item, page-footer, page-header,
    - picture, section-header, table, text, title, document-index,
    - code, checkbox-selected, checkbox-unselected, form, key-value-region, chart

    Also includes V2 label fallbacks for mixed V2/V3 responses from the hosted API.
    """

    # Shared central mapping.
    MAPPING: dict[str, tuple[CanonicalLabel, dict[str, str]]] = LLAMAPARSE_V3_RAW_TO_CANONICAL

    def to_canonical(
        self,
        label: str,
        score: float,
        bbox: list[float],
    ) -> CanonicalLayoutPrediction | None:
        """Convert LlamaParse V3 string label to canonical prediction."""
        mapping = self.MAPPING.get(label)
        if mapping is None:
            return None

        canonical_class, attributes = mapping

        return CanonicalLayoutPrediction(
            bbox=bbox,
            score=score,
            canonical_class=canonical_class,
            attributes=attributes,
            original_label=label,
        )

    def to_core(
        self,
        label: str,
        score: float,
        bbox: list[float],
    ) -> CoreLayoutPrediction | None:
        """Convert V3 label to Core11 prediction."""
        canonical = self.to_canonical(label, score, bbox)
        if canonical is None:
            return None
        return canonical_to_core(canonical)

    def to_basic(
        self,
        label: str,
        score: float,
        bbox: list[float],
    ) -> tuple[str, dict[str, str]] | None:
        """Convert V3 label to Basic label and merged attributes."""
        canonical = self.to_canonical(label, score, bbox)
        if canonical is None:
            return None
        return canonical_to_basic(canonical)


# =============================================================================
# Chunkr Adapter (ChunkrLabel string -> Canonical17)
# =============================================================================


class ChunkrLayoutDetLabelAdapter:
    """Adapter for Chunkr layout labels -> Canonical17 + attributes.

    Chunkr outputs 17 segment types as string labels. Mapping based on
    semantic equivalence to DocLayNet/Canonical17 labels.

    Chunkr segment types (from docs):
    Caption, Footnote, Formula, FormRegion, GraphicalItem, Legend,
    LineNumber, ListItem, PageFooter, PageHeader, PageNumber, Picture,
    Table, Text, Title, Unknown, Page
    """

    # String label -> (CanonicalLabel, attributes)
    # Using exact Chunkr segment_type values from docs
    MAPPING: dict[str, tuple[CanonicalLabel, dict[str, str]]] = {
        # Core11 direct mappings (exact Chunkr names)
        "Caption": (CanonicalLabel.CAPTION, {}),
        "Footnote": (CanonicalLabel.FOOTNOTE, {}),
        "Formula": (CanonicalLabel.FORMULA, {}),
        "ListItem": (CanonicalLabel.LIST_ITEM, {}),
        "PageFooter": (CanonicalLabel.PAGE_FOOTER, {}),
        "PageHeader": (CanonicalLabel.PAGE_HEADER, {}),
        "Picture": (CanonicalLabel.PICTURE, {}),
        "Table": (CanonicalLabel.TABLE, {}),
        "Text": (CanonicalLabel.TEXT, {}),
        "Title": (CanonicalLabel.TITLE, {}),
        # Text variants with attributes
        "LineNumber": (CanonicalLabel.TEXT, {"text_role": "line_number"}),
        "PageNumber": (CanonicalLabel.TEXT, {"text_role": "page_number"}),
        # Caption variant
        "Legend": (CanonicalLabel.CAPTION, {"caption_of": "chart"}),
        # Picture variant
        "GraphicalItem": (CanonicalLabel.PICTURE, {"picture_type": "chart"}),
        # Extended mappings (no Core11 equivalent)
        "FormRegion": (CanonicalLabel.FORM, {}),
        # Unknown and Page are skipped (no mapping)
    }

    def to_canonical(
        self,
        label: str,
        score: float,
        bbox: list[float],
    ) -> CanonicalLayoutPrediction | None:
        """
        Convert Chunkr string label to canonical prediction.

        :param label: Chunkr segment_type string (e.g., "Text Block", "Table")
        :param score: Confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: CanonicalLayoutPrediction or None if unknown/unmapped label
        """
        mapping = self.MAPPING.get(label)
        if mapping is None:
            # Unknown or Page label, skip
            return None

        canonical_class, attributes = mapping

        return CanonicalLayoutPrediction(
            bbox=bbox,
            score=score,
            canonical_class=canonical_class,
            attributes=attributes,
            original_label=label,
        )

    def to_core(
        self,
        label: str,
        score: float,
        bbox: list[float],
    ) -> CoreLayoutPrediction | None:
        """
        Convert Chunkr string label to Core11 prediction.

        First converts to Canonical17, then maps to Core11.

        :param label: Chunkr segment_type string
        :param score: Confidence score (0-1)
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: CoreLayoutPrediction or None if no Core11 equivalent
        """
        canonical = self.to_canonical(label, score, bbox)
        if canonical is None:
            return None
        return canonical_to_core(canonical)
