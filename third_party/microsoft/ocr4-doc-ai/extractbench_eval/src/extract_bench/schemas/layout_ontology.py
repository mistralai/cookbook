"""Layout detection ontology definitions.

This module defines:
- CanonicalLabel: Canonical17 labels (the standardized dataset label set)
- CORE_LABELS: Core11 subset (DocLayNet-compatible labels)
- CANONICAL_TO_CORE: Mapping from Canonical17 to Core11
- DOCLAYNET_ID_TO_LABEL: DocLayNet ground truth category ID mapping
- Ontology classes for dataset-side label management

Reference: layout_detection_class_label_canonicalization_proposal.md
"""

from abc import ABC, abstractmethod
from enum import StrEnum

# =============================================================================
# Canonical17 Label Enum (Standardized Dataset Labels)
# =============================================================================


class CanonicalLabel(StrEnum):
    """Canonical17 layout detection labels.

    This is the standardized label set based on Docling Heron's 17-class schema.
    It serves as the canonical representation for all dataset ground truth.

    Core11 (DocLayNet-compatible) is a subset: use CORE_LABELS to check membership.
    """

    # Core11 labels (DocLayNet-compatible) - these 11 are the minimum required
    CAPTION = "Caption"
    FOOTNOTE = "Footnote"
    FORMULA = "Formula"
    LIST_ITEM = "List-item"
    PAGE_FOOTER = "Page-footer"
    PAGE_HEADER = "Page-header"
    PICTURE = "Picture"
    SECTION_HEADER = "Section-header"
    TABLE = "Table"
    TEXT = "Text"
    TITLE = "Title"

    # Extended Canonical17 labels (not in Core11)
    CHECKBOX_SELECTED = "Checkbox-Selected"
    CHECKBOX_UNSELECTED = "Checkbox-Unselected"
    CODE = "Code"
    DOCUMENT_INDEX = "Document Index"
    FORM = "Form"
    KEY_VALUE_REGION = "Key-Value Region"


# =============================================================================
# Core11 Subset (DocLayNet-compatible labels)
# =============================================================================

CORE_LABELS: frozenset[CanonicalLabel] = frozenset(
    {
        CanonicalLabel.CAPTION,
        CanonicalLabel.FOOTNOTE,
        CanonicalLabel.FORMULA,
        CanonicalLabel.LIST_ITEM,
        CanonicalLabel.PAGE_FOOTER,
        CanonicalLabel.PAGE_HEADER,
        CanonicalLabel.PICTURE,
        CanonicalLabel.SECTION_HEADER,
        CanonicalLabel.TABLE,
        CanonicalLabel.TEXT,
        CanonicalLabel.TITLE,
    }
)


# =============================================================================
# Canonical17 to Core11 Mapping
# =============================================================================

# Labels in Core11 map to themselves, others map to closest Core11 equivalent or None
CANONICAL_TO_CORE: dict[CanonicalLabel, CanonicalLabel | None] = {
    # Core11 labels map to themselves
    CanonicalLabel.CAPTION: CanonicalLabel.CAPTION,
    CanonicalLabel.FOOTNOTE: CanonicalLabel.FOOTNOTE,
    CanonicalLabel.FORMULA: CanonicalLabel.FORMULA,
    CanonicalLabel.LIST_ITEM: CanonicalLabel.LIST_ITEM,
    CanonicalLabel.PAGE_FOOTER: CanonicalLabel.PAGE_FOOTER,
    CanonicalLabel.PAGE_HEADER: CanonicalLabel.PAGE_HEADER,
    CanonicalLabel.PICTURE: CanonicalLabel.PICTURE,
    CanonicalLabel.SECTION_HEADER: CanonicalLabel.SECTION_HEADER,
    CanonicalLabel.TABLE: CanonicalLabel.TABLE,
    CanonicalLabel.TEXT: CanonicalLabel.TEXT,
    CanonicalLabel.TITLE: CanonicalLabel.TITLE,
    # Extended labels map to closest Core11 equivalent or None
    CanonicalLabel.CHECKBOX_SELECTED: None,  # No Core11 equivalent
    CanonicalLabel.CHECKBOX_UNSELECTED: None,  # No Core11 equivalent
    CanonicalLabel.CODE: CanonicalLabel.TEXT,  # Code is text content
    CanonicalLabel.DOCUMENT_INDEX: CanonicalLabel.TEXT,  # Index is text content
    CanonicalLabel.FORM: None,  # No Core11 equivalent
    CanonicalLabel.KEY_VALUE_REGION: CanonicalLabel.TEXT,  # KV regions contain text
}


# =============================================================================
# Basic7 Label Enum (Simplified Ontology)
# =============================================================================


class BasicLabel(StrEnum):
    """Basic7 layout detection labels (simplified ontology).

    Merges Title + Section-header → Section
    Merges List-item → Text (with text_role attribute)
    Merges Caption + Footnote + Formula → Text (with text_role attribute)

    This ontology is useful for evaluation when finer distinctions
    between Title/Section-header or Text/List-item are not needed.
    """

    # Kept as a legacy accepted label so older on-disk "basic" datasets remain loadable.
    FORMULA = "Formula"
    PAGE_FOOTER = "Page-footer"
    PAGE_HEADER = "Page-header"
    PICTURE = "Picture"
    SECTION = "Section"  # Title + Section-header
    TABLE = "Table"
    TEXT = "Text"  # Includes List-item, Caption, Footnote


BASIC_LABELS: frozenset[BasicLabel] = frozenset(BasicLabel)

# Evaluation ontology defaults/choices (used by evaluation-time adaptation).
DEFAULT_LAYOUT_EVALUATION_ONTOLOGY = "basic"
SUPPORTED_LAYOUT_EVALUATION_ONTOLOGIES = frozenset({"basic", "canonical", "core"})


# =============================================================================
# Canonical17 to Basic7 Mapping
# =============================================================================


# Canonical17 → Basic7 mapping with attributes for reversibility
CANONICAL_TO_BASIC: dict[CanonicalLabel, tuple[BasicLabel, dict[str, str]]] = {
    # Merged into Section
    CanonicalLabel.TITLE: (BasicLabel.SECTION, {"title_level": "title"}),
    CanonicalLabel.SECTION_HEADER: (BasicLabel.SECTION, {"title_level": "section-header"}),
    # Merged into Text
    CanonicalLabel.TEXT: (BasicLabel.TEXT, {}),
    CanonicalLabel.LIST_ITEM: (BasicLabel.TEXT, {"text_role": "list-item"}),
    CanonicalLabel.CAPTION: (BasicLabel.TEXT, {"text_role": "caption"}),
    CanonicalLabel.FOOTNOTE: (BasicLabel.TEXT, {"text_role": "footnote"}),
    CanonicalLabel.FORMULA: (BasicLabel.TEXT, {"text_role": "formula"}),
    # Identity mappings (unchanged)
    CanonicalLabel.PAGE_FOOTER: (BasicLabel.PAGE_FOOTER, {}),
    CanonicalLabel.PAGE_HEADER: (BasicLabel.PAGE_HEADER, {}),
    CanonicalLabel.PICTURE: (BasicLabel.PICTURE, {}),
    CanonicalLabel.TABLE: (BasicLabel.TABLE, {}),
    # Extended Canonical17 labels → Text (not present in current datasets)
    CanonicalLabel.CODE: (BasicLabel.TEXT, {"text_role": "code"}),
    CanonicalLabel.DOCUMENT_INDEX: (BasicLabel.TEXT, {"text_role": "document-index"}),
    CanonicalLabel.KEY_VALUE_REGION: (BasicLabel.TEXT, {"text_role": "key-value"}),
    CanonicalLabel.FORM: (BasicLabel.TEXT, {"text_role": "form"}),
    CanonicalLabel.CHECKBOX_SELECTED: (BasicLabel.TEXT, {"text_role": "checkbox-selected"}),
    CanonicalLabel.CHECKBOX_UNSELECTED: (BasicLabel.TEXT, {"text_role": "checkbox-unselected"}),
}


# =============================================================================
# DocLayNet Ground Truth Mapping (1-indexed category IDs)
# =============================================================================

DOCLAYNET_ID_TO_LABEL: dict[int, CanonicalLabel] = {
    1: CanonicalLabel.CAPTION,
    2: CanonicalLabel.FOOTNOTE,
    3: CanonicalLabel.FORMULA,
    4: CanonicalLabel.LIST_ITEM,
    5: CanonicalLabel.PAGE_FOOTER,
    6: CanonicalLabel.PAGE_HEADER,
    7: CanonicalLabel.PICTURE,
    8: CanonicalLabel.SECTION_HEADER,
    9: CanonicalLabel.TABLE,
    10: CanonicalLabel.TEXT,
    11: CanonicalLabel.TITLE,
}


# =============================================================================
# Ontology Base Class
# =============================================================================


class LayoutDetectionOntology(ABC):
    """Abstract base class for layout detection ontologies.

    Ontologies define which labels are valid for a given dataset evaluation view.
    """

    def __init__(self) -> None:
        self._labels: frozenset[CanonicalLabel] = frozenset()
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """Initialize the label set."""
        pass

    @property
    def labels(self) -> frozenset[CanonicalLabel]:
        """Return all labels in this ontology."""
        return self._labels

    @property
    def label_names(self) -> list[str]:
        """Return sorted list of label names (string values)."""
        return sorted(label.value for label in self._labels)

    def has_label(self, label: CanonicalLabel | str) -> bool:
        """Check if a label exists in this ontology.

        :param label: CanonicalLabel enum or string value
        :return: True if label exists
        """
        if isinstance(label, str):
            return label in {lbl.value for lbl in self._labels}
        return label in self._labels

    def map_label(
        self,
        source_label: str,
        source_schema: str,
    ) -> tuple[str, dict[str, str]]:
        """Map a label from a source schema to canonical class and attributes.

        :param source_label: The label in the source schema
        :param source_schema: Name of the source schema (e.g., "doclaynet")
        :return: Tuple of (canonical_class_name, attributes_dict)
        """
        # For DocLayNet, it's an identity mapping (Core11 labels)
        if source_schema == "doclaynet":
            if self.has_label(source_label):
                return (source_label, {})
            # Fallback to Text if label not found
            return ("Text", {})

        # Unknown schema, fallback to Text
        return ("Text", {})


# =============================================================================
# Core11 Ontology (DocLayNet compatible)
# =============================================================================


class CoreLayoutDetectionOntology(LayoutDetectionOntology):
    """Core11 layout detection ontology.

    This ontology contains the 11 classes from DocLayNet, which is the
    intersection view for compatibility with DocLayNet and many other benchmarks.

    Labels: Caption, Footnote, Formula, List-item, Page-footer, Page-header,
            Picture, Section-header, Table, Text, Title
    """

    def _setup(self) -> None:
        """Initialize Core11 labels."""
        self._labels = CORE_LABELS


# =============================================================================
# Canonical17 Ontology (Full canonical schema)
# =============================================================================


class CanonicalLayoutDetectionOntology(LayoutDetectionOntology):
    """Canonical17 layout detection ontology.

    This ontology adopts Docling Heron's 17-class schema as the primary
    canonical class set, because it is a superset of DocLayNet's 11 labels
    and covers several additional document elements commonly needed in
    real pipelines.

    Additional labels beyond Core11: Checkbox-Selected, Checkbox-Unselected,
                                     Code, Document Index, Form, Key-Value Region
    """

    def _setup(self) -> None:
        """Initialize Canonical17 labels."""
        self._labels = frozenset(CanonicalLabel)


# =============================================================================
# Ontology Type Enum and Factory
# =============================================================================


class OntologyType(StrEnum):
    """Supported ontology types."""

    CORE = "core"
    CANONICAL = "canonical"


def get_ontology(ontology_type: OntologyType | str) -> LayoutDetectionOntology:
    """Factory function to get an ontology instance.

    :param ontology_type: Either OntologyType enum or string ("core" or "canonical")
    :return: LayoutDetectionOntology instance
    :raises ValueError: If unknown ontology type
    """
    if isinstance(ontology_type, str):
        ontology_type = OntologyType(ontology_type.lower())

    if ontology_type == OntologyType.CORE:
        return CoreLayoutDetectionOntology()
    elif ontology_type == OntologyType.CANONICAL:
        return CanonicalLayoutDetectionOntology()
    else:
        raise ValueError(f"Unknown ontology type: {ontology_type}")
