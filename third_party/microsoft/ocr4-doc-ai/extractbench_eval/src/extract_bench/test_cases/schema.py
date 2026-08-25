"""Schema definitions for test cases."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator

from extract_bench.schemas.pipeline_io import InputFile
from extract_bench.test_cases.parse_rule_schemas import (
    ParseRule,
    coerce_parse_rule,
)

# =============================================================================
# Layout Content Types
# =============================================================================


class LayoutTextContent(BaseModel):
    """Text content for layout elements (paragraphs, headers, captions, etc.)."""

    type: Literal["text"] = "text"
    text: str = Field(description="Aggregated text content from PDF cells")
    start_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional inclusive start char index of this rule's text inside its "
            "parent line's text. Set on word rules whose ``parent_test_id`` "
            "points at a line and whose text appears verbatim in that line; "
            "``None`` when the parent is not a line or when the substring "
            "could not be located."
        ),
    )
    end_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional exclusive end char index of this rule's text in its "
            "parent line's text. Mirror of ``start_index``."
        ),
    )


class LayoutTableContent(BaseModel):
    """Table content with HTML representation."""

    type: Literal["table"] = "table"
    html: str = Field(description="HTML table representation")


# Union type for layout content with discriminator
LayoutContent = Annotated[
    Annotated[LayoutTextContent, Tag("text")] | Annotated[LayoutTableContent, Tag("table")],
    Discriminator("type"),
]


# =============================================================================
# Test Case Schemas
# =============================================================================


class QAConfig(BaseModel):
    """Configuration for question-answering test cases."""

    question: str = Field(description="The question text to be answered")
    answer: str = Field(description="Expected answer(s)")
    question_type: str = Field(description="Type of question: 'single_choice', 'multiple_choice', or 'numerical'")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional QA metadata (options, tolerance, etc.)"
    )


class LayoutAnnotation(BaseModel):
    """Single layout region annotation."""

    id: str | None = Field(
        default=None,
        description="Optional stable identifier for this layout element",
    )
    bbox: list[float] = Field(description="Bounding box [x, y, width, height] in COCO format")
    canonical_class: str = Field(description="Canonical class name from ontology")
    page: int = Field(default=0, description="Page index (0-based) for multi-page documents")
    attributes: dict[str, str | bool] = Field(
        default_factory=dict, description="Optional attributes that refine the class"
    )
    source_label: str | None = Field(default=None, description="Original label from source dataset")
    source_category_id: int | None = Field(default=None, description="Original category ID from source dataset")
    content: LayoutContent | None = Field(
        default=None,
        description="Attributed content for this layout element",
    )
    ro_index: int | None = Field(
        default=None,
        description="Reading order index (0-based position in reading sequence)",
    )
    tags: list[str] = Field(default_factory=list, description="Optional per-rule tags")
    r: float | None = Field(
        default=None,
        description=("Optional page/SVG rotation in degrees applied clockwise around the center of the literal bbox."),
    )


class LayoutTestRule(BaseModel):
    """Layout annotation as test rule with normalized bbox."""

    type: Literal["layout"] = "layout"
    id: str | None = Field(
        default=None,
        description="Optional stable identifier for this layout element",
    )
    parent_test_id: str | None = Field(
        default=None,
        description="Optional parent layout test rule id for derived line/word annotations",
    )
    page: int = Field(ge=1, description="Page number (1-indexed)")
    bbox: list[float] = Field(description="Normalized bbox [x, y, w, h] in [0,1] range (COCO format)")
    canonical_class: str = Field(description="Class from ontology")
    attributes: dict[str, str | bool] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list, description="Optional per-rule tags")
    source_label: str | None = Field(default=None)
    source_category_id: int | None = Field(default=None)
    content: LayoutContent | None = Field(
        default=None,
        description="Attributed content (text or table HTML)",
    )
    ro_index: int | None = Field(
        default=None,
        description="Reading order index (0-based position in reading sequence)",
    )
    granularity: Literal["region", "line", "word", "cell"] | None = Field(
        default=None,
        description=(
            "Optional geometric granularity of this rule. ``None`` (default) "
            "and ``'region'`` both mean the bbox covers a semantic region "
            "(paragraph, table, figure, ...). ``'line'`` marks a single text "
            "line; ``'word'`` marks a single word; ``'cell'`` marks a table "
            "cell. Evaluators partition detection / attribution metrics by "
            "this key so region-level layout scoring does not mix with line-, "
            "word-, or cell-level OCR scoring when a document carries both."
        ),
    )
    r: float | None = Field(
        default=None,
        description=("Optional page/SVG rotation in degrees applied clockwise around the center of the literal bbox."),
    )

    def to_layout_annotation(self) -> LayoutAnnotation:
        """Convert to LayoutAnnotation (coordinates remain normalized)."""
        return LayoutAnnotation(
            id=self.id,
            bbox=self.bbox,  # Keep normalized
            canonical_class=self.canonical_class,
            page=self.page - 1,  # Convert to 0-indexed for internal use
            attributes=self.attributes,
            source_label=self.source_label,
            source_category_id=self.source_category_id,
            content=self.content,
            ro_index=self.ro_index,
            tags=self.tags,
            r=self.r,
        )


class ExtractFieldBbox(BaseModel):
    """One evidence bbox attached to an extract_field rule."""

    page: int = Field(ge=1, description="Page number (1-indexed)")
    bbox: list[float] = Field(description="Normalized bbox [x, y, w, h] in [0,1] range (COCO format)")
    source_bbox_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Zero-indexed position in the original source bbox list for the "
            "underlying column. Enables lossless round-trip to the source "
            "export. Null when no source-export index is available."
        ),
    )


ComparatorSpec = str | dict[str, str]

# Canonical vocabulary for ExtractFieldTestRule.normalizers. Unknown names are
# rejected at load time: a typo'd normalizer would otherwise silently no-op in
# the metrics and the field would score strictly with no warning.
EXTRACT_FIELD_NORMALIZERS: frozenset[str] = frozenset(
    {
        "optional_terminal_punctuation",
        "punctuation_spacing",
        "case_insensitive",
        "null_equals_false",
        "phone_digits",
        "lenient_date",
    }
)


class FieldEvidence(BaseModel):
    """One accepted location for a field's value (v0.2 evidence list entry).

    A leaf can have multiple evidence entries (the same answer printed in different
    places, or alternate canonical forms). Pipelines pass v0.2 evidence metrics
    if they match an accepted entry. Values may be scalars, arrays, or shallow
    objects depending on the extraction schema.
    """

    page: int | None = Field(default=None, ge=1, description="1-indexed page number; None for value-only evidence")
    bbox: list[float] | None = Field(
        default=None,
        description=("Normalized COCO [x, y, w, h] bbox; None for page-only / parent-level cites."),
    )
    quote: str | None = Field(
        default=None,
        description="Verbatim text from the parsed PDF; None when page-only.",
    )
    value: Any = Field(
        default=None,
        description="Canonical schema-shaped value at this location.",
    )
    coarse: bool = Field(
        default=False,
        description="True when this is a parent-level cite or page-only evidence.",
    )


class ExtractFieldTestRule(BaseModel):
    """Self-contained extract field test: value + evidence bboxes + verified flag."""

    type: Literal["extract_field"] = "extract_field"
    id: str | None = Field(default=None, description="Optional stable identifier")
    field_path: str = Field(
        description=('Dotted + bracketed path into expected_output, e.g. "line_items[0].description".'),
    )
    expected_value: str | int | float | bool | None = Field(
        default=None,
        description="Expected primitive value for this field path",
    )
    bboxes: list[ExtractFieldBbox] = Field(
        default_factory=list,
        description="Zero or more evidence bboxes for this field (multi-line cells have multiple)",
    )
    verified: bool = Field(
        default=True,
        description=(
            "Whether the grounding has been human-verified. False for entries "
            "assigned heuristically that need manual review (wrap extras, "
            "suspected header clicks)."
        ),
    )
    tags: list[str] = Field(default_factory=list, description="Optional per-rule tags")
    evidence: list[FieldEvidence] | None = Field(
        default=None,
        description=(
            "v0.2 evidence list: per-location (page, bbox, quote, value, coarse) entries. "
            "When present, supersedes the legacy bboxes+expected_value pair for matching."
        ),
    )
    comparator: ComparatorSpec | None = Field(
        default=None,
        description=(
            "v0.2 per-field comparator name (e.g. 'exact', 'enum', 'number_with_unit', "
            "'string_substring') or a shallow object comparator map such as "
            "{'text': 'case_insensitive', 'category': 'enum'}. None = legacy default."
        ),
    )
    normalizers: list[str] = Field(
        default_factory=list,
        description=(
            "Opt-in value normalizers for this field, applied by metrics that support them "
            "as fail->pass-only fallbacks after exact matching misses. Supported "
            "(see EXTRACT_FIELD_NORMALIZERS; unknown names are rejected at load time): "
            "'optional_terminal_punctuation' (ignore ONE trailing .,;: char on both sides), "
            "'punctuation_spacing' (whitespace hugging ,.;: is insignificant), "
            "'case_insensitive' (casefold both sides), "
            "'null_equals_false' (a blank checkbox may read as false or null/omitted), "
            "'phone_digits' (compare digit sequences; last-10 rule when both are full numbers), "
            "'lenient_date' (scanned-form dates: join split preprinted years like '19 55', "
            "expand trailing 2-digit years into either century; rewrites must parse to a date)."
        ),
    )

    @field_validator("normalizers")
    @classmethod
    def _validate_normalizers(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - EXTRACT_FIELD_NORMALIZERS)
        if unknown:
            raise ValueError(f"Unknown normalizer(s) {unknown}; supported: {sorted(EXTRACT_FIELD_NORMALIZERS)}")
        return value

    structural: str | None = Field(
        default=None,
        description=(
            "v0.2 array-parent structural rule ('ordered', 'set', 'multiset', 'match_by:<key>'). "
            "None for scalar/object fields."
        ),
    )
    exhaustive: bool | None = Field(
        default=None,
        description="v0.2 array-only: whether the predicted array may contain extra unmatched items.",
    )
    expected_min: int | None = Field(
        default=None,
        ge=0,
        description="v0.2 array-only: minimum number of expected items required for a pass.",
    )
    evidence_required: bool = Field(
        default=True,
        description="v0.2: when False, pipeline may pass M1 with no grounding evidence.",
    )
    source_policy: Literal["verbatim", "computed", "inferred"] = Field(
        default="verbatim",
        description="v0.2: how the value originates in the source (verbatim, computed, inferred).",
    )
    max_evidence: int | None = Field(
        default=None,
        ge=1,
        description="v0.2: cap on evidence list length (None = unlimited).",
    )
    iou_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Legacy/v0.2 diagnostic per-rule strict-IoU override. None = harness default (0.5). "
            "Replaces the strict 0.5 threshold only; the relaxed value-conditioned branch "
            "(IoU>=0.3 AND IoA>=0.7 AND canonical_exact) is unaffected."
        ),
    )


ExtractRuleUnion = ExtractFieldTestRule | dict[str, Any]


class BaseTestCase(BaseModel):
    """Base test case with common fields."""

    model_config = ConfigDict(populate_by_name=True)

    test_id: str = Field(description="Unique identifier (e.g., 'group/pdf_name')")
    group: str = Field(description="Group/subfolder name")
    file_path: Path = Field(description="Path to the input file (PDF, image, etc.)")
    tags: list[str] = Field(
        default_factory=list,
        description="Optional top-level tags for document-level provenance, filtering, and grouping.",
    )


class ExtractTestCase(BaseTestCase):
    """Test case for EXTRACT product type."""

    data_schema: dict[str, Any] = Field(
        description="JSON schema for extraction (from test.json data_schema)",
        alias="schema",
    )
    config: dict[str, Any] | None = Field(
        default=None, description="Extraction config override (from test.json config)"
    )
    input_files: list[InputFile] | None = Field(
        default=None,
        description=(
            "Optional ordered, role-tagged input files for multi-file extract "
            "test cases (base + amendments -> merged master). When None, this is "
            "a single-file test case using ``file_path`` alone."
        ),
    )
    expected_output: dict[str, Any] | None = Field(
        default=None,
        description="Expected output for evaluation (from test.json expected_output)",
    )
    test_rules: list[ExtractRuleUnion] | None = Field(
        default=None,
        description="List of rule-based test definitions (from test.json test_rules)",
    )
    schema_version: str | None = Field(
        default=None,
        alias="_schema_version",
        description="v0.2 schema version tag (e.g. 'sample/v0.2'); informational only.",
    )
    row_identity: dict[str, Any] | None = Field(
        default=None,
        alias="_eval_row_identity",
        description=(
            "Authoritative row-identity declarations, keyed by array path "
            "({'holdings': {'identity_key': 'cusip'}}). EVALUATION-ONLY: it rides "
            "beside data_schema rather than inside it, because a "
            "``repeated_structure`` block within the schema is a LlamaExtract-only "
            "annotation that every other provider has to strip back out of its "
            "request payload. Read it via ``eval_data_schema``; never send it to a "
            "provider.\n\n"
            "Deliberately NOT spelled ``_repeated_structure``: many existing "
            "sidecars carry an inert ``_repeated_structure`` block that nothing "
            "has ever read. Honoring that name would silently activate all of "
            "them and move their scores, so a file must opt in under a name no "
            "legacy file uses."
        ),
    )

    @property
    def eval_data_schema(self) -> dict[str, Any]:
        """``data_schema`` with row identity re-attached, for scoring only.

        Inference must keep using ``data_schema`` (plain, provider-agnostic JSON
        Schema). Alignment code that honors a schema-declared ``identity_key``
        treats it as authoritative, so it has to see the same block it saw when
        the declaration still lived inside the schema.
        """
        if not self.row_identity:
            return self.data_schema
        return {**self.data_schema, "repeated_structure": self.row_identity}

    schema_field_metric_groups: dict[str, list[str]] | None = Field(
        default=None,
        alias="_schema_field_metric_groups",
        description=(
            "Named groups of dot-separated field paths (from test.json "
            "_schema_field_metric_groups). Each group is scored as 'schema_field_accuracy_<name>' "
            "by projecting expected_output onto the group's paths and running the "
            "JSON subset match."
        ),
    )

    @field_validator("test_rules", mode="before")
    @classmethod
    def _coerce_extract_rules(cls, value: list[dict[str, Any]] | None) -> list[ExtractRuleUnion] | None:
        if value is None:
            return None
        out: list[ExtractRuleUnion] = []
        for rule in value:
            if isinstance(rule, ExtractFieldTestRule):
                out.append(rule)
                continue
            if isinstance(rule, dict) and rule.get("type") == "extract_field":
                out.append(ExtractFieldTestRule.model_validate(rule))
                continue
            out.append(rule)
        return out

    def get_extract_field_rules(self) -> list[ExtractFieldTestRule]:
        """Return only typed extract_field rules from test_rules."""
        if not self.test_rules:
            return []
        return [rule for rule in self.test_rules if isinstance(rule, ExtractFieldTestRule)]


def _coerce_mixed_rule_list(
    value: list[dict[str, Any]] | list[LayoutTestRule | ParseRule | ExtractFieldTestRule] | None,
) -> list[LayoutTestRule | ParseRule | ExtractFieldTestRule]:
    """Coerce a mixed rule list (dicts + typed models) into typed models.

    Used by ``ParseTestCase`` so the same on-disk payload (``type=layout``
    entries alongside ``type=present`` / ``type=extract_field`` etc.)
    validates identically regardless of which mix is present.
    """
    if value is None:
        return []

    typed_rules: list[LayoutTestRule | ParseRule | ExtractFieldTestRule] = []
    for rule in value:
        if isinstance(rule, (LayoutTestRule, ExtractFieldTestRule)):
            typed_rules.append(rule)
            continue

        if isinstance(rule, dict):
            rule_type = rule.get("type")
            if rule_type == "layout":
                typed_rules.append(LayoutTestRule.model_validate(rule))
            elif rule_type == "extract_field":
                typed_rules.append(ExtractFieldTestRule.model_validate(rule))
            else:
                typed_rules.append(coerce_parse_rule(rule))
            continue

        typed_rules.append(coerce_parse_rule(rule))

    return typed_rules


class ParseTestCase(BaseTestCase):
    """Test case for PARSE product type.

    Accepts parse rules (``present``, ``order``, ``table``, ...), layout rules,
    and ``extract_field`` grounding rules on the same ``test_rules`` list, so
    parse pipelines with granular bboxes can be cross-evaluated against
    extract-field grounding ground truth.
    """

    test_rules: list[LayoutTestRule | ParseRule | ExtractFieldTestRule] | None = Field(
        default=None,
        description="List of rule-based test definitions (from test.json test_rules)",
    )
    expected_markdown: str | None = Field(
        default=None,
        description="Ground truth markdown for comparison (from test.json expected_markdown)",
    )
    qa_config: QAConfig | None = Field(
        default=None,
        description=(
            "Optional QA configuration for question-answering evaluation (from test.json question/answer fields)"
        ),
    )
    qa_configs: list[QAConfig] | None = Field(
        default=None,
        description=(
            "Optional list of QA configurations (from test.json qa_configs array). "
            "Expanded into per-question evaluation tasks by the evaluation runner."
        ),
    )
    allow_splitting_ambiguous_merged_tables: bool = Field(
        default=False,
        description=(
            "When True, try splitting a merged predicted table to match GT table "
            "structure for ambiguous side-by-side layouts"
        ),
    )
    trm_unsupported: bool = Field(
        default=False,
        description=(
            "When True, the table_record_match metric is unreliable for this "
            "document's tables; grits_trm_composite falls back to grits_con."
        ),
    )
    max_top_title_rows: int = Field(
        default=1,
        description=(
            "Cap on how many top <th> spanning title rows the strip stage "
            "removes from the header block of each table (both GT and "
            "predicted) before any table metric runs. Default 1 matches "
            "today's behavior. Set to 0 to disable all top-of-table "
            "stripping (no leading <td> title rows, no top <th> titles). "
            "A rowspanning title row consumes 1 slot from the cap "
            "regardless of its original rowspan. The bottom-<th> title "
            "strip is independent and not governed by this field."
        ),
    )

    data_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional extraction JSON schema when layout rules coexist with extract annotations",
    )

    @field_validator("test_rules", mode="before")
    @classmethod
    def _coerce_parse_rules(
        cls,
        value: list[dict[str, Any]] | None,
    ) -> list[LayoutTestRule | ParseRule | ExtractFieldTestRule] | None:
        if value is None:
            return None
        return _coerce_mixed_rule_list(value)

    def get_extract_field_rules(self) -> list[ExtractFieldTestRule]:
        """Return only the typed `extract_field` rules from `test_rules`."""
        if not self.test_rules:
            return []
        return [rule for rule in self.test_rules if isinstance(rule, ExtractFieldTestRule)]

    # -- Layout accessors so PARSE-hint mixed corpora expose the same layout
    #    interface as LayoutDetectionTestCase. --

    def get_layout_rules(self, page: int | None = None) -> list[LayoutTestRule]:
        """Return layout rules, optionally filtered by 1-indexed page."""
        rules: list[LayoutTestRule] = []
        for rule in self.test_rules or []:
            if isinstance(rule, LayoutTestRule) and (page is None or rule.page == page):
                rules.append(rule)
        return rules

    def get_layout_annotations(self, page: int | None = None) -> list[LayoutAnnotation]:
        """Return layout annotations as ``LayoutAnnotation`` objects."""
        return [rule.to_layout_annotation() for rule in self.get_layout_rules(page)]

    def get_page_indices(self) -> list[int]:
        """Return sorted 0-indexed page indices that carry layout rules."""
        pages: set[int] = set()
        for rule in self.test_rules or []:
            if isinstance(rule, LayoutTestRule):
                pages.add(rule.page - 1)  # Convert to 0-indexed
        return sorted(pages)


class LayoutDetectionTestCase(BaseTestCase):
    """Test case for LAYOUT_DETECTION product type.

    Layout annotations are stored in test_rules with type="layout" and normalized
    bounding boxes in [0,1] range. This enables unified loading with other test types.
    """

    test_rules: list[LayoutTestRule | ParseRule] = Field(
        default_factory=list,
        description="Test rules including layout annotations (type='layout')",
    )
    ontology: str | None = Field(
        default=None,
        description="Target ontology for evaluation (e.g., basic, canonical)",
    )
    source_ontology: str | None = Field(
        default=None,
        description="Original ontology of the source dataset (if provided)",
    )
    source_dataset: str | None = Field(default=None, description="Source dataset identifier (e.g., 'DocLayNet-v1.2')")
    source_id: str | None = Field(default=None, description="Original ID in source dataset")
    page_index: int = Field(default=0, description="Page index for single-page test (0-indexed)")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata (doc_category, complexity_rank, etc.)"
    )

    @field_validator("test_rules", mode="before")
    @classmethod
    def _coerce_layout_or_parse_rules(cls, value: list[dict[str, Any]] | None) -> list[LayoutTestRule | ParseRule]:
        if value is None:
            return []

        typed_rules: list[LayoutTestRule | ParseRule] = []
        for rule in value:
            if isinstance(rule, (LayoutTestRule,)):
                typed_rules.append(rule)
                continue

            if isinstance(rule, dict):
                rule_type = rule.get("type")
                if rule_type == "layout":
                    typed_rules.append(LayoutTestRule.model_validate(rule))
                else:
                    typed_rules.append(coerce_parse_rule(rule))
                continue

            typed_rules.append(coerce_parse_rule(rule))

        return typed_rules

    def get_layout_rules(self, page: int | None = None) -> list[LayoutTestRule]:
        """Get layout test rules, optionally filtered by page.

        :param page: 1-indexed page number to filter by (None for all pages)
        :return: List of LayoutTestRule objects
        """
        layout_rules: list[LayoutTestRule] = []
        for rule in self.test_rules:
            if isinstance(rule, LayoutTestRule):
                if page is None or rule.page == page:
                    layout_rules.append(rule)
        return layout_rules

    def get_layout_annotations(self, page: int | None = None) -> list[LayoutAnnotation]:
        """Get layout annotations as LayoutAnnotation objects.

        :param page: 1-indexed page number to filter by (None for all pages)
        :return: List of LayoutAnnotation objects (with normalized coordinates)
        """
        return [rule.to_layout_annotation() for rule in self.get_layout_rules(page)]

    def get_page_indices(self) -> list[int]:
        """Get list of unique page indices (0-indexed) with layout annotations."""
        pages: set[int] = set()
        for rule in self.test_rules:
            if isinstance(rule, LayoutTestRule):
                pages.add(rule.page - 1)  # Convert to 0-indexed
        return sorted(pages)


# Union type for backward compatibility
TestCase = ExtractTestCase | ParseTestCase | LayoutDetectionTestCase


def iter_rule_evidence(rule: ExtractFieldTestRule) -> list[FieldEvidence]:
    """Normalize legacy bboxes/expected_value or v0.2 evidence list into FieldEvidence entries.

    Single normalization point — every metric path consumes this. When ``rule.evidence`` is
    set, returns it verbatim. Otherwise synthesizes one entry per legacy ``ExtractFieldBbox``
    using ``rule.expected_value``, with ``coarse=False``. Returns an empty list when the rule
    has neither (in which case M1 falls back to comparing ``rule.expected_value`` directly).
    """
    if rule.evidence is not None:
        return list(rule.evidence)
    return [
        FieldEvidence(
            page=bbox.page,
            bbox=list(bbox.bbox),
            quote=None,
            value=rule.expected_value,
            coarse=False,
        )
        for bbox in rule.bboxes
    ]
