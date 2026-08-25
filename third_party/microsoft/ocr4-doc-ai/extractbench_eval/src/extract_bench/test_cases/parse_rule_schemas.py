"""Pydantic schemas for parse rule payloads.

The parse rule pipeline uses loosely typed dictionaries today. These models add
structured validation while keeping compatibility with legacy payloads via
`extra="allow"`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from extract_bench.evaluation.metrics.parse.test_types import ParseRuleType


class DictCompatPydanticModel(BaseModel):
    """Base model with a light dict-like `get` interface."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    def get(self, key: str, default: Any = None) -> Any:
        # O(1) lookup: check field names, extra fields, then aliases
        cls_fields = type(self).model_fields
        if key in cls_fields or (self.model_extra and key in self.model_extra):
            return getattr(self, key)
        # Check aliases (e.g. "_csv_path" → csv_path)
        for field_name, field_info in cls_fields.items():
            if field_info.alias == key:
                return getattr(self, field_name)
        return default


class ParseRuleBase(DictCompatPydanticModel):
    """Base schema shared by all typed parse rules.

    `id` is intentionally optional so existing datasets without rule ids keep
    loading cleanly while new ids can be added incrementally.
    """

    type: str
    id: str | None = Field(
        default=None,
        description="Optional stable identifier for the rule",
    )
    page: int | None = Field(default=None)
    max_diffs: int | float = Field(default=0, description="Allowed Levenshtein distance")
    tags: list[str] = Field(default_factory=list)
    layout_id: str | None = Field(
        default=None,
        description="Primary linked layout element id for visual grounding.",
    )
    layout_ids: list[str] = Field(
        default_factory=list,
        description="All linked layout element ids for visual grounding.",
    )
    layout_bindings: dict[str, str | list[str]] = Field(
        default_factory=dict,
        description="Optional role-based grounding map (e.g. before/after for order rules).",
    )

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    @model_validator(mode="after")
    def normalize_layout_grounding(self) -> ParseRuleBase:
        """Normalize grounding fields into a consistent, deduped representation."""

        ordered_ids: list[str] = []
        if self.layout_id:
            ordered_ids.append(self.layout_id)

        if self.layout_ids:
            ordered_ids.extend(layout_id for layout_id in self.layout_ids if isinstance(layout_id, str))

        if self.layout_bindings:
            for value in self.layout_bindings.values():
                if isinstance(value, str):
                    ordered_ids.append(value)
                elif isinstance(value, list):
                    ordered_ids.extend(v for v in value if isinstance(v, str))

        deduped_ids: list[str] = []
        seen: set[str] = set()
        for layout_id in ordered_ids:
            if layout_id and layout_id not in seen:
                seen.add(layout_id)
                deduped_ids.append(layout_id)

        self.layout_ids = deduped_ids
        if self.layout_id is None and self.layout_ids:
            self.layout_id = self.layout_ids[0]
        return self


class _HasAnchorCells(BaseModel):
    """Shared table anchor cells field."""

    table_anchor_cells: list[Any] | None = Field(default_factory=list)
    model_config = ConfigDict(extra="allow")

    @field_validator("table_anchor_cells", mode="before")
    @classmethod
    def _coerce_null_to_list(cls, v: Any) -> list[Any]:
        return v if v is not None else []


class ParsePresenceRule(ParseRuleBase):
    """Schema for `present` and `absent` rules."""

    type: Literal[ParseRuleType.PRESENT.value, ParseRuleType.ABSENT.value]
    text: str = ""
    keep_formatting_text_normalisation: bool = False
    case_sensitive: bool = True
    first_n: int | None = None
    last_n: int | None = None
    count: int | None = None


class _SentenceBagRule(ParseRuleBase):
    bag_of_sentence: dict[str, int] = Field(default_factory=dict)


class ParseUnexpectedSentenceRule(_SentenceBagRule):
    type: Literal[ParseRuleType.UNEXPECTED_SENTENCE.value]


class ParseUnexpectedSentencePercentRule(_SentenceBagRule):
    type: Literal[ParseRuleType.UNEXPECTED_SENTENCE_PERCENT.value]
    original_md: str | None = Field(
        default=None,
        description=(
            "Original ground-truth markdown; when provided, sentences found in this text are not counted as unexpected."
        ),
    )


class ParseTooManySentenceOccurrenceRule(_SentenceBagRule):
    type: Literal[ParseRuleType.TOO_MANY_SENTENCE_OCCURENCE.value]


class ParseTooManySentenceOccurrencePercentRule(_SentenceBagRule):
    type: Literal[ParseRuleType.TOO_MANY_SENTENCE_OCCURENCE_PERCENT.value]


class ParseMissingSentenceRule(_SentenceBagRule):
    type: Literal[ParseRuleType.MISSING_SENTENCE.value]


class ParseMissingSentencePercentRule(_SentenceBagRule):
    type: Literal[ParseRuleType.MISSING_SENTENCE_PERCENT.value]


class ParseMissingSpecificSentenceRule(ParseRuleBase):
    """Check that a specific sentence is present (fail if missing)."""

    type: Literal[ParseRuleType.MISSING_SPECIFIC_SENTENCE.value]
    sentence: str = Field(description="The specific sentence to look for.")


class _WordBagRule(ParseRuleBase):
    bag_of_word: dict[str, int] = Field(default_factory=dict)


class ParseUnexpectedWordRule(_WordBagRule):
    type: Literal[ParseRuleType.UNEXPECTED_WORD.value]


class ParseUnexpectedWordPercentRule(_WordBagRule):
    type: Literal[ParseRuleType.UNEXPECTED_WORD_PERCENT.value]


class ParseTooManyWordOccurrenceRule(_WordBagRule):
    type: Literal[ParseRuleType.TOO_MANY_WORD_OCCURENCE.value]


class ParseTooManyWordOccurrencePercentRule(_WordBagRule):
    type: Literal[ParseRuleType.TOO_MANY_WORD_OCCURENCE_PERCENT.value]


class ParseMissingWordRule(_WordBagRule):
    type: Literal[ParseRuleType.MISSING_WORD.value]


class ParseMissingWordPercentRule(_WordBagRule):
    type: Literal[ParseRuleType.MISSING_WORD_PERCENT.value]


class ParseMissingSpecificWordRule(ParseRuleBase):
    """Check that a specific word is present (fail if missing)."""

    type: Literal[ParseRuleType.MISSING_SPECIFIC_WORD.value]
    word: str = Field(description="The specific word to look for.")


class ParseExtraContentRule(_SentenceBagRule):
    type: Literal[ParseRuleType.EXTRA_CONTENT.value]


class ParseBaselineRule(ParseRuleBase):
    type: Literal[ParseRuleType.BASELINE.value]
    max_length: int | None = None
    max_length_skips_image_alt_tags: bool = False
    max_repeats: int = 30
    check_disallowed_characters: bool = True


class ParseOrderRule(ParseRuleBase):
    type: Literal[ParseRuleType.ORDER.value]
    before: str = ""
    after: str = ""
    keep_formatting_text_normalisation: bool = False

    @model_validator(mode="after")
    def validate_layout_bindings(self) -> ParseOrderRule:
        """Validate role-based layout grounding for order rules."""

        if not self.layout_bindings:
            return self

        allowed_roles = {"before", "after"}
        invalid_roles = sorted(role for role in self.layout_bindings if role not in allowed_roles)
        if invalid_roles:
            invalid = ", ".join(invalid_roles)
            raise ValueError(f"Order rule layout_bindings only supports roles 'before' and 'after'; got: {invalid}")

        for role, value in self.layout_bindings.items():
            if isinstance(value, list):
                raise ValueError(f"Order rule layout_bindings.{role} must be a single layout id string")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Order rule layout_bindings.{role} must be a non-empty layout id string")

        before_layout = self.layout_bindings.get("before")
        if isinstance(before_layout, str):
            # Some legacy datasets contain stale `layout_id` values on order rules.
            # Canonicalize to the role-specific binding instead of failing load.
            if self.layout_id != before_layout:
                self.layout_id = before_layout

            if before_layout in self.layout_ids:
                ordered_ids = [layout_id for layout_id in self.layout_ids if layout_id != before_layout]
                self.layout_ids = [before_layout, *ordered_ids]
            else:
                self.layout_ids = [before_layout, *self.layout_ids]
        return self


class ParseTableRule(ParseRuleBase):
    type: Literal[ParseRuleType.TABLE.value]
    cell: str = ""
    up: str | None = None
    down: str | None = None
    left: str | None = None
    right: str | None = None
    top_heading: str | None = None
    left_heading: str | None = None
    ignore_markdown_tables: bool = False


class ParseTablesValuesRule(ParseRuleBase):
    type: Literal[ParseRuleType.TABLES_VALUES.value]
    table_variations: list[Any] | None = None
    json_path: str | None = None
    table_match_threshold: float = 0.8
    table_values_match_threshold: float = 0.9
    add_check_num_rows_test: bool = True
    add_check_num_cols_test: bool = True


class ParseTablesNumRowsRule(ParseRuleBase):
    type: Literal[ParseRuleType.TABLES_NUM_ROWS.value]
    expected_num_rows: int = 0
    actual_num_rows: int | None = None


class ParseTablesNumColsRule(ParseRuleBase):
    type: Literal[ParseRuleType.TABLES_NUM_COLS.value]
    expected_num_cols: int = 0
    actual_num_cols: int | None = None


class ParseTableColspanRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_COLSPAN.value]
    cell: str = ""
    expected_colspan: int = 1


class ParseTableRowspanRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_ROWSPAN.value]
    cell: str = ""
    expected_rowspan: int = 1


class ParseTableSameRowRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_SAME_ROW.value]
    cell_a: str = ""
    cell_b: str = ""


class ParseTableSameColumnRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_SAME_COLUMN.value]
    cell_a: str = ""
    cell_b: str = ""


class ParseTableHeaderChainRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_HEADER_CHAIN.value]
    data_cell: str = ""
    column_headers: list[Any] = Field(default_factory=list)
    row_headers: list[Any] = Field(default_factory=list)


class ParseTableAdjacentUpRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_ADJACENT_UP.value]
    anchor_cell: str = ""
    expected_neighbor: str = ""


class ParseTableAdjacentDownRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_ADJACENT_DOWN.value]
    anchor_cell: str = ""
    expected_neighbor: str = ""


class ParseTableAdjacentLeftRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_ADJACENT_LEFT.value]
    anchor_cell: str = ""
    expected_neighbor: str = ""


class ParseTableAdjacentRightRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_ADJACENT_RIGHT.value]
    anchor_cell: str = ""
    expected_neighbor: str = ""


class ParseTableTopHeaderRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_TOP_HEADER.value]
    data_cell: str = ""
    expected_header: str = ""


class ParseTableLeftHeaderRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_LEFT_HEADER.value]
    data_cell: str = ""
    expected_header: str = ""


class ParseTableNoLeftRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_NO_LEFT.value]
    cell: str = ""


class ParseTableNoRightRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_NO_RIGHT.value]
    cell: str = ""


class ParseTableNoAboveRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_NO_ABOVE.value]
    cell: str = ""


class ParseTableNoBelowRule(ParseRuleBase, _HasAnchorCells):
    type: Literal[ParseRuleType.TABLE_NO_BELOW.value]
    cell: str = ""


class ParseChartDataPointRule(ParseRuleBase):
    type: Literal[ParseRuleType.CHART_DATA_POINT.value]
    value: str | int | float | None = None
    labels: list[str] = Field(default_factory=list)
    normalize_numbers: bool = True
    relative_tolerance: float = 0.01


class ParseChartDataArrayLabelsRule(ParseRuleBase):
    type: Literal[ParseRuleType.CHART_DATA_ARRAY_LABELS.value]
    data: list[Any] = Field(default_factory=list)
    x_axis_shuffle: bool = False
    transposed: bool = False
    csv_path: str | None = Field(default=None, alias="_csv_path")


class ParseChartDataArrayDataRule(ParseRuleBase):
    type: Literal[ParseRuleType.CHART_DATA_ARRAY_DATA.value]
    data: list[Any] = Field(default_factory=list)
    x_axis_shuffle: bool = False
    y_axis_shuffle: bool = False
    normalize_numbers: bool = True
    transposed: bool = False
    csv_path: str | None = Field(default=None, alias="_csv_path")


class ParseFormattingRule(ParseRuleBase):
    type: Literal[
        ParseRuleType.IS_UNDERLINE.value,
        ParseRuleType.IS_NOT_UNDERLINE.value,
        ParseRuleType.IS_BOLD.value,
        ParseRuleType.IS_NOT_BOLD.value,
        ParseRuleType.IS_STRIKEOUT.value,
        ParseRuleType.IS_NOT_STRIKEOUT.value,
        ParseRuleType.IS_ITALIC.value,
        ParseRuleType.IS_NOT_ITALIC.value,
        ParseRuleType.IS_MARK.value,
        ParseRuleType.IS_NOT_MARK.value,
        ParseRuleType.IS_SUP.value,
        ParseRuleType.IS_NOT_SUP.value,
        ParseRuleType.IS_SUB.value,
        ParseRuleType.IS_NOT_SUB.value,
    ]
    text: str = ""


class ParseMarkColorRule(ParseRuleBase):
    """Schema for `mark_color` rules.

    Validates that text is wrapped in a ``<mark>`` tag AND that the tag
    contains the expected color string in one of its attributes
    (e.g. ``style="background-color: yellow"``, ``background="yellow"``,
    ``backgroundColor="yellow"``).
    """

    type: Literal[ParseRuleType.MARK_COLOR.value]
    text: str = ""
    color: str = ""


class ParseTitleRule(ParseRuleBase):
    type: Literal[ParseRuleType.IS_TITLE.value]
    text: str = ""
    level: int | None = None


class ParseLatexRule(ParseRuleBase):
    type: Literal[ParseRuleType.IS_LATEX.value]
    formula: str = ""


class ParseCodeBlockRule(ParseRuleBase):
    type: Literal[ParseRuleType.IS_CODE_BLOCK.value]
    language: str = ""
    code: str = ""


class ParseTitleHierarchyPercentRule(ParseRuleBase):
    type: Literal[ParseRuleType.TITLE_HIERARCHY_PERCENT.value]
    title_hierarchy: dict[str, Any] = Field(default_factory=dict)


class ParsePageSectionRule(ParseRuleBase):
    type: Literal[ParseRuleType.IS_HEADER.value, ParseRuleType.IS_FOOTER.value]
    text: str = ""


class ParseBagOfDigitPercentRule(ParseRuleBase):
    """Schema for `bag_of_digit_percent` rule.

    Compares digit frequency (0-9) between expected markdown and actual output.
    Digits inside HTML tag attributes (e.g. colspan="2") are excluded.
    """

    type: Literal[ParseRuleType.BAG_OF_DIGIT_PERCENT.value]
    bag_of_digit: dict[str, int] = Field(default_factory=dict)


class ParseRotateCheckRule(ParseRuleBase):
    type: Literal[ParseRuleType.ROTATE_CHECK.value]
    value: int | float | str | None = None


class ParseFormFieldRule(ParseRuleBase):
    """Schema for `form_field` rules.

    Locates a labeled field in the parsed markdown/HTML and checks its value.
    Used for benchmarking form (key-value, checkbox, signature) extraction.

    `value` may be a list of strings to declare acceptable alternatives for
    genuinely ambiguous fields (e.g. illegible handwriting). The evaluator
    passes if the parsed value matches *any* alternative. Use sparingly —
    most rules should be a single string. List form is only for value_type
    "text" and "signature".
    """

    type: Literal[ParseRuleType.FORM_FIELD.value]
    label: str = ""
    value: str | bool | list[str] = ""
    value_type: Literal["text", "checkbox", "signature"] = "text"


type ParseRule = (
    ParsePresenceRule
    | ParseUnexpectedSentenceRule
    | ParseUnexpectedSentencePercentRule
    | ParseTooManySentenceOccurrenceRule
    | ParseTooManySentenceOccurrencePercentRule
    | ParseMissingSentenceRule
    | ParseMissingSentencePercentRule
    | ParseMissingSpecificSentenceRule
    | ParseUnexpectedWordRule
    | ParseUnexpectedWordPercentRule
    | ParseTooManyWordOccurrenceRule
    | ParseTooManyWordOccurrencePercentRule
    | ParseMissingWordRule
    | ParseMissingWordPercentRule
    | ParseMissingSpecificWordRule
    | ParseExtraContentRule
    | ParseBaselineRule
    | ParseOrderRule
    | ParseTableRule
    | ParseTablesValuesRule
    | ParseTablesNumRowsRule
    | ParseTablesNumColsRule
    | ParseTableColspanRule
    | ParseTableRowspanRule
    | ParseTableSameRowRule
    | ParseTableSameColumnRule
    | ParseTableHeaderChainRule
    | ParseTableAdjacentUpRule
    | ParseTableAdjacentDownRule
    | ParseTableAdjacentLeftRule
    | ParseTableAdjacentRightRule
    | ParseTableTopHeaderRule
    | ParseTableLeftHeaderRule
    | ParseTableNoLeftRule
    | ParseTableNoRightRule
    | ParseTableNoAboveRule
    | ParseTableNoBelowRule
    | ParseChartDataPointRule
    | ParseChartDataArrayLabelsRule
    | ParseChartDataArrayDataRule
    | ParseFormattingRule
    | ParseMarkColorRule
    | ParseLatexRule
    | ParseCodeBlockRule
    | ParseTitleRule
    | ParseTitleHierarchyPercentRule
    | ParsePageSectionRule
    | ParseBagOfDigitPercentRule
    | ParseRotateCheckRule
    | ParseFormFieldRule
)

type ParseRuleInput = ParseRule | dict[str, Any]

_RULE_TYPE_TO_MODEL: dict[str, type[ParseRule]] = {
    ParseRuleType.PRESENT.value: ParsePresenceRule,
    ParseRuleType.ABSENT.value: ParsePresenceRule,
    ParseRuleType.UNEXPECTED_SENTENCE.value: ParseUnexpectedSentenceRule,
    ParseRuleType.UNEXPECTED_SENTENCE_PERCENT.value: ParseUnexpectedSentencePercentRule,
    ParseRuleType.TOO_MANY_SENTENCE_OCCURENCE.value: ParseTooManySentenceOccurrenceRule,
    ParseRuleType.TOO_MANY_SENTENCE_OCCURENCE_PERCENT.value: ParseTooManySentenceOccurrencePercentRule,
    ParseRuleType.MISSING_SENTENCE.value: ParseMissingSentenceRule,
    ParseRuleType.MISSING_SENTENCE_PERCENT.value: ParseMissingSentencePercentRule,
    ParseRuleType.MISSING_SPECIFIC_SENTENCE.value: ParseMissingSpecificSentenceRule,
    ParseRuleType.UNEXPECTED_WORD.value: ParseUnexpectedWordRule,
    ParseRuleType.UNEXPECTED_WORD_PERCENT.value: ParseUnexpectedWordPercentRule,
    ParseRuleType.TOO_MANY_WORD_OCCURENCE.value: ParseTooManyWordOccurrenceRule,
    ParseRuleType.TOO_MANY_WORD_OCCURENCE_PERCENT.value: ParseTooManyWordOccurrencePercentRule,
    ParseRuleType.MISSING_WORD.value: ParseMissingWordRule,
    ParseRuleType.MISSING_WORD_PERCENT.value: ParseMissingWordPercentRule,
    ParseRuleType.MISSING_SPECIFIC_WORD.value: ParseMissingSpecificWordRule,
    ParseRuleType.EXTRA_CONTENT.value: ParseExtraContentRule,
    ParseRuleType.BASELINE.value: ParseBaselineRule,
    ParseRuleType.ORDER.value: ParseOrderRule,
    ParseRuleType.TABLE.value: ParseTableRule,
    ParseRuleType.TABLES_VALUES.value: ParseTablesValuesRule,
    ParseRuleType.TABLES_NUM_ROWS.value: ParseTablesNumRowsRule,
    ParseRuleType.TABLES_NUM_COLS.value: ParseTablesNumColsRule,
    ParseRuleType.TABLE_COLSPAN.value: ParseTableColspanRule,
    ParseRuleType.TABLE_ROWSPAN.value: ParseTableRowspanRule,
    ParseRuleType.TABLE_SAME_ROW.value: ParseTableSameRowRule,
    ParseRuleType.TABLE_SAME_COLUMN.value: ParseTableSameColumnRule,
    ParseRuleType.TABLE_HEADER_CHAIN.value: ParseTableHeaderChainRule,
    ParseRuleType.TABLE_ADJACENT_UP.value: ParseTableAdjacentUpRule,
    ParseRuleType.TABLE_ADJACENT_DOWN.value: ParseTableAdjacentDownRule,
    ParseRuleType.TABLE_ADJACENT_LEFT.value: ParseTableAdjacentLeftRule,
    ParseRuleType.TABLE_ADJACENT_RIGHT.value: ParseTableAdjacentRightRule,
    ParseRuleType.TABLE_TOP_HEADER.value: ParseTableTopHeaderRule,
    ParseRuleType.TABLE_LEFT_HEADER.value: ParseTableLeftHeaderRule,
    ParseRuleType.TABLE_NO_LEFT.value: ParseTableNoLeftRule,
    ParseRuleType.TABLE_NO_RIGHT.value: ParseTableNoRightRule,
    ParseRuleType.TABLE_NO_ABOVE.value: ParseTableNoAboveRule,
    ParseRuleType.TABLE_NO_BELOW.value: ParseTableNoBelowRule,
    ParseRuleType.CHART_DATA_POINT.value: ParseChartDataPointRule,
    ParseRuleType.CHART_DATA_ARRAY_LABELS.value: ParseChartDataArrayLabelsRule,
    ParseRuleType.CHART_DATA_ARRAY_DATA.value: ParseChartDataArrayDataRule,
    ParseRuleType.IS_UNDERLINE.value: ParseFormattingRule,
    ParseRuleType.IS_NOT_UNDERLINE.value: ParseFormattingRule,
    ParseRuleType.IS_BOLD.value: ParseFormattingRule,
    ParseRuleType.IS_NOT_BOLD.value: ParseFormattingRule,
    ParseRuleType.IS_STRIKEOUT.value: ParseFormattingRule,
    ParseRuleType.IS_NOT_STRIKEOUT.value: ParseFormattingRule,
    ParseRuleType.IS_ITALIC.value: ParseFormattingRule,
    ParseRuleType.IS_NOT_ITALIC.value: ParseFormattingRule,
    ParseRuleType.IS_MARK.value: ParseFormattingRule,
    ParseRuleType.IS_NOT_MARK.value: ParseFormattingRule,
    ParseRuleType.MARK_COLOR.value: ParseMarkColorRule,
    ParseRuleType.IS_SUP.value: ParseFormattingRule,
    ParseRuleType.IS_NOT_SUP.value: ParseFormattingRule,
    ParseRuleType.IS_SUB.value: ParseFormattingRule,
    ParseRuleType.IS_NOT_SUB.value: ParseFormattingRule,
    ParseRuleType.IS_LATEX.value: ParseLatexRule,
    ParseRuleType.IS_CODE_BLOCK.value: ParseCodeBlockRule,
    ParseRuleType.IS_TITLE.value: ParseTitleRule,
    ParseRuleType.TITLE_HIERARCHY_PERCENT.value: ParseTitleHierarchyPercentRule,
    ParseRuleType.IS_HEADER.value: ParsePageSectionRule,
    ParseRuleType.IS_FOOTER.value: ParsePageSectionRule,
    ParseRuleType.BAG_OF_DIGIT_PERCENT.value: ParseBagOfDigitPercentRule,
    ParseRuleType.ROTATE_CHECK.value: ParseRotateCheckRule,
    ParseRuleType.FORM_FIELD.value: ParseFormFieldRule,
}


def get_rule_type(rule: ParseRuleInput | Any) -> str | None:
    """Return the `type` field from either a typed schema or raw dict."""

    if isinstance(rule, ParseRuleBase):
        return rule.type
    if isinstance(rule, BaseModel):
        value = getattr(rule, "type", None)
        return value if isinstance(value, str) else None
    if isinstance(rule, dict):
        raw_type = rule.get("type")
        return raw_type if isinstance(raw_type, str) else None
    return None


def get_rule_id(rule: ParseRuleInput | Any) -> str | None:
    """Return optional `id` from either typed schema or raw dict."""

    if isinstance(rule, ParseRuleBase):
        return rule.id
    if isinstance(rule, BaseModel):
        value = getattr(rule, "id", None)
        return value if isinstance(value, str) else None
    if isinstance(rule, dict):
        value = rule.get("id")
        return value if isinstance(value, str) else None
    return None


def get_rule_page(rule: ParseRuleInput | Any) -> int | None:
    """Return optional `page` from either typed schema or raw dict."""

    if isinstance(rule, ParseRuleBase):
        return rule.page
    if isinstance(rule, BaseModel):
        value = getattr(rule, "page", None)
        return value if isinstance(value, int) else None
    if isinstance(rule, dict):
        value = rule.get("page")
        return value if isinstance(value, int) else None
    return None


def get_rule_layout_id(rule: ParseRuleInput | Any) -> str | None:
    """Return primary `layout_id` from either typed schema or raw dict."""

    if isinstance(rule, ParseRuleBase):
        return rule.layout_id
    if isinstance(rule, BaseModel):
        value = getattr(rule, "layout_id", None)
        if isinstance(value, str):
            return value
        layout_ids = get_rule_layout_ids(rule)
        return layout_ids[0] if layout_ids else None
    if isinstance(rule, dict):
        value = rule.get("layout_id")
        if isinstance(value, str):
            return value
        layout_ids = get_rule_layout_ids(rule)
        return layout_ids[0] if layout_ids else None
    return None


def get_rule_layout_ids(rule: ParseRuleInput | Any) -> list[str]:
    """Return normalized `layout_ids` from either typed schema or raw dict."""

    if isinstance(rule, ParseRuleBase):
        return list(rule.layout_ids)
    if isinstance(rule, BaseModel):
        value = getattr(rule, "layout_ids", None)
        if isinstance(value, list):
            return [v for v in value if isinstance(v, str)]
        return []
    if isinstance(rule, dict):
        value = rule.get("layout_ids")
        if isinstance(value, list):
            return [v for v in value if isinstance(v, str)]
        fallback = rule.get("layout_id")
        if isinstance(fallback, str):
            return [fallback]
    return []


def get_rule_layout_bindings(rule: ParseRuleInput | Any) -> dict[str, str | list[str]]:
    """Return `layout_bindings` from either typed schema or raw dict."""

    if isinstance(rule, ParseRuleBase):
        return dict(rule.layout_bindings)

    def _is_valid_binding_value(value: Any) -> bool:
        return isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value))

    if isinstance(rule, BaseModel):
        value = getattr(rule, "layout_bindings", None)
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items() if _is_valid_binding_value(v)}
        return {}
    if isinstance(rule, dict):
        value = rule.get("layout_bindings")
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items() if _is_valid_binding_value(v)}
    return {}


def rule_to_dict(rule: ParseRuleInput | Any) -> dict[str, Any]:
    """Serialize a typed or raw rule to a mutable dict for reporting/evals."""

    if isinstance(rule, BaseModel):
        return rule.model_dump(by_alias=True)
    if isinstance(rule, dict):
        return dict(rule)
    raise TypeError(f"Expected ParseRule or dict, got {type(rule)!r}")


def coerce_parse_rule(rule_data: ParseRuleInput | Any) -> ParseRule:
    """Coerce raw rule payloads into the typed parse rule model.

    Idempotent: already-typed rules are returned as-is.
    """

    if isinstance(rule_data, ParseRuleBase):
        return rule_data  # type: ignore[return-value]

    if isinstance(rule_data, dict):
        rule_dict = rule_data
    else:
        raise TypeError(f"Expected rule payload as dict or ParseRule, got {type(rule_data)!r}")

    rule_type = rule_dict.get("type")
    if not isinstance(rule_type, str):
        raise ValueError("Rule payload must contain a string 'type' field")

    model_cls = _RULE_TYPE_TO_MODEL.get(rule_type)
    if model_cls is None:
        raise ValueError(f"Unknown parse rule type: {rule_type}")

    return model_cls.model_validate(rule_dict)


def coerce_parse_rule_list(rules: list[dict[str, Any]] | list[ParseRule]) -> list[ParseRule]:
    """Validate a list of parse rule payloads and return typed objects."""

    parsed: list[ParseRule] = []
    for rule in rules:
        parsed.append(coerce_parse_rule(rule))
    return parsed


def coerce_parse_rule_list_or_none(
    rules: list[dict[str, Any]] | list[ParseRule] | None,
) -> list[ParseRule] | None:
    """Coerce rule lists while preserving explicit None for optional fields."""

    if rules is None:
        return None
    return coerce_parse_rule_list(rules)
