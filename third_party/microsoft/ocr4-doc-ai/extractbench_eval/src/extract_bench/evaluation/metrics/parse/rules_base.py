"""Base class and shared helpers for parse test rules."""

import logging
import re
from html import unescape
from pathlib import Path
from typing import Any

from extract_bench.evaluation.metrics.parse.table_parsing import (
    parse_html_tables,
    parse_markdown_tables,
)
from extract_bench.evaluation.metrics.parse.test_types import ParseRuleType
from extract_bench.schemas.parse_output import ParseOutput
from extract_bench.test_cases.parse_rule_schemas import (
    ParseExtraContentRule,
    ParseMissingSentencePercentRule,
    ParseMissingSentenceRule,
    ParseMissingWordPercentRule,
    ParseMissingWordRule,
    ParseRule,
    ParseRuleInput,
    ParseTableAdjacentDownRule,
    ParseTableAdjacentLeftRule,
    ParseTableAdjacentRightRule,
    ParseTableAdjacentUpRule,
    ParseTableNoAboveRule,
    ParseTableNoBelowRule,
    ParseTableNoLeftRule,
    ParseTableNoRightRule,
    ParseTooManySentenceOccurrencePercentRule,
    ParseTooManySentenceOccurrenceRule,
    ParseTooManyWordOccurrencePercentRule,
    ParseTooManyWordOccurrenceRule,
    ParseUnexpectedSentencePercentRule,
    ParseUnexpectedSentenceRule,
    ParseUnexpectedWordPercentRule,
    ParseUnexpectedWordRule,
    coerce_parse_rule,
    get_rule_type,
)

SentenceBagRuleData = (
    ParseUnexpectedSentenceRule
    | ParseUnexpectedSentencePercentRule
    | ParseTooManySentenceOccurrenceRule
    | ParseTooManySentenceOccurrencePercentRule
    | ParseMissingSentenceRule
    | ParseMissingSentencePercentRule
    | ParseExtraContentRule
)

WordBagRuleData = (
    ParseUnexpectedWordRule
    | ParseUnexpectedWordPercentRule
    | ParseTooManyWordOccurrenceRule
    | ParseTooManyWordOccurrencePercentRule
    | ParseMissingWordRule
    | ParseMissingWordPercentRule
)

AdjacentTableRuleData = (
    ParseTableAdjacentUpRule | ParseTableAdjacentDownRule | ParseTableAdjacentLeftRule | ParseTableAdjacentRightRule
)

NoBorderTableRuleData = ParseTableNoLeftRule | ParseTableNoRightRule | ParseTableNoAboveRule | ParseTableNoBelowRule

# Minimum fuzzy match ratio (0.0-1.0) for cell text comparison in table tests
CELL_FUZZY_MATCH_THRESHOLD = 0.8


logger = logging.getLogger(__name__)


_DATETIME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\s+\d{2}:\d{2}:\d{2}$")
_MONTH_TO_QUARTER = {"01": "1", "04": "2", "07": "3", "10": "4"}

# Scale keywords found in table headers (e.g. "Spending (Millions of national currency)")
_HEADER_SCALE_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\btrillions?\b", re.IGNORECASE), 1e12),
    (re.compile(r"\bbillions?\b", re.IGNORECASE), 1e9),
    (re.compile(r"\bmillions?\b", re.IGNORECASE), 1e6),
    (re.compile(r"\bthousands?\b", re.IGNORECASE), 1e3),
    (re.compile(r"\bhundreds?\b", re.IGNORECASE), 1e2),
]


def _detect_header_scale(header: str) -> float:
    """Detect a numeric scale keyword in a table header string.

    Returns the divisor to apply to raw expected values so they match the
    header's scale.  E.g. header "Millions of USD" → 1e6, meaning raw
    ``495926400`` should become ``~495.9`` for comparison with the table cell.

    Returns 1.0 when no scale keyword is found.
    """
    for pattern, scale in _HEADER_SCALE_PATTERNS:
        if pattern.search(header):
            return scale
    return 1.0


def _normalize_date_str(value: str) -> list[str]:
    """Return canonical date representations for a datetime or short-date string.

    For a full datetime like ``2008-04-01 00:00:00`` returns multiple forms so
    that any common chart label format can match:
      - ``2008`` (year only)
      - ``04-2008`` (MM-YYYY)
      - ``q2-2008`` (quarter)

    For an already-short label like ``Q2-2008`` or ``01-2008`` the function
    returns a lowercase canonical form.

    Returns an empty list when the value is not date-like.
    """
    value = value.strip()

    # Full datetime from CSV: "2008-04-01 00:00:00"
    m = _DATETIME_RE.match(value)
    if m:
        year, month, _ = m.group(1), m.group(2), m.group(3)
        forms = [year, f"{month}-{year}"]
        q = _MONTH_TO_QUARTER.get(month)
        if q:
            forms.append(f"q{q}-{year}")
        return forms

    # Short label from chart: "Q2-2008", "01-2008", "2008"
    low = value.lower().strip()
    # "q2-2008" style
    if re.match(r"^q\d-\d{4}$", low):
        return [low]
    # "01-2008" style (MM-YYYY)
    mm_yyyy = re.match(r"^(\d{1,2})-(\d{4})$", low)
    if mm_yyyy:
        month_padded = mm_yyyy.group(1).zfill(2)
        year = mm_yyyy.group(2)
        forms = [f"{month_padded}-{year}", year]
        q = _MONTH_TO_QUARTER.get(month_padded)
        if q:
            forms.append(f"q{q}-{year}")
        return forms
    # Plain year
    if re.match(r"^\d{4}$", low):
        return [low]

    return []


def _dates_match(expected: str, actual: str) -> bool:
    """Check whether two date strings represent the same point in time."""
    e_forms = _normalize_date_str(expected)
    if not e_forms:
        return False
    a_forms = _normalize_date_str(actual)
    if not a_forms:
        return False
    return bool(set(e_forms) & set(a_forms))


def _detect_csv_skip_rows(csv_path: str | Path) -> int:
    """Detect leading metadata rows in a CSV file.

    Some CSV files (e.g. OECD exports) prepend a title and subtitle row before
    the actual column headers. These metadata rows typically contain only a
    single value per line (or have trailing empty fields due to a trailing
    comma).  This function counts how many leading rows to skip by checking
    whether only the first field has meaningful content.
    """
    import csv as csv_mod

    skip = 0
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv_mod.reader(f)
        for row in reader:
            non_empty = [c for c in row if c.strip()]
            if len(non_empty) <= 1:
                skip += 1
            else:
                break
    return skip


# --- LaTeX stripping for bag-of-sentence/word rules ---
_INLINE_LATEX_PATTERN = re.compile(r"\$(?!\d)(?:[^$\\\n]|\\.)+\$")
_BLOCK_LATEX_PATTERN = re.compile(r"\$\$(?:[^$]|\$[^$])+\$\$", re.DOTALL)
# LaTeX commands to strip (retain only numbers, operators, and variable letters)
_LATEX_COMMAND_PATTERN = re.compile(r"\\(?:[a-zA-Z]+|[^a-zA-Z])")
_LATEX_KEEP_PATTERN = re.compile(r"[0-9a-zA-Z+\-*/=<>().,]")


def _simplify_latex_body(body: str) -> str:
    """Simplify a LaTeX formula body to its essential numbers, operators, and variables.

    Strips LaTeX commands (\\frac, \\int, etc.), delimiters ({, }), and
    whitespace, keeping only digits, ASCII letters, and math operators.
    This produces a stable, comparable token like "a2+b2=c2" instead of
    a generic "LATEX" placeholder, so different formulas remain distinguishable.
    """
    # Strip LaTeX commands but keep their arguments
    simplified = _LATEX_COMMAND_PATTERN.sub(" ", body)
    # Strip braces and other delimiters
    simplified = simplified.replace("{", " ").replace("}", " ").replace("^", "").replace("_", "")
    # Keep only meaningful characters
    result = "".join(ch for ch in simplified if _LATEX_KEEP_PATTERN.match(ch))
    return result if result else "LATEX"


def _strip_and_replace_latex(md_content: str) -> str:
    """
    Remove all block LaTeX (delimited by $$...$$) and replace inline LaTeX
    ($...$) with a simplified representation retaining numbers, operators,
    and variable letters.

    Inline matching ignores ``$`` followed by a digit so currency values like
    ``$15.00`` are preserved and do not accidentally consume large spans between
    multiple dollar amounts in financial prose.
    """

    # Replace block latex with simplified body
    def _replace_block(m: re.Match) -> str:
        body = m.group(0)[2:-2]  # strip $$...$$
        return " " + _simplify_latex_body(body) + " "

    no_block = _BLOCK_LATEX_PATTERN.sub(_replace_block, md_content)

    # Replace inline latex with simplified body
    def _replace_inline(m: re.Match) -> str:
        body = m.group(0)[1:-1]  # strip $...$
        return _simplify_latex_body(body)

    replaced = _INLINE_LATEX_PATTERN.sub(_replace_inline, no_block)
    return replaced


# --- Fenced code block stripping for bag-of-sentence/word rules ---
# ``mermaid`` and ``description`` blocks contain generated diagram syntax or
# metadata that should never produce bag entries.  Matches ``` (3+ backticks)
# with the language tag, through to the closing fence.
_FENCED_CODE_BLOCK_PATTERN = re.compile(
    r"^`{3,}(?:mermaid|description)[^\S\n]*\n.*?^`{3,}[^\S\n]*$",
    flags=re.MULTILINE | re.DOTALL,
)


def _strip_fenced_code_blocks(md_content: str) -> str:
    """Remove ``mermaid`` and ``description`` fenced code blocks from markdown."""
    return _FENCED_CODE_BLOCK_PATTERN.sub(" ", md_content)


_HTML_TABLE_BLOCK_PATTERN = re.compile(
    r"<table\b[^>]*>.*?</table>",
    flags=re.IGNORECASE | re.DOTALL,
)
_HTML_TABLE_START_PATTERN = re.compile(r"<table\b[^>]*>", flags=re.IGNORECASE)


def _strip_html_tables_and_content(md_content: str) -> str:
    """Remove HTML tables and their content from markdown-like text.

    This keeps bag-of-word/sentence rules focused on document prose and avoids
    table payload noise across heterogeneous layouts.
    """
    if "<table" not in md_content.lower():
        return md_content

    stripped = md_content
    total_removed = 0

    while True:
        next_stripped, removed = _HTML_TABLE_BLOCK_PATTERN.subn(" ", stripped)
        if removed == 0:
            break
        stripped = next_stripped
        total_removed += removed

    # Handle malformed HTML where <table ...> has no matching </table>.
    dangling_match = _HTML_TABLE_START_PATTERN.search(stripped)
    if dangling_match:
        stripped = stripped[: dangling_match.start()] + " "
        total_removed += 1

    if total_removed > 0:
        logger.debug("Removed %d HTML table block(s) before bag extraction", total_removed)

    return stripped


def _extract_table_cell_texts(md_content: str) -> list[str]:
    """Extract non-empty cell texts from markdown and HTML tables.

    This is used by missing-content bag rules so table content remains searchable
    while still allowing per-cell sentence/word extraction behavior.

    In addition to individual cell texts, concatenated row texts are emitted so
    that sentences spanning multiple cells in the same row can be matched as
    contiguous substrings after normalization.
    """
    cell_texts: list[str] = []

    for table in parse_markdown_tables(md_content):
        for row in table.data:
            row_parts: list[str] = []
            for cell in row:
                text = str(cell).strip()
                if text:
                    cell_texts.append(text)
                    row_parts.append(text)
            if len(row_parts) > 1:
                cell_texts.append(" ".join(row_parts))

    for table in parse_html_tables(md_content):
        for row in table.data:
            row_parts = []
            for cell in row:
                text = str(cell).strip()
                if text:
                    cell_texts.append(text)
                    row_parts.append(text)
            if len(row_parts) > 1:
                cell_texts.append(" ".join(row_parts))

    return cell_texts


def _augment_with_table_cell_text(md_content: str) -> str:
    """Append table cell text to content without removing original markdown.

    Why: missing_* rules should search substrings everywhere (including tables)
    and treat each table cell as an independent text unit for splitting.
    """
    cell_texts = _extract_table_cell_texts(md_content)
    if not cell_texts:
        return md_content
    return f"{md_content}\n" + "\n".join(cell_texts)


def _unescape_html_entities(text: str) -> str:
    """Decode HTML entities with bounded passes to handle double-escaped payloads.

    Stops early if a decoding pass would introduce *new* ``&`` characters
    (indicating over-decoding, e.g. ``&amp;gt;`` → ``&gt;`` → ``>``).
    At most 2 passes are applied to handle double-encoded entities without
    over-decoding content that contains intentional entities.
    """
    decoded = text
    for _ in range(2):
        next_decoded = unescape(decoded)
        if next_decoded == decoded:
            break
        # Stop if decoding created more ampersands than it resolved — this
        # signals over-decoding (e.g. literal "&gt;" in source code).
        if next_decoded.count("&") > decoded.count("&"):
            break
        decoded = next_decoded
    return decoded


class ParseTestRule:
    """Base class for parse test rules."""

    def __init__(self, rule_data: ParseRule | dict):
        """
        Initialize a test rule from a typed payload or raw dict.

        :param rule_data: Rule payload containing rule definition
        """
        if isinstance(rule_data, dict):
            rule_data = coerce_parse_rule(rule_data)
        self._rule_data = rule_data  # Coerced model for subclass access
        self.type = rule_data.type
        self.page = rule_data.page  # Optional page number (1-indexed)
        self.max_diffs = rule_data.max_diffs
        # Structured parse payload is injected by RuleBasedMetric when available.
        # Keep this optional so direct unit tests can still run rules against raw markdown only.
        self.parse_output: ParseOutput | None = None

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str] | tuple[bool, str, float]:
        """
        Run the test rule against markdown content.

        :param md_content: Markdown content to test
        :param normalized_content: Optional pre-normalized content (performance optimization)
        :return: Tuple of (passed, explanation) or (passed, explanation, score).
            Score is 0.0-1.0 for graduated rules. If omitted, defaults to
            1.0 for passed and 0.0 for failed.
        """
        raise NotImplementedError("Subclasses must implement run()")

    def _get_structured_page_sections(self) -> dict[str, list[str]] | None:
        """Return structured page sections from parse_output when available.

        Why: header/footer/page number checks should prefer dedicated page-level fields
        over markdown tag parsing to avoid coupling rule correctness to markdown stitching.
        """
        if self.parse_output is None or not self.parse_output.layout_pages:
            return None

        target_pages = self.parse_output.layout_pages
        if self.page is not None:
            target_pages = [p for p in target_pages if p.page_number == self.page]

        headers = [p.page_header_markdown for p in target_pages if p.page_header_markdown]
        footers = [p.page_footer_markdown for p in target_pages if p.page_footer_markdown]
        page_numbers = [p.printed_page_number for p in target_pages if p.printed_page_number]

        logger.debug(
            "Structured page sections extracted: page_filter=%s pages=%d headers=%d footers=%d page_numbers=%d",
            self.page,
            len(target_pages),
            len(headers),
            len(footers),
            len(page_numbers),
        )

        return {
            "page_header": headers,
            "page_footer": footers,
            "page_number": page_numbers,
        }


def create_test_rule(rule_data: ParseRuleInput) -> "ParseTestRule":
    """
    Create a test rule from a typed payload.

    :param rule_data: Rule payload containing rule definition
    :return: ParseTestRule instance
    :raises ValueError: If rule type is unknown or invalid
    """
    # Lazy imports to avoid circular dependencies
    from extract_bench.evaluation.metrics.parse.rules_bag import (
        BagOfDigitPercentRule,
        ExtraContentRule,
        MissingSentencePercentRule,
        MissingSentenceRule,
        MissingSpecificSentenceRule,
        MissingSpecificWordRule,
        MissingWordPercentRule,
        MissingWordRule,
        TooManySentenceOccurencePercentRule,
        TooManySentenceOccurenceRule,
        TooManyWordOccurencePercentRule,
        TooManyWordOccurenceRule,
        UnexpectedSentencePercentRule,
        UnexpectedSentenceRule,
        UnexpectedWordPercentRule,
        UnexpectedWordRule,
    )
    from extract_bench.evaluation.metrics.parse.rules_chart import (
        ChartDataArrayDataRule,
        ChartDataArrayLabelsRule,
        ChartDataPointRule,
        RotateCheckRule,
    )
    from extract_bench.evaluation.metrics.parse.rules_form import (
        FormFieldRule,
    )
    from extract_bench.evaluation.metrics.parse.rules_formatting import (
        _FORMATTING_TEST_TYPES,
        CodeBlockRule,
        FormattingRule,
        LatexRule,
        MarkColorRule,
        PageSectionRule,
        TitleHierarchyPercentRule,
        TitleLevelRule,
    )
    from extract_bench.evaluation.metrics.parse.rules_table import (
        TableAdjacentDownRule,
        TableAdjacentLeftRule,
        TableAdjacentRightRule,
        TableAdjacentUpRule,
        TableColspanRule,
        TableHeaderChainRule,
        TableLeftHeaderRule,
        TableNoAboveRule,
        TableNoBelowRule,
        TableNoLeftRule,
        TableNoRightRule,
        TableRowspanRule,
        TableRule,
        TableSameColumnRule,
        TableSameRowRule,
        TablesNumColsRule,
        TablesNumRowsRule,
        TablesValuesRule,
        TableTopHeaderRule,
    )
    from extract_bench.evaluation.metrics.parse.rules_text import (
        BaselineRule,
        TextOrderRule,
        TextPresenceRule,
    )

    typed_rule: Any = coerce_parse_rule(rule_data)
    rule_type = get_rule_type(typed_rule)
    if not rule_type:
        raise ValueError("Rule must have a 'type' field")

    if rule_type in {ParseRuleType.PRESENT.value, ParseRuleType.ABSENT.value}:
        return TextPresenceRule(typed_rule)
    elif rule_type == ParseRuleType.UNEXPECTED_SENTENCE.value:
        return UnexpectedSentenceRule(typed_rule)
    elif rule_type == ParseRuleType.UNEXPECTED_SENTENCE_PERCENT.value:
        return UnexpectedSentencePercentRule(typed_rule)
    elif rule_type == ParseRuleType.TOO_MANY_SENTENCE_OCCURENCE.value:
        return TooManySentenceOccurenceRule(typed_rule)
    elif rule_type == ParseRuleType.TOO_MANY_SENTENCE_OCCURENCE_PERCENT.value:
        return TooManySentenceOccurencePercentRule(typed_rule)
    elif rule_type == ParseRuleType.MISSING_SENTENCE.value:
        return MissingSentenceRule(typed_rule)
    elif rule_type == ParseRuleType.MISSING_SENTENCE_PERCENT.value:
        return MissingSentencePercentRule(typed_rule)
    elif rule_type == ParseRuleType.MISSING_SPECIFIC_SENTENCE.value:
        return MissingSpecificSentenceRule(typed_rule)
    elif rule_type == ParseRuleType.UNEXPECTED_WORD.value:
        return UnexpectedWordRule(typed_rule)
    elif rule_type == ParseRuleType.UNEXPECTED_WORD_PERCENT.value:
        return UnexpectedWordPercentRule(typed_rule)
    elif rule_type == ParseRuleType.TOO_MANY_WORD_OCCURENCE.value:
        return TooManyWordOccurenceRule(typed_rule)
    elif rule_type == ParseRuleType.TOO_MANY_WORD_OCCURENCE_PERCENT.value:
        return TooManyWordOccurencePercentRule(typed_rule)
    elif rule_type == ParseRuleType.MISSING_WORD.value:
        return MissingWordRule(typed_rule)
    elif rule_type == ParseRuleType.MISSING_WORD_PERCENT.value:
        return MissingWordPercentRule(typed_rule)
    elif rule_type == ParseRuleType.MISSING_SPECIFIC_WORD.value:
        return MissingSpecificWordRule(typed_rule)
    elif rule_type == ParseRuleType.EXTRA_CONTENT.value:
        return ExtraContentRule(typed_rule)
    elif rule_type == ParseRuleType.BASELINE.value:
        return BaselineRule(typed_rule)
    elif rule_type == ParseRuleType.ORDER.value:
        return TextOrderRule(typed_rule)
    elif rule_type == ParseRuleType.TABLE.value:
        return TableRule(typed_rule)
    elif rule_type == ParseRuleType.TABLES_VALUES.value:
        return TablesValuesRule(typed_rule)
    elif rule_type == ParseRuleType.TABLES_NUM_ROWS.value:
        return TablesNumRowsRule(typed_rule)
    elif rule_type == ParseRuleType.TABLES_NUM_COLS.value:
        return TablesNumColsRule(typed_rule)
    # Table hierarchy rules
    elif rule_type == ParseRuleType.TABLE_COLSPAN.value:
        return TableColspanRule(typed_rule)
    elif rule_type == ParseRuleType.TABLE_ROWSPAN.value:
        return TableRowspanRule(typed_rule)
    elif rule_type == ParseRuleType.TABLE_SAME_ROW.value:
        return TableSameRowRule(typed_rule)
    elif rule_type == ParseRuleType.TABLE_SAME_COLUMN.value:
        return TableSameColumnRule(typed_rule)
    elif rule_type == ParseRuleType.TABLE_HEADER_CHAIN.value:
        return TableHeaderChainRule(typed_rule)
    # Table adjacency and header rules
    elif rule_type == ParseRuleType.TABLE_ADJACENT_UP.value:
        return TableAdjacentUpRule(typed_rule)
    elif rule_type == ParseRuleType.TABLE_ADJACENT_DOWN.value:
        return TableAdjacentDownRule(typed_rule)
    elif rule_type == ParseRuleType.TABLE_ADJACENT_LEFT.value:
        return TableAdjacentLeftRule(typed_rule)
    elif rule_type == ParseRuleType.TABLE_ADJACENT_RIGHT.value:
        return TableAdjacentRightRule(typed_rule)
    elif rule_type == ParseRuleType.TABLE_TOP_HEADER.value:
        return TableTopHeaderRule(typed_rule)
    elif rule_type == ParseRuleType.TABLE_LEFT_HEADER.value:
        return TableLeftHeaderRule(typed_rule)
    # Table border rules (negative tests)
    elif rule_type == ParseRuleType.TABLE_NO_LEFT.value:
        return TableNoLeftRule(typed_rule)
    elif rule_type == ParseRuleType.TABLE_NO_RIGHT.value:
        return TableNoRightRule(typed_rule)
    elif rule_type == ParseRuleType.TABLE_NO_ABOVE.value:
        return TableNoAboveRule(typed_rule)
    elif rule_type == ParseRuleType.TABLE_NO_BELOW.value:
        return TableNoBelowRule(typed_rule)
    # Chart rules
    elif rule_type == ParseRuleType.CHART_DATA_POINT.value:
        return ChartDataPointRule(typed_rule)
    elif rule_type == ParseRuleType.CHART_DATA_ARRAY_LABELS.value:
        return ChartDataArrayLabelsRule(typed_rule)
    elif rule_type == ParseRuleType.CHART_DATA_ARRAY_DATA.value:
        return ChartDataArrayDataRule(typed_rule)
    # Form rules
    elif rule_type == ParseRuleType.FORM_FIELD.value:
        return FormFieldRule(typed_rule)
    # Formatting rules (bold, italic, underline, strikeout, mark, sup, sub)
    elif rule_type in _FORMATTING_TEST_TYPES:
        if rule_type == ParseRuleType.MARK_COLOR.value:
            return MarkColorRule(typed_rule)
        return FormattingRule(typed_rule)
    elif rule_type == ParseRuleType.IS_LATEX.value:
        return LatexRule(typed_rule)
    elif rule_type == ParseRuleType.IS_CODE_BLOCK.value:
        return CodeBlockRule(typed_rule)
    # Title / heading level rule
    elif rule_type == ParseRuleType.IS_TITLE.value:
        return TitleLevelRule(typed_rule)
    elif rule_type == ParseRuleType.TITLE_HIERARCHY_PERCENT.value:
        return TitleHierarchyPercentRule(typed_rule)
    # Page header / footer rules
    elif rule_type in {ParseRuleType.IS_HEADER.value, ParseRuleType.IS_FOOTER.value}:
        return PageSectionRule(typed_rule)
    # Digit bag rule
    elif rule_type == ParseRuleType.BAG_OF_DIGIT_PERCENT.value:
        return BagOfDigitPercentRule(typed_rule)
    # Rotation check
    elif rule_type == ParseRuleType.ROTATE_CHECK.value:
        return RotateCheckRule(typed_rule)
    else:
        raise ValueError(f"Unknown test type: {rule_type}")
