"""Test rule implementations for parse evaluation.

This module re-exports all rule classes and helpers from the split submodules
for backward compatibility. New code should import directly from the
specific submodule (rules_base, rules_text, rules_bag, rules_formatting,
rules_table, rules_chart, rules_form).
"""

# Base class, helpers, and factory
# Bag rules
from extract_bench.evaluation.metrics.parse.rules_bag import (  # noqa: F401
    BagOfDigitPercentRule,
    ExtraContentRule,
    MissingSentencePercentRule,
    MissingSentenceRule,
    MissingSpecificSentenceRule,
    MissingSpecificWordRule,
    MissingWordPercentRule,
    MissingWordRule,
    SentenceBagRule,
    TooManySentenceOccurencePercentRule,
    TooManySentenceOccurenceRule,
    TooManyWordOccurencePercentRule,
    TooManyWordOccurenceRule,
    UnexpectedSentencePercentRule,
    UnexpectedSentenceRule,
    UnexpectedWordPercentRule,
    UnexpectedWordRule,
    WordBagRule,
)
from extract_bench.evaluation.metrics.parse.rules_base import (  # noqa: F401
    CELL_FUZZY_MATCH_THRESHOLD,
    AdjacentTableRuleData,
    NoBorderTableRuleData,
    ParseTestRule,
    SentenceBagRuleData,
    WordBagRuleData,
    _augment_with_table_cell_text,
    _dates_match,
    _detect_csv_skip_rows,
    _detect_header_scale,
    _extract_table_cell_texts,
    _normalize_date_str,
    _strip_and_replace_latex,
    _strip_fenced_code_blocks,
    _strip_html_tables_and_content,
    _unescape_html_entities,
    create_test_rule,
)

# Chart rules
from extract_bench.evaluation.metrics.parse.rules_chart import (  # noqa: F401
    ChartDataArrayDataRule,
    ChartDataArrayLabelsRule,
    ChartDataPointRule,
    RotateCheckRule,
    extract_numeric_parts,
    normalize_number_string,
    numbers_match,
    numeric_similarity,
)

# Form rules
from extract_bench.evaluation.metrics.parse.rules_form import (  # noqa: F401
    FormFieldRule,
)

# Formatting rules
from extract_bench.evaluation.metrics.parse.rules_formatting import (  # noqa: F401
    _FORMATTING_TEST_TYPES,
    CodeBlockRule,
    FormattingRule,
    LatexRule,
    MarkColorRule,
    PageSectionRule,
    TitleHierarchyPercentRule,
    TitleLevelRule,
)

# Table rules
from extract_bench.evaluation.metrics.parse.rules_table import (  # noqa: F401
    TableAdjacentDownRule,
    TableAdjacentLeftRule,
    TableAdjacentRightRule,
    TableAdjacentRule,
    TableAdjacentUpRule,
    TableColspanRule,
    TableHeaderChainRule,
    TableLeftHeaderRule,
    TableNoAboveRule,
    TableNoBelowRule,
    TableNoBorderRule,
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

# Text rules
from extract_bench.evaluation.metrics.parse.rules_text import (  # noqa: F401
    BaselineRule,
    TextOrderRule,
    TextPresenceRule,
)
