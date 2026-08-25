"""Test type definitions for parse evaluation."""

from enum import StrEnum


class ParseRuleType(StrEnum):
    """Test types for parse evaluation."""

    BASELINE = "baseline"
    PRESENT = "present"
    UNEXPECTED_SENTENCE = "unexpected_sentence"
    UNEXPECTED_SENTENCE_PERCENT = "unexpected_sentence_percent"
    TOO_MANY_SENTENCE_OCCURENCE = "too_many_sentence_occurence"
    TOO_MANY_SENTENCE_OCCURENCE_PERCENT = "too_many_sentence_occurence_percent"
    MISSING_SENTENCE = "missing_sentence"
    MISSING_SENTENCE_PERCENT = "missing_sentence_percent"
    MISSING_SPECIFIC_SENTENCE = "missing_specific_sentence"
    UNEXPECTED_WORD = "unexpected_word"
    UNEXPECTED_WORD_PERCENT = "unexpected_word_percent"
    TOO_MANY_WORD_OCCURENCE = "too_many_word_occurence"
    TOO_MANY_WORD_OCCURENCE_PERCENT = "too_many_word_occurence_percent"
    MISSING_WORD = "missing_word"
    MISSING_WORD_PERCENT = "missing_word_percent"
    MISSING_SPECIFIC_WORD = "missing_specific_word"
    # Keep for backward compatibility with earlier combined behavior
    EXTRA_CONTENT = "extra_content"
    ABSENT = "absent"
    ORDER = "order"
    TABLE = "table"
    TABLES_VALUES = "tables_values"
    TABLES_NUM_ROWS = "tables_num_rows"
    TABLES_NUM_COLS = "tables_num_cols"
    MATH = "math"
    # Table hierarchy test types
    TABLE_COLSPAN = "table_colspan"
    TABLE_ROWSPAN = "table_rowspan"
    TABLE_SAME_ROW = "table_same_row"
    TABLE_SAME_COLUMN = "table_same_column"
    TABLE_HEADER_CHAIN = "table_header_chain"  # Keep for backward compatibility
    # New table adjacency test types
    TABLE_ADJACENT_UP = "table_adjacent_up"
    TABLE_ADJACENT_DOWN = "table_adjacent_down"
    TABLE_ADJACENT_LEFT = "table_adjacent_left"
    TABLE_ADJACENT_RIGHT = "table_adjacent_right"
    TABLE_TOP_HEADER = "table_top_header"
    TABLE_LEFT_HEADER = "table_left_header"
    # Table border tests - verify absence of cells beyond boundaries
    TABLE_NO_LEFT = "table_no_left"
    TABLE_NO_RIGHT = "table_no_right"
    TABLE_NO_ABOVE = "table_no_above"
    TABLE_NO_BELOW = "table_no_below"
    # Formatting test types - check if text has specific formatting
    IS_UNDERLINE = "is_underline"
    IS_NOT_UNDERLINE = "is_not_underline"
    IS_BOLD = "is_bold"
    IS_NOT_BOLD = "is_not_bold"
    IS_STRIKEOUT = "is_strikeout"
    IS_NOT_STRIKEOUT = "is_not_strikeout"
    IS_ITALIC = "is_italic"
    IS_NOT_ITALIC = "is_not_italic"
    IS_MARK = "is_mark"
    IS_NOT_MARK = "is_not_mark"
    MARK_COLOR = "mark_color"
    IS_SUP = "is_sup"
    IS_NOT_SUP = "is_not_sup"
    IS_SUB = "is_sub"
    IS_NOT_SUB = "is_not_sub"
    IS_LATEX = "is_latex"
    IS_CODE_BLOCK = "is_code_block"
    # Title / heading level test
    IS_TITLE = "is_title"
    TITLE_HIERARCHY_PERCENT = "title_hierarchy_percent"
    # Page header / footer tests
    IS_HEADER = "is_header"
    IS_FOOTER = "is_footer"
    # Chart test types
    CHART_DATA_POINT = "chart_data_point"
    CHART_DATA_ARRAY_LABELS = "chart_data_array_labels"
    CHART_DATA_ARRAY_DATA = "chart_data_array_data"
    # Layout detection
    LAYOUT = "layout"
    # Digit bag rule
    BAG_OF_DIGIT_PERCENT = "bag_of_digit_percent"
    # Rotation check
    ROTATE_CHECK = "rotate_check"
    # Form field (key/value, checkbox, signature) extraction
    FORM_FIELD = "form_field"
