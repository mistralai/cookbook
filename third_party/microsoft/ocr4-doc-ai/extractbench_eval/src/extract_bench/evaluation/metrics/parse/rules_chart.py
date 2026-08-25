"""Chart data validation and rotation check test rules."""

import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
from rapidfuzz import fuzz

from extract_bench.evaluation.metrics.parse.rules_base import (
    CELL_FUZZY_MATCH_THRESHOLD,
    ParseTestRule,
    _dates_match,
    _detect_csv_skip_rows,
)
from extract_bench.evaluation.metrics.parse.table_parsing import (
    TableData,
    parse_html_tables,
    parse_markdown_tables,
)
from extract_bench.evaluation.metrics.parse.test_types import ParseRuleType
from extract_bench.evaluation.metrics.parse.utils import normalize_text
from extract_bench.schemas.parse_output import ParseOutput
from extract_bench.test_cases.parse_rule_schemas import (
    ParseChartDataArrayDataRule,
    ParseChartDataArrayLabelsRule,
    ParseChartDataPointRule,
    ParseRotateCheckRule,
)

# =============================================================================
# Number Normalization Utilities for Chart Tests
# =============================================================================


def normalize_number_string(s: str) -> float | None:
    """
    Convert various number formats to a normalized float.

    Handles:
    - Currency symbols: $, €, £, ¥
    - Thousands separators: commas
    - Suffixes: k, K, m, M, million, billion, b, B, etc.
    - Percentage signs

    Examples:
        "39.2m" → 39.2
        "$39.2M" → 39.2
        "39,200,000" → 39200000.0
        "1.5k" → 1.5
        "45%" → 45.0

    Returns None if string cannot be parsed as a number.
    """
    if not s:
        return None

    # Remove whitespace and normalize
    s = s.strip()

    # Remove currency symbols
    s = re.sub(r"^[$€£¥]\s*", "", s)
    s = re.sub(r"\s*[$€£¥]$", "", s)

    # Remove approximate prefixes
    s = re.sub(r"^[~≈]\s*", "", s)

    # Remove percentage sign (but remember the value)
    s = s.rstrip("%")

    # Remove thousands separators (commas)
    s = s.replace(",", "")

    # Remove space-as-thousands-separator (e.g., "6 888" → "6888")
    s = s.replace(" ", "")

    # Handle suffixes — apply actual multipliers so values on different
    # scales (e.g. "485k" vs "485567") can be compared numerically.
    multiplier = 1.0
    suffix_patterns = [
        (r"(?i)\s*(trillion|trill|trn)$", 1e12),
        (r"(?i)\s*(billion|bill|bln)$", 1e9),
        (r"(?i)\s*(million|mill|mln)$", 1e6),
        (r"(?i)\s*t$", 1e12),
        (r"(?i)\s*g$", 1e9),  # G = giga = billion
        (r"(?i)\s*b$", 1e9),
        (r"(?i)\s*m$", 1e6),
        (r"(?i)\s*k$", 1e3),
    ]

    for pattern, mult in suffix_patterns:
        if re.search(pattern, s):
            s = re.sub(pattern, "", s)
            multiplier = mult
            break

    # Try to parse as float
    try:
        return float(s) * multiplier
    except ValueError:
        return None


def _normalize_number_candidates(s: str) -> list[float]:
    """Return all plausible numeric interpretations of *s*.

    When *s* contains commas the comma is ambiguous: it could be a thousands
    separator ("3,125" → 3125) or a French/European decimal separator
    ("3,125" → 3.125).  This helper returns both interpretations so callers
    can try each.
    """
    candidates: list[float] = []
    val = normalize_number_string(s)
    if val is not None:
        candidates.append(val)
    # If s contains a comma, also try the decimal-separator interpretation.
    if "," in s:
        val2 = normalize_number_string(s.replace(",", "."))
        if val2 is not None and val2 not in candidates:
            candidates.append(val2)
    return candidates


def numbers_match(val1: str, val2: str, tolerance: float = 0.01) -> bool:
    """
    Check if two number strings represent the same value.

    Args:
        val1: First value string
        val2: Second value string
        tolerance: Relative tolerance for comparison (default 1%)

    Returns:
        True if values match within tolerance, False otherwise
    """
    # Get all plausible numeric interpretations for each value.
    # E.g. "2,08" produces [208.0, 2.08] (thousands-sep vs decimal-sep).
    candidates1 = _normalize_number_candidates(val1)
    candidates2 = _normalize_number_candidates(val2)

    if not candidates1 or not candidates2:
        return False

    # Try every pair of interpretations — match if any combo agrees.
    for num1 in candidates1:
        for num2 in candidates2:
            if num1 == 0 and num2 == 0:
                return True
            if num1 == 0 or num2 == 0:
                if abs(num1 - num2) < tolerance:
                    return True
                continue
            relative_diff = abs(num1 - num2) / max(abs(num1), abs(num2))
            if relative_diff <= tolerance:
                return True

    return False


def numeric_similarity(val1: str, val2: str) -> float | None:
    """
    Calculate similarity score between two numeric values using relative error.

    Uses: score = max(0, 1 - |expected - actual| / |expected|)

    This is equivalent to 1 - NRMSE for a single observation, providing
    a statistically principled similarity measure.

    References:
        - Chai & Draxler (2014). RMSE or MAE? Geosci. Model Dev.
        - Hyndman & Koehler (2006). Forecast accuracy. Int. J. Forecasting.

    Returns:
        Float between 0.0 and 1.0, or None if not both numbers.
        - 0% error → 1.0
        - 50% error → 0.5
        - 100%+ error → 0.0
    """
    # Same comma-ambiguity handling as numbers_match: try all
    # interpretations and return the best (highest) similarity score.
    candidates1 = _normalize_number_candidates(val1)  # expected
    candidates2 = _normalize_number_candidates(val2)  # actual

    if not candidates1 or not candidates2:
        return None

    best: float | None = None
    for num1 in candidates1:
        for num2 in candidates2:
            if num1 == 0 and num2 == 0:
                return 1.0
            if num1 == 0:
                score = 1.0 if abs(num2) < 0.001 else 0.0
            else:
                relative_error = abs(num1 - num2) / abs(num1)
                score = max(0.0, 1.0 - relative_error)
            if best is None or score > best:
                best = score

    return best


def extract_numeric_parts(value: str) -> list[str]:
    """Extract numeric parts from a composite value string.

    Examples:
        "25 (13.0%)" -> ["25", "13.0%"]
        "25, 13.0%"  -> ["25", "13.0%"]
        "100/50"     -> ["100", "50"]
        "25"         -> ["25"]
    """
    pattern = r"[-+]?\d(?:,\d{3}|\d)*\.?\d*%?"
    return re.findall(pattern, value)


class ChartDataPointRule(ParseTestRule):
    """
    Test rule for chart-to-table conversions.

    Verifies that a value is associated with given labels in a table,
    regardless of whether the table is row-oriented or column-oriented.

    This is useful for testing chart conversions where the same data
    can be represented in multiple valid table orientations.
    """

    def __init__(self, rule_data: ParseChartDataPointRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseChartDataPointRule, self._rule_data)

        if self.type != ParseRuleType.CHART_DATA_POINT.value:
            raise ValueError(f"Invalid type for ChartDataPointRule: {self.type}")

        self.value = normalize_text(str(rule_data.value))
        self.labels = [normalize_text(re.sub(r"<br\s*/?>", " ", str(label))) for label in rule_data.labels]
        self.normalize_numbers = rule_data.normalize_numbers
        self.relative_tolerance = rule_data.relative_tolerance

        if not self.value:
            raise ValueError("value field cannot be empty")
        if not self.labels:
            raise ValueError("labels field must contain at least one label")

    @staticmethod
    def _strip_for_label_compare(text: str) -> str:
        """Strip whitespace and special characters for label comparison."""
        return re.sub(r"[^a-z0-9]", "", text)

    def _find_value_in_table(self, table_array, value: str) -> list[tuple[int, int]]:  # type: ignore[no-untyped-def]
        """Find all cells matching the value in a table."""
        matches = []
        rows, cols = table_array.shape

        for row_idx in range(rows):
            for col_idx in range(cols):
                cell_text = normalize_text(str(table_array[row_idx, col_idx]))

                # Try exact fuzzy match first
                threshold = max(0.5, 1.0 - (self.max_diffs / max(len(value), 1)))
                similarity = fuzz.ratio(value, cell_text) / 100.0

                if similarity >= threshold:
                    matches.append((row_idx, col_idx))
                elif self.normalize_numbers and numbers_match(value, cell_text, self.relative_tolerance):
                    matches.append((row_idx, col_idx))

        # Fallback: try composite value decomposition for values like "25 (13.0%)"
        # where the number and percentage are in adjacent cells.
        if not matches and self.normalize_numbers:
            matches = self._find_composite_value_in_table(table_array, value)

        return matches

    def _find_composite_value_in_table(self, table_array, value: str) -> list[tuple[int, int]]:  # type: ignore[no-untyped-def]
        """Match a composite value against adjacent cells in a table.

        Handles values like "25 (13.0%)" where "25" is in one cell and
        "13.0%" is in an adjacent cell to the right.
        """
        parts = extract_numeric_parts(value)
        if len(parts) < 2:
            return []

        rows, cols = table_array.shape
        matches = []

        for row_idx in range(rows):
            for col_idx in range(cols):
                cell_text = normalize_text(str(table_array[row_idx, col_idx]))

                # Check if this cell matches the first numeric part
                if not numbers_match(parts[0], cell_text, self.relative_tolerance):
                    continue

                # Check if adjacent cells to the right match remaining parts
                all_parts_found = True
                for part_offset, part in enumerate(parts[1:], start=1):
                    adj_col = col_idx + part_offset
                    if adj_col >= cols:
                        all_parts_found = False
                        break
                    adj_text = normalize_text(str(table_array[row_idx, adj_col]))
                    if not numbers_match(part, adj_text, self.relative_tolerance):
                        all_parts_found = False
                        break

                if all_parts_found:
                    matches.append((row_idx, col_idx))

        return matches

    def _check_label_association(  # type: ignore[no-untyped-def]
        self,
        table_array,
        value_row: int,
        value_col: int,
        label: str,
        table_data: TableData | None = None,
    ) -> bool:
        """
        Check if a label is associated with a value cell.

        A label is associated if it appears in the same row OR same column,
        including colspan/rowspan headers tracked by col_headers/row_headers.
        """
        rows, cols = table_array.shape
        threshold = max(0.5, 1.0 - (self.max_diffs / max(len(label), 1)))
        stripped_label = self._strip_for_label_compare(label)

        def _label_matches(cell_text: str) -> bool:
            if fuzz.partial_ratio(label, cell_text) / 100.0 >= threshold:
                return True
            stripped_cell = self._strip_for_label_compare(cell_text)
            if stripped_label and stripped_cell and stripped_label in stripped_cell:
                return True
            return False

        # Check same row
        for col_idx in range(cols):
            if col_idx == value_col:
                continue
            cell_text = normalize_text(str(table_array[value_row, col_idx]))
            if _label_matches(cell_text):
                return True

        # Check same column
        for row_idx in range(rows):
            if row_idx == value_row:
                continue
            cell_text = normalize_text(str(table_array[row_idx, value_col]))
            if _label_matches(cell_text):
                return True

        # Check col_headers for the value's column (handles colspan headers)
        if table_data and table_data.col_headers:
            for header_entry in table_data.col_headers.get(value_col, []):
                header_text = normalize_text(str(header_entry[1]))
                if _label_matches(header_text):
                    return True

        # Check row_headers for the value's row (handles rowspan headers)
        if table_data and table_data.row_headers:
            for header_entry in table_data.row_headers.get(value_row, []):
                header_text = normalize_text(str(header_entry[1]))
                if _label_matches(header_text):
                    return True

        return False

    def _extract_formatted_labels(self, context: str) -> set[str]:
        """Extract bold text and headings from markdown/HTML context."""
        formatted_labels = set()

        # Extract markdown headings: # Title, ## Title, etc.
        heading_pattern = r"^#{1,6}\s+(.+?)$"
        for match in re.finditer(heading_pattern, context, re.MULTILINE):
            formatted_labels.add(normalize_text(match.group(1)))

        # Extract bold text: **Bold Text**
        bold_pattern = r"\*\*(.+?)\*\*"
        for match in re.finditer(bold_pattern, context):
            formatted_labels.add(normalize_text(match.group(1)))

        # Extract HTML headings: <h1>Title</h1>, etc.
        html_heading_pattern = r"<h[1-6][^>]*>(.+?)</h[1-6]>"
        for match in re.finditer(html_heading_pattern, context, re.IGNORECASE):
            formatted_labels.add(normalize_text(match.group(1)))

        # Extract HTML bold: <strong>Text</strong>, <b>Text</b>
        html_bold_pattern = r"<(?:strong|b)[^>]*>(.+?)</(?:strong|b)>"
        for match in re.finditer(html_bold_pattern, context, re.IGNORECASE):
            formatted_labels.add(normalize_text(match.group(1)))

        return formatted_labels

    def _label_exists_in_table(  # type: ignore[no-untyped-def]
        self, table_array, label: str, table_data: TableData | None = None
    ) -> bool:
        """Check if label matches ANY cell in the table (not just same row/col).

        This prevents labels like "Retail Ecommerce Sales" from being treated
        as title labels when they are actually column headers in the table.
        """
        threshold = max(0.5, 1.0 - (self.max_diffs / max(len(label), 1)))
        stripped_label = self._strip_for_label_compare(label)
        rows, cols = table_array.shape

        def _label_matches(cell_text: str) -> bool:
            if fuzz.partial_ratio(label, cell_text) / 100.0 >= threshold:
                return True
            stripped_cell = self._strip_for_label_compare(cell_text)
            if stripped_label and stripped_cell and stripped_label in stripped_cell:
                return True
            return False

        for r in range(rows):
            for c in range(cols):
                cell_text = normalize_text(str(table_array[r, c]))
                if _label_matches(cell_text):
                    return True

        # Also check col_headers and row_headers
        if table_data:
            for headers in (table_data.col_headers, table_data.row_headers):
                if headers:
                    for entries in headers.values():
                        for entry in entries:
                            header_text = normalize_text(str(entry[1]))
                            if _label_matches(header_text):
                                return True

        return False

    def _is_label_in_formatted_context(self, context: str, label: str) -> bool:
        """Check if label appears as formatted text (bold/heading) in context.

        Uses full-string ratio (not partial/substring) to avoid matching a label
        that is merely one term inside a longer heading.  For example,
        "Summary innovation index" should match "Summary innovation index (Individual Countries)"
        but "Workforce" must NOT match "Evolution of the workforce, revenue and productivity".
        """
        formatted_labels = self._extract_formatted_labels(context)
        normalized_label = normalize_text(label)
        stripped_label = self._strip_for_label_compare(normalized_label)

        # Use a fixed threshold with full-string ratio so that the label must
        # cover most of the formatted text (or vice-versa).
        threshold = 0.60

        for formatted_label in formatted_labels:
            similarity = fuzz.ratio(normalized_label, formatted_label) / 100.0
            if similarity >= threshold:
                return True
            # Fallback: compare with whitespace/special chars stripped
            stripped_formatted = self._strip_for_label_compare(formatted_label)
            if stripped_label and stripped_formatted:
                similarity = fuzz.ratio(stripped_label, stripped_formatted) / 100.0
                if similarity >= threshold:
                    return True

        return False

    def _is_label_in_heading_or_caption(self, context: str, label: str) -> bool:
        """Check if label appears in a heading or <caption> element in context.

        Headings and captions are strong table-identity signals (e.g. "## LDC/LLDCs"
        or "<caption>Solar PV (modules) ...</caption>").  Uses partial_ratio since
        headings/captions are typically longer than the label.
        """
        normalized_label = normalize_text(label)
        stripped_label = self._strip_for_label_compare(normalized_label)
        threshold = max(0.5, 1.0 - (self.max_diffs / max(len(normalized_label), 1)))

        def _matches(text: str) -> bool:
            text = normalize_text(text)
            if fuzz.partial_ratio(normalized_label, text) / 100.0 >= threshold:
                return True
            stripped = self._strip_for_label_compare(text)
            if stripped_label and stripped and stripped_label in stripped:
                return True
            return False

        # Check <caption> elements
        for match in re.finditer(r"<caption[^>]*>(.+?)</caption>", context, re.IGNORECASE):
            if _matches(match.group(1)):
                return True

        # Check markdown headings (# Title, ## Title, etc.)
        for match in re.finditer(r"^#{1,6}\s+(.+?)$", context, re.MULTILINE):
            if _matches(match.group(1)):
                return True

        return False

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        """Check if value is associated with all labels in any table."""
        # Parse all tables
        tables_to_check = []

        # Parse markdown tables
        md_tables = parse_markdown_tables(content)
        tables_to_check.extend(md_tables)

        # Parse HTML tables
        html_tables = parse_html_tables(content)
        tables_to_check.extend(html_tables)

        if not tables_to_check:
            return False, "No tables found in content", 0.0

        all_failed_reasons = []

        for table_data in tables_to_check:
            table_array = table_data.data
            # Only use context BEFORE the table (chart titles/headings).
            # context_after is excluded to avoid matching labels that
            context = table_data.context_before.strip()

            # Find all cells matching the value
            value_matches = self._find_value_in_table(table_array, self.value)

            if not value_matches:
                continue  # Try next table

            # For each matching cell, try validation phases
            for value_row, value_col in value_matches:
                # PHASE 1: Try strict matching - all labels in table cells
                missing_labels_strict = []
                for label in self.labels:
                    if not self._check_label_association(table_array, value_row, value_col, label, table_data):
                        missing_labels_strict.append(label)

                if not missing_labels_strict:
                    # Success - all labels found in table
                    return (
                        True,
                        f"Value '{self.value}' found with all labels at ({value_row}, {value_col})",
                        1.0,
                    )

                # PHASE 2: Try context-aware matching if context available
                if context:
                    # Classify labels based on WHERE they are found:
                    # 1. First check if label is in table (associated with value)
                    # 2. If not in table, check if it's in formatted context
                    # This ensures labels that appear in BOTH table and context
                    # are correctly classified as data labels (found in table)

                    data_labels = []  # Labels found in table cells
                    title_labels = []  # Labels found in formatted context only
                    missing_labels = []  # Labels not found anywhere

                    for label in self.labels:
                        # Check if label is associated with value in table
                        if self._check_label_association(table_array, value_row, value_col, label, table_data):
                            data_labels.append(label)
                        # Only allow formatted-context matching for labels that
                        # do NOT exist anywhere in the current table.
                        elif not self._label_exists_in_table(
                            table_array, label, table_data
                        ) and self._is_label_in_formatted_context(context, label):
                            title_labels.append(label)
                        # Headings and captions are strong table-identity signals;
                        # allow them even when the label partially matches a table
                        # cell (e.g. "LDC/LLDCs" heading vs "Average OSI for
                        # LDC/LLDCs" column header).
                        elif self._is_label_in_heading_or_caption(context, label):
                            title_labels.append(label)
                        else:
                            missing_labels.append(label)

                    # Success if all labels are found (either in table or in context)
                    if not missing_labels:
                        # Success - all labels found
                        if title_labels:
                            return (
                                True,
                                (
                                    f"Value '{self.value}' found with data labels {data_labels} "
                                    f"in table and title labels "
                                    f"{title_labels} in context "
                                    f"at ({value_row}, {value_col})"
                                ),
                                1.0,
                            )
                        else:
                            # All labels in table (no title labels needed)
                            return (
                                True,
                                f"Value '{self.value}' found with all labels at ({value_row}, {value_col})",
                                1.0,
                            )

                    # Track failure reason
                    all_failed_reasons.append(
                        f"Value at ({value_row}, {value_col}) missing labels: {missing_labels} "
                        f"(data labels {data_labels} in table, "
                        f"title labels {title_labels} in context)"
                    )
                else:
                    # No context available, use strict failure reason
                    all_failed_reasons.append(
                        f"Value at ({value_row}, {value_col}) missing labels: {missing_labels_strict}"
                    )

        if not all_failed_reasons:
            return False, f"Value '{self.value}' not found in any table", 0.0

        return (
            False,
            f"Value found but labels not associated: {'; '.join(all_failed_reasons[:3])}",
            0.0,
        )


class ChartDataArrayLabelsRule(ParseTestRule):
    """
    Test rule for validating chart data array labels (headers) in table conversions.

    Computes a similarity score for each label, so partial matches get proportionally
    lower scores. For example, "Length (in months)" matching against
    "Length of the current and previous bull market (in months)" will get a partial
    score rather than full credit.
    """

    def __init__(self, rule_data: ParseChartDataArrayLabelsRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseChartDataArrayLabelsRule, self._rule_data)

        if self.type != ParseRuleType.CHART_DATA_ARRAY_LABELS.value:
            raise ValueError(f"Invalid type for ChartDataArrayLabelsRule: {self.type}")

        # Load data from CSV if available (auto-detected by loader)
        csv_path = rule_data.csv_path
        if csv_path and Path(csv_path).exists():
            skip = _detect_csv_skip_rows(csv_path)
            df = pd.read_csv(csv_path, skiprows=skip)
            # Convert to list of lists: [headers, row1, row2, ...]
            self.data = [df.columns.tolist()] + df.values.tolist()
        else:
            self.data = rule_data.data

        self.x_axis_shuffle = rule_data.x_axis_shuffle
        self.transposed = rule_data.transposed

        if not self.data or len(self.data) < 1:
            raise ValueError("data field must contain at least one row (headers)")

        # Headers are the first row
        self.headers = self.data[0]

    def _normalize_cell(self, value: Any) -> str:
        """Normalize a cell value for comparison."""
        return normalize_text(str(value))

    def _label_similarity(self, expected: Any, actual: str) -> float:
        """
        Compute similarity score between expected and actual label (0.0 to 1.0).

        Returns 1.0 for exact match, lower scores for partial matches.
        """
        expected_str = self._normalize_cell(expected)
        actual_str = normalize_text(actual)

        # Exact match
        if expected_str == actual_str:
            return 1.0

        # Date-aware match (e.g. "2008-01-01 00:00:00" vs "Q1-2008")
        if _dates_match(str(expected), actual):
            return 1.0

        # Numeric match (e.g. "415,000" vs "415000" or "415 000")
        num_score = numeric_similarity(str(expected), actual)
        if num_score is not None and num_score >= CELL_FUZZY_MATCH_THRESHOLD:
            return num_score

        # Use ratio for overall similarity (stricter than partial_ratio)
        ratio_score = fuzz.ratio(expected_str, actual_str) / 100.0

        # Also check partial_ratio for cases where one is substring of other
        partial_score = fuzz.partial_ratio(expected_str, actual_str) / 100.0

        # Weight: prefer ratio_score but give some credit for partial matches
        # If actual is much shorter than expected, penalize more
        length_ratio = min(len(actual_str), len(expected_str)) / max(len(actual_str), len(expected_str), 1)

        # Combined score: ratio is primary, partial helps when lengths differ
        score = ratio_score * 0.7 + partial_score * length_ratio * 0.3

        return score

    def _check_labels_ordered(  # type: ignore[no-untyped-def]
        self, table_array
    ) -> tuple[float, float, list[tuple[str, str, float]]]:
        """
        Check labels in order (no shuffle).

        Returns: (total_score, max_possible_score, [(expected, actual, score), ...])
        """
        _, cols = table_array.shape
        if cols != len(self.headers):
            return 0.0, float(len(self.headers)), [("", "", 0.0)]

        total_score = 0.0
        label_scores: list[tuple[str, str, float]] = []

        for col_idx, expected_label in enumerate(self.headers):
            actual_label = str(table_array[0, col_idx])
            score = self._label_similarity(expected_label, actual_label)
            total_score += score
            label_scores.append((str(expected_label), actual_label, score))

        return total_score, float(len(self.headers)), label_scores

    def _check_labels_shuffled(  # type: ignore[no-untyped-def]
        self, table_array
    ) -> tuple[float, float, list[tuple[str, str, float]]]:
        """
        Check labels with x-axis shuffle (columns can be reordered).

        Returns: (total_score, max_possible_score, [(expected, actual, score), ...])
        """
        _, cols = table_array.shape
        if cols != len(self.headers):
            return 0.0, float(len(self.headers)), [("", "", 0.0)]

        actual_labels = [str(table_array[0, col_idx]) for col_idx in range(cols)]
        total_score = 0.0
        label_scores: list[tuple[str, str, float]] = []
        used_actual: set[int] = set()

        for expected_label in self.headers:
            best_score = 0.0
            best_idx = -1
            best_actual = ""

            for act_idx, actual_label in enumerate(actual_labels):
                if act_idx in used_actual:
                    continue
                score = self._label_similarity(expected_label, actual_label)
                if score > best_score:
                    best_score = score
                    best_idx = act_idx
                    best_actual = actual_label

            if best_idx >= 0:
                used_actual.add(best_idx)
                total_score += best_score
                label_scores.append((str(expected_label), best_actual, best_score))
            else:
                label_scores.append((str(expected_label), "", 0.0))

        return total_score, float(len(self.headers)), label_scores

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        """
        Check if expected labels match any table in content.

        Returns a score-based result where partial matches get proportionally lower scores.
        """
        tables_to_check = []

        md_tables = parse_markdown_tables(content)
        tables_to_check.extend(md_tables)

        html_tables = parse_html_tables(content)
        tables_to_check.extend(html_tables)

        if not tables_to_check:
            return False, "No tables found in content", 0.0

        best_score = 0.0
        best_total = float(len(self.headers))
        best_label_scores: list[tuple[str, str, float]] = []

        for table_data in tables_to_check:
            # Try both orientations and keep the best score
            orientations = [table_data.data, table_data.data.T]
            for data in orientations:
                if self.x_axis_shuffle:
                    score, total, label_scores = self._check_labels_shuffled(data)
                else:
                    score, total, label_scores = self._check_labels_ordered(data)

                if score > best_score:
                    best_score = score
                    best_total = total
                    best_label_scores = label_scores

        # Format the result with individual label scores
        if best_total == 0:
            return False, "No labels to check", 0.0

        score_pct = (best_score / best_total) * 100 if best_total > 0 else 0
        score_normalized = score_pct / 100.0

        # Build details for labels that aren't perfect matches
        imperfect = [f"'{exp}' vs '{act}' ({sc:.0%})" for exp, act, sc in best_label_scores if sc < 1.0]

        if score_pct == 100:
            return True, f"Labels: {score_pct:.1f}% ({best_score:.2f}/{best_total:.0f})", 1.0

        if imperfect:
            return (
                False,
                (f"Labels: {score_pct:.1f}% ({best_score:.2f}/{best_total:.0f}). Partial: {'; '.join(imperfect[:3])}"),
                score_normalized,
            )
        return (
            False,
            f"Labels: {score_pct:.1f}% ({best_score:.2f}/{best_total:.0f})",
            score_normalized,
        )


class ChartDataArrayDataRule(ParseTestRule):
    """
    Test rule for validating chart data array values (excluding headers) in table conversions.

    Uses strict matching with number normalization. Supports row/column shuffling
    for charts where axis order doesn't matter.
    """

    def __init__(self, rule_data: ParseChartDataArrayDataRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseChartDataArrayDataRule, self._rule_data)

        if self.type != ParseRuleType.CHART_DATA_ARRAY_DATA.value:
            raise ValueError(f"Invalid type for ChartDataArrayDataRule: {self.type}")

        # Load data from CSV if available (auto-detected by loader)
        csv_path = rule_data.csv_path
        if csv_path and Path(csv_path).exists():
            skip = _detect_csv_skip_rows(csv_path)
            df = pd.read_csv(csv_path, skiprows=skip)
            # Convert to list of lists: [headers, row1, row2, ...]
            self.data = [df.columns.tolist()] + df.values.tolist()
        else:
            self.data = rule_data.data

        self.x_axis_shuffle = rule_data.x_axis_shuffle  # columns can reorder
        self.y_axis_shuffle = rule_data.y_axis_shuffle  # rows can reorder
        self.normalize_numbers = rule_data.normalize_numbers
        self.transposed = rule_data.transposed

        if not self.data or len(self.data) < 2:
            raise ValueError("data field must contain at least header row and one data row")

        # Data rows are everything after the header
        self.data_rows = self.data[1:]
        self.expected_cols = len(self.data[0]) if self.data else 0

    def _normalize_cell(self, value: Any) -> str:
        """Normalize a cell value for comparison."""
        return normalize_text(str(value))

    def _cells_match(self, expected: Any, actual: str) -> bool:
        """Check if two cell values match (with fuzzy/number matching)."""
        return self._cell_score(expected, actual) >= CELL_FUZZY_MATCH_THRESHOLD

    @staticmethod
    def _is_empty_or_nan(value: Any) -> bool:
        """Check if a value represents an empty/missing cell."""
        if value is None:
            return True
        s = str(value).strip()
        return s in ("", "nan", "NaN", "NAN", "none", "None", "-", "—", "n/a", "N/A")

    def _cell_score(self, expected: Any, actual: str) -> float:
        """
        Calculate similarity score between expected and actual cell values.

        Returns float between 0.0 and 1.0.
        """
        # Expected nan/empty matches any empty-like or zero actual value
        if self._is_empty_or_nan(expected):
            if self._is_empty_or_nan(actual):
                return 1.0
            # Also accept "0" / 0.0 as a representation of missing data
            num = normalize_number_string(actual)
            if num is not None and num == 0:
                return 1.0

        expected_str = self._normalize_cell(expected)
        actual_str = normalize_text(actual)

        # Exact match after normalization
        if expected_str == actual_str:
            return 1.0

        # Date-aware match (e.g. "2008-01-01 00:00:00" vs "Q1-2008")
        if _dates_match(str(expected), actual):
            return 1.0

        # Try numeric similarity first (graduated scoring)
        # Use original strings (not normalized) for numeric comparison
        # since normalize_text removes decimal points
        if self.normalize_numbers:
            num_score = numeric_similarity(str(expected), actual)
            if num_score is not None:
                # If direct comparison is poor, the CSV value may be in a
                # different magnitude than the displayed value (e.g. CSV has
                # raw 495926400 but chart shows ~495 meaning "millions").
                # Try dividing expected by common scales and keep best score.
                if num_score < CELL_FUZZY_MATCH_THRESHOLD:
                    exp_num = normalize_number_string(str(expected))
                    if exp_num is not None and exp_num != 0:
                        for scale in (1e3, 1e6, 1e9, 1e12):
                            scaled_score = numeric_similarity(str(exp_num / scale), actual)
                            if scaled_score is not None and scaled_score > num_score:
                                num_score = scaled_score
                return num_score

        # Fall back to fuzzy string matching
        similarity = fuzz.ratio(expected_str, actual_str) / 100.0
        return similarity

    def _find_matching_column(  # type: ignore[no-untyped-def]
        self, expected_col_data: list, table_array, used_cols: set, start_row: int = 1
    ) -> int | None:
        """Find a column in table that matches expected column data (rows in order)."""
        rows, cols = table_array.shape
        data_rows = rows - start_row
        if len(expected_col_data) != data_rows:
            return None

        for col_idx in range(cols):
            if col_idx in used_cols:
                continue
            all_match = True
            for data_row_idx, expected_val in enumerate(expected_col_data):
                actual_row_idx = data_row_idx + start_row
                actual_val = str(table_array[actual_row_idx, col_idx])
                if not self._cells_match(expected_val, actual_val):
                    all_match = False
                    break
            if all_match:
                return col_idx
        return None

    def _find_matching_column_unordered(  # type: ignore[no-untyped-def]
        self, expected_col_data: list, table_array, used_cols: set, start_row: int = 1
    ) -> int | None:
        """
        Find a column in table that matches expected column data as a multiset.

        This is used when both x and y axis shuffle are enabled - the column values
        must all be present but can be in any row order.
        """
        rows, cols = table_array.shape
        data_rows = rows - start_row
        if len(expected_col_data) != data_rows:
            return None

        for col_idx in range(cols):
            if col_idx in used_cols:
                continue

            # Get actual column values
            actual_vals = [str(table_array[row_idx, col_idx]) for row_idx in range(start_row, rows)]

            # Try to match each expected value to an actual value (multiset match)
            used_actual: set[int] = set()
            all_matched = True

            for expected_val in expected_col_data:
                found = False
                for act_idx, actual_val in enumerate(actual_vals):
                    if act_idx not in used_actual and self._cells_match(expected_val, actual_val):
                        used_actual.add(act_idx)
                        found = True
                        break
                if not found:
                    all_matched = False
                    break

            if all_matched:
                return col_idx
        return None

    def _find_matching_row(  # type: ignore[no-untyped-def]
        self, expected_row: list, table_array, used_rows: set, start_row: int = 1
    ) -> int | None:
        """Find a row in table that matches expected row data."""
        rows, cols = table_array.shape
        if len(expected_row) != cols:
            return None

        for row_idx in range(start_row, rows):
            if row_idx in used_rows:
                continue
            all_match = True
            for col_idx, expected_val in enumerate(expected_row):
                actual_val = str(table_array[row_idx, col_idx])
                if not self._cells_match(expected_val, actual_val):
                    all_match = False
                    break
            if all_match:
                return row_idx
        return None

    def _check_data(self, table_array) -> tuple[float, float, list[str]]:  # type: ignore[no-untyped-def]
        """
        Check how many data cells match using score-based comparison.

        Returns: (score, max_score, mismatched_details)
        """
        actual_rows, actual_cols = table_array.shape
        expected_data_rows = len(self.data_rows)

        # Data starts at row 1 (row 0 is header)
        actual_data_rows = actual_rows - 1

        # Columns must match; rows may differ (partial credit for missing rows)
        if actual_cols != self.expected_cols:
            return (
                0.0,
                float(expected_data_rows * self.expected_cols),
                [f"Column mismatch: expected {self.expected_cols} cols, got {actual_cols}"],
            )

        total_cells = float(expected_data_rows * self.expected_cols)
        mismatches: list[str] = []

        if actual_data_rows != expected_data_rows:
            mismatches.append(f"Row count mismatch: expected {expected_data_rows}, got {actual_data_rows}")

        data_rows = self.data_rows

        # Case 1: No shuffling - direct position comparison
        if not self.x_axis_shuffle and not self.y_axis_shuffle:
            total_score = 0.0
            row_details: list[tuple[int, float, list[str], list[str]]] = []
            matchable_rows = min(expected_data_rows, actual_data_rows)
            for row_idx, expected_row in enumerate(data_rows):
                if row_idx >= matchable_rows:
                    # Remaining expected rows have no actual counterpart (score 0)
                    break
                actual_row_idx = row_idx + 1  # Skip header row
                row_score = 0.0
                actual_row_values: list[str] = []
                for col_idx, expected_val in enumerate(expected_row):
                    actual_val = str(table_array[actual_row_idx, col_idx])
                    actual_row_values.append(actual_val)
                    cell_score = self._cell_score(expected_val, actual_val)
                    row_score += cell_score
                total_score += row_score
                max_row_score = len(expected_row)
                row_pct = row_score / max_row_score if max_row_score > 0 else 0
                if row_pct < 1.0:
                    row_details.append(
                        (
                            row_idx + 1,
                            row_pct,
                            [str(v) for v in expected_row],
                            actual_row_values,
                        )
                    )
            if row_details:
                row_details.sort(key=lambda x: x[1])  # worst rows first
                for r_idx, r_pct, exp_vals, act_vals in row_details[:5]:
                    mismatches.append(f"Row {r_idx}: {r_pct:.0%} | Expected: {exp_vals} | Actual: {act_vals}")
            return total_score, total_cells, mismatches

        # Case 2: Y-axis shuffle only (rows can reorder, columns fixed)
        if self.y_axis_shuffle and not self.x_axis_shuffle:
            used_rows: set[int] = set()
            total_score = 0.0
            row_details: list[tuple[int, float, list[str], list[str]]] = []  # type: ignore[no-redef]

            for exp_row_idx, expected_row in enumerate(data_rows):
                # Find best matching row using scores
                best_row_score = 0.0
                best_row_idx = -1
                best_actual_values: list[str] = []

                for act_row_idx in range(1, actual_rows):  # Skip header
                    if act_row_idx in used_rows:
                        continue
                    # Calculate score for this row pairing
                    row_score = 0.0
                    actual_values: list[str] = []
                    for col_idx, expected_val in enumerate(expected_row):
                        actual_val = str(table_array[act_row_idx, col_idx])
                        actual_values.append(actual_val)
                        row_score += self._cell_score(expected_val, actual_val)

                    if row_score > best_row_score:
                        best_row_score = row_score
                        best_row_idx = act_row_idx
                        best_actual_values = actual_values

                if best_row_idx >= 0:
                    used_rows.add(best_row_idx)
                    total_score += best_row_score
                    max_row_score = len(expected_row)
                    row_pct = best_row_score / max_row_score if max_row_score > 0 else 0
                    if row_pct < 1.0:
                        row_details.append(
                            (
                                exp_row_idx + 1,
                                row_pct,
                                [str(v) for v in expected_row],
                                best_actual_values,
                            )
                        )

            if row_details:
                row_details.sort(key=lambda x: x[1])  # worst rows first
                for r_idx, r_pct, exp_vals, act_vals in row_details[:5]:
                    mismatches.append(f"Row {r_idx}: {r_pct:.0%} | Expected: {exp_vals} | Actual: {act_vals}")
            return total_score, total_cells, mismatches

        # Case 3: X-axis shuffle only (columns can reorder, rows fixed)
        if self.x_axis_shuffle and not self.y_axis_shuffle:
            matchable_rows = min(expected_data_rows, actual_data_rows)
            expected_cols_data = [
                [row[col_idx] for row in data_rows[:matchable_rows]] for col_idx in range(self.expected_cols)
            ]
            used_cols: set[int] = set()
            total_score = 0.0
            # Track column mapping: expected_col_idx -> actual_col_idx
            col_mapping: dict[int, int] = {}

            for col_idx, expected_col in enumerate(expected_cols_data):
                # Find best matching column using scores
                best_col_score = 0.0
                best_col_idx = -1

                for act_col_idx in range(actual_cols):
                    if act_col_idx in used_cols:
                        continue
                    # Calculate score for this column pairing
                    col_score = 0.0
                    for row_idx, expected_val in enumerate(expected_col):
                        actual_row_idx = row_idx + 1  # Skip header
                        actual_val = str(table_array[actual_row_idx, act_col_idx])
                        col_score += self._cell_score(expected_val, actual_val)

                    if col_score > best_col_score:
                        best_col_score = col_score
                        best_col_idx = act_col_idx

                if best_col_idx >= 0:
                    used_cols.add(best_col_idx)
                    col_mapping[col_idx] = best_col_idx
                    total_score += best_col_score

            # Generate row-by-row comparison using the column mapping
            row_details: list[tuple[int, float, list[str], list[str]]] = []  # type: ignore[no-redef]
            for row_idx, expected_row in enumerate(data_rows):
                if row_idx >= matchable_rows:
                    break
                actual_row_idx = row_idx + 1
                row_score = 0.0
                actual_row_values: list[str] = []  # type: ignore[no-redef]
                for col_idx, expected_val in enumerate(expected_row):
                    act_col_idx = col_mapping.get(col_idx, col_idx)
                    actual_val = str(table_array[actual_row_idx, act_col_idx])
                    actual_row_values.append(actual_val)
                    row_score += self._cell_score(expected_val, actual_val)
                max_row_score = len(expected_row)
                row_pct = row_score / max_row_score if max_row_score > 0 else 0
                if row_pct < 1.0:
                    row_details.append(
                        (
                            row_idx + 1,
                            row_pct,
                            [str(v) for v in expected_row],
                            actual_row_values,
                        )
                    )

            if row_details:
                row_details.sort(key=lambda x: x[1])  # worst rows first
                for r_idx, r_pct, exp_vals, act_vals in row_details[:5]:
                    mismatches.append(f"Row {r_idx}: {r_pct:.0%} | Expected: {exp_vals} | Actual: {act_vals}")
            return total_score, total_cells, mismatches

        # Case 4: Both axes can shuffle
        # Use score-based matching: find best column mapping, then best row mapping
        total_score = 0.0

        # For each expected column, find the best matching actual column
        expected_cols_data = [[row[col_idx] for row in data_rows] for col_idx in range(self.expected_cols)]

        used_cols_set: set[int] = set()
        col_mapping: dict[int, int] = {}  # type: ignore[no-redef]

        for exp_col_idx, expected_col in enumerate(expected_cols_data):
            best_col_score = 0.0
            best_col_idx = -1

            for act_col_idx in range(actual_cols):
                if act_col_idx in used_cols_set:
                    continue
                # Calculate unordered score (find best row matches)
                col_score = 0.0
                used_rows_temp: set[int] = set()
                for expected_val in expected_col:
                    best_cell_score = 0.0
                    for row_idx in range(1, actual_rows):
                        if row_idx in used_rows_temp:
                            continue
                        actual_val = str(table_array[row_idx, act_col_idx])
                        cell_score = self._cell_score(expected_val, actual_val)
                        if cell_score > best_cell_score:
                            best_cell_score = cell_score
                    col_score += best_cell_score

                if col_score > best_col_score:
                    best_col_score = col_score
                    best_col_idx = act_col_idx

            if best_col_idx >= 0:
                used_cols_set.add(best_col_idx)
                col_mapping[exp_col_idx] = best_col_idx
                total_score += best_col_score

        # Generate row-by-row comparison with best-effort column mapping
        # Since rows can also shuffle, find best row matches for reporting
        row_details: list[tuple[int, float, list[str], list[str]]] = []  # type: ignore[no-redef]
        for exp_row_idx, expected_row in enumerate(data_rows):
            best_row_score = 0.0
            best_actual_values: list[str] = []  # type: ignore[no-redef]
            # Find the best matching actual row
            for act_row_idx in range(1, actual_rows):
                row_score = 0.0
                actual_values: list[str] = []  # type: ignore[no-redef]
                for col_idx, expected_val in enumerate(expected_row):
                    act_col_idx = col_mapping.get(col_idx, col_idx)
                    if act_col_idx < actual_cols:
                        actual_val = str(table_array[act_row_idx, act_col_idx])
                    else:
                        actual_val = ""
                    actual_values.append(actual_val)
                    row_score += self._cell_score(expected_val, actual_val)
                if row_score > best_row_score:
                    best_row_score = row_score
                    best_actual_values = actual_values
            max_row_score = len(expected_row)
            row_pct = best_row_score / max_row_score if max_row_score > 0 else 0
            if row_pct < 1.0:
                row_details.append(
                    (
                        exp_row_idx + 1,
                        row_pct,
                        [str(v) for v in expected_row],
                        best_actual_values,
                    )
                )

        if row_details:
            row_details.sort(key=lambda x: x[1])  # worst rows first
            for r_idx, r_pct, exp_vals, act_vals in row_details[:5]:
                mismatches.append(f"Row {r_idx}: {r_pct:.0%} | Expected: {exp_vals} | Actual: {act_vals}")
        return total_score, total_cells, mismatches

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        """Check if expected data values match any table in content."""
        tables_to_check = []

        md_tables = parse_markdown_tables(content)
        tables_to_check.extend(md_tables)

        html_tables = parse_html_tables(content)
        tables_to_check.extend(html_tables)

        if not tables_to_check:
            return False, "No tables found in content", 0.0

        best_score = 0.0
        best_total = float(len(self.data_rows) * self.expected_cols)
        best_mismatches: list[str] = []

        for table_data in tables_to_check:
            # Try both orientations and keep the best score
            orientations = [table_data.data, table_data.data.T]
            for data in orientations:
                score, total, mismatches = self._check_data(data)

                if score == total:
                    return True, f"Data: 100% ({score:.1f}/{total:.0f})", 1.0

                if score > best_score or (score == best_score == 0.0 and not best_mismatches):
                    best_score = score
                    best_total = total
                    best_mismatches = mismatches

        score_pct = (best_score / best_total) * 100 if best_total > 0 else 0
        score_normalized = score_pct / 100.0

        if score_pct >= 99.5:
            return (
                True,
                f"Data: {score_pct:.1f}% ({best_score:.1f}/{best_total:.0f})",
                score_normalized,
            )

        if best_mismatches:
            return (
                False,
                (f"Data: {score_pct:.1f}% ({best_score:.1f}/{best_total:.0f}). {'; '.join(best_mismatches[:3])}"),
                score_normalized,
            )
        return (
            False,
            f"Data: {score_pct:.1f}% ({best_score:.1f}/{best_total:.0f})",
            score_normalized,
        )


class RotateCheckRule(ParseTestRule):
    """Test rule that validates the detected original_orientation_angle.

    parse_output and raw_output are set by RuleBasedMetric before calling run().
    parse_output.layout_pages is the primary source; raw_output.pages is the
    legacy fallback for historical artifacts.
    """

    def __init__(self, rule_data: ParseRotateCheckRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseRotateCheckRule, self._rule_data)

        self.expected_angle = rule_data.value
        if self.expected_angle is None:
            raise ValueError("rotate_check rule must have a 'value' field (expected angle)")
        self.parse_output: ParseOutput | None = None
        self.raw_output: dict[str, Any] | None = None

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        actual_angle = self._angle_from_parse_output()
        if actual_angle is None:
            actual_angle = self._angle_from_raw_output()
        if actual_angle is None:
            return False, "No original_orientation_angle found in output metadata"

        try:
            if actual_angle == self.expected_angle:
                return True, ""
            return False, (f"Expected orientation angle {self.expected_angle}, got {actual_angle}")
        except Exception as e:
            return False, f"Error checking orientation angle: {e}"

    def _angle_from_parse_output(self) -> int | float | str | None:
        if self.parse_output is None:
            return None
        if not self.parse_output.layout_pages:
            return None
        return self.parse_output.layout_pages[0].original_orientation_angle

    def _angle_from_raw_output(self) -> int | float | str | None:
        if self.raw_output is None:
            return None
        pages = self.raw_output.get("pages")
        if not isinstance(pages, list) or not pages:
            return None
        first_page = pages[0]
        if not isinstance(first_page, dict):
            return None
        return first_page.get("original_orientation_angle")
