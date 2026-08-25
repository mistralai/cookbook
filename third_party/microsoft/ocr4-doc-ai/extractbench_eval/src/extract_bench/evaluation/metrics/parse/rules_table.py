"""Table structure and hierarchy test rules."""

import json
import re
from collections import Counter
from typing import Any, cast

import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

from extract_bench.evaluation.metrics.parse.rules_base import (
    CELL_FUZZY_MATCH_THRESHOLD,
    AdjacentTableRuleData,
    NoBorderTableRuleData,
    ParseTestRule,
)
from extract_bench.evaluation.metrics.parse.table_parsing import (
    ResolvedGrid,
    TableData,
    find_all_html_tables,
    find_cell_in_grids,
    find_table_by_anchors,
    parse_html_tables,
    parse_markdown_tables,
)
from extract_bench.evaluation.metrics.parse.test_types import ParseRuleType
from extract_bench.evaluation.metrics.parse.utils import normalize_text
from extract_bench.test_cases.parse_rule_schemas import (
    ParseTableAdjacentDownRule,
    ParseTableAdjacentLeftRule,
    ParseTableAdjacentRightRule,
    ParseTableAdjacentUpRule,
    ParseTableColspanRule,
    ParseTableHeaderChainRule,
    ParseTableLeftHeaderRule,
    ParseTableNoAboveRule,
    ParseTableNoBelowRule,
    ParseTableNoLeftRule,
    ParseTableNoRightRule,
    ParseTableRowspanRule,
    ParseTableRule,
    ParseTableSameColumnRule,
    ParseTableSameRowRule,
    ParseTablesNumColsRule,
    ParseTablesNumRowsRule,
    ParseTablesValuesRule,
    ParseTableTopHeaderRule,
)


class TableRule(ParseTestRule):
    """Test rule to verify table cell relationships."""

    def __init__(self, rule_data: ParseTableRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableRule, self._rule_data)

        if self.type != ParseRuleType.TABLE.value:
            raise ValueError(f"Invalid type for TableRule: {self.type}")

        # Normalize the search text
        self.cell = normalize_text(rule_data.cell)
        self.up = normalize_text(rule_data.up or "")
        self.down = normalize_text(rule_data.down or "")
        self.left = normalize_text(rule_data.left or "")
        self.right = normalize_text(rule_data.right or "")
        self.top_heading = normalize_text(rule_data.top_heading or "")
        self.left_heading = normalize_text(rule_data.left_heading or "")
        self.ignore_markdown_tables = rule_data.ignore_markdown_tables

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if table cell relationships are satisfied."""
        tables_to_check = []
        failed_reasons = []

        # Threshold for fuzzy matching derived from max_diffs
        threshold = 1.0 - (self.max_diffs / (len(self.cell) if len(self.cell) > 0 else 1))
        threshold = max(0.5, threshold)

        # Parse tables
        if not self.ignore_markdown_tables:
            md_tables = parse_markdown_tables(content)
            tables_to_check.extend(md_tables)

        html_tables = parse_html_tables(content)
        tables_to_check.extend(html_tables)

        # If no tables found, return failure
        if not tables_to_check:
            return False, "No tables found in the content"

        # Check each table
        for table_data in tables_to_check:
            table_array = table_data.data
            header_rows = table_data.header_rows
            header_cols = table_data.header_cols

            # Find all cells that match the target cell using fuzzy matching
            matches = []
            for i in range(table_array.shape[0]):
                for j in range(table_array.shape[1]):
                    cell_content = normalize_text(str(table_array[i, j]))
                    similarity = fuzz.ratio(self.cell, cell_content) / 100.0

                    if similarity >= threshold:
                        matches.append((i, j))

            # If no matches found in this table, continue to the next table
            if not matches:
                continue

            # Check the relationships for each matching cell
            for row_idx, col_idx in matches:
                all_relationships_satisfied = True
                current_failed_reasons = []

                # Check up relationship
                if self.up and row_idx > 0:
                    up_cell = normalize_text(str(table_array[row_idx - 1, col_idx]))
                    up_similarity = fuzz.ratio(self.up, up_cell) / 100.0
                    up_threshold = max(0.5, 1.0 - (self.max_diffs / (len(self.up) if len(self.up) > 0 else 1)))
                    if up_similarity < up_threshold:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(
                            f"Cell above '{up_cell}' doesn't match "
                            f"expected '{self.up}' "
                            f"(similarity: {up_similarity:.2f})"
                        )

                # Check down relationship
                if self.down and row_idx < table_array.shape[0] - 1:
                    down_cell = normalize_text(str(table_array[row_idx + 1, col_idx]))
                    down_similarity = fuzz.ratio(self.down, down_cell) / 100.0
                    down_threshold = max(0.5, 1.0 - (self.max_diffs / (len(self.down) if len(self.down) > 0 else 1)))
                    if down_similarity < down_threshold:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(
                            f"Cell below '{down_cell}' doesn't match "
                            f"expected '{self.down}' "
                            f"(similarity: {down_similarity:.2f})"
                        )

                # Check left relationship
                if self.left and col_idx > 0:
                    left_cell = normalize_text(str(table_array[row_idx, col_idx - 1]))
                    left_similarity = fuzz.ratio(self.left, left_cell) / 100.0
                    left_threshold = max(0.5, 1.0 - (self.max_diffs / (len(self.left) if len(self.left) > 0 else 1)))
                    if left_similarity < left_threshold:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(
                            f"Cell to the left '{left_cell}' doesn't "
                            f"match expected '{self.left}' "
                            f"(similarity: {left_similarity:.2f})"
                        )

                # Check right relationship
                if self.right and col_idx < table_array.shape[1] - 1:
                    right_cell = normalize_text(str(table_array[row_idx, col_idx + 1]))
                    right_similarity = fuzz.ratio(self.right, right_cell) / 100.0
                    right_threshold = max(
                        0.5,
                        1.0 - (self.max_diffs / (len(self.right) if len(self.right) > 0 else 1)),
                    )
                    if right_similarity < right_threshold:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(
                            f"Cell to the right '{right_cell}' doesn't "
                            f"match expected '{self.right}' "
                            f"(similarity: {right_similarity:.2f})"
                        )

                # Check top heading relationship
                if self.top_heading:
                    top_heading_found = False
                    best_match = ""
                    best_similarity = 0.0

                    # Check the col_headers dictionary first
                    if col_idx in table_data.col_headers:
                        for _, header_text in table_data.col_headers[col_idx]:
                            header_text = normalize_text(header_text)
                            similarity = fuzz.ratio(self.top_heading, header_text) / 100.0
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = header_text
                                top_threshold = max(
                                    0.5,
                                    1.0
                                    - (self.max_diffs / (len(self.top_heading) if len(self.top_heading) > 0 else 1)),
                                )
                                if best_similarity >= top_threshold:
                                    top_heading_found = True
                                    break

                    # If no match found in col_headers, fall back to checking header rows
                    if not top_heading_found and header_rows:
                        for i in sorted(header_rows):
                            if i < row_idx and str(table_array[i, col_idx]).strip():
                                header_text = normalize_text(str(table_array[i, col_idx]))
                                similarity = fuzz.ratio(self.top_heading, header_text) / 100.0
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = header_text
                                    top_threshold = max(
                                        0.5,
                                        1.0
                                        - (
                                            self.max_diffs / (len(self.top_heading) if len(self.top_heading) > 0 else 1)
                                        ),
                                    )
                                    if best_similarity >= top_threshold:
                                        top_heading_found = True
                                        break

                    # If still no match, use any non-empty cell above as a last resort
                    if not top_heading_found and not best_match and row_idx > 0:
                        for i in range(row_idx):
                            if str(table_array[i, col_idx]).strip():
                                header_text = normalize_text(str(table_array[i, col_idx]))
                                similarity = fuzz.ratio(self.top_heading, header_text) / 100.0
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = header_text

                    if not best_match:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(f"No top heading found for cell at ({row_idx}, {col_idx})")
                    else:
                        top_threshold = max(
                            0.5,
                            1.0 - (self.max_diffs / (len(self.top_heading) if len(self.top_heading) > 0 else 1)),
                        )
                        if best_similarity < top_threshold:
                            all_relationships_satisfied = False
                            current_failed_reasons.append(
                                f"Top heading '{best_match}' doesn't "
                                f"match expected '{self.top_heading}' "
                                f"(similarity: {best_similarity:.2f})"
                            )

                # Check left heading relationship
                if self.left_heading:
                    left_heading_found = False
                    best_match = ""
                    best_similarity = 0.0

                    # Check the row_headers dictionary first
                    if row_idx in table_data.row_headers:
                        for _, header_text in table_data.row_headers[row_idx]:
                            header_text = normalize_text(header_text)
                            similarity = fuzz.ratio(self.left_heading, header_text) / 100.0
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = header_text
                                left_threshold = max(
                                    0.5,
                                    1.0
                                    - (self.max_diffs / (len(self.left_heading) if len(self.left_heading) > 0 else 1)),
                                )
                                if best_similarity >= left_threshold:
                                    left_heading_found = True
                                    break

                    # If no match found in row_headers, fall back to checking header columns
                    if not left_heading_found and header_cols:
                        for j in sorted(header_cols):
                            if j < col_idx and str(table_array[row_idx, j]).strip():
                                header_text = normalize_text(str(table_array[row_idx, j]))
                                similarity = fuzz.ratio(self.left_heading, header_text) / 100.0
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = header_text
                                    left_threshold = max(
                                        0.5,
                                        1.0
                                        - (
                                            self.max_diffs
                                            / (len(self.left_heading) if len(self.left_heading) > 0 else 1)
                                        ),
                                    )
                                    if best_similarity >= left_threshold:
                                        left_heading_found = True
                                        break

                    # If still no match, use any non-empty cell to the left as a last resort
                    if not left_heading_found and not best_match and col_idx > 0:
                        for j in range(col_idx):
                            if str(table_array[row_idx, j]).strip():
                                header_text = normalize_text(str(table_array[row_idx, j]))
                                similarity = fuzz.ratio(self.left_heading, header_text) / 100.0
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = header_text

                    if not best_match:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(f"No left heading found for cell at ({row_idx}, {col_idx})")
                    else:
                        left_threshold = max(
                            0.5,
                            1.0 - (self.max_diffs / (len(self.left_heading) if len(self.left_heading) > 0 else 1)),
                        )
                        if best_similarity < left_threshold:
                            all_relationships_satisfied = False
                            current_failed_reasons.append(
                                f"Left heading '{best_match}' doesn't "
                                f"match expected '{self.left_heading}' "
                                f"(similarity: {best_similarity:.2f})"
                            )

                # If all relationships are satisfied for this cell, the test passes
                if all_relationships_satisfied:
                    return True, ""
                else:
                    failed_reasons.extend(current_failed_reasons)

        if not failed_reasons:
            return (
                False,
                f"No cell matching '{self.cell}' found in any table with threshold {threshold}",
            )
        else:
            return (
                False,
                f"Found cells matching '{self.cell}' but relationships were not satisfied: {'; '.join(failed_reasons)}",
            )


class TablesValuesRule(ParseTestRule):
    """Test rule to verify that tables match ground truth tables."""

    def __init__(self, rule_data: ParseTablesValuesRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTablesValuesRule, self._rule_data)

        if self.type != ParseRuleType.TABLES_VALUES.value:
            raise ValueError(f"Invalid type for TablesValuesRule: {self.type}")

        self.table_variations = rule_data.table_variations
        self.json_path = rule_data.json_path
        self.table_match_threshold = rule_data.table_match_threshold
        self.table_values_match_threshold = rule_data.table_values_match_threshold
        self.add_check_num_rows_test = rule_data.add_check_num_rows_test
        self.add_check_num_cols_test = rule_data.add_check_num_cols_test

        # Must have either table_variations or json_path
        if not self.table_variations and not self.json_path:
            raise ValueError("Either table_variations or json_path must be provided")

        self.relevant_gt: pd.DataFrame | None = None
        self.relevant_pred: pd.DataFrame | None = None

    def _load_json_table(self, json_file_path: str) -> dict[str, Any]:
        """Load the ground truth table JSON file."""
        with open(json_file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Validate the structure
        required_keys = ["pdf", "page", "table_variations", "id", "table_match_threshold"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"JSON file missing required key: {key}")

        return data  # type: ignore[no-any-return]

    def _tabledata_to_dataframe(self, table_data: TableData) -> pd.DataFrame:
        """Convert a TableData object to a pandas DataFrame."""
        return pd.DataFrame(table_data.data)

    def _table_schema_to_dataframe(self, table_schema: dict[str, Any]) -> pd.DataFrame:
        """Convert a table schema (with rowspan/colspan) to a pandas DataFrame."""
        if "rows" not in table_schema:
            raise ValueError("table_schema missing 'rows' key")

        rows_data = table_schema["rows"]

        # First pass: determine the grid size and build a cell position map
        grid = {}  # (row_idx, col_idx) -> cell_text
        max_cols = 0

        for row_idx, row_dict in enumerate(rows_data):
            if "cells" not in row_dict:
                raise ValueError(f"Row {row_idx} missing 'cells' key")

            cells = row_dict["cells"]
            col_idx = 0

            for cell_dict in cells:
                # Skip columns that are already filled by previous rowspan/colspan
                while (row_idx, col_idx) in grid:
                    col_idx += 1

                # Extract cell properties
                text = cell_dict.get("text", "")
                colspan = cell_dict.get("colspan", 1)
                rowspan = cell_dict.get("rowspan", 1)

                # Fill the grid for this cell and all its spans
                for r_offset in range(rowspan):
                    for c_offset in range(colspan):
                        grid[(row_idx + r_offset, col_idx + c_offset)] = text

                col_idx += colspan
                max_cols = max(max_cols, col_idx)

        # Second pass: build the DataFrame from the grid
        num_rows = max(r for r, c in grid.keys()) + 1 if grid else 0
        num_cols = max_cols

        # Create a 2D list for the DataFrame
        data_array = []
        for r in range(num_rows):
            row = []
            for c in range(num_cols):
                cell_value = grid.get((r, c), "")
                row.append(cell_value)
            data_array.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(data_array)

        return df

    def _normalize_cell(self, cell: str) -> str:
        """Normalize a cell value for comparison."""
        text = unidecode(str(cell)).lower()
        # Remove all whitespace
        text = re.sub(r"\s+", "", text)
        # Remove zero-width characters
        text = re.sub(r"[\u200B-\u200D\uFEFF\u00AD]", "", text)

        # Handle numbers with commas
        number_pattern = r"(-?\d{1,3}(?:,\d{3})*\.?\d*)"
        match = re.search(number_pattern, text)

        if match:
            number_str = match.group(1)
            try:
                clean_number = number_str.replace(",", "")
                float_val = float(clean_number)

                if "," in number_str:
                    if float_val >= 1000:
                        normalized_number = f"{float_val:,.10g}".rstrip("0").rstrip(".")
                    else:
                        normalized_number = f"{float_val:g}"
                else:
                    normalized_number = f"{float_val:g}"

                return text.replace(number_str, normalized_number)
            except ValueError:
                pass

        return text

    def _compute_single_table_similarity(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
        # Extract and normalize all words from ground truth table
        gt_words = []
        for row_idx in range(gt_df.shape[0]):
            for col_idx in range(gt_df.shape[1]):
                cell_value = str(gt_df.iloc[row_idx, col_idx])
                normalized = self._normalize_cell(cell_value)
                if normalized:
                    gt_words.append(normalized)

        gt_counter = Counter(gt_words)

        # If ground truth is empty, return 0
        if not gt_words:
            return 0.0

        # Extract and normalize all words from predicted table
        pred_words = []
        for row_idx in range(pred_df.shape[0]):
            for col_idx in range(pred_df.shape[1]):
                cell_value = str(pred_df.iloc[row_idx, col_idx])
                normalized = self._normalize_cell(cell_value)
                if normalized:
                    pred_words.append(normalized)

        pred_counter = Counter(pred_words)

        # If predicted table is empty, return 0
        if not pred_words:
            return 0.0

        # Compute intersection (minimum counts for each word)
        intersection = sum((gt_counter & pred_counter).values())

        # Compute union (maximum counts for each word)
        union = sum((gt_counter | pred_counter).values())

        # Compute Jaccard similarity with counts
        if union > 0:
            return intersection / union

        return 0.0

    def _compare_cells_exactly(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[int, int, float]:
        """Compare cells one-by-one between ground truth and predicted DataFrames."""
        # Compare only the overlapping region
        min_rows = min(len(gt_df), len(pred_df))
        min_cols = min(len(gt_df.columns), len(pred_df.columns))

        matching_cells = 0
        total_cells = min_rows * min_cols

        if total_cells == 0:
            return 0, 0, 0.0

        for row_idx in range(min_rows):
            for col_idx in range(min_cols):
                gt_cell = str(gt_df.iloc[row_idx, col_idx])
                pred_cell = str(pred_df.iloc[row_idx, col_idx])

                # Normalize both cells
                gt_normalized = self._normalize_cell(gt_cell)
                pred_normalized = self._normalize_cell(pred_cell)

                # Exact match after normalization
                if gt_normalized == pred_normalized:
                    matching_cells += 1

        match_ratio = matching_cells / total_cells if total_cells > 0 else 0.0
        return matching_cells, total_cells, match_ratio

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Run the table values test on provided content."""
        # Extract tables from content
        pred_tables = []

        # Parse markdown tables
        md_tables = parse_markdown_tables(content)
        pred_tables.extend(md_tables)

        # Parse HTML tables
        html_tables = parse_html_tables(content)
        pred_tables.extend(html_tables)

        if not pred_tables:
            return False, "No tables found in the content"

        pred_tables = [self._tabledata_to_dataframe(table) for table in pred_tables]

        # Load table variations either from embedded data or external JSON
        try:
            if self.table_variations:
                table_variations = self.table_variations
            elif self.json_path:
                gt_data = self._load_json_table(self.json_path)
                table_variations = gt_data["table_variations"]
            else:
                return False, "No table variations available (neither embedded nor in json_path)"

            if not table_variations:
                return False, "No table variations found"

        except Exception as e:
            return False, f"Error loading ground truth data: {e}"

        # Track the best variation and its score
        best_variation_idx = -1
        best_score = 0.0
        best_pred_idx = -1

        for var_idx, table_schema in enumerate(table_variations):
            try:
                # Convert the table schema to a DataFrame
                gt_df = self._table_schema_to_dataframe(table_schema)

                # Compare this GT variation with each predicted table
                for pred_idx, pred_table in enumerate(pred_tables):
                    # Compute similarity between this specific GT and this specific pred table
                    similarity = self._compute_single_table_similarity(gt_df, pred_table)

                    # Track the overall best score across all GT-pred pairs
                    if similarity > best_score:
                        best_score = similarity
                        best_variation_idx = var_idx
                        best_pred_idx = pred_idx
                        self.relevant_pred = pred_table
                        self.relevant_gt = gt_df

            except Exception:
                # If conversion fails, continue to next variation
                continue

        # Check if the best variation passes the threshold
        threshold = self.table_match_threshold

        if best_score < threshold:
            return (
                False,
                f"Best match: GT variation {best_variation_idx} with pred table {best_pred_idx} "
                f"scored {best_score:.3f}, below threshold {threshold:.3f}",
            )

        # Perform exact cell-by-cell comparison on the best match
        if self.relevant_gt is None or self.relevant_pred is None:
            return False, "No relevant GT or pred table found for cell comparison"

        matching_cells, total_cells, match_ratio = self._compare_cells_exactly(self.relevant_gt, self.relevant_pred)
        cell_threshold = self.table_values_match_threshold

        if match_ratio < cell_threshold:
            return (  # type: ignore[return-value]
                False,
                f"Best match: GT variation {best_variation_idx} with pred table {best_pred_idx} "
                f"scored {best_score:.3f} (>= {threshold:.3f}), "
                f"but cell exact match "
                f"{matching_cells}/{total_cells} "
                f"({match_ratio:.3f}) below threshold "
                f"{cell_threshold:.3f}",
                f"({match_ratio:.3f}) below threshold {cell_threshold:.3f}",
            )

        return (
            True,
            f"Best match: GT variation {best_variation_idx} with pred table {best_pred_idx} "
            f"scored {best_score:.3f} (>= {threshold:.3f}), "
            f"cell exact match {matching_cells}/{total_cells} "
            f"({match_ratio:.3f})",
        )


class TablesNumRowsRule(ParseTestRule):
    """Test rule to verify that predicted table has the correct number of rows."""

    def __init__(self, rule_data: ParseTablesNumRowsRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTablesNumRowsRule, self._rule_data)

        if self.type != ParseRuleType.TABLES_NUM_ROWS.value:
            raise ValueError(f"Invalid type for TablesNumRowsRule: {self.type}")

        self.expected_num_rows = rule_data.expected_num_rows
        self.actual_num_rows = rule_data.actual_num_rows

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if row count matches."""
        if self.actual_num_rows is None:
            return False, "Row count not populated"

        if self.actual_num_rows == self.expected_num_rows:
            return True, f"Row count matches: {self.actual_num_rows}"
        else:
            return (
                False,
                f"Row count mismatch: expected {self.expected_num_rows}, got {self.actual_num_rows}",
            )


class TablesNumColsRule(ParseTestRule):
    """Test rule to verify that predicted table has the correct number of columns."""

    def __init__(self, rule_data: ParseTablesNumColsRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTablesNumColsRule, self._rule_data)

        if self.type != ParseRuleType.TABLES_NUM_COLS.value:
            raise ValueError(f"Invalid type for TablesNumColsRule: {self.type}")

        self.expected_num_cols = rule_data.expected_num_cols
        self.actual_num_cols = rule_data.actual_num_cols

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if column count matches."""
        if self.actual_num_cols is None:
            return False, "Column count not populated"

        if self.actual_num_cols == self.expected_num_cols:
            return True, f"Column count matches: {self.actual_num_cols}"
        else:
            return (
                False,
                f"Column count mismatch: expected {self.expected_num_cols}, got {self.actual_num_cols}",
            )


# =============================================================================
# Table Hierarchy Rules
# =============================================================================


class TableColspanRule(ParseTestRule):
    """Test rule to verify a cell has the expected colspan attribute."""

    def __init__(self, rule_data: ParseTableColspanRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableColspanRule, self._rule_data)

        if self.type != ParseRuleType.TABLE_COLSPAN.value:
            raise ValueError(f"Invalid type for TableColspanRule: {self.type}")

        self.cell = normalize_text(rule_data.cell)
        self.expected_colspan = rule_data.expected_colspan
        self.table_anchor_cells = rule_data.table_anchor_cells

        if not self.cell:
            raise ValueError("cell must be provided")
        if self.expected_colspan < 1:
            raise ValueError("expected_colspan must be >= 1")

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if cell has expected colspan attribute."""
        grids = find_all_html_tables(content)
        if not grids:
            return False, "No HTML tables found in content"

        # Step 1: Find the correct table using anchor cells if provided
        if self.table_anchor_cells:
            anchor_result = find_table_by_anchors(grids, self.table_anchor_cells)
            if anchor_result.grid is not None:
                grids = [anchor_result.grid]
            elif anchor_result.is_ambiguous:
                return False, (
                    f"[AMBIGUOUS ANCHORS] Anchors matched {anchor_result.num_candidates} "
                    f"tables - could not uniquely identify target table"
                )
            else:
                return False, "Table anchor cells not found in any table"

        # Step 2: Find cell and check colspan
        match = find_cell_in_grids(grids, self.cell)
        if not match:
            return False, f"Cell '{self.cell}' not found in target table"

        grid, cell, row_idx, col_idx = match

        if cell.colspan == self.expected_colspan:
            return True, f"Cell '{self.cell}' has correct colspan={cell.colspan}"
        else:
            return (
                False,
                f"Cell '{self.cell}' has colspan={cell.colspan}, expected {self.expected_colspan}",
            )


class TableRowspanRule(ParseTestRule):
    """Test rule to verify a cell has the expected rowspan attribute."""

    def __init__(self, rule_data: ParseTableRowspanRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableRowspanRule, self._rule_data)

        if self.type != ParseRuleType.TABLE_ROWSPAN.value:
            raise ValueError(f"Invalid type for TableRowspanRule: {self.type}")

        self.cell = normalize_text(rule_data.cell)
        self.expected_rowspan = rule_data.expected_rowspan
        self.table_anchor_cells = rule_data.table_anchor_cells

        if not self.cell:
            raise ValueError("cell must be provided")
        if self.expected_rowspan < 1:
            raise ValueError("expected_rowspan must be >= 1")

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if cell has expected rowspan attribute."""
        grids = find_all_html_tables(content)
        if not grids:
            return False, "No HTML tables found in content"

        # Step 1: Find the correct table using anchor cells if provided
        if self.table_anchor_cells:
            anchor_result = find_table_by_anchors(grids, self.table_anchor_cells)
            if anchor_result.grid is not None:
                grids = [anchor_result.grid]
            elif anchor_result.is_ambiguous:
                return False, (
                    f"[AMBIGUOUS ANCHORS] Anchors matched {anchor_result.num_candidates} "
                    f"tables - could not uniquely identify target table"
                )
            else:
                return False, "Table anchor cells not found in any table"

        # Step 2: Find cell and check rowspan
        match = find_cell_in_grids(grids, self.cell)
        if not match:
            return False, f"Cell '{self.cell}' not found in target table"

        grid, cell, row_idx, col_idx = match

        if cell.rowspan == self.expected_rowspan:
            return True, f"Cell '{self.cell}' has correct rowspan={cell.rowspan}"
        else:
            return (
                False,
                f"Cell '{self.cell}' has rowspan={cell.rowspan}, expected {self.expected_rowspan}",
            )


class TableSameRowRule(ParseTestRule):
    """Test rule to verify two cells share a logical row (considering rowspan)."""

    def __init__(self, rule_data: ParseTableSameRowRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableSameRowRule, self._rule_data)

        if self.type != ParseRuleType.TABLE_SAME_ROW.value:
            raise ValueError(f"Invalid type for TableSameRowRule: {self.type}")

        self.cell_a = normalize_text(rule_data.cell_a)
        self.cell_b = normalize_text(rule_data.cell_b)
        self.table_anchor_cells = rule_data.table_anchor_cells

        if not self.cell_a or not self.cell_b:
            raise ValueError("Both cell_a and cell_b must be provided")

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if two cells share a logical row."""
        grids = find_all_html_tables(content)
        if not grids:
            return False, "No HTML tables found in content"

        # Step 1: Find the correct table using anchor cells if provided
        if self.table_anchor_cells:
            anchor_result = find_table_by_anchors(grids, self.table_anchor_cells)
            if anchor_result.grid is not None:
                grids = [anchor_result.grid]
            elif anchor_result.is_ambiguous:
                return False, (
                    f"[AMBIGUOUS ANCHORS] Anchors matched {anchor_result.num_candidates} "
                    f"tables - could not uniquely identify target table"
                )
            else:
                return False, "Table anchor cells not found in any table"

        match_a = find_cell_in_grids(grids, self.cell_a)
        if not match_a:
            return False, f"Cell '{self.cell_a}' not found in target table"

        match_b = find_cell_in_grids(grids, self.cell_b)
        if not match_b:
            return False, f"Cell '{self.cell_b}' not found in target table"

        grid_a, cell_a, row_a, col_a = match_a
        grid_b, cell_b, row_b, col_b = match_b

        # Must be in the same table
        if grid_a is not grid_b:
            return False, "Cells are in different tables"

        # Calculate row ranges for each cell (considering rowspan)
        rows_a = set(range(cell_a.original_row, cell_a.original_row + cell_a.rowspan))
        rows_b = set(range(cell_b.original_row, cell_b.original_row + cell_b.rowspan))

        if rows_a & rows_b:  # Intersection
            return True, f"Cells share rows: {rows_a & rows_b}"
        else:
            return False, f"Cells do not share any row. A: rows {rows_a}, B: rows {rows_b}"


class TableSameColumnRule(ParseTestRule):
    """Test rule to verify two cells share a logical column (considering colspan)."""

    def __init__(self, rule_data: ParseTableSameColumnRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableSameColumnRule, self._rule_data)

        if self.type != ParseRuleType.TABLE_SAME_COLUMN.value:
            raise ValueError(f"Invalid type for TableSameColumnRule: {self.type}")

        self.cell_a = normalize_text(rule_data.cell_a)
        self.cell_b = normalize_text(rule_data.cell_b)
        self.table_anchor_cells = rule_data.table_anchor_cells

        if not self.cell_a or not self.cell_b:
            raise ValueError("Both cell_a and cell_b must be provided")

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if two cells share a logical column."""
        grids = find_all_html_tables(content)
        if not grids:
            return False, "No HTML tables found in content"

        # Step 1: Find the correct table using anchor cells if provided
        if self.table_anchor_cells:
            anchor_result = find_table_by_anchors(grids, self.table_anchor_cells)
            if anchor_result.grid is not None:
                grids = [anchor_result.grid]
            elif anchor_result.is_ambiguous:
                return False, (
                    f"[AMBIGUOUS ANCHORS] Anchors matched {anchor_result.num_candidates} "
                    f"tables - could not uniquely identify target table"
                )
            else:
                return False, "Table anchor cells not found in any table"

        match_a = find_cell_in_grids(grids, self.cell_a)
        if not match_a:
            return False, f"Cell '{self.cell_a}' not found in target table"

        match_b = find_cell_in_grids(grids, self.cell_b)
        if not match_b:
            return False, f"Cell '{self.cell_b}' not found in target table"

        grid_a, cell_a, row_a, col_a = match_a
        grid_b, cell_b, row_b, col_b = match_b

        # Must be in the same table
        if grid_a is not grid_b:
            return False, "Cells are in different tables"

        # Calculate column ranges for each cell (considering colspan)
        cols_a = set(range(cell_a.original_col, cell_a.original_col + cell_a.colspan))
        cols_b = set(range(cell_b.original_col, cell_b.original_col + cell_b.colspan))

        if cols_a & cols_b:  # Intersection
            return True, f"Cells share columns: {cols_a & cols_b}"
        else:
            return False, f"Cells do not share any column. A: cols {cols_a}, B: cols {cols_b}"


class TableHeaderChainRule(ParseTestRule):
    """Test rule to verify a data cell has the correct header chain."""

    def __init__(self, rule_data: ParseTableHeaderChainRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableHeaderChainRule, self._rule_data)

        if self.type != ParseRuleType.TABLE_HEADER_CHAIN.value:
            raise ValueError(f"Invalid type for TableHeaderChainRule: {self.type}")

        self.data_cell = normalize_text(rule_data.data_cell)
        self.column_headers = rule_data.column_headers
        self.row_headers = rule_data.row_headers
        self.table_anchor_cells = rule_data.table_anchor_cells

        if not self.data_cell:
            raise ValueError("data_cell must be provided")
        if not self.column_headers and not self.row_headers:
            raise ValueError("At least one of column_headers or row_headers must be provided")

    def _get_column_headers(self, grid: ResolvedGrid, data_row: int, data_col: int) -> list[str]:
        """Get all column headers above the data cell."""
        headers = []
        seen_cells: set[tuple[int, int]] = set()

        for row_idx in range(data_row):
            cell = grid.cells[row_idx][data_col]
            if cell is None:
                continue
            # Use original position as key to avoid duplicates
            cell_key = (cell.original_row, cell.original_col)
            if cell_key in seen_cells:
                continue
            seen_cells.add(cell_key)

            if cell.text:
                headers.append(cell.text)

        return headers

    def _get_row_headers(self, grid: ResolvedGrid, data_row: int, data_col: int) -> list[str]:
        """Get all row headers to the left of the data cell."""
        headers = []
        seen_cells: set[tuple[int, int]] = set()

        for col_idx in range(data_col):
            cell = grid.cells[data_row][col_idx]
            if cell is None:
                continue
            # Use original position as key to avoid duplicates
            cell_key = (cell.original_row, cell.original_col)
            if cell_key in seen_cells:
                continue
            seen_cells.add(cell_key)

            if cell.text:
                headers.append(cell.text)

        return headers

    def _fuzzy_list_match(self, expected: list[str], actual: list[str], threshold: float = 0.8) -> tuple[bool, str]:
        """Check if two lists match using fuzzy matching."""
        if len(expected) != len(actual):
            return (
                False,
                f"Length mismatch: expected {len(expected)} headers, got {len(actual)}",
            )

        for i, (exp, act) in enumerate(zip(expected, actual, strict=False)):
            exp_norm = normalize_text(exp)
            act_norm = normalize_text(act)
            similarity = fuzz.ratio(exp_norm, act_norm) / 100.0
            if similarity < threshold:
                return (
                    False,
                    f"Header {i} mismatch: expected '{exp}', got '{act}' (similarity: {similarity:.2f})",
                )

        return True, ""

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if data cell has correct header chain."""
        grids = find_all_html_tables(content)
        if not grids:
            return False, "No HTML tables found in content"

        # Step 1: Find the correct table using anchor cells if provided
        if self.table_anchor_cells:
            anchor_result = find_table_by_anchors(grids, self.table_anchor_cells)
            if anchor_result.grid is not None:
                grids = [anchor_result.grid]
            elif anchor_result.is_ambiguous:
                return False, (
                    f"[AMBIGUOUS ANCHORS] Anchors matched {anchor_result.num_candidates} "
                    f"tables - could not uniquely identify target table"
                )
            else:
                return False, "Table anchor cells not found in any table"

        match = find_cell_in_grids(grids, self.data_cell)
        if not match:
            return False, f"Data cell '{self.data_cell}' not found in target table"

        grid, cell, row_idx, col_idx = match

        errors = []

        # Check column headers if expected
        if self.column_headers:
            actual_col_headers = self._get_column_headers(grid, row_idx, col_idx)
            passed, err = self._fuzzy_list_match(self.column_headers, actual_col_headers)
            if not passed:
                errors.append(f"Column headers: {err}. Expected: {self.column_headers}, Got: {actual_col_headers}")

        # Check row headers if expected
        if self.row_headers:
            actual_row_headers = self._get_row_headers(grid, row_idx, col_idx)
            passed, err = self._fuzzy_list_match(self.row_headers, actual_row_headers)
            if not passed:
                errors.append(f"Row headers: {err}. Expected: {self.row_headers}, Got: {actual_row_headers}")

        if errors:
            return False, "; ".join(errors)
        else:
            return True, f"Header chain verified for '{self.data_cell}'"


# =============================================================================
# Table Adjacency and Header Rules
# =============================================================================


class TableAdjacentRule(ParseTestRule):
    """
    Base class for table adjacency rules.

    Tests that anchor_cell has expected_neighbor in a specific direction.
    Handles duplicate anchor cells by checking ALL occurrences.
    """

    def __init__(self, rule_data: AdjacentTableRuleData | dict):
        super().__init__(rule_data)
        rule_data = cast(AdjacentTableRuleData, self._rule_data)

        self.anchor_cell = normalize_text(rule_data.anchor_cell)
        self.expected_neighbor = normalize_text(rule_data.expected_neighbor)
        self.table_anchor_cells = rule_data.table_anchor_cells
        self.direction = ""  # Set by subclass

        if not self.anchor_cell:
            raise ValueError("anchor_cell must be provided")
        if not self.expected_neighbor:
            raise ValueError("expected_neighbor must be provided")

    def _get_neighbor_position(self, row: int, col: int, grid: ResolvedGrid) -> tuple[int, int] | None:
        """Get neighbor position based on direction."""
        if self.direction == "up" and row > 0:
            return (row - 1, col)
        elif self.direction == "down" and row < grid.num_rows - 1:
            return (row + 1, col)
        elif self.direction == "left" and col > 0:
            return (row, col - 1)
        elif self.direction == "right" and col < grid.num_cols - 1:
            return (row, col + 1)
        return None

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        grids = find_all_html_tables(content)
        if not grids:
            return False, "No HTML tables found"

        # Step 1: Find the correct table using anchor cells if provided
        if self.table_anchor_cells:
            anchor_result = find_table_by_anchors(grids, self.table_anchor_cells)
            if anchor_result.grid is not None:
                grids = [anchor_result.grid]
            elif anchor_result.is_ambiguous:
                return False, (
                    f"[AMBIGUOUS ANCHORS] Anchors matched {anchor_result.num_candidates} "
                    f"tables - could not uniquely identify target table"
                )
            else:
                return False, "Table anchor cells not found in any table"

        for grid in grids:
            for row_idx, row in enumerate(grid.cells):
                for col_idx, cell in enumerate(row):
                    if cell is None:
                        continue
                    if cell.original_row != row_idx or cell.original_col != col_idx:
                        continue

                    similarity = fuzz.ratio(self.anchor_cell, cell.text) / 100.0
                    if similarity < CELL_FUZZY_MATCH_THRESHOLD:
                        continue

                    neighbor_pos = self._get_neighbor_position(row_idx, col_idx, grid)
                    if neighbor_pos is None:
                        continue

                    neighbor = grid.cells[neighbor_pos[0]][neighbor_pos[1]]
                    if neighbor is None:
                        continue

                    neighbor_sim = fuzz.ratio(self.expected_neighbor, neighbor.text) / 100.0
                    if neighbor_sim >= CELL_FUZZY_MATCH_THRESHOLD:
                        return True, ""

        return False, f"No '{self.anchor_cell}' has '{self.expected_neighbor}' {self.direction}"


class TableAdjacentUpRule(TableAdjacentRule):
    def __init__(self, rule_data: ParseTableAdjacentUpRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableAdjacentUpRule, self._rule_data)
        if self.type != ParseRuleType.TABLE_ADJACENT_UP.value:
            raise ValueError(f"Invalid type: {self.type}")
        self.direction = "up"


class TableAdjacentDownRule(TableAdjacentRule):
    def __init__(self, rule_data: ParseTableAdjacentDownRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableAdjacentDownRule, self._rule_data)
        if self.type != ParseRuleType.TABLE_ADJACENT_DOWN.value:
            raise ValueError(f"Invalid type: {self.type}")
        self.direction = "down"


class TableAdjacentLeftRule(TableAdjacentRule):
    def __init__(self, rule_data: ParseTableAdjacentLeftRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableAdjacentLeftRule, self._rule_data)
        if self.type != ParseRuleType.TABLE_ADJACENT_LEFT.value:
            raise ValueError(f"Invalid type: {self.type}")
        self.direction = "left"


class TableAdjacentRightRule(TableAdjacentRule):
    def __init__(self, rule_data: ParseTableAdjacentRightRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableAdjacentRightRule, self._rule_data)
        if self.type != ParseRuleType.TABLE_ADJACENT_RIGHT.value:
            raise ValueError(f"Invalid type: {self.type}")
        self.direction = "right"


class TableTopHeaderRule(ParseTestRule):
    """
    Test that a data cell has a specific column header above it.

    Handles duplicate data cells by checking ALL occurrences.
    """

    def __init__(self, rule_data: ParseTableTopHeaderRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableTopHeaderRule, self._rule_data)

        if self.type != ParseRuleType.TABLE_TOP_HEADER.value:
            raise ValueError(f"Invalid type: {self.type}")
        self.data_cell = normalize_text(rule_data.data_cell)
        self.expected_header = normalize_text(rule_data.expected_header)
        self.table_anchor_cells = rule_data.table_anchor_cells

        if not self.data_cell:
            raise ValueError("data_cell must be provided")
        if not self.expected_header:
            raise ValueError("expected_header must be provided")

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        grids = find_all_html_tables(content)
        if not grids:
            return False, "No HTML tables found"

        # Step 1: Find the correct table using anchor cells if provided
        if self.table_anchor_cells:
            anchor_result = find_table_by_anchors(grids, self.table_anchor_cells)
            if anchor_result.grid is not None:
                grids = [anchor_result.grid]
            elif anchor_result.is_ambiguous:
                return False, (
                    f"[AMBIGUOUS ANCHORS] Anchors matched {anchor_result.num_candidates} "
                    f"tables - could not uniquely identify target table"
                )
            else:
                return False, "Table anchor cells not found in any table"

        for grid in grids:
            for row_idx, row in enumerate(grid.cells):
                for col_idx, cell in enumerate(row):
                    if cell is None:
                        continue
                    if cell.original_row != row_idx or cell.original_col != col_idx:
                        continue

                    similarity = fuzz.ratio(self.data_cell, cell.text) / 100.0
                    if similarity < CELL_FUZZY_MATCH_THRESHOLD:
                        continue

                    # Look above for header
                    for header_row in range(row_idx):
                        header_cell = grid.cells[header_row][col_idx]
                        if header_cell is None:
                            continue

                        header_sim = fuzz.ratio(self.expected_header, header_cell.text) / 100.0
                        if header_sim >= CELL_FUZZY_MATCH_THRESHOLD:
                            return True, ""

        return False, f"No '{self.data_cell}' has header '{self.expected_header}' above"


class TableLeftHeaderRule(ParseTestRule):
    """
    Test that a data cell has a specific row header to its left.

    Handles duplicate data cells by checking ALL occurrences.
    """

    def __init__(self, rule_data: ParseTableLeftHeaderRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableLeftHeaderRule, self._rule_data)

        if self.type != ParseRuleType.TABLE_LEFT_HEADER.value:
            raise ValueError(f"Invalid type: {self.type}")
        self.data_cell = normalize_text(rule_data.data_cell)
        self.expected_header = normalize_text(rule_data.expected_header)
        self.table_anchor_cells = rule_data.table_anchor_cells

        if not self.data_cell:
            raise ValueError("data_cell must be provided")
        if not self.expected_header:
            raise ValueError("expected_header must be provided")

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        grids = find_all_html_tables(content)
        if not grids:
            return False, "No HTML tables found"

        # Step 1: Find the correct table using anchor cells if provided
        if self.table_anchor_cells:
            anchor_result = find_table_by_anchors(grids, self.table_anchor_cells)
            if anchor_result.grid is not None:
                grids = [anchor_result.grid]
            elif anchor_result.is_ambiguous:
                return False, (
                    f"[AMBIGUOUS ANCHORS] Anchors matched {anchor_result.num_candidates} "
                    f"tables - could not uniquely identify target table"
                )
            else:
                return False, "Table anchor cells not found in any table"

        for grid in grids:
            for row_idx, row in enumerate(grid.cells):
                for col_idx, cell in enumerate(row):
                    if cell is None:
                        continue
                    if cell.original_row != row_idx or cell.original_col != col_idx:
                        continue

                    similarity = fuzz.ratio(self.data_cell, cell.text) / 100.0
                    if similarity < CELL_FUZZY_MATCH_THRESHOLD:
                        continue

                    # Look left for header
                    for header_col in range(col_idx):
                        header_cell = grid.cells[row_idx][header_col]
                        if header_cell is None:
                            continue

                        header_sim = fuzz.ratio(self.expected_header, header_cell.text) / 100.0
                        if header_sim >= CELL_FUZZY_MATCH_THRESHOLD:
                            return True, ""

        return False, f"No '{self.data_cell}' has header '{self.expected_header}' to left"


# =============================================================================
# Table Border Rules (Negative Tests)
# =============================================================================


class TableNoBorderRule(ParseTestRule):
    """
    Base class for table border rules that verify absence of cells.

    These are "negative tests" that ensure predicted tables don't have
    extra rows/columns beyond the ground truth boundaries.
    """

    def __init__(self, rule_data: NoBorderTableRuleData | dict):
        super().__init__(rule_data)
        rule_data = cast(NoBorderTableRuleData, self._rule_data)

        self.cell = normalize_text(rule_data.cell)
        self.table_anchor_cells = rule_data.table_anchor_cells
        self.direction = ""  # Set by subclass: "left", "right", "up", "down"

        if not self.cell:
            raise ValueError("cell must be provided")

    def _get_neighbor_position(self, row: int, col: int, grid: ResolvedGrid) -> tuple[int, int] | None:
        """Get neighbor position based on direction. Returns None if out of bounds."""
        if self.direction == "up":
            return (row - 1, col) if row > 0 else None
        elif self.direction == "down":
            return (row + 1, col) if row < grid.num_rows - 1 else None
        elif self.direction == "left":
            return (row, col - 1) if col > 0 else None
        elif self.direction == "right":
            return (row, col + 1) if col < grid.num_cols - 1 else None
        return None

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        grids = find_all_html_tables(content)
        if not grids:
            return False, "No HTML tables found"

        # Find the correct table using anchor cells if provided
        if self.table_anchor_cells:
            anchor_result = find_table_by_anchors(grids, self.table_anchor_cells)
            if anchor_result.grid is not None:
                grids = [anchor_result.grid]
            elif anchor_result.is_ambiguous:
                return False, (
                    f"[AMBIGUOUS ANCHORS] Anchors matched {anchor_result.num_candidates} "
                    f"tables - could not uniquely identify target table"
                )
            else:
                return False, "Table anchor cells not found in any table"

        for grid in grids:
            for row_idx, row in enumerate(grid.cells):
                for col_idx, cell in enumerate(row):
                    if cell is None:
                        continue
                    if cell.original_row != row_idx or cell.original_col != col_idx:
                        continue

                    similarity = fuzz.ratio(self.cell, cell.text) / 100.0
                    if similarity < CELL_FUZZY_MATCH_THRESHOLD:
                        continue

                    # Found the cell - now check if there's NO neighbor in the direction
                    neighbor_pos = self._get_neighbor_position(row_idx, col_idx, grid)

                    if neighbor_pos is None:
                        # No neighbor position possible (at grid boundary) - PASS
                        return True, ""

                    neighbor = grid.cells[neighbor_pos[0]][neighbor_pos[1]]
                    if neighbor is None or not neighbor.text.strip():
                        # No neighbor cell or empty cell - PASS
                        return True, ""

                    # There IS a neighbor - this is a FAILURE for border tests
                    return (
                        False,
                        f"Cell '{self.cell}' has unexpected neighbor '{neighbor.text}' to {self.direction}",
                    )

        return False, f"Could not find cell '{self.cell}' in any table"


class TableNoLeftRule(TableNoBorderRule):
    """Test that a cell has no cell to its left (leftmost column boundary)."""

    def __init__(self, rule_data: ParseTableNoLeftRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableNoLeftRule, self._rule_data)
        if self.type != ParseRuleType.TABLE_NO_LEFT.value:
            raise ValueError(f"Invalid type: {self.type}")
        self.direction = "left"


class TableNoRightRule(TableNoBorderRule):
    """Test that a cell has no cell to its right (rightmost column boundary)."""

    def __init__(self, rule_data: ParseTableNoRightRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableNoRightRule, self._rule_data)
        if self.type != ParseRuleType.TABLE_NO_RIGHT.value:
            raise ValueError(f"Invalid type: {self.type}")
        self.direction = "right"


class TableNoAboveRule(TableNoBorderRule):
    """Test that a cell has no cell above it (top row boundary)."""

    def __init__(self, rule_data: ParseTableNoAboveRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableNoAboveRule, self._rule_data)
        if self.type != ParseRuleType.TABLE_NO_ABOVE.value:
            raise ValueError(f"Invalid type: {self.type}")
        self.direction = "up"


class TableNoBelowRule(TableNoBorderRule):
    """Test that a cell has no cell below it (bottom row boundary)."""

    def __init__(self, rule_data: ParseTableNoBelowRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTableNoBelowRule, self._rule_data)
        if self.type != ParseRuleType.TABLE_NO_BELOW.value:
            raise ValueError(f"Invalid type: {self.type}")
        self.direction = "down"
