"""Structural consistency metric for HTML table comparison.

A binary metric that flags when a predicted table has an inconsistent
internal structure — specifically, when rows or columns have an
inconsistent number of cells (after resolving colspan/rowspan).

This is a *self-consistency* check on the predicted table alone (not a
comparison against ground truth). A structurally consistent table has
every row spanning the same total number of columns and every column
spanning the same total number of rows.

Returns 1.0 (consistent) or 0.0 (inconsistent) per table, averaged
across all tables in the document.
"""

from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup

from extract_bench.evaluation.metrics.base import Metric
from extract_bench.evaluation.metrics.parse.table_extraction import extract_html_tables
from extract_bench.schemas.evaluation import MetricValue


def _mode_value(counts: list[int]) -> int:
    """Return the modal (most common) value from *counts*.

    Ties are broken by first occurrence, matching the platform logic.
    """
    if not counts:
        return 0
    freq: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    for i, c in enumerate(counts):
        freq[c] = freq.get(c, 0) + 1
        if c not in first_seen:
            first_seen[c] = i

    best_count = 0
    best_val = counts[0]
    for val, f in freq.items():
        if f > best_count or (f == best_count and first_seen[val] < first_seen[best_val]):
            best_count = f
            best_val = val
    return best_val


def _check_table_consistency(table_html: str) -> dict[str, Any]:
    """Check structural consistency of a single HTML table.

    Returns a dict with:
        - consistent: bool
        - num_rows: int
        - num_cols: int (max column span across all rows)
        - row_cell_counts: list[int] — effective column count per row
        - col_cell_counts: list[int] — effective row count per column
        - row_inconsistency: bool — True if rows have different widths
        - col_inconsistency: bool — True if columns have different heights
        - row_inconsistency_details: list[str] — per-row mismatch descriptions
        - col_inconsistency_details: list[str] — per-column mismatch descriptions
    """
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table")
    if not table:
        return {"consistent": True, "num_rows": 0, "num_cols": 0}

    rows = table.find_all("tr")
    if not rows:
        return {"consistent": True, "num_rows": 0, "num_cols": 0}

    num_rows = len(rows)

    # Build an occupancy grid by resolving rowspan/colspan
    occupied: dict[tuple[int, int], bool] = {}

    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row.find_all(["td", "th"]):
            while (row_idx, col_idx) in occupied:
                col_idx += 1

            rowspan = int(str(cell.get("rowspan", "1")))
            colspan = int(str(cell.get("colspan", "1")))

            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    occupied[(r, c)] = True

            col_idx += colspan

    if not occupied:
        return {"consistent": True, "num_rows": num_rows, "num_cols": 0}

    max_row = max(r for r, c in occupied) + 1
    max_col = max(c for r, c in occupied) + 1

    # Count how many columns each row spans
    row_cell_counts = []
    for r in range(max_row):
        count = sum(1 for c in range(max_col) if (r, c) in occupied)
        row_cell_counts.append(count)

    # Count how many rows each column spans
    col_cell_counts = []
    for c in range(max_col):
        count = sum(1 for r in range(max_row) if (r, c) in occupied)
        col_cell_counts.append(count)

    # Check consistency: all rows should have the same width,
    # all columns should have the same height
    row_inconsistent = len(set(row_cell_counts)) > 1
    col_inconsistent = len(set(col_cell_counts)) > 1

    consistent = not row_inconsistent and not col_inconsistent

    # Build per-row/col inconsistency details using the modal expected value
    row_inconsistency_details: list[str] = []
    col_inconsistency_details: list[str] = []

    if row_inconsistent:
        expected_cols = _mode_value(row_cell_counts)
        for i, count in enumerate(row_cell_counts):
            if count != expected_cols:
                row_inconsistency_details.append(f"row {i + 1} has {count} cols, expected {expected_cols}")

    if col_inconsistent:
        expected_rows = _mode_value(col_cell_counts)
        for i, count in enumerate(col_cell_counts):
            if count != expected_rows:
                col_inconsistency_details.append(f"col {i + 1} has {count} rows, expected {expected_rows}")

    return {
        "consistent": consistent,
        "num_rows": max_row,
        "num_cols": max_col,
        "row_cell_counts": row_cell_counts,
        "col_cell_counts": col_cell_counts,
        "row_inconsistency": row_inconsistent,
        "col_inconsistency": col_inconsistent,
        "row_inconsistency_details": row_inconsistency_details,
        "col_inconsistency_details": col_inconsistency_details,
    }


class StructuralConsistencyMetric(Metric):
    """Binary structural consistency metric for predicted HTML tables.

    Checks that each predicted table is internally consistent: every row
    spans the same number of columns and every column spans the same
    number of rows (after resolving colspan/rowspan).

    Only evaluates the *actual* (predicted) tables. Ground truth is used
    only for table matching (so we report per-table diagnostics aligned
    with the GT table order).

    Returns a single MetricValue with value 1.0 (all tables consistent)
    or 0.0 (at least one inconsistent), with per-table details in metadata.
    """

    @property
    def name(self) -> str:
        return "structural_consistency"

    def compute(  # type: ignore[override]
        self,
        expected: str,
        actual: str,
        **kwargs: Any,
    ) -> list[MetricValue]:
        actual_tables = extract_html_tables(actual)

        if not actual_tables:
            return [
                MetricValue(
                    metric_name="structural_consistency",
                    value=1.0,
                    metadata={"tables_found_actual": 0, "note": "No tables to check"},
                )
            ]

        per_table: list[dict[str, Any]] = []
        scores: list[float] = []
        details: list[str] = []

        for idx, table_html in enumerate(actual_tables):
            result = _check_table_consistency(table_html)
            score = 1.0 if result["consistent"] else 0.0
            scores.append(score)
            per_table.append(
                {
                    "table_index": idx,
                    "consistent": result["consistent"],
                    "num_rows": result["num_rows"],
                    "num_cols": result["num_cols"],
                    "row_inconsistency": result.get("row_inconsistency", False),
                    "col_inconsistency": result.get("col_inconsistency", False),
                }
            )

            nr = result["num_rows"]
            nc = result["num_cols"]

            if idx > 0:
                details.append("=" * 40)

            if result["consistent"]:
                details.append(f"Table {idx + 1}: 1.0 — consistent ({nr}×{nc})")
            else:
                row_details: list[str] = result.get("row_inconsistency_details", [])
                col_details: list[str] = result.get("col_inconsistency_details", [])
                issues = row_details + col_details
                row_counts = result.get("row_cell_counts", [])
                col_counts = result.get("col_cell_counts", [])
                n_bad_rows = len(row_details)
                n_bad_cols = len(col_details)
                summary = f"Table {idx + 1}: 0.0 — inconsistent ({nr}×{nc})"
                if n_bad_rows:
                    summary += f" {n_bad_rows}/{nr} rows"
                if n_bad_cols:
                    summary += f" {n_bad_cols}/{nc} cols"
                details.append(summary)
                if row_counts:
                    details.append(f"  row widths:  {row_counts}")
                if col_counts:
                    details.append(f"  col heights: {col_counts}")
                for issue in issues:
                    details.append(f"  {issue}")

        avg_score = sum(scores) / len(scores)
        n_ok = sum(1 for s in scores if s == 1.0)
        n_bad = sum(1 for s in scores if s == 0.0)
        details.insert(
            0,
            f"{avg_score:.3f} — {len(actual_tables)} table(s) checked, {n_ok} consistent, {n_bad} inconsistent",
        )

        return [
            MetricValue(
                metric_name="structural_consistency",
                value=avg_score,
                metadata={
                    "tables_found_actual": len(actual_tables),
                    "tables_consistent": n_ok,
                    "tables_inconsistent": n_bad,
                    "per_table_details": per_table,
                },
                details=details,
            )
        ]
