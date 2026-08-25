"""Table parsing utilities for parse evaluation.

Ported from OlmOCR bench table parsing logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from bs4 import BeautifulSoup, Tag
from rapidfuzz import fuzz

from extract_bench.evaluation.metrics.parse.utils import normalize_text

# Mappings from ASCII digits to Unicode super-/subscript equivalents.
# Used so that ``<sup>1</sup>`` is stored as ``¹`` — the same codepoint a
# provider might already emit — and ``normalize_text`` can strip both
# representations uniformly.
_SUPERSCRIPT_DIGITS = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
_SUBSCRIPT_DIGITS = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
_ASCII_TO_SUPERSCRIPT = dict(zip("0123456789", _SUPERSCRIPT_DIGITS, strict=True))
_ASCII_TO_SUBSCRIPT = dict(zip("0123456789", _SUBSCRIPT_DIGITS, strict=True))


def _sup_sub_to_unicode(cell: Tag) -> None:
    """Convert ``<sup>``/``<sub>`` digit content to Unicode equivalents.

    ASCII digits inside the tags are mapped to their Unicode super-/subscript
    codepoints (``1`` → ``¹``).  Non-digit characters (parens, letters,
    whitespace) are preserved as-is — e.g. ``<sup>(2)</sup>`` becomes
    ``(²)`` rather than being silently dropped to ``²``, which would glue
    a footnote digit onto the preceding number.  After this call,
    ``cell.get_text()`` will contain Unicode super-/subscript digits in
    place of bare ASCII digits that were hidden inside markup.
    """
    for tag in cell.find_all("sup"):
        text = tag.get_text()
        converted = "".join(_ASCII_TO_SUPERSCRIPT.get(c, c) for c in text)
        tag.replace_with(converted)
    for tag in cell.find_all("sub"):
        text = tag.get_text()
        converted = "".join(_ASCII_TO_SUBSCRIPT.get(c, c) for c in text)
        tag.replace_with(converted)


@dataclass
class TableData:
    """Class to hold table data and metadata about headers."""

    data: np.ndarray  # The actual table data
    header_rows: set[int] = field(default_factory=set)  # Indices of rows that are headers
    header_cols: set[int] = field(default_factory=set)  # Indices of columns that are headers
    col_headers: dict = field(default_factory=dict)  # Maps column index to header text, handling colspan
    row_headers: dict = field(default_factory=dict)  # Maps row index to header text, handling rowspan
    # Grid cells that originate from a <th> element (including all cells
    # covered by colspan/rowspan expansion).  This lets downstream code
    # answer "is (row, col) from a <th>?" without conflating span expansion
    # with hierarchical header levels in col_headers.
    header_cells: set[tuple[int, int]] = field(default_factory=set)
    context_before: str = field(default="")  # Text before table (for chart titles)
    context_after: str = field(default="")  # Text after table (for captions)


def _process_table_lines(table_lines: list[str]) -> list[list[str]]:
    """
    Process a list of lines that potentially form a markdown table.

    Args:
        table_lines: List of strings, each representing a line in a potential markdown table

    Returns:
        A list of rows, each a list of cell values
    """
    table_data = []
    separator_row_index = None

    # First, identify the separator row (the row with dashes)
    for i, line in enumerate(table_lines):
        # Check if this looks like a separator row (contains mostly dashes)
        content_without_pipes = line.replace("|", "").strip()
        if content_without_pipes and all(c in "- :" for c in content_without_pipes):
            separator_row_index = i
            break

    # Process each line, filtering out the separator row
    for i, line in enumerate(table_lines):
        # Skip the separator row
        if i == separator_row_index:
            continue

        # Skip lines that are entirely formatting
        if line.strip() and all(c in "- :|" for c in line):
            continue

        # Process the cells in this row
        cells = [cell.strip() for cell in line.split("|")]

        # Remove empty cells at the beginning and end (caused by leading/trailing pipes)
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]

        if cells:  # Only add non-empty rows
            table_data.append(cells)

    return table_data


def parse_markdown_tables(md_content: str) -> list[TableData]:
    """
    Extract and parse all markdown tables from the provided content.

    Args:
        md_content: The markdown content containing tables

    Returns:
        A list of TableData objects, each containing the table data and header information
    """
    # Split the content into lines and process line by line
    lines = md_content.strip().split("\n")

    parsed_tables = []
    current_table_lines = []
    table_start_line = -1
    in_table = False

    # Identify potential tables by looking for lines with pipe characters
    for line_idx, line in enumerate(lines):
        # Check if this line has pipe characters (a table row indicator)
        if "|" in line:
            # If we weren't in a table before, start a new one
            if not in_table:
                in_table = True
                table_start_line = line_idx
                current_table_lines = [line]
            else:
                # Continue adding to the current table
                current_table_lines.append(line)
        else:
            # No pipes in this line, so if we were in a table, we've reached its end
            if in_table:
                table_end_line = line_idx
                # Process the completed table if it has at least 2 rows
                if len(current_table_lines) >= 2:
                    table_data = _process_table_lines(current_table_lines)
                    if table_data and len(table_data) > 0:
                        # Convert to numpy array for easier manipulation
                        max_cols = max(len(row) for row in table_data)
                        padded_data = [row + [""] * (max_cols - len(row)) for row in table_data]
                        table_array = np.array(padded_data, dtype=object)

                        # In markdown tables, the first row is typically a header row
                        header_rows = {0} if len(table_array) > 0 else set()

                        # Set up col_headers with first row headers for each column
                        col_headers = {}
                        if len(table_array) > 0:
                            for col_idx in range(table_array.shape[1]):
                                if col_idx < len(table_array[0]):
                                    col_headers[col_idx] = [(0, str(table_array[0, col_idx]))]

                        # Set up row_headers with first column headers for each row
                        row_headers = {}
                        if table_array.shape[1] > 0:
                            # Skip header row
                            for row_idx in range(1, table_array.shape[0]):
                                # First column as heading
                                row_headers[row_idx] = [(0, str(table_array[row_idx, 0]))]

                        # Extract context (up to 5 lines before, 2 lines after)
                        context_before = "\n".join(lines[max(0, table_start_line - 5) : table_start_line])
                        context_after = "\n".join(lines[table_end_line : min(len(lines), table_end_line + 2)])

                        # Create TableData object
                        parsed_tables.append(
                            TableData(
                                data=table_array,
                                header_rows=header_rows,
                                # First column as header
                                header_cols={0} if table_array.shape[1] > 0 else set(),
                                col_headers=col_headers,
                                row_headers=row_headers,
                                context_before=context_before,
                                context_after=context_after,
                            )
                        )
                in_table = False

    # Process the last table if we're still tracking one at the end of the file
    if in_table and len(current_table_lines) >= 2:
        table_end_line = len(lines)
        table_data = _process_table_lines(current_table_lines)
        if table_data and len(table_data) > 0:
            # Convert to numpy array
            max_cols = max(len(row) for row in table_data)
            padded_data = [row + [""] * (max_cols - len(row)) for row in table_data]
            table_array = np.array(padded_data, dtype=object)

            # In markdown tables, the first row is typically a header row
            header_rows = {0} if len(table_array) > 0 else set()

            # Set up col_headers with first row headers for each column
            col_headers = {}
            if len(table_array) > 0:
                for col_idx in range(table_array.shape[1]):
                    if col_idx < len(table_array[0]):
                        col_headers[col_idx] = [(0, str(table_array[0, col_idx]))]

            # Set up row_headers with first column headers for each row
            row_headers = {}
            if table_array.shape[1] > 0:
                # Skip header row
                for row_idx in range(1, table_array.shape[0]):
                    # First column as heading
                    row_headers[row_idx] = [(0, str(table_array[row_idx, 0]))]

            # Extract context (up to 5 lines before, 2 lines after)
            context_before = "\n".join(lines[max(0, table_start_line - 5) : table_start_line])
            context_after = "\n".join(lines[table_end_line : min(len(lines), table_end_line + 2)])

            # Create TableData object
            parsed_tables.append(
                TableData(
                    data=table_array,
                    header_rows=header_rows,
                    # First column as header
                    header_cols={0} if table_array.shape[1] > 0 else set(),
                    col_headers=col_headers,
                    row_headers=row_headers,
                    context_before=context_before,
                    context_after=context_after,
                )
            )

    return parsed_tables


def parse_html_tables(html_content: str) -> list[TableData]:
    """
    Extract and parse all HTML tables from the provided content.
    Identifies header rows and columns, and maps them properly handling rowspan/colspan.

    Args:
        html_content: The HTML content containing tables

    Returns:
        A list of TableData objects, each containing the table data and header information
    """
    soup = BeautifulSoup(html_content, "lxml")
    all_tables = soup.find_all("table")

    # Filter to top-level tables (skip tables nested inside other tables)
    top_level_tables = []
    for t in all_tables:
        if not any(p.name == "table" for p in t.parents if hasattr(p, "name")):
            top_level_tables.append(t)

    parsed_tables = []

    for table in top_level_tables:
        # Replace nested tables with their text content so their <tr>
        # elements don't leak into the outer table's row list
        for nested in table.find_all("table"):
            nested.replace_with(nested.get_text(" ", strip=True))

        rows = table.find_all(["tr"])

        # Extract <caption> text if present (used as chart title context)
        caption_elem = table.find("caption")
        caption_text = caption_elem.get_text(strip=True) if caption_elem else ""

        header_rows = set()
        header_cols = set()
        # Maps column index to all header cells above it
        col_headers: dict[int, list[tuple[int, str]]] = {}
        # Maps row index to all header cells to its left
        row_headers: dict[int, list[tuple[int, str]]] = {}

        # Find rows inside thead tags - these are definitely header rows
        thead = table.find("thead")
        if thead:
            thead_rows = thead.find_all("tr")
            for tr in thead_rows:
                if tr in rows:
                    header_rows.add(rows.index(tr))

        # Initialize a grid to track filled cells due to rowspan/colspan
        cell_grid = {}
        header_cells: set[tuple[int, int]] = set()
        col_span_info = {}  # Tracks which columns contain headers
        row_span_info = {}  # Tracks which rows contain headers

        # First pass: process each row to build the raw table data and identify headers
        for row_idx, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            col_idx = 0

            for cell in cells:
                # Skip cells that are already filled by rowspan/colspan
                while (row_idx, col_idx) in cell_grid:
                    col_idx += 1

                # Get cell text — replace <br> with space before extraction
                # so that tag boundaries don't merge adjacent words
                # (mirrors the approach used in _resolve_table at line ~451)
                for br in cell.find_all("br"):
                    br.replace_with(" ")
                # Convert <sup>/<sub> digit content to Unicode equivalents
                # so that "Name<sup>1</sup>" becomes "Name¹", matching the
                # representation when sources already use Unicode superscripts.
                _sup_sub_to_unicode(cell)
                cell_text = cell.get_text().strip()

                # Check if this is a header cell
                is_header = cell.name == "th"
                if is_header:
                    header_rows.add(row_idx)
                    header_cols.add(col_idx)
                    col_span_info[col_idx] = True
                    row_span_info[row_idx] = True

                # Get rowspan and colspan
                rowspan = int(cell.get("rowspan", 1))  # type: ignore[arg-type]
                colspan = int(cell.get("colspan", 1))  # type: ignore[arg-type]

                # Fill the grid for all cells spanned by this cell
                for r in range(row_idx, row_idx + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        cell_grid[(r, c)] = cell_text
                        if is_header:
                            header_cells.add((r, c))

                # Update col_headers and row_headers if this is a header
                if is_header:
                    # Add to col_headers for all columns this cell spans
                    for c in range(col_idx, col_idx + colspan):
                        if c not in col_headers:
                            col_headers[c] = []
                        col_headers[c].append((row_idx, cell_text))

                    # Add to row_headers for all rows this cell spans
                    for r in range(row_idx, row_idx + rowspan):
                        if r not in row_headers:
                            row_headers[r] = []
                        row_headers[r].append((col_idx, cell_text))

                col_idx += colspan

        if cell_grid:
            max_grid_row = max(r for r, c in cell_grid) + 1
            max_grid_col = max(c for r, c in cell_grid) + 1
            table_array = np.empty((max_grid_row, max_grid_col), dtype=object)
            table_array[:] = ""
            for (r, c), text in cell_grid.items():
                table_array[r, c] = text
        else:
            table_array = np.array([[]], dtype=object)

        # Extract context IMMEDIATELY before and after the table
        # Simple approach: just get the previous and next siblings up to next table
        context_before_parts = []  # type: ignore[var-annotated]
        context_after_parts = []

        # Get text IMMEDIATELY before table (up to 3 siblings or until we hit another table)
        prev = table.previous_sibling
        for _ in range(3):
            if prev is None:
                break
            # Stop if we hit another table
            if hasattr(prev, "name") and prev.name == "table":
                break
            # Get text content
            if hasattr(prev, "get_text"):
                text = prev.get_text(strip=True)
            elif isinstance(prev, str):
                text = prev.strip()
            else:
                text = ""
            if text:
                context_before_parts.insert(0, text)
            prev = prev.previous_sibling

        # Take last 300 chars (closest to table), not first 300
        full_context_before = " ".join(context_before_parts)
        context_before = full_context_before[-300:] if len(full_context_before) > 300 else full_context_before

        # Prepend caption to context_before (caption acts as chart title)
        if caption_text:
            caption_markup = f"<caption>{caption_text}</caption>"
            context_before = f"{caption_markup} {context_before}" if context_before else caption_markup

        # Get text IMMEDIATELY after table (up to 2 siblings or until we hit another table)
        next_elem = table.next_sibling
        for _ in range(2):
            if next_elem is None:
                break
            # Stop if we hit another table
            if hasattr(next_elem, "name") and next_elem.name == "table":
                break
            # Get text content
            if hasattr(next_elem, "get_text"):
                text = next_elem.get_text(strip=True)
            elif isinstance(next_elem, str):
                text = next_elem.strip()
            else:
                text = ""
            if text:
                context_after_parts.append(text)
            next_elem = next_elem.next_sibling

        context_after = " ".join(context_after_parts)[:200]

        # Create TableData object
        parsed_tables.append(
            TableData(
                data=table_array,
                header_rows=header_rows,
                header_cols=header_cols,
                col_headers=col_headers,
                row_headers=row_headers,
                header_cells=header_cells,
                context_before=context_before,
                context_after=context_after,
            )
        )

    return parsed_tables


# =============================================================================
# Grid-based table parsing for hierarchy tests
# =============================================================================


@dataclass
class ResolvedCell:
    """Metadata for a cell in the resolved grid."""

    text: str
    original_row: int
    original_col: int
    colspan: int
    rowspan: int
    is_header: bool  # True if <th> element


@dataclass
class ResolvedGrid:
    """A 2D grid of cells with span information resolved."""

    cells: list[list[ResolvedCell | None]]  # [row][col] -> ResolvedCell or None
    num_rows: int
    num_cols: int
    cell_positions: dict[str, list[tuple[int, int]]]  # text -> list of (row, col) positions


def resolve_html_table_grid(table_html: str) -> ResolvedGrid | None:
    """
    Parse an HTML table and resolve colspan/rowspan to a 2D grid.

    Args:
        table_html: HTML string containing a single table

    Returns:
        ResolvedGrid with cell positions and metadata, or None if no table found
    """
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table")
    if not table:
        return None

    rows = table.find_all("tr")
    if not rows:
        return None

    # First pass: determine grid dimensions
    max_cols = 0
    for row in rows:
        col_count = sum(int(cell.get("colspan", 1)) for cell in row.find_all(["td", "th"]))  # type: ignore[arg-type, misc]
        max_cols = max(max_cols, col_count)

    num_rows = len(rows)
    num_cols = max_cols

    if num_rows == 0 or num_cols == 0:
        return None

    # Initialize empty grid
    grid: list[list[ResolvedCell | None]] = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    cell_positions: dict[str, list[tuple[int, int]]] = {}

    # Second pass: fill the grid
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row.find_all(["td", "th"]):
            # Skip positions already filled by rowspan from above
            while col_idx < num_cols and grid[row_idx][col_idx] is not None:
                col_idx += 1

            if col_idx >= num_cols:
                break

            # Get cell properties
            colspan = int(cell.get("colspan", 1))  # type: ignore[arg-type]
            rowspan = int(cell.get("rowspan", 1))  # type: ignore[arg-type]
            is_header = cell.name == "th"

            # Get text, replacing <br> with newlines first
            for br in cell.find_all("br"):
                br.replace_with("\n")
            # Convert <sup>/<sub> digit content to Unicode equivalents
            _sup_sub_to_unicode(cell)
            text = cell.get_text().strip()
            text = normalize_text(text)

            # Create cell metadata
            resolved_cell = ResolvedCell(
                text=text,
                original_row=row_idx,
                original_col=col_idx,
                colspan=colspan,
                rowspan=rowspan,
                is_header=is_header,
            )

            # Fill grid positions for this cell
            for r_offset in range(rowspan):
                for c_offset in range(colspan):
                    target_row = row_idx + r_offset
                    target_col = col_idx + c_offset
                    if target_row < num_rows and target_col < num_cols:
                        grid[target_row][target_col] = resolved_cell
                        # Track positions by text
                        if text:
                            if text not in cell_positions:
                                cell_positions[text] = []
                            cell_positions[text].append((target_row, target_col))

            col_idx += colspan

    return ResolvedGrid(
        cells=grid,
        num_rows=num_rows,
        num_cols=num_cols,
        cell_positions=cell_positions,
    )


def find_all_html_tables(content: str) -> list[ResolvedGrid]:
    """
    Find and resolve all HTML tables in content.

    Args:
        content: HTML or markdown content containing tables

    Returns:
        List of ResolvedGrid objects
    """
    soup = BeautifulSoup(content, "lxml")
    tables = soup.find_all("table")

    grids = []
    for table in tables:
        grid = resolve_html_table_grid(str(table))
        if grid:
            grids.append(grid)

    return grids


class AnchorMatchResult:
    """Result of find_table_by_anchors with match quality info."""

    def __init__(
        self,
        grid: ResolvedGrid | None,
        status: str,
        num_candidates: int = 0,
    ):
        self.grid = grid
        # "unique" = one table matched uniquely
        # "ambiguous" = anchors found in multiple tables, no unique winner
        # "no_match" = anchors not found in any table
        self.status = status
        self.num_candidates = num_candidates

    @property
    def is_ambiguous(self) -> bool:
        return self.status == "ambiguous"


def find_table_by_anchors(
    grids: list[ResolvedGrid],
    anchor_cells: list[str],
    threshold: float = 0.8,
) -> AnchorMatchResult:
    """
    Find the table that contains ANY of the anchor cells uniquely.

    Strategy:
    1. For each anchor cell, find which tables contain it (using fuzzy matching)
    2. If an anchor appears in exactly ONE table, that's a strong signal
    3. Return the table with the most anchor matches

    This handles OCR errors: even if some anchors don't match, others will.

    Args:
        grids: List of ResolvedGrid objects to search
        anchor_cells: List of anchor cell texts that uniquely identify the target table
        threshold: Minimum similarity ratio (0-1) for fuzzy matching

    Returns:
        AnchorMatchResult with the matched grid (if any) and match quality status
    """
    if not anchor_cells or not grids:
        return AnchorMatchResult(None, "no_match")

    # Track votes: grid_index -> number of unique anchor matches
    table_votes: dict[int, int] = {}
    # Track all tables that contain any anchor (for ambiguity detection)
    all_matching_tables: set[int] = set()

    for anchor in anchor_cells:
        normalized_anchor = normalize_text(anchor)
        tables_with_anchor: list[int] = []

        for grid_idx, grid in enumerate(grids):
            found_in_grid = False
            for row in grid.cells:
                if found_in_grid:
                    break
                for cell in row:
                    if cell is None:
                        continue
                    similarity = fuzz.ratio(normalized_anchor, cell.text) / 100.0
                    if similarity >= threshold:
                        tables_with_anchor.append(grid_idx)
                        found_in_grid = True
                        break

        all_matching_tables.update(tables_with_anchor)

        # If this anchor appears in exactly ONE table, it's a strong signal
        unique_tables = set(tables_with_anchor)
        if len(unique_tables) == 1:
            matched_idx = unique_tables.pop()
            table_votes[matched_idx] = table_votes.get(matched_idx, 0) + 1

    if table_votes:
        best_table_idx = max(table_votes, key=lambda x: table_votes[x])
        return AnchorMatchResult(grids[best_table_idx], "unique")

    if all_matching_tables:
        return AnchorMatchResult(None, "ambiguous", len(all_matching_tables))

    return AnchorMatchResult(None, "no_match")


def merge_preceding_titles_into_tables(expected: str, actual: str) -> str:
    """Normalize predicted HTML by merging preceding text into tables as full-width title rows.

    When a ground-truth table starts with a single full-width colspan row (acting
    as a table title), but the predicted output has that same text as a heading or
    paragraph immediately before the ``<table>``, the predicted table will be
    missing that row and score lower on structural metrics.

    This function detects such cases and inserts the preceding text into the
    predicted table as a ``<tr><th colspan="...">title</th></tr>`` first row,
    so the two tables align structurally.

    Args:
        expected: Ground-truth markdown/HTML content.
        actual: Predicted markdown/HTML content to normalize.

    Returns:
        The ``actual`` string with preceding titles merged into tables where
        appropriate.
    """
    if not expected or not actual:
        return actual

    # --- Step 1: collect title texts from GT tables whose first row is full-width ---
    gt_titles: list[str] = []
    gt_soup = BeautifulSoup(expected, "lxml")
    for table in gt_soup.find_all("table"):
        first_row = table.find("tr")
        if first_row is None:
            continue
        cells = first_row.find_all(["th", "td"])
        if len(cells) != 1:
            continue
        cell = cells[0]
        colspan = int(cell.get("colspan", 1))  # type: ignore[arg-type]
        if colspan <= 1:
            continue
        # Verify this cell actually spans all columns by checking the next row
        second_row = first_row.find_next_sibling("tr")
        if second_row is None:
            continue
        second_row_col_count = sum(
            int(c.get("colspan", 1))  # type: ignore[arg-type, misc]
            for c in second_row.find_all(["th", "td"])
        )
        if colspan < second_row_col_count:
            continue
        title_text = cell.get_text(strip=True)
        if title_text:
            gt_titles.append(title_text)

    if not gt_titles:
        return actual

    # --- Step 2: for each predicted table, check if preceding text matches a GT title ---
    pred_soup = BeautifulSoup(actual, "lxml")
    modified = False

    for table in pred_soup.find_all("table"):
        # Skip tables that already start with a full-width row
        first_row = table.find("tr")
        if first_row is not None:
            first_cells = first_row.find_all(["th", "td"])
            if len(first_cells) == 1:
                first_colspan = int(first_cells[0].get("colspan", 1))  # type: ignore[arg-type]
                if first_colspan > 1:
                    continue  # Already has a title row

        # Collect preceding sibling text (headings, paragraphs, bare text)
        preceding_text = ""
        prev_elem = table.previous_sibling
        # Skip whitespace-only text nodes
        while prev_elem is not None and isinstance(prev_elem, str) and not prev_elem.strip():
            prev_elem = prev_elem.previous_sibling
        if prev_elem is not None:
            if hasattr(prev_elem, "get_text"):
                preceding_text = prev_elem.get_text(strip=True)
            elif isinstance(prev_elem, str):
                preceding_text = prev_elem.strip()

        if not preceding_text:
            continue

        # Fuzzy-match against GT titles
        normalized_preceding = normalize_text(preceding_text)
        best_match_ratio = 0.0
        for gt_title in gt_titles:
            normalized_gt = normalize_text(gt_title)
            ratio = fuzz.ratio(normalized_preceding, normalized_gt) / 100.0
            if ratio > best_match_ratio:
                best_match_ratio = ratio

        if best_match_ratio < 0.8:
            continue

        # Determine column count from the table's header/first data row
        col_count = 0
        for row in table.find_all("tr"):
            row_cols = sum(
                int(c.get("colspan", 1))  # type: ignore[arg-type, misc]
                for c in row.find_all(["th", "td"])
            )
            if row_cols > col_count:
                col_count = row_cols

        if col_count < 2:
            continue

        # Insert a new full-width title row at the top of the table
        new_row = pred_soup.new_tag("tr")
        new_th = pred_soup.new_tag("th", colspan=str(col_count))
        new_th.string = preceding_text
        new_row.append(new_th)

        # Insert into <thead> if it exists, otherwise at the start of the table
        thead = table.find("thead")
        if thead:
            thead.insert(0, new_row)
        else:
            # Insert before the first <tr>
            first_tr = table.find("tr")
            if first_tr:
                first_tr.insert_before(new_row)
            else:
                table.append(new_row)

        # Remove the preceding element that contained the title
        if prev_elem is not None:
            prev_elem.extract()

        modified = True

    if not modified:
        return actual

    # Serialize back to string — use the body content to avoid extra <html><body> wrapper
    body = pred_soup.find("body")
    if body:
        return body.decode_contents()
    return str(pred_soup)


def find_cell_in_grids(
    grids: list[ResolvedGrid],
    cell_text: str,
    threshold: float = 0.8,
) -> tuple[ResolvedGrid, ResolvedCell, int, int] | None:
    """
    Find a cell by text in a list of grids using fuzzy matching.

    Args:
        grids: List of ResolvedGrid objects to search
        cell_text: Text to search for
        threshold: Minimum similarity ratio (0-1)

    Returns:
        Tuple of (grid, cell, row_idx, col_idx) or None if not found
    """
    normalized_search = normalize_text(cell_text)
    best_match: tuple[ResolvedGrid, ResolvedCell, int, int] | None = None
    best_score = 0.0

    for grid in grids:
        for row_idx, row in enumerate(grid.cells):
            for col_idx, cell in enumerate(row):
                if cell is None:
                    continue
                # Only check original cell position to avoid duplicate checks
                if cell.original_row != row_idx or cell.original_col != col_idx:
                    continue

                similarity = fuzz.ratio(normalized_search, cell.text) / 100.0
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = (grid, cell, row_idx, col_idx)

    return best_match
