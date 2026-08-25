"""Generalized, domain-agnostic HTML table parser (Layer 1 of the IR).

This is the first half of the "document -> structured representation ->
target schema" generalization. It knows *nothing* about any particular vendor,
options grids, comparables, or checkboxes. Given the HTML emitted by a
parser (we feed it LlamaParse Agentic page text), it produces a faithful
*physical* view of every ``<table>`` on the page. All semantic
interpretation — orientation, un-pivoting, column-role assignment, type
coercion, "is this a total row" — is deliberately left to Layer 2 (the
dynamic mapper).

Design lineage
--------------
This is a deliberately simplified port of the table-parsing structures in
the parse metrics (``evaluation/metrics/parse/table_parsing.py``):
``ResolvedGrid`` / ``ResolvedCell`` below mirror the bench's grid resolver
(span-resolved physical grid with per-cell provenance). Everything the
bench file does *beyond* that — Hungarian column alignment, record
matching, title-row stripping, header promotion / demotion, cell
normalization (currency/comma/sup-sub) — is *scoring* concern and is
intentionally NOT ported. (A naive header-keyed records view existed
here once too, but generated scripts work from the raw grid, so it was
removed rather than risk presenting record semantics the scripts don't
use.) We keep cell text as raw strings (whitespace-collapsed, ``<br>``
-> space) with no value coercion.

One faithful output
-------------------
``Table.grid`` / ``text_grid()``: the dense R x C grid with
``rowspan``/``colspan`` resolved by fill, each cell carrying
``{text, row, col, rowspan, colspan, is_header}``. No orientation, no
records, no un-pivoting — generated scripts work from raw cell strings.

Header detection (``header_row_indices()``) is purely syntactic and
deliberately conservative: a row counts as a column-header row only if it
belongs to the leading block of rows whose every cell is a ``<th>`` —
the cell element name is the sole signal (``<thead>`` placement is not
consulted; a ``<td>`` row inside ``<thead>`` does not count). The first
row that contains a ``<td>`` ends the header block. This avoids the
degenerate case where a row-header table (``<th>`` label in column 0 of
every row) marks every row as a header and yields zero data rows — those
``<th>`` label cells simply remain ordinary grid cells.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

_WS_RE = re.compile(r"\s+")


def _clean_text(cell: Tag) -> str:
    """Cell element -> raw text. ``<br>`` becomes a space so adjacent
    lines don't glue together; whitespace is collapsed. No value
    normalization (currency, commas, case) — Layer 2's job."""
    for br in cell.find_all("br"):
        br.replace_with(" ")
    return _WS_RE.sub(" ", cell.get_text()).strip()


@dataclass
class Cell:
    """One logical cell with full span provenance.

    A cell that spans multiple grid positions is stored once and shared
    (by reference) across every position it covers; ``row``/``col`` are
    the span's top-left origin, so ``grid[r][c].row == r and .col == c``
    tests whether ``(r, c)`` is the cell's origin vs. a covered position.
    """

    text: str
    row: int  # origin (top-left) row of the span
    col: int  # origin (top-left) col of the span
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False  # came from a <th> element


@dataclass
class Table:
    """A single parsed table: faithful grid + provenance + context.

    ``grid`` is dense and span-resolved: ``grid[r][c]`` is the ``Cell``
    occupying that position (shared object for spanned cells), or ``None``
    only for ragged tails where a row had fewer cells than the widest row.
    """

    grid: list[list[Cell | None]]
    n_rows: int
    n_cols: int
    page_num: int | None = None
    table_index: int = 0  # index of this table within its page
    context_before: str = ""  # text immediately preceding the table (heading breadcrumb)
    context_after: str = ""  # text immediately following the table
    caption: str = ""

    # ---- Phase-1 helpers (grid-only) ----

    def text_grid(self) -> list[list[str]]:
        """The dense grid as plain strings (empty string for ``None``)."""
        return [[(c.text if c is not None else "") for c in row] for row in self.grid]

    def header(self) -> list[str]:
        """First row's cell texts, stripped (``[]`` if the table has no rows).
        A convenience for role detection — compare against an expected column
        signature."""
        grid = self.text_grid()
        return [c.strip() for c in grid[0]] if grid else []

    # ---- header detection (syntactic) ----

    def header_row_indices(self) -> list[int]:
        """Leading column-header rows: the run of rows (from row 0) whose
        every present cell is a ``<th>``. Stops at the first row that
        contains a ``<td>``. Returns ``[]`` when there is no such block
        (e.g. tables with no ``<th>`` at all)."""
        header_rows: list[int] = []
        for r in range(self.n_rows):
            cells = [c for c in self.grid[r] if c is not None]
            if not cells:
                break
            if all(c.is_header for c in cells):
                header_rows.append(r)
            else:
                break
        return header_rows


def _span(cell: Tag, attr: str) -> int:
    """Read a colspan/rowspan attribute as a positive int (default 1)."""
    try:
        return max(1, int(str(cell.get(attr))))
    except (TypeError, ValueError):
        return 1


def _resolve_table(table: Tag) -> tuple[list[list[Cell | None]], int, int]:
    """Resolve one ``<table>`` Tag into a dense, span-filled grid of
    ``Cell`` objects. Mirrors the bench grid resolver: two passes, the
    first to size the grid (summing colspans), the second to place cells
    and fill spanned positions with the shared ``Cell``."""
    rows = table.find_all("tr")
    if not rows:
        return [], 0, 0

    # Pass 1: widest row by summed colspan determines column count.
    n_cols = 0
    for row in rows:
        width = sum(_span(c, "colspan") for c in row.find_all(["td", "th"]))
        n_cols = max(n_cols, width)
    n_rows = len(rows)
    if n_rows == 0 or n_cols == 0:
        return [], 0, 0

    grid: list[list[Cell | None]] = [[None] * n_cols for _ in range(n_rows)]

    # Pass 2: place each cell, skipping positions already filled by a
    # rowspan from above, and fill all spanned positions with the shared
    # Cell object.
    for r, row in enumerate(rows):
        c = 0
        for el in row.find_all(["td", "th"]):
            while c < n_cols and grid[r][c] is not None:
                c += 1
            if c >= n_cols:
                break
            colspan = _span(el, "colspan")
            rowspan = _span(el, "rowspan")
            cell = Cell(
                text=_clean_text(el),
                row=r,
                col=c,
                rowspan=rowspan,
                colspan=colspan,
                is_header=(el.name == "th"),
            )
            for dr in range(rowspan):
                for dc in range(colspan):
                    rr, cc = r + dr, c + dc
                    if rr < n_rows and cc < n_cols:
                        grid[rr][cc] = cell
            c += colspan

    return grid, n_rows, n_cols


def _context_before(table: Tag, limit: int = 300) -> str:
    """Text immediately preceding the table (up to ``limit`` chars,
    nearest-first), stopping at a previous table. Captures the markdown
    heading breadcrumb that LlamaParse leaves above a table."""
    parts: list[str] = []
    prev = table.previous_sibling
    for _ in range(4):
        if prev is None:
            break
        if isinstance(prev, Tag) and prev.name == "table":
            break
        text = prev.get_text(" ", strip=True) if isinstance(prev, Tag) else str(prev).strip()
        if text:
            parts.insert(0, text)
        prev = prev.previous_sibling
    joined = " ".join(parts)
    return joined[-limit:] if len(joined) > limit else joined


def _context_after(table: Tag, limit: int = 200) -> str:
    """Text immediately following the table, stopping at the next table."""
    parts: list[str] = []
    nxt = table.next_sibling
    for _ in range(2):
        if nxt is None:
            break
        if isinstance(nxt, Tag) and nxt.name == "table":
            break
        text = nxt.get_text(" ", strip=True) if isinstance(nxt, Tag) else str(nxt).strip()
        if text:
            parts.append(text)
        nxt = nxt.next_sibling
    return " ".join(parts)[:limit]


def parse_html_tables(html: str, *, page_num: int | None = None) -> list[Table]:
    """Parse every top-level ``<table>`` in an HTML/markdown blob into a
    list of :class:`Table`. Uses the lenient ``lxml`` parser (matches the
    bench convention; tolerates the malformed/OCR HTML real parsers
    emit). Nested tables are flattened to their text before resolving the
    outer table so their ``<tr>`` rows don't leak in."""
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    # Top-level only: skip tables nested inside another table.
    top_level = [t for t in tables if not any(getattr(p, "name", None) == "table" for p in t.parents)]

    out: list[Table] = []
    for idx, table in enumerate(top_level):
        for nested in table.find_all("table"):
            nested.replace_with(nested.get_text(" ", strip=True))
        caption_el = table.find("caption")
        caption = caption_el.get_text(" ", strip=True) if caption_el else ""
        grid, n_rows, n_cols = _resolve_table(table)
        # Empty tables (0 rows/cols) are KEPT: every top-level <table> tag gets
        # a Table, so table_index, list position, and the prose markers can
        # never diverge — what the model sees enumerated is what its generated
        # script enumerates. They render as "(empty)" in the baseline context.
        out.append(
            Table(
                grid=grid,
                n_rows=n_rows,
                n_cols=n_cols,
                page_num=page_num,
                table_index=idx,
                context_before=_context_before(table),
                context_after=_context_after(table),
                caption=caption,
            )
        )
    return out


def parse_document(pages: list[dict]) -> list[Table]:
    """Parse a LlamaParse-style ``parsed.json`` ``pages`` list (each
    ``{page_num?, text}``) into a flat list of :class:`Table`, tagged with
    their page number. ``table_index`` is per-page (resets each page)."""
    out: list[Table] = []
    for i, pg in enumerate(pages):
        page_num = pg.get("page_num", i)
        out.extend(parse_html_tables(pg.get("text", ""), page_num=page_num))
    return out


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Parse tables from a LlamaParse parsed.json")
    ap.add_argument("parsed_json", type=Path)
    ap.add_argument("--page", type=int, default=None, help="only show tables on this page_num")
    args = ap.parse_args()

    doc = json.loads(args.parsed_json.read_text())
    tables = parse_document(doc.get("pages", []))
    for t in tables:
        if args.page is not None and t.page_num != args.page:
            continue
        print(f"\n=== page {t.page_num} table {t.table_index} ({t.n_rows}x{t.n_cols}) ===")
        if t.context_before:
            print(f"  context_before: {t.context_before[-120:]!r}")
        for row in t.text_grid()[:8]:
            print("  ", row)
