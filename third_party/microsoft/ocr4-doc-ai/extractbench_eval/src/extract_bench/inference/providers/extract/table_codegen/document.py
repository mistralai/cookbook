"""Document / Page wrapper + baseline-context rendering for the extract harness.

The phase-1 and direct generation agents work over a whole *document*, not a
flat ``list[Table]``. This module wraps a LlamaParse-style ``parsed.json`` into
a small object graph the generated scripts call at runtime:

    document.tables()        -> flat list[Table] across all pages
    document.pages           -> list[Page]
    document.page(n)         -> the Page whose page_num == n

    page.page_num            -> int
    page.text                -> raw page text (prose + <table> HTML)
    page.non_table_text      -> the prose with each table lifted to a
                                "[table P.I]" placeholder (positional reference)
    page.tables()            -> list[Table] on this page

``Table`` itself is unchanged (``table_parser.Table``): text_grid / header /
header_row_indices / context_before / context_after / caption / page_num /
table_index / n_rows / n_cols.

Beyond the runtime object, this module renders the **baseline context** shown to
the agent up front (the whole document, to start): per page, the non-table prose
followed by a *compressed* view of each table — its header row(s), the first 5
non-header rows, and the total non-header row count. (We'll iterate on the
compression later; first-5 is the agreed starting point.)
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

try:
    from .table_parser import Table, parse_html_tables
except ImportError:  # pragma: no cover — the sandbox shim loads this file as a top-level module
    from table_parser import Table, parse_html_tables  # type: ignore[import-not-found,no-redef]


def _table_label(page_num: int | None, table_index: int) -> str:
    """Human-readable, unambiguous table reference. Page and index are kept as
    separate integers (``page 4 table 0``) rather than a dotted ``4.0`` form,
    which reads like a float."""
    return f"[page {page_num} table {table_index}]"


# --------------------------------------------------------------------------- #
# runtime object graph
# --------------------------------------------------------------------------- #
class Page:
    """One document page: raw text, prose-with-tables-lifted, and its tables."""

    def __init__(self, page_num: int, text: str):
        self.page_num = page_num
        self.text = text
        self._tables = parse_html_tables(text, page_num=page_num)
        self.non_table_text = _lift_tables(text, page_num)

    def tables(self) -> list[Table]:
        return self._tables

    def __repr__(self) -> str:
        return f"Page(page_num={self.page_num}, n_tables={len(self._tables)})"


class Document:
    """A parsed document: page access (``.page(n)`` / ``.pages``) and a flat
    ``.tables()`` view across every page."""

    def __init__(self, pages: list[dict]):
        self.pages: list[Page] = [Page(pg.get("page_num", i), pg.get("text", "")) for i, pg in enumerate(pages)]
        self._by_num = {p.page_num: p for p in self.pages}

    def page(self, page_num: int) -> Page | None:
        return self._by_num.get(page_num)

    def tables(self) -> list[Table]:
        out: list[Table] = []
        for p in self.pages:
            out.extend(p.tables())
        return out

    def to_pages(self) -> list[dict]:
        """The JSON-serializable form this document round-trips through (the
        sandbox boundary and the provider's ``parse_pages`` observability)."""
        return [{"page_num": p.page_num, "text": p.text} for p in self.pages]

    def __repr__(self) -> str:
        return f"Document(n_pages={len(self.pages)}, n_tables={len(self.tables())})"


def load_document(parsed_json: Path) -> Document:
    import json

    return Document(json.loads(Path(parsed_json).read_text()).get("pages", []))


# --------------------------------------------------------------------------- #
# non-table prose: lift each top-level <table> to a positional placeholder
# --------------------------------------------------------------------------- #
def _lift_tables(text: str, page_num: int) -> str:
    """Replace each top-level ``<table>`` with a ``[page P table I]`` marker so
    the prose keeps a positional reference to where each table sat (``I`` matches
    the Table.table_index that ``parse_html_tables`` uses). Markdown formatting in
    the prose is preserved (it is semantically meaningful)."""
    soup = BeautifulSoup(text, "lxml")
    top_level = [t for t in soup.find_all("table") if not any(getattr(p, "name", None) == "table" for p in t.parents)]
    for idx, t in enumerate(top_level):
        t.replace_with(_table_label(page_num, idx))
    # lxml wraps fragments in <html><body>; pull the body text back out.
    body = soup.body or soup
    txt = body.get_text() if isinstance(body, Tag) else str(body)
    # collapse the runs of blank lines bs4 leaves behind, keep single breaks.
    txt = re.sub(r"\n[ \t]*\n[ \t\n]*", "\n\n", txt)
    return txt.strip()


# --------------------------------------------------------------------------- #
# baseline-context rendering (shown to the agent up front)
# --------------------------------------------------------------------------- #
def compressed_table(t: Table, n_rows: int | None = 5) -> str:
    """Compressed view of one table: a label + header row(s) + first ``n_rows``
    non-header rows + the total non-header row count. ``header:`` lines appear
    only for rows ``header_row_indices()`` actually detects (no row-0 guessing
    for ``<th>``-less tables — the runtime treats those rows as data, so the
    rendering must too). An empty table (a ``<table>`` tag with no rows/cells)
    renders as a single ``(empty)`` line so it stays visible — and enumerable —
    in flow. ``n_rows=None`` shows ALL non-header rows (no truncation) — used by
    the ``view_document`` re-view tool."""
    grid = t.text_grid()
    if not grid:
        return f"{_table_label(t.page_num, t.table_index)}  (empty)"
    hdr_idx = t.header_row_indices()
    hdr_set = set(hdr_idx)
    data = [grid[i] for i in range(len(grid)) if i not in hdr_set]

    lines = [f"{_table_label(t.page_num, t.table_index)}  ({t.n_rows}×{t.n_cols})"]
    for i in hdr_idx:
        lines.append(f"  header: {grid[i]}")
    for row in data if n_rows is None else data[:n_rows]:
        lines.append(f"    {row}")
    ellipsis = "… " if (n_rows is not None and len(data) > n_rows) else ""
    lines.append(f"    {ellipsis}({len(data)} non-header rows total)")
    return "\n".join(lines)


def render_page(p: Page, n_rows: int | None = 5) -> str:
    """A page rendered for the agent: the prose with each table inlined *in
    place* (the compressed view substituted where its marker sits), rather than
    appended at the end. Any table whose marker isn't found in the prose is
    appended as a fallback so nothing is dropped."""
    text = p.non_table_text
    trailing: list[str] = []
    for t in p.tables():
        block = compressed_table(t, n_rows)
        # Look up by the table's ORIGINAL index — _lift_tables marks every
        # top-level <table> (including empty ones the parser skips), so a
        # sequential enumerate index would drift past an empty table and
        # substitute into the wrong marker.
        marker = _table_label(t.page_num, t.table_index)
        if marker in text:
            text = text.replace(marker, "\n" + block + "\n")
        else:
            trailing.append(block)
    if trailing:
        text = (text + "\n\n" + "\n\n".join(trailing)).strip()
    return text


def render_document(doc: Document, n_rows: int = 5) -> str:
    """The full baseline context: each page's prose with its tables inlined in
    place (see :func:`render_page`)."""
    blocks: list[str] = []
    for p in doc.pages:
        body = render_page(p, n_rows)
        blocks.append(f"===== PAGE {p.page_num} =====\n{body}" if body else f"===== PAGE {p.page_num} =====")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render the baseline document context")
    ap.add_argument("parsed_json", type=Path)
    ap.add_argument("--rows", type=int, default=5)
    args = ap.parse_args()
    doc = load_document(args.parsed_json)
    print(repr(doc))
    print("=" * 70)
    print(render_document(doc, args.rows))
