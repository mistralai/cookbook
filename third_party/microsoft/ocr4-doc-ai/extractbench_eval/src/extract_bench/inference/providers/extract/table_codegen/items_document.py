"""Items-based Document for the extract harness (structured LlamaParse items).

Mirror of ``document.py``, but built from LlamaParse's expanded ``items`` result
— the per-page reading-order sequence of typed blocks (text / heading / table /
list / header / footer / …) — instead of per-page markdown text. Generated
scripts iterate the items in order and filter by type rather than regexing one
prose blob, so formatting drift in the parse can't silently zero an array the
way it can with ``non_table_text`` string matching.

Representation choices (deliberate):
  - Tables are surfaced by their typed ``rows`` grid — the preview shows a
    *compressed* view of ``item.rows`` (header rows labeled, first rows + a
    count) and the script iterates the same ``item.rows`` at runtime, so the
    model writes code against the shape it sees. Header rows are detected from
    the item's HTML ``<th>`` (``rows`` itself has no header flag). (``html`` /
    ``csv`` remain on the item but are not the rendered form.)
  - Cross-page tables stay as separate per-page items — the server-side table
    merge flag is buggy and stays off, so ``merged_from_pages`` is never relied
    upon.
  - Container items (list/header/footer) keep their nested items as
    ``children``; iteration yields the container once, not its children.

This module is loaded standalone (by file path) inside the sandbox subprocess,
so it must stay stdlib-only — no bs4, no extract_bench imports.

Boundary format (``ItemsDocument(pages)`` / ``to_pages()``): the LlamaParse
items expand shape, ``[{"page_number": n, "items": [<raw item dict>, ...]}]``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _type_set(types: str | Iterable[str] | None) -> set[str] | None:
    """Normalize a type filter: a bare string means a single-type set."""
    if types is None:
        return None
    if isinstance(types, str):
        return {types}
    return set(types)


# Typographic-quote normalization (opt-in). The parse mixes curly “ ” / ‘ ’ with
# straight quotes within structurally-identical regions, so a regex keyed on a
# single quote form silently matches only a subset — on lp_06 the alias regexes hit
# the straight-quoted records (~21%) and dropped the 378 curly-quoted ones. Folding
# curly→straight at construction makes the model's prompt and the script's runtime
# see one quote form. Stdlib-only (this module is reloaded inside the sandbox).
# Scope: quotes only — no dashes/ellipses/other punctuation.
_QUOTE_FOLD = str.maketrans(
    {
        "“": '"',  # “
        "”": '"',  # ”
        "„": '"',  # „
        "‟": '"',  # ‟
        "‘": "'",  # ‘
        "’": "'",  # ’
        "‚": "'",  # ‚
        "‛": "'",  # ‛
    }
)


def _fold_quotes(v: Any) -> Any:
    """Fold curly quotes to straight on a string; pass non-strings (numbers/None) through."""
    return v.translate(_QUOTE_FOLD) if isinstance(v, str) else v


def _normalize_quotes_in_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy the raw item dicts, folding curly quotes in the text fields the harness
    consumes (``md``, ``value``, ``rows`` cells) and recursing into nested ``items``.
    Returns new dicts/lists so the caller's input is never mutated."""
    out: list[dict[str, Any]] = []
    for it in items:
        ni = dict(it)
        if "md" in ni:
            ni["md"] = _fold_quotes(ni["md"])
        if "value" in ni:
            ni["value"] = _fold_quotes(ni["value"])
        if ni.get("rows"):
            ni["rows"] = [[_fold_quotes(c) for c in row] for row in ni["rows"]]
        if ni.get("items"):
            ni["items"] = _normalize_quotes_in_items(ni["items"])
        out.append(ni)
    return out


class Item:
    """One parsed block, wrapping the raw LlamaParse item dict.

    Field availability follows the item type: ``level`` is set on headings,
    ``html``/``rows``/``csv`` on tables, ``children`` on container items
    (list/header/footer). Everything else is ``None``/empty. The raw dict stays
    reachable via ``.raw`` so fields this wrapper doesn't surface (bbox, …)
    aren't lost.
    """

    def __init__(self, raw: dict[str, Any], page_num: int):
        self.raw = raw
        self.page_num = page_num
        self.type: str = raw.get("type") or ""
        self.md: str = raw.get("md") or ""
        self.value: str = raw.get("value") or ""
        self.level: int | None = raw.get("level")
        self.html: str = raw.get("html") or ""
        self.rows: list[list[Any]] = raw.get("rows") or []
        self.csv: str = raw.get("csv") or ""
        self.children: list[Item] = [Item(c, page_num) for c in (raw.get("items") or [])]

    def __repr__(self) -> str:
        head = self.md.splitlines()[0][:40] if self.md else ""
        return f"Item(type={self.type!r}, page={self.page_num}, md={head!r})"


class ItemsPage:
    """One document page: its number and its items in reading order."""

    def __init__(self, page_num: int, items: list[dict[str, Any]]):
        self.page_num = page_num
        self._items = [Item(raw, page_num) for raw in items]

    def items(
        self,
        types: str | Iterable[str] | None = None,
        exclude: str | Iterable[str] | None = None,
    ) -> list[Item]:
        inc = _type_set(types)
        exc = _type_set(exclude) or set()
        return [it for it in self._items if (inc is None or it.type in inc) and it.type not in exc]

    def __repr__(self) -> str:
        return f"ItemsPage(page_num={self.page_num}, n_items={len(self._items)})"


class ItemsDocument:
    """A parsed document as an in-order sequence of typed items.

    ``items()`` is the primary view: every item across all pages, in reading
    order, optionally filtered in (``types=``) or out (``exclude=``) by type.
    """

    def __init__(self, pages: list[dict[str, Any]], normalize_quotes: bool = False):
        self._raw_pages: list[dict[str, Any]] = []
        self.pages: list[ItemsPage] = []
        for i, pg in enumerate(pages):
            page_num = int(pg.get("page_number", pg.get("page_num", i + 1)))
            items = pg.get("items") or []  # failed-page entries carry no items; the page stays visible
            if normalize_quotes:
                # Fold the page items BEFORE storing/wrapping so both the rendered prompt
                # AND to_pages() (the sandbox boundary) carry the normalized text — the
                # script in the sandbox rebuilds from to_pages(), so it must match the view.
                items = _normalize_quotes_in_items(items)
            self._raw_pages.append({"page_number": page_num, "items": items})
            self.pages.append(ItemsPage(page_num, items))
        self._by_num = {p.page_num: p for p in self.pages}

    def page(self, page_num: int) -> ItemsPage | None:
        return self._by_num.get(page_num)

    def items(
        self,
        types: str | Iterable[str] | None = None,
        exclude: str | Iterable[str] | None = None,
    ) -> list[Item]:
        out: list[Item] = []
        for p in self.pages:
            out.extend(p.items(types=types, exclude=exclude))
        return out

    def to_pages(self) -> list[dict[str, Any]]:
        """The JSON-serializable form this document round-trips through (the
        sandbox boundary and the provider's ``parse_pages`` observability)."""
        return self._raw_pages

    def __repr__(self) -> str:
        return f"ItemsDocument(n_pages={len(self.pages)}, n_items={len(self.items())})"


# Uniform handle for the sandbox shim: every document module exposes `Document`.
Document = ItemsDocument


# --------------------------------------------------------------------------- #
# baseline-context rendering (shown to the agent up front)
#
# These run ONLY in the parent process (building the agent prompt), never in the
# sandbox — so, unlike the Item/ItemsDocument classes above, they may use bs4 via
# table_parser. Keep any such import lazy/inside the function.
# --------------------------------------------------------------------------- #
def _table_header_rows(item: Item) -> list[int]:
    """Best-effort header-row indices into ``item.rows``. The typed rows grid
    carries no header flag, so detect headers from the item's HTML (``<th>``) via
    the shared table parser — but only when the HTML grid aligns row-for-row with
    ``item.rows`` (LlamaParse sometimes explodes a stacked header into extra
    ``<tr>`` rows in the HTML that ``rows`` keeps as one). On any divergence or
    parse failure, treat the single leading row as the header."""
    if not item.rows:
        return []
    parsed = []
    if item.html:
        try:
            try:
                from .table_parser import parse_html_tables
            except ImportError:  # pragma: no cover — defensive; render is parent-only
                from table_parser import parse_html_tables  # type: ignore[import-not-found,no-redef]
            parsed = parse_html_tables(item.html)
        except Exception:  # noqa: BLE001 — rendering must never crash on odd HTML
            parsed = []
    if parsed and parsed[0].n_rows == len(item.rows):
        return parsed[0].header_row_indices()  # faithful; [] when the table truly has no header
    return [0]


def compressed_table(item: Item, n_rows: int | None = 5) -> str:
    """A table compressed for the prompt: a ``(R×C)`` label, each detected header
    row on a ``header:`` line, the first ``n_rows`` data rows, then the total
    data-row count. Rows are shown verbatim from ``item.rows`` — the same object
    the script iterates — so the preview and runtime never diverge. ``n_rows=None``
    shows ALL data rows (no truncation, no ellipsis) — used by the on-demand
    ``view_document`` tool to re-view a table in full."""
    rows = item.rows
    if not rows:
        return "[table]  (empty)"
    hdr = set(_table_header_rows(item))
    data = [r for i, r in enumerate(rows) if i not in hdr]
    n_cols = max((len(r) for r in rows), default=0)
    lines = [f"[table] ({len(rows)}×{n_cols})"]
    lines.extend(f"  header: {rows[i]}" for i in sorted(hdr))
    shown = data if n_rows is None else data[:n_rows]
    lines.extend(f"    {r}" for r in shown)
    ellipsis = "… " if (n_rows is not None and len(data) > n_rows) else ""
    lines.append(f"    {ellipsis}({len(data)} data rows total)")
    return "\n".join(lines)


def render_item(item: Item, full_tables: bool = False) -> str:
    """``[type]`` tag followed by the item's content. A table is shown as a
    compressed view of its ``rows`` grid (header rows labeled, first rows + a
    count) — the same object the script iterates at runtime (``item.rows``), so
    the preview never shows a different shape than the code receives. Everything
    else is shown as its md. ``full_tables=True`` shows every table row (for the
    ``view_document`` re-view tool); the default keeps the first-rows preview."""
    if item.type == "table":
        return compressed_table(item, n_rows=None if full_tables else 5)
    return f"[{item.type}] {item.md}".rstrip()


def render_items_document(doc: ItemsDocument) -> str:
    """The full baseline context: per page, each item on its own ``[type]``-
    tagged block, in reading order."""
    blocks: list[str] = []
    for p in doc.pages:
        lines = [f"===== PAGE {p.page_num} ====="]
        lines.extend(render_item(it) for it in p.items())
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
