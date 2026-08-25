"""Column-coverage warning: point the model at table rows it likely missed.

After a generated extractor runs, this compares each parsed table *column* against
the produced output. When the output reflects MOST but not all of a column's
non-empty cell values, the un-reflected rows are likely dropped or merged — exactly
the omissions self-review is otherwise blind to (you can't review a record you never
produced). It emits a warning naming the specific page / table / row and showing the
full row, so the model can decide whether it missed content.

Matching is on STANDALONE values, not substrings: an over-merged value (folded into a
neighbouring record's longer string) is correctly flagged as not appearing on its own.
Numeric-heavy columns (amounts, balances) and date columns are skipped — repeated /
reformatted values there are noisy; the signal lives in the distinctive text columns
(descriptions, payees, ids). A column the model reformatted wholesale falls below the
coverage floor and is skipped rather than flooding false positives.

Runs in the parent process (inside the run_script tool handler), so it may use
``items_document``'s bs4-backed header detection — never inside the sandbox.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .document import Document
from .items_document import ItemsDocument

# A column must have at least this many non-empty cells before we judge "most".
_MIN_CELLS = 5
# Fire only when the output reflects at least this fraction (but < 1.0) of the column.
_COVERAGE_MIN = 0.5
# Skip a column whose cells are mostly numeric (amounts/balances) — too noisy.
_NUMERIC_COL_FRAC = 0.6
# Cap the rows we name so a badly-off run doesn't produce a wall of text.
_MAX_WARN_ROWS = 50

_WS = re.compile(r"\s+")
_NUM = re.compile(r"^[-+]?[\d,]*\.?\d+%?$")


def _norm(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).replace("<br/>", " ").replace("<br>", " ")
    return _WS.sub(" ", s).strip().upper()


def _is_numeric(cell: str) -> bool:
    return bool(_NUM.match(cell.replace("$", "").replace(" ", "")))


def _output_value_counts(output: Any) -> Counter[str]:
    """Multiset of normalized STANDALONE leaf values in the output (strings +
    numbers; bools excluded). 'Standalone' is the point: an over-merged value that
    only survives as a substring of a longer fused string is not counted here, so it
    still shows up as missing."""
    counts: Counter[str] = Counter()

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, bool):
            return
        elif isinstance(v, (str, int, float)):
            n = _norm(v)
            if n:
                counts[n] += 1

    walk(output)
    return counts


def _iter_tables(document: Document | ItemsDocument) -> list[tuple[int, int, list[list[Any]], set[int]]]:
    """Yield (page_num, table_ordinal_on_page, grid, header_row_indices) for every
    table, uniform across the items and pages document shapes."""
    out: list[tuple[int, int, list[list[Any]], set[int]]] = []
    if isinstance(document, ItemsDocument):
        from .items_document import _table_header_rows

        for ipage in document.pages:
            for t_ord, item in enumerate(ipage.items(types="table")):
                out.append((ipage.page_num, t_ord, item.rows, set(_table_header_rows(item))))
    else:
        for ppage in document.pages:
            for t in ppage.tables():
                out.append((ppage.page_num, t.table_index, t.text_grid(), set(t.header_row_indices())))
    return out


def coverage_warnings(
    document: Document | ItemsDocument,
    output: Any,
    *,
    coverage_min: float = _COVERAGE_MIN,
    min_cells: int = _MIN_CELLS,
) -> str:
    """Return a warning block (or '' if nothing is flagged) naming table rows whose
    column value does not appear as a distinct value in ``output``."""
    # ONE depleting budget across all tables: each standalone output value can
    # account for exactly one parse cell. A description that recurs across pages
    # (so the output has several copies) is covered that many times — the extra
    # occurrence that was actually dropped/merged finds no budget left and is
    # flagged, on whichever table it lands in (document order).
    remaining = dict(_output_value_counts(output))
    flagged: list[str] = []
    total_flagged = 0

    for page_num, t_ord, grid, header_idx in _iter_tables(document):
        if not grid:
            continue
        n_cols = max((len(r) for r in grid), default=0)
        data = [(i, r) for i, r in enumerate(grid) if i not in header_idx]
        for c in range(n_cols):
            cells = [(i, _norm(r[c])) for i, r in data if c < len(r) and _norm(r[c])]
            if len(cells) < min_cells:
                continue
            if sum(_is_numeric(v) for _, v in cells) / len(cells) > _NUMERIC_COL_FRAC:
                continue  # amount/balance column — noisy, skip
            # Tentatively account each cell against the running budget; commit the
            # debit only if the column clears the coverage floor (so a wholesale-
            # reformatted column neither flags nor wrongly consumes budget).
            tentative = dict(remaining)
            covered = 0
            missing_here: list[tuple[int, list[Any]]] = []
            for i, v in cells:
                if tentative.get(v, 0) > 0:
                    tentative[v] -= 1
                    covered += 1
                else:
                    missing_here.append((i, grid[i]))
            coverage = covered / len(cells)
            if coverage < coverage_min:
                continue  # reformatted column — not a reliable signal
            remaining = tentative  # commit
            if coverage >= 1.0 or not missing_here:
                continue
            total_flagged += len(missing_here)
            head = (
                f"page {page_num} table {t_ord} column {c}: your output accounts for "
                f"{covered}/{len(cells)} of this column's values. These rows look unaccounted "
                f"for (likely dropped or merged into a neighbour) — verify each is in your output:"
            )
            shown_so_far = sum(b.count("\n    row ") for b in flagged)
            room = max(_MAX_WARN_ROWS - shown_so_far, 0)
            body = [f"    row {i}: {row}" for i, row in missing_here[:room]]
            flagged.append(head + "\n" + "\n".join(body))
        if sum(b.count("\n    row ") for b in flagged) >= _MAX_WARN_ROWS:
            break

    if not flagged:
        return ""
    note = ""
    shown = sum(b.count("\n    row ") for b in flagged)
    if total_flagged > shown:
        note = f"\n(+{total_flagged - shown} more unaccounted rows not shown)"
    return (
        "COVERAGE CHECK — some table columns are only partly reflected in your output. "
        "Each row below carries a value that does not appear as a distinct value in your "
        "output; a row with its own date/payee/amount is its own transaction even if a "
        "cell (e.g. the date) is blank — do not fold it into a neighbour:\n\n" + "\n\n".join(flagged) + note
    )
