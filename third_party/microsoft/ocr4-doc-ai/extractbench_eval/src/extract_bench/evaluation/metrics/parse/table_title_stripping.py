"""Title-row stripping stage.

Detects table-title rows (leading ``<td>`` colspan titles and ``<th>``
spanning titles inside the header block) and physically removes them
from a table's grid before any metric (GriTS, TRM) consumes it. The
stripped texts and the surviving header geometry — both translated to
the **trimmed-table** coordinate system — are packed into a
``HeaderHints`` payload and attached to the returned ``ExtractedTable``.

This module owns **all** title detection. ``table_record_match_metric``
imports nothing from here except the ``HeaderHints`` dataclass; it never
runs a detector itself.

Detector functions are moved verbatim from the previous TRM
implementation. Behavior is preserved bit-for-bit; the only change is
*where* the rows are removed (upstream, once, for everyone) instead of
inside TRM record-building.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from extract_bench.evaluation.metrics.parse.table_extraction import ExtractedTable
from extract_bench.evaluation.metrics.parse.table_parsing import TableData

# ---------------------------------------------------------------------------
# Detectors (moved verbatim from table_record_match_metric.py)
# ---------------------------------------------------------------------------


def detect_td_title_rows(table: TableData, nominal_header_rows: set[int]) -> set[int]:
    """Detect leading <td> rows where uniform text spans all columns (table titles).

    Rows before the first <th> row that have a single non-empty text value
    repeated across all columns (e.g. <td colspan="5">Title</td>) are titles.
    Single-column tables are excluded since they trivially have "uniform" text.
    """
    n_rows, n_cols = table.data.shape
    if n_cols <= 1:
        return set()

    candidates: set[int] = set()
    for r in range(n_rows):
        if r in nominal_header_rows:
            break
        row_values = [str(table.data[r, c]).strip() for c in range(n_cols)]
        unique_nonempty = {v for v in row_values if v}
        if len(unique_nonempty) == 1 and all(v for v in row_values):
            candidates.add(r)
        else:
            break

    if candidates and len(candidates) < n_rows:
        return candidates
    return set()


def find_col_header_rows(table: TableData, nominal_header_rows: set[int], td_title_rows: set[int]) -> set[int]:
    """Identify the leading block of consecutive full-width <th> header rows.

    Starts scanning after any <td> title rows. Stops at the first row that
    either isn't marked as a header or has partial <th> coverage with novel
    (non-header) values in non-<th> columns (dual-axis data rows).
    """
    n_rows, n_cols = table.data.shape
    first_possible_header = max(td_title_rows) + 1 if td_title_rows else 0
    leading_header_end = first_possible_header

    for r in range(first_possible_header, n_rows):
        if r not in nominal_header_rows:
            break
        has_non_th_content = any(
            str(table.data[r, c]).strip() and (r, c) not in table.header_cells for c in range(n_cols)
        )
        if has_non_th_content:
            break
        leading_header_end = r + 1

    col_header_rows = set(range(leading_header_end)) & nominal_header_rows

    if col_header_rows:
        bottom_row = max(col_header_rows)
        for c in range(n_cols):
            cell_val = str(table.data[bottom_row, c]).strip()
            if cell_val and (bottom_row, c) not in table.header_cells:
                col_header_rows.discard(bottom_row)
                break

    return col_header_rows


def detect_th_title_rows(table: TableData, col_header_rows: set[int], *, max_top: int = 1) -> set[int]:
    """Detect <th> header rows that are spanning titles, not real column headers.

    Returns the set of title rows to strip from the header block:
    up to ``max_top`` *top* title rows (the topmost contiguous title
    rows) plus at most one *bottom* title (a title row that sits at the
    very bottom of the header block, e.g. ``(in millions)``). Each
    removal is independently guarded — never strip a title if doing so
    would leave zero header rows.

    A row's rowspan does not affect the count: each grid row counts as
    one slot regardless of how many original rows it represents.

    The bottom-title strip is independent of ``max_top`` and is always
    applied when present.
    """
    n_cols = table.data.shape[1]

    cols_with_headers: set[int] = set()
    for c in range(n_cols):
        for row_idx, text in table.col_headers.get(c, []):
            if row_idx in col_header_rows and text.strip():
                cols_with_headers.add(c)
                break

    title_rows: set[int] = set()
    for r in col_header_rows:
        col_texts: list[str] = []
        for c in range(n_cols):
            text = ""
            for row_idx, t in table.col_headers.get(c, []):
                if row_idx == r:
                    text = t.strip()
            col_texts.append(text)

        unique_nonempty = {t for t in col_texts if t}
        if len(unique_nonempty) != 1:
            continue

        covered = {c for c, t in enumerate(col_texts) if t}

        other_header_cols: set[int] = set()
        for other_r in col_header_rows:
            if other_r == r:
                continue
            for c in range(n_cols):
                for row_idx, t in table.col_headers.get(c, []):
                    if row_idx == other_r and t.strip():
                        other_header_cols.add(c)

        target_cols = other_header_cols if other_header_cols else cols_with_headers
        if not target_cols:
            continue
        uncovered = target_cols - covered
        if uncovered:
            edge_cols = {min(target_cols), max(target_cols)}
            if not uncovered <= edge_cols:
                continue

        rh = table.row_headers.get(r, [])
        n_nonempty_th = sum(1 for _, text in rh if text.strip())
        if n_nonempty_th < len(covered):
            title_rows.add(r)

    if not title_rows:
        return set()

    to_strip: set[int] = set()

    # Helper: a row is a "rowspan continuation" of a title above it
    # when it sits inside col_header_rows but has no col_headers
    # entries originated at its own row index. Its grid cells were
    # filled by rowspan expansion, not by a fresh <th>. Such rows
    # should be stripped together with their originating title row but
    # must NOT consume a slot from the cap (a rowspanning title is one
    # logical title, not N).
    def _is_rowspan_continuation(r: int) -> bool:
        for c in range(n_cols):
            for row_idx, _t in table.col_headers.get(c, []):
                if row_idx == r:
                    return False
        return True

    sorted_header = sorted(col_header_rows)
    bottom_of_block = sorted_header[-1]

    # Top strip: walk title rows contiguously from the top of the
    # header block. Each row must itself be a title (rowspan-
    # continuation rows are absorbed without consuming a slot). Break
    # on the first non-title, non-continuation row. Cap is hard-clamped
    # to ``len(header) - 1`` so the top strip alone can never empty
    # the block — the bottom strip below decides separately whether to
    # also fire.
    #
    # When ``max_top == 0`` we skip the top walk entirely and only run
    # the bottom strip.
    if max_top > 0:
        top_cap = min(len(col_header_rows) - 1, max_top)
        n_taken = 0
        i = 0
        while i < len(sorted_header) and n_taken < top_cap:
            r = sorted_header[i]
            if r not in title_rows:
                break
            tentative = {r}
            j = i + 1
            while j < len(sorted_header) and _is_rowspan_continuation(sorted_header[j]):
                tentative.add(sorted_header[j])
                j += 1
            to_strip |= tentative
            n_taken += 1
            i = j

    # Bottom strip: independently guarded — only fires if the bottom
    # row of the header block is itself a title AND more than one
    # header row would remain after the strip.
    if bottom_of_block in title_rows and bottom_of_block not in to_strip:
        if len(col_header_rows - to_strip - {bottom_of_block}) >= 1:
            to_strip.add(bottom_of_block)

    return to_strip


def collect_stripped_titles(
    table: TableData,
    title_rows: set[int],
    td_title_rows: set[int],
) -> set[str]:
    """Collect text from title rows that were excluded from column keys.

    Returns the unique non-empty text values from both <th> title rows
    and <td> title rows, used for prefix-stripping fallback in alignment.
    """
    n_cols = table.data.shape[1]
    stripped_titles: set[str] = set()

    for r in title_rows:
        col_texts = []
        for c in range(n_cols):
            for row_idx, t in table.col_headers.get(c, []):
                if row_idx == r:
                    col_texts.append(t.strip())
        nonempty = {t for t in col_texts if t}
        stripped_titles.update(nonempty)

    for r in td_title_rows:
        td_vals = {str(table.data[r, c]).strip() for c in range(n_cols)}
        stripped_titles.update(v for v in td_vals if v)

    return stripped_titles


# ---------------------------------------------------------------------------
# HeaderInfo + extract_header_info (un-hinted; for callers that run *before*
# the strip stage, currently just ``table_splitting.try_split_by_period``)
# ---------------------------------------------------------------------------


@dataclass
class HeaderInfo:
    """Results of header analysis for a single (un-stripped) table."""

    keys: list[str]
    synthetic_keys: frozenset[str]
    col_header_rows: set[int]
    th_title_rows: set[int]
    td_title_rows: set[int]
    stripped_titles: set[str]


def extract_header_info(table: TableData) -> HeaderInfo:
    """Analyze table headers on an un-stripped table.

    This is the convenience entry point for callers that operate *before*
    ``strip_title_rows`` (currently only ``table_splitting``). It runs the
    detectors and packages the result. TRM does **not** call this function
    — it consumes precomputed ``HeaderHints`` instead.
    """
    # Imported lazily to avoid a circular import: table_record_match_metric
    # imports HeaderHints from this module.
    from extract_bench.evaluation.metrics.parse.table_record_match_metric import (
        _build_column_keys,
    )

    if table.data.size == 0:
        return HeaderInfo(
            keys=[],
            synthetic_keys=frozenset(),
            col_header_rows=set(),
            th_title_rows=set(),
            td_title_rows=set(),
            stripped_titles=set(),
        )

    nominal_header_rows = table.header_rows if table.header_rows else set()

    td_title_rows = detect_td_title_rows(table, nominal_header_rows)
    col_header_rows = find_col_header_rows(table, nominal_header_rows, td_title_rows)
    th_title_rows = detect_th_title_rows(table, col_header_rows)

    keys, synthetic_keys = _build_column_keys(table, col_header_rows, th_title_rows)
    stripped_titles = collect_stripped_titles(table, th_title_rows, td_title_rows)

    return HeaderInfo(
        keys=keys,
        synthetic_keys=synthetic_keys,
        col_header_rows=col_header_rows,
        th_title_rows=th_title_rows,
        td_title_rows=td_title_rows,
        stripped_titles=stripped_titles,
    )


# ---------------------------------------------------------------------------
# HeaderHints + strip_title_rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeaderHints:
    """Precomputed header geometry for a *trimmed* table.

    All row indices are in the trimmed-table coordinate system (i.e. after
    the rows in ``to_remove`` have been physically deleted).
    ``stripped_titles`` is the un-normalized text the title rows
    contained; TRM applies its own normalization to these strings before
    using them in the alignment fallback.
    """

    col_header_rows: frozenset[int]
    th_title_rows: frozenset[int]
    stripped_titles: frozenset[str]


def _leading_all_empty_row_indices(table: TableData) -> frozenset[int]:
    """Return the indices of consecutive all-empty rows at the top of the table.

    A row is all-empty if every cell's stripped text is the empty string.
    Scanning stops at the first row with any non-empty cell.
    """
    n_rows, n_cols = table.data.shape
    leading: set[int] = set()
    for r in range(n_rows):
        if any(str(table.data[r, c]).strip() for c in range(n_cols)):
            break
        leading.add(r)
    return frozenset(leading)


def _remove_rows(table: TableData, rows_to_remove: frozenset[int]) -> tuple[TableData, dict[int, int]]:
    """Drop ``rows_to_remove`` from ``table`` and remap all row-indexed metadata.

    Returns the new table plus an ``old_to_new`` index map for surviving
    rows.  Rows in ``rows_to_remove`` have no entry in the map.
    """
    n_rows = table.data.shape[0]
    survivors = [r for r in range(n_rows) if r not in rows_to_remove]
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(survivors)}

    if not rows_to_remove:
        return table, old_to_new

    new_data = np.delete(table.data, sorted(rows_to_remove), axis=0)

    new_header_rows = {old_to_new[r] for r in table.header_rows if r in old_to_new}
    new_header_cells = {(old_to_new[r], c) for (r, c) in table.header_cells if r in old_to_new}

    new_col_headers: dict[int, list[tuple[int, str]]] = {}
    for col_idx, entries in table.col_headers.items():
        remapped = [(old_to_new[r], t) for (r, t) in entries if r in old_to_new]
        if remapped:
            new_col_headers[col_idx] = remapped

    new_row_headers: dict[int, list[tuple[int, str]]] = {}
    for row_idx, entries in table.row_headers.items():
        if row_idx in old_to_new:
            new_row_headers[old_to_new[row_idx]] = list(entries)

    new_table = TableData(
        data=new_data,
        header_rows=new_header_rows,
        header_cols=set(table.header_cols),
        col_headers=new_col_headers,
        row_headers=new_row_headers,
        header_cells=new_header_cells,
        context_before=table.context_before,
        context_after=table.context_after,
    )
    return new_table, old_to_new


def strip_title_rows(et: ExtractedTable, *, max_top_title_rows: int = 1) -> ExtractedTable:
    """Physically remove title rows from ``et.table_data`` and attach hints.

    ``max_top_title_rows`` caps the *top* title strip:

    - ``0`` → strip nothing from the top: no leading ``<td>`` titles, no
      top ``<th>`` titles. The bottom-``<th>`` title strip is independent
      and still applied.
    - ``>= 1`` (default ``1``) → strip *all* leading uniform ``<td>``
      colspan title rows (today's behavior is unchanged here), and strip
      up to ``max_top_title_rows`` top ``<th>`` spanning title rows from
      the header block.

    A rowspanning title row consumes 1 slot from the cap, not N — each
    grid row counts as one slot regardless of original rowspan.

    The bottom-of-header ``<th>`` title row (e.g. ``(in millions)``) is
    *not* removed from the grid — TRM re-emits it as a data row via
    ``_data_row_indices`` when the header block has more than one row,
    and GriTS treats it as a header cell either way. It is, however,
    recorded in ``hints.th_title_rows`` so TRM still excludes it from
    column-key construction.
    """
    td = et.table_data

    if td.data.size == 0:
        empty_hints = HeaderHints(
            col_header_rows=frozenset(),
            th_title_rows=frozenset(),
            stripped_titles=frozenset(),
        )
        return ExtractedTable(raw_html=et.raw_html, table_data=td, header_hints=empty_hints)

    # Phase 0: strip leading all-empty rows before detectors run.
    # An all-empty leading row (e.g. ``<tr><td></td><td></td></tr>``) is
    # not a title — there is no text to record in ``stripped_titles`` —
    # but if left in place it breaks ``find_col_header_rows``, which
    # breaks at row 0 when row 0 isn't a ``<th>`` row and would lose the
    # real header block behind it. Physically remove these rows so the
    # detectors see the table as if they were never there.
    leading_empty = _leading_all_empty_row_indices(td)
    if leading_empty:
        td, _ = _remove_rows(td, leading_empty)

    # Run detectors against a normalized copy of the table so the
    # row-set decisions match what TRM previously did when it ran
    # extract_header_info() on its already-normalized table. We then
    # apply the resulting row removals to the *original* table_data so
    # GriTS / downstream consumers still see un-normalized cell text.
    from extract_bench.evaluation.metrics.parse.table_record_match_metric import normalize_table

    norm_td = normalize_table(td)
    nominal = norm_td.header_rows if norm_td.header_rows else set()
    if max_top_title_rows > 0:
        td_titles = detect_td_title_rows(norm_td, nominal)
    else:
        td_titles = set()
    col_header_rows = find_col_header_rows(norm_td, nominal, td_titles)
    th_titles = detect_th_title_rows(norm_td, col_header_rows, max_top=max_top_title_rows)

    bottom_title: int | None = None
    if col_header_rows and max(col_header_rows) in th_titles:
        bottom_title = max(col_header_rows)
    top_th_titles = th_titles - ({bottom_title} if bottom_title is not None else set())

    stripped_titles = collect_stripped_titles(norm_td, th_titles, td_titles)

    to_remove = frozenset(td_titles | top_th_titles)
    trimmed_td, old_to_new = _remove_rows(td, to_remove)

    new_col_header_rows = frozenset(old_to_new[r] for r in col_header_rows if r in old_to_new)
    new_th_title_rows = frozenset(old_to_new[r] for r in th_titles if r in old_to_new)

    hints = HeaderHints(
        col_header_rows=new_col_header_rows,
        th_title_rows=new_th_title_rows,
        stripped_titles=frozenset(stripped_titles),
    )
    return ExtractedTable(raw_html=et.raw_html, table_data=trimmed_td, header_hints=hints)
