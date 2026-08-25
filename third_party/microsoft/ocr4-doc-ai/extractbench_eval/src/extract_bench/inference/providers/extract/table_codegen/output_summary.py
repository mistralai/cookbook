"""Output summary: fill-rate + value-distribution of the model's OWN extraction.

After a generated extractor runs, this distills the produced output into per-field
fill rates and value distributions (cardinality + top values), so the model can
RECONCILE them against what the source document implies — rather than being told what
to fix. A field far less filled than the document implies (e.g. an alias the document
states in every record but the script captured for only a fifth of them), or one
collapsed to a single value (a count assignment that mapped every record to the same
count), is a likely mapping bug. It is deliberately non-prescriptive: it states facts
about the output, not corrections — some fields are legitimately sparse or constant,
and the model has the document in context to judge which.

Motivation (lp_06): review draws echo the full output and the model still
eyeball-blessed ``actor_alias`` as correct while it was 79% null, and only one draw
caught ``primary_count`` collapsing to a single value. A scannable 481-record list
invites that confirmation bias; a printed ``actor_alias: 103/481 (21%)`` or
``primary_count: 1 distinct {…}`` does not.

Document-agnostic: it summarizes the OUTPUT structure (not parsed tables), so unlike
the column-coverage check it fires on prose documents with no tables. Runs in the
parent process (inside the run_script tool handler).

When the provenance experiment is on, ``summarize_output(..., provenance=True)`` also
appends a PROVENANCE COVERAGE block (per-record source-page fill rate + page
distribution) and drops the reserved ``_provenance`` key from the normal per-field
summary so it doesn't read as a junk field.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .schema_utils import PROVENANCE_KEY

# Show the full value distribution when a field has at most this many distinct
# values (degeneracy/enum-skew lives here); above it, show distinct count + samples.
_MAX_DISTINCT_SHOWN = 12
# Cap how many (value: count) pairs we print for a shown distribution.
_TOP_K = 8
# Sample values to show for a high-cardinality field (to spot format mistakes,
# e.g. paragraph_id "8" vs "1.01").
_SAMPLES = 3
# Left-pad field names to this width for a readable column.
_NAME_W = 22


def _is_filled(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    if isinstance(v, (list, dict)):
        return len(v) > 0
    return True  # numbers (incl. 0), bools — present is present


def _summ_scalar(values: list[Any], n: int) -> str:
    nonnull = [v for v in values if _is_filled(v)]
    filled = len(nonnull)
    keys = [str(v) for v in nonnull]
    distinct = list(dict.fromkeys(keys))  # order-preserving unique
    d = len(distinct)
    pct = f" ({round(100 * filled / n)}%)" if n and filled < n else ""
    head = f"{filled}/{n} filled{pct}, {d} distinct"
    if d == 0:
        return head
    if d <= _MAX_DISTINCT_SHOWN:
        top = Counter(keys).most_common(_TOP_K)
        body = ", ".join(f"{k}: {c}" for k, c in top)
        more = ", …" if d > _TOP_K else ""
        return f"{head}  →  {{{body}{more}}}"
    return f"{head}  e.g. {', '.join(distinct[:_SAMPLES])}"


def _summ_list(values: list[Any], n: int) -> str:
    """A list-valued subfield (e.g. defendants, referenced_exhibits): report the
    fraction of records with a non-empty list and the distribution of the flattened
    element values."""
    non_empty = sum(1 for v in values if isinstance(v, list) and v)
    flat = [str(x) for v in values if isinstance(v, list) for x in v if _is_filled(x)]
    distinct = list(dict.fromkeys(flat))
    d = len(distinct)
    pct = f" ({round(100 * non_empty / n)}%)" if n and non_empty < n else ""
    head = f"{non_empty}/{n} non-empty{pct}, {d} distinct elems"
    if d == 0:
        return head
    if d <= _MAX_DISTINCT_SHOWN:
        top = Counter(flat).most_common(_TOP_K)
        body = ", ".join(f"{k}: {c}" for k, c in top)
        more = ", …" if d > _TOP_K else ""
        return f"{head}  →  {{{body}{more}}}"
    return f"{head}  e.g. {', '.join(distinct[:_SAMPLES])}"


def _summ_record_array(records: list[dict[str, Any]], skip: str | None = None) -> list[str]:
    """Per-subfield summary across a list of record dicts. ``skip`` drops one
    subfield name (used to keep the reserved ``_provenance`` key out of the normal
    field summary — it gets its own coverage block instead)."""
    subfields: list[str] = list(dict.fromkeys(k for r in records if isinstance(r, dict) for k in r if k != skip))
    n = len(records)
    lines: list[str] = []
    for f in subfields:
        col = [r.get(f) if isinstance(r, dict) else None for r in records]
        is_list = any(isinstance(v, list) for v in col)
        label = f"{f} (list)" if is_list else f
        summ = _summ_list(col, n) if is_list else _summ_scalar(col, n)
        lines.append(f"  {label:<{_NAME_W}}: {summ}")
    return lines


def _page_sort_key(p: Any) -> tuple[int, float, str]:
    """Sort pages numerically when possible (so 2 < 10), non-numeric tokens last."""
    try:
        return (0, float(p), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(p))


def _provenance_coverage(output: dict[str, Any]) -> str:
    """A page-coverage summary of the reserved ``_provenance`` key the script attaches
    to each record (only built when the provenance experiment is on). Per record array
    (and the top-level object for a flat schema): how many records carry a source page
    vs are missing/null, plus the page distribution — so the model can reconcile against
    how many pages the document actually spans. Every record on one page, or many with no
    page, usually means the page wiring is wrong (not read from the source item's
    ``page_num``). Non-prescriptive, like the rest of the summary."""
    lines: list[str] = []

    def _one(label: str, recs: list[Any]) -> None:
        n = len(recs)
        pages: list[Any] = []
        missing = 0
        for r in recs:
            prov = r.get(PROVENANCE_KEY) if isinstance(r, dict) else None
            page = prov.get("page") if isinstance(prov, dict) else None
            if page is None:
                missing += 1
            else:
                pages.append(page)
        miss_txt = f", {missing} missing/null" if missing else ""
        counts = Counter(pages)
        head = f"  {label:<{_NAME_W}}: {len(pages)}/{n} carry a page{miss_txt}, {len(counts)} distinct pages"
        if pages:
            shown = sorted(counts.items(), key=lambda kv: _page_sort_key(kv[0]))[:_TOP_K]
            body = ", ".join(f"{k}: {c}" for k, c in shown)
            more = f", … (+{len(counts) - _TOP_K} more)" if len(counts) > _TOP_K else ""
            head += f"  →  {{{body}{more}}}"
        lines.append(head)

    for key, val in output.items():
        if isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
            _one(key, val)
    top = output.get(PROVENANCE_KEY)  # flat schema: provenance on the top-level object
    if isinstance(top, dict):
        page = top.get("page")
        lines.append(f"  {'(top-level)':<{_NAME_W}}: page={page if page is not None else 'null'}")
    if not lines:
        return ""
    return (
        "PROVENANCE COVERAGE — the reserved _provenance.page you attached to each record. RECONCILE "
        "against how many pages the DOCUMENT spans: records with NO page, or all records collapsing "
        "onto a single page, usually mean the page wiring is wrong — read each record's page from its "
        "source item's page_num, do not default or hardcode it.\n" + "\n".join(lines)
    )


def summarize_output(output: Any, provenance: bool = False) -> str:
    """Return a fill-rate + value-distribution summary of ``output`` (or '' if it is
    not a dict). Appended to the run_script result so the model can reconcile its own
    output's coverage against the source document. When ``provenance`` is True, the
    reserved ``_provenance`` key is dropped from the per-field summary and a separate
    PROVENANCE COVERAGE block (per-record source-page fill + page distribution) is
    appended."""
    if not isinstance(output, dict):
        return ""
    skip = PROVENANCE_KEY if provenance else None
    blocks: list[str] = []
    scalars: list[str] = []
    for key, val in output.items():
        if provenance and key == PROVENANCE_KEY:
            continue  # reserved key — covered by the PROVENANCE COVERAGE block, not here
        if isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
            blocks.append(f"{key}: {len(val)} records\n" + "\n".join(_summ_record_array(val, skip=skip)))
        elif isinstance(val, list):
            blocks.append(f"{key}: {len(val)} items\n  {'(elements)':<{_NAME_W}}: {_summ_list([val], 1)}")
        else:
            scalars.append(f"{key}={'filled' if _is_filled(val) else 'NULL/empty'}")
    if scalars:
        blocks.append("top-level scalars: " + ", ".join(scalars))
    if not blocks:
        return ""
    summary = (
        "OUTPUT SUMMARY — fill rates and value distributions of fields in YOUR output. RECONCILE each "
        "against what the DOCUMENT implies that field should look like: a field far less filled "
        "than the source states, or collapsed to a single value, is usually a mapping bug — but "
        "some fields are legitimately sparse or constant, so judge against the document, do not "
        "just pad them.\n\n" + "\n\n".join(blocks)
    )
    if provenance:
        cov = _provenance_coverage(output)
        if cov:
            summary += "\n\n" + cov
    return summary
