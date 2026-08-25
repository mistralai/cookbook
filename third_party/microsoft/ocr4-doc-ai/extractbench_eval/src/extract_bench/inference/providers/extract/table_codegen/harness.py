"""Drive a model to author a Python extractor over a parsed document.

The agent is shown the Document/Table API, the (resolved) OUTPUT schema, and a
compressed baseline context (prose + first-rows of each table). It writes a
``extract_direct(document) -> dict`` script via the write_file/submit tools; the
submit gate validates it (non-empty, parses, defines the contract function, runs
over the real document **in a sandboxed subprocess** — see ``sandbox.py`` for
the trust model — returns a dict, references only schema fields, includes every
required field, and every non-null value matches its declared type/enum) and the
validated output is returned.

Parameterized entrypoint: ``generate_extractor(document, output_schema, ...)``.
Ported from doc-extraction-gt (generalized/extract_harness/generate_extract.py),
stripped of experiment scaffolding (fixed doc, on-disk target file, CLI, phase-1).
"""

from __future__ import annotations

import ast
import json
import re
import traceback
from typing import Any

from .agent_loop import Tool, ToolLoopDone, run_agent
from .document import Document, render_document, render_page
from .document import compressed_table as _pages_compressed_table
from .items_document import ItemsDocument, ItemsPage, render_item, render_items_document
from .sandbox import run_extractor_subprocess
from .schema_utils import (
    PROVENANCE_KEY,
    invalid_output_values,
    missing_required_fields,
    render_output_schema,
    resolve_refs,
    unknown_output_fields,
)

FUNCTION_NAME = "extract_direct"
DEFAULT_SCRIPT_TIMEOUT_S = 60.0


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def validate(
    content: str,
    document: Document | ItemsDocument,
    output_schema: dict[str, Any],
    *,
    timeout_s: float = DEFAULT_SCRIPT_TIMEOUT_S,
    reserved_keys: frozenset[str] = frozenset(),
) -> tuple[bool, str, Any]:
    """Validate a generated extractor: non-empty, parses, defines
    ``extract_direct``, runs over the real document (in a sandboxed subprocess
    with a hard timeout), returns a dict whose fields are all in the
    (ref-resolved) schema and include every required field. ``reserved_keys`` are
    tolerated at every object level by the unknown-field check (e.g. the provenance
    experiment passes ``{PROVENANCE_KEY}``); empty by default → gate unchanged."""
    schema = resolve_refs(output_schema)  # inline $defs/$ref so field checks can descend
    if not content.strip():
        return (
            False,
            f"The code file is empty — write a non-empty script defining {FUNCTION_NAME}(document) -> dict.",
            None,
        )

    # Fast parent-side checks (no execution): syntax + contract function.
    try:
        tree = ast.parse(content, "<generated>")
    except SyntaxError:
        return False, f"Your script failed to parse:\n{traceback.format_exc()[-1500:]}", None
    if not any(isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == FUNCTION_NAME for n in tree.body):
        return False, f"Your file must define a function {FUNCTION_NAME}(document) -> dict.", None

    # Execution happens in an isolated subprocess (scrubbed env, hard timeout).
    document_module = "items_document" if isinstance(document, ItemsDocument) else "document"
    ok, err, out = run_extractor_subprocess(
        content,
        document.to_pages(),
        function_name=FUNCTION_NAME,
        timeout_s=timeout_s,
        document_module=document_module,
    )
    if not ok:
        return False, f"Your {FUNCTION_NAME} failed to run over the document:\n{err}", None

    if not isinstance(out, dict):
        return False, f"{FUNCTION_NAME} must return a dict (got {type(out).__name__}).", None
    unknown = unknown_output_fields(out, schema, allow_keys=reserved_keys)
    if unknown:
        return (
            False,
            (
                f"Your output includes fields not in the OUTPUT schema: {unknown[:12]}"
                f"{' …' if len(unknown) > 12 else ''}. Use only the output schema's fields."
            ),
            None,
        )
    missing = missing_required_fields(out, schema)
    if missing:
        uniq = sorted({re.sub(r"\[\d+\]", "[]", m) for m in missing})
        return (
            False,
            (
                f"Your output is missing required fields ({len(missing)} occurrence(s); the schema "
                f"marks these required): {uniq[:12]}{' …' if len(uniq) > 12 else ''}. "
                "Every required field must be present on each object (a null value is allowed)."
            ),
            None,
        )
    invalid = invalid_output_values(out, schema)
    if invalid:
        # Collapse array indices so 27 copies of the same per-record mistake read
        # as one line; distinct values at the same path survive the dedup.
        uniq = sorted({re.sub(r"\[\d+\]", "[]", m) for m in invalid})
        return (
            False,
            (
                f"Your output has values that don't match the OUTPUT schema's declared types/enums "
                f"({len(invalid)} occurrence(s)):\n" + "\n".join(uniq[:10]) + ("\n…" if len(uniq) > 10 else "") + "\n"
                "Coerce each value to the declared type, and map onto enum tokens EXACTLY as listed "
                "in the schema (a null value is allowed where you cannot determine one)."
            ),
            None,
        )
    return (
        True,
        "OK — non-empty, runs over the document, returns a dict with only valid output fields "
        "and schema-conformant value types/enums.",
        out,
    )


# Per-array cap when echoing a run's output back to the model. None (the default)
# shows the FULL output — the whole point of the run preview is to see exactly what
# the script produced, so the model can catch a mapping that is valid-but-wrong. A
# small integer cap is for cheap smoke tests, where echoing every record each run is
# wasteful; it trades self-review fidelity (tail records become invisible) for cost.
DEFAULT_OUTPUT_PREVIEW_MAX_ITEMS: int | None = None


def _compact_for_preview(value: Any, max_items: int | None) -> Any:
    """Shrink long arrays for the run-output preview: keep the first ``max_items``
    elements of every list and replace the tail with a ``(+K more, M total)``
    sentinel. ``max_items is None`` returns the value unchanged (full output)."""
    if max_items is None:
        return value
    if isinstance(value, list):
        shown = [_compact_for_preview(v, max_items) for v in value[:max_items]]
        if len(value) > max_items:
            shown.append(f"... (+{len(value) - max_items} more, {len(value)} total)")
        return shown
    if isinstance(value, dict):
        return {k: _compact_for_preview(v, max_items) for k, v in value.items()}
    return value


def render_output_preview(out: Any, max_items: int | None = DEFAULT_OUTPUT_PREVIEW_MAX_ITEMS) -> str:
    """Pretty-print a run's output for the model to inspect. Full by default; with
    an integer ``max_items`` long arrays are compacted (see
    :func:`_compact_for_preview`). Scalars are always shown in full."""
    return json.dumps(_compact_for_preview(out, max_items), indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# the edit tool (str_replace) — surgical edits instead of full rewrites
# --------------------------------------------------------------------------- #
def _edit_context(content: str, idx: int, new_len: int, pad: int = 2) -> str:
    """The changed line(s) ± ``pad`` lines, line-numbered, so the model can re-sync
    after an edit WITHOUT re-echoing the whole script (which would undo the saving)."""
    lines = content.splitlines()
    start_line = content[:idx].count("\n")
    end_line = content[: idx + new_len].count("\n")
    lo, hi = max(0, start_line - pad), min(len(lines), end_line + pad + 1)
    return "\n".join(f"{i + 1:4d}| {lines[i]}" for i in range(lo, hi))


def _make_str_replace_tool(holder: dict[str, str]) -> Tool:
    """A ``str_replace(old_str, new_str)`` tool that edits the held script in place:
    apply iff ``old_str`` matches EXACTLY once (else an actionable error), then echo
    a small context window around the change. Shared by the one-shot and review
    tool sets so the edit semantics can't diverge between them."""

    def str_replace(args: dict[str, Any]) -> str:
        old, new = args.get("old_str", ""), args.get("new_str", "")
        content = holder["content"]
        if not old:
            return "str_replace: old_str is empty — provide the exact text to replace."
        n = content.count(old)
        if n == 0:
            return (
                "str_replace: old_str NOT FOUND (0 matches) — it may be stale after a prior edit. "
                "Quote the exact current text (use the context the last edit/run echoed)."
            )
        if n > 1:
            return (
                f"str_replace: old_str matches {n} times; it must match EXACTLY once. "
                "Add surrounding lines to make the match unique."
            )
        idx = content.index(old)
        holder["content"] = content[:idx] + new + content[idx + len(old) :]
        return (
            "Edited (1 replacement). Context around the change:\n"
            + _edit_context(holder["content"], idx, len(new))
            + "\nRe-run the script (run_script) — or submit, in one-shot — to apply and check the change."
        )

    return Tool(
        "str_replace",
        "Replace EXACTLY ONE occurrence of old_str with new_str in the current script (error if it matches 0 "
        "or >1 times — add surrounding context to disambiguate). Use this to fix a specific line instead of "
        "rewriting the whole script with write_file.",
        {
            "type": "object",
            "properties": {"old_str": {"type": "string"}, "new_str": {"type": "string"}},
            "required": ["old_str", "new_str"],
        },
        str_replace,
    )


def _make_oneshot_tools(
    document: Document | ItemsDocument,
    output_schema: dict[str, Any],
    timeout_s: float,
    edit_tool: bool = False,
    provenance: bool = False,
) -> tuple[list[Tool], dict[str, Any]]:
    """The original one-shot tools: ``submit`` validates AND finalizes in one step
    (the model never sees its output). Kept byte-for-byte as the default so the
    baseline pipelines are unaffected by the review-loop experiment. ``edit_tool``
    adds ``str_replace`` for surgical edits (default off → baseline unchanged).
    ``provenance`` tells the gate to tolerate the reserved ``_provenance`` key."""
    holder: dict[str, Any] = {"content": ""}
    reserved = frozenset({PROVENANCE_KEY}) if provenance else frozenset()

    def write_file(args: dict[str, Any]) -> str:
        holder["content"] = args.get("content", "")
        return f"wrote {len(holder['content'])} chars to the script."

    def submit(args: dict[str, Any]) -> str:
        ok, msg, out = validate(holder["content"], document, output_schema, timeout_s=timeout_s, reserved_keys=reserved)
        if ok:
            raise ToolLoopDone({"output": out, "script": holder["content"], "message": msg})
        return f"REJECTED. {msg}"

    tools = [
        Tool(
            "write_file",
            f"Write (overwrite) the single target script. It must define {FUNCTION_NAME}(document) -> dict.",
            {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
            write_file,
        )
    ]
    if edit_tool:
        tools.append(_make_str_replace_tool(holder))
    tools.append(
        Tool(
            "submit",
            "Validate the current script: non-empty, runs over the real document, returns a dict, references "
            "only valid output-schema fields, includes every field the schema marks required on each object, "
            "and every value matches its declared type/enum (null allowed).",
            {"type": "object", "properties": {}},
            submit,
        )
    )
    return tools, holder


def make_tools(
    document: Document | ItemsDocument,
    output_schema: dict[str, Any],
    *,
    timeout_s: float = DEFAULT_SCRIPT_TIMEOUT_S,
    preview_max_items: int | None = DEFAULT_OUTPUT_PREVIEW_MAX_ITEMS,
    clock: dict[str, int] | None = None,
    review_loop: bool = False,
    coverage_warning: bool = False,
    output_summary: bool = False,
    view_tools: bool = False,
    edit_tool: bool = False,
    echo_run_output: bool = True,
    provenance: bool = False,
) -> tuple[list[Tool], dict[str, Any]]:
    """Build the agent's tools. The script content is held in the returned dict
    (no on-disk file) so callers can recover the final script.

    ``provenance=True`` tells the validity gate to tolerate the reserved
    ``_provenance`` key the script attaches to each record, and (when
    ``output_summary`` is also on) augments the run summary with per-record
    page-coverage. Works in either path; default off → gate/summary unchanged.

    ``view_tools=True`` (review loop only) adds ``view_document`` (re-view a page /
    a table in full) and ``view_output`` (re-view a field or the whole last-run
    output, optionally only its null records) — on-demand inspection so the model
    can verify a suspicious field against the source instead of rationalizing it.

    ``edit_tool=True`` adds ``str_replace`` (surgical single-occurrence edits)
    alongside ``write_file`` — cheaper than re-emitting the whole script each turn.
    Works in both paths.

    ``review_loop=False`` (default) is the original one-shot path — ``submit``
    validates and finalizes in one step. ``review_loop=True`` is the two-phase
    experiment: the model must ``run_script`` (which validates AND shows it the
    actual extracted output) before ``submit`` will finalize. ``submit`` is then a
    thin gate over the cached run — it never re-runs, returns the output the last
    run already produced, and requires that run to have happened on a STRICTLY
    EARLIER turn (a model emits several tool calls per response and only sees their
    results next turn, so a same-turn run+submit would commit output it never saw;
    ``clock["turn"]`` is advanced by the agent loop's ``on_turn_start``).

    ``coverage_warning=True`` (review loop only) augments each viable ``run_script``
    with a column-coverage check pointing the model at table rows it likely dropped
    or merged (see :mod:`coverage`)."""
    if not review_loop:
        return _make_oneshot_tools(document, output_schema, timeout_s, edit_tool=edit_tool, provenance=provenance)

    holder: dict[str, Any] = {"content": ""}
    clock = {"turn": 0} if clock is None else clock
    reserved = frozenset({PROVENANCE_KEY}) if provenance else frozenset()
    # The most recent run_script result, keyed to the exact content it ran over and
    # the turn it ran on. submit consults this instead of re-validating: it may
    # finalize only when a viable run exists for the content currently in `holder`,
    # produced on an earlier turn (so its output was already returned to the model).
    run_state: dict[str, Any] = {"content": None, "ok": False, "output": None, "turn": -1}

    def write_file(args: dict[str, Any]) -> str:
        holder["content"] = args.get("content", "")
        return (
            f"wrote {len(holder['content'])} chars to the script. "
            "Call run_script to run it over the document and see its output."
        )

    def run_script(args: dict[str, Any]) -> str:
        ok, msg, out = validate(holder["content"], document, output_schema, timeout_s=timeout_s, reserved_keys=reserved)
        run_state.update(content=holder["content"], ok=ok, output=out, turn=clock["turn"])
        if not ok:
            return f"NOT VIABLE — submit stays blocked until this passes. {msg}"
        # Remember the last validated run so generate_extractor can salvage it if the
        # model never reaches submit (e.g. runs out of turns mid-iteration).
        holder["last_viable"] = {"output": out, "content": holder["content"]}
        if echo_run_output:
            cap = "full output" if preview_max_items is None else f"arrays truncated to {preview_max_items} items each"
            head = (
                "Ran over the document; the validity gate PASSED. Below is the actual output your script "
                f"produced ({cap}) — INSPECT it for wrong-column values, flipped signs, mis-parsed "
                "dates/amounts, or missing records before you finalize:\n\n"
                f"{render_output_preview(out, preview_max_items)}"
            )
        else:
            counts = (
                ", ".join(f"{k}={len(v)}" for k, v in out.items() if isinstance(v, list))
                if isinstance(out, dict)
                else ""
            )
            # With the view tools on, point the model at them; without them it has only the
            # summary below (+ the document already in context) to reconcile against.
            inspect = (
                "Inspect it with view_output / view_document (and the summary below)"
                if view_tools
                else "Reconcile the per-field summary below against the document"
            )
            head = (
                "Ran over the document; the validity gate PASSED. The full output is NOT shown here — "
                f"record counts: {counts or '(no top-level arrays)'}. {inspect} and verify before you finalize."
            )
        parts = [head]
        if coverage_warning:
            from .coverage import coverage_warnings

            warn = coverage_warnings(document, out)
            if warn:
                parts.append(warn)
        if output_summary:
            from .output_summary import summarize_output

            summ = summarize_output(out, provenance=provenance)
            if summ:
                parts.append(summ)
        parts.append(
            "If the output is correct, call submit to finalize it. Otherwise fix the script with write_file "
            "and run_script again."
        )
        return "\n\n".join(parts)

    def submit(args: dict[str, Any]) -> str:
        if run_state["content"] != holder["content"]:
            return (
                "REJECTED — submit is only available after run_script produces a viable output for the "
                "CURRENT script. You either have not run the current script yet, or edited it since the last "
                "run. Call run_script first."
            )
        if not run_state["ok"]:
            return (
                "REJECTED — your most recent run_script did not produce a viable output, so there is nothing "
                "valid to submit. Fix the issues it reported and run_script again."
            )
        if clock["turn"] <= run_state["turn"]:
            return (
                "REJECTED — you called submit in the same turn as run_script, before its output was returned "
                "to you. Wait for the run_script result, read the output, and call submit on a later turn."
            )
        raise ToolLoopDone(
            {"output": run_state["output"], "script": holder["content"], "message": "submitted after a viable run"}
        )

    def view_document(args: dict[str, Any]) -> str:
        """Re-view a page in full (tables un-truncated), or one table on it. Reuses
        the same renderers as the baseline context, so what's shown can't diverge."""
        page_num = args.get("page")
        if not isinstance(page_num, int):
            return "view_document: pass an integer `page`. Pages: " + str([p.page_num for p in document.pages][:40])
        pg = document.page(page_num)
        if pg is None:
            return f"view_document: no page {page_num}. Available pages: {[p.page_num for p in document.pages][:40]}"
        ti = args.get("table_index")

        def _bad_ti(n: int) -> str:
            return f"view_document: page {page_num} has {n} table(s) (valid table_index 0..{n - 1})."

        if isinstance(pg, ItemsPage):
            tables = pg.items(types="table")
            if ti is not None:
                if not isinstance(ti, int) or not (0 <= ti < len(tables)):
                    return _bad_ti(len(tables))
                return render_item(tables[ti], full_tables=True)
            rendered = "\n".join(render_item(it, full_tables=True) for it in pg.items())
            return rendered or f"(page {page_num}: no items)"
        # pages (markdown) Page
        if ti is not None:
            tbls = pg.tables()
            if not isinstance(ti, int) or not (0 <= ti < len(tbls)):
                return _bad_ti(len(tbls))
            return _pages_compressed_table(tbls[ti], n_rows=None)
        return render_page(pg, n_rows=None) or f"(page {page_num}: empty)"

    def view_output(args: dict[str, Any]) -> str:
        """Re-view the last viable run's output: the whole thing, one field, or just
        the records where a field is null. Reuses ``render_output_preview`` (the same
        renderer as the run echo)."""
        from .output_summary import _is_filled

        if not run_state["ok"] or run_state["output"] is None:
            return "view_output: no viable run yet — call run_script first to produce an output to inspect."
        out = run_state["output"]
        field, only_nulls = args.get("field"), bool(args.get("only_nulls"))
        if not field:
            return render_output_preview(out, None)
        if "." in field:  # array.subfield
            arr_key, sub = field.split(".", 1)
            arr = out.get(arr_key) if isinstance(out, dict) else None
            if not isinstance(arr, list):
                return f"view_output: '{arr_key}' is not a top-level array. Top-level keys: {list(out)}."
            recs = [r for r in arr if isinstance(r, dict)]
            if only_nulls:
                hits = [r for r in recs if not _is_filled(r.get(sub))]
                label = f"{arr_key}[] where {sub} is null/empty ({len(hits)} of {len(recs)})"
                return render_output_preview({label: hits}, None)
            return render_output_preview({field: [r.get(sub) for r in recs]}, None)
        if not isinstance(out, dict) or field not in out:
            keys = list(out) if isinstance(out, dict) else "(output is not a dict)"
            return f"view_output: no field '{field}'. Top-level keys: {keys}."
        if only_nulls:
            return "view_output: only_nulls applies to an 'array.subfield' field (e.g. 'charged_facts.actor_alias')."
        return render_output_preview({field: out[field]}, None)

    tools = [
        Tool(
            "write_file",
            f"Write (overwrite) the single target script. It must define {FUNCTION_NAME}(document) -> dict.",
            {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
            write_file,
        )
    ]
    if edit_tool:
        tools.append(_make_str_replace_tool(holder))
    tools.append(
        Tool(
            "run_script",
            "Run the current script over the real document and return its ACTUAL output for you to inspect, "
            "plus a validity check (parses, runs, returns a dict, references only valid output-schema fields, "
            "includes every required field, every value matches its declared type/enum). You must run_script "
            "and get a viable output before you can submit.",
            {"type": "object", "properties": {}},
            run_script,
        )
    )
    if view_tools:
        tools += [
            Tool(
                "view_document",
                "Re-view part of the parsed document: view_document(page=N) shows that page's items in FULL "
                "(tables un-truncated); add table_index to show just one table's full rows. Use it to check the "
                "source for a field you suspect your script got wrong or dropped.",
                {
                    "type": "object",
                    "properties": {"page": {"type": "integer"}, "table_index": {"type": "integer"}},
                    "required": ["page"],
                },
                view_document,
            ),
            Tool(
                "view_output",
                "Re-view your last run's output: view_output() shows the whole output; "
                "view_output(field='charged_facts.actor_alias') shows that field across records; add "
                "only_nulls=true to list ONLY the records where it is null/empty (then view_document the matching "
                "source to confirm the value is genuinely absent before accepting it).",
                {"type": "object", "properties": {"field": {"type": "string"}, "only_nulls": {"type": "boolean"}}},
                view_output,
            ),
        ]
    tools.append(
        Tool(
            "submit",
            "Finalize and return the output from your most recent viable run_script. Only available after "
            "run_script has produced a viable output for the current script; if you edited the script since "
            "the last run, run_script again first.",
            {"type": "object", "properties": {}},
            submit,
        )
    )
    return tools, holder


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #
API_REFERENCE = """Your script receives one argument, `document`, with this API:

  document.tables()        -> list[Table]   # every table, all pages, in order
  document.pages           -> list[Page]
  document.page(n)         -> Page | None    # the page whose .page_num == n

  page.page_num            -> int
  page.non_table_text      -> str            # the page's prose, each table lifted
                                             #   to a "[page P table I]" marker
  page.tables()            -> list[Table]    # tables on this page

  table.text_grid()        -> list[list[str]]   # dense, span-resolved cells (raw strings)
  table.header()           -> list[str]         # row-0 cell texts (quick role detection)
  table.header_row_indices()-> list[int]        # leading all-<th> header rows
  table.context_before     -> str               # heading/breadcrumb text just above the table
  table.context_after      -> str
  table.caption            -> str
  table.page_num           -> int
  table.table_index        -> int               # index within its page
  table.n_rows, table.n_cols-> int

You may import value helpers:  from common import money, iso_date
  money('$1,086.44') -> 1086.44 ; money('') -> None
  iso_date('08/17', 2025) -> '2025-08-17'

Work from raw cell strings via text_grid() when a table needs positional logic
(merging page-wrapped rows, un-stacking repeated column groups). Use header() /
context_before to tell tables apart by role. Do NOT hardcode row values.

The parsed text is INCONSISTENTLY FORMATTED across structurally-identical
regions: the same field may carry Markdown emphasis in some places and not
others (e.g. `**VIN**` JTD… in one block, plain `VIN` JTD… in another), and the
same semantic marker may render as different tokens (e.g. a checked box as `[x]`
in one table and `[checkmark]` in another). Make your regexes and string matching
TOLERANT of this — allow optional `**`/`__` around labels, and match a marker
against a SET of accepted tokens — rather than keying on a single exact form, or
you will silently capture only the subset that matches."""

API_REFERENCE_ITEMS = """Your script receives one argument, `document`, with this API:

  document.items()                  -> list[Item]  # every item, all pages, in reading order
  document.items(types="table")     -> list[Item]  # only this type (str or iterable of str)
  document.items(exclude={"header", "footer"}) -> list[Item]  # all but these types
  document.pages                    -> list[Page]
  document.page(n)                  -> Page | None  # the page whose .page_num == n

  page.page_num                     -> int
  page.items(types=..., exclude=...)-> list[Item]   # this page only, in reading order

  item.type     -> str          # 'text' | 'heading' | 'table' | 'list' | 'header' | 'footer' | ...
  item.md       -> str          # markdown representation (preserves formatting)
  item.value    -> str          # plain text without markdown ('' on tables/containers)
  item.level    -> int | None   # heading level (headings only)
  item.html     -> str          # tables: the HTML table ('' otherwise)
  item.rows     -> list[list]   # tables: cell grid of str/number/None ([] otherwise)
  item.children -> list[Item]   # list/header/footer: their nested items
  item.page_num -> int

You may import value helpers:  from common import money, iso_date
  money('$1,086.44') -> 1086.44 ; money('') -> None
  iso_date('08/17', 2025) -> '2025-08-17'

Iterate items in order and filter by type — e.g. walk headings to track which
section you are in, take prose records from text items, and read tables from
item.rows (already split into cells; no HTML parsing needed). A table that
continues across pages appears as a separate table item on each page — stitch
continuation tables (same column shape, often no repeated header) when a record
sequence spans pages. Do NOT hardcode row values: every record in your output
must be produced by code reading the items, never typed out as literals — the
context below is a preview, and literals break on any other document.

The parsed text is INCONSISTENTLY FORMATTED across structurally-identical
regions: the same field may carry Markdown emphasis in some places and not
others, and the same semantic marker may render as different tokens. Make your
regexes and string matching TOLERANT of this — allow optional `**`/`__` around
labels, and match a marker against a SET of accepted tokens — rather than
keying on a single exact form, or you will silently capture only the subset
that matches."""

SYSTEM_DIRECT = """You are a code-generation agent. Write ONE Python script that maps a parsed
document straight onto a target OUTPUT schema, defining exactly:

    def extract_direct(document) -> dict:
        ...
        return output   # a dict matching the OUTPUT schema

Detect each table's role, taking into account any physical-layout quirks (rows/cells split across
page boundaries, horizontally-merged tables), coerce values, broadcast per-document constants,
and emit the schema shape. You do NOT need to reproduce any particular code —
just a correct mapping.

{tools}

{api}

=== OUTPUT schema (what extract_direct must return) ===
{output_schema}

=== DOCUMENT (full baseline context: prose + compressed tables) ===
{document}
"""

# One-shot: submit validates AND finalizes (the original behaviour).
TOOLS_ONESHOT = """Tools:
- write_file(content): write your script (overwrites each time).
- submit(): validate it — non-empty, runs over the real document, returns a dict,
  and references only OUTPUT-schema fields. Fix and resubmit until it passes."""

# Review loop: the model must run and inspect its output before it can submit.
TOOLS_REVIEW = """Tools (write -> run -> inspect -> submit):
- write_file(content): write your script (overwrites each time).
- run_script(): run your current script over the real document and see its ACTUAL
  output, plus a validity check (runs, returns a dict, references only OUTPUT-schema
  fields, includes every required field, every value matches its declared type/enum).
  READ the output back: a script can be valid yet map values wrongly — if a value is
  in the wrong column, has a flipped sign, or a mis-parsed date/amount, or records are
  missing, fix the script with write_file and run_script again.
- submit(): finalize. Available ONLY after run_script produces a viable output for the
  CURRENT script; it returns that output. If you edit the script, run_script again
  before you can submit.

Always run_script and inspect the output before submitting — the run output is your
chance to catch a mapping that passes validation but is wrong."""

# Optional tool docs the prompt assembler appends per the active flags.
TOOLS_EDIT_DOC = """
- str_replace(old_str, new_str): edit ONE occurrence in the current script in place
  (it must match exactly once). Once your draft is mostly right, prefer this over
  rewriting the whole script with write_file — it's cheaper and won't disturb the
  parts you already got correct."""

TOOLS_VIEW_DOC = """
- view_document(page[, table_index]): re-read a page of the SOURCE in full (tables
  un-truncated), or just one table on it.
- view_output(field[, only_nulls]): re-read your last run's OUTPUT — the whole thing,
  one field (e.g. 'charged_facts.actor_alias'), or, with only_nulls=true, only the
  records where that field is null/empty."""

ECHO_OFF_NOTE = """
NOTE: run_script will NOT echo your full output — only record counts plus a per-field
summary. Use view_output / view_document to inspect specific fields and verify them."""

# Echo-off note when the view tools are NOT enabled: there is nothing to inspect with, so
# the summary is the signal — reconcile it against the document already in context.
ECHO_OFF_NOTE_SUMMARY_ONLY = """
NOTE: run_script will NOT echo your full output — only record counts plus a per-field
summary. Reconcile that summary against the document above to catch under-filled or
mis-mapped fields before you finalize."""

VERIFY_INSTRUCTION = """VERIFY, don't rationalize: when a field is filled far below what the document implies,
or its values collapse to a single value, do NOT assume it's correct. Use
view_output(field, only_nulls=true) to pull the under-filled records, then
view_document(page) the matching source and check whether the value is genuinely
absent before you accept it — many "missing" values are really a too-narrow pattern."""

# Appended (verbatim) to the system prompt when provenance is on. Doc-agnostic — no
# doc specifics (per the no-calibration-specifics-in-prompts convention).
PROVENANCE_INSTRUCTION = """=== PROVENANCE (additional output) ===
For every object you emit in an output array (or the top-level object for a flat
schema), ALSO set a reserved key:  record["_provenance"] = {"page": N}
where N is the .page_num of the item you read that record from. If a record is
stitched from rows on more than one page, use the page where its primary row
appears; if a value is derived/computed with no single source item, use null.
This key is reserved — it is NOT part of the OUTPUT schema and will NOT be scored;
it only records where each record came from, so attach it from the source item's
page_num in code (do not hardcode page numbers)."""

# Appended (verbatim) to the system prompt for Google/Gemini only. gemini-3.x flash
# models tend to write a print()-the-structure "exploration" script that returns empty
# arrays as a first step — but stdout is NEVER shown back, and in one-shot `submit`
# commits that empty result as the final answer (the validity gate accepts an empty
# array as shape-valid). This nudges the model to author the real extractor on its first
# script. Doc-agnostic (no doc specifics, per the no-calibration-specifics convention).
GEMINI_DIRECT_NOTE = """=== WRITE THE EXTRACTOR DIRECTLY (do not "explore first") ===
Do NOT write a script that merely print()s the document structure and returns empty
arrays to inspect it first: print() output is NOT shown back to you, and submit COMMITS
whatever the script returns (an empty array passes the validity gate and becomes your
final answer). The full document content is already provided above. On your FIRST script,
write the complete extract_direct that reads the items/tables and actually POPULATES every
output array — never submit a result whose arrays are empty when the document has rows."""


def build_system(
    document: Document | ItemsDocument,
    output_schema: dict[str, Any],
    *,
    provider: str = "anthropic",
    review_loop: bool = False,
    edit_tool: bool = False,
    view_tools: bool = False,
    echo_run_output: bool = True,
    provenance: bool = False,
    turn_budget_hint: bool = False,
    max_turns: int = 10,
) -> str:
    if isinstance(document, ItemsDocument):
        api, document_text = API_REFERENCE_ITEMS, render_items_document(document)
    else:
        api, document_text = API_REFERENCE, render_document(document)
    tools_text = TOOLS_REVIEW if review_loop else TOOLS_ONESHOT
    if edit_tool:
        tools_text += TOOLS_EDIT_DOC
    if review_loop and view_tools:
        tools_text += TOOLS_VIEW_DOC
    if review_loop and not echo_run_output:
        tools_text += ECHO_OFF_NOTE if view_tools else ECHO_OFF_NOTE_SUMMARY_ONLY
    if review_loop and view_tools:
        tools_text += "\n\n" + VERIFY_INSTRUCTION
    system = SYSTEM_DIRECT.format(
        tools=tools_text,
        api=api,
        # resolve $defs/$ref + full descriptions so nested model structure is visible
        output_schema=render_output_schema(resolve_refs(output_schema)),
        document=document_text,
    )
    if provenance:
        system += "\n" + PROVENANCE_INSTRUCTION + "\n"
    if review_loop and turn_budget_hint:
        # State the budget up front; the live "turn X of N" counter rides every turn's
        # tool results (agent loop's turn_note). Without this the model is told nothing
        # about max_turns and tends to keep editing a correct output until it's cut off.
        system += (
            f"\nTURN BUDGET: you have at most {max_turns} turns, and submit must land on a turn AFTER the "
            "run_script whose output you are finalizing. So do not spend every turn editing — once run_script "
            "shows a correct output, call submit on the next turn rather than polishing. Every turn's results "
            "carry a reminder of the turn you are on and how many remain; there is no reward for extra edits "
            "once the output is right, and you forfeit the result entirely if you run out of turns.\n"
        )
    if provider == "google":
        system += "\n" + GEMINI_DIRECT_NOTE + "\n"
    return system


def turn_budget_message(turn: int, max_turns: int) -> str:
    """The live turn-budget reminder the agent loop appends to each turn's tool
    results when ``turn_budget_hint`` is on. ``turn`` is the 0-indexed loop index
    (the loop runs ``for turn in range(max_turns)``), so turn 0 is the 1st turn."""
    used = turn + 1
    remaining = max(0, max_turns - used)
    return (
        f"[TURN BUDGET — turn {used} of {max_turns}, {remaining} remaining. submit must run on a turn AFTER "
        "run_script, so leave a spare turn: once the output looks correct, call submit instead of editing "
        "further. No reward for polishing a correct result; you forfeit it if you run out of turns.]"
    )


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def generate_extractor(
    document: Document | ItemsDocument,
    output_schema: dict[str, Any],
    *,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    max_turns: int = 10,
    max_tokens: int = 16000,
    script_timeout_s: float = DEFAULT_SCRIPT_TIMEOUT_S,
    api_key: str | None = None,
    trace_result_limit: int | None = 2000,
    review_loop: bool = False,
    coverage_warning: bool = False,
    output_summary: bool = False,
    view_tools: bool = False,
    edit_tool: bool = False,
    echo_run_output: bool = True,
    preview_max_items: int | None = DEFAULT_OUTPUT_PREVIEW_MAX_ITEMS,
    provenance: bool = False,
    turn_budget_hint: bool = False,
    thinking_level: str | None = None,
    effort: str | None = None,
) -> dict[str, Any]:
    """Have the model author + validate an extractor, run it over ``document``,
    and return ``{status, output, script, trace, usage, turns, latency_s}``.
    ``output`` is the extraction dict (None if the agent never passed the gate).
    ``trace_result_limit`` clips tool results in the returned trace (default 2000
    chars; ``None`` keeps them whole for offline review). ``review_loop`` /
    ``coverage_warning`` / ``output_summary`` / ``view_tools`` / ``edit_tool`` /
    ``echo_run_output`` / ``preview_max_items`` / ``provenance`` select the
    run→inspect→submit experiment and its options (see :func:`make_tools`)."""
    # Shared turn counter: the agent loop advances it before each turn; the submit
    # gate reads it so a same-turn run+submit (output the model never saw) is refused.
    clock: dict[str, int] = {"turn": 0}
    tools, holder = make_tools(
        document,
        output_schema,
        timeout_s=script_timeout_s,
        clock=clock,
        review_loop=review_loop,
        coverage_warning=coverage_warning,
        output_summary=output_summary,
        view_tools=view_tools,
        edit_tool=edit_tool,
        echo_run_output=echo_run_output,
        preview_max_items=preview_max_items,
        provenance=provenance,
    )
    system = build_system(
        document,
        output_schema,
        provider=provider,
        review_loop=review_loop,
        edit_tool=edit_tool,
        view_tools=view_tools,
        echo_run_output=echo_run_output,
        provenance=provenance,
        turn_budget_hint=turn_budget_hint,
        max_turns=max_turns,
    )
    result = run_agent(
        provider,
        model,
        system,
        tools,
        max_turns=max_turns,
        max_tokens=max_tokens,
        api_key=api_key,
        on_turn_start=lambda turn: clock.update(turn=turn),
        trace_result_limit=trace_result_limit,
        thinking_level=thinking_level,
        turn_note=turn_budget_message if turn_budget_hint else None,
        effort=effort,
    )
    payload = result.get("payload") or {}
    status = result["status"]
    output = payload.get("output")
    script = payload.get("script", holder["content"])
    salvaged = False
    # If the agent ran out of turns (or stopped) WITHOUT submitting but had already
    # produced a viable run, return that last validated output rather than nothing —
    # tagged with a distinct *_salvaged status so it's never mistaken for a real submit.
    if output is None and status in ("max_turns", "stopped_no_tool"):
        last = holder.get("last_viable")
        if isinstance(last, dict) and last.get("output") is not None:
            output, script, salvaged = last["output"], last["content"], True
            status = f"{status}_salvaged"
    return {
        "status": status,
        "output": output,
        "script": script,
        "salvaged": salvaged,
        "trace": result.get("trace", []),
        "usage": result.get("usage", {}),
        "turns": result.get("turns"),
        "latency_s": result.get("latency_s"),
    }
