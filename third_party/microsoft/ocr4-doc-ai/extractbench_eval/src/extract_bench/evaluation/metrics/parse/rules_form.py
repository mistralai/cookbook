"""Form field test rule.

A `form_field` rule locates a labeled field in the parsed markdown/HTML and
checks its value. Three value types are supported in v0.1: ``text``,
``checkbox``, and ``signature``. The matcher tries a small set of
high-confidence patterns:

- Bold-colon (``**Label:** value`` and ``**Label**: value``) — supports
  multiple bold-colon pairs on the same line.
- Plain colon on its own line (``Label: value``).
- 2-column markdown tables AND 2-column HTML tables (label in first cell,
  value in second).
- Per-line checkbox tokenization for inline groups, handling both
  glyph-first (``☐ Single ☑ Married``) and label-first
  (``Single ☐ Married ☑``) orderings.
- Markdown task-list checkboxes (``- [x] Label`` / ``- [ ] Label``).

When the rule has a ``page`` and the metric injects a ``parse_output``,
matching is scoped to that page's markdown only.
"""

from __future__ import annotations

import re
from typing import cast

from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from extract_bench.evaluation.metrics.parse.rules_base import (
    CELL_FUZZY_MATCH_THRESHOLD,
    ParseTestRule,
)
from extract_bench.evaluation.metrics.parse.rules_chart import normalize_number_string
from extract_bench.evaluation.metrics.parse.table_parsing import (
    TableData,
    parse_html_tables,
    parse_markdown_tables,
)
from extract_bench.evaluation.metrics.parse.test_types import ParseRuleType
from extract_bench.evaluation.metrics.parse.utils import normalize_text
from extract_bench.test_cases.parse_rule_schemas import ParseFormFieldRule

# Glyphs that represent a checked / unchecked state. Sourced from the most
# common Unicode shapes parsers emit when surfacing form widgets. The extra
# circle/dot glyphs (◉●⦿/○◯⊙) appear in Gemini and OpenAI outputs which mirror
# radio-button widgets; the extra X glyphs (⊠⊗) appear in IRS/USCIS forms.
_CHECKED_GLYPHS = "☑☒▣✓✔◉●⦿⊠⊗"
_UNCHECKED_GLYPHS = "☐□○◯⊙"
_CHECKBOX_GLYPHS = _CHECKED_GLYPHS + _UNCHECKED_GLYPHS
_GLYPH_RE = re.compile(f"[{_CHECKBOX_GLYPHS}]")
# Combined marker regex: either a single Unicode glyph OR an ASCII bracket
# pair ``[x]`` / ``[ ]`` (with optional ``\`` escapes around the brackets).
# Used by the per-line tokenizer so inline ASCII checkbox groups
# (``\[x] Single \[ ] Married``) are parsed the same as Unicode-glyph groups.
_MARKER_RE = re.compile(rf"[{_CHECKBOX_GLYPHS}]|\\?\[[ xX]\\?\]")


def _marker_is_checked(token: str) -> bool:
    """Decide if a checkbox marker token represents the checked state."""

    if len(token) == 1:
        return token in _CHECKED_GLYPHS
    return any(c in "xX" for c in token)


# Boolean coercion table for textual yes/no values.
_TRUTHY_TEXT = {"yes", "y", "true", "t", "1", "checked", "x", "selected", "on"}
_FALSY_TEXT = {"no", "n", "false", "f", "0", "unchecked", "unselected", "off", ""}


_PARTIAL_RATIO_THRESHOLD = 0.90
_PARTIAL_RATIO_MIN_LEN = 6
# Penalty applied to the partial-ratio score so a partial hit cannot beat
# an equally strong strict-ratio hit. Empirically 0.05 keeps partial 1.0
# above strict 0.86 (so e.g. ``API NO. (if available)`` still resolves
# against GT ``API NO.`` when the only candidate is the full noisy label)
# while preventing partial 0.95 (``County`` ⊂ ``Country``) from beating a
# strict 1.0 ``Country`` match on the *correct* row.
_PARTIAL_RATIO_PENALTY = 0.05


def _strip_label_punct(s: str) -> str:
    """Remove punctuation that varies between abbreviation styles.

    ``K.B.`` vs ``KB``, ``D.F.`` vs ``DF``, ``API NO:`` vs ``API NO``,
    ``Tel.`` vs ``Tel`` are the same label semantically. This strips
    dots, colons, and commas — separators that the parser may add or drop
    while preserving the underlying tokens. Operates after
    ``normalize_text`` so it sees a case-folded, whitespace-collapsed
    string.
    """

    s = s.replace(".", "")
    s = s.replace(":", "")
    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _label_match_score(candidate: str, label: str) -> float:
    """Score how well *candidate* matches *label* on the ``_label_matches`` axes.

    Returns ``0.0`` when the candidate fails every path (same as
    ``_label_matches`` returning False); otherwise returns a value in
    ``(0.0, 1.0]`` where higher means a better label match.

    Why scoring instead of a bool: when the document contains two visually
    similar labels (the classic ``Country`` / ``County`` collision), the
    old first-match-wins iteration would latch onto whichever fuzzy hit
    came first and return that row's value. With a scoring function, the
    caller can collect every candidate and pick the *best* match — an
    exact ``Country`` (score ``1.0``) beats a fuzzy ``County`` (score
    ``~0.92``) even when ``County`` appears earlier in the markdown.

    Scoring:

    - Strict-ratio path (``fuzz.ratio >= CELL_FUZZY_MATCH_THRESHOLD``):
      score = the ratio itself.
    - Partial-ratio fallback (``fuzz.partial_ratio >= _PARTIAL_RATIO_THRESHOLD``
      with ``shorter >= _PARTIAL_RATIO_MIN_LEN``): score = the partial
      ratio minus ``_PARTIAL_RATIO_PENALTY`` (currently 0.05). The penalty
      keeps partial hits strictly below same-strength strict hits — a
      partial 1.0 (``County`` substring inside ``County, TX``) scores
      ``0.95``, which still loses to any strict-ratio match >= 0.95 but
      wins over a strict-ratio 0.86 fuzzy match.
    - Punctuation-stripped exact path
      (``_strip_label_punct(cand) == _strip_label_punct(lbl)``): score
      ``1.0``. Exact-equality (not fuzz) keeps this path narrow — short
      labels like ``KB`` won't collide with ``KBC`` (``fuzz.ratio`` happens
      to hit exactly 0.80 between those two strings, which would leak
      through if we ran fuzz on the stripped variants) while still
      matching dotted abbreviation variants like ``K.B.`` ≡ ``KB``.
    - Best path wins when several fire.
    """

    cand = normalize_text(candidate)
    lbl = normalize_text(label)
    if not cand or not lbl:
        return 0.0

    best = 0.0
    ratio = fuzz.ratio(cand, lbl) / 100.0
    if ratio >= CELL_FUZZY_MATCH_THRESHOLD:
        best = ratio
    shorter = min(len(cand), len(lbl))
    if shorter >= _PARTIAL_RATIO_MIN_LEN:
        partial = fuzz.partial_ratio(cand, lbl) / 100.0
        if partial >= _PARTIAL_RATIO_THRESHOLD:
            penalized = max(partial - _PARTIAL_RATIO_PENALTY, 0.0)
            if penalized > best:
                best = penalized

    # Punctuation-stripped exact equality — narrowest of the three paths,
    # only fires when the strip actually collapses two different surface
    # forms onto the same string. Scored at 1.0 so legitimate abbreviation
    # variants beat a coincidental ratio-0.80 collision (the ``K.B.``/
    # ``KBC`` boundary case) when both candidates appear in the document.
    cand_stripped = _strip_label_punct(cand)
    lbl_stripped = _strip_label_punct(lbl)
    if cand_stripped and lbl_stripped and cand_stripped == lbl_stripped:
        if 1.0 > best:
            best = 1.0

    return best


def _label_matches(candidate: str, label: str) -> bool:
    """Boolean predicate over :func:`_label_match_score` for legacy callers.

    Used by callers that only need a boolean (adjacent-line fallback,
    underscore-blank label-seen detection, checkbox-state matching). The
    main text-value lookup uses :func:`_label_match_score` directly so it
    can score-and-pick-best across multiple candidate KV pairs.
    """

    return _label_match_score(candidate, label) > 0.0


def _coerce_bool(value: str | bool | list[str]) -> bool | None:
    """Coerce a value to True/False, or None if ambiguous.

    List inputs are not supported by checkbox semantics and return None;
    the caller surfaces a "must be coercible to bool" error in that case.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, list):
        return None
    text = str(value).strip().lower()
    if text in _TRUTHY_TEXT:
        return True
    if text in _FALSY_TEXT:
        return False
    return None


def _value_alternatives(value: str | bool | list[str]) -> list[str]:
    """Return the list of acceptable string values for a text-typed rule.

    Supports both single-string and list-of-strings GTs. A list lets a rule
    declare multiple acceptable readings for genuinely ambiguous fields
    (e.g. illegible handwriting). The single-string form is the default and
    keeps the GT clean for the common case.
    """

    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _multi_col_header_data_pairs(table: TableData) -> list[tuple[str, str]]:
    """For a >2-col table with header rows, yield (col_header, data_value) for
    every (column, data row) pair so a label that names a column matches the
    value in that column's data row(s)."""

    out: list[tuple[str, str]] = []
    rows, cols = table.data.shape
    if cols <= 2 or rows == 0:
        return out
    header_rows = getattr(table, "header_rows", set()) or set()
    n_header = (max(header_rows) + 1) if header_rows else 1
    for col_idx in range(cols):
        header_text = _column_header_for_index(table, col_idx)
        if not header_text:
            continue
        for row_idx in range(n_header, rows):
            cell_value = str(table.data[row_idx, col_idx]).strip()
            out.append((header_text, cell_value))
    return out


def _iter_html_cell_kv_pairs(content: str) -> list[tuple[str, str]]:
    """Yield (label, value) pairs extracted from HTML cells whose internal
    layout stacks the label above the value via ``<br/>``.

    Pattern: ``<td>Label<br/><strong>Value</strong></td>``. Common in parsers
    that try to mirror the visual two-line widget within a single cell."""

    out: list[tuple[str, str]] = []
    if "<table" not in content.lower():
        return out
    soup = BeautifulSoup(content, "lxml")
    for table in soup.find_all("table"):
        for cell in table.find_all(["td", "th"]):
            for br in cell.find_all("br"):
                br.replace_with("\n")
            cell_text = cell.get_text().strip()
            if "\n" not in cell_text:
                continue
            parts = [p.strip() for p in cell_text.split("\n", 1)]
            if len(parts) != 2:
                continue
            label_part, value_part = parts
            label_part = label_part.strip("*_ \t")
            value_part = value_part.strip("*_ \t")
            if label_part:
                out.append((label_part, value_part))
    return out


def _iter_html_table_kv_rows(content: str) -> list[tuple[str, str]]:
    """Yield (label, value) tuples from HTML tables.

    - 2-col tables: yield each row as ``(col0, col1)`` (label-then-value layout).
    - >2-col tables with header rows: yield ``(col_header, data_row_value)``
      for every column × data row, so a label naming a column matches the
      value in that column's data row.
    - Any cell that contains in-cell ``<br/>`` separators: yield
      ``(top_half, bottom_half)`` so ``<td>Label<br/><strong>Value</strong></td>``
      is captured.
    """

    out: list[tuple[str, str]] = []
    if "<table" not in content.lower():
        return out
    # Cell-internal label/value (label<br/>value inside one cell) takes
    # precedence over the row-wise 2-col interpretation; otherwise a cell
    # like ``<td>Last Name<br/>Nguyen</td>`` would be mangled into a single
    # blob ``Last Name Nguyen`` by the row-wise path before the cell-level
    # pair is ever consulted.
    out.extend(_iter_html_cell_kv_pairs(content))
    for table in parse_html_tables(content):
        rows, cols = table.data.shape
        if cols == 2:
            for row_idx in range(rows):
                label_text = str(table.data[row_idx, 0]).strip()
                value_text = str(table.data[row_idx, 1]).strip()
                out.append((label_text, value_text))
        elif cols > 2:
            # Header-then-data-row binding (one record per data row).
            # Interleaved label/value layouts inside wide HTML tables
            # (well-log report headers, rotated form pages) are handled
            # downstream by ``_iter_html_cell_neighbor_pairs`` — that
            # iterator classifies each neighbor as label-shaped vs
            # value-shaped before pairing, so the score path never sees
            # spurious ``(LABEL, OTHER_LABEL)`` candidates.
            out.extend(_multi_col_header_data_pairs(table))
    return out


# Heuristic: signals that a cell *looks like* a form-label rather than a value.
# Used as a tie-breaker by ``_iter_html_cell_neighbor_pairs`` when picking
# between a right-neighbor and a below-neighbor in wide HTML form tables —
# we only want to return value-shaped neighbors, not adjacent label cells.
#
# A cell is considered label-like when any of these holds:
#   1. trailing colon (``FILE NO:``);
#   2. short ALL-CAPS with no digits / no value-style punctuation
#      (``WELL``, ``COMPANY``, ``OTHER SERVICES``);
#   3. structurally repeats elsewhere in the same table — handled by the
#      caller, which threads the per-table text-count map in.
#
# Values like ``LEHMAN #1``, ``42-157-33282``, ``KEBO OIL & GAS, INC.``,
# ``15-MAY-2023`` keep digits / hashes / commas / parens so they fail
# heuristic (2) and are correctly classified as value-shaped.
#
# Note: ``&`` is intentionally absent from the disqualifier so common
# value strings like ``"KEBO OIL & GAS, INC."`` (rescued by the comma)
# stay value-shaped without forcing every label with ``&`` (e.g. an
# ``"OIL & GAS"`` column header) to be misread as a value. A naked
# ``"X & Y"`` value with no other punctuation would be misclassified as a
# label, but that pattern hasn't surfaced in real benchmark data.
_LABEL_LIKE_DISQUALIFIER_RE = re.compile(r"[\d#@/_,()$%]")


def _cell_text_is_label_like(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if s.endswith(":"):
        return True
    # Length cap: typical form labels are short (1-3 words). Long ALL-CAPS
    # strings like ``"PERMITTED FOR RECOMPLETION TO PRODUCE FROM"`` skip
    # heuristic (2) and stay value-shaped, which is the safer default — the
    # cost of mis-flagging a long label is a missed neighbor, but the cost
    # of flagging a long value is returning the wrong neighbor.
    if len(s) > 30:
        return False
    if _LABEL_LIKE_DISQUALIFIER_RE.search(s):
        return False
    if s != s.upper():
        return False
    if not re.search(r"[A-Z]", s):
        return False
    return True


def _iter_md_table_kv_rows(content: str) -> list[tuple[str, str]]:
    """Yield (label, value) tuples from markdown tables.

    - 2-col tables: yield each row as ``(col0, col1)`` (existing behavior).
    - >2-col tables: yield ``(col_header, data_row_value)`` for every
      column × data row.
    """

    out: list[tuple[str, str]] = []
    for table in parse_markdown_tables(content):
        rows, cols = table.data.shape
        if cols == 2:
            for row_idx in range(rows):
                label_text = str(table.data[row_idx, 0]).strip()
                value_text = str(table.data[row_idx, 1]).strip()
                out.append((label_text, value_text))
        elif cols > 2:
            out.extend(_multi_col_header_data_pairs(table))
    return out


# Generic HTML tag stripper for label/value normalization. Form-field values
# never legitimately contain ``<tag>...</tag>`` markup — names, addresses, IDs,
# and currency don't — but parsers leak HTML wrappers into extracted spans
# (haiku preserves ``<strong>``/``<td>``, gemini emits ``<u>`` underline-fill,
# OpenAI sometimes leaves ``</p>``). The pattern is restricted to well-formed
# HTML element opens/closes: a tag name must start with an ASCII letter and
# contain only alphanumerics afterwards, optionally followed by a
# whitespace-introduced attribute run. This deliberately excludes markdown
# email/URL autolinks like ``<wei.lin@host.com>`` and ``<https://...>``,
# whose first character after ``<`` is a letter but whose body contains
# ``.``/``@``/``:`` that disqualify them from the tag-name shape.
_HTML_TAG_RE = re.compile(r"<\s*/?\s*[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?\s*/?\s*>")


def _strip_html_tags(s: str) -> str:
    return _HTML_TAG_RE.sub("", s)


# Tagged-line prefix used by some parsers to mark a field-extraction event,
# e.g. ``[FORM FIELD] Label: value``. The bracketed prefix is parser noise,
# not part of the label. We only strip it from the start of a candidate
# label, never mid-string, so legitimate labels containing brackets like
# ``[Effective Date]`` (uncommon but possible) are preserved unless the
# bracket is the leading token.
_LABEL_TAG_PREFIX_RE = re.compile(r"^\s*\[[^\]\n]+\]\s+")


def _trim_value_at_next_field(value: str) -> str:
    """Trim a captured value at a ``| Next Label: ...`` boundary.

    Some parsers concatenate multiple labelled fields onto one line with
    ``|`` separators (e.g. ``Date: 2026-04-27 | Borrower's Name: Maya | ...``).
    Without this trim, the plain-colon regex captures the entire tail as the
    value of the first field. We only split when the part after the ``|``
    looks like another labelled field (contains ``:``), so legitimate values
    with embedded ``|`` (rare in form data) are preserved.
    """

    parts = re.split(r"\s+\|\s+", value, maxsplit=1)
    if len(parts) == 2 and ":" in parts[1]:
        return parts[0].strip()
    return value


def _split_pipe_concatenated_pairs(value: str) -> list[tuple[str, str]]:
    """Split a run-on ``Label1: v1 | Label2: v2 | ...`` value tail into pairs.

    Companion to :func:`_trim_value_at_next_field`. The first call trims the
    value of the *initial* labelled field; this function recovers any
    *subsequent* ``Label: value`` pairs that were riding along on the same
    line so a single-line run-on yields one pair per labelled field.
    """

    out: list[tuple[str, str]] = []
    if " | " not in value:
        return out
    for segment in re.split(r"\s+\|\s+", value):
        if ":" not in segment:
            continue
        # Same horizontal-only colon split as _PLAIN_COLON_RE so we don't
        # accidentally bleed time-of-day strings ("11:30 AM") into pairs.
        m = re.match(r"^[ \t]*([^:\n*][^:\n]{0,200}?)[ \t]*:[ \t]*(.+?)[ \t]*$", segment)
        if not m:
            continue
        seg_label = _strip_html_tags(m.group(1).strip()).strip()
        seg_label = _LABEL_TAG_PREFIX_RE.sub("", seg_label).strip()
        seg_value = _strip_html_tags(m.group(2).strip()).strip()
        if seg_label:
            out.append((seg_label, seg_value))
    return out


# Bullet-line shape for safe aggregation: ``- item`` / ``* item`` / ``+ item``
# (with optional leading ``\`` escape some renderers emit). The negative
# lookahead rejects checkbox-bearing bullets (``- [x] ...``) — those rows
# describe their own state, not a continuation of the preceding label.
_AGGREGATE_BULLET_RE = re.compile(r"^\\?[-*+]\s+(?!\\?\[)")


def _aggregate_following_lines(content: str, after_offset: int, max_lines: int = 8) -> str:
    """Collect bullet-list lines after *after_offset* into a single value
    string, joined with ``, ``.

    This is a narrow fallback for the audit-A3 pattern: a bold-colon header
    with an empty inline value followed by a multi-line address laid out as
    bullets (HUD voucher ``Mail Payments To`` blocks, etc.). Strict gating
    keeps it from pulling unrelated form structure into the value:

    1. Every line must be a clean bullet (``-``/``*``/``+`` with no
       ``[x]``/``[ ]`` checkbox marker — those rows belong to a different
       field).
    2. No line may carry any checkbox glyph or ASCII bracket marker.
    3. At least 2 collected bullets are required. A single bullet is too
       ambiguous to attribute as the value — leaving the value empty is
       safer than risking a wrong attribution.
    4. Stops at blank line, ATX heading, HTML boundary, or another bold-
       colon header. Returns ``""`` if any constraint fails so the caller
       falls back to the normal empty-value path.
    """

    tail = content[after_offset:]
    lines = tail.splitlines()
    # Skip the line containing the header itself (we matched into it).
    start_idx = 1 if lines else 0
    collected: list[str] = []
    for raw in lines[start_idx : start_idx + max_lines]:
        stripped = raw.strip()
        if not stripped:
            break
        if stripped.startswith(("#", ">", "|", "<")):
            break
        # Stop at the start of a new bold-colon header.
        if "**" in stripped and ":" in stripped:
            break
        if not _AGGREGATE_BULLET_RE.match(stripped):
            return ""
        if _MARKER_RE.search(stripped):
            return ""
        cleaned = re.sub(r"^\\?[-*+]\s+", "", stripped).strip()
        cleaned = _strip_html_tags(cleaned).strip()
        if cleaned:
            collected.append(cleaned)
    if len(collected) < 2:
        return ""
    return ", ".join(collected)


# Bold-colon pattern. Matches **Label:** value and **Label**: value, allowing
# multiple pairs on a single line. All inter-token whitespace is restricted
# to horizontal whitespace ([ \t]) so a match cannot span blank lines or
# headings — without this, an empty "**Label**:\n\n# Heading\n\n**Other**:"
# would attribute the heading text to Label as the value.
_BOLD_COLON_RE = re.compile(
    r"\*\*[ \t]*([^*\n]+?)[ \t]*\*\*[ \t]*:?[ \t]*([^\n*]*?)(?=[ \t]*\*\*|$)",
    re.MULTILINE,
)

# Connector words that the parser sometimes wraps in bold inside a numeric
# range, e.g. ``**Depth Drilled**: 105 **to**: 15437`` or
# ``Temperature: 32 **to** 100 F``. Without special-casing, the bold-colon
# value regex stops at the connector's leading ``**`` and only captures the
# left half. We re-join the trailing value when the bold span between two
# value chunks is one of these connectors. The connectors are matched whole-
# word, case-insensitively. Allows leading horizontal whitespace so the
# splice cursor doesn't have to land exactly on the ``**``.
_BOLD_CONNECTOR_RE = re.compile(
    r"[ \t]*\*\*[ \t]*(to|and|or|&|thru|through|until)[ \t]*\*\*[ \t]*:?[ \t]*([^\n*]*?)"
    r"(?=[ \t]*\*\*|$)",
    re.IGNORECASE | re.MULTILINE,
)


def _extend_value_across_bold_connectors(content: str, value_end_offset: int, base_value: str) -> str:
    """Re-join a bold-colon value that was clipped at a bold connector token.

    The bold-colon regex terminates the value at the next ``**``. When the
    next bold span is a connector word (``to``, ``and``, ...), the value
    actually continues across it. This helper looks at the content
    immediately following the captured value and, while it sees a bold
    connector followed by more inline content, splices everything into a
    single value string.

    Stops as soon as the next bold span is anything other than a recognized
    connector — that's a real label boundary, not a continuation.
    """

    if not base_value:
        return base_value
    cursor = value_end_offset
    joined = base_value
    while True:
        match = _BOLD_CONNECTOR_RE.match(content, cursor)
        if not match:
            break
        connector = match.group(1)
        extra = match.group(2).strip()
        joined = f"{joined} {connector} {extra}".strip()
        cursor = match.end()
    return joined


def _iter_bold_colon_pairs(content: str) -> list[tuple[str, str]]:
    """Yield every (label, value) pair surfaced via bold-colon syntax.

    Generic post-processing applied to every yielded pair: HTML tags
    stripped from both label and value, leading ``[tag]`` prefix removed
    from the label, ``| Next Label:`` boundary trimmed from the value, and
    when the inline value is empty, the next few non-blank list/text lines
    are aggregated into the value (multi-line address pattern).
    """

    out: list[tuple[str, str]] = []
    for match in _BOLD_COLON_RE.finditer(content):
        cand_label = match.group(1).strip(": ").strip()
        # Strip trailing markdown line-continuation backslash before whitespace.
        # Some parsers emit ``**Label**: \`` for empty fields; without this
        # strip the value would be ``"\\"``, never matching empty expected.
        raw_value = match.group(2).strip().rstrip("\\").strip()
        # Splice bold connectors (``**to**``, ``**and**``) back into the value
        # so numeric ranges like ``**Depth Drilled**: 105 **to** 15437`` aren't
        # truncated at the connector.
        raw_value = _extend_value_across_bold_connectors(content, match.end(), raw_value)
        cand_label = _strip_html_tags(cand_label).strip()
        cand_label = _LABEL_TAG_PREFIX_RE.sub("", cand_label).strip()
        cand_value = _strip_html_tags(raw_value).strip()
        cand_value = _trim_value_at_next_field(cand_value)
        if not cand_value:
            cand_value = _aggregate_following_lines(content, match.end())
        if cand_label:
            out.append((cand_label, cand_value))
        # Recover any sibling pipe-concatenated pairs riding the same line.
        out.extend(_split_pipe_concatenated_pairs(raw_value))
    return out


# Plain-colon pattern. Inter-token whitespace is restricted to horizontal
# whitespace ([ \t]) so a colon at end-of-line cannot consume the next line as
# the value (parallel to the bold-colon regex; same blank-line crossing bug).
_PLAIN_COLON_RE = re.compile(r"^[ \t]*([^:\n*][^:\n]{0,200}?)[ \t]*:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^\\?[-*+]\s+")

# Underscore blank field: ``Processor's Name _________________``. The label
# sits before a run of three or more underscores acting as a fill-in line for
# an empty field. No colon, no bold, just a label-then-underscore-blank.
_UNDERSCORE_BLANK_RE = re.compile(r"^\s*([^_\n]+?)\s+_{3,}\s*$", re.MULTILINE)


def _iter_plain_colon_pairs(content: str) -> list[tuple[str, str]]:
    """Yield (label, value) pairs from `Label: value` lines (plain text).

    Plain bullet items with the ``Label: value`` shape (``- Defendant: Devon``)
    are stripped of their leading marker and yielded — markdown task lists
    (``- [x] Foo``) are still skipped because they are handled by the checkbox
    scanners. Headings, fenced code, blockquotes, and bold-formatted lines
    are skipped here too.

    Generic post-processing on every yielded pair: HTML tags stripped from
    both label and value, leading ``[tag]`` prefix removed from the label,
    and the value trimmed at any ``| Next Label:`` boundary so a single line
    like ``A: x | B: y`` yields two pairs instead of one with a run-on
    value.
    """

    out: list[tuple[str, str]] = []
    for match in _PLAIN_COLON_RE.finditer(content):
        cand_label = match.group(1).strip()
        raw_value = match.group(2).strip()
        # Skip multi-cell HTML table rows: a single line that opens more than
        # one ``<th>`` / ``<td>`` is a wide table row, not a single
        # ``label: value`` line. Without this guard
        # ``<tr><th>API NO:</th><th>WELL</th><th>LEHMAN #1</th></tr>`` matches
        # the plain-colon regex and yields ``("API NO", "WELLLEHMAN #1")``
        # because HTML-tag stripping collapses adjacent cells into a single
        # value run. Single-cell rows (``<th>Company: CIMARRON ...</th>``)
        # carry exactly one inline KV pair and stay on this path — wide HTML
        # form tables are handled by ``_iter_html_table_kv_rows`` and
        # ``_iter_html_cell_neighbor_pairs``.
        raw_line = match.group(0)
        if len(re.findall(r"<t[hd]\b", raw_line)) > 1:
            continue
        if cand_label.startswith(("#", "`", ">")):
            continue
        if "**" in cand_label:
            continue
        if cand_label.startswith(("\\-", "-", "*", "+")):
            stripped = _LIST_MARKER_RE.sub("", cand_label).strip()
            # Tasklist-shaped bullets (``[x] ...``) belong to the checkbox path.
            if stripped.startswith(("\\[", "[")):
                continue
            if not stripped:
                continue
            cand_label = stripped
        cand_label = _strip_html_tags(cand_label).strip()
        cand_label = _LABEL_TAG_PREFIX_RE.sub("", cand_label).strip()
        cand_value = _strip_html_tags(raw_value).strip()
        cand_value = _trim_value_at_next_field(cand_value)
        if cand_label:
            out.append((cand_label, cand_value))
        # Recover any sibling pipe-concatenated pairs riding the same line.
        out.extend(_split_pipe_concatenated_pairs(raw_value))
    return out


# Underline fill-in pattern: parsers that preserve the form's "fill in the
# blank" layout emit the filled value wrapped in ``<u>...</u>`` tags inline
# in the surrounding prose, e.g.
#
#   **2. PROPERTY:** Lot <u>12</u>, Block <u>C</u>, City of <u>Austin</u>...
#
# The label sits immediately before the underline span, terminated by a
# punctuation/whitespace boundary on its left side. We yield (label, value)
# for each such span so the standard ``_label_matches`` fuzzy-matcher can
# bridge GT labels like "Block" or "City of (Street Address and City)".
_UNDERLINE_FILL_RE = re.compile(r"<u>([^<\n]+)</u>")
_LABEL_LEFT_TERMINATORS = ".,;:()\n>"


def _iter_underline_fill_pairs(content: str) -> list[tuple[str, str]]:
    """Yield (preceding_label, underlined_value) pairs from ``<u>...</u>`` runs."""

    out: list[tuple[str, str]] = []
    for match in _UNDERLINE_FILL_RE.finditer(content):
        value = match.group(1).strip()
        if not value:
            continue
        before = content[max(0, match.start() - 100) : match.start()]
        # Walk backward to the nearest sentence/clause terminator. Anything
        # left of that terminator belongs to a different label (or to a
        # heading/inline header), so we stop there.
        cut = -1
        for ch in _LABEL_LEFT_TERMINATORS:
            cut = max(cut, before.rfind(ch))
        label_chunk = before[cut + 1 :]
        # Strip markdown noise: leading bullet, bold/italic markers, stray
        # backslashes, and trailing whitespace. The bracketed-tag prefix
        # (``[FORM FIELD] ``) is dropped here too so it never bleeds into
        # candidate labels.
        label_chunk = re.sub(r"^[\s\\*_#>\-]+", "", label_chunk)
        label_chunk = _LABEL_TAG_PREFIX_RE.sub("", label_chunk)
        label_chunk = _strip_html_tags(label_chunk).strip()
        label_chunk = label_chunk.strip("*_ \t").strip()
        if not label_chunk:
            continue
        # Only the trailing 1-6 words can plausibly be the label — the rest
        # is sentence context.
        words = label_chunk.split()
        if not words:
            continue
        label = " ".join(words[-6:])
        if label:
            out.append((label, value))
    return out


def _iter_underscore_blank_pairs(content: str) -> list[tuple[str, str]]:
    """Yield (label, "") pairs for ``Label ____`` underscore-blank fields."""

    out: list[tuple[str, str]] = []
    for match in _UNDERSCORE_BLANK_RE.finditer(content):
        cand_label = match.group(1).strip()
        if not cand_label:
            continue
        if cand_label.startswith(("#", "-", "*", "`", ">", "|")):
            continue
        if "**" in cand_label or ":" in cand_label:
            continue
        out.append((cand_label, ""))
    return out


# Italic line shape: ``*Plaintiff*`` or ``_Address_`` (optionally with a
# trailing space + ``)`` from court-form layouts like ``*Plaintiff* )``).
_ITALIC_LABEL_LINE_RE = re.compile(r"^\s*([*_])\s*(\S.*?\S)\s*\1[\s)\\]*$")


def _find_text_value_adjacent_line(content: str, label: str) -> tuple[bool, str | None]:
    """Fallback for label-on-its-own-line layouts adjacent to an unlabelled value.

    Two layouts share this scanner:

    - **Italic caption below value** (federal court forms — AO398):
      ``Anthony Cole Jackson )\\n*Plaintiff* )``. The label sits italicized on
      the line below the value.
    - **Numbered/heading-style label above value** (UCC5, gemini sub-sections):
      ``1a. INITIAL FINANCING STATEMENT FILE NUMBER\\nOR-UCC-2025-00532600``.
      The label is its own line above the value.

    Conservative heuristic: only fires for short label-shaped lines (≤ 80
    chars after stripping markers, no ``:`` and no ``**``) and only on a
    *strict* ratio match (≥ ``CELL_FUZZY_MATCH_THRESHOLD``). This means a
    long paragraph that *contains* the label as a substring is **not**
    treated as the label line — partial-ratio matching is reserved for the
    other (label-then-value) scanners.

    Direction: italic line → look ABOVE first (caption convention); plain
    line → look BELOW first (label-then-value convention). Whichever
    direction lands a non-blank line wins.
    """

    lines = content.splitlines()
    lbl_norm = normalize_text(label)
    if not lbl_norm:
        return False, None
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or len(stripped) > 100:
            continue
        if stripped.startswith(("#", ">", "|", "<", "`")):
            continue
        if "**" in stripped or ":" in stripped:
            continue
        italic_match = _ITALIC_LABEL_LINE_RE.match(raw)
        if italic_match:
            cleaned = italic_match.group(2).strip()
        else:
            cleaned = re.sub(r"\s*[)\\]+\s*$", "", stripped)
            cleaned = re.sub(r"^\\?[-*+]\s+", "", cleaned).strip()
            cleaned = cleaned.strip("*_ \t").strip()
        if not cleaned or len(cleaned) > 80:
            continue
        cand_norm = normalize_text(cleaned)
        if not cand_norm:
            continue
        if fuzz.ratio(cand_norm, lbl_norm) / 100.0 < CELL_FUZZY_MATCH_THRESHOLD:
            continue
        if italic_match:
            search_orders = [
                range(i - 1, max(i - 4, -1), -1),
                range(i + 1, min(i + 4, len(lines))),
            ]
        else:
            search_orders = [
                range(i + 1, min(i + 4, len(lines))),
                range(i - 1, max(i - 4, -1), -1),
            ]
        for order in search_orders:
            for j in order:
                cand_line = lines[j].strip()
                if not cand_line:
                    continue
                if cand_line.startswith(("#", "|", ">")):
                    break
                if "**" in cand_line and ":" in cand_line:
                    break
                value = re.sub(r"^\\?[-*+]\s+", "", cand_line).strip()
                value = re.sub(r"\s*[)\\]+\s*$", "", value).strip()
                value = value.strip("*_ \t").strip()
                if value:
                    return True, value
        return True, ""
    return False, None


def _build_cell_text_counts(data, rows: int, cols: int) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Per-table map of text → number of distinct *origin* cells.

    ``parse_html_tables`` expands ``colspan``/``rowspan`` by duplicating cell
    text across every covered grid position, so a single ``<th
    colspan="4">KEBO</th>`` looks like four ``"KEBO"`` cells in the expanded
    grid. Counting raw grid cells would mis-classify any spanned value as a
    repeated label. Dedupe by skipping cells whose text equals the left or
    above neighbor — those are colspan / rowspan runs of the same origin.
    """

    counts: dict[str, int] = {}
    for r in range(rows):
        for c in range(cols):
            t = str(data[r, c]).strip()
            if not t:
                continue
            if c > 0 and str(data[r, c - 1]).strip() == t:
                continue
            if r > 0 and str(data[r - 1, c]).strip() == t:
                continue
            counts[t] = counts.get(t, 0) + 1
    return counts


def _neighbor_is_label_like(neighbor: str, text_counts: dict[str, int]) -> bool:
    if _cell_text_is_label_like(neighbor):
        return True
    # Short text that repeats elsewhere in the same table → structural label.
    if len(neighbor) <= 30 and text_counts.get(neighbor, 0) >= 2:
        return True
    return False


def _is_value_shaped_cell(neighbor: str | None) -> bool:
    if not neighbor:
        return False
    s = neighbor.strip()
    if len(s) < 2:
        return False
    # Lone checkbox glyphs aren't useful values for text rules.
    if _GLYPH_RE.search(s) and len(s) <= 2:
        return False
    return True


def _iter_html_cell_neighbor_pairs(content: str) -> list[tuple[str, str]]:
    """Yield ``(cell_text, neighbor_value)`` pairs for wide (>2 col) HTML
    tables, intended as a low-priority fallback source for
    ``_find_text_value_for_label``.

    Targets form-style layouts where labels and values are spatially
    interleaved inside a single wide ``<table>`` rather than separated into
    a clean header row + data rows, e.g. well-log report headers::

        <tr><th colspan="2">FILE NO:</th>
            <th colspan="2">COMPANY</th>
            <th colspan="4">KEBO OIL &amp; GAS, INC.</th></tr>
        <tr><th colspan="2">API NO:</th>
            <th colspan="2">WELL</th>
            <th colspan="4">LEHMAN #1</th></tr>
        <tr><th colspan="2">42-157-33282</th>
            <th colspan="2">FIELD</th>
            <th colspan="4">NEEDVILLE</th></tr>

    For each non-empty cell in the expanded grid the iterator looks at two
    candidate neighbors:

      * the first non-empty cell to the right in the same row, skipping
        colspan duplicates (cell text equal to the cell itself);
      * the first non-empty cell below in the same column, similarly skipping
        rowspan duplicates.

    The chosen neighbor is the first one that is *value-shaped* (length ≥ 2,
    not a lone checkbox glyph) and *not label-shaped* per
    ``_cell_text_is_label_like`` or structural repetition in the same table.
    Right is preferred over below (matches left-to-right reading).

    Cells with no value-shaped neighbor still emit ``(cell_text, "")`` so the
    caller's ``_collect`` records ``label_seen=True`` for empty-expected
    rules — same contract as the other pair sources.

    The caller scores ``cell_text`` against the rule label via
    ``_label_match_score`` and picks the best candidate. We don't filter by
    label here so the caller can resolve adjacent-label collisions (the
    same way #978 made other sources do).
    """

    if "<table" not in content.lower():
        return []

    out: list[tuple[str, str]] = []

    for table in parse_html_tables(content):
        rows, cols = table.data.shape
        if cols <= 2 or rows == 0:
            continue

        # Per-table text-count map — a short text that exactly repeats in
        # ≥2 *distinct origin* cells (after collapsing colspan/rowspan runs
        # via ``_build_cell_text_counts``) is structurally likely to be a
        # column label / section header (e.g. ``KB`` / ``DF`` / ``GL`` rows
        # in well-log elevation blocks). Used as a tie-breaker for which
        # neighbor cell is value-shaped.
        text_counts = _build_cell_text_counts(table.data, rows, cols)

        for r in range(rows):
            for c in range(cols):
                cell = str(table.data[r, c]).strip()
                if not cell:
                    continue

                # Right scan: first non-empty cell to the right that is not
                # a colspan duplicate (text != label cell text).
                right_val: str | None = None
                for cc in range(c + 1, cols):
                    nxt = str(table.data[r, cc]).strip()
                    if nxt and nxt != cell:
                        right_val = nxt
                        break

                # Below scan: first non-empty cell directly below that is
                # not a rowspan duplicate.
                below_val: str | None = None
                for rr in range(r + 1, rows):
                    nxt = str(table.data[rr, c]).strip()
                    if nxt and nxt != cell:
                        below_val = nxt
                        break

                # Score each candidate. Want a value-shaped neighbor that
                # does *not* itself look label-like. Right is preferred over
                # below when both qualify (matches left-to-right reading).
                right_ok = _is_value_shaped_cell(right_val) and not _neighbor_is_label_like(
                    right_val or "", text_counts
                )
                below_ok = _is_value_shaped_cell(below_val) and not _neighbor_is_label_like(
                    below_val or "", text_counts
                )

                if right_ok:
                    out.append((cell, right_val or ""))
                elif below_ok:
                    out.append((cell, below_val or ""))
                else:
                    # No value-shaped neighbor at this position. Still emit
                    # an empty-value pair so a matching label sets the
                    # caller's ``label_seen`` flag (mirrors the other pair
                    # iterators that surface ``""`` for label-only hits).
                    out.append((cell, ""))

    return out


def _find_text_value_for_label(
    content: str,
    label: str,
    expected_values: list[str] | None = None,
) -> tuple[bool, str | None]:
    """Look up the value for *label*. Returns (label_found, value).

    The boolean tracks whether the label was located **at all** — useful for
    distinguishing "label missing from content" from "label present but value
    blank" (signature evaluation depends on this distinction). When the label
    is found only with empty values, returns ``(True, "")`` so callers can
    decide what to do (text rules with empty expected values pass; signature
    rules treat it as unsigned).

    Matching strategy is **best-score across all sources**: every candidate
    KV pair from every source iterator is scored against the target label
    via :func:`_label_match_score`, and the highest-scoring non-empty value
    wins. Tie-breaks fall back to source priority (bold-colon > plain-colon
    > md-table > html-table > underline-fill) and then document order. This
    eliminates the classic ``Country`` / ``County`` adjacent-label
    collision: an exact ``Country`` hit (score 1.0) always wins over a
    fuzzy ``County`` hit (score ~0.86–0.95) no matter which comes first.

    Multi-occurrence disambiguation via ``expected_values``
    -------------------------------------------------------
    A label text can legitimately appear multiple times at the **same**
    best score: ``KB`` / ``DF`` / ``GL`` are exact-match labels in well-
    log elevation blocks while also appearing as values of
    ``LOG MEASURED FROM`` / ``DRILL. MEAS. FROM`` (where the cell-
    neighbor matcher surfaces them with score 1.0). Without
    disambiguation, source-priority + doc-order tie-breaks would lock
    onto an arbitrary occurrence — which one happens to come first
    has no relation to which page occurrence the GT refers to.

    When ``expected_values`` is supplied, the matcher applies the rule's
    expected value(s) as an oracle **among candidates at the top score
    level only**. That is: it picks the highest score; collects every
    candidate at that score; and returns the first one whose value
    matches any expected via :func:`_values_match_text`. If no
    top-score candidate matches, the legacy source-priority / doc-order
    tie-break fires — same as without ``expected_values``.

    The "top score only" gate is what keeps the GT oracle from leaking
    across adjacent labels: ``Country`` (score 1.0) vs ``County``
    (partial 0.95) live at *different* score levels, so even if a
    ``County`` row's value coincidentally equals the GT's expected
    ``Country`` value, ``County`` is not eligible. Only when two
    candidates are equally good *label matches* does the value oracle
    intervene.
    """

    label_seen = False
    # (negated_score, negated_priority, doc_order, value, source_name)
    # — we'll sort ascending so the *best* candidate (highest score, then
    # highest priority, then earliest doc order) sits at the top.
    candidates: list[tuple[float, int, int, str, str]] = []

    def _collect(
        pairs: list[tuple[str, str]],
        priority: int,
        source_name: str,
    ) -> None:
        nonlocal label_seen
        for idx, (cand_label, cand_value) in enumerate(pairs):
            score = _label_match_score(cand_label, label)
            if score <= 0.0:
                continue
            label_seen = True
            if cand_value:
                candidates.append((-score, -priority, idx, cand_value, source_name))

    # Higher priority numbers = more confident sources. The ordering matches
    # the original first-match-wins precedence so tie-breaks preserve legacy
    # behavior on documents where multiple sources produce equally strong
    # label matches.
    _collect(_iter_bold_colon_pairs(content), priority=4, source_name="bold_colon")
    _collect(_iter_plain_colon_pairs(content), priority=3, source_name="plain_colon")
    _collect(_iter_md_table_kv_rows(content), priority=2, source_name="md_table")
    _collect(_iter_html_table_kv_rows(content), priority=1, source_name="html_table")

    # Underscore blank fields (``Label ____``) — label seen, value empty.
    # The yielded value is always ""; we just record label presence so an
    # empty-expected text rule can pass via the ``label_seen`` short-circuit
    # below.
    for cand_label, _ in _iter_underscore_blank_pairs(content):
        if _label_matches(cand_label, label):
            label_seen = True

    # Last-resort sources — only fire when no higher-confidence source
    # surfaced a non-empty value, so they never overwrite a strong-source
    # extraction. Both are gated on ``not candidates`` and added with
    # priorities below the strong sources; if both fire and both produce
    # candidates, ``priority`` breaks the tie in favor of underline_fill.
    if not candidates:
        # Underline fill-in (``Label <u>value</u>`` inline in prose).
        if "<u>" in content:
            _collect(
                _iter_underline_fill_pairs(content),
                priority=0,
                source_name="underline_fill",
            )
        # Wide-form HTML table cell-neighbor fallback. Targets layouts
        # where labels and values are spatially interleaved inside a single
        # wide ``<table>`` (well-log report headers, rotated form pages),
        # which neither the 2-col nor the multi-col header×data pair
        # iterator covers. Priority -1 keeps it strictly below
        # underline_fill on tie-breaks.
        _collect(
            _iter_html_cell_neighbor_pairs(content),
            priority=-1,
            source_name="html_cell_neighbor",
        )

    if candidates:
        candidates.sort()
        # ``candidates`` is sorted ascending by (-score, -priority, doc_order,
        # ...), so the head is the best (label-match-score, source-priority,
        # doc-order) tuple. We use the rule's expected value as a tie-breaker
        # **only among candidates at the head score**, which keeps the GT
        # oracle from leaking across adjacent labels (Country score 1.0 vs
        # County score ~0.95 live at different levels, so County is never
        # eligible when Country is present).
        best_score_key = candidates[0][0]
        if expected_values:
            for neg_score, _prio, _doc, value, _src in candidates:
                if neg_score != best_score_key:
                    break
                for exp in expected_values:
                    if _values_match_text(value, exp):
                        return True, value
        return True, candidates[0][3]

    # Adjacent-line fallback (italic caption below value, or numbered label
    # above value). Only fires when no other matcher located the label.
    if not label_seen:
        adj_seen, adj_value = _find_text_value_adjacent_line(content, label)
        if adj_seen:
            return True, adj_value

    if label_seen:
        return True, ""
    return False, None


def _tokenize_checkbox_line(line: str) -> list[tuple[str, bool]]:
    """Pair every checkbox marker on *line* with its associated label.

    Markers may be Unicode glyphs (``☐``/``☑``/``◉``/``○``/...) OR ASCII
    bracket pairs (``[x]``, ``\\[x\\]``, ``[ ]``). Handles both orderings:

      - marker-first: ``☐ Single ☑ Married`` or ``\\[x] A \\[ ] B`` — each
        label sits between a marker and the next marker (or end of line).
      - label-first: ``Single ☐  Married ☑`` or ``Checking \\[x] Savings \\[ ]``
        — each label sits between the previous marker (or start) and the
        next marker.

    Direction is decided by what comes before the first marker: if the line
    starts with the marker (after optional whitespace), use marker-first;
    otherwise use label-first. This covers the inline mid-line bracket
    pattern ``**Inaccuracy in financing statement** \\[ ]`` since the
    closing bracket is treated as a marker and the bold-label segment to
    its left becomes the label.
    """

    marker_matches = list(_MARKER_RE.finditer(line))
    if not marker_matches:
        return []

    text_before_first = line[: marker_matches[0].start()].strip()
    pairs: list[tuple[str, bool]] = []

    if not text_before_first:
        # marker-first: label runs from marker end to next marker start (or EOL).
        for i, m in enumerate(marker_matches):
            label_start = m.end()
            label_end = marker_matches[i + 1].start() if i + 1 < len(marker_matches) else len(line)
            label_text = line[label_start:label_end].strip()
            if label_text:
                pairs.append((label_text, _marker_is_checked(m.group())))
    else:
        # label-first: label runs from previous marker end (or 0) to current marker.
        prev_end = 0
        for m in marker_matches:
            label_text = line[prev_end : m.start()].strip()
            prev_end = m.end()
            if label_text:
                pairs.append((label_text, _marker_is_checked(m.group())))
    return pairs


# Markdown task-list. Allows optional ``\`` escapes around the list marker
# AND the brackets — some parsers emit ``\[x\]`` (or even ``\- \[x\]`` for a
# nested escaped bullet) so the markdown source survives literal-character
# rendering. The marker accepts ``-``/``*``/``+`` and numbered-list ``\d+.`` —
# USCIS citizenship attestations are rendered as ``1. [x] A citizen ...``.
_LIST_MARKER_RE_INLINE = r"(?:[-*+]|\d+\.)"
_MD_TASKLIST_RE = re.compile(
    rf"^\s*\\?{_LIST_MARKER_RE_INLINE}\s*\\?\[([ xX])\\?\]\s*(.+?)\s*$",
    re.MULTILINE,
)

# Label-first bullet checkbox: ``* Checking \[x]`` or ``\- Savings [ ]``. The
# label sits between the list marker and the bracket. Common in forms where
# the parser surfaces the option label as the bullet text and the state as a
# trailing widget marker. Both the bullet marker and the brackets may be
# preceded by a literal backslash escape.
_MD_BULLET_LABEL_FIRST_RE = re.compile(
    rf"^\s*\\?{_LIST_MARKER_RE_INLINE}\s+(.+?)\s+\\?\[([ xX])\\?\]\s*$",
    re.MULTILINE,
)

# Bullet-less task-list: a line that starts with ``\[x]`` / ``[ ]`` directly
# with no leading bullet marker. ours_cost_effective and gemini render IRS
# W-9 / USCIS / UCC5 checkboxes this way (``\[x] Individual/sole proprietor``
# on its own line). The label group disallows ``[`` so a line with multiple
# inline bracket markers (``\[ ] A \[x] B``) does NOT match here — those go
# through the per-line tokenizer below where each bracket is paired with its
# own label.
_MD_BARE_TASKLIST_RE = re.compile(
    r"^\s*\\?\[([ xX])\\?\]\s*([^\[\n]+?)\s*$",
    re.MULTILINE,
)


def _find_checkbox_state_for_label(content: str, label: str) -> bool | None:
    """Return True/False if *label* has a checkbox-style state nearby, else None.

    Like :func:`_find_text_value_for_label`, this collects every candidate
    ``(label, state)`` across every checkbox source, scores the label, and
    returns the state attached to the highest-scoring candidate. Avoids
    adjacent-label collisions where two visually similar labels share a
    line and the wrong one gets picked just because it came first.
    """

    # (negated_score, negated_priority, doc_order, state)
    candidates: list[tuple[float, int, int, bool]] = []

    def _try_add(cand_label: str, state: bool, priority: int, idx: int) -> None:
        score = _label_match_score(cand_label, label)
        if score > 0.0:
            candidates.append((-score, -priority, idx, state))

    # Markdown task-list: - [x] Label   or   - [ ] Label   or   1. [x] Label
    for idx, match in enumerate(_MD_TASKLIST_RE.finditer(content)):
        state_char, cand_label = match.group(1), match.group(2).strip()
        _try_add(cand_label, state_char.strip().lower() == "x", priority=4, idx=idx)

    # Label-first bullet: - Label [x]   or   * Label \[x]
    for idx, match in enumerate(_MD_BULLET_LABEL_FIRST_RE.finditer(content)):
        cand_label, state_char = match.group(1).strip(), match.group(2)
        _try_add(cand_label, state_char.strip().lower() == "x", priority=3, idx=idx)

    # Bullet-less task-list: \[x] Label  (no leading -/*/+/digit.)
    for idx, match in enumerate(_MD_BARE_TASKLIST_RE.finditer(content)):
        state_char, cand_label = match.group(1), match.group(2).strip()
        _try_add(cand_label, state_char.strip().lower() == "x", priority=2, idx=idx)

    # Per-line marker tokenization (handles inline groups in either direction
    # and mid-line ASCII bracket markers after a bold label).
    inline_idx = 0
    for line in content.splitlines():
        if not _MARKER_RE.search(line):
            continue
        for cand_label, state in _tokenize_checkbox_line(line):
            _try_add(cand_label, state, priority=1, idx=inline_idx)
            inline_idx += 1

    if candidates:
        candidates.sort()
        return candidates[0][3]
    return None


# Strikethrough span — match a ``~~...~~`` block AND its contents so an edit
# history like ``~~old~~ new`` collapses to just ``new``. ``normalize_text``
# only strips the ``~~`` markers (leaving the crossed-out text behind), which
# is the wrong shape when the GT records the final clean value. The pattern
# is non-greedy and bounded to a single line so it can't span paragraphs.
_STRIKETHROUGH_SPAN_RE = re.compile(r"~~[^~\n]+~~")


def _strip_strikethrough_spans(s: str) -> str:
    return _STRIKETHROUGH_SPAN_RE.sub("", s).strip()


def _values_match_text(found: str, expected: str) -> bool:
    """Compare two form-field text values strictly.

    Form values are *extracted*, not estimated, so there is no role for
    similarity ratios or relative tolerance — a wrong digit, a wrong letter,
    or a swapped name component is a real mismatch. Two paths only:

    1. Exact match after normalization (``normalize_text`` already case-folds
       and collapses whitespace), which handles ``Madison`` vs ``madison``,
       trailing whitespace, and unicode quote variants.
    2. Strict numeric equality via ``normalize_number_string``, which lets
       ``1,234`` match ``1234`` and ``$1,234.00`` match ``1234`` (the same
       value written differently) — but rejects ``53703`` vs ``53704``.

    Strikethrough spans (``~~old~~ new``) are stripped from the *found*
    value before comparison so the parser's edit-history rendering matches
    the GT's clean final value. The expected side is left untouched on the
    assumption GT never contains ``~~``.
    """

    found_stripped = _strip_strikethrough_spans(found)

    f_norm = normalize_text(found_stripped)
    e_norm = normalize_text(expected)
    if f_norm == e_norm:
        return True
    if not f_norm or not e_norm:
        return False

    f_num = normalize_number_string(found_stripped)
    e_num = normalize_number_string(expected)
    if f_num is not None and e_num is not None and f_num == e_num:
        return True

    return False


# Trailing ``(row N)`` annotation used by the form-field test generator to
# point a label at a specific data row of a multi-column table. The column is
# named by the prefix; ``N`` is 1-indexed over data rows (header rows are
# skipped). We deliberately keep this strict and simple — anything else stays
# under the bold-colon / 2-col / glyph paths.
_ROW_LABEL_RE = re.compile(r"\s*\(row\s+(\d+)\)\s*$", re.IGNORECASE)


def _split_row_label(label: str) -> tuple[str, int] | None:
    """Return ``(column_label, row_index_1based)`` if *label* has a ``(row N)``
    suffix, else None."""

    m = _ROW_LABEL_RE.search(label)
    if not m:
        return None
    col_label = label[: m.start()].strip()
    if not col_label:
        return None
    return col_label, int(m.group(1))


def _column_header_for_index(table: TableData, col_idx: int) -> str:
    """Concatenate every header cell stacked above column *col_idx* into one
    label. If the table has no recorded column headers (e.g. a markdown table
    where row 0 is the de facto header), fall back to row 0 of that column."""

    parts: list[str] = []
    seen: set[str] = set()
    headers = getattr(table, "col_headers", {}) or {}
    for _, text in headers.get(col_idx, []):
        clean = (text or "").strip()
        if clean and clean not in seen:
            parts.append(clean)
            seen.add(clean)
    if parts:
        return " ".join(parts)
    if table.data.size and col_idx < table.data.shape[1]:
        return str(table.data[0, col_idx]).strip()
    return ""


def _find_table_cell_for_row_label(content: str, label: str) -> tuple[bool, str | None]:
    """Look up ``"<col_label> (row N)"`` in any multi-column table.

    Returns ``(label_seen, value_or_None)``. ``label_seen`` is True if a
    matching column was found in some table, even when the data row is out
    of range or the cell is empty — that distinction lets text rules with
    empty expected values pass on real empty cells without giving signature
    rules a free pass for missing labels.
    """

    parsed = _split_row_label(label)
    if parsed is None:
        return False, None
    col_label, row_n = parsed

    label_seen = False
    for table in parse_html_tables(content) + parse_markdown_tables(content):
        if table.data.size == 0:
            continue
        rows, cols = table.data.shape
        # Determine which rows are headers. For HTML tables, header_rows is
        # populated from <thead>/<th>. For markdown tables, parse_markdown_tables
        # records header_rows={0} when a separator row is present.
        header_rows = getattr(table, "header_rows", set()) or set()
        n_header = (max(header_rows) + 1) if header_rows else 0
        data_row_idx = n_header + (row_n - 1)

        for col_idx in range(cols):
            header_text = _column_header_for_index(table, col_idx)
            if not header_text:
                continue
            if not _label_matches(header_text, col_label):
                continue
            label_seen = True
            if 0 <= data_row_idx < rows:
                cell_value = str(table.data[data_row_idx, col_idx]).strip()
                if cell_value:
                    return True, cell_value
            # Column matched but cell out of range or empty — keep looking
            # in case a sibling table has the same header populated.

    if label_seen:
        return True, ""
    return False, None


def _scope_to_page(content: str, parse_output, page: int | None) -> str:  # type: ignore[no-untyped-def]
    """Return per-page markdown when ``parse_output`` and ``page`` are both set.

    Fail-closed: once per-page IR is present (``pages`` or ``layout_pages``),
    scoping is strict — if the requested page has no entry (or its markdown
    is empty), return ``""`` rather than the full document. The old lenient
    fallback let repeated header/footer fields satisfy page-N rules on the
    wrong page and silently masked page-level extraction failures (see
    PR #897 for the reducto/extend variant of the same bug).

    Only when no per-page IR is available (both lists empty) do we fall
    back to the document-level ``content``. Providers that emit neither
    list never had fair per-page scoring; the fallback preserves prior
    behavior rather than introducing a silent regression.

    When ``layout_pages`` carries the per-page split but ``md`` is empty,
    synthesize from ``items`` (priority ``md > html > value``). ``html``
    ranks above ``value`` so table items keep their structure for the
    HTML cell-neighbor matcher.

    ``parse_output`` is typed as ``ParseOutput`` upstream but kept loose
    here to avoid an import cycle.
    """

    if parse_output is None or page is None:
        return content

    pages = getattr(parse_output, "pages", None) or []
    layout_pages = getattr(parse_output, "layout_pages", None) or []

    if not pages and not layout_pages:
        # Provider produced no per-page IR at all — fall back to full doc.
        return content

    if pages:
        for p in pages:
            # PageIR.page_index is 0-indexed; rule.page is 1-indexed.
            if getattr(p, "page_index", None) == page - 1:
                return getattr(p, "markdown", "") or ""
        # ``pages`` populated but no matching page — fail closed.
        if not layout_pages:
            return ""
        # Fall through to ``layout_pages`` lookup; some providers populate
        # only one of the two lists per page.

    for lp in layout_pages:
        if getattr(lp, "page_number", None) != page:
            continue
        md = getattr(lp, "md", "") or ""
        if md:
            return md
        # Synthesize from items: md > html > value (html ranks above value
        # so table items keep their structure for the HTML cell matcher).
        parts: list[str] = []
        for it in getattr(lp, "items", None) or []:
            text = getattr(it, "md", "") or getattr(it, "html", "") or getattr(it, "value", "")
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    # Per-page IR present but page not found — fail closed.
    return ""


class FormFieldRule(ParseTestRule):
    """Test rule for form-field key-value extraction.

    Locates a labeled field by its visible label in the parsed markdown/HTML
    and checks the extracted value matches the expected one.
    """

    def __init__(self, rule_data: ParseFormFieldRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseFormFieldRule, self._rule_data)

        if self.type != ParseRuleType.FORM_FIELD.value:
            raise ValueError(f"Invalid type for FormFieldRule: {self.type}")

        self.label = rule_data.label
        self.value = rule_data.value
        self.value_type = rule_data.value_type

        if not self.label:
            raise ValueError("label field cannot be empty")

    def _content_for_match(self, md_content: str) -> str:
        return _scope_to_page(md_content, self.parse_output, self.page)

    def run(
        self,
        md_content: str,
        normalized_content: str | None = None,
    ) -> tuple[bool, str, float]:
        scoped = self._content_for_match(md_content)
        if self.value_type == "text":
            return self._run_text(scoped)
        if self.value_type == "checkbox":
            return self._run_checkbox(scoped)
        if self.value_type == "signature":
            return self._run_signature(scoped)
        return False, f"unknown value_type: {self.value_type}", 0.0

    def _run_text(self, content: str) -> tuple[bool, str, float]:
        # `self.value` can be a list of acceptable alternatives — pass if any matches.
        expected_alternatives = _value_alternatives(self.value)
        # Multi-col table cell lookup ("Column Name (row N)") takes precedence
        # over the bold-colon / 2-col / glyph paths because the suffix
        # explicitly names a tabular position.
        if _split_row_label(self.label) is not None:
            label_found, value = _find_table_cell_for_row_label(content, self.label)
        else:
            # Thread expected_alternatives so the matcher can disambiguate
            # among candidates at the same top label-match score. The rule's
            # position in the test list says nothing about which page
            # occurrence the GT refers to when a label legitimately repeats
            # (e.g. ``KB`` appearing as both an elevation label and as the
            # value of ``LOG MEASURED FROM`` in well-log headers). The GT
            # oracle is applied **only** to candidates tied at the head
            # label-match score, so it never leaks across adjacent labels
            # like Country vs County which live at different score levels.
            label_found, value = _find_text_value_for_label(content, self.label, expected_alternatives)
        if not label_found:
            return False, f"label not found: {self.label!r}", 0.0
        # value may be "" (label found, cell/value empty); _values_match_text
        # handles empty == empty correctly so empty-expected rules can pass.
        for expected in expected_alternatives:
            if _values_match_text(value or "", expected):
                return True, "match", 1.0
        if len(expected_alternatives) == 1:
            return False, f"expected {expected_alternatives[0]!r}, got {(value or '')!r}", 0.0
        return False, f"expected any of {expected_alternatives!r}, got {(value or '')!r}", 0.0

    def _run_checkbox(self, content: str) -> tuple[bool, str, float]:
        expected_bool = _coerce_bool(self.value)
        if expected_bool is None:
            return False, f"checkbox value must be coercible to bool, got {self.value!r}", 0.0

        # Prefer a real checkbox-shaped match.
        state = _find_checkbox_state_for_label(content, self.label)
        if state is None:
            # Fall back to a text-shaped value (e.g. **Married:** Yes / No).
            if _split_row_label(self.label) is not None:
                label_found, text_value = _find_table_cell_for_row_label(content, self.label)
            else:
                label_found, text_value = _find_text_value_for_label(content, self.label)
            if not label_found or not text_value:
                return False, f"label not found: {self.label!r}", 0.0
            state = _coerce_bool(text_value)
            if state is None:
                return False, f"could not interpret {text_value!r} as checkbox state", 0.0

        if state == expected_bool:
            return True, "match", 1.0
        return False, f"expected {expected_bool}, got {state}", 0.0

    def _run_signature(self, content: str) -> tuple[bool, str, float]:
        # Relaxed semantics: the rule's value is treated as a presence indicator,
        # not a strict bool. A non-empty string (e.g. the actual signed name) is
        # equivalent to True — the matcher only checks "is something signed here"
        # rather than the exact handwriting. An empty string / False / None means
        # "expected unsigned". A list value collapses the same way: any non-empty
        # alternative means "expected signed".
        if isinstance(self.value, bool):
            expected_signed = self.value
        elif isinstance(self.value, list):
            expected_signed = any(bool(str(v).strip()) for v in self.value)
        else:
            expected_signed = bool(str(self.value).strip())

        # Track label presence separately from value presence — an absent label
        # must NOT pass an "expected unsigned" rule. A form tuple benchmark
        # requires the parser to surface the field at all.
        label_found, text_value = _find_text_value_for_label(content, self.label)
        if not label_found:
            return False, f"label not found: {self.label!r}", 0.0
        signed = bool(text_value and text_value.strip())
        if signed == expected_signed:
            return True, "match", 1.0
        return False, f"expected signed={expected_signed}, got signed={signed}", 0.0
