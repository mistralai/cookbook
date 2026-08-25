"""Unified evidence + array-record extract metric.

One metric that subsumes both `array_record_*` (keyless Hungarian row matching,
order-insensitive, value-aware) and the v0.2 `extract_evidence_*` family
(OR-acceptable evidence values + page/bbox grounding) -- with NONE of the
`match_by`-rule dependence that makes the evidence metric cascade on dropped or
reordered rows.

Design
------
The value scorer is ``array_record``'s keyless Hungarian row assignment with
three additions:

1. **OR-acceptable values.** A ground-truth cell passes if the prediction
   matches its ``expected_output`` value *or any* alternate value declared in
   that leaf's ``evidence[]``. (When a leaf has a single evidence value equal to
   the expected value, this is just ``array_record``.)
2. **Sub-record recursion.** A list-of-objects subfield (e.g. OFAC
   ``entities[].aliases``) is recursed into and aligned per sub-row, scoring
   each of its fields, instead of being compared as one opaque cell.
   Scalars, scalar lists, and **bare objects stay opaque cells** -- an
   object-valued cell is scored all-or-nothing, NOT field-by-field, so this is
   not universal leaf-level scoring for object-valued cells (that would diverge
   from array_record's structure).
3. **Grounding (bbox + page).** Two parallel sets of counters, both value-gated
   and both nesting under the value score. A cell is *grounded-correct* when the
   value matches AND the prediction's citation has a bbox that overlaps an
   evidence bbox on the same page (IoU >= threshold) -- a value-gated
   **bbox-IoU** metric, emitted as ``*_grounded_*``. A cell is *page-correct*
   when the value matches AND the citation's page is in the evidence's page set
   -- the coarser **page** check (no IoU), emitted as ``*_page_*``. Page is a
   per-cell superset of bbox (a box match requires equal pages, and page
   evidence/claims are supersets of bbox evidence/claims), so the three F1s nest:
   ``value_f1 >= page_f1 >= grounded_f1``. Together ``*_grounded_*`` and
   ``*_page_*`` form one nested precision/recall/F1 trio.
   Grounding is only defined where the GT carries the corresponding annotation
   (a page for ``*_page_*``, a bbox for ``*_grounded_*``), on
   BOTH sides of the P/R pair: recall's denominator is the GT cells that carry
   an evidence bbox, and precision's denominator is the predicted citation-bbox
   claims on cells *aligned to such a GT cell*. A claim on a cell whose GT has
   no bbox (or on an extra predicted row with no GT counterpart) is ungradeable
   -- it can never be verified right or wrong -- so it is excluded from the
   denominator rather than counted as a miss. Without this, a sparsely
   annotated document (e.g. one bbox-bearing cover field among 30k value-only
   cells) pins precision to ~0 for every pipeline that cites everything. A
   document whose GT has no evidence bbox at all emits **no** ``*_grounded_*``
   metric (excluded from the dataset average), rather than a 0.0 that would
   punish a document that simply cannot be graded for grounding.

On a document with no object-array subfields and no alternate evidence values,
the ``*_value_*`` metrics are bit-identical to ``array_record_*`` -- same
alignment, subfields, cost matrix, and denominators -- with no ``match_by`` rule
anywhere. Object-array subfields make the value metrics diverge upward (the
per-sub-record credit array_record cannot give); the ``*_grounded_*`` metrics
are the other new signal (0 for pipelines that emit no citations *on documents
whose GT carries bboxes*; omitted entirely on documents whose GT has none).
Truncation
drops recall exactly as in ``array_record``: an unaligned ground-truth row
contributes 0 correct against a full denominator, so there is no vacuous null
pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from extract_bench.evaluation.metrics.extract.array_record_match_metric import (
    DEFAULT_FUZZY_FIELD_THRESHOLDS,
    RESERVED_OUTPUT_KEYS,
    _array_subfields,
    _cell_match,
    _is_array_schema,
    _mismatch_cost_matrix,
    _normalize_dates_deep,
    _normalize_ws,
    _peel_exact_row_matches,
    _unwrap_value,
)
from extract_bench.evaluation.metrics.extract.json_subset_match import normalize_date_string
from extract_bench.schemas.evaluation import MetricValue
from extract_bench.test_cases.schema import ExtractFieldTestRule, iter_rule_evidence

# Opt-in per-field normalizers (``_field_rules[path].normalizers``). Each is a
# fail->pass-only fallback applied after the exact cell match misses, so setting
# one can never turn a passing cell into a failure. They exist for scanned-form
# fields where the printed template makes one strict reading unfair (preprinted
# units, checkbox blanks, typewriter case, split preprinted years).
_OPTIONAL_TERMINAL_PUNCT = "optional_terminal_punctuation"
_CASE_INSENSITIVE = "case_insensitive"
_NULL_EQUALS_FALSE = "null_equals_false"
_PHONE_DIGITS = "phone_digits"
_LENIENT_DATE = "lenient_date"
_PUNCTUATION_SPACING = "punctuation_spacing"
# Above this many cells (GT rows x predicted rows) in one flat array, the
# grounded assignment matrix -- an int16 cost matrix plus scipy's internal
# float64 copy -- is several GB and a memory-limited runner OOMs building it.
# Such an array skips grounding entirely (``grounded_incomplete``) rather than
# scoring it partially, and the value score takes the exact-row peel, which is
# value-identical for opaque un-grounded cells. Only arrays with no sub-records
# qualify, so the peel never shifts nested TP. 100M cells keeps the matrix under
# ~1 GB while sparing every ordinary document.
_GROUNDED_MAX_CELLS = 100_000_000
# Everything ``configured_cell_match`` handles; kept equal to the schema's
# EXTRACT_FIELD_NORMALIZERS vocabulary (guarded by a unit test) so a name can't
# validate at load time yet silently no-op here.
SUPPORTED_NORMALIZERS: frozenset[str] = frozenset(
    {
        _OPTIONAL_TERMINAL_PUNCT,
        _CASE_INSENSITIVE,
        _NULL_EQUALS_FALSE,
        _PHONE_DIGITS,
        _LENIENT_DATE,
        _PUNCTUATION_SPACING,
    }
)
# One optional terminal punctuation char, not a whole run: "item 5." forgives
# the dot, but "item 5;,,." stays distinct from "item 5".
_TERMINAL_PUNCT_RE = re.compile(r"\s*[.,;:]\s*$")
_PUNCT_SPACING_RE = re.compile(r"\s*([,.;:])\s*", re.U)
_NON_DIGIT_RE = re.compile(r"\D+")
_DIGIT_RE = re.compile(r"\d")
# Preprinted decade split: a filer types "55" after the printed "19", so
# transcriptions render "19 55". Only joined when a digit (a day) already
# appears before the pair -- "June 19 44" is a day 19 plus a 2-digit year,
# not a split 1944.
_SPLIT_YEAR_RE = re.compile(r"\b(19|20)\s+(\d{2})\b")
_TRAILING_2DIGIT_YEAR_RE = re.compile(r"^(.*[ ,/-])(\d{2})\s*$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@lru_cache(maxsize=8192)
def _lenient_date_probes(value: str) -> frozenset[str]:
    """Date readings of a scanned-form value: split-year joined, and 2-digit
    trailing years expanded into both centuries ("June 17, 43" -> 1943/2043).

    Guardrails against equating different values:
    - Both rewrites require a digit (a day) before the year token, so a lone
      trailing pair ("May 11") is never century-expanded.
    - A rewritten probe only counts when it parses to an ISO date; unparseable
      rewrites are discarded rather than matched as raw strings ("Permit 12-34"
      vs "Permit 12-19 34" must not intersect on a fabricated "Permit 12-1934").
    """
    probes = {normalize_date_string(value)}
    rewrites: list[str] = []
    m = _SPLIT_YEAR_RE.search(value)
    if m and _DIGIT_RE.search(value[: m.start()]):
        rewrites.append(_SPLIT_YEAR_RE.sub(r"\1\2", value))
    for base in (value, *rewrites):
        m = _TRAILING_2DIGIT_YEAR_RE.match(base)
        if m and _DIGIT_RE.search(m.group(1)):
            rewrites += [f"{m.group(1)}19{m.group(2)}", f"{m.group(1)}20{m.group(2)}"]
    for rewrite in rewrites:
        normalized = normalize_date_string(rewrite)
        if _ISO_DATE_RE.match(normalized):
            probes.add(normalized)
    return frozenset(probes)


def configured_cell_match(
    expected: Any,
    actual: Any,
    field: str,
    fuzzy: dict[str, float],
    normalizers: tuple[str, ...] | None,
) -> bool:
    """``_cell_match`` plus the field's opt-in normalizer fallbacks."""
    if _cell_match(expected, actual, field, fuzzy):
        return True
    if not normalizers:
        return False
    if _NULL_EQUALS_FALSE in normalizers:
        # A blank checkbox reads as False or as an omitted/null field with equal
        # right; identity checks keep 0/"" from sneaking in via == coercion.
        if (expected is False or expected is None) and (actual is False or actual is None):
            return True
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    if _OPTIONAL_TERMINAL_PUNCT in normalizers:
        expected_norm = _TERMINAL_PUNCT_RE.sub("", expected.strip())
        actual_norm = _TERMINAL_PUNCT_RE.sub("", actual.strip())
        # Both sides must survive the strip: punctuation-only values ('.' vs
        # ';') would otherwise collapse to '' == '' and match each other.
        if expected_norm and actual_norm:
            if _cell_match(expected_norm, actual_norm, field, fuzzy):
                return True
    if _PUNCTUATION_SPACING in normalizers:
        expected_norm = _PUNCT_SPACING_RE.sub(r"\1", expected)
        actual_norm = _PUNCT_SPACING_RE.sub(r"\1", actual)
        if _cell_match(expected_norm, actual_norm, field, fuzzy):
            return True
    if _CASE_INSENSITIVE in normalizers:
        if _normalize_ws(expected).casefold() == _normalize_ws(actual).casefold():
            return True
    if _PHONE_DIGITS in normalizers:
        expected_digits = _NON_DIGIT_RE.sub("", expected)
        actual_digits = _NON_DIGIT_RE.sub("", actual)
        if len(expected_digits) >= 10 and len(actual_digits) >= 10:
            if expected_digits[-10:] == actual_digits[-10:]:
                return True
        elif expected_digits and expected_digits == actual_digits:
            return True
    if _LENIENT_DATE in normalizers:
        if _lenient_date_probes(expected) & _lenient_date_probes(actual):
            return True
    return False


@dataclass
class _Counts:
    """Pooled tp/denominator counts for value and grounded scoring."""

    v_correct: int = 0  # value matches over aligned cells (tp for precision AND recall)
    expected: int = 0  # GT leaf cells (value recall denominator)
    predicted: int = 0  # predicted leaf cells (value precision denominator)
    g_correct: int = 0  # value+grounding correct cells (grounded tp)
    g_expected: int = 0  # GT cells whose evidence carries a bbox (grounded recall denom)
    # Predicted citation-bbox claims on cells aligned to a bbox-bearing GT cell
    # (grounded precision denom). Claims on cells whose GT has no bbox -- or on
    # extra predicted rows with no GT counterpart -- are ungradeable and stay out.
    g_claims: int = 0
    # Page-level counterpart of the grounded (bbox) counters: a cell is
    # page-correct when the value matches AND the prediction cites a page in the
    # GT evidence's page set. Coarser than bbox (no IoU), so page is a superset:
    # every bbox-bearing cell also carries a page, and a bbox match implies a
    # page match -- hence p_correct >= g_correct and p_expected >= g_expected,
    # giving value_f1 >= page_f1 >= grounded_f1.
    p_correct: int = 0  # value+page correct cells (page tp)
    p_expected: int = 0  # GT cells whose evidence carries a page (page recall denom)
    p_claims: int = 0  # predicted page claims on cells aligned to a page-bearing GT cell (page precision denom)
    expected_rows: int = 0
    predicted_rows: int = 0
    grounded_incomplete: bool = False  # a giant array skipped grounding -> emit no grounded metric
    value_incomplete: bool = False  # a giant flat array skipped the residual value matrix -> value is a lower bound


def _is_record_subfield(rows: list[Any], sub: str) -> bool:
    """True when a subfield holds a list of objects in any row (a real sub-record).

    Scalar lists (footnote markers, tag strings) and bare objects are NOT
    sub-records: they stay opaque cells scored exactly as array_record does, so
    a doc with no object-array subfields reduces to array_record bit-for-bit.
    """
    for row in rows:
        if isinstance(row, dict):
            value = row.get(sub)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return True
    return False


def _iou_xywh(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a[0], a[1], a[2], a[3]
    bx, by, bw, bh = b[0], b[1], b[2], b[3]
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class _Scorer:
    def __init__(
        self,
        alt_values: dict[str, list[Any]],
        evidence_boxes: dict[str, list[tuple[int, list[float]]]],
        evidence_pages: dict[str, set[int]],
        normalizers: dict[str, tuple[str, ...]],
        pred_boxes: dict[str, list[tuple[int, list[float]]]],
        pred_pages: dict[str, set[int]],
        fuzzy: dict[str, float],
        iou_threshold: float,
    ) -> None:
        self._alt = alt_values
        self._ev_boxes = evidence_boxes
        self._ev_pages = evidence_pages
        self._normalizers = normalizers
        self._pred_boxes = pred_boxes
        self._pred_pages = pred_pages
        self._fuzzy = fuzzy
        self._iou = iou_threshold

    def _configured_cell_match(self, gt_path: str, expected: Any, actual: Any, field: str) -> bool:
        return configured_cell_match(expected, actual, field, self._fuzzy, self._normalizers.get(gt_path))

    def _value_match(self, gt_path: str, canonical: Any, actual: Any, field: str) -> bool:
        """OR-acceptable: the prediction matches the expected value or any evidence value."""
        for candidate in (canonical, *self._alt.get(gt_path, ())):
            if self._configured_cell_match(gt_path, candidate, actual, field):
                return True
        return False

    def _box_match(self, gt_path: str, pred_path: str) -> bool:
        gt_boxes = self._ev_boxes.get(gt_path)
        pred = self._pred_boxes.get(pred_path)
        if not gt_boxes or not pred:
            return False
        return any(gp == pp and _iou_xywh(gb, pb) >= self._iou for gp, gb in gt_boxes for pp, pb in pred)

    def _page_match(self, gt_path: str, pred_path: str) -> bool:
        """A cited page for pred_path lands in the GT evidence's page set for
        gt_path. Coarser than ``_box_match`` (no IoU) and implied by it: a box
        match requires the pages to be equal, so every box match is a page
        match."""
        gt_pages = self._ev_pages.get(gt_path)
        pred_pages = self._pred_pages.get(pred_path)
        if not gt_pages or not pred_pages:
            return False
        return not gt_pages.isdisjoint(pred_pages)

    def _score_cell(
        self, gt_path: str, pred_path: str, canonical: Any, actual: Any, field: str, c: _Counts, ground: bool = True
    ) -> bool:
        """Score one aligned cell (value + page + grounding). Caller owns the denominators.

        ``ground=False`` scores value only, for cells in a giant array whose
        grounding-family scoring was skipped (their g_expected/g_claims and
        p_expected/p_claims were not counted either, so g_correct/p_correct must
        not be counted here or the grounded/page precision/recall would be
        computed against a truncated denominator).
        """
        if self._value_match(gt_path, canonical, actual, field):
            c.v_correct += 1
            if ground:
                if self._box_match(gt_path, pred_path):
                    c.g_correct += 1
                if self._page_match(gt_path, pred_path):
                    c.p_correct += 1
            return True
        return False

    def _count_subtree(self, path: str, value: Any, schema: dict[str, Any], c: _Counts, *, expected: bool) -> None:
        """Count an unmatched subtree's leaves on one side only (recall or precision miss)."""
        if isinstance(value, list) and any(isinstance(r, dict) for r in value):
            subs = _array_subfields(schema if isinstance(schema, dict) else {}, value)
            item_sch = (schema or {}).get("items", {}).get("properties", {})
            for i, row in enumerate(value):
                rd = row if isinstance(row, dict) else {}
                for s in subs:
                    self._count_subtree(f"{path}[{i}].{s}", rd.get(s), item_sch.get(s, {}), c, expected=expected)
        elif expected:
            c.expected += 1
            if self._ev_boxes.get(path):
                c.g_expected += 1
            if self._ev_pages.get(path):
                c.p_expected += 1
        else:
            # Extra predicted subtree: no GT counterpart exists, so its citation
            # bboxes are ungradeable and do not enter the grounded precision
            # denominator (see _Counts.g_claims).
            c.predicted += 1

    def _score_array(
        self, gt_field: str, pred_field: str, exp_rows: Any, act_rows: Any, schema: dict[str, Any], c: _Counts
    ) -> None:
        # gt_field / pred_field are separate base paths: ground-truth cells (alt
        # values, evidence boxes) key off gt_field; predicted cells (citations)
        # key off pred_field. They differ once the outer array is reordered, so
        # threading both is what keeps nested grounding correct after a remap.
        exp_rows = exp_rows if isinstance(exp_rows, list) else []
        act_rows = act_rows if isinstance(act_rows, list) else []
        subs = _array_subfields(schema if isinstance(schema, dict) else {}, exp_rows)
        if not subs:
            return
        item_sch = (schema or {}).get("items", {}).get("properties", {})
        # A list-of-objects subfield is a real sub-record -> recurse (the depth
        # array_record lacks). Scalars, scalar lists, and bare objects stay
        # opaque cells, so a doc with no sub-record arrays scores bit-identically
        # to array_record.
        record_subs = [s for s in subs if _is_record_subfield(exp_rows, s)]
        cell_subs = [s for s in subs if s not in record_subs]
        # Opaque-cell denominators: array_record-exact (rows x cell subfields).
        c.expected += len(exp_rows) * len(cell_subs)
        c.predicted += len(act_rows) * len(cell_subs)
        c.expected_rows += len(exp_rows)
        c.predicted_rows += len(act_rows)
        # A flat array whose full grounded matrix would be multi-GB: skip its
        # grounding-family scoring (bbox AND page). Leaving has_ev_page/
        # has_pred_page False routes the value score to the exact-row peel below
        # (value-identical for opaque cells), and not counting
        # g_expected/g_claims/p_expected/p_claims here keeps the grounded and
        # page denominators consistent with the skipped g_correct/p_correct.
        # ``not record_subs`` guarantees no nested recursion, so the peel cannot
        # shift nested TP. The memory win only lands on the plain peel path: an
        # array with genuine alternate values (``multi``) or field normalizers
        # still builds the full candidate matrix below regardless of ``giant`` --
        # there grounding is withheld but the allocation is unchanged. The
        # threshold bounds the FULL matrix; the peel's residual matrix over
        # unmatched rows can still grow if few rows match exactly, so treat it as
        # a heuristic, not a hard ceiling.
        giant = bool(
            exp_rows
            and act_rows
            and cell_subs
            and not record_subs
            and len(exp_rows) * len(act_rows) > _GROUNDED_MAX_CELLS
        )
        # Page presence drives both the grounded-family denominators and the
        # pairing-sensitive branch selection below. It is a SUPERSET of bbox
        # presence (every bbox-bearing evidence/citation also carries a page), so
        # probing pages alone detects any dropped grounding -- bbox or page-only.
        has_ev_page = False
        has_pred_page = False
        if giant:
            # Existence-only probe (early-exit) so we still know whether any
            # grounding-family scoring was actually dropped -- only then is the
            # document's grounded/page score incomplete and must be withheld
            # entirely. Pages are the superset, so this covers dropped bboxes too.
            dropped_grounding = any(
                self._ev_pages.get(f"{gt_field}[{i}].{s}") for i in range(len(exp_rows)) for s in cell_subs
            ) or any(self._pred_pages.get(f"{pred_field}[{j}].{s}") for j in range(len(act_rows)) for s in cell_subs)
            if dropped_grounding:
                c.grounded_incomplete = True
        else:
            for i in range(len(exp_rows)):
                for s in cell_subs:
                    gt_path = f"{gt_field}[{i}].{s}"
                    if self._ev_boxes.get(gt_path):
                        c.g_expected += 1
                    if self._ev_pages.get(gt_path):
                        c.p_expected += 1
                        has_ev_page = True
            # Flag-only scan: g_claims/p_claims are counted per aligned pair
            # below, and only for cells whose GT side carries a bbox/page
            # (ungradeable claims stay out of the precision denominator). The flag
            # reflects ANY predicted page (the superset of any predicted bbox) so
            # the pairing-sensitive branch selection below covers both families.
            for j in range(len(act_rows)):
                for s in cell_subs:
                    if self._pred_pages.get(f"{pred_field}[{j}].{s}"):
                        has_pred_page = True
                        break
                if has_pred_page:
                    break
        # Sub-record subfields are counted by the recursion below, per matched
        # pair (and as one-sided misses for unmatched rows).
        match_for_exp: dict[int, int] = {}
        if exp_rows and act_rows:
            cost_subs = cell_subs or subs  # align on identity cells; fall back to all
            fuzzy = self._fuzzy
            exp_dicts = [e if isinstance(e, dict) else {} for e in exp_rows]
            # The alt index holds every leaf's evidence value, so most entries
            # equal the expected value. A genuinely different alternate expands
            # the zero-cost graph; in that case use full assignment semantics and
            # do not greedily peel canonical exact matches.
            multi = False
            for i, ed in enumerate(exp_dicts):
                for s in cost_subs:
                    canon = ed.get(s)
                    alt = self._alt.get(f"{gt_field}[{i}].{s}")
                    if alt:
                        seen = [canon]
                        for value in alt:
                            if value != canon and value not in seen:
                                multi = True
                                break
                            seen.append(value)
                    if multi:
                        break
                if multi:
                    break

            # Skipped when `multi` already forces the candidate path or no rule
            # carries normalizers at all (the default): the path scan is
            # O(rows x subfields) per array and would otherwise run for nothing.
            has_normalizer = (
                bool(self._normalizers)
                and not multi
                and any(self._normalizers.get(f"{gt_field}[{i}].{s}") for i in range(len(exp_rows)) for s in cost_subs)
            )

            if multi or has_normalizer:
                exp_cands = []
                for i, ed in enumerate(exp_dicts):
                    row_cands = []
                    for s in cost_subs:
                        canon = ed.get(s)
                        alt = self._alt.get(f"{gt_field}[{i}].{s}")
                        cand = [canon]
                        if alt:
                            cand += [v for v in alt if v != canon and v not in cand]
                        row_cands.append(cand)
                    exp_cands.append(row_cands)
                cost = np.empty((len(act_rows), len(exp_rows)), dtype=np.int16)
                for j, arow in enumerate(act_rows):
                    ad = arow if isinstance(arow, dict) else {}
                    avals = [ad.get(s) for s in cost_subs]
                    for i, cands in enumerate(exp_cands):
                        cost[j, i] = sum(
                            not any(
                                self._configured_cell_match(
                                    f"{gt_field}[{i}].{cost_subs[si]}", cv, avals[si], cost_subs[si]
                                )
                                for cv in cands[si]
                            )
                            for si in range(len(cost_subs))
                        )
                row_ind, col_ind = linear_sum_assignment(cost)
                match_for_exp = {int(i): int(j) for j, i in zip(row_ind, col_ind, strict=True)}
            elif record_subs or (has_ev_page and has_pred_page):
                # Pairing-sensitive scoring: nested sub-records recurse on the
                # matched predicted index, and per-cell grounding + page grading
                # key off it too. Among equal-cost optima the exact-row peel could
                # pick a different pairing than the full assignment and shift
                # grounded / page / nested TP, so fall back to full assignment to
                # preserve the exact tie-break. ``has_ev_page``/``has_pred_page``
                # are the bbox supersets, so this branch also covers every
                # bbox-grounded array.
                cost = _mismatch_cost_matrix(act_rows, exp_rows, cost_subs, fuzzy)
                row_ind, col_ind = linear_sum_assignment(cost)
                match_for_exp = {int(i): int(j) for j, i in zip(row_ind, col_ind, strict=True)}
            else:
                # Opaque-cell-only, un-grounded: the value score is pairing-
                # independent (n_pairs*k - total_cost), so peeling exact rows is
                # safe. Reuse the vectorized interned matrix on the residual rows.
                assignment = _peel_exact_row_matches(act_rows, exp_rows, cost_subs, fuzzy)
                match_for_exp = {int(i): int(j) for j, i in assignment.pairs}
                if assignment.unmatched_actual_indices and assignment.unmatched_expected_indices:
                    residual_actual = [act_rows[idx] for idx in assignment.unmatched_actual_indices]
                    residual_expected = [exp_rows[idx] for idx in assignment.unmatched_expected_indices]
                    # Memory ceiling on the residual assignment matrix (parallel to
                    # the _GROUNDED_MAX_CELLS grounding guard above). After the
                    # exact-row peel, a flat array whose prediction diverges leaves
                    # a residual ~= the full array; the dense int16 cost matrix plus
                    # scipy's internal float64 copy is tens of GB at ~50k+ rows and
                    # OOMs the eval runner (a 77,400-row array needs ~67 GB). Over
                    # the cell budget, skip the residual assignment: those rows stay
                    # unmatched (scored as value misses below) and the value score is
                    # flagged ``value_incomplete`` -- a lower bound, not a crash.
                    if len(residual_actual) * len(residual_expected) > _GROUNDED_MAX_CELLS:
                        c.value_incomplete = True
                    else:
                        cost = _mismatch_cost_matrix(residual_actual, residual_expected, cost_subs, fuzzy)
                        row_ind, col_ind = linear_sum_assignment(cost)
                        match_for_exp.update(
                            {
                                int(assignment.unmatched_expected_indices[int(expected_idx)]): int(
                                    assignment.unmatched_actual_indices[int(actual_idx)]
                                )
                                for actual_idx, expected_idx in zip(row_ind, col_ind, strict=True)
                            }
                        )
        matched_act = set(match_for_exp.values())
        for i, erow in enumerate(exp_rows):
            ed = erow if isinstance(erow, dict) else {}
            mj = match_for_exp.get(i)
            if mj is not None:
                ad = act_rows[mj] if isinstance(act_rows[mj], dict) else {}
                for s in cell_subs:
                    gt_path = f"{gt_field}[{i}].{s}"
                    pred_path = f"{pred_field}[{mj}].{s}"
                    if not giant:
                        if self._ev_boxes.get(gt_path) and self._pred_boxes.get(pred_path):
                            c.g_claims += 1
                        if self._ev_pages.get(gt_path) and self._pred_pages.get(pred_path):
                            c.p_claims += 1
                    self._score_cell(gt_path, pred_path, ed.get(s), ad.get(s), s, c, ground=not giant)
                for s in record_subs:  # recurse with the matched predicted index, not the GT index
                    self._score_array(
                        f"{gt_field}[{i}].{s}", f"{pred_field}[{mj}].{s}", ed.get(s), ad.get(s), item_sch.get(s, {}), c
                    )
            else:  # unmatched GT row: cell subfields already counted; recurse sub-records as recall misses
                for s in record_subs:
                    self._count_subtree(f"{gt_field}[{i}].{s}", ed.get(s), item_sch.get(s, {}), c, expected=True)
        for j, arow in enumerate(act_rows):  # extra predicted rows: sub-records are precision misses
            if j in matched_act:
                continue
            ad = arow if isinstance(arow, dict) else {}
            for s in record_subs:
                self._count_subtree(f"{pred_field}[{j}].{s}", ad.get(s), item_sch.get(s, {}), c, expected=False)

    def score_root(self, expected: dict[str, Any], actual: dict[str, Any], schema_props: dict[str, Any]) -> _Counts:
        c = _Counts()
        for field in sorted((set(expected) | set(actual)) - RESERVED_OUTPUT_KEYS):
            fsch = schema_props.get(field, {})
            ev = expected.get(field)
            av = actual.get(field)
            if _is_array_schema(fsch, ev):
                self._score_array(field, field, ev, av, fsch, c)
            else:
                # Scalar or shallow-object cell, scored opaquely (array_record parity).
                c.expected += 1
                if self._ev_boxes.get(field):
                    c.g_expected += 1
                    if self._pred_boxes.get(field):
                        c.g_claims += 1
                if self._ev_pages.get(field):
                    c.p_expected += 1
                    if self._pred_pages.get(field):
                        c.p_claims += 1
                self._score_cell(field, field, ev, av, field, c)
                # An omitted key is an implicit null prediction and always
                # enters the precision denominator, right or wrong -- exactly as
                # if the model had asserted null explicitly. Counting it only
                # when correct would let tp exceed predicted (precision > 1.0),
                # and counting it never would make omitting an uncertain field
                # strictly outscore asserting it.
                c.predicted += 1
        return c


def _build_rule_indexes(
    field_rules: list[ExtractFieldTestRule],
) -> tuple[
    dict[str, list[Any]],
    dict[str, list[tuple[int, list[float]]]],
    dict[str, set[int]],
    dict[str, tuple[str, ...]],
]:
    """field_path -> alternate values; -> evidence boxes; -> evidence pages; -> normalizers.

    ``pages`` is a superset of the pages implied by ``boxes``: it holds every
    page any evidence entry carries, even entries with no bbox, so a page-only
    annotation still makes a cell page-gradeable.
    """
    alt: dict[str, list[Any]] = {}
    boxes: dict[str, list[tuple[int, list[float]]]] = {}
    pages: dict[str, set[int]] = {}
    normalizers: dict[str, tuple[str, ...]] = {}
    for rule in field_rules:
        path = rule.field_path
        if rule.normalizers:
            normalizers[path] = tuple(rule.normalizers)
        entries = iter_rule_evidence(rule)
        vals = [e.value for e in entries if e.value is not None]
        if vals:
            alt.setdefault(path, []).extend(vals)
        for e in entries:
            if e.page is not None:
                pages.setdefault(path, set()).add(int(e.page))
                if e.bbox is not None and len(e.bbox) >= 4:
                    boxes.setdefault(path, []).append((int(e.page), [float(x) for x in e.bbox[:4]]))
    return alt, boxes, pages, normalizers


def _index_citations(
    field_citations: list[Any],
) -> tuple[dict[str, list[tuple[int, list[float]]]], dict[str, set[int]]]:
    """field_path -> citation boxes; field_path -> citation pages.

    A page-only citation (page present, bbox missing) is indexed for pages but
    not for boxes, so it can satisfy a page match without ever being credited as
    a bbox match.
    """
    boxes: dict[str, list[tuple[int, list[float]]]] = {}
    pages: dict[str, set[int]] = {}
    for cit in field_citations or []:
        path = cit.get("field_path") if isinstance(cit, dict) else getattr(cit, "field_path", None)
        page = cit.get("page") if isinstance(cit, dict) else getattr(cit, "page", None)
        bbox = cit.get("bbox") if isinstance(cit, dict) else getattr(cit, "bbox", None)
        if not path or page is None:
            continue
        pages.setdefault(path, set()).add(int(page))
        if bbox and len(bbox) >= 4:
            boxes.setdefault(path, []).append((int(page), [float(x) for x in bbox[:4]]))
    return boxes, pages


def _prf(tp: int, denom_p: int, denom_r: int) -> tuple[float, float, float]:
    precision = tp / denom_p if denom_p else 0.0
    recall = tp / denom_r if denom_r else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def compute_unified_evidence_metrics(
    expected_output: Any,
    extracted_data: Any,
    field_rules: list[ExtractFieldTestRule],
    field_citations: list[Any] | None = None,
    data_schema: dict[str, Any] | None = None,
    *,
    fuzzy_field_thresholds: dict[str, float] | None = None,
    normalize_dates: bool = True,
    bbox_iou_threshold: float = 0.5,
) -> list[MetricValue]:
    """Value + grounded precision/recall/F1 under keyless Hungarian alignment.

    Returns an empty list when either side is not a dict (no array structure to
    score), matching ``array_record``'s ``counts is None`` guard.
    """
    expected = _unwrap_value(expected_output)
    actual = _unwrap_value(extracted_data)
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return []
    if normalize_dates:
        expected = _normalize_dates_deep(expected)
        actual = _normalize_dates_deep(actual)

    fuzzy = dict(DEFAULT_FUZZY_FIELD_THRESHOLDS if fuzzy_field_thresholds is None else fuzzy_field_thresholds)
    alt, ev_boxes, ev_pages, normalizers = _build_rule_indexes(field_rules)
    if normalize_dates:
        alt = {p: _normalize_dates_deep(v) for p, v in alt.items()}
    pred_boxes, pred_pages = _index_citations(field_citations or [])
    schema_props = (data_schema or {}).get("properties", {})

    c = _Scorer(alt, ev_boxes, ev_pages, normalizers, pred_boxes, pred_pages, fuzzy, bbox_iou_threshold).score_root(
        expected, actual, schema_props
    )
    if c.expected == 0 and c.predicted == 0:
        return []

    vp, vr, vf1 = _prf(c.v_correct, c.predicted, c.expected)
    meta = {
        "v_correct": c.v_correct,
        "expected_cells": c.expected,
        "predicted_cells": c.predicted,
        "g_correct": c.g_correct,
        "grounded_expected_cells": c.g_expected,
        "grounded_pred_claims": c.g_claims,
        "p_correct": c.p_correct,
        "page_expected_cells": c.p_expected,
        "page_pred_claims": c.p_claims,
        "expected_rows": c.expected_rows,
        "predicted_rows": c.predicted_rows,
        "bbox_iou_threshold": bbox_iou_threshold,
        "grounded_incomplete": c.grounded_incomplete,
        "value_incomplete": c.value_incomplete,
    }
    metrics = [
        MetricValue(metric_name="extract_unified_value_precision", value=vp, metadata={**meta, "tp": c.v_correct}),
        MetricValue(metric_name="extract_unified_value_recall", value=vr, metadata={**meta, "tp": c.v_correct}),
        MetricValue(metric_name="extract_unified_value_f1", value=vf1, metadata=meta),
    ]
    # Page grounding is the coarse counterpart of the bbox grounding below:
    # value + correct-page, no IoU. It is emitted under the SAME
    # bbox-bearing-only guard shape as grounded (``c.p_expected`` counts GT cells
    # whose evidence carries a page; ``grounded_incomplete`` withholds it when a
    # giant array skipped pairing-sensitive scoring), and it sits between value
    # and grounded so the three F1s nest: value_f1 >= page_f1 >= grounded_f1.
    # (Page is a per-cell superset of bbox -- box match implies page match and
    # page evidence/claims are supersets of bbox evidence/claims -- so a
    # document that emits a grounded metric always emits a page metric too.)
    if c.p_expected > 0 and not c.grounded_incomplete:
        pp, pr, pf1 = _prf(c.p_correct, c.p_claims, c.p_expected)
        metrics += [
            MetricValue(metric_name="extract_unified_page_precision", value=pp, metadata={**meta, "tp": c.p_correct}),
            MetricValue(metric_name="extract_unified_page_recall", value=pr, metadata={**meta, "tp": c.p_correct}),
            MetricValue(metric_name="extract_unified_page_f1", value=pf1, metadata=meta),
        ]
    # Grounding is only defined where the ground truth carries bounding boxes.
    # A document whose GT has NO evidence bbox (``c.g_expected == 0``) cannot be
    # scored for grounding at all, so we emit NO ``*_grounded_*`` metric for it
    # -- rather than a 0.0 that the runner would average in as if the pipeline
    # had failed to ground. This makes the dataset-level grounded P/R/F1 an
    # average over bbox-bearing documents only (mirroring how the runner already
    # excludes docs that don't emit a metric). Within a bbox-bearing document the
    # same principle applies per cell: precision's denominator (``g_claims``)
    # only counts citation-bbox claims on cells aligned to a bbox-bearing GT
    # cell, so a sparsely annotated GT (few bbox cells among many value-only
    # cells) grades only what it can verify instead of pinning precision to ~0.
    # A document whose GT *does* carry boxes but whose prediction emits none on
    # those cells still scores 0.0 here (``g_claims``/``g_correct == 0``): that
    # is a real grounding miss and the 0.0 is kept.
    # ``grounded_incomplete`` means at least one giant array's grounding was
    # skipped to bound eval memory, so any grounded score would be partial and
    # misleading. Withhold all grounded metrics for the document instead (the
    # value metrics above are unaffected -- they are exact).
    if c.g_expected > 0 and not c.grounded_incomplete:
        gp, gr, gf1 = _prf(c.g_correct, c.g_claims, c.g_expected)
        metrics += [
            MetricValue(
                metric_name="extract_unified_grounded_precision", value=gp, metadata={**meta, "tp": c.g_correct}
            ),
            MetricValue(metric_name="extract_unified_grounded_recall", value=gr, metadata={**meta, "tp": c.g_correct}),
            MetricValue(metric_name="extract_unified_grounded_f1", value=gf1, metadata=meta),
        ]
    return metrics
