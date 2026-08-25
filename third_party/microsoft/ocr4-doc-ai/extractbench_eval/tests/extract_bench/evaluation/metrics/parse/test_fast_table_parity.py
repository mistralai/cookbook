"""Exact-parity tests for the fast table-metric paths.

Two speedups are covered here, both of which must leave the *scores* unchanged:

1. ``fast_tree_edit.tree_edit_distance`` (Zhang-Shasha, numba/pure-Python) is an
   exact replacement for APTED inside TEDS. Zhang-Shasha and APTED both compute
   the optimal *ordered* tree-edit distance, so they agree by construction. We
   pin this down on crafted adversarial tables and on many randomly generated
   tables, for every TEDS variant.

2. ``grits_metric._lcs_similarity`` is now memoized. Memoization is a pure
   speedup, so the similarity — and therefore GriTS-Con — must be byte-for-byte
   identical to a fresh, un-memoized ``difflib`` computation.

If these ever diverge, the benchmark's published table scores would silently
change, so the tolerance is tight (1e-9).
"""

from __future__ import annotations

import random
from difflib import SequenceMatcher

import numpy as np
import pytest
from apted import APTED
from lxml import html

from extract_bench.evaluation.metrics.parse import fast_tree_edit
from extract_bench.evaluation.metrics.parse.grits_metric import (
    _lcs_similarity,
    cells_to_grid,
    factored_2dmss,
    html_to_cells,
)
from extract_bench.evaluation.metrics.parse.teds_metric import (
    ALL_TEDS_VARIANTS,
    TEDS,
    VARIANT_CONFIGS,
)

TOL = 1e-9

# --------------------------------------------------------------------------- #
# Crafted adversarial table pairs — the failure modes that stress tree edits.
# --------------------------------------------------------------------------- #
_GT = (
    "<table><tr><td>Item</td><td>Qty</td><td>Price</td></tr>"
    "<tr><td>Apple</td><td>3</td><td>$1.20</td></tr>"
    "<tr><td>Pear</td><td>5</td><td>$2.00</td></tr></table>"
)

CRAFTED_PAIRS = [
    # (name, gt_html, pred_html)
    ("perfect", _GT, _GT),
    (
        "single_typo",
        _GT,
        _GT.replace("Apple", "Apple"),
    ),
    (
        "swapped_columns",
        _GT,
        "<table><tr><td>Qty</td><td>Item</td><td>Price</td></tr>"
        "<tr><td>3</td><td>Apple</td><td>$1.20</td></tr>"
        "<tr><td>5</td><td>Pear</td><td>$2.00</td></tr></table>",
    ),
    (
        "missing_row",
        _GT,
        "<table><tr><td>Item</td><td>Qty</td><td>Price</td></tr>"
        "<tr><td>Apple</td><td>3</td><td>$1.20</td></tr></table>",
    ),
    (
        "hallucinated_row",
        _GT,
        _GT.replace("</table>", "<tr><td>Plum</td><td>9</td><td>$4.00</td></tr></table>"),
    ),
    (
        "bold_markup",
        _GT,
        _GT.replace("<td>Item</td>", "<td><b>Item</b></td>"),
    ),
    (
        "colspan_header",
        "<table><tr><td colspan='2'>Fruit</td><td>Price</td></tr>"
        "<tr><td>Apple</td><td>Red</td><td>$1.20</td></tr></table>",
        "<table><tr><td>Fruit</td><td>Kind</td><td>Price</td></tr>"
        "<tr><td>Apple</td><td>Red</td><td>$1.20</td></tr></table>",
    ),
    (
        "rowspan",
        "<table><tr><td rowspan='2'>A</td><td>1</td></tr><tr><td>2</td></tr></table>",
        "<table><tr><td>A</td><td>1</td></tr><tr><td>A</td><td>2</td></tr></table>",
    ),
    (
        "empty_cells",
        "<table><tr><td>A</td><td></td></tr><tr><td></td><td>B</td></tr></table>",
        "<table><tr><td>A</td><td>x</td></tr><tr><td></td><td>B</td></tr></table>",
    ),
    (
        "th_header",
        "<table><tr><th>Item</th><th>Qty</th></tr><tr><td>Apple</td><td>3</td></tr></table>",
        "<table><tr><td>Item</td><td>Qty</td></tr><tr><td>Apple</td><td>3</td></tr></table>",
    ),
]


def _table_element(table_html: str):
    """Parse an HTML string down to its <table> element, mirroring TEDS.evaluate."""
    parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
    doc = html.fromstring(table_html, parser=parser)
    if doc.tag == "table":
        return doc
    return doc.xpath(".//table")[0]


def _apted_distance(pred_html: str, gt_html: str, variant: str) -> float:
    teds = TEDS(variants={variant})
    tp = teds._load_html_tree(_table_element(pred_html))
    tt = teds._load_html_tree(_table_element(gt_html))
    return float(APTED(tp, tt, VARIANT_CONFIGS[variant]()).compute_edit_distance())


def _zss_distance(pred_html: str, gt_html: str, variant: str) -> float:
    teds = TEDS(variants={variant})
    tp = teds._load_html_tree(_table_element(pred_html))
    tt = teds._load_html_tree(_table_element(gt_html))
    return fast_tree_edit.tree_edit_distance(tp, tt, VARIANT_CONFIGS[variant]())


# --------------------------------------------------------------------------- #
# 1. Zhang-Shasha == APTED (the load-bearing exactness guarantee)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,gt,pred", CRAFTED_PAIRS, ids=[c[0] for c in CRAFTED_PAIRS])
@pytest.mark.parametrize("variant", sorted(ALL_TEDS_VARIANTS))
def test_zss_matches_apted_crafted(name: str, gt: str, pred: str, variant: str) -> None:
    zss = _zss_distance(pred, gt, variant)
    apted = _apted_distance(pred, gt, variant)
    assert zss == pytest.approx(apted, abs=TOL), f"{name}/{variant}: zss={zss} apted={apted}"


def _random_table_html(rng: random.Random) -> str:
    n_rows = rng.randint(1, 6)
    n_cols = rng.randint(1, 5)
    words = ["Item", "Qty", "Price", "Apple", "Pear", "3", "5", "$1.20", "", "Total", "N/A"]
    rows = []
    for _ in range(n_rows):
        cells = []
        for _ in range(n_cols):
            txt = rng.choice(words)
            # Occasional colspan to exercise span-aware rename costs.
            if rng.random() < 0.1:
                cells.append(f"<td colspan='2'>{txt}</td>")
            else:
                cells.append(f"<td>{txt}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


@pytest.mark.parametrize("variant", sorted(ALL_TEDS_VARIANTS))
def test_zss_matches_apted_random(variant: str) -> None:
    rng = random.Random(20240708)
    max_delta = 0.0
    for _ in range(200):
        gt = _random_table_html(rng)
        pred = _random_table_html(rng)
        zss = _zss_distance(pred, gt, variant)
        apted = _apted_distance(pred, gt, variant)
        max_delta = max(max_delta, abs(zss - apted))
    assert max_delta < TOL, f"variant={variant} max|delta|={max_delta}"


def test_teds_score_parity_fast_vs_apted(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: TEDS().evaluate scores are identical with the fast path on/off."""
    for _name, gt, pred in CRAFTED_PAIRS:
        teds = TEDS(variants=set(ALL_TEDS_VARIANTS))

        monkeypatch.setattr(fast_tree_edit, "FAST_TEDS_ENABLED", True)
        fast_scores, _, _ = teds.evaluate(pred, gt)

        monkeypatch.setattr(fast_tree_edit, "FAST_TEDS_ENABLED", False)
        apted_scores, _, _ = teds.evaluate(pred, gt)

        for variant in ALL_TEDS_VARIANTS:
            assert fast_scores[variant] == pytest.approx(apted_scores[variant], abs=TOL)


@pytest.mark.skipif(not fast_tree_edit._HAVE_NUMBA, reason="numba not installed")
def test_numba_and_python_kernels_agree() -> None:
    """The numba kernel and the pure-Python kernel must return the same value."""
    rng = random.Random(7)
    teds = TEDS(variants={"teds"})
    config = VARIANT_CONFIGS["teds"]()
    for _ in range(50):
        gt = _random_table_html(rng)
        pred = _random_table_html(rng)
        tp = teds._load_html_tree(_table_element(pred))
        tt = teds._load_html_tree(_table_element(gt))
        a = fast_tree_edit._build_postorder(tp)
        b = fast_tree_edit._build_postorder(tt)
        # Same (1-based) cost matrix feeds both kernels; _zss_dp_python indexes
        # it as c[i][j], which works for both numpy arrays and lists of lists.
        c = fast_tree_edit._cost_matrix(a, b, config)
        py = fast_tree_edit._zss_dp_python(a, b, c)

        l_a = np.array(a["l"], dtype=np.int64)
        l_b = np.array(b["l"], dtype=np.int64)
        kr_a = np.array(a["keyroots"], dtype=np.int64)
        kr_b = np.array(b["keyroots"], dtype=np.int64)
        nb = float(fast_tree_edit._zss_dp_numba(a["n"], b["n"], l_a, l_b, kr_a, kr_b, c))
        assert py == pytest.approx(nb, abs=TOL)


# --------------------------------------------------------------------------- #
# 2. GriTS LCS memoization is exact
# --------------------------------------------------------------------------- #
def _reference_lcs_similarity(string1: object, string2: object) -> float:
    """Fresh, un-memoized copy of the original ``_lcs_similarity`` body."""
    s1 = str(string1) if not isinstance(string1, str) else string1
    s2 = str(string2) if not isinstance(string2, str) else string2
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    s = SequenceMatcher(None, s1, s2)
    lcs = "".join([s1[block.a : (block.a + block.size)] for block in s.get_matching_blocks()])
    return 2 * len(lcs) / (len(s1) + len(s2))


def test_lcs_memo_matches_reference_on_random_strings() -> None:
    rng = random.Random(99)
    alphabet = "abc 0123$,.%"
    for _ in range(5000):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        assert _lcs_similarity(a, b) == pytest.approx(_reference_lcs_similarity(a, b), abs=TOL)
    # Non-string grid values (scalar 0 for unoccupied cells) must still work.
    assert _lcs_similarity(0, 0) == pytest.approx(_reference_lcs_similarity(0, 0), abs=TOL)
    assert _lcs_similarity(0, "x") == pytest.approx(_reference_lcs_similarity(0, "x"), abs=TOL)


def test_grits_con_parity_memoized_vs_reference() -> None:
    """GriTS-Con computed with the memoized primitive equals the reference."""
    for _name, gt, pred in CRAFTED_PAIRS:
        gt_cells = html_to_cells(gt)
        pred_cells = html_to_cells(pred)
        if not gt_cells or not pred_cells:
            continue
        true_grid = np.array(cells_to_grid(gt_cells), dtype=object)
        pred_grid = np.array(cells_to_grid(pred_cells), dtype=object)
        memoized = factored_2dmss(true_grid, pred_grid, _lcs_similarity)
        reference = factored_2dmss(true_grid, pred_grid, _reference_lcs_similarity)
        assert memoized == pytest.approx(reference, abs=TOL)
