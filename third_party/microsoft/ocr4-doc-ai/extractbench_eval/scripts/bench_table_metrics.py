"""Measure the previous vs. proposed table-metric implementations.

Reports, for a range of table sizes:
  * TEDS distance: APTED (previous) vs. Zhang-Shasha (proposed), with the
    max score delta across all TEDS variants (proves exactness) and the
    wall-clock speedup.
  * GriTS-Con: difflib un-memoized (previous) vs. memoized (proposed), with
    the score delta and speedup.

This is a diagnostic script, not a CI test — timings are machine-dependent, so
the exactness guarantee is enforced by ``tests/.../test_fast_table_parity.py``
instead. Run it to sanity-check the win on this machine:

    uv run python scripts/bench_table_metrics.py
"""

from __future__ import annotations

import time
from difflib import SequenceMatcher

import numpy as np
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


def _reference_lcs(string1: object, string2: object) -> float:
    s1 = str(string1) if not isinstance(string1, str) else string1
    s2 = str(string2) if not isinstance(string2, str) else string2
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    s = SequenceMatcher(None, s1, s2)
    lcs = "".join([s1[b.a : (b.a + b.size)] for b in s.get_matching_blocks()])
    return 2 * len(lcs) / (len(s1) + len(s2))


def _make_table(rows: int, cols: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    words = ["Item", "Qty", "Price", "Apple", "Pear", "Total", "N/A", "1,024", "$3.50", ""]
    trs = []
    for _ in range(rows):
        tds = "".join(f"<td>{words[rng.integers(0, len(words))]}</td>" for _ in range(cols))
        trs.append(f"<tr>{tds}</tr>")
    return "<table>" + "".join(trs) + "</table>"


def _table_element(table_html: str):
    parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
    doc = html.fromstring(table_html, parser=parser)
    return doc if doc.tag == "table" else doc.xpath(".//table")[0]


def _time(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0  # ms


def bench_teds(sizes: list[tuple[int, int]]) -> None:
    print(f"\nTEDS distance — APTED vs Zhang-Shasha (numba={fast_tree_edit._HAVE_NUMBA})")
    print(f"{'size':>8} | {'APTED ms':>10} | {'ZSS ms':>10} | {'speedup':>8} | {'max|Δscore|':>12}")
    print("-" * 62)
    fast_tree_edit.warmup()  # pay the numba compile once, outside the timing
    for rows, cols in sizes:
        gt = _make_table(rows, cols, seed=1)
        pred = _make_table(rows, cols, seed=2)  # independent → real edit distance
        teds = TEDS(variants=set(ALL_TEDS_VARIANTS))

        def run_apted() -> dict[str, float]:
            tp, tt = teds._load_html_tree(_table_element(pred)), teds._load_html_tree(_table_element(gt))
            n = max(len(_table_element(pred).xpath(".//*")) + 1, len(_table_element(gt).xpath(".//*")) + 1)
            return {
                v: max(0.0, 1.0 - APTED(tp, tt, VARIANT_CONFIGS[v]()).compute_edit_distance() / n)
                for v in ALL_TEDS_VARIANTS
            }

        def run_zss() -> dict[str, float]:
            tp, tt = teds._load_html_tree(_table_element(pred)), teds._load_html_tree(_table_element(gt))
            n = max(len(_table_element(pred).xpath(".//*")) + 1, len(_table_element(gt).xpath(".//*")) + 1)
            return {
                v: max(0.0, 1.0 - fast_tree_edit.tree_edit_distance(tp, tt, VARIANT_CONFIGS[v]()) / n)
                for v in ALL_TEDS_VARIANTS
            }

        a_scores, z_scores = run_apted(), run_zss()
        max_delta = max(abs(a_scores[v] - z_scores[v]) for v in ALL_TEDS_VARIANTS)
        a_ms, z_ms = _time(run_apted), _time(run_zss)
        print(f"{rows}x{cols:>5} | {a_ms:10.2f} | {z_ms:10.2f} | {a_ms / z_ms:7.1f}x | {max_delta:12.2e}")


def bench_grits(sizes: list[tuple[int, int]]) -> None:
    print("\nGriTS-Con — difflib (un-memoized) vs memoized")
    print(f"{'size':>8} | {'ref ms':>10} | {'memo ms':>10} | {'speedup':>8} | {'|Δscore|':>12}")
    print("-" * 62)
    for rows, cols in sizes:
        gt = _make_table(rows, cols, seed=1)
        pred = _make_table(rows, cols, seed=2)
        true_grid = np.array(cells_to_grid(html_to_cells(gt)), dtype=object)
        pred_grid = np.array(cells_to_grid(html_to_cells(pred)), dtype=object)

        def run_ref() -> float:
            return factored_2dmss(true_grid, pred_grid, _reference_lcs)[0]

        def run_memo() -> float:
            return factored_2dmss(true_grid, pred_grid, _lcs_similarity)[0]

        ref_score, memo_score = run_ref(), run_memo()
        r_ms, m_ms = _time(run_ref), _time(run_memo)
        delta = abs(ref_score - memo_score)
        print(f"{rows}x{cols:>5} | {r_ms:10.2f} | {m_ms:10.2f} | {r_ms / m_ms:7.1f}x | {delta:12.2e}")


if __name__ == "__main__":
    sizes = [(8, 5), (15, 8), (25, 10), (40, 12)]
    bench_teds(sizes)
    bench_grits(sizes)
