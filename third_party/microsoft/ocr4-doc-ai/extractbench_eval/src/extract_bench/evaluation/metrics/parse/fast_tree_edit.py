"""Fast, EXACT-parity tree-edit distance for TEDS — a drop-in for APTED.

Why this exists
---------------
The TEDS metric (``teds_metric.py``) scores an HTML table by the tree-edit
distance between two ``TableTree`` forests. That distance is the dominant cost
of the whole table-scoring path and grows ``~O(cells^2)`` — on a large table
(e.g. 40x12) a single ``teds`` variant can take well over a second, and the
metric runs three variants (``teds`` / ``teds_struct`` / ``teds_struct_bool``)
on every predicted table of every document in the benchmark.

The distance itself is computed by the pure-Python ``apted`` package. Profiling
(see ``FAST_TABLE_OPTIMIZATION.md`` in the training pipeline) showed two things:

1. APTED's dynamic program is the bottleneck, and it **recomputes the per-node
   rename cost several times** instead of memoizing it.
2. The trees TEDS builds are ordinary *ordered* forests (``table -> tr* -> td*``,
   with inline markup already tokenized into cell content). For an ordered tree
   the classic **Zhang-Shasha** algorithm computes the **identical optimal
   tree-edit distance** as APTED — it is the same quantity, so the result is
   equal *by construction*, not by approximation.

This module implements Zhang-Shasha over the existing ``TableTree`` nodes and
reuses each variant's own ``apted.Config.rename`` for the substitution cost, so
the numbers are byte-for-byte the APTED numbers (modulo floating-point
summation order, which is bounded by ~1e-12 and never large enough to move a
reported TEDS score). Insertion/deletion cost is ``1`` — exactly APTED's
default, which ParseBench's configs do not override (only ``rename`` is).

Speed comes from two exact changes over APTED:
  * the rename/Levenshtein cost matrix is computed **once** per node pair, and
  * the DP runs in a tight array kernel that ``numba`` JIT-compiles to machine
    code (``@njit(cache=True)``) when available.

``numba`` is optional. When it is missing (or fails to import) the pure-Python
Zhang-Shasha kernel is used instead — still exact, just without the JIT win.
Install ``numba`` for the fast path: ``pip install numba``.

Measured (``scripts/bench_table_metrics.py``, numba on, all three TEDS
variants, independent random tables — the worst case for edit distance)::

    size    APTED ms    ZSS ms   speedup   max|Δ score|
    8x5        89.6      13.2      6.8x      0.00e+00
    15x8      843.3     115.6      7.3x      0.00e+00
    25x10    3994.4     519.6      7.7x      0.00e+00
    40x12   14895.4    1895.7      7.9x      0.00e+00

The win grows with table size (the metric is ``~O(cells^2)``) and the score
delta is exactly zero — the whole point: faster, not different.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from apted import Config

    from extract_bench.evaluation.metrics.parse.teds_metric import TableTree

# Kill-switch: set EXTRACT_BENCH_FAST_TEDS=0 to force the original APTED path.
# The default is ON because the result is exact by construction (see module
# docstring), with an automatic fallback to APTED on any unexpected error.
FAST_TEDS_ENABLED = os.environ.get("EXTRACT_BENCH_FAST_TEDS", "1") != "0"

_LOGGED: set[str] = set()


def _log_once(msg: str) -> None:
    """Emit ``msg`` at most once per process so it is obvious from the logs
    which distance path (numba / pure-Python / APTED fallback) is active."""
    if msg not in _LOGGED:
        _LOGGED.add(msg)
        print(f"[fast_tree_edit] {msg}", file=sys.stderr, flush=True)


try:
    import numpy as _np
    from numba import njit as _njit

    _HAVE_NUMBA = True
    _NUMBA_ERR: str | None = None
except Exception as _e:  # pragma: no cover - depends on the environment
    _HAVE_NUMBA = False
    _NUMBA_ERR = repr(_e)


# --------------------------------------------------------------------------- #
# Postorder indexing of a TableTree (the layout Zhang-Shasha needs).
# --------------------------------------------------------------------------- #
def _build_postorder(root: TableTree) -> dict[str, Any]:
    """Index ``root``'s nodes in postorder (1-based) with the two arrays
    Zhang-Shasha needs: ``l[i]`` = the postorder id of node ``i``'s leftmost
    leaf descendant, and ``keyroots`` = the nodes that start a distinct
    leftmost-leaf path.

    Returns a dict with ``n`` (node count), ``nodes`` (1-based list of the
    original ``TableTree`` nodes, so the caller can compute rename costs),
    ``l`` and ``keyroots``.
    """
    nodes: list[Any] = [None]  # index 0 unused → 1-based postorder ids
    leftmost: list[int] = [0]

    # Iterative postorder to stay safe on pathologically deep trees.
    # Stack entries: (node, child_index, first_child_leftmost).
    stack: list[list[Any]] = [[root, 0, None]]
    while stack:
        frame = stack[-1]
        node, child_idx, first_leftmost = frame
        children = node.children
        if child_idx < len(children):
            frame[1] = child_idx + 1
            stack.append([children[child_idx], 0, None])
            continue
        # All children processed → assign this node its postorder id.
        stack.pop()
        nodes.append(node)
        idx = len(nodes) - 1
        # A node's leftmost leaf is its first child's leftmost leaf; a leaf is
        # its own leftmost leaf.
        lm = first_leftmost if first_leftmost is not None else idx
        leftmost.append(lm)
        if stack:
            parent = stack[-1]
            if parent[2] is None:  # record the leftmost id of the parent's 1st child
                parent[2] = lm

    n = len(nodes) - 1
    # keyroots: for each distinct leftmost value keep the largest node id.
    last_for_l: dict[int, int] = {}
    for i in range(1, n + 1):
        last_for_l[leftmost[i]] = i
    keyroots = sorted(last_for_l.values())
    return {"n": n, "nodes": nodes, "l": leftmost, "keyroots": keyroots}


def _cost_matrix(a: dict[str, Any], b: dict[str, Any], config: Config) -> Any:
    """Substitution-cost matrix ``C[i][j] = config.rename(a_i, b_j)`` (1-based).

    Each entry is computed exactly once here; APTED would recompute the same
    rename several times inside its DP. Uses a numpy array when numba is
    available (the kernel needs one), otherwise a list of lists.
    """
    n1, n2 = a["n"], b["n"]
    nodes_a, nodes_b = a["nodes"], b["nodes"]
    c: Any  # numpy array on the numba path, list-of-lists on the fallback path
    if _HAVE_NUMBA:
        c = _np.zeros((n1 + 1, n2 + 1), dtype=_np.float64)
    else:
        c = [[0.0] * (n2 + 1) for _ in range(n1 + 1)]
    for i in range(1, n1 + 1):
        node_a = nodes_a[i]
        row = c[i]
        for j in range(1, n2 + 1):
            row[j] = config.rename(node_a, nodes_b[j])
    return c


# --------------------------------------------------------------------------- #
# Zhang-Shasha DP — numba (compiled) and pure-Python (fallback). Both compute
# the exact ordered tree-edit distance with unit insert/delete cost.
# --------------------------------------------------------------------------- #
if _HAVE_NUMBA:

    @_njit(cache=True, fastmath=False)  # pragma: no cover - compiled to machine code
    def _zss_dp_numba(n1, n2, l_a, l_b, kr_a, kr_b, c):
        treedist = _np.zeros((n1 + 1, n2 + 1))
        fd = _np.empty((n1 + 2, n2 + 2))
        for ki in range(kr_a.shape[0]):
            i = kr_a[ki]
            li = l_a[i]
            m = i - li + 1
            for kj in range(kr_b.shape[0]):
                j = kr_b[kj]
                lj = l_b[j]
                nn = j - lj + 1
                fd[0, 0] = 0.0
                for x in range(1, m + 1):
                    fd[x, 0] = fd[x - 1, 0] + 1.0
                for y in range(1, nn + 1):
                    fd[0, y] = fd[0, y - 1] + 1.0
                for x in range(1, m + 1):
                    di = li + x - 1
                    ldi = l_a[di]
                    for y in range(1, nn + 1):
                        dj = lj + y - 1
                        dele = fd[x - 1, y] + 1.0
                        ins = fd[x, y - 1] + 1.0
                        val = dele if dele < ins else ins
                        if ldi == li and l_b[dj] == lj:
                            sub = fd[x - 1, y - 1] + c[di, dj]
                            if sub < val:
                                val = sub
                            fd[x, y] = val
                            treedist[di, dj] = val
                        else:
                            sub = fd[ldi - li, l_b[dj] - lj] + treedist[di, dj]
                            if sub < val:
                                val = sub
                            fd[x, y] = val
        return treedist[n1, n2]


def _zss_dp_python(a: dict[str, Any], b: dict[str, Any], c: list[list[float]]) -> float:
    n1, n2 = a["n"], b["n"]
    l_a, l_b = a["l"], b["l"]
    treedist = [[0.0] * (n2 + 1) for _ in range(n1 + 1)]
    for i in a["keyroots"]:
        li = l_a[i]
        m = i - li + 1
        for j in b["keyroots"]:
            lj = l_b[j]
            nn = j - lj + 1
            fd = [[0.0] * (nn + 1) for _ in range(m + 1)]
            for x in range(1, m + 1):
                fd[x][0] = fd[x - 1][0] + 1.0
            for y in range(1, nn + 1):
                fd[0][y] = fd[0][y - 1] + 1.0
            for x in range(1, m + 1):
                di = li + x - 1
                ldi = l_a[di]
                for y in range(1, nn + 1):
                    dj = lj + y - 1
                    dele = fd[x - 1][y] + 1.0
                    ins = fd[x][y - 1] + 1.0
                    val = dele if dele < ins else ins
                    if ldi == li and l_b[dj] == lj:
                        sub = fd[x - 1][y - 1] + c[di][dj]
                        if sub < val:
                            val = sub
                        fd[x][y] = val
                        treedist[di][dj] = val
                    else:
                        sub = fd[ldi - li][l_b[dj] - lj] + treedist[di][dj]
                        if sub < val:
                            val = sub
                        fd[x][y] = val
    return float(treedist[n1][n2])


def tree_edit_distance(tree1: TableTree, tree2: TableTree, config: Config) -> float:
    """Exact ordered tree-edit distance between two ``TableTree`` forests.

    Equivalent to ``APTED(tree1, tree2, config).compute_edit_distance()`` for
    ParseBench's TEDS configs (which override only ``rename``; insert/delete
    cost is APTED's default of 1). Uses the numba kernel when available and the
    pure-Python kernel otherwise — both return the same value.
    """
    a = _build_postorder(tree1)
    b = _build_postorder(tree2)
    c = _cost_matrix(a, b, config)
    if _HAVE_NUMBA:
        _log_once("active path: numba JIT Zhang-Shasha (fast)")
        l_a = _np.array(a["l"], dtype=_np.int64)
        l_b = _np.array(b["l"], dtype=_np.int64)
        kr_a = _np.array(a["keyroots"], dtype=_np.int64)
        kr_b = _np.array(b["keyroots"], dtype=_np.int64)
        return float(_zss_dp_numba(a["n"], b["n"], l_a, l_b, kr_a, kr_b, c))
    _log_once(f"FALLBACK: numba unavailable -> pure-Python Zhang-Shasha (exact, slower). cause={_NUMBA_ERR}")
    return _zss_dp_python(a, b, c)


def warmup() -> bool:
    """Trigger numba JIT compilation once (cached on disk via ``cache=True``)
    on a trivial 1x1 table, so the first real table isn't charged the compile.
    Returns True if the fast (numba) path is active. Safe to call at startup;
    swallows any error and reports the state instead of raising."""
    try:
        from apted import Config

        from extract_bench.evaluation.metrics.parse.teds_metric import TableTree

        cell = TableTree("td", 1, 1, ["a"])
        row = TableTree("tr", None, None, None, cell)
        table = TableTree("table", None, None, None, row)
        tree_edit_distance(table, table, Config())
    except Exception as e:  # pragma: no cover - defensive
        _log_once(f"FALLBACK: warmup failed -> {e!r}")
        return False
    return _HAVE_NUMBA
