"""Header accuracy metric for HTML table comparison.

Evaluates how accurately table headers are reproduced by comparing
predicted HTML tables against ground-truth HTML tables. Produces a
composite score from eight submetrics (``header_perfect`` is still
emitted but excluded from the composite):

1. **header_cell_count**: Ratio of predicted header cell count to expected.
   Penalises both missing and extra header cells symmetrically.
2. **header_grits**: GriTS-Con applied to contiguous header blocks,
   measuring how well the mapping between header regions is preserved.
   Extra predicted blocks are penalised.
3. **header_content_bag**: Bag-of-cells exact content overlap — counts how
   many expected header texts appear (exact match after formatting
   normalization) in the prediction.
4. **header_perfect**: Binary — 1.0 iff the header structure
   (cell texts, positions, colspan/rowspan) matches exactly.
   *Emitted but not included in the composite.*
5. **header_block_extent**: Measures how well each header block's location
   and size within the full table matches what is expected.
6. **header_block_proximity**: Averaged nearest-edge distance similarity
   between matched header block pairs. Extra predicted blocks are
   penalised via ``max(gt_pairs, pred_pairs)`` denominator.
7. **header_block_relative_direction**: Averaged cosine similarity of
   signed edge vectors between matched header block pairs, mapped to
   [0, 1]. Extra predicted blocks are penalised via
   ``max(gt_pairs, pred_pairs)`` denominator.
8. **multilevel_header_depth**: Compares the depth of the header hierarchy
   tree (number of nesting levels) between GT and prediction using
   ``min(depth_gt, depth_pred) / max(depth_gt, depth_pred)``.
9. **header_data_alignment**: Uses GriTS row/col alignment maps to check
    whether each GT header cell's text appears at the corresponding
    position in the prediction grid. When GriTS alignment is not
    available, computes its own alignment as a fallback.

The overall ``header_composite_v3`` score is the mean of *applicable*
submetrics from 1–3 and 5–9 (i.e. excluding ``header_perfect``).
Submetrics that are trivially 1.0 because both GT and prediction have
the same degenerate block count (e.g. both have ≤1 block, so proximity
and direction metrics are vacuous) are excluded from the composite
denominator.  When the counts *differ* (e.g. GT has 0 blocks but
prediction has 2), the submetric is kept so the mismatch is still
penalised.

Table-level matching is delegated to the caller (typically from GriTS
matching) via the ``table_pairs`` parameter. Block-level matching is
computed once (via GriTS-Con Hungarian) and shared across all block
submetrics for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from bs4 import BeautifulSoup
from scipy.optimize import linear_sum_assignment

from extract_bench.evaluation.metrics.base import Metric
from extract_bench.evaluation.metrics.parse.grits_metric import (
    _lcs_similarity,
    factored_2dmss,
)
from extract_bench.evaluation.metrics.parse.table_extraction import extract_html_tables
from extract_bench.evaluation.metrics.parse.table_parsing import _sup_sub_to_unicode
from extract_bench.evaluation.metrics.parse.utils import normalize_cell_text, normalize_text
from extract_bench.schemas.evaluation import MetricValue


def _normalize_header_text(raw: str) -> str:
    """Normalize cell text for header comparison.

    Applies cell-level normalization (formatting stripping, dash/dot
    normalization) via ``normalize_cell_text``, then the full
    ``normalize_text`` pipeline (lowercasing, accent removal, etc.).
    """
    text = normalize_cell_text(raw)
    return normalize_text(text)


# ---------------------------------------------------------------------------
# Header block extraction
# ---------------------------------------------------------------------------


@dataclass
class HeaderCell:
    """A single header cell with its grid position and span info."""

    text: str
    row: int
    col: int
    rowspan: int
    colspan: int


@dataclass
class HeaderBlock:
    """A contiguous rectangular region of header cells."""

    cells: list[HeaderCell] = field(default_factory=list)
    min_row: int = 0
    max_row: int = 0  # exclusive
    min_col: int = 0
    max_col: int = 0  # exclusive

    def extent(self, table_rows: int, table_cols: int) -> tuple[float, float, float, float]:
        """Normalised (row_start, col_start, row_end, col_end) in [0, 1]."""
        if table_rows == 0 or table_cols == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            self.min_row / table_rows,
            self.min_col / table_cols,
            self.max_row / table_rows,
            self.max_col / table_cols,
        )

    def center(self, table_rows: int, table_cols: int) -> tuple[float, float]:
        """Normalised center (row, col) of this block."""
        if table_rows == 0 or table_cols == 0:
            return (0.0, 0.0)
        return (
            (self.min_row + self.max_row) / 2.0 / table_rows,
            (self.min_col + self.max_col) / 2.0 / table_cols,
        )


def _parse_header_cells(table_html: str) -> tuple[list[HeaderCell], int, int]:
    """Extract header cells from an HTML table string.

    Returns (header_cells, num_rows, num_cols).
    """
    # Use lxml for robustness with malformed HTML (e.g. <th>...</td> mismatches)
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table")
    if not table:
        return [], 0, 0

    rows = table.find_all("tr")
    if not rows:
        return [], 0, 0

    # Build occupied grid to resolve spans
    occupied: dict[tuple[int, int], bool] = {}
    header_cells: list[HeaderCell] = []
    max_col = 0

    # Identify thead rows
    thead = table.find("thead")
    thead_row_indices: set[int] = set()
    if thead:
        for tr in thead.find_all("tr"):
            if tr in rows:
                thead_row_indices.add(rows.index(tr))

    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row.find_all(["td", "th"]):
            while (row_idx, col_idx) in occupied:
                col_idx += 1

            rowspan = int(str(cell.get("rowspan", "1")))
            colspan = int(str(cell.get("colspan", "1")))
            is_header = cell.name == "th" or row_idx in thead_row_indices

            # Mark occupied
            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    occupied[(r, c)] = True

            if is_header:
                # Convert <sup>/<sub> digit content to Unicode equivalents
                # so "Name<sup>1</sup>" becomes "Name¹", matching sources
                # that already use Unicode superscripts.
                _sup_sub_to_unicode(cell)
                text = _normalize_header_text(cell.get_text(strip=True))
                header_cells.append(
                    HeaderCell(
                        text=text,
                        row=row_idx,
                        col=col_idx,
                        rowspan=rowspan,
                        colspan=colspan,
                    )
                )

            max_col = max(max_col, col_idx + colspan)
            col_idx += colspan

    num_rows = len(rows)
    num_cols = max_col if max_col > 0 else 0
    return header_cells, num_rows, num_cols


def _find_header_blocks(cells: list[HeaderCell]) -> list[HeaderBlock]:
    """Group header cells into contiguous rectangular blocks.

    Two header cells belong to the same block if they are adjacent
    (horizontally or vertically, using 4-connected adjacency).
    """
    if not cells:
        return []

    # Build adjacency via grid positions occupied by each cell
    cell_positions: dict[int, set[tuple[int, int]]] = {}
    for idx, cell in enumerate(cells):
        positions = set()
        for r in range(cell.row, cell.row + cell.rowspan):
            for c in range(cell.col, cell.col + cell.colspan):
                positions.add((r, c))
        cell_positions[idx] = positions

    # Union-Find
    parent = list(range(len(cells)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Two cells are in the same block if any of their occupied positions
    # are adjacent (4-connected: up/down/left/right only).
    all_pos_to_idx: dict[tuple[int, int], int] = {}
    for idx, positions in cell_positions.items():
        for pos in positions:
            all_pos_to_idx[pos] = idx

    for idx, positions in cell_positions.items():
        for r, c in positions:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (r + dr, c + dc)
                if neighbor in all_pos_to_idx:
                    union(idx, all_pos_to_idx[neighbor])

    # Group by root
    groups: dict[int, list[int]] = {}
    for idx in range(len(cells)):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    blocks: list[HeaderBlock] = []
    for indices in groups.values():
        block_cells = [cells[i] for i in indices]
        min_row = min(c.row for c in block_cells)
        max_row = max(c.row + c.rowspan for c in block_cells)
        min_col = min(c.col for c in block_cells)
        max_col = max(c.col + c.colspan for c in block_cells)
        blocks.append(
            HeaderBlock(
                cells=block_cells,
                min_row=min_row,
                max_row=max_row,
                min_col=min_col,
                max_col=max_col,
            )
        )

    # Sort blocks by (min_row, min_col) for stable ordering
    blocks.sort(key=lambda b: (b.min_row, b.min_col))
    return blocks


# ---------------------------------------------------------------------------
# Block matching (shared across all block submetrics)
# ---------------------------------------------------------------------------


def _block_to_text_grid(block: HeaderBlock) -> np.ndarray:
    """Build a text grid for a header block (for GriTS-Con)."""
    rows = block.max_row - block.min_row
    cols = block.max_col - block.min_col
    if rows <= 0 or cols <= 0:
        return np.array([[""]], dtype=object)
    grid = np.full((rows, cols), "", dtype=object)
    for cell in block.cells:
        for r in range(cell.row, cell.row + cell.rowspan):
            for c in range(cell.col, cell.col + cell.colspan):
                grid[r - block.min_row, c - block.min_col] = cell.text
    return grid


def _block_extent_iou(
    gt_b: HeaderBlock,
    pred_b: HeaderBlock,
    gt_rows: int,
    gt_cols: int,
    pred_rows: int,
    pred_cols: int,
) -> float:
    """IoU between normalised extents of two blocks in their respective tables."""
    e1 = gt_b.extent(gt_rows, gt_cols)
    e2 = pred_b.extent(pred_rows, pred_cols)
    inter_r1, inter_c1 = max(e1[0], e2[0]), max(e1[1], e2[1])
    inter_r2, inter_c2 = min(e1[2], e2[2]), min(e1[3], e2[3])
    inter = max(0.0, inter_r2 - inter_r1) * max(0.0, inter_c2 - inter_c1)
    area1 = (e1[2] - e1[0]) * (e1[3] - e1[1])
    area2 = (e2[2] - e2[0]) * (e2[3] - e2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# Small weight for extent IoU tiebreaker — enough to break ties in GriTS-Con
# but not enough to override a genuine content difference.
_EXTENT_TIEBREAK_WEIGHT = 1e-4


def _match_blocks(
    gt_blocks: list[HeaderBlock],
    pred_blocks: list[HeaderBlock],
    gt_rows: int = 0,
    gt_cols: int = 0,
    pred_rows: int = 0,
    pred_cols: int = 0,
) -> tuple[dict[int, int], dict[tuple[int, int], float]]:
    """Match GT blocks to pred blocks via GriTS-Con Hungarian matching.

    When GriTS-Con scores tie, extent IoU (positional overlap) is used
    as a tiebreaker so that blocks in similar positions are preferred.

    Returns:
        gt_to_pred: mapping from GT block index to pred block index
        grits_scores: dict of (gt_idx, pred_idx) -> GriTS-Con f-score
            for all pairs considered during matching
    """
    if not gt_blocks or not pred_blocks:
        return {}, {}

    n_gt = len(gt_blocks)
    n_pred = len(pred_blocks)

    # Infer table dimensions from block extents if not provided
    if gt_rows == 0:
        gt_rows = max(b.max_row for b in gt_blocks)
    if gt_cols == 0:
        gt_cols = max(b.max_col for b in gt_blocks)
    if pred_rows == 0:
        pred_rows = max(b.max_row for b in pred_blocks)
    if pred_cols == 0:
        pred_cols = max(b.max_col for b in pred_blocks)

    cost = np.zeros((n_gt, n_pred))
    grits_scores: dict[tuple[int, int], float] = {}
    for i, gt_b in enumerate(gt_blocks):
        gt_grid = _block_to_text_grid(gt_b)
        for j, pred_b in enumerate(pred_blocks):
            pred_grid = _block_to_text_grid(pred_b)
            fscore, _, _, _ = factored_2dmss(gt_grid, pred_grid, _lcs_similarity)
            grits_scores[(i, j)] = fscore
            # Extent IoU tiebreaker: when GriTS-Con ties, prefer positional match
            iou = _block_extent_iou(
                gt_b,
                pred_b,
                gt_rows,
                gt_cols,
                pred_rows,
                pred_cols,
            )
            cost[i, j] = -(fscore + _EXTENT_TIEBREAK_WEIGHT * iou)

    row_ind, col_ind = linear_sum_assignment(cost)

    gt_to_pred: dict[int, int] = {}
    for r, c in zip(row_ind, col_ind, strict=True):
        gt_to_pred[int(r)] = int(c)

    return gt_to_pred, grits_scores


# ---------------------------------------------------------------------------
# Submetric implementations
# ---------------------------------------------------------------------------


_COMPOSITE_KEYS = [
    "header_cell_count",
    "header_grits",
    "header_content_bag",
    "header_block_extent",
    "header_block_proximity",
    "header_block_relative_direction",
    "multilevel_header_depth",
    "header_data_alignment",
]


def _build_text_lookup(table_html: str) -> dict[tuple[int, int], str]:
    """Build a (row, col) -> normalized_text lookup for ALL cells in a table.

    Unlike _parse_header_cells which only returns <th>/thead cells, this
    returns text for every cell (th and td) so we can check what text
    appears at any grid position.
    """
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table")
    if not table:
        return {}

    rows = table.find_all("tr")
    if not rows:
        return {}

    occupied: dict[tuple[int, int], bool] = {}
    lookup: dict[tuple[int, int], str] = {}

    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row.find_all(["td", "th"]):
            while (row_idx, col_idx) in occupied:
                col_idx += 1
            rowspan = int(str(cell.get("rowspan", "1")))
            colspan = int(str(cell.get("colspan", "1")))
            text = _normalize_header_text(cell.get_text(strip=True))
            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    occupied[(r, c)] = True
                    lookup[(r, c)] = text
            col_idx += colspan

    return lookup


_GENEROUS_LCS_THRESHOLD = 0.8


def _is_bottom_left_block(block: HeaderBlock, num_rows: int) -> bool:
    """Check if a header block is a rectangle in the bottom-left of the table.

    A block is bottom-left if:
    - It touches the leftmost column (min_col == 0)
    - It has at least one cell in the bottom row (max_row == num_rows)
    - It does NOT extend to the top row (min_row > 0)
    - It is a perfect rectangle (no holes or irregular shape)
    """
    if not (block.min_col == 0 and block.max_row == num_rows and block.min_row > 0):
        return False
    # Check rectangularity: count occupied grid positions across all cells
    occupied: set[tuple[int, int]] = set()
    for cell in block.cells:
        for r in range(cell.row, cell.row + cell.rowspan):
            for c in range(cell.col, cell.col + cell.colspan):
                occupied.add((r, c))
    expected_area = (block.max_row - block.min_row) * (block.max_col - block.min_col)
    return len(occupied) == expected_area


def _find_contiguous_groups(
    positions: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    """Group positions into contiguous sets using 4-connected adjacency."""
    if not positions:
        return []

    parent: dict[tuple[int, int], tuple[int, int]] = {p: p for p in positions}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r, c in positions:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (r + dr, c + dc)
            if neighbor in positions:
                union((r, c), neighbor)

    groups: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for p in positions:
        root = find(p)
        groups.setdefault(root, set()).add(p)

    return list(groups.values())


def _promote_cells_at_positions(pred_html: str, positions: set[tuple[int, int]]) -> str:
    """Promote <td> cells at specific grid positions to <th>."""
    soup = BeautifulSoup(pred_html, "lxml")
    table = soup.find("table")
    if not table:
        return pred_html

    rows = table.find_all("tr")
    occupied: dict[tuple[int, int], bool] = {}

    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row.find_all(["td", "th"]):
            while (row_idx, col_idx) in occupied:
                col_idx += 1
            rowspan = int(str(cell.get("rowspan", "1")))
            colspan = int(str(cell.get("colspan", "1")))
            # Check if any of this cell's positions should be promoted
            should_promote = False
            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    occupied[(r, c)] = True
                    if (r, c) in positions:
                        should_promote = True
            if should_promote and cell.name == "td":
                cell.name = "th"
            col_idx += colspan

    return str(table)


def _promote_bottom_left_to_header(gt_html: str, pred_html: str, threshold: float = _GENEROUS_LCS_THRESHOLD) -> str:
    """Promote pred cells to <th> where GT has a bottom-left header block.

    For each bottom-left header block in the GT, check if the corresponding
    cells in the pred table have similar text. Promote matching pred cells
    that form a contiguous block to <th>.
    """
    gt_cells, gt_num_rows, gt_num_cols = _parse_header_cells(gt_html)
    if not gt_cells or gt_num_rows == 0:
        return pred_html

    gt_blocks = _find_header_blocks(gt_cells)
    bl_blocks = [b for b in gt_blocks if _is_bottom_left_block(b, gt_num_rows)]
    if not bl_blocks:
        return pred_html

    pred_text_lookup = _build_text_lookup(pred_html)
    if not pred_text_lookup:
        return pred_html

    # Check if pred already has headers in the bottom-left region
    pred_cells, pred_num_rows, _ = _parse_header_cells(pred_html)
    pred_blocks = _find_header_blocks(pred_cells)
    pred_bl_blocks = [b for b in pred_blocks if _is_bottom_left_block(b, pred_num_rows)]
    if pred_bl_blocks:
        return pred_html  # pred already has bottom-left headers

    # Collect positions to promote
    positions_to_promote: set[tuple[int, int]] = set()
    for block in bl_blocks:
        matching_positions: set[tuple[int, int]] = set()
        for cell in block.cells:
            gt_text = cell.text
            for r in range(cell.row, cell.row + cell.rowspan):
                for c in range(cell.col, cell.col + cell.colspan):
                    pred_text = pred_text_lookup.get((r, c), "")
                    if _lcs_similarity(gt_text, pred_text) >= threshold:
                        matching_positions.add((r, c))

        # Check contiguity: find the contiguous group that contains
        # the bottom-left cell of the *pred* table (not the GT table,
        # since the pred may be truncated).
        if matching_positions:
            contiguous_groups = _find_contiguous_groups(matching_positions)
            pred_bottom_left = (pred_num_rows - 1, 0)
            for group in contiguous_groups:
                if pred_bottom_left in group:
                    positions_to_promote.update(group)
                    break

    if not positions_to_promote:
        return pred_html

    # Promote the cells at those positions
    return _promote_cells_at_positions(pred_html, positions_to_promote)


def _header_data_alignment_score(
    gt_cells: list[HeaderCell],
    pred_text_lookup: dict[tuple[int, int], str],
    row_map: dict[int, int],
    col_map: dict[int, int],
) -> float:
    """Submetric 10: header-data alignment via GriTS grid mapping.

    For each GT header cell at anchor (row, col) with normalized text T,
    maps to the prediction grid via (row_map[row], col_map[col]) and
    checks whether the text at that position matches T.

    Returns fraction of GT header cells whose text matches at the
    aligned position. Returns 1.0 when GT has no headers.
    """
    if not gt_cells:
        return 1.0
    if not row_map or not col_map:
        return 0.0

    hits = 0
    for gc in gt_cells:
        mapped_r = row_map.get(gc.row)
        mapped_c = col_map.get(gc.col)
        if mapped_r is not None and mapped_c is not None:
            pred_text = pred_text_lookup.get((mapped_r, mapped_c), "")
            if gc.text == pred_text:
                hits += 1

    return hits / len(gt_cells)


def _header_data_alignment_score_fallback(
    gt_html: str,
    pred_html: str,
    gt_cells: list[HeaderCell],
    pred_text_lookup: dict[tuple[int, int], str],
) -> float:
    """Compute header_data_alignment without pre-computed GriTS alignment.

    Builds text grids from the HTML and runs _align_2d_outer to get
    row/col mappings, then delegates to _header_data_alignment_score.
    """
    from extract_bench.evaluation.metrics.parse.grits_metric import (
        _align_2d_outer,
        _lcs_similarity,
        cells_to_grid,
        html_to_cells,
    )

    if not gt_cells:
        return 1.0

    true_cells = html_to_cells(gt_html)
    pred_cells_parsed = html_to_cells(pred_html)
    if not true_cells or not pred_cells_parsed:
        return 0.0

    true_text = np.array(cells_to_grid(true_cells, key="cell_text"), dtype=object)
    pred_text = np.array(cells_to_grid(pred_cells_parsed, key="cell_text"), dtype=object)

    # Compute reward lookup (same as factored_2dmss)
    pre_computed: dict[tuple[int, int, int, int], float] = {}
    transpose_rewards: dict[tuple[int, int, int, int], float] = {}
    for trow in range(true_text.shape[0]):
        for tcol in range(true_text.shape[1]):
            for prow in range(pred_text.shape[0]):
                for pcol in range(pred_text.shape[1]):
                    reward = _lcs_similarity(true_text[trow, tcol], pred_text[prow, pcol])
                    pre_computed[(trow, tcol, prow, pcol)] = reward
                    transpose_rewards[(tcol, trow, pcol, prow)] = reward

    true_row_nums, pred_row_nums, _ = _align_2d_outer(true_text.shape[:2], pred_text.shape[:2], pre_computed)
    true_col_nums, pred_col_nums, _ = _align_2d_outer(
        true_text.shape[:2][::-1], pred_text.shape[:2][::-1], transpose_rewards
    )

    row_map = dict(zip(true_row_nums, pred_row_nums, strict=True))
    col_map = dict(zip(true_col_nums, pred_col_nums, strict=True))

    return _header_data_alignment_score(gt_cells, pred_text_lookup, row_map, col_map)


def _header_cell_count_score(
    gt_cells: list[HeaderCell],
    pred_cells: list[HeaderCell],
) -> float:
    """Submetric 1: ratio-based cell count similarity.

    Returns min(gt_count, pred_count) / max(gt_count, pred_count).
    If both are 0, returns 1.0 (both agree there are no headers).
    """
    gt_n = len(gt_cells)
    pred_n = len(pred_cells)
    if gt_n == 0 and pred_n == 0:
        return 1.0
    if gt_n == 0 or pred_n == 0:
        return 0.0
    return min(gt_n, pred_n) / max(gt_n, pred_n)


def _header_grits_score(
    gt_blocks: list[HeaderBlock],
    pred_blocks: list[HeaderBlock],
    gt_to_pred: dict[int, int],
    grits_scores: dict[tuple[int, int], float],
) -> float:
    """Submetric 2: GriTS-Con on matched header blocks.

    Uses the shared block matching. Unmatched GT blocks score 0. Extra
    pred blocks are penalised by averaging over max(n_gt, n_pred).
    """
    if not gt_blocks and not pred_blocks:
        return 1.0
    if not gt_blocks or not pred_blocks:
        return 0.0

    n_gt = len(gt_blocks)
    n_pred = len(pred_blocks)

    matched_scores = [grits_scores[(gi, pi)] for gi, pi in gt_to_pred.items()]

    denom = max(n_gt, n_pred)
    return sum(matched_scores) / denom


def _header_content_bag_score(
    gt_cells: list[HeaderCell],
    pred_cells: list[HeaderCell],
) -> float:
    """Submetric 3: bag-of-cells exact content overlap.

    For each GT header cell text, check if any pred header cell text
    matches exactly (after formatting normalization).
    Score = matched / total GT.
    """
    if not gt_cells and not pred_cells:
        return 1.0
    if not gt_cells or not pred_cells:
        return 0.0

    pred_texts = [c.text for c in pred_cells]
    used = [False] * len(pred_texts)
    matched = 0

    for gt_cell in gt_cells:
        for j, pt in enumerate(pred_texts):
            if used[j]:
                continue
            if gt_cell.text == pt:
                used[j] = True
                matched += 1
                break

    return matched / len(gt_cells)


def _header_perfect_score(
    gt_cells: list[HeaderCell],
    pred_cells: list[HeaderCell],
) -> float:
    """Submetric 4: binary exact structure match.

    Returns 1.0 iff the header cells have the same count and each
    (text, row, col, rowspan, colspan) matches exactly (in sorted order).
    """
    if not gt_cells and not pred_cells:
        return 1.0
    if len(gt_cells) != len(pred_cells):
        return 0.0

    def _key(c: HeaderCell) -> tuple[int, int, int, int, str]:
        return (c.row, c.col, c.rowspan, c.colspan, c.text)

    gt_sorted = sorted(gt_cells, key=_key)
    pred_sorted = sorted(pred_cells, key=_key)

    for g, p in zip(gt_sorted, pred_sorted, strict=True):
        if _key(g) != _key(p):
            return 0.0
    return 1.0


def _header_block_extent_score(
    gt_blocks: list[HeaderBlock],
    pred_blocks: list[HeaderBlock],
    gt_to_pred: dict[int, int],
    table_rows_gt: int,
    table_cols_gt: int,
    table_rows_pred: int,
    table_cols_pred: int,
) -> float:
    """Submetric 7: header block location/extent similarity.

    For each matched GT-pred block pair, computes IoU of their normalised
    extents within their respective tables. Averages over max(n_gt, n_pred).
    """
    if not gt_blocks and not pred_blocks:
        return 1.0
    if not gt_blocks or not pred_blocks:
        return 0.0

    n_gt = len(gt_blocks)
    n_pred = len(pred_blocks)

    def _extent_iou(
        gt_b: HeaderBlock,
        pred_b: HeaderBlock,
    ) -> float:
        e1 = gt_b.extent(table_rows_gt, table_cols_gt)
        e2 = pred_b.extent(table_rows_pred, table_cols_pred)

        # IoU on normalised rectangles
        r1, c1, r2, c2 = e1
        r3, c3, r4, c4 = e2

        inter_r1 = max(r1, r3)
        inter_c1 = max(c1, c3)
        inter_r2 = min(r2, r4)
        inter_c2 = min(c2, c4)

        inter_area = max(0.0, inter_r2 - inter_r1) * max(0.0, inter_c2 - inter_c1)
        area1 = (r2 - r1) * (c2 - c1)
        area2 = (r4 - r3) * (c4 - c3)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    total = 0.0
    for gi, pi in gt_to_pred.items():
        total += _extent_iou(gt_blocks[gi], pred_blocks[pi])

    denom = max(n_gt, n_pred)
    return total / denom


def _block_edge_vector(a: HeaderBlock, b: HeaderBlock) -> tuple[float, float]:
    """Signed vector from nearest edge of *a* to nearest edge of *b*, in cells.

    Returns (dr, dc) where positive dr means *b* is below *a* and
    positive dc means *b* is to the right of *a*. Components are zero
    when the blocks overlap along that axis.
    """
    # Row component (signed gap)
    if b.min_row >= a.max_row:
        dr = float(b.min_row - a.max_row)
    elif a.min_row >= b.max_row:
        dr = -float(a.min_row - b.max_row)
    else:
        dr = 0.0

    # Col component (signed gap)
    if b.min_col >= a.max_col:
        dc = float(b.min_col - a.max_col)
    elif a.min_col >= b.max_col:
        dc = -float(a.min_col - b.max_col)
    else:
        dc = 0.0

    return (dr, dc)


def _block_edge_distance(a: HeaderBlock, b: HeaderBlock) -> float:
    """Shortest Euclidean distance between edges/corners of two blocks, in cells.

    If blocks overlap or are adjacent, returns 0.
    """
    dr, dc = _block_edge_vector(a, b)
    return float((dr**2 + dc**2) ** 0.5)


def _direction_similarity(
    gt_dr: float,
    gt_dc: float,
    pred_dr: float,
    pred_dc: float,
) -> float:
    """Cosine similarity between two edge vectors, mapped to [0, 1].

    Returns (cos_sim + 1) / 2 so that:
    - parallel vectors → 1.0
    - perpendicular → 0.5
    - opposite → 0.0

    If either vector is zero-length, returns 1.0 (co-located blocks,
    direction is irrelevant).
    """
    gt_mag = (gt_dr**2 + gt_dc**2) ** 0.5
    pred_mag = (pred_dr**2 + pred_dc**2) ** 0.5
    if gt_mag < 1e-9 or pred_mag < 1e-9:
        return 1.0
    cos_sim = (gt_dr * pred_dr + gt_dc * pred_dc) / (gt_mag * pred_mag)
    # Clamp for floating-point safety
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return float((cos_sim + 1.0) / 2.0)


def _header_block_relative_position_score(
    gt_blocks: list[HeaderBlock],
    pred_blocks: list[HeaderBlock],
    gt_to_pred: dict[int, int],
    table_rows_gt: int,
    table_cols_gt: int,
    table_rows_pred: int,
    table_cols_pred: int,
) -> tuple[float, float]:
    """Proximity and direction scores for header block pairs.

    For every pair of matched GT blocks, computes:
    - **proximity**: similarity of nearest-edge distances in cell units
      ``1 - |dist_gt - dist_pred| / max(dist_gt, dist_pred)``
    - **direction**: cosine similarity of the signed edge vectors, mapped
      to [0, 1] via ``(cos + 1) / 2``

    The denominator is ``max(gt_pairs, pred_pairs)`` so that extra
    predicted blocks are penalised (unmatched pairs contribute 0).

    Returns (proximity, direction) averages.
    If <=1 block on both sides with the same count, returns (1.0, 1.0).
    If counts differ (0 vs 1), returns (0.0, 0.0).
    """
    n_gt = len(gt_blocks)
    n_pred = len(pred_blocks)

    if n_gt <= 1 and n_pred <= 1:
        # Both sides have ≤1 block — but if counts differ (0 vs 1)
        # that is a mismatch, not vacuous agreement.
        if n_gt != n_pred:
            return 0.0, 0.0
        return 1.0, 1.0
    if n_pred == 0:
        return 0.0, 0.0

    matched_gt_indices = sorted(gt_to_pred.keys())
    prox_scores: list[float] = []
    dir_scores: list[float] = []

    for idx_a in range(len(matched_gt_indices)):
        for idx_b in range(idx_a + 1, len(matched_gt_indices)):
            gi_a = matched_gt_indices[idx_a]
            gi_b = matched_gt_indices[idx_b]
            pi_a = gt_to_pred[gi_a]
            pi_b = gt_to_pred[gi_b]

            gt_dr, gt_dc = _block_edge_vector(gt_blocks[gi_a], gt_blocks[gi_b])
            pred_dr, pred_dc = _block_edge_vector(pred_blocks[pi_a], pred_blocks[pi_b])

            gt_dist = (gt_dr**2 + gt_dc**2) ** 0.5
            pred_dist = (pred_dr**2 + pred_dc**2) ** 0.5

            max_dist = max(gt_dist, pred_dist)
            if max_dist < 1e-9:
                prox_scores.append(1.0)
            else:
                prox_scores.append(1.0 - abs(gt_dist - pred_dist) / max_dist)

            dir_scores.append(_direction_similarity(gt_dr, gt_dc, pred_dr, pred_dc))

    gt_pairs = n_gt * (n_gt - 1) // 2
    pred_pairs = n_pred * (n_pred - 1) // 2
    total_pairs = max(gt_pairs, pred_pairs)

    if total_pairs == 0:
        return 1.0, 1.0

    # Sum matched pair scores and divide by total (unmatched pairs contribute 0)
    sum_prox = sum(prox_scores)
    sum_dir = sum(dir_scores)
    avg_prox = sum_prox / total_pairs
    avg_dir = sum_dir / total_pairs
    return avg_prox, avg_dir


# ---------------------------------------------------------------------------
# Header hierarchy depth
# ---------------------------------------------------------------------------


def _header_hierarchy_depth(cells: list[HeaderCell]) -> int:
    """Compute the depth of the header hierarchy.

    The depth is the number of distinct levels in the header tree.
    A cell with ``colspan > 1`` (or ``rowspan > 1``) is a parent that
    groups child cells underneath (or beside) it.

    The algorithm assigns each header cell to a *level* by tracing how
    many ancestor cells span over it from rows above.  A cell at
    ``(row, col)`` is a child of the innermost cell in a prior row whose
    column span covers ``col``.  The depth is the maximum nesting depth
    across all cells.

    Returns 0 when there are no header cells.
    """
    if not cells:
        return 0

    # Build occupancy: for each grid position, record the cell that owns it
    # sorted by row so we process top-down
    # For each cell, compute its level in the hierarchy.
    # A cell's level = 1 + level of its nearest ancestor (a cell in a
    # prior row whose column span covers this cell's columns).
    # Cells in the first header row (or with no ancestor) are at level 1.

    # Sort cells by (row, col) for top-down processing
    sorted_cells = sorted(cells, key=lambda c: (c.row, c.col))

    # Map grid positions to the cell that "owns" them and its level
    # For each column, track the stack of spanning cells
    # Simple approach: for each cell, find the deepest ancestor

    # Build a grid mapping (row, col) -> cell index
    cell_index: dict[tuple[int, int], int] = {}
    for idx, cell in enumerate(sorted_cells):
        for r in range(cell.row, cell.row + cell.rowspan):
            for c in range(cell.col, cell.col + cell.colspan):
                cell_index[(r, c)] = idx

    cell_level: dict[int, int] = {}

    for idx, cell in enumerate(sorted_cells):
        # Look for an ancestor: a cell in a prior row whose column span
        # covers at least one column of this cell, and which is a
        # *different* cell (not the same cell spanning multiple rows).
        best_ancestor_level = 0
        # Check the row just above this cell's start row
        if cell.row > 0:
            # Look at all columns this cell spans
            ancestor_candidates: set[int] = set()
            for c in range(cell.col, cell.col + cell.colspan):
                for r in range(cell.row - 1, -1, -1):
                    if (r, c) in cell_index:
                        anc_idx = cell_index[(r, c)]
                        if anc_idx != idx:
                            ancestor_candidates.add(anc_idx)
                        break  # found the nearest cell above in this column

            for anc_idx in ancestor_candidates:
                # The ancestor must have colspan > 1 OR be a spanning cell
                # that groups this cell — i.e. its column span must be
                # strictly wider than this cell's, OR it must be in a
                # different row. A same-width cell in a row above still
                # forms a parent if it spans across.
                # Actually: any cell in a row above that covers our columns
                # is a potential parent in the hierarchy.
                if anc_idx in cell_level:
                    best_ancestor_level = max(best_ancestor_level, cell_level[anc_idx])

        cell_level[idx] = best_ancestor_level + 1

    return max(cell_level.values()) if cell_level else 0


def _header_hierarchy_depth_score(
    gt_cells: list[HeaderCell],
    pred_cells: list[HeaderCell],
) -> float:
    """Submetric 9: header hierarchy depth similarity.

    Compares the depth of the header hierarchy tree between GT and
    prediction using ``min(d_gt, d_pred) / max(d_gt, d_pred)``.
    Returns 1.0 when both have the same depth (including both 0).
    """
    gt_depth = _header_hierarchy_depth(gt_cells)
    pred_depth = _header_hierarchy_depth(pred_cells)
    if gt_depth == 0 and pred_depth == 0:
        return 1.0
    if gt_depth == 0 or pred_depth == 0:
        return 0.0
    return min(gt_depth, pred_depth) / max(gt_depth, pred_depth)


# ---------------------------------------------------------------------------
# Per-table composite computation
# ---------------------------------------------------------------------------


def compute_header_composite_for_table_pair(
    gt_html: str,
    pred_html: str,
) -> dict[str, float]:
    """Compute all header accuracy submetrics for a single table pair.

    Returns a dict with keys for each submetric plus the composite
    header_composite_v3 (mean of all submetrics).
    """
    result = _detailed_header_composite_for_table_pair(gt_html, pred_html)
    return result[0]


def _detailed_header_composite_for_table_pair(
    gt_html: str,
    pred_html: str,
    row_map: dict[int, int] | None = None,
    col_map: dict[int, int] | None = None,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Compute header accuracy scores and rich per-submetric diagnostics.

    Returns:
        (scores, details) where scores is a dict of metric_name -> float
        and details is a dict of metric_name -> list of detail strings.
    """
    gt_cells, gt_rows, gt_cols = _parse_header_cells(gt_html)
    pred_cells, pred_rows, pred_cols = _parse_header_cells(pred_html)

    gt_blocks = _find_header_blocks(gt_cells)
    pred_blocks = _find_header_blocks(pred_cells)

    gt_to_pred, block_grits_scores = _match_blocks(
        gt_blocks,
        pred_blocks,
        gt_rows,
        gt_cols,
        pred_rows,
        pred_cols,
    )

    scores: dict[str, float] = {}
    details: dict[str, list[str]] = {}

    gt_n = len(gt_cells)
    pred_n = len(pred_cells)
    gt_b = len(gt_blocks)
    pred_b = len(pred_blocks)

    # --- 1. header_cell_count ---
    s = _header_cell_count_score(gt_cells, pred_cells)
    scores["header_cell_count"] = s
    if gt_n == 0 and pred_n == 0:
        details["header_cell_count"] = [f"{s:.3f} — no header cells"]
    else:
        details["header_cell_count"] = [
            f"{s:.3f} — {pred_n}/{gt_n} cells predicted (min/max = {min(gt_n, pred_n)}/{max(gt_n, pred_n)})"
        ]

    # --- 2. header_grits ---
    s = _header_grits_score(gt_blocks, pred_blocks, gt_to_pred, block_grits_scores)
    scores["header_grits"] = s
    grits_lines: list[str] = [f"{s:.3f} — {pred_b}/{gt_b} blocks predicted"]
    for gi, pi in sorted(gt_to_pred.items()):
        gs = block_grits_scores.get((gi, pi), 0.0)
        gt_texts = sorted({c.text for c in gt_blocks[gi].cells if c.text})
        pred_texts = sorted({c.text for c in pred_blocks[pi].cells if c.text})
        if gt_texts or pred_texts:
            grits_lines.append(
                f"  block {gi + 1}↔{pi + 1}: grits={gs:.3f}"
                f" | expected [{', '.join(repr(t) for t in gt_texts[:5])}]"
                f" predicted [{', '.join(repr(t) for t in pred_texts[:5])}]"
            )
        else:
            gb, pb = gt_blocks[gi], pred_blocks[pi]
            grits_lines.append(
                f"  block {gi + 1}↔{pi + 1}: grits={gs:.3f}"
                f" | GT rows [{gb.min_row},{gb.max_row})"
                f" cols [{gb.min_col},{gb.max_col})"
                f"  pred rows [{pb.min_row},{pb.max_row})"
                f" cols [{pb.min_col},{pb.max_col})"
            )
    # Flag unmatched GT blocks
    for gi in range(gt_b):
        if gi not in gt_to_pred:
            gt_texts = sorted({c.text for c in gt_blocks[gi].cells if c.text})
            if gt_texts:
                grits_lines.append(
                    f"  block {gi + 1}: unmatched | expected [{', '.join(repr(t) for t in gt_texts[:5])}]"
                )
            else:
                gb = gt_blocks[gi]
                grits_lines.append(
                    f"  block {gi + 1}: unmatched | rows [{gb.min_row},{gb.max_row}) cols [{gb.min_col},{gb.max_col})"
                )
    details["header_grits"] = grits_lines

    # --- 3. header_content_bag (with matched/missing/unexpected) ---
    s = _header_content_bag_score(gt_cells, pred_cells)
    scores["header_content_bag"] = s
    # Recompute matching to get per-cell info
    pred_texts_list = [c.text for c in pred_cells]
    used = [False] * len(pred_texts_list)
    matched_texts: list[str] = []
    missing_texts: list[str] = []
    for gc in gt_cells:
        found = False
        for j, pt in enumerate(pred_texts_list):
            if used[j]:
                continue
            if gc.text == pt:
                used[j] = True
                matched_texts.append(gc.text)
                found = True
                break
        if not found:
            missing_texts.append(gc.text)
    unexpected_texts = [pred_texts_list[j] for j in range(len(pred_texts_list)) if not used[j]]

    bag_lines: list[str] = [f"{s:.3f} — {len(matched_texts)}/{gt_n} expected cells found"]
    if missing_texts:
        bag_lines.append(f"  missing: {list(missing_texts)}")
    if unexpected_texts:
        bag_lines.append(f"  unexpected: {list(unexpected_texts)}")
    details["header_content_bag"] = bag_lines

    # --- 4. perfect_header ---
    s = _header_perfect_score(gt_cells, pred_cells)
    scores["header_perfect"] = s
    if s == 1.0:
        details["header_perfect"] = [f"{s:.3f} — exact match ({gt_n} cells)"]
    elif gt_n != pred_n:
        details["header_perfect"] = [f"{s:.3f} — cell count differs ({gt_n} expected, {pred_n} predicted)"]
    else:
        # Same count but position/span mismatch — show first difference
        def _key(c: HeaderCell) -> tuple[int, int, int, int, str]:
            return (c.row, c.col, c.rowspan, c.colspan, c.text)

        gt_sorted = sorted(gt_cells, key=_key)
        pred_sorted = sorted(pred_cells, key=_key)
        diffs: list[str] = []
        for g, p in zip(gt_sorted, pred_sorted, strict=True):
            gk, pk = _key(g), _key(p)
            if gk != pk:
                diffs.append(
                    f"  expected ({g.row},{g.col}) {g.rowspan}x{g.colspan} {g.text!r}"
                    f" vs predicted ({p.row},{p.col}) {p.rowspan}x{p.colspan} {p.text!r}"
                )
                if len(diffs) >= 3:
                    break
        struct_lines = [f"{s:.3f} — position/span mismatch ({gt_n} cells)"]
        struct_lines.extend(diffs)
        details["header_perfect"] = struct_lines

    # --- 5. header_block_extent ---
    s = _header_block_extent_score(
        gt_blocks,
        pred_blocks,
        gt_to_pred,
        gt_rows,
        gt_cols,
        pred_rows,
        pred_cols,
    )
    scores["header_block_extent"] = s
    extent_lines: list[str] = [f"{s:.3f} — {len(gt_to_pred)}/{max(gt_b, pred_b)} blocks matched"]
    for gi, pi in sorted(gt_to_pred.items()):
        e1 = gt_blocks[gi].extent(gt_rows, gt_cols)
        e2 = pred_blocks[pi].extent(pred_rows, pred_cols)
        # Compute IoU inline
        inter_r1, inter_c1 = max(e1[0], e2[0]), max(e1[1], e2[1])
        inter_r2, inter_c2 = min(e1[2], e2[2]), min(e1[3], e2[3])
        inter = max(0.0, inter_r2 - inter_r1) * max(0.0, inter_c2 - inter_c1)
        a1 = (e1[2] - e1[0]) * (e1[3] - e1[1])
        a2 = (e2[2] - e2[0]) * (e2[3] - e2[1])
        union = a1 + a2 - inter
        iou = inter / union if union > 0 else 0.0
        extent_lines.append(
            f"  block {gi + 1}↔{pi + 1}: IoU={iou:.3f}"
            f" | expected rows [{e1[0]:.2f},{e1[2]:.2f}] cols [{e1[1]:.2f},{e1[3]:.2f}]"
            f" predicted rows [{e2[0]:.2f},{e2[2]:.2f}] cols [{e2[1]:.2f},{e2[3]:.2f}]"
        )
    details["header_block_extent"] = extent_lines

    # --- 7/8. header_block_proximity & header_block_relative_direction ---
    s_prox, s_dir = _header_block_relative_position_score(
        gt_blocks,
        pred_blocks,
        gt_to_pred,
        gt_rows,
        gt_cols,
        pred_rows,
        pred_cols,
    )
    scores["header_block_proximity"] = s_prox
    scores["header_block_relative_direction"] = s_dir
    if gt_b == pred_b and gt_b <= 1:
        details["header_block_proximity"] = [f"{s_prox:.3f} — ≤1 block, no pairwise distances"]
        details["header_block_relative_direction"] = [f"{s_dir:.3f} — ≤1 block, no pairwise distances"]
    elif gt_b <= 1 and pred_b <= 1:
        details["header_block_proximity"] = [
            f"{s_prox:.3f} — block count mismatch ({gt_b} expected, {pred_b} predicted)"
        ]
        details["header_block_relative_direction"] = [
            f"{s_dir:.3f} — block count mismatch ({gt_b} expected, {pred_b} predicted)"
        ]
    else:
        matched_indices = sorted(gt_to_pred.keys())
        prox_pair_details: list[str] = []
        dir_pair_details: list[str] = []
        for idx_a in range(len(matched_indices)):
            for idx_b in range(idx_a + 1, len(matched_indices)):
                gi_a, gi_b = matched_indices[idx_a], matched_indices[idx_b]
                pi_a, pi_b = gt_to_pred[gi_a], gt_to_pred[gi_b]
                gt_dr, gt_dc = _block_edge_vector(gt_blocks[gi_a], gt_blocks[gi_b])
                pred_dr, pred_dc = _block_edge_vector(pred_blocks[pi_a], pred_blocks[pi_b])
                gt_dist = (gt_dr**2 + gt_dc**2) ** 0.5
                pred_dist = (pred_dr**2 + pred_dc**2) ** 0.5
                max_dist = max(gt_dist, pred_dist)
                pair_prox = 1.0 if max_dist < 1e-9 else 1.0 - abs(gt_dist - pred_dist) / max_dist
                pair_dir = _direction_similarity(gt_dr, gt_dc, pred_dr, pred_dc)
                prox_pair_details.append(
                    f"  blocks {gi_a + 1}↔{gi_b + 1}: proximity={pair_prox:.3f}"
                    f" | gt_dist={gt_dist:.1f} pred_dist={pred_dist:.1f}"
                )
                dir_pair_details.append(
                    f"  blocks {gi_a + 1}↔{gi_b + 1}: direction={pair_dir:.3f}"
                    f" | gt_vec=({gt_dr:.1f},{gt_dc:.1f})"
                    f" pred_vec=({pred_dr:.1f},{pred_dc:.1f})"
                )
        gt_pairs_count = gt_b * (gt_b - 1) // 2
        pred_pairs_count = pred_b * (pred_b - 1) // 2
        total_pairs_count = max(gt_pairs_count, pred_pairs_count)
        prox_lines = [f"{s_prox:.3f} — {len(prox_pair_details)} matched / {total_pairs_count} total pair(s)"]
        dir_lines = [f"{s_dir:.3f} — {len(dir_pair_details)} matched / {total_pairs_count} total pair(s)"]
        if pred_b > gt_b:
            extra_msg = f"  extra pred blocks: {pred_b - gt_b} (penalised via denominator)"
            prox_lines.append(extra_msg)
            dir_lines.append(extra_msg)
        prox_lines.extend(prox_pair_details[:5])
        dir_lines.extend(dir_pair_details[:5])
        details["header_block_proximity"] = prox_lines
        details["header_block_relative_direction"] = dir_lines

    # --- 9. multilevel_header_depth ---
    s = _header_hierarchy_depth_score(gt_cells, pred_cells)
    scores["multilevel_header_depth"] = s
    gt_depth = _header_hierarchy_depth(gt_cells)
    pred_depth = _header_hierarchy_depth(pred_cells)
    if gt_depth == 0 and pred_depth == 0:
        details["multilevel_header_depth"] = [f"{s:.3f} — no header hierarchy"]
    else:
        details["multilevel_header_depth"] = [f"{s:.3f} — expected depth {gt_depth}, predicted depth {pred_depth}"]

    # --- 10. header_data_alignment ---
    pred_text_lookup = _build_text_lookup(pred_html)
    has_grits_alignment = bool(row_map) and bool(col_map)
    if has_grits_alignment:
        assert row_map is not None and col_map is not None
        s = _header_data_alignment_score(gt_cells, pred_text_lookup, row_map, col_map)
        scores["header_data_alignment"] = s
        aligned = int(s * len(gt_cells)) if gt_cells else 0
        details["header_data_alignment"] = [
            f"{s:.3f} — {aligned}/{len(gt_cells)} GT headers aligned (via GriTS row/col mapping)"
        ]
    else:
        s = _header_data_alignment_score_fallback(
            gt_html,
            pred_html,
            gt_cells,
            pred_text_lookup,
        )
        scores["header_data_alignment"] = s
        aligned = int(s * len(gt_cells)) if gt_cells else 0
        details["header_data_alignment"] = [
            f"{s:.3f} — {aligned}/{len(gt_cells)} GT headers aligned (computed via standalone alignment, no GriTS data)"
        ]

    # --- composite (mean of applicable _COMPOSITE_KEYS, excludes header_perfect) ---
    # Exclude submetrics that are trivially 1.0 because both GT and prediction
    # fall in the degenerate case (e.g. ≤1 block on both sides).  When counts
    # differ (e.g. GT has 0 blocks but prediction has 2) the metric is kept so
    # that the mismatch is penalised.
    trivial_keys: set[str] = set()
    if gt_b == pred_b and gt_b <= 1:
        trivial_keys.add("header_block_proximity")
        trivial_keys.add("header_block_relative_direction")

    applicable_keys = [k for k in _COMPOSITE_KEYS if k not in trivial_keys]
    if applicable_keys:
        scores["header_composite_v3"] = sum(scores[k] for k in applicable_keys) / len(applicable_keys)
    else:
        # All submetrics trivial — perfect by default
        scores["header_composite_v3"] = 1.0

    sub_strs = [f"{k}={scores[k]:.3f}" for k in _COMPOSITE_KEYS]
    skipped_strs = [f"{k} (trivial, excluded)" for k in _COMPOSITE_KEYS if k in trivial_keys]
    composite_lines = [
        f"{scores['header_composite_v3']:.3f} — " + ", ".join(sub_strs),
        f"{pred_n}/{gt_n} cells, {pred_b}/{gt_b} blocks",
    ]
    if skipped_strs:
        composite_lines.append(f"excluded from composite: {', '.join(skipped_strs)}")
    details["header_composite_v3"] = composite_lines

    return scores, details


# ---------------------------------------------------------------------------
# Metric class (multi-table document level)
# ---------------------------------------------------------------------------

# All submetric keys emitted by this metric
SUBMETRIC_KEYS = [
    "header_cell_count",
    "header_grits",
    "header_content_bag",
    "header_perfect",
    "header_block_extent",
    "header_block_proximity",
    "header_block_relative_direction",
    "multilevel_header_depth",
    "header_data_alignment",
    "header_composite_v3",
]


class HeaderAccuracyMetric(Metric):
    """Header accuracy metric for comparing HTML tables in markdown content.

    Computes header accuracy between expected and actual HTML tables.
    Table-level matching can be provided externally (e.g. from GriTS) via
    the ``table_pairs`` parameter, or computed internally as a fallback.
    """

    @property
    def name(self) -> str:
        return "header_composite_v3"

    def compute(  # type: ignore[override]
        self,
        expected: str,
        actual: str,
        table_pairs: list[tuple[str, str]] | None = None,
        table_alignments: list[tuple[dict[int, int], dict[int, int]]] | None = None,
        **kwargs: Any,
    ) -> list[MetricValue]:
        """Compute header accuracy scores between expected and actual content.

        Args:
            expected: Full document markdown/HTML with ground-truth tables.
            actual: Full document markdown/HTML with predicted tables.
            table_pairs: Optional pre-matched list of (gt_html, pred_html)
                table pairs (e.g. from GriTS matching). If None, tables are
                extracted and matched internally via Hungarian on the
                overall header_composite_v3 score.
            table_alignments: Optional per-table GriTS row/col alignment
                maps as [(row_map, col_map), ...]. Used for the
                header_data_alignment submetric.

        Returns a list of MetricValues: one for the overall header_composite_v3
        and one for each submetric.
        """
        if table_pairs is not None:
            return self._compute_from_pairs(table_pairs, table_alignments)

        # Fallback: extract and match tables internally
        expected_tables = extract_html_tables(expected)
        actual_tables = extract_html_tables(actual)

        if not expected_tables:
            meta: dict[str, Any] = {
                "tables_found_expected": 0,
                "tables_found_actual": len(actual_tables),
            }
            return [MetricValue(metric_name="header_composite_v3", value=0.0, metadata=meta)]

        if not actual_tables:
            meta = {
                "tables_found_expected": len(expected_tables),
                "tables_found_actual": 0,
            }
            return [MetricValue(metric_name="header_composite_v3", value=0.0, metadata=meta)]

        n_gt = len(expected_tables)
        n_pred = len(actual_tables)

        # Compute all pairwise scores
        pair_scores: dict[tuple[int, int], dict[str, float]] = {}
        for i, gt_t in enumerate(expected_tables):
            for j, pred_t in enumerate(actual_tables):
                pair_scores[(i, j)] = compute_header_composite_for_table_pair(gt_t, pred_t)

        # Hungarian matching on overall header_composite_v3
        cost = np.zeros((n_gt, n_pred))
        for i in range(n_gt):
            for j in range(n_pred):
                cost[i, j] = -pair_scores[(i, j)]["header_composite_v3"]

        row_ind, col_ind = linear_sum_assignment(cost)

        # Build paired list
        pairs: list[tuple[str, str]] = []
        matched_gt: set[int] = set()
        for gt_idx, pred_idx in zip(row_ind, col_ind, strict=True):
            pairs.append((expected_tables[int(gt_idx)], actual_tables[int(pred_idx)]))
            matched_gt.add(int(gt_idx))

        # Unmatched GT tables get paired with empty string
        for i in range(n_gt):
            if i not in matched_gt:
                pairs.append((expected_tables[i], ""))

        return self._compute_from_pairs(pairs)

    def _compute_from_pairs(
        self,
        table_pairs: list[tuple[str, str]],
        table_alignments: list[tuple[dict[int, int], dict[int, int]]] | None = None,
    ) -> list[MetricValue]:
        """Compute header accuracy from pre-matched table pairs."""
        if not table_pairs:
            return [MetricValue(metric_name="header_composite_v3", value=0.0, metadata={})]

        accumulators: dict[str, list[float]] = {k: [] for k in SUBMETRIC_KEYS}
        per_table_details: list[dict[str, Any]] = []
        # Per-table rich detail strings keyed by submetric
        per_table_rich_details: list[dict[str, list[str]]] = []
        per_table_gt_cells: list[int] = []
        per_table_pred_cells: list[int] = []
        per_table_gt_blocks: list[int] = []
        per_table_pred_blocks: list[int] = []

        for idx, (gt_html, pred_html) in enumerate(table_pairs):
            if not pred_html:
                # Unmatched GT table
                for k in SUBMETRIC_KEYS:
                    accumulators[k].append(0.0)
                per_table_details.append({"table_pair_index": idx, **dict.fromkeys(SUBMETRIC_KEYS, 0.0)})
                gt_cells, _, _ = _parse_header_cells(gt_html)
                gt_n = len(gt_cells)
                gt_b = len(_find_header_blocks(gt_cells))
                per_table_gt_cells.append(gt_n)
                per_table_pred_cells.append(0)
                per_table_gt_blocks.append(gt_b)
                per_table_pred_blocks.append(0)
                # All-zero details for unmatched table
                unmatched_details: dict[str, list[str]] = {}
                for k in SUBMETRIC_KEYS:
                    if k == "header_composite_v3":
                        unmatched_details[k] = [f"0.000 — unmatched table ({gt_n} expected cells, 0 predicted)"]
                    else:
                        unmatched_details[k] = ["0.000 — no predicted table to compare"]
                per_table_rich_details.append(unmatched_details)
                continue

            gt_cells_parsed, _, _ = _parse_header_cells(gt_html)
            pred_cells_parsed, _, _ = _parse_header_cells(pred_html)
            per_table_gt_cells.append(len(gt_cells_parsed))
            per_table_pred_cells.append(len(pred_cells_parsed))
            per_table_gt_blocks.append(len(_find_header_blocks(gt_cells_parsed)))
            per_table_pred_blocks.append(len(_find_header_blocks(pred_cells_parsed)))

            if table_alignments and idx < len(table_alignments):
                pair_row_map, pair_col_map = table_alignments[idx]
            else:
                pair_row_map, pair_col_map = None, None

            scores, rich_details = _detailed_header_composite_for_table_pair(
                gt_html,
                pred_html,
                row_map=pair_row_map,
                col_map=pair_col_map,
            )
            for k in SUBMETRIC_KEYS:
                accumulators[k].append(scores[k])
            per_table_details.append({"table_pair_index": idx, **scores})
            per_table_rich_details.append(rich_details)

        shared_meta: dict[str, Any] = {
            "table_pairs": len(table_pairs),
            "per_table_details": per_table_details,
            "alignment_source": "grits" if table_alignments else "fallback",
        }

        # Build human-readable detail strings per submetric
        total_gt = sum(per_table_gt_cells)
        total_pred = sum(per_table_pred_cells)
        summary_line = f"{total_gt} header cell(s) expected, {total_pred} predicted across {len(table_pairs)} table(s)"

        submetric_details: dict[str, list[str]] = {}
        for k in SUBMETRIC_KEYS:
            lines: list[str] = [summary_line]
            for idx in range(len(table_pairs)):
                table_detail_lines = per_table_rich_details[idx].get(k, [])
                if len(table_pairs) > 1:
                    # Prefix first line with table number
                    if table_detail_lines:
                        lines.append(f"Table {idx + 1}: {table_detail_lines[0]}")
                        lines.extend(table_detail_lines[1:])
                    else:
                        td = per_table_details[idx]
                        lines.append(f"Table {idx + 1}: {k}={td.get(k, 0.0):.3f}")
                else:
                    # Single table — just append detail lines directly
                    lines.extend(table_detail_lines)
            submetric_details[k] = lines

        results: list[MetricValue] = []
        for k in SUBMETRIC_KEYS:
            vals = accumulators[k]
            avg = sum(vals) / len(vals) if vals else 0.0
            results.append(
                MetricValue(
                    metric_name=k,
                    value=avg,
                    metadata=shared_meta,
                    details=submetric_details.get(k, []),
                )
            )

        return results


# ---------------------------------------------------------------------------
# Generous header normalization
# ---------------------------------------------------------------------------


def _promote_top_row_to_header(pred_html: str) -> str:
    """Convert all <td> cells in the top row of *pred_html* to <th> cells."""
    soup = BeautifulSoup(pred_html, "lxml")
    table = soup.find("table")
    if not table:
        return pred_html
    rows = table.find_all("tr")
    if not rows:
        return pred_html
    for cell in rows[0].find_all("td"):
        cell.name = "th"
    return str(table)


def _apply_generous_header_normalization(gt_html: str, pred_html: str) -> str:
    """Promote pred's top row to a header if GT has headers and pred has none.
    Also promote bottom-left cells if GT has a bottom-left header block."""
    gt_cells, _, _ = _parse_header_cells(gt_html)
    if not gt_cells:
        return pred_html
    pred_cells, _, _ = _parse_header_cells(pred_html)
    if not pred_cells:
        pred_html = _promote_top_row_to_header(pred_html)
    # Always try bottom-left promotion (pred may have top headers but no
    # bottom-left headers, or we just promoted the top row above)
    pred_html = _promote_bottom_left_to_header(gt_html, pred_html)
    return pred_html


class HeaderAccuracyMetricGenerous(HeaderAccuracyMetric):
    """Variant of HeaderAccuracyMetric with generous header normalization.

    When the GT table has header cells but the prediction has none,
    the prediction's top row is promoted to a header before scoring.
    Only the composite score is emitted (as ``header_composite_v3_generous``);
    sub-metrics are omitted to avoid name collisions with the base metric.
    """

    @property
    def name(self) -> str:
        return "exp_header_composite_v3_generous"

    def compute(  # type: ignore[override]
        self,
        expected: str,
        actual: str,
        table_pairs: list[tuple[str, str]] | None = None,
        table_alignments: list[tuple[dict[int, int], dict[int, int]]] | None = None,
        **kwargs: Any,
    ) -> list[MetricValue]:
        if table_pairs is not None:
            table_pairs = [
                (gt, _apply_generous_header_normalization(gt, pred) if pred else pred) for gt, pred in table_pairs
            ]
        results = super().compute(expected, actual, table_pairs, table_alignments, **kwargs)
        return self._rename_composite(results)

    @staticmethod
    def _rename_composite(results: list[MetricValue]) -> list[MetricValue]:
        """Keep only the composite MetricValue, renamed to exp_header_composite_v3_generous."""
        for mv in results:
            if mv.metric_name == "header_composite_v3":
                mv.metric_name = "exp_header_composite_v3_generous"
                return [mv]
        return []
