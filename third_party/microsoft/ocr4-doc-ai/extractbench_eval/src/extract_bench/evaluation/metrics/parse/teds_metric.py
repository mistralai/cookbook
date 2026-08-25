"""TEDS (Tree Edit Distance based Similarity) metric for HTML table comparison.

This metric computes structural and content similarity between HTML tables
using the TEDS metric from the PubTabNet paper.

Three variants are supported:
- teds: Full content comparison (structure + Levenshtein on cell text)
- teds_struct: Structure-only (ignores cell text entirely)
- teds_struct_bool: Structure + boolean content awareness (penalizes when one
  cell is empty and the other is not, but ignores the actual text)

Reference:
    https://github.com/ibm-aur-nlp/PubTabNet/blob/master/src/metric.py
"""

from collections import deque
from typing import Any

import Levenshtein
import numpy as np
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from scipy.optimize import linear_sum_assignment

from extract_bench.evaluation.metrics.base import Metric
from extract_bench.evaluation.metrics.parse import fast_tree_edit
from extract_bench.evaluation.metrics.parse.utils import normalize_cell_text
from extract_bench.schemas.evaluation import MetricValue

# =============================================================================
# Variant names
# =============================================================================

TEDS_CONTENT = "teds"
TEDS_STRUCT = "teds_struct"
TEDS_STRUCT_BOOL = "teds_struct_bool"
ALL_TEDS_VARIANTS = frozenset({TEDS_CONTENT, TEDS_STRUCT, TEDS_STRUCT_BOOL})

# =============================================================================
# TEDS Implementation (adapted from PubTabNet)
# =============================================================================


class TableTree(Tree):
    """
    Custom tree node for HTML table elements.

    Stores tag name, colspan/rowspan for cells, and tokenized content.
    """

    def __init__(
        self,
        tag: str,
        colspan: int | None = None,
        rowspan: int | None = None,
        content: list[str] | None = None,
        *children: "TableTree",
    ):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self) -> str:
        """Show tree using brackets notation (for debugging)."""
        if self.tag == "td":
            result = f'"tag": {self.tag}, "colspan": {self.colspan}, "rowspan": {self.rowspan}, "text": {self.content}'
        else:
            result = f'"tag": {self.tag}'
        for child in self.children:
            result += child.bracket()
        return "{" + result + "}"


class ContentConfig(Config):
    """
    APTED configuration for full TEDS (structure + content).

    Compares HTML table nodes by:
    - Tag name (must match exactly)
    - Colspan/rowspan attributes (must match exactly)
    - Cell text content (compared using normalized Levenshtein distance)
    """

    @staticmethod
    def maximum(*sequences: str) -> int:
        """Get maximum possible value for normalization."""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences: str) -> float:
        """Get Levenshtein distance normalized to 0-1 range."""
        if not sequences[0] and not sequences[1]:
            return 0.0
        max_len = self.maximum(*sequences)
        if max_len == 0:
            return 0.0
        return float(Levenshtein.distance(sequences[0], sequences[1])) / max_len

    def rename(self, node1: TableTree, node2: TableTree) -> float:
        """
        Compute the cost of renaming node1 to node2.

        Returns:
            0.0 if nodes are identical
            1.0 if tags or spans differ
            Normalized Levenshtein distance for cell content
        """
        # Tags must match
        if node1.tag != node2.tag:
            return 1.0

        # For cells, colspan and rowspan must match
        if node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
            return 1.0

        # For td cells, compare content
        if node1.tag == "td":
            if node1.content or node2.content:
                s1 = normalize_cell_text("".join(node1.content) if node1.content else "")
                s2 = normalize_cell_text("".join(node2.content) if node2.content else "")
                return self.normalized_distance(s1, s2)

        return 0.0


class StructConfig(Config):
    """
    APTED configuration for TEDS-Struct (structure only).

    Compares HTML table nodes by structure alone:
    - Tag name (must match exactly)
    - Colspan/rowspan attributes (must match exactly)
    - Cell text content is ignored entirely
    """

    def rename(self, node1: TableTree, node2: TableTree) -> float:
        """
        Compute the cost of renaming node1 to node2 (structure only).

        Returns:
            0.0 if tag and spans match
            1.0 if tags or spans differ
        """
        if node1.tag != node2.tag:
            return 1.0
        if node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
            return 1.0
        return 0.0


class StructBooleanContentConfig(Config):
    """
    APTED configuration for TEDS-Struct with boolean content awareness.

    Like StructConfig, but additionally penalizes mismatches in cell
    emptiness: cost is 1.0 if one cell is empty and the other is not,
    0.0 if both are empty or both are non-empty. The actual text content
    is ignored — only its presence or absence matters.
    """

    def rename(self, node1: TableTree, node2: TableTree) -> float:
        """
        Compute the cost of renaming node1 to node2 (structure + boolean content).

        Returns:
            0.0 if tag/spans match and both cells are empty or both non-empty
            1.0 if tags or spans differ, or one cell is empty and the other isn't
        """
        if node1.tag != node2.tag:
            return 1.0
        if node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
            return 1.0
        if node1.tag == "td":
            s1 = normalize_cell_text("".join(node1.content) if node1.content else "")
            s2 = normalize_cell_text("".join(node2.content) if node2.content else "")
            if bool(s1) != bool(s2):
                return 1.0
        return 0.0


# Map variant names to their Config classes
VARIANT_CONFIGS: dict[str, type[Config]] = {
    TEDS_CONTENT: ContentConfig,
    TEDS_STRUCT: StructConfig,
    TEDS_STRUCT_BOOL: StructBooleanContentConfig,
}


def _edit_distance(tree_pred: "TableTree", tree_true: "TableTree", config: Config) -> float:
    """Ordered tree-edit distance between two table trees.

    Prefers the fast Zhang-Shasha implementation in ``fast_tree_edit`` (numba
    JIT when available, pure-Python otherwise). Zhang-Shasha computes the
    *same* optimal ordered tree-edit distance as APTED, so the TEDS score is
    unchanged — this is a speedup, not a re-definition of the metric. Any
    unexpected failure (or ``EXTRACT_BENCH_FAST_TEDS=0``) falls back to APTED so
    the metric never breaks on an edge case the fast path hasn't seen.
    """
    if fast_tree_edit.FAST_TEDS_ENABLED:
        try:
            return fast_tree_edit.tree_edit_distance(tree_pred, tree_true, config)
        except Exception as e:  # pragma: no cover - defensive: never fail scoring
            fast_tree_edit._log_once(f"FALLBACK: fast tree-edit failed -> APTED. cause={e!r}")
    return float(APTED(tree_pred, tree_true, config).compute_edit_distance())


class TEDS:
    """
    Tree Edit Distance based Similarity metric for HTML tables.

    Computes similarity between two HTML tables using tree edit distance.
    Supports multiple scoring variants in a single call by sharing the
    HTML parsing and tree construction work.
    """

    def __init__(
        self,
        ignore_nodes: list[str] | None = None,
        variants: set[str] | None = None,
    ):
        self.ignore_nodes = ignore_nodes
        self.variants = variants if variants is not None else ALL_TEDS_VARIANTS
        self._tokens: list[str] = []

    def _tokenize(self, node: Any) -> None:
        """Tokenize table cell content into a list of tokens."""
        self._tokens.append(f"<{node.tag}>")
        if node.text is not None:
            self._tokens.extend(list(node.text))
        for child in node.getchildren():
            self._tokenize(child)
        if node.tag != "unk":
            self._tokens.append(f"</{node.tag}>")
        if node.tag != "td" and node.tail is not None:
            self._tokens.extend(list(node.tail))

    def _load_html_tree(self, node: Any, parent: TableTree | None = None) -> TableTree | None:
        """
        Convert an lxml HTML element to a TableTree for APTED.

        Args:
            node: lxml element node
            parent: Parent TableTree node (for recursive building)

        Returns:
            Root TableTree node if parent is None, else None
        """
        if node.tag == "td" or node.tag == "th":
            # For cells, extract content tokens
            self._tokens = []
            self._tokenize(node)
            cell_content = self._tokens[1:-1].copy()

            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell_content,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())

        if parent is not None:
            parent.children.append(new_node)

        # Recursively process children (but not for cells - their content is tokenized)
        if node.tag not in ("td", "th"):
            for child in node.getchildren():
                self._load_html_tree(child, new_node)

        if parent is None:
            return new_node
        return None

    def evaluate(self, pred: str, true: str) -> tuple[dict[str, float], int, int]:
        """
        Compute TEDS scores between predicted and ground truth HTML tables.

        Parses HTML and builds trees once, then runs APTED for each requested
        variant. APTED does not mutate the TableTree nodes, so the same trees
        are safely reused across calls.

        Args:
            pred: Predicted HTML table string
            true: Ground truth HTML table string

        Returns:
            Tuple of (scores_dict, gt_nodes, pred_nodes) where scores_dict
            maps variant name to its TEDS score.
        """
        empty_scores = dict.fromkeys(self.variants, 0.0)

        if not pred or not true:
            return (empty_scores, 0, 0)

        try:
            parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
            pred_doc = html.fromstring(pred, parser=parser)
            true_doc = html.fromstring(true, parser=parser)
        except Exception:
            return (empty_scores, 0, 0)

        # Find table elements
        if pred_doc.tag == "table":
            pred_table = pred_doc
        else:
            pred_tables = pred_doc.xpath(".//table")
            if not pred_tables:
                return (empty_scores, 0, 0)
            pred_table = pred_tables[0]

        if true_doc.tag == "table":
            true_table = true_doc
        else:
            true_tables = true_doc.xpath(".//table")
            if not true_tables:
                return (empty_scores, 0, 0)
            true_table = true_tables[0]

        # Optionally strip certain nodes
        if self.ignore_nodes:
            etree.strip_tags(pred_table, *self.ignore_nodes)
            etree.strip_tags(true_table, *self.ignore_nodes)

        # Count nodes for normalization
        n_nodes_pred = len(pred_table.xpath(".//*")) + 1  # +1 for root
        n_nodes_true = len(true_table.xpath(".//*")) + 1
        n_nodes = max(n_nodes_pred, n_nodes_true)

        if n_nodes == 0:
            return (empty_scores, n_nodes_true, n_nodes_pred)

        # Compute edit distance for each variant.
        # We rebuild the TableTree for each variant since it's cheap (O(n))
        # and avoids any potential issues with APTED's internal indexing.
        scores: dict[str, float] = {}
        for variant in self.variants:
            tree_pred = self._load_html_tree(pred_table)
            tree_true = self._load_html_tree(true_table)
            # n_nodes > 0 (checked above) guarantees both trees are non-None.
            assert tree_pred is not None and tree_true is not None
            config = VARIANT_CONFIGS[variant]()
            distance = _edit_distance(tree_pred, tree_true, config)
            scores[variant] = max(0.0, 1.0 - (float(distance) / n_nodes))

        return (scores, n_nodes_true, n_nodes_pred)


def extract_html_tables(content: str) -> list[str]:
    """
    Extract all HTML tables from markdown content.

    Args:
        content: Markdown content potentially containing HTML tables

    Returns:
        List of HTML table strings
    """
    if not content:
        return []

    try:
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        doc = html.fromstring(content, parser=parser)
    except Exception:
        return []

    tables = []

    # If the root is a table, include it
    if doc.tag == "table":
        tables.append(etree.tostring(doc, encoding="unicode"))
    else:
        # Find all nested tables
        for table in doc.xpath(".//table"):
            tables.append(etree.tostring(table, encoding="unicode"))

    return tables


# =============================================================================
# TEDSMetric class
# =============================================================================


class TEDSMetric(Metric):
    """
    TEDS metric for comparing HTML tables in markdown content.

    Computes Tree Edit Distance based Similarity between expected and actual
    HTML tables. Auto-detects tables in both expected and actual markdown.

    Supports multiple TEDS variants (content, struct, struct+empty) and
    returns one MetricValue per variant.
    """

    def __init__(self, variants: set[str] | None = None):
        """
        Initialize TEDSMetric.

        Args:
            variants: Set of variant names to compute. Defaults to all variants.
                Valid names: "teds", "teds_struct", "teds_struct_bool".
        """
        self.variants = variants if variants is not None else set(ALL_TEDS_VARIANTS)

    @property
    def name(self) -> str:
        """Return the name of this metric."""
        return "teds"

    def compute(  # type: ignore[override]
        self,
        expected: str,
        actual: str,
        **kwargs: Any,
    ) -> list[MetricValue]:
        """
        Compute TEDS scores between expected and actual markdown content.

        Uses Hungarian algorithm for optimal table matching (based on
        TEDS-Content scores when available, otherwise the first variant).
        Then applies the same matching to all requested variants.

        Args:
            expected: Expected markdown with HTML tables (ground truth)
            actual: Actual markdown with HTML tables (from inference)
            kwargs: Additional parameters (not used)

        Returns:
            List of MetricValues, one per requested variant.
        """
        # Extract tables from both
        expected_tables = extract_html_tables(expected)
        actual_tables = extract_html_tables(actual)

        # No tables in expected means we can't compute TEDS
        if not expected_tables:
            return [
                MetricValue(
                    metric_name=variant,
                    value=0.0,
                    metadata={
                        "note": "No tables found in expected markdown",
                        "tables_found_expected": 0,
                        "tables_found_actual": len(actual_tables),
                    },
                )
                for variant in sorted(self.variants)
            ]

        # No tables in actual means all tables are missing
        if not actual_tables:
            return [
                MetricValue(
                    metric_name=variant,
                    value=0.0,
                    metadata={
                        "note": "No tables found in actual markdown",
                        "tables_found_expected": len(expected_tables),
                        "tables_found_actual": 0,
                        "tables_matched": 0,
                    },
                )
                for variant in sorted(self.variants)
            ]

        teds_calculator = TEDS(variants=self.variants)
        n_expected = len(expected_tables)
        n_actual = len(actual_tables)
        total_pairs = n_expected * n_actual

        print(
            f"  TEDS: comparing {n_expected} expected x {n_actual} actual = {total_pairs} table pair(s)",
            flush=True,
        )

        # Build cost matrix (negative TEDS scores for minimization)
        # Rows: expected tables, Columns: actual tables
        # Also store full results to avoid recomputation later
        cost_matrix = np.zeros((n_expected, n_actual))
        results_cache: dict[tuple[int, int], tuple[dict[str, float], int, int]] = {}

        # Use TEDS-Content for matching if available, else first variant
        matching_variant = TEDS_CONTENT if TEDS_CONTENT in self.variants else next(iter(sorted(self.variants)))

        pair_idx = 0
        for i, gt_table in enumerate(expected_tables):
            for j, pred_table in enumerate(actual_tables):
                pair_idx += 1
                if total_pairs > 1:
                    print(f"  TEDS: table pair {pair_idx}/{total_pairs}", flush=True)
                scores, gt_nodes, pred_nodes = teds_calculator.evaluate(pred_table, gt_table)
                results_cache[(i, j)] = (scores, gt_nodes, pred_nodes)
                cost_matrix[i, j] = -scores[matching_variant]

        # Solve assignment problem using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Build one MetricValue per variant
        metric_values: list[MetricValue] = []
        for variant in sorted(self.variants):
            per_table_scores: list[float] = []
            per_table_details: list[dict[str, Any]] = []
            matched_gt_indices: set[int] = set()

            for gt_idx, pred_idx in zip(row_ind, col_ind, strict=True):
                gt_idx_int = int(gt_idx)
                pred_idx_int = int(pred_idx)
                scores, gt_nodes, pred_nodes = results_cache[(gt_idx_int, pred_idx_int)]
                score = scores[variant]
                per_table_scores.append(score)
                per_table_details.append(
                    {
                        "gt_table_index": gt_idx_int,
                        "pred_table_index": pred_idx_int,
                        "score": score,
                        "gt_nodes": gt_nodes,
                        "pred_nodes": pred_nodes,
                    }
                )
                matched_gt_indices.add(gt_idx_int)

            # If there are unmatched expected tables, count them as 0
            for i in range(n_expected):
                if i not in matched_gt_indices:
                    per_table_scores.append(0.0)
                    per_table_details.append(
                        {
                            "gt_table_index": i,
                            "pred_table_index": None,
                            "score": 0.0,
                            "gt_nodes": 0,
                            "pred_nodes": 0,
                            "note": "No matching table in actual",
                        }
                    )

            aggregate_score = sum(per_table_scores) / len(per_table_scores) if per_table_scores else 0.0

            # Build human-readable detail strings
            details: list[str] = []
            details.append(f"{n_expected} table(s) expected, {n_actual} found, {len(row_ind)} matched")
            for td in per_table_details:
                gi = td["gt_table_index"]
                pi = td.get("pred_table_index")
                if pi is None:
                    details.append(f"Table {gi + 1}: no match found in prediction")
                else:
                    details.append(
                        f"Table {gi + 1}: {variant}={td['score']:.3f}"
                        f" (gt_nodes={td['gt_nodes']}, pred_nodes={td['pred_nodes']})"
                    )

            metric_values.append(
                MetricValue(
                    metric_name=variant,
                    value=aggregate_score,
                    metadata={
                        "tables_predicted": True,
                        "tables_found_expected": n_expected,
                        "tables_found_actual": n_actual,
                        "tables_matched": len(row_ind),
                        "per_table_scores": per_table_scores,
                        "per_table_details": per_table_details,
                    },
                    details=details,
                )
            )

        variant_str = ", ".join(
            f"{v}={m.value:.4f}" for v, m in zip(sorted(self.variants), metric_values, strict=False)
        )
        print(f"  TEDS: done, {variant_str}", flush=True)

        return metric_values
