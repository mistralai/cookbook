"""Wrapper around the vendored Microsoft table-transformer GriTS reference.

This module provides a GriTS metric class backed by the official reference
implementation from https://github.com/microsoft/table-transformer/blob/main/src/grits.py
so we can compare results and timing against our own implementation.

**Remove this file (and _vendor_grits_reference.py) before deploying.**
"""

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from extract_bench.evaluation.metrics.base import Metric
from extract_bench.evaluation.metrics.parse._vendor_grits_reference import (
    grits_from_html as ref_grits_from_html,
)
from extract_bench.evaluation.metrics.parse.table_extraction import extract_html_tables
from extract_bench.schemas.evaluation import MetricValue


class ReferenceGriTSMetric(Metric):
    """GriTS metric backed by the Microsoft table-transformer reference.

    Same interface as GriTSMetric so it can be swapped in with one line.
    Reports metrics with a 'ref_grits_' prefix to distinguish from ours.
    """

    @property
    def name(self) -> str:
        return "ref_grits"

    def compute(  # type: ignore[override]
        self,
        expected: str,
        actual: str,
        **kwargs: Any,
    ) -> list[MetricValue]:
        expected_tables = extract_html_tables(expected)
        actual_tables = extract_html_tables(actual)

        shared_meta: dict[str, Any] = {}

        if not expected_tables:
            shared_meta = {
                "note": "No tables found in expected markdown",
                "tables_found_expected": 0,
                "tables_found_actual": len(actual_tables),
            }
            return [
                MetricValue(metric_name="ref_grits_top", value=0.0, metadata=shared_meta),
                MetricValue(metric_name="ref_grits_con", value=0.0, metadata=shared_meta),
            ]

        if not actual_tables:
            shared_meta = {
                "note": "No tables found in actual markdown",
                "tables_found_expected": len(expected_tables),
                "tables_found_actual": 0,
                "tables_matched": 0,
            }
            return [
                MetricValue(metric_name="ref_grits_top", value=0.0, metadata=shared_meta),
                MetricValue(metric_name="ref_grits_con", value=0.0, metadata=shared_meta),
            ]

        n_expected = len(expected_tables)
        n_actual = len(actual_tables)
        total_pairs = n_expected * n_actual

        print(f"  ref_GriTS: comparing {n_expected} expected x {n_actual} actual = {total_pairs} table pair(s)")

        results_cache: dict[tuple[int, int], dict[str, float]] = {}
        cost_matrix = np.zeros((n_expected, n_actual))

        pair_idx = 0
        for i, gt_table in enumerate(expected_tables):
            for j, pred_table in enumerate(actual_tables):
                pair_idx += 1
                if total_pairs > 1:
                    print(f"  ref_GriTS: table pair {pair_idx}/{total_pairs}")
                try:
                    result = ref_grits_from_html(gt_table, pred_table)
                except Exception:
                    result = None
                if result is None:
                    result = {
                        "grits_top": 0.0,
                        "grits_con": 0.0,
                        "grits_precision_top": 0.0,
                        "grits_recall_top": 0.0,
                        "grits_top_upper_bound": 0.0,
                        "grits_precision_con": 0.0,
                        "grits_recall_con": 0.0,
                        "grits_con_upper_bound": 0.0,
                    }
                results_cache[(i, j)] = result
                cost_matrix[i, j] = -result["grits_con"]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        per_table_details: list[dict[str, Any]] = []
        top_scores: list[float] = []
        con_scores: list[float] = []
        matched_gt: set[int] = set()

        for gt_idx, pred_idx in zip(row_ind, col_ind, strict=True):
            gi, pi = int(gt_idx), int(pred_idx)
            result = results_cache[(gi, pi)]
            top_scores.append(result["grits_top"])
            con_scores.append(result["grits_con"])
            per_table_details.append(
                {
                    "gt_table_index": gi,
                    "pred_table_index": pi,
                    "grits_top": result["grits_top"],
                    "grits_con": result["grits_con"],
                }
            )
            matched_gt.add(gi)

        for i in range(n_expected):
            if i not in matched_gt:
                top_scores.append(0.0)
                con_scores.append(0.0)

        avg_top = sum(top_scores) / len(top_scores) if top_scores else 0.0
        avg_con = sum(con_scores) / len(con_scores) if con_scores else 0.0
        print(f"  ref_GriTS: done, top = {avg_top:.4f}, con = {avg_con:.4f}")

        shared_meta = {
            "tables_found_expected": n_expected,
            "tables_found_actual": n_actual,
            "tables_matched": len(row_ind),
            "per_table_details": per_table_details,
        }

        return [
            MetricValue(metric_name="ref_grits_top", value=avg_top, metadata=shared_meta),
            MetricValue(metric_name="ref_grits_con", value=avg_con, metadata=shared_meta),
        ]
