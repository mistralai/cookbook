"""Confidence-scoped summary metrics.

All ``confidence_scoped_*`` metrics here are computed over the
emitted-fields-with-confidence row set (``scoring_scope`` in the metric
metadata). GT-denominator signals live in the ``confidence_full_gt_*`` metrics
instead.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

from extract_bench.schemas.evaluation import MetricValue

from .gt_normalization import lift_repeated_structure_gt, lift_repeated_structure_schema
from .paths import iter_leaf_values
from .scoring import build_confidence_field_rows, expected_doc_from_rules
from .types import ConfidenceFieldRow

DEFAULT_CONFIDENCE_THRESHOLDS: tuple[float, ...] = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
)


def auc(labels: list[bool], scores: list[float]) -> float | None:
    """Rank-based ROC AUC (Mann-Whitney U with tie correction), O(n log n).

    Equivalent to the pairwise definition with ties counted as half-wins; the
    aggregate path scores every row across a run, so quadratic pairing is not
    an option there.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    pos_total = sum(1 for label in labels if label)
    neg_total = len(labels) - pos_total
    if not pos_total or not neg_total:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    positive_rank_sum = 0.0
    start = 0
    while start < len(order):
        tie_end = start
        while tie_end + 1 < len(order) and scores[order[tie_end + 1]] == scores[order[start]]:
            tie_end += 1
        average_rank = (start + tie_end) / 2 + 1
        positive_rank_sum += average_rank * sum(1 for position in range(start, tie_end + 1) if labels[order[position]])
        start = tie_end + 1
    return (positive_rank_sum - pos_total * (pos_total + 1) / 2) / (pos_total * neg_total)


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


_MISSING = object()


def _row_gt_path(row: Any) -> str:
    """Ground-truth path a row matched.

    Prefers the authoritative ``matched_gt_path`` (where ``""`` means *no match*
    and is preserved as such). Falls back to the fused ``field_path`` only for
    legacy rows that predate ``matched_gt_path`` and lack the key entirely.
    """
    matched = _row_value(row, "matched_gt_path", _MISSING)
    if matched is not _MISSING:
        return str(matched or "")
    return str(_row_value(row, "field_path") or "")


def _thresholds_with_observed_confidences(
    rows: list[Any],
    thresholds: tuple[float, ...],
) -> tuple[float, ...]:
    observed = {
        round(float(confidence), 6) for row in rows if (confidence := _row_value(row, "confidence")) is not None
    }
    requested = {round(float(threshold), 6) for threshold in thresholds}
    return tuple(sorted(observed | requested))


def _threshold_rows(rows: list[Any], thresholds: tuple[float, ...]) -> list[dict[str, Any]]:
    # Sorted confidences + suffix counts turn each threshold into a binary
    # search; the observed-confidence curve makes len(thresholds) ~ len(rows),
    # so the naive per-threshold scan is quadratic on the aggregate path.
    scored = sorted(
        (float(confidence), bool(_row_value(row, "correct", False)))
        for row in rows
        if (confidence := _row_value(row, "confidence")) is not None
    )
    confidences = [confidence for confidence, _ in scored]
    correct_at_or_above = [0] * (len(scored) + 1)
    for index in range(len(scored) - 1, -1, -1):
        correct_at_or_above[index] = correct_at_or_above[index + 1] + (1 if scored[index][1] else 0)
    correct_total = sum(1 for row in rows if _row_value(row, "correct", False))
    threshold_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        start = bisect_left(confidences, threshold)
        selected_count = len(scored) - start
        correct_selected = correct_at_or_above[start]
        threshold_rows.append(
            {
                "threshold": threshold,
                "selected": selected_count,
                "correct_selected": correct_selected,
                "precision": correct_selected / selected_count if selected_count else 0.0,
                "precision_defined": selected_count > 0,
                "coverage": selected_count / len(rows) if rows else 0.0,
                "scored_coverage": selected_count / len(scored) if scored else 0.0,
                "recall_correct_fields": correct_selected / correct_total if correct_total else 0.0,
            }
        )
    return threshold_rows


def compute_confidence_threshold_rows(
    rows: list[Any],
    thresholds: tuple[float, ...] = DEFAULT_CONFIDENCE_THRESHOLDS,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    return _threshold_rows(rows, thresholds)


def coverage_precision_curve_points(threshold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_coverage: dict[float, dict[str, Any]] = {
        0.0: {
            "threshold": None,
            "selected": 0,
            "correct_selected": 0,
            "precision": 1.0,
            "coverage": 0.0,
            "scored_coverage": 0.0,
            "recall_correct_fields": 0.0,
            "precision_defined": False,
            "synthetic": True,
        }
    }
    for row in threshold_rows:
        coverage = float(row["coverage"])
        precision = float(row["precision"])
        candidate = {**row, "precision_defined": bool(row.get("selected", 0)), "synthetic": False}
        existing = by_coverage.get(coverage)
        if existing is None or precision > float(existing["precision"]):
            by_coverage[coverage] = candidate
    return [by_coverage[coverage] for coverage in sorted(by_coverage)]


def coverage_precision_auc(threshold_rows: list[dict[str, Any]]) -> float:
    points = [
        (float(row["coverage"]), float(row["precision"])) for row in coverage_precision_curve_points(threshold_rows)
    ]
    if len(points) < 2:
        return 0.0
    area = 0.0
    for (x0, _y0), (x1, y1) in zip(points, points[1:], strict=False):
        area += (x1 - x0) * y1
    return area


def _best_precision_at_min_coverage(threshold_rows: list[dict[str, Any]], target: float) -> tuple[float, bool]:
    """Best precision at >= target coverage, plus whether the target is reachable.

    An unreachable target reports 0.0, which is indistinguishable from true
    zero precision — callers must surface the reached flag alongside the value.
    """
    candidates = [row["precision"] for row in threshold_rows if row["coverage"] >= target]
    return (max(candidates), True) if candidates else (0.0, False)


def _best_coverage_at_min_precision(threshold_rows: list[dict[str, Any]], target: float) -> tuple[float, bool]:
    candidates = [row["coverage"] for row in threshold_rows if row["precision"] >= target]
    return (max(candidates), True) if candidates else (0.0, False)


def _schema_extra_granted_count(rows: list[Any]) -> int:
    """Rows counted correct only because a citation supports a GT-absent value.

    These ``schema_extra_*`` grants inflate emitted-field precision relative to
    strict GT matching, so their count is surfaced next to the headline numbers.
    """
    return sum(1 for row in rows if str(_row_value(row, "comparison_mode", "") or "").startswith("schema_extra_"))


def _row_metadata(row: ConfidenceFieldRow) -> dict[str, Any]:
    return {
        "field_path": row.field_path,
        "matched_gt_path": row.matched_gt_path,
        "field_pattern": row.field_pattern,
        "predicted_path": row.predicted_path,
        "confidence": row.confidence,
        "correct": row.correct,
        "comparison_score": row.comparison_score,
        "comparison_mode": row.comparison_mode,
        "verified": row.verified,
        "alignment_mode": row.alignment_mode,
        "alignment_score": row.alignment_score,
        "schema_valid": row.schema_valid,
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _full_gt_metrics(
    rows: list[Any],
    *,
    expected_output: Any | None = None,
    full_gt_totals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute GT-denominator metrics alongside confidence-scoped metrics."""
    full_gt_totals = full_gt_totals or {}
    expected_leaf_paths = (
        {path for path, _ in iter_leaf_values(expected_output)} if expected_output is not None else set()
    )
    if "expected_leaf_total" in full_gt_totals:
        expected_leaf_total = int(full_gt_totals["expected_leaf_total"])
    else:
        expected_leaf_total = len(expected_leaf_paths)

    if "unique_expected_leaf_covered" in full_gt_totals:
        unique_expected_leaf_covered = int(full_gt_totals["unique_expected_leaf_covered"])
    else:
        unique_expected_leaf_covered = len(
            {
                gt_path
                for row in rows
                if _row_value(row, "correct", False) and (gt_path := _row_gt_path(row)) in expected_leaf_paths
            }
        )

    invalid_schema_fields = int(
        full_gt_totals.get(
            "invalid_schema_fields",
            sum(1 for row in rows if _row_value(row, "schema_valid", True) is False),
        )
    )
    correct_emitted_rows = sum(1 for row in rows if _row_value(row, "correct", False))
    emitted_field_total = len(rows)
    scoped_precision = _safe_ratio(correct_emitted_rows, emitted_field_total)
    full_gt_recall = _safe_ratio(unique_expected_leaf_covered, expected_leaf_total)
    return {
        "confidence_scoped_field_count": emitted_field_total,
        "confidence_scoped_correct_fields": correct_emitted_rows,
        "confidence_scoped_accuracy": scoped_precision,
        "confidence_scoped_output_keyed_precision": scoped_precision,
        "confidence_full_gt_expected_leaf_count": expected_leaf_total,
        "confidence_full_gt_unique_leaf_covered": unique_expected_leaf_covered,
        "confidence_full_gt_recall": full_gt_recall,
        "confidence_full_gt_f1": _f1(scoped_precision, full_gt_recall),
        "confidence_full_gt_invalid_schema_fields": invalid_schema_fields,
    }


def _confidence_scoped_roc_auc(rows: list[Any]) -> float | None:
    scored_rows = [row for row in rows if _row_value(row, "confidence") is not None]
    labels = [bool(_row_value(row, "correct", False)) for row in scored_rows]
    scores = [float(_row_value(row, "confidence")) for row in scored_rows]
    return auc(labels, scores)


def summarize_confidence_rows(
    rows: list[Any],
    *,
    thresholds: tuple[float, ...] = DEFAULT_CONFIDENCE_THRESHOLDS,
    precision_coverage_target: float = 0.8,
    coverage_precision_target: float = 0.95,
    expected_output: Any | None = None,
    full_gt_totals: dict[str, Any] | None = None,
    data_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Rows are matched against the lifted GT shape; the full-GT denominator
    # must count that same shape, whichever caller we came from.
    expected_output = lift_repeated_structure_gt(expected_output, lift_repeated_structure_schema(data_schema or {}))
    threshold_rows = compute_confidence_threshold_rows(rows, thresholds)
    curve_thresholds = _thresholds_with_observed_confidences(rows, thresholds)
    curve_threshold_rows = compute_confidence_threshold_rows(rows, curve_thresholds)
    curve_points = coverage_precision_curve_points(curve_threshold_rows)
    full_gt = _full_gt_metrics(rows, expected_output=expected_output, full_gt_totals=full_gt_totals)
    roc_auc = _confidence_scoped_roc_auc(rows)
    if not rows:
        return {
            "total_fields": 0,
            "correct_fields": 0,
            "accuracy": 0.0,
            "fields_with_confidence": 0,
            "confidence_scoped_unique_value_count": 0,
            "unique_confidence_values_sample": [],
            "curve_threshold_count": len(curve_thresholds),
            "confidence_scoped_auc": 0.0,
            "confidence_scoped_normalized_auc": 0.0,
            "confidence_scoped_roc_auc": roc_auc,
            f"confidence_scoped_precision_at_{int(precision_coverage_target * 100)}pct_coverage": 0.0,
            f"confidence_scoped_precision_at_{int(precision_coverage_target * 100)}pct_coverage_reached": False,
            f"confidence_scoped_coverage_at_{int(coverage_precision_target * 100)}pct_precision": 0.0,
            f"confidence_scoped_coverage_at_{int(coverage_precision_target * 100)}pct_precision_reached": False,
            "confidence_scoped_schema_extra_granted": 0,
            "thresholds": threshold_rows,
            "curve_points": curve_points,
            "scoring_scope": "emitted_fields_with_confidence",
            **full_gt,
        }

    scored_rows = [row for row in rows if _row_value(row, "confidence") is not None]
    unique_confidences = sorted(
        {
            round(float(confidence), 6)
            for confidence in (_row_value(row, "confidence") for row in scored_rows)
            if confidence is not None
        }
    )
    auc_value = coverage_precision_auc(curve_threshold_rows)
    max_coverage = max((row["coverage"] for row in curve_points), default=0.0)
    correct_fields = sum(1 for row in rows if _row_value(row, "correct", False))
    precision_at_coverage, coverage_target_reached = _best_precision_at_min_coverage(
        curve_threshold_rows, precision_coverage_target
    )
    coverage_at_precision, precision_target_reached = _best_coverage_at_min_precision(
        curve_threshold_rows, coverage_precision_target
    )
    return {
        "total_fields": len(rows),
        "correct_fields": correct_fields,
        "accuracy": correct_fields / len(rows),
        "fields_with_confidence": len(scored_rows),
        "confidence_scoped_unique_value_count": len(unique_confidences),
        "unique_confidence_values_sample": unique_confidences[:100],
        "curve_threshold_count": len(curve_thresholds),
        "confidence_scoped_auc": auc_value,
        "confidence_scoped_normalized_auc": auc_value / max_coverage if max_coverage > 0 else 0.0,
        "confidence_scoped_roc_auc": roc_auc,
        f"confidence_scoped_precision_at_{int(precision_coverage_target * 100)}pct_coverage": precision_at_coverage,
        f"confidence_scoped_precision_at_{int(precision_coverage_target * 100)}pct_coverage_reached": (
            coverage_target_reached
        ),
        f"confidence_scoped_coverage_at_{int(coverage_precision_target * 100)}pct_precision": coverage_at_precision,
        f"confidence_scoped_coverage_at_{int(coverage_precision_target * 100)}pct_precision_reached": (
            precision_target_reached
        ),
        "confidence_scoped_schema_extra_granted": _schema_extra_granted_count(rows),
        "thresholds": threshold_rows,
        "curve_points": curve_points,
        "scoring_scope": "emitted_fields_with_confidence",
        **full_gt,
    }


def _threshold_suffix(threshold: float) -> str:
    return str(threshold).replace(".", "_")


def _threshold_metric_values(threshold_rows: list[dict[str, Any]]) -> list[MetricValue]:
    metrics: list[MetricValue] = []
    for row in threshold_rows:
        suffix = _threshold_suffix(row["threshold"])
        metadata = {
            "threshold": row["threshold"],
            "passed": row["correct_selected"],
            "total": row["selected"],
            "selected": row["selected"],
            "correct_selected": row["correct_selected"],
            "coverage": row["coverage"],
            "scored_coverage": row["scored_coverage"],
            "recall_correct_fields": row["recall_correct_fields"],
            "precision_defined": row.get("precision_defined", row["selected"] > 0),
            "scoring_scope": "emitted_fields_with_confidence",
        }
        metrics.append(
            MetricValue(
                metric_name=f"confidence_scoped_precision_at_{suffix}",
                value=row["precision"],
                metadata=metadata,
            )
        )
        metrics.append(
            MetricValue(
                metric_name=f"confidence_scoped_coverage_at_{suffix}",
                value=row["coverage"],
                metadata=metadata,
            )
        )
    return metrics


def compute_confidence_scoped_metrics(
    *,
    extracted_data: Any,
    field_rules: list[Any],
    field_citations: list[Any],
    data_schema: dict[str, Any] | None = None,
    skip_field_paths: set[str] | None = None,
    thresholds: tuple[float, ...] = DEFAULT_CONFIDENCE_THRESHOLDS,
    precision_coverage_target: float = 0.8,
    coverage_precision_target: float = 0.95,
    expected_output: Any | None = None,
) -> list[MetricValue]:
    expected_doc = (
        expected_output
        if expected_output is not None
        else expected_doc_from_rules(
            field_rules,
            skip_field_paths or set(),
        )
    )
    rows = build_confidence_field_rows(
        extracted_data=extracted_data,
        field_rules=field_rules,
        field_citations=field_citations,
        data_schema=data_schema,
        skip_field_paths=skip_field_paths,
        expected_output=expected_doc,
    )
    summary = summarize_confidence_rows(
        rows,
        thresholds=thresholds,
        precision_coverage_target=precision_coverage_target,
        coverage_precision_target=coverage_precision_target,
        expected_output=expected_doc,
        data_schema=data_schema,
    )
    if not rows and not summary["confidence_full_gt_expected_leaf_count"]:
        return []

    threshold_rows = summary["thresholds"]
    curve_points = summary["curve_points"]
    scored_rows = [row for row in rows if _row_value(row, "confidence") is not None]
    unique_confidences = sorted(
        {
            round(float(_row_value(row, "confidence")), 6)
            for row in scored_rows
            if _row_value(row, "confidence") is not None
        }
    )
    base_metadata = {
        # Bumped when scoring semantics change enough that series are not
        # comparable across the boundary.
        "scorer_version": "2026-07-09",
        "total_fields": len(rows),
        "correct_fields": summary["correct_fields"],
        "scoped_accuracy": summary["confidence_scoped_accuracy"],
        "output_keyed_precision": summary["confidence_scoped_output_keyed_precision"],
        "fields_with_confidence": len(scored_rows),
        "unique_confidence_count": len(unique_confidences),
        "schema_extra_granted_count": summary["confidence_scoped_schema_extra_granted"],
        "confidence_source": "ExtractOutput.field_citations[*].confidence",
        "scoring_scope": "emitted_fields_with_confidence",
        "full_gt": {
            "expected_leaf_count": summary["confidence_full_gt_expected_leaf_count"],
            "unique_leaf_covered": summary["confidence_full_gt_unique_leaf_covered"],
            "recall": summary["confidence_full_gt_recall"],
            "f1": summary["confidence_full_gt_f1"],
            "invalid_schema_fields": summary["confidence_full_gt_invalid_schema_fields"],
        },
        "roc_auc_defined": summary["confidence_scoped_roc_auc"] is not None,
    }
    # The heavyweight diagnostics (threshold table, curve, row sample) ride on
    # exactly one carrier metric; sharing them across every confidence metric
    # multiplied the serialized _evaluation_report.json roughly ninefold.
    detailed_metadata = {
        **base_metadata,
        "thresholds": threshold_rows,
        "curve_threshold_count": summary.get("curve_threshold_count", 0),
        "curve_points": curve_points,
        "unique_confidence_values_sample": unique_confidences[:25],
        "field_rows_sample": [_row_metadata(row) for row in rows[:100]],
    }

    metrics = [
        MetricValue(
            metric_name="confidence_scoped_auc",
            value=summary["confidence_scoped_auc"],
            metadata=detailed_metadata,
        ),
        MetricValue(
            metric_name="confidence_scoped_normalized_auc",
            value=summary["confidence_scoped_normalized_auc"],
            metadata=dict(base_metadata),
        ),
        MetricValue(
            metric_name=f"confidence_scoped_precision_at_{int(precision_coverage_target * 100)}pct_coverage",
            value=summary[f"confidence_scoped_precision_at_{int(precision_coverage_target * 100)}pct_coverage"],
            metadata={
                **base_metadata,
                "coverage_target": precision_coverage_target,
                "coverage_target_reached": summary[
                    f"confidence_scoped_precision_at_{int(precision_coverage_target * 100)}pct_coverage_reached"
                ],
            },
        ),
        MetricValue(
            metric_name=f"confidence_scoped_coverage_at_{int(coverage_precision_target * 100)}pct_precision",
            value=summary[f"confidence_scoped_coverage_at_{int(coverage_precision_target * 100)}pct_precision"],
            metadata={
                **base_metadata,
                "precision_target": coverage_precision_target,
                "precision_target_reached": summary[
                    f"confidence_scoped_coverage_at_{int(coverage_precision_target * 100)}pct_precision_reached"
                ],
            },
        ),
        MetricValue(
            metric_name="confidence_scoped_unique_value_count",
            value=float(len(unique_confidences)),
            metadata={"unique_confidence_values_sample": unique_confidences[:100]},
        ),
    ]
    # Only meaningful when the pipeline emitted a confidence signal. With none
    # they collapse to 0.0 and read as total extraction failure, so gate them on
    # signal presence like the confidence_scoped_roc_auc gate below.
    if scored_rows:
        metrics.extend(
            [
                MetricValue(
                    metric_name="confidence_scoped_output_keyed_precision",
                    value=summary["confidence_scoped_output_keyed_precision"],
                    metadata=dict(base_metadata),
                ),
                MetricValue(
                    metric_name="confidence_full_gt_recall",
                    value=summary["confidence_full_gt_recall"],
                    metadata=dict(base_metadata),
                ),
                MetricValue(
                    metric_name="confidence_full_gt_f1",
                    value=summary["confidence_full_gt_f1"],
                    metadata=dict(base_metadata),
                ),
            ]
        )
    if summary["confidence_scoped_roc_auc"] is not None:
        metrics.append(
            MetricValue(
                metric_name="confidence_scoped_roc_auc",
                value=summary["confidence_scoped_roc_auc"],
                metadata=dict(base_metadata),
            )
        )
    metrics.extend(_threshold_metric_values(threshold_rows))
    return metrics
