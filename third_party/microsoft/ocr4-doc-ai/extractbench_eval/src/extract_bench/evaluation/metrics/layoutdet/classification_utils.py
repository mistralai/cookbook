"""Classification utilities for layout detection evaluation.

This module provides functions for matching predictions to ground truth
and computing per-class precision/recall/F1 metrics.
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

from extract_bench.evaluation.metrics.layoutdet.iou import compute_iou_matrix


def match_predictions_to_gt(
    pred_boxes: np.ndarray,  # (N, 4) xyxy format
    pred_scores: np.ndarray,  # (N,)
    pred_classes: np.ndarray,  # (N,) class indices
    gt_boxes: np.ndarray,  # (M, 4) xyxy format
    gt_classes: np.ndarray,  # (M,) class indices
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match predictions to ground truth using IoU threshold.

    Uses greedy matching: predictions are processed in descending score order,
    each prediction matches the best available GT with same class and IoU >= threshold.

    :param pred_boxes: Prediction bounding boxes, shape (N, 4) in xyxy format
    :param pred_scores: Prediction confidence scores, shape (N,)
    :param pred_classes: Prediction class indices, shape (N,)
    :param gt_boxes: Ground truth bounding boxes, shape (M, 4) in xyxy format
    :param gt_classes: Ground truth class indices, shape (M,)
    :param iou_threshold: IoU threshold for matching (default 0.5)
    :return: Tuple of (y_true, y_score) where:
             - y_true: Binary array (N,) - 1 if prediction matches GT, 0 otherwise
             - y_score: Confidence scores (N,)
    """
    if len(pred_boxes) == 0:
        return np.array([]), np.array([])
    if len(gt_boxes) == 0:
        return np.zeros(len(pred_boxes)), pred_scores.copy()

    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    # Sort predictions by score (descending)
    sorted_indices = np.argsort(-pred_scores)

    y_true = np.zeros(len(pred_boxes))
    matched_gt: set[int] = set()

    for pred_idx in sorted_indices:
        pred_class = pred_classes[pred_idx]

        # Find best matching GT with same class
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx in range(len(gt_boxes)):
            if gt_idx in matched_gt:
                continue
            if gt_classes[gt_idx] != pred_class:
                continue

            iou = iou_matrix[pred_idx, gt_idx]
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            y_true[pred_idx] = 1
            matched_gt.add(best_gt_idx)

    return y_true, pred_scores.copy()


def compute_per_class_metrics(
    predictions: list[dict],  # [{bbox, class_name, score}, ...]
    ground_truth: list[dict],  # [{bbox, class_name}, ...]
    class_names: list[str],
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """
    Compute per-class precision, recall, F1 at given IoU threshold.

    :param predictions: List of predictions, each with 'bbox', 'class_name', 'score'
    :param ground_truth: List of ground truth, each with 'bbox', 'class_name'
    :param class_names: List of class names to evaluate
    :param iou_threshold: IoU threshold for matching (default 0.5)
    :return: Dict mapping class_name to metrics dict {precision, recall, f1, ap, support}
    """
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    class_set = set(class_names)
    results: dict[str, dict[str, float]] = {}

    has_pages = False
    for entry in predictions:
        if "example_id" in entry:
            has_pages = True
            break
    if not has_pages:
        for entry in ground_truth:
            if "example_id" in entry:
                has_pages = True
                break

    if not has_pages:
        for class_name in class_names:
            class_idx = class_to_idx[class_name]

            # Filter by class
            class_preds = [p for p in predictions if p["class_name"] == class_name]
            class_gt = [g for g in ground_truth if g["class_name"] == class_name]

            if len(class_gt) == 0:
                results[class_name] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "ap": 0.0,
                    "support": 0,
                }
                continue

            if len(class_preds) == 0:
                results[class_name] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "ap": 0.0,
                    "support": len(class_gt),
                }
                continue

            # Convert to numpy arrays
            pred_boxes = np.array([p["bbox"] for p in class_preds])
            pred_scores = np.array([p["score"] for p in class_preds])
            pred_classes = np.full(len(class_preds), class_idx)

            gt_boxes = np.array([g["bbox"] for g in class_gt])
            gt_classes = np.full(len(class_gt), class_idx)

            # Match predictions to GT
            y_true, y_scores = match_predictions_to_gt(
                pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold
            )

            # Compute metrics
            tp = int(np.sum(y_true))
            fp = len(y_true) - tp
            fn = len(class_gt) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # Compute AP using sklearn
            if len(np.unique(y_true)) > 1:
                ap = float(average_precision_score(y_true, y_scores))
            else:
                ap = precision if np.all(y_true == 1) else 0.0

            results[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap": ap,
                "support": len(class_gt),
            }
        return results

    # Per-page matching to keep IoU matrices bounded by per-page box counts.
    predictions_by_page = defaultdict(lambda: defaultdict(list))  # type: ignore[var-annotated]
    ground_truth_by_page = defaultdict(lambda: defaultdict(list))  # type: ignore[var-annotated]

    for pred in predictions:
        if pred.get("class_name") not in class_set:
            continue
        page_id = str(pred.get("example_id", "__missing__"))
        predictions_by_page[page_id][pred["class_name"]].append(pred)

    for gt in ground_truth:
        if gt.get("class_name") not in class_set:
            continue
        page_id = str(gt.get("example_id", "__missing__"))
        ground_truth_by_page[page_id][gt["class_name"]].append(gt)

    page_ids = set(predictions_by_page.keys()) | set(ground_truth_by_page.keys())

    for class_name in class_names:
        class_idx = class_to_idx[class_name]
        total_true = 0
        total_false = 0
        total_support = 0
        all_y_true: list[int] = []
        all_y_scores: list[float] = []

        for page_id in page_ids:
            class_preds = predictions_by_page[page_id].get(class_name, [])
            class_gt = ground_truth_by_page[page_id].get(class_name, [])

            total_support += len(class_gt)
            if not class_gt:
                total_false += len(class_preds)
                all_y_true.extend([0] * len(class_preds))
                all_y_scores.extend([float(p["score"]) for p in class_preds])
                continue
            if not class_preds:
                continue

            pred_boxes = np.array([p["bbox"] for p in class_preds])
            pred_scores = np.array([p["score"] for p in class_preds])
            pred_classes = np.full(len(class_preds), class_idx)
            gt_boxes = np.array([g["bbox"] for g in class_gt])
            gt_classes = np.full(len(class_gt), class_idx)

            y_true, y_scores = match_predictions_to_gt(
                pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold
            )
            y_true_list = [int(v) for v in y_true.tolist()]
            y_score_list = [float(v) for v in y_scores.tolist()]

            true_count = int(np.sum(y_true))
            total_true += true_count
            total_false += int(len(class_preds) - true_count)
            all_y_true.extend(y_true_list)
            all_y_scores.extend(y_score_list)

        if total_support == 0:
            results[class_name] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap": 0.0,
                "support": 0,
            }
            continue

        total_false_neg = total_support - total_true
        precision = total_true / (total_true + total_false) if (total_true + total_false) > 0 else 0.0
        recall = total_true / (total_true + total_false_neg) if (total_true + total_false_neg) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if len(set(all_y_true)) > 1:
            ap = float(average_precision_score(np.array(all_y_true), np.array(all_y_scores)))
        else:
            ap = precision if all_y_true and all(v == 1 for v in all_y_true) else 0.0

        results[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": ap,
            "support": total_support,
        }

    return results


def compute_map_at_thresholds(
    predictions: list[dict],  # [{bbox, class_name, score}, ...]
    ground_truth: list[dict],  # [{bbox, class_name}, ...]
    class_names: list[str],
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """
    Compute mAP at multiple IoU thresholds (COCO-style).

    :param predictions: List of predictions
    :param ground_truth: List of ground truth
    :param class_names: List of class names to evaluate
    :param iou_thresholds: List of IoU thresholds (default: [0.5, 0.55, ..., 0.95])
    :return: Dict with mAP@[.50:.95], AP50, AP75
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + i * 0.05 for i in range(10)]  # [0.5, 0.55, ..., 0.95]

    # Compute AP at each threshold
    ap_per_threshold: list[float] = []
    ap50 = 0.0
    ap75 = 0.0

    for threshold in iou_thresholds:
        per_class = compute_per_class_metrics(predictions, ground_truth, class_names, iou_threshold=threshold)

        # Mean AP across classes (only classes with support > 0)
        aps = [m["ap"] for m in per_class.values() if m["support"] > 0]
        mean_ap = float(np.mean(aps)) if aps else 0.0
        ap_per_threshold.append(mean_ap)

        if abs(threshold - 0.5) < 0.01:
            ap50 = mean_ap
        if abs(threshold - 0.75) < 0.01:
            ap75 = mean_ap

    # mAP@[.50:.95] is the mean across all thresholds
    map_50_95 = float(np.mean(ap_per_threshold)) if ap_per_threshold else 0.0

    return {
        "mAP@[.50:.95]": map_50_95,
        "AP50": ap50,
        "AP75": ap75,
    }
