"""Bounding box geometry utilities for layout attribution evaluation.

Key concept: IoA (Intersection over Area) is used instead of IoU for
merge/split tolerance. If a predicted bbox fully contains a GT bbox,
IoA(gt, pred) = 1.0 even if IoU is small. This handles the case where
a service merges multiple GT elements into one predicted region.
"""

import numpy as np


def coco_to_xyxy(bbox: list[float]) -> list[float]:
    """Convert COCO format bbox [x, y, width, height] to xyxy format [x1, y1, x2, y2].

    :param bbox: Bounding box in COCO format [x, y, width, height]
    :return: Bounding box in xyxy format [x1, y1, x2, y2]
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def normalize_bbox_to_unit(
    bbox_pixel: dict | list,
    page_width: float,
    page_height: float,
) -> list[float]:
    """Convert pixel-coordinate bbox to normalized [0,1] COCO format [x, y, w, h].

    :param bbox_pixel: Bbox as dict with x/y/w/h keys or list [x, y, w, h] in pixels
    :param page_width: Page width in pixels
    :param page_height: Page height in pixels
    :return: Normalized bbox [x, y, w, h] in [0,1] range
    """
    if isinstance(bbox_pixel, dict):
        x = bbox_pixel["x"]
        y = bbox_pixel["y"]
        w = bbox_pixel["w"]
        h = bbox_pixel["h"]
    else:
        x, y, w, h = bbox_pixel

    return [
        x / page_width,
        y / page_height,
        w / page_width,
        h / page_height,
    ]


def _intersection_area(box1_xyxy: list[float], box2_xyxy: list[float]) -> float:
    """Compute intersection area between two xyxy boxes.

    :param box1_xyxy: First box [x1, y1, x2, y2]
    :param box2_xyxy: Second box [x1, y1, x2, y2]
    :return: Intersection area
    """
    x_left = max(box1_xyxy[0], box2_xyxy[0])
    y_top = max(box1_xyxy[1], box2_xyxy[1])
    x_right = min(box1_xyxy[2], box2_xyxy[2])
    y_bottom = min(box1_xyxy[3], box2_xyxy[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


def _box_area(box_xyxy: list[float]) -> float:
    """Compute area of an xyxy box.

    :param box_xyxy: Box [x1, y1, x2, y2]
    :return: Area
    """
    return max(0.0, (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1]))


def compute_ioa(gt_box_xyxy: list[float], pred_box_xyxy: list[float]) -> float:
    """Compute Intersection over Area (IoA) of the GT box.

    IoA(gt, pred) = area(intersection(gt, pred)) / area(gt)

    This is merge/split tolerant: if pred fully contains gt, IoA = 1.0
    regardless of how much extra area pred covers.

    :param gt_box_xyxy: Ground truth box [x1, y1, x2, y2]
    :param pred_box_xyxy: Predicted box [x1, y1, x2, y2]
    :return: IoA value in [0, 1]
    """
    gt_area = _box_area(gt_box_xyxy)
    if gt_area <= 0:
        return 0.0
    inter = _intersection_area(gt_box_xyxy, pred_box_xyxy)
    return inter / gt_area


def compute_ioa_matrix(
    gt_boxes: np.ndarray,  # shape (N, 4) xyxy
    pred_boxes: np.ndarray,  # shape (M, 4) xyxy
) -> np.ndarray:  # shape (N, M)
    """Compute pairwise IoA matrix: IoA[i, j] = intersection(gt_i, pred_j) / area(gt_i).

    :param gt_boxes: Array of shape (N, 4) with GT boxes in xyxy format
    :param pred_boxes: Array of shape (M, 4) with predicted boxes in xyxy format
    :return: IoA matrix of shape (N, M)
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)))

    gt_boxes = np.asarray(gt_boxes, dtype=float)
    pred_boxes = np.asarray(pred_boxes, dtype=float)

    # Compute GT areas
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute intersection
    lt = np.maximum(gt_boxes[:, None, :2], pred_boxes[None, :, :2])
    rb = np.minimum(gt_boxes[:, None, 2:], pred_boxes[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    # IoA = intersection / gt_area
    return intersection / np.clip(gt_areas[:, None], 1e-10, None)  # type: ignore[no-any-return]


def compute_overlap_matrix(
    gt_boxes: np.ndarray,  # shape (N, 4) xyxy
    pred_boxes: np.ndarray,  # shape (M, 4) xyxy
) -> np.ndarray:  # shape (N, M)
    """Compute bidirectional overlap matrix for merge/split tolerance.

    overlap[i, j] = max(IoA(gt_i, pred_j), IoA(pred_j, gt_i))

    This handles both:
    - Merge: pred fully contains GT → IoA(gt, pred) = 1.0
    - Split: GT fully contains pred → IoA(pred, gt) = 1.0

    :param gt_boxes: Array of shape (N, 4) with GT boxes in xyxy format
    :param pred_boxes: Array of shape (M, 4) with predicted boxes in xyxy format
    :return: Overlap matrix of shape (N, M)
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)))

    gt_boxes = np.asarray(gt_boxes, dtype=float)
    pred_boxes = np.asarray(pred_boxes, dtype=float)

    # Compute areas
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    pred_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])

    # Compute intersection
    lt = np.maximum(gt_boxes[:, None, :2], pred_boxes[None, :, :2])
    rb = np.minimum(gt_boxes[:, None, 2:], pred_boxes[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    # IoA in both directions
    ioa_gt = intersection / np.clip(gt_areas[:, None], 1e-10, None)  # intersection / gt_area
    ioa_pred = intersection / np.clip(pred_areas[None, :], 1e-10, None)  # intersection / pred_area

    return np.maximum(ioa_gt, ioa_pred)  # type: ignore[no-any-return]
