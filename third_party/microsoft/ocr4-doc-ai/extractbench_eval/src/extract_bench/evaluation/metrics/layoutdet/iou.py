"""IoU (Intersection over Union) computation utilities for layout detection."""

import numpy as np


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """
    Compute IoU between two boxes in xyxy format [x1, y1, x2, y2].

    :param box1: First bounding box [x1, y1, x2, y2]
    :param box2: Second bounding box [x1, y1, x2, y2]
    :return: IoU value between 0.0 and 1.0
    """
    # Determine intersection coordinates
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute intersection area
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def compute_iou_matrix(
    boxes1: np.ndarray,  # shape (N, 4)
    boxes2: np.ndarray,  # shape (M, 4)
) -> np.ndarray:  # shape (N, M)
    """
    Compute pairwise IoU matrix between two sets of boxes (vectorized).

    Both box sets should be in xyxy format [x1, y1, x2, y2].

    :param boxes1: Array of shape (N, 4) with N bounding boxes
    :param boxes2: Array of shape (M, 4) with M bounding boxes
    :return: IoU matrix of shape (N, M)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute intersection
    # boxes1[:, None, :2] has shape (N, 1, 2), boxes2[None, :, :2] has shape (1, M, 2)
    # Result has shape (N, M, 2)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # left-top
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # right-bottom
    wh = np.clip(rb - lt, 0, None)  # width-height, clipped to non-negative
    intersection = wh[:, :, 0] * wh[:, :, 1]

    # Compute union
    # area1[:, None] has shape (N, 1), area2[None, :] has shape (1, M)
    union = area1[:, None] + area2[None, :] - intersection

    return intersection / np.clip(union, 1e-10, None)  # type: ignore[no-any-return]


def coco_to_xyxy(bbox: list[float]) -> list[float]:
    """
    Convert COCO format bbox [x, y, width, height] to xyxy format [x1, y1, x2, y2].

    :param bbox: Bounding box in COCO format [x, y, width, height]
    :return: Bounding box in xyxy format [x1, y1, x2, y2]
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]
