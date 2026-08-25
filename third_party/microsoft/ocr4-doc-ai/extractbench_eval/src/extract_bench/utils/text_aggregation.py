"""Text aggregation utilities for matching PDF cells to layout bounding boxes."""

from collections.abc import Callable


def point_in_bbox(point_x: float, point_y: float, bbox: list[float]) -> bool:
    """Check if point is inside COCO bbox [x, y, w, h].

    :param point_x: X coordinate of the point
    :param point_y: Y coordinate of the point
    :param bbox: Bounding box in COCO format [x, y, width, height]
    :return: True if point is inside the bbox
    """
    x, y, w, h = bbox
    return x <= point_x <= x + w and y <= point_y <= y + h


def match_cell_to_bbox(cell_bbox: list[float], layout_bboxes: list[list[float]]) -> int | None:
    """Find which layout bbox contains the cell center.

    :param cell_bbox: Cell bounding box in COCO format [x, y, w, h]
    :param layout_bboxes: List of layout bounding boxes in COCO format
    :return: Index of matching layout bbox, or None if no match
    """
    cx = cell_bbox[0] + cell_bbox[2] / 2
    cy = cell_bbox[1] + cell_bbox[3] / 2
    for idx, bbox in enumerate(layout_bboxes):
        if point_in_bbox(cx, cy, bbox):
            return idx
    return None


def group_cells_into_lines(cells: list[dict], y_threshold_ratio: float = 0.5) -> list[list[dict]]:
    """Group cells into lines based on y-coordinate proximity.

    :param cells: List of cell dicts with 'bbox' and 'text' keys
    :param y_threshold_ratio: Cells within this fraction of avg cell height are same line
    :return: List of lines, where each line is a list of cells sorted by x-coordinate
    """
    if not cells:
        return []

    sorted_cells = sorted(cells, key=lambda c: (c["bbox"][1], c["bbox"][0]))
    avg_height = sum(c["bbox"][3] for c in sorted_cells) / len(sorted_cells)
    y_threshold = avg_height * y_threshold_ratio

    lines: list[list[dict]] = []
    current_line = [sorted_cells[0]]

    for cell in sorted_cells[1:]:
        last_y = current_line[-1]["bbox"][1]
        curr_y = cell["bbox"][1]

        if abs(curr_y - last_y) <= y_threshold:
            current_line.append(cell)
        else:
            current_line.sort(key=lambda c: c["bbox"][0])
            lines.append(current_line)
            current_line = [cell]

    current_line.sort(key=lambda c: c["bbox"][0])
    lines.append(current_line)
    return lines


def aggregate_text_by_bbox(
    pdf_cells: list[list[dict]],
    layout_bboxes: list[list[float]],
    transform_fn: Callable[[list[float]], list[float]] | None = None,
) -> tuple[dict[int, str], list[dict]]:
    """Aggregate pdf_cells text into layout bboxes.

    :param pdf_cells: Nested list of cell dicts with 'bbox' and 'text' keys
    :param layout_bboxes: List of COCO format bboxes [x, y, w, h]
    :param transform_fn: Optional function to transform cell bbox coordinates
    :return: Tuple of (aggregated_texts dict mapping bbox_idx -> text, unmatched cells list)
    """
    all_cells = [cell for group in pdf_cells for cell in group]

    if transform_fn:
        all_cells = [{**cell, "bbox": transform_fn(cell["bbox"])} for cell in all_cells]

    bbox_cells: dict[int, list[dict]] = {i: [] for i in range(len(layout_bboxes))}
    unmatched: list[dict] = []

    for cell in all_cells:
        match_idx = match_cell_to_bbox(cell["bbox"], layout_bboxes)
        if match_idx is not None:
            bbox_cells[match_idx].append(cell)
        else:
            unmatched.append(cell)

    result: dict[int, str] = {}
    for idx, cells in bbox_cells.items():
        if cells:
            lines = group_cells_into_lines(cells)
            line_texts = [" ".join(c["text"] for c in line) for line in lines]
            result[idx] = "\n".join(line_texts)

    return result, unmatched
