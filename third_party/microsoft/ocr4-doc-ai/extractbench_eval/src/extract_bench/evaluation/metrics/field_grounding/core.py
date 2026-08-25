"""Formula-only helpers for field value and bbox grounding metrics."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, cast

from dateutil import parser as date_parser
from rapidfuzz.distance import JaroWinkler

STRING_MATCH_THRESHOLD = 0.90
NUMERIC_ABSOLUTE_TOLERANCE = 1e-6
NUMERIC_RELATIVE_TOLERANCE = 1e-6
FIELD_GROUNDING_STRICT_IOU_THRESHOLD = 0.50
FIELD_GROUNDING_RELAXED_IOU_THRESHOLD = 0.30
FIELD_GROUNDING_RELAXED_MAX_IOA_THRESHOLD = 0.70
FIELD_GROUNDING_CANONICAL_EXACT_SCORE_THRESHOLD = 0.999

_IGNORED_INVISIBLE_CODEPOINTS = {
    0x00AD,  # soft hyphen
    0x200B,  # zero width space
    0x2060,  # word joiner
    0xFEFF,  # zero width no-break space / BOM
}
_TRUE_STRINGS = frozenset({"true", "yes", "y", "1", "checked"})
_FALSE_STRINGS = frozenset({"false", "no", "n", "0", "unchecked"})
_DATE_PATTERNS = (
    re.compile(r"\d{4}-\d{1,2}-\d{1,2}"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}"),
    re.compile(r"\d{1,2}-\d{1,2}-\d{2,4}"),
    # Optional day-of-week prefix + month, both tolerating a trailing period —
    # covers "Mon. Jan. 02 2023", "Monday January 2, 2023", "Jan 02 2023".
    re.compile(r"(?:[A-Za-z]{3,9}\.?\s+)?[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}"),
    re.compile(r"\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4}"),
)


@dataclass(frozen=True)
class ValueComparison:
    """Result of comparing one GT field value against one prediction."""

    passed: bool
    score: float
    mode: str
    reason: str


@dataclass(frozen=True)
class BBox:
    """One normalized COCO bbox attached to a page and optional field group."""

    page: int
    bbox: tuple[float, float, float, float]
    group: str | None = None


@dataclass(frozen=True)
class BBoxMetrics:
    """Continuous bbox grounding scores plus raw area metadata."""

    iou: float
    bbox_recall: float
    gt_area: float
    best_intersection_area: float
    covered_gt_area: float


@dataclass(frozen=True)
class StandardIoUMetrics:
    """Standard set IoU over the union of GT and predicted rectangles."""

    iou: float
    gt_area: float
    pred_area: float
    intersection_area: float
    union_area: float


def normalize_text(text: Any) -> str:
    """Normalize text for OCR-tolerant comparison without dropping visible glyphs."""
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", str(text))
    chars: list[str] = []
    for char in normalized:
        if ord(char) in _IGNORED_INVISIBLE_CODEPOINTS:
            continue
        if char.isspace():
            chars.append(" ")
            continue
        if unicodedata.category(char) == "Cc":
            continue
        chars.append(char)
    return " ".join("".join(chars).split()).casefold().strip()


def compare_field_value(expected: Any, actual: Any) -> ValueComparison:
    """Compare field values with customer-compatible typed semantics."""
    if expected is None:
        passed = actual is None or normalize_text(actual) == ""
        return ValueComparison(passed=passed, score=1.0 if passed else 0.0, mode="null", reason=_reason(passed, "null"))

    if isinstance(expected, bool):
        expected_bool = expected
        actual_bool = _parse_bool(actual)
        passed = actual_bool is not None and expected_bool is actual_bool
        return ValueComparison(
            passed=passed,
            score=1.0 if passed else 0.0,
            mode="boolean",
            reason=_reason(passed, "boolean_mismatch"),
        )

    if isinstance(expected, int) and not isinstance(expected, bool):
        actual_number = _parse_number(actual)
        passed = actual_number is not None and _is_integer_like(actual_number) and int(round(actual_number)) == expected
        return ValueComparison(
            passed=passed,
            score=1.0 if passed else 0.0,
            mode="integer",
            reason=_reason(passed, "integer_mismatch"),
        )

    if isinstance(expected, float):
        actual_number = _parse_number(actual)
        passed = actual_number is not None and math.isclose(
            float(expected),
            actual_number,
            rel_tol=NUMERIC_RELATIVE_TOLERANCE,
            abs_tol=NUMERIC_ABSOLUTE_TOLERANCE,
        )
        return ValueComparison(
            passed=passed,
            score=1.0 if passed else 0.0,
            mode="number",
            reason=_reason(passed, "number_mismatch"),
        )

    expected_date = _parse_date(expected)
    actual_date = _parse_date(actual)
    if expected_date is not None and actual_date is not None:
        passed = expected_date == actual_date
        return ValueComparison(
            passed=passed,
            score=1.0 if passed else 0.0,
            mode="date",
            reason=_reason(passed, "date_mismatch"),
        )

    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    score = float(JaroWinkler.normalized_similarity(expected_norm, actual_norm))
    passed = score >= STRING_MATCH_THRESHOLD
    return ValueComparison(
        passed=passed,
        score=score,
        mode="jaro_winkler",
        reason=_reason(passed, "jaro_winkler_below_threshold"),
    )


def compute_bbox_metrics(gt_boxes: list[BBox], pred_boxes: list[BBox]) -> BBoxMetrics:
    """Compute field grounding IoU and bbox recall with page/group scoping."""
    valid_gt = [box for box in gt_boxes if _valid_xywh(box.bbox)]
    valid_pred = [box for box in pred_boxes if _valid_xywh(box.bbox)]
    gt_area = sum(_area_xywh(box.bbox) for box in valid_gt)
    if gt_area <= 0.0:
        return BBoxMetrics(iou=0.0, bbox_recall=0.0, gt_area=0.0, best_intersection_area=0.0, covered_gt_area=0.0)

    best_intersection_area = 0.0
    for gt in valid_gt:
        scoped_preds = [pred for pred in valid_pred if _same_scope(gt, pred)]
        best_intersection_area += max(
            (_intersection_area_xywh(gt.bbox, pred.bbox) for pred in scoped_preds),
            default=0.0,
        )

    covered_gt_area = 0.0
    scopes = {(box.page, box.group) for box in valid_gt}
    for page, group in scopes:
        scope_gt = [box for box in valid_gt if box.page == page and box.group == group]
        scope_pred = [box for box in valid_pred if box.page == page and box.group == group]
        clipped: list[tuple[float, float, float, float]] = []
        for gt in scope_gt:
            gt_xyxy = _xywh_to_xyxy(gt.bbox)
            for pred in scope_pred:
                if (intersection := _intersect_xyxy(gt_xyxy, _xywh_to_xyxy(pred.bbox))) is not None:
                    clipped.append(intersection)
        covered_gt_area += _rect_union_area(clipped)

    return BBoxMetrics(
        iou=best_intersection_area / gt_area,
        bbox_recall=covered_gt_area / gt_area,
        gt_area=gt_area,
        best_intersection_area=best_intersection_area,
        covered_gt_area=covered_gt_area,
    )


def compute_standard_iou_metrics(gt_boxes: list[BBox], pred_boxes: list[BBox]) -> StandardIoUMetrics:
    """Compute standard IoU between GT and predicted bbox sets.

    Rectangles are scoped by page and group. Within each scope, GT boxes and
    predicted boxes are independently unioned before intersection/union area
    are accumulated. This differs from :func:`compute_bbox_metrics`, whose
    historic ``iou`` field is GT-coverage shaped.
    """
    valid_gt = [box for box in gt_boxes if _valid_xywh(box.bbox)]
    valid_pred = [box for box in pred_boxes if _valid_xywh(box.bbox)]
    scopes = {(box.page, box.group) for box in valid_gt} | {(box.page, box.group) for box in valid_pred}

    gt_area = 0.0
    pred_area = 0.0
    intersection_area = 0.0
    for page, group in scopes:
        scope_gt = [box for box in valid_gt if box.page == page and box.group == group]
        scope_pred = [box for box in valid_pred if box.page == page and box.group == group]
        gt_rects = [_xywh_to_xyxy(box.bbox) for box in scope_gt]
        pred_rects = [_xywh_to_xyxy(box.bbox) for box in scope_pred]

        gt_area += _rect_union_area(gt_rects)
        pred_area += _rect_union_area(pred_rects)

        intersections: list[tuple[float, float, float, float]] = []
        for gt_rect in gt_rects:
            for pred_rect in pred_rects:
                if (intersection := _intersect_xyxy(gt_rect, pred_rect)) is not None:
                    intersections.append(intersection)
        intersection_area += _rect_union_area(intersections)

    union_area = gt_area + pred_area - intersection_area
    iou = intersection_area / union_area if union_area > 0.0 else 0.0
    return StandardIoUMetrics(
        iou=iou,
        gt_area=gt_area,
        pred_area=pred_area,
        intersection_area=intersection_area,
        union_area=union_area,
    )


def field_grounding_max_ioa(summary: StandardIoUMetrics) -> float:
    """Return the best directional intersection-over-area for a set IoU summary."""
    gt_ioa = summary.intersection_area / summary.gt_area if summary.gt_area > 0.0 else 0.0
    pred_ioa = summary.intersection_area / summary.pred_area if summary.pred_area > 0.0 else 0.0
    return max(gt_ioa, pred_ioa)


def field_grounding_has_canonical_exact_text_match(comparison: ValueComparison | None) -> bool:
    """True only for typed exact/canonical equivalences, not fuzzy string passes."""
    return bool(
        comparison is not None
        and comparison.passed
        and comparison.score >= FIELD_GROUNDING_CANONICAL_EXACT_SCORE_THRESHOLD
    )


def field_grounding_has_null_empty_match(comparison: ValueComparison | None) -> bool:
    """True when attribution verifies a visual dash/blank/null placeholder."""
    return bool(comparison is not None and comparison.passed and comparison.mode == "null_empty")


def field_grounding_localization_passes(
    *,
    iou: float,
    max_ioa: float,
    comparison: ValueComparison | None,
    strict_iou_threshold: float | None = None,
) -> bool:
    """Evaluate strict-or-relaxed field localization semantics.

    The relaxed branch is reserved for small granularity mismatches: it still
    requires meaningful overlap and an exact typed text/value match.

    ``strict_iou_threshold`` overrides only the strict branch (default 0.5). The
    relaxed value-conditioned branch (IoU>=0.3 AND IoA>=0.7 AND canonical_exact)
    is unaffected by the override — it's a fallback escape hatch for granularity
    mismatches and is intentionally not tunable per rule.
    """
    strict = FIELD_GROUNDING_STRICT_IOU_THRESHOLD if strict_iou_threshold is None else strict_iou_threshold
    if iou >= strict:
        return True
    if field_grounding_has_null_empty_match(comparison):
        return True
    return (
        iou >= FIELD_GROUNDING_RELAXED_IOU_THRESHOLD
        and max_ioa >= FIELD_GROUNDING_RELAXED_MAX_IOA_THRESHOLD
        and field_grounding_has_canonical_exact_text_match(comparison)
    )


def field_grounding_localization_reason(
    *,
    iou: float,
    max_ioa: float,
    comparison: ValueComparison | None,
    strict_iou_threshold: float | None = None,
) -> str:
    strict = FIELD_GROUNDING_STRICT_IOU_THRESHOLD if strict_iou_threshold is None else strict_iou_threshold
    if iou >= strict:
        return "pass"
    if field_grounding_has_null_empty_match(comparison):
        return "pass_null_empty_overlap" if max_ioa > 0.0 else "pass_null_empty_no_support"
    if field_grounding_localization_passes(
        iou=iou,
        max_ioa=max_ioa,
        comparison=comparison,
        strict_iou_threshold=strict_iou_threshold,
    ):
        return "pass_relaxed_iou_canonical_exact"
    return "iou_below_threshold"


def field_iou(gt_boxes: list[BBox], pred_boxes: list[BBox]) -> float:
    """Return the customer-spec field grounding IoU score."""
    return compute_bbox_metrics(gt_boxes, pred_boxes).iou


def bbox_recall(gt_boxes: list[BBox], pred_boxes: list[BBox]) -> float:
    """Return the customer-spec field grounding bbox recall score."""
    return compute_bbox_metrics(gt_boxes, pred_boxes).bbox_recall


def _reason(passed: bool, failure_reason: str) -> str:
    return "pass" if passed else failure_reason


def _parse_bool(value: Any) -> bool | None:
    normalized = normalize_text(value)
    if normalized in _TRUE_STRINGS:
        return True
    if normalized in _FALSE_STRINGS:
        return False
    return None


def _parse_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    normalized = normalize_text(value)
    if not normalized:
        return None

    negative = False
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1].strip()
        negative = True

    normalized = re.sub(r"^[~≈]", "", normalized).strip()
    normalized = re.sub(r"^[$€£¥₹]\s*", "", normalized)
    normalized = re.sub(r"\s*[$€£¥₹]$", "", normalized)
    normalized = normalized.rstrip("%")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace(" ", "")

    try:
        parsed = float(normalized)
    except ValueError:
        return None
    return -parsed if negative else parsed


def _is_integer_like(value: float) -> bool:
    return math.isclose(value, round(value), abs_tol=NUMERIC_ABSOLUTE_TOLERANCE)


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    normalized = normalize_text(value)
    if not normalized or not any(pattern.search(normalized) for pattern in _DATE_PATTERNS):
        return None
    try:
        parsed = cast(datetime, date_parser.parse(normalized, fuzzy=False))
    except (ValueError, OverflowError, TypeError):
        return None
    return parsed.date()


def _same_scope(a: BBox, b: BBox) -> bool:
    return a.page == b.page and a.group == b.group


def _valid_xywh(bbox: tuple[float, float, float, float]) -> bool:
    return len(bbox) == 4 and bbox[2] > 0.0 and bbox[3] > 0.0


def _area_xywh(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2]) * max(0.0, bbox[3])


def _xywh_to_xyxy(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])


def _intersection_area_xywh(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    intersection = _intersect_xyxy(_xywh_to_xyxy(a), _xywh_to_xyxy(b))
    if intersection is None:
        return 0.0
    return _area_xyxy(intersection)


def _intersect_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _area_xyxy(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _rect_union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    if not rectangles:
        return 0.0

    xs = sorted({coord for rect in rectangles for coord in (rect[0], rect[2])})
    ys = sorted({coord for rect in rectangles for coord in (rect[1], rect[3])})
    total = 0.0
    for left, right in zip(xs, xs[1:], strict=False):
        if right <= left:
            continue
        for top, bottom in zip(ys, ys[1:], strict=False):
            if bottom <= top:
                continue
            if any(
                rect[0] <= left and rect[2] >= right and rect[1] <= top and rect[3] >= bottom for rect in rectangles
            ):
                total += (right - left) * (bottom - top)
    return total
