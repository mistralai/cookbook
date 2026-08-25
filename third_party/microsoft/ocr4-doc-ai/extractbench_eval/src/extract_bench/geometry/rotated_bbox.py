"""Canonical literal ``xywh+r`` geometry helpers.

The layout contract is:

* ``x, y, w, h`` describe the literal unrotated rectangle.
* ``r`` is a page/SVG angle in degrees applied around the rectangle center.
* Positive angles are clockwise in page/image coordinates because y increases
  downward.

This module intentionally does not recover a rectangle from an AABB envelope.
Provider polygons are converted directly from their ordered corners.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_MIN_EDGE_LENGTH = 1e-12
_CONTAINMENT_EPSILON = 1e-9


@dataclass(frozen=True)
class LiteralRotatedBox:
    """Literal unrotated rectangle plus optional page/SVG rotation."""

    x: float
    y: float
    w: float
    h: float
    r: float | None


def normalize_angle_degrees(angle: float) -> float:
    """Normalize an angle to ``[-180, 180)`` degrees."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def polygon_angle_degrees(raw_polygon: Any) -> float | None:
    """Return the first non-degenerate edge angle in page/SVG degrees."""
    points = polygon_points(raw_polygon)
    if len(points) < 2:
        return None

    for (x1, y1), (x2, y2) in zip(points, points[1:], strict=False):
        dx = x2 - x1
        dy = y2 - y1
        if math.hypot(dx, dy) <= _MIN_EDGE_LENGTH:
            continue
        return normalize_angle_degrees(math.degrees(math.atan2(dy, dx)))
    return None


def polygon_to_literal_xywh_r(
    raw_polygon: Any,
    *,
    page_width: float,
    page_height: float,
    normalized: bool = True,
) -> LiteralRotatedBox | None:
    """Convert an ordered provider polygon into canonical literal ``xywh+r``.

    The first four points are interpreted as ``top-left, top-right,
    bottom-right, bottom-left`` in the provider's page coordinate system.
    """
    points = polygon_points(raw_polygon)
    if len(points) < 4:
        return None
    if page_width <= 0 or page_height <= 0:
        return None

    p0, p1, p2, p3 = points[:4]
    width = (_distance(p0, p1) + _distance(p2, p3)) / 2.0
    height = (_distance(p1, p2) + _distance(p3, p0)) / 2.0
    if width <= _MIN_EDGE_LENGTH or height <= _MIN_EDGE_LENGTH:
        return None

    center_x = (p0[0] + p1[0] + p2[0] + p3[0]) / 4.0
    center_y = (p0[1] + p1[1] + p2[1] + p3[1]) / 4.0
    x = center_x - width / 2.0
    y = center_y - height / 2.0
    r = normalize_angle_degrees(math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])))

    if normalized:
        return LiteralRotatedBox(
            x=x / page_width,
            y=y / page_height,
            w=width / page_width,
            h=height / page_height,
            r=r,
        )
    return LiteralRotatedBox(x=x, y=y, w=width, h=height, r=r)


def xywh_r_to_polygon(
    x: float,
    y: float,
    w: float,
    h: float,
    r: float | None,
    *,
    page_width: float = 1.0,
    page_height: float = 1.0,
    normalized: bool = True,
) -> list[tuple[float, float]]:
    """Return the four rotated corners for a canonical literal box."""
    if page_width <= 0 or page_height <= 0:
        raise ValueError("page_width and page_height must be positive")

    if normalized:
        x_px = float(x) * page_width
        y_px = float(y) * page_height
        w_px = float(w) * page_width
        h_px = float(h) * page_height
    else:
        x_px = float(x)
        y_px = float(y)
        w_px = float(w)
        h_px = float(h)

    cx = x_px + w_px / 2.0
    cy = y_px + h_px / 2.0
    angle = 0.0 if r is None else math.radians(float(r))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    corners = [
        (-w_px / 2.0, -h_px / 2.0),
        (w_px / 2.0, -h_px / 2.0),
        (w_px / 2.0, h_px / 2.0),
        (-w_px / 2.0, h_px / 2.0),
    ]
    polygon_px = [(cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a) for dx, dy in corners]
    if not normalized:
        return polygon_px
    return [(px / page_width, py / page_height) for px, py in polygon_px]


def rotated_rect_contains_point(
    box_xywh: Sequence[float],
    r: float | None,
    point_xy: tuple[float, float],
    *,
    page_width: float = 1.0,
    page_height: float = 1.0,
    normalized: bool = True,
) -> bool:
    """Return whether a point falls inside a literal rotated rectangle."""
    if len(box_xywh) < 4 or page_width <= 0 or page_height <= 0:
        return False

    try:
        x, y, w, h = (float(value) for value in box_xywh[:4])
        point_x, point_y = float(point_xy[0]), float(point_xy[1])
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (x, y, w, h, point_x, point_y)):
        return False
    if w < 0 or h < 0:
        return False

    if normalized:
        x *= page_width
        w *= page_width
        point_x *= page_width
        y *= page_height
        h *= page_height
        point_y *= page_height

    cx = x + w / 2.0
    cy = y + h / 2.0
    dx = point_x - cx
    dy = point_y - cy
    angle = 0.0 if r is None else math.radians(float(r))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a

    return abs(local_x) <= w / 2.0 + _CONTAINMENT_EPSILON and abs(local_y) <= h / 2.0 + _CONTAINMENT_EPSILON


def polygon_points(raw_polygon: Any) -> list[tuple[float, float]]:
    """Coerce flat or nested polygon coordinates into finite point tuples."""
    if not isinstance(raw_polygon, list | tuple):
        return []
    if not raw_polygon:
        return []

    first = raw_polygon[0]
    if isinstance(first, list | tuple):
        points: list[tuple[float, float]] = []
        for point in raw_polygon:
            if not isinstance(point, list | tuple) or len(point) < 2:
                return []
            coerced = _finite_point(point[0], point[1])
            if coerced is None:
                return []
            points.append(coerced)
        return points

    if len(raw_polygon) < 4:
        return []
    points = []
    for index in range(0, len(raw_polygon) - 1, 2):
        coerced = _finite_point(raw_polygon[index], raw_polygon[index + 1])
        if coerced is None:
            return []
        points.append(coerced)
    return points


def _finite_point(raw_x: Any, raw_y: Any) -> tuple[float, float] | None:
    try:
        x = float(raw_x)
        y = float(raw_y)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return (x, y)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])
