"""Canonical geometry helpers for parse provider normalization.

Layout ``bbox+r`` means a literal unrotated ``xywh`` rectangle with ``r`` as
the page/SVG clockwise rotation applied around the rectangle center.
"""

from __future__ import annotations

from extract_bench.geometry.rotated_bbox import (
    LiteralRotatedBox,
    normalize_angle_degrees,
    polygon_angle_degrees,
    polygon_points,
    polygon_to_literal_xywh_r,
    rotated_rect_contains_point,
    xywh_r_to_polygon,
)

__all__ = [
    "LiteralRotatedBox",
    "normalize_angle_degrees",
    "polygon_angle_degrees",
    "polygon_points",
    "polygon_to_literal_xywh_r",
    "rotated_rect_contains_point",
    "xywh_r_to_polygon",
]
