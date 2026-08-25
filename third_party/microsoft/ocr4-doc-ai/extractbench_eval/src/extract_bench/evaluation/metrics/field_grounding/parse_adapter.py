"""Field grounding metrics for parse pipeline outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from extract_bench.evaluation.layout_adapters import create_layout_adapter_for_result
from extract_bench.evaluation.metrics.field_grounding.core import (
    FIELD_GROUNDING_CANONICAL_EXACT_SCORE_THRESHOLD,
    FIELD_GROUNDING_RELAXED_IOU_THRESHOLD,
    FIELD_GROUNDING_RELAXED_MAX_IOA_THRESHOLD,
    FIELD_GROUNDING_STRICT_IOU_THRESHOLD,
    BBox,
    ValueComparison,
    compare_field_value,
    compute_bbox_metrics,
    compute_standard_iou_metrics,
    field_grounding_has_canonical_exact_text_match,
    field_grounding_localization_passes,
    field_grounding_localization_reason,
    field_grounding_max_ioa,
    normalize_text,
)
from extract_bench.evaluation.metrics.field_grounding.value_compare import (
    COMPARATOR_VERSION,
    ExpectedType,
    compare_attributed_value,
    expected_type_for_field_path,
)
from extract_bench.schemas.evaluation import MetricValue
from extract_bench.schemas.parse_output import LayoutSegmentIR, ParseLayoutPageIR, ParseOutput
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.test_cases.schema import ExtractFieldTestRule

EXTRACT_FIELD_LOCALIZATION_IOU_THRESHOLD = FIELD_GROUNDING_STRICT_IOU_THRESHOLD


@dataclass(frozen=True)
class _SupportUnit:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    order_index: int
    granularity: str  # "word" | "line"


@dataclass(frozen=True)
class _SupportMatch:
    comparison: ValueComparison
    boxes: tuple[BBox, ...]
    iou: float  # compute_standard_iou_metrics(rule_gts, boxes).iou
    max_ioa: float
    granularity: str  # "word" | "line"
    units: tuple[_SupportUnit, ...]  # winning candidate group (source of matched_pred_text)


def compute_parse_field_grounding_metrics(
    *,
    inference_result: InferenceResult,
    field_rules: list[ExtractFieldTestRule],
    data_schema: dict[str, Any] | None = None,
) -> list[MetricValue]:
    """Compute the 9-metric extract_field_* attribution taxonomy for parse outputs."""
    if not field_rules or not isinstance(inference_result.output, ParseOutput):
        return []

    support_sets = _build_support_sets(inference_result)
    ungrounded_sources = _build_ungrounded_text_sources(inference_result)
    value_rules = [rule for rule in field_rules if not _is_stray_rule(rule)]

    loc_passes = 0
    cls_passes = 0  # trivial: equals len(value_rules)
    attr_passes = 0
    element_passes = 0
    verified_total = 0
    verified_loc_passes = 0
    verified_cls_passes = 0
    verified_attr_passes = 0
    verified_element_passes = 0
    text_sim_sum = 0.0
    string_rule_count = 0
    iou_sum = 0.0
    matched_iou_sum = 0.0
    unmatched_iou_sum = 0.0
    granularity_mix: dict[str, int] = {"word": 0, "line": 0, "none": 0}
    pred_boxes: list[BBox] = []
    rule_results: list[dict[str, Any]] = []

    for rule in value_rules:
        expected_type = expected_type_for_field_path(data_schema, rule.field_path, rule.expected_value)
        match = _select_best_match(rule, support_sets, expected_type=expected_type)
        ungrounded_source = _find_ungrounded_text_source(rule.expected_value, ungrounded_sources)

        if match is not None:
            pred_boxes.extend(match.boxes)
            granularity_mix[match.granularity] = granularity_mix.get(match.granularity, 0) + 1
            loc_pass = field_grounding_localization_passes(
                iou=match.iou,
                max_ioa=match.max_ioa,
                comparison=match.comparison,
            )
        else:
            granularity_mix["none"] += 1
            loc_pass = False

        cls_pass = True  # trivial — parse field rules have no class label
        attr_pass = loc_pass and match is not None and match.comparison.passed
        element_pass = loc_pass and cls_pass and attr_pass

        loc_passes += int(loc_pass)
        cls_passes += 1
        attr_passes += int(attr_pass)
        element_passes += int(element_pass)
        if rule.verified:
            verified_total += 1
            verified_loc_passes += int(loc_pass)
            verified_cls_passes += int(cls_pass)
            verified_attr_passes += int(attr_pass)
            verified_element_passes += int(element_pass)
        iou = match.iou if match else 0.0
        iou_sum += iou
        if loc_pass:
            matched_iou_sum += iou
        else:
            unmatched_iou_sum += iou

        if expected_type == "string" and match is not None:
            text_sim_sum += match.comparison.score
            string_rule_count += 1

        # Derive a localization reason so the viz can distinguish between
        # "no candidate ever landed near the GT bbox" and "candidate landed
        # but overlapped poorly".
        if not loc_pass and ungrounded_source is not None:
            localization_reason = "text_present_but_ungrounded"
        elif match is None:
            localization_reason = "no_support_match"
        elif loc_pass:
            localization_reason = field_grounding_localization_reason(
                iou=match.iou,
                max_ioa=match.max_ioa,
                comparison=match.comparison,
            )
        else:
            localization_reason = "iou_below_threshold"

        rule_results.append(
            {
                "field_path": rule.field_path,
                "verified": rule.verified,
                "loc_pass": loc_pass,
                "cls_pass": cls_pass,
                "attr_pass": attr_pass,
                "element_pass": element_pass,
                "granularity": match.granularity if match else "none",
                "iou": iou,
                "max_ioa": match.max_ioa if match else 0.0,
                "score": match.comparison.score if match else 0.0,
                "mode": match.comparison.mode if match else "missing",
                "reason": _rule_reason(match, loc_pass, ungrounded_source=ungrounded_source),
                "expected_type": expected_type,
                "attr_source": "selected_support_text" if match else "none",
                "comparator_version": COMPARATOR_VERSION,
                "canonical_exact": (
                    field_grounding_has_canonical_exact_text_match(match.comparison) if match else False
                ),
                "localization_reason": localization_reason,
                "ungrounded_text_source": ungrounded_source[:200] if ungrounded_source is not None else None,
                "matched_pred_bboxes": [list(b.bbox) for b in match.boxes] if match else [],
                "matched_pred_text": (" ".join(u.text for u in match.units) if match else ""),
            }
        )

    total = len(value_rules)
    gt_boxes_all = _rule_gt_boxes(value_rules)
    metrics: list[MetricValue] = []

    if total == 0:
        return metrics

    unmatched = total - loc_passes
    avg_iou_meta = {
        "total": total,
        "matched": loc_passes,
        "unmatched": unmatched,
        "iou_threshold": FIELD_GROUNDING_STRICT_IOU_THRESHOLD,
        "relaxed_iou_threshold": FIELD_GROUNDING_RELAXED_IOU_THRESHOLD,
        "relaxed_max_ioa_threshold": FIELD_GROUNDING_RELAXED_MAX_IOA_THRESHOLD,
        "canonical_exact_score_threshold": FIELD_GROUNDING_CANONICAL_EXACT_SCORE_THRESHOLD,
    }
    rule_meta = {
        "gt_count": total,
        "rule_results": rule_results,
        "granularity_mix": granularity_mix,
        "verified_total": verified_total,
    }

    metrics.extend(
        [
            MetricValue(
                metric_name="extract_field_element_pass_rate",
                value=element_passes / total,
                metadata={
                    **rule_meta,
                    "passed": element_passes,
                    "total": total,
                    "verified_passed": verified_element_passes,
                },
            ),
            MetricValue(
                metric_name="extract_field_rule_pass_rate",
                value=(loc_passes + cls_passes + attr_passes) / (3 * total),
                metadata={
                    "passed": loc_passes + cls_passes + attr_passes,
                    "loc_passed": loc_passes,
                    "cls_passed": cls_passes,
                    "attr_passed": attr_passes,
                    "verified_passed": verified_loc_passes + verified_cls_passes + verified_attr_passes,
                    "verified_loc_passed": verified_loc_passes,
                    "verified_cls_passed": verified_cls_passes,
                    "verified_attr_passed": verified_attr_passes,
                    "verified_total": 3 * verified_total,
                    "total": 3 * total,
                },
            ),
            MetricValue(
                metric_name="extract_field_localization_pass_rate",
                value=loc_passes / total,
                metadata={
                    "passed": loc_passes,
                    "verified_passed": verified_loc_passes,
                    "verified_total": verified_total,
                    "total": total,
                    "iou_threshold": FIELD_GROUNDING_STRICT_IOU_THRESHOLD,
                    "relaxed_iou_threshold": FIELD_GROUNDING_RELAXED_IOU_THRESHOLD,
                    "relaxed_max_ioa_threshold": FIELD_GROUNDING_RELAXED_MAX_IOA_THRESHOLD,
                    "canonical_exact_score_threshold": FIELD_GROUNDING_CANONICAL_EXACT_SCORE_THRESHOLD,
                },
            ),
            MetricValue(
                metric_name="extract_field_classification_pass_rate",
                value=1.0,
                metadata={
                    "passed": cls_passes,
                    "verified_passed": verified_cls_passes,
                    "verified_total": verified_total,
                    "total": total,
                },
            ),
            MetricValue(
                metric_name="extract_field_attribution_pass_rate",
                value=attr_passes / total,
                metadata={
                    "passed": attr_passes,
                    "verified_passed": verified_attr_passes,
                    "verified_total": verified_total,
                    "total": total,
                },
            ),
            MetricValue(
                metric_name="extract_field_avg_iou",
                value=iou_sum / total,
                metadata=avg_iou_meta,
            ),
            MetricValue(
                metric_name="extract_field_avg_iou_matched",
                value=matched_iou_sum / loc_passes if loc_passes > 0 else 0.0,
                metadata=avg_iou_meta,
            ),
            MetricValue(
                metric_name="extract_field_avg_iou_unmatched",
                value=unmatched_iou_sum / unmatched if unmatched > 0 else 0.0,
                metadata=avg_iou_meta,
            ),
        ]
    )

    if gt_boxes_all:
        summary = compute_standard_iou_metrics(gt_boxes_all, pred_boxes)
        recall_summary = compute_bbox_metrics(gt_boxes_all, pred_boxes)
        bbox_meta = {
            "gt_count": len(gt_boxes_all),
            "pred_count": len(pred_boxes),
            "gt_area": summary.gt_area,
            "pred_area": summary.pred_area,
            "intersection_area": summary.intersection_area,
            "union_area": summary.union_area,
        }
        metrics.extend(
            [
                MetricValue(metric_name="extract_field_iou", value=summary.iou, metadata=bbox_meta),
                MetricValue(
                    metric_name="extract_field_bbox_recall",
                    value=recall_summary.bbox_recall,
                    metadata={
                        **bbox_meta,
                        "covered_gt_area": recall_summary.covered_gt_area,
                    },
                ),
            ]
        )

    if string_rule_count > 0:
        metrics.append(
            MetricValue(
                metric_name="extract_field_text_similarity",
                value=text_sim_sum / string_rule_count,
                metadata={"string_rule_count": string_rule_count, "total_rule_count": total},
            )
        )

    metrics.append(
        MetricValue(
            metric_name="extract_field_gt_count",
            value=float(total),
            metadata={"granularity_mix": granularity_mix},
        )
    )

    return metrics


def _is_string_expected(value: Any) -> bool:
    return isinstance(value, str) and not isinstance(value, bool)


def _build_support_sets(inference_result: InferenceResult) -> list[list[_SupportUnit]]:
    word_units, line_units = _adapter_units(inference_result)
    layout_text_units = _layout_text_units(
        inference_result.output.layout_pages if isinstance(inference_result.output, ParseOutput) else []
    )
    return [units for units in (word_units, line_units, layout_text_units) if units]


def _adapter_units(inference_result: InferenceResult) -> tuple[list[_SupportUnit], list[_SupportUnit]]:
    try:
        adapter = create_layout_adapter_for_result(inference_result)
        to_granular_pages = getattr(adapter, "to_granular_pages", None)
        if not callable(to_granular_pages):
            return [], []
        granular_pages = to_granular_pages(inference_result)
    except Exception:
        return [], []

    words: list[_SupportUnit] = []
    lines: list[_SupportUnit] = []
    for page in granular_pages:
        page_number = int(getattr(page, "page_number", 0) or 0)
        for bucket_name, bucket, granularity in (
            ("words", words, "word"),
            ("lines", lines, "line"),
        ):
            for order_index, unit in enumerate(getattr(page, bucket_name, []) or []):
                bbox = getattr(unit, "bbox", None)
                text = str(getattr(unit, "text", "") or "")
                if bbox is None or not text.strip():
                    continue
                bucket.append(
                    _SupportUnit(
                        page=page_number,
                        text=text,
                        bbox=(float(bbox.x), float(bbox.y), float(bbox.w), float(bbox.h)),
                        order_index=int(getattr(unit, "order_index", order_index) or order_index),
                        granularity=granularity,
                    )
                )
    return words, lines


def _layout_text_units(layout_pages: list[ParseLayoutPageIR]) -> list[_SupportUnit]:
    units: list[_SupportUnit] = []
    for page in layout_pages:
        already_normalized = _page_bboxes_are_normalized(page)
        width = page.width or 0.0
        height = page.height or 0.0
        for order_index, item in enumerate(page.items):
            if item.type.casefold() not in {"text", "line", "word"}:
                continue
            text = item.value or item.md or item.html
            if not text.strip():
                continue
            segments = item.layout_segments if item.layout_segments else ([item.bbox] if item.bbox is not None else [])
            for segment in segments:
                bbox = _segment_to_normalized_xywh(
                    segment, width=width, height=height, already_normalized=already_normalized
                )
                if bbox is not None:
                    units.append(
                        _SupportUnit(
                            page=page.page_number,
                            text=text,
                            bbox=bbox,
                            order_index=order_index,
                            granularity="line",
                        )
                    )
    return units


def _build_ungrounded_text_sources(inference_result: InferenceResult) -> list[str]:
    if not isinstance(inference_result.output, ParseOutput):
        return []

    sources: list[str] = []
    for page_payload in inference_result.output.grounded_pages:
        if isinstance(page_payload, dict):
            sources.extend(_collect_ungrounded_text_sources(page_payload.get("items")))
    return sources


def _collect_ungrounded_text_sources(raw_items: Any) -> list[str]:
    if not isinstance(raw_items, list):
        return []

    sources: list[str] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        grounding = item.get("grounding")
        if isinstance(grounding, dict):
            source_rows = item.get("rows")
            grounded_rows = grounding.get("rows")
            if isinstance(source_rows, list) and isinstance(grounded_rows, list):
                for source_row, grounded_row in zip(source_rows, grounded_rows, strict=False):
                    if not isinstance(source_row, list) or not isinstance(grounded_row, list):
                        continue
                    for source_cell, grounded_cell in zip(source_row, grounded_row, strict=False):
                        source_text = _coerce_source_text(source_cell)
                        if source_text and not _has_grounding_geometry(grounded_cell):
                            sources.append(source_text)

        child_items = item.get("items")
        if isinstance(child_items, list):
            sources.extend(_collect_ungrounded_text_sources(child_items))

    return sources


def _coerce_source_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("value", "md", "text", "html"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def _has_grounding_geometry(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if isinstance(value.get("bbox"), dict):
        return True
    lines = value.get("lines")
    if not isinstance(lines, list):
        return False
    for line in lines:
        if not isinstance(line, dict):
            continue
        if isinstance(line.get("bbox"), dict):
            return True
        words = line.get("words")
        if isinstance(words, list) and any(
            isinstance(word, dict) and isinstance(word.get("bbox"), dict) for word in words
        ):
            return True
    return False


def _find_ungrounded_text_source(expected: Any, sources: list[str]) -> str | None:
    if not sources:
        return None
    expected_norm = normalize_text(expected)
    if not expected_norm:
        return None

    expected_tokens = _meaningful_tokens(expected_norm)
    for source in sources:
        source_norm = normalize_text(source)
        if not source_norm:
            continue
        if expected_norm in source_norm or source_norm in expected_norm:
            return source
        if compare_field_value(expected, source).score >= 0.90:
            return source
        source_tokens = _meaningful_tokens(source_norm)
        if expected_tokens and _token_coverage(expected_tokens, source_tokens) >= 0.80:
            return source
    return None


def _meaningful_tokens(value: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", value.casefold()))
    return {token for token in tokens if len(token) > 1 or token.isdigit()}


def _token_coverage(expected_tokens: set[str], source_tokens: set[str]) -> float:
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & source_tokens) / len(expected_tokens)


def _select_best_match(
    rule: ExtractFieldTestRule,
    support_sets: list[list[_SupportUnit]],
    *,
    expected_type: ExpectedType,
) -> _SupportMatch | None:
    gt_boxes = _rule_gt_boxes([rule])
    if not gt_boxes:
        return None
    rule_pages = {box.page for box in gt_boxes}
    best: _SupportMatch | None = None
    best_key: tuple[float, float, float, float, float, float, float, float, float] | None = None

    for support_units in support_sets:
        candidates = [
            unit for unit in support_units if unit.page in rule_pages and _unit_near_any_gt_box(unit, gt_boxes)
        ]
        for group in _candidate_groups(candidates, rule.expected_value):
            comparison = _compare_support_text(
                rule.expected_value,
                " ".join(unit.text for unit in group),
                expected_type=expected_type,
            )
            boxes = tuple(BBox(page=unit.page, bbox=unit.bbox, group=rule.field_path) for unit in group)
            bbox_summary = compute_standard_iou_metrics(gt_boxes, list(boxes))
            max_ioa = field_grounding_max_ioa(bbox_summary)
            loc_candidate = field_grounding_localization_passes(
                iou=bbox_summary.iou,
                max_ioa=max_ioa,
                comparison=comparison,
            )
            key = (
                float(loc_candidate),
                float(field_grounding_has_canonical_exact_text_match(comparison)),
                float(comparison.passed),
                comparison.score,
                -float(len(group)),
                _granularity_rank(group[0].granularity),
                bbox_summary.iou,
                max_ioa,
                -abs(bbox_summary.pred_area - bbox_summary.gt_area),
            )
            if best_key is None or key > best_key:
                best_key = key
                # All units in one candidate group are sourced from a single
                # support pool, so the granularity label is consistent across
                # the group — read it off the first unit.
                best = _SupportMatch(
                    comparison=comparison,
                    boxes=boxes,
                    iou=bbox_summary.iou,
                    max_ioa=max_ioa,
                    granularity=group[0].granularity,
                    units=tuple(group),
                )

    # Return even on failure (was: `return best if best is not None and
    # best.comparison.passed else None`). Downstream rung computation needs
    # to distinguish "no localization" from "localized but attribution
    # failed" — those cases have different metadata shape.
    return best


def _granularity_rank(granularity: str) -> float:
    return {"word": 2.0, "line": 1.0}.get(granularity, 0.0)


def _candidate_groups(units: list[_SupportUnit], expected: Any) -> list[tuple[_SupportUnit, ...]]:
    ordered = sorted(units, key=lambda unit: (unit.page, unit.order_index, unit.bbox[1], unit.bbox[0]))
    groups: list[tuple[_SupportUnit, ...]] = [(unit,) for unit in ordered]
    expected_len = max(len(normalize_text(expected)), 1)
    max_norm_len = expected_len * 2 + 20

    by_page: dict[int, list[_SupportUnit]] = {}
    for unit in ordered:
        by_page.setdefault(unit.page, []).append(unit)
    for page_units in by_page.values():
        for start in range(len(page_units)):
            parts: list[_SupportUnit] = []
            for unit in page_units[start : start + 20]:
                parts.append(unit)
                joined_norm = normalize_text(" ".join(part.text for part in parts))
                if len(parts) > 1:
                    groups.append(tuple(parts))
                if len(joined_norm) > max_norm_len:
                    break
    return groups


def _compare_support_text(expected: Any, actual: str, *, expected_type: ExpectedType) -> ValueComparison:
    return compare_attributed_value(expected, actual, expected_type=expected_type, source_kind="native")


def _rule_reason(match: _SupportMatch | None, loc_pass: bool, *, ungrounded_source: str | None) -> str:
    if not loc_pass and ungrounded_source is not None:
        return "text_present_but_ungrounded"
    if match is None:
        return "no_support_match"
    if not loc_pass:
        return "localization_failed"
    return match.comparison.reason


def _rule_gt_boxes(field_rules: list[ExtractFieldTestRule]) -> list[BBox]:
    boxes: list[BBox] = []
    for rule in field_rules:
        for bbox in rule.bboxes:
            normalized = _as_xywh(bbox.bbox)
            if normalized is not None:
                boxes.append(BBox(page=bbox.page, bbox=normalized, group=rule.field_path))
    return boxes


def _is_stray_rule(rule: ExtractFieldTestRule) -> bool:
    tags = {tag.casefold() for tag in rule.tags}
    return (
        rule.expected_value is None
        or "stray" in tags
        or "no_value" in tags
        or any(tag.endswith(":stray") for tag in tags)
    )


def _unit_near_any_gt_box(unit: _SupportUnit, gt_boxes: list[BBox], *, margin: float = 0.01) -> bool:
    unit_xyxy = _xywh_to_xyxy(unit.bbox)
    for gt in gt_boxes:
        if unit.page != gt.page:
            continue
        gt_xyxy = _expand_xyxy(_xywh_to_xyxy(gt.bbox), margin=margin)
        if _xyxy_intersects(unit_xyxy, gt_xyxy):
            return True
        if _xyxy_contains_point(gt_xyxy, _xyxy_center(unit_xyxy)):
            return True
        if _xyxy_contains_point(unit_xyxy, _xyxy_center(gt_xyxy)):
            return True
    return False


def _page_bboxes_are_normalized(page: ParseLayoutPageIR) -> bool:
    for item in page.items:
        segment = item.layout_segments[0] if item.layout_segments else item.bbox
        if segment is not None:
            return max(segment.x + segment.w, segment.y + segment.h) <= 1.0
    return False


def _segment_to_normalized_xywh(
    segment: LayoutSegmentIR | None,
    *,
    width: float,
    height: float,
    already_normalized: bool,
) -> tuple[float, float, float, float] | None:
    if segment is None:
        return None
    x, y, w, h = float(segment.x), float(segment.y), float(segment.w), float(segment.h)
    if not already_normalized:
        if width <= 0.0 or height <= 0.0:
            return None
        x /= width
        w /= width
        y /= height
        h /= height
    return _as_xywh((x, y, w, h))


def _as_xywh(value: Any) -> tuple[float, float, float, float] | None:
    if value is None or len(value) != 4:
        return None
    x, y, w, h = value
    x_f = float(x)
    y_f = float(y)
    w_f = float(w)
    h_f = float(h)
    if w_f <= 0.0 or h_f <= 0.0:
        return None
    return (x_f, y_f, w_f, h_f)


def _xywh_to_xyxy(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])


def _expand_xyxy(
    bbox: tuple[float, float, float, float],
    *,
    margin: float,
) -> tuple[float, float, float, float]:
    return (
        max(0.0, bbox[0] - margin),
        max(0.0, bbox[1] - margin),
        min(1.0, bbox[2] + margin),
        min(1.0, bbox[3] + margin),
    )


def _xyxy_intersects(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _xyxy_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _xyxy_contains_point(bbox: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]
