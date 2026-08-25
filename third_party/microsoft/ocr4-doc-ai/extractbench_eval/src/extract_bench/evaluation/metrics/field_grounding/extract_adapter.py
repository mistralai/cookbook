"""Field grounding metrics for extract pipeline outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from extract_bench.evaluation.metrics.field_grounding.core import (
    FIELD_GROUNDING_CANONICAL_EXACT_SCORE_THRESHOLD,
    FIELD_GROUNDING_RELAXED_IOU_THRESHOLD,
    FIELD_GROUNDING_RELAXED_MAX_IOA_THRESHOLD,
    FIELD_GROUNDING_STRICT_IOU_THRESHOLD,
    BBox,
    ValueComparison,
    compute_bbox_metrics,
    compute_standard_iou_metrics,
    field_grounding_has_canonical_exact_text_match,
    field_grounding_localization_passes,
    field_grounding_localization_reason,
    field_grounding_max_ioa,
)
from extract_bench.evaluation.metrics.field_grounding.evidence_comparator import (
    compare_evidence_value,
    parse_match_by_keys,
    stable_value_key,
)
from extract_bench.evaluation.metrics.field_grounding.value_compare import (
    COMPARATOR_VERSION,
    ExpectedType,
    compare_array_against_rule,
    compare_attributed_value,
    compare_value_against_rule,
    expected_type_for_field_path,
)
from extract_bench.schemas.evaluation import MetricValue
from extract_bench.test_cases.extract_field_paths import get_path, parse_field_path
from extract_bench.test_cases.schema import ExtractFieldTestRule, iter_rule_evidence

_MISSING = object()


def compute_extract_field_grounding_metrics(
    *,
    extracted_data: Any,
    field_rules: list[ExtractFieldTestRule],
    field_citations: list[Any],
    data_schema: dict[str, Any] | None = None,
    skip_field_paths: Iterable[str] = (),
    list_unwrap_applied: bool = False,
    list_unwrap_mode: str = "no_op",
    alias_skipped_field_paths: Iterable[str] = (),
    normalized_top_level_keys: Iterable[str] = (),
    list_unwrap_warnings: Iterable[str] = (),
) -> list[MetricValue]:
    """Compute value and bbox field grounding metrics for extract outputs.

    ``skip_field_paths`` lists rule ``field_path`` values that are known not
    to be scorable against the current ``extracted_data`` shape (typically
    scalar rules excluded after a per_table_row list-unwrap). They are
    dropped from value-metric denominators but still participate in bbox
    metrics, which don't depend on the prediction shape.

    ``list_unwrap_applied`` (and ``skip_field_paths``) are recorded in the
    metadata of the emitted precision/recall/f1 metrics so downstream
    reports can tell whether the root-level list-unwrap fired and which
    rules were excluded. No new metric names are introduced.
    """
    if not field_rules:
        return []

    legacy_rules = [rule for rule in field_rules if rule.evidence is None]
    v02_rules = [rule for rule in field_rules if rule.evidence is not None]
    metrics: list[MetricValue] = []
    metrics.extend(
        _compute_value_metrics(
            extracted_data,
            legacy_rules,
            skip_field_paths=skip_field_paths,
            list_unwrap_applied=list_unwrap_applied,
            list_unwrap_mode=list_unwrap_mode,
            alias_skipped_field_paths=alias_skipped_field_paths,
            normalized_top_level_keys=normalized_top_level_keys,
            list_unwrap_warnings=list_unwrap_warnings,
            data_schema=data_schema,
        )
    )
    metrics.extend(_compute_bbox_metrics(legacy_rules, field_citations))
    metrics.extend(_compute_record_metrics(legacy_rules, extracted_data, field_citations, data_schema=data_schema))
    metrics.extend(_compute_null_hallucination_metrics(legacy_rules, extracted_data))
    metrics.extend(
        _compute_extract_pass_rate_metrics(
            legacy_rules,
            extracted_data,
            field_citations,
            skip_field_paths=skip_field_paths,
            data_schema=data_schema,
        )
    )
    metrics.extend(
        _compute_v02_evidence_metrics(
            v02_rules,
            extracted_data,
            field_citations,
            skip_field_paths=skip_field_paths,
            data_schema=data_schema,
        )
    )
    return metrics


def _compute_value_metrics(
    extracted_data: Any,
    field_rules: list[ExtractFieldTestRule],
    *,
    skip_field_paths: Iterable[str] = (),
    list_unwrap_applied: bool = False,
    list_unwrap_mode: str = "no_op",
    alias_skipped_field_paths: Iterable[str] = (),
    normalized_top_level_keys: Iterable[str] = (),
    list_unwrap_warnings: Iterable[str] = (),
    data_schema: dict[str, Any] | None = None,
) -> list[MetricValue]:
    skip_set = set(skip_field_paths)
    value_rules = [rule for rule in field_rules if not _is_stray_rule(rule) and rule.field_path not in skip_set]
    if not value_rules:
        return []

    expected_by_pattern: dict[tuple[str | None, ...], list[ExtractFieldTestRule]] = defaultdict(list)
    for rule in value_rules:
        pattern = _field_pattern(rule.field_path)
        if pattern is not None:
            expected_by_pattern[pattern].append(rule)

    tp = 0
    fp = 0
    fn = 0
    rule_results: list[dict[str, Any]] = []

    for pattern, rules in expected_by_pattern.items():
        predictions = list(_iter_values_for_pattern(extracted_data, pattern))
        matches, group_rule_results = _match_value_group(rules, predictions, data_schema=data_schema)
        group_tp = len(matches)
        group_fp = len(predictions) - group_tp
        group_fn = len(rules) - group_tp
        tp += group_tp
        fp += group_fp
        fn += group_fn
        rule_results.extend(group_rule_results)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = _harmonic_mean(precision, recall)
    metadata = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gt": len(value_rules),
        "total_pred": tp + fp,
        "rule_results": rule_results,
        "list_unwrap_applied": bool(list_unwrap_applied),
        "list_unwrap_mode": list_unwrap_mode,
        "skipped_field_paths": sorted(skip_set),
        "alias_skipped_field_paths": sorted(set(alias_skipped_field_paths)),
        "normalized_top_level_keys": sorted(set(normalized_top_level_keys)),
        "list_unwrap_warnings": list(list_unwrap_warnings),
    }
    return [
        MetricValue(metric_name="precision", value=precision, metadata=metadata),
        MetricValue(metric_name="recall", value=recall, metadata=metadata),
        MetricValue(metric_name="f1", value=f1, metadata=metadata),
    ]


def _compute_bbox_metrics(field_rules: list[ExtractFieldTestRule], field_citations: list[Any]) -> list[MetricValue]:
    gt_boxes: list[BBox] = []
    for rule in field_rules:
        group = _pattern_group(rule.field_path)
        for bbox in rule.bboxes:
            normalized = _as_xywh(bbox.bbox)
            if normalized is not None:
                gt_boxes.append(BBox(page=bbox.page, bbox=normalized, group=group))

    if not gt_boxes:
        return []

    pred_boxes = [
        BBox(page=page, bbox=citation_bbox, group=_pattern_group(field_path))
        for citation in field_citations
        if (field_path := getattr(citation, "field_path", None))
        and (page := _as_int(getattr(citation, "page", None))) is not None
        and (citation_bbox := _as_xywh(getattr(citation, "bbox", None))) is not None
    ]
    summary = compute_standard_iou_metrics(gt_boxes, pred_boxes)
    recall_summary = compute_bbox_metrics(gt_boxes, pred_boxes)
    metadata = {
        "gt_count": len(gt_boxes),
        "pred_count": len(pred_boxes),
        "gt_area": summary.gt_area,
        "pred_area": summary.pred_area,
        "intersection_area": summary.intersection_area,
        "union_area": summary.union_area,
    }
    return [
        MetricValue(metric_name="iou", value=summary.iou, metadata=metadata),
        MetricValue(
            metric_name="bbox_recall",
            value=recall_summary.bbox_recall,
            metadata={**metadata, "covered_gt_area": recall_summary.covered_gt_area},
        ),
    ]


_HALLUCINATED_PATHS_SAMPLE_CAP = 20


def _compute_null_hallucination_metrics(
    field_rules: list[ExtractFieldTestRule],
    extracted_data: Any,
) -> list[MetricValue]:
    """Score whether the model hallucinates values for null-expected rules.

    Scope: rules with ``expected_value is None`` and ``verified=True``. The
    7 known-bad bronze ``expected_null_got_text`` annotations in v0.7 are
    excluded by the verified filter.

    Outcomes per rule:
    - **Correct skip** (``tp``): ``extracted_data`` has no value at the
      field_path (missing key, list index out of range) or the value is
      ``None``.
    - **Hallucination** (``fp``): ``extracted_data`` has any non-``None``
      value at the field_path. Booleans, numbers (incl. ``0``/``False``),
      strings (incl. ``""``), and non-empty containers all count — the
      model committed to *some* concrete value.

    The headline ``null_hallucination_rate`` ∈ [0, 1] is ``fp / (tp + fp)``;
    lower is better. ``fn`` is always 0 (the null cohort has no
    "missed-null" outcome). The runner's standard tp/fp/fn pooling
    produces ``total_null_hallucination_rate_*`` for the global view.
    """

    # v0.2: a rule is null-expected only when neither expected_value nor any
    # evidence entry carries a value. Legacy rules (no evidence) collapse to
    # the original ``expected_value is None`` check.
    def _is_null_expected(rule: ExtractFieldTestRule) -> bool:
        if rule.expected_value is not None:
            return False
        if rule.evidence:
            return all(entry.value is None for entry in rule.evidence)
        return True

    null_rules = [rule for rule in field_rules if _is_null_expected(rule) and rule.verified]
    if not null_rules:
        return []

    correct_skips = 0
    hallucinations = 0
    hallucinated_paths: list[dict[str, Any]] = []

    for rule in null_rules:
        emitted = _get_field_value(extracted_data, rule.field_path)
        if emitted is _MISSING or emitted is None:
            correct_skips += 1
            continue
        hallucinations += 1
        if len(hallucinated_paths) < _HALLUCINATED_PATHS_SAMPLE_CAP:
            hallucinated_paths.append(
                {
                    "field_path": rule.field_path,
                    "emitted_value": emitted,
                    "tags": list(rule.tags),
                }
            )

    rate = hallucinations / len(null_rules)
    return [
        MetricValue(
            metric_name="null_hallucination_rate",
            value=rate,
            metadata={
                "tp": correct_skips,
                "fp": hallucinations,
                "fn": 0,
                "total_null_rules": len(null_rules),
                "hallucinated_count": hallucinations,
                "hallucinated_paths": hallucinated_paths,
            },
        ),
    ]


_PASS_RATE_IOU_THRESHOLD = FIELD_GROUNDING_STRICT_IOU_THRESHOLD


def _compute_extract_pass_rate_metrics(
    field_rules: list[ExtractFieldTestRule],
    extracted_data: Any,
    field_citations: list[Any],
    *,
    skip_field_paths: Iterable[str] = (),
    data_schema: dict[str, Any] | None = None,
) -> list[MetricValue]:
    """Per-rule loc / attr / element pass-rate metrics, mirroring parse semantics.

    For each non-stray rule we compute:

    - ``loc_pass``: best per-rule standard set IoU, scoped by field family via
      ``_pattern_group``. Strict pass is IoU >= 0.5; relaxed pass is IoU >= 0.3,
      max directional IoA >= 0.7, and exact typed value match.
    - ``attr_pass``: ``loc_pass`` AND the predicted value at the rule's
      ``field_path`` matches the rule's ``expected_value`` under
      :func:`compare_field_value`.
    - ``element_pass``: ``loc_pass`` AND ``attr_pass`` (no class-pass concept
      on extract, just the AND of the two).

    Each metric is emitted with ``tp/fp/fn`` metadata so the runner pools
    them into ``total_extract_*_tp/fp/fn`` automatically (mirrors the
    ``null_hallucination_rate`` pattern). ``fn`` is always 0 — every rule
    yields a definite pass/fail, there is no "missed" outcome.

    Rules in ``skip_field_paths`` are excluded entirely (no per-rule metric,
    not counted in tp/fp denominators). This mirrors ``_compute_value_metrics``
    so list-unwrapped per-table-row predictions don't artificially fail
    attribution on scalar fields they structurally cannot reach via
    ``_get_field_value``.

    Naming differs from the parse-side ``extract_field_*`` metrics on
    purpose: the parse adapter already emits
    ``extract_field_localization_pass_rate`` / ``extract_field_attribution_pass_rate``
    / ``extract_field_element_pass_rate`` from a different code path, so
    extract-side variants drop ``field`` to keep the report unambiguous.
    """
    skip_set = set(skip_field_paths)
    value_rules = [rule for rule in field_rules if not _is_stray_rule(rule) and rule.field_path not in skip_set]
    if not value_rules:
        return []

    citations_by_field_path: dict[str, list[BBox]] = defaultdict(list)
    citation_paths_by_pattern: dict[tuple[str | None, ...], set[str]] = defaultdict(set)
    for citation in field_citations:
        cit_field_path = getattr(citation, "field_path", None)
        if not cit_field_path:
            continue
        page = _as_int(getattr(citation, "page", None))
        if page is None:
            continue
        cit_bbox = _as_xywh(getattr(citation, "bbox", None))
        if cit_bbox is None:
            continue
        group = _pattern_group(cit_field_path)
        pred_box = BBox(page=page, bbox=cit_bbox, group=group)
        citations_by_field_path[cit_field_path].append(pred_box)
        pattern = _field_pattern(cit_field_path)
        if pattern is not None:
            citation_paths_by_pattern[pattern].add(cit_field_path)

    value_match_by_rule: dict[int, ValueComparison] = {}
    matched_pred_path_by_rule: dict[int, str] = {}
    for pattern, rules in _rules_by_field_pattern(value_rules).items():
        path_predictions = _iter_values_for_pattern_with_paths(extracted_data, pattern)
        _, comparisons, matches = _match_value_group_detailed_with_geometry(
            rules,
            path_predictions,
            candidate_pred_paths=sorted(citation_paths_by_pattern.get(pattern, set())),
            citations_by_field_path=citations_by_field_path,
            data_schema=data_schema,
        )
        for rule_index, value_comparison in comparisons.items():
            value_match_by_rule[id(rules[rule_index])] = value_comparison
        for rule_index, pred_path in matches:
            matched_pred_path_by_rule[id(rules[rule_index])] = pred_path

    loc_passes = 0
    attr_passes = 0
    element_passes = 0
    graded_total = 0  # rules whose GT carries a bbox (grounding-gradeable)
    verified_total = 0
    verified_loc_passes = 0
    verified_attr_passes = 0
    verified_element_passes = 0
    iou_sum = 0.0
    matched_iou_sum = 0.0
    unmatched_iou_sum = 0.0
    rule_results: list[dict[str, Any]] = []

    for rule in value_rules:
        rule_iou_threshold = rule.iou_threshold
        group = _pattern_group(rule.field_path)
        gt_boxes: list[BBox] = []
        if rule.evidence is not None:
            for entry in rule.evidence:
                if entry.page is None or entry.bbox is None:
                    continue
                normalized = _as_xywh(entry.bbox)
                if normalized is not None:
                    gt_boxes.append(BBox(page=entry.page, bbox=normalized, group=group))
        else:
            for gt_bbox in rule.bboxes:
                normalized = _as_xywh(gt_bbox.bbox)
                if normalized is not None:
                    gt_boxes.append(BBox(page=gt_bbox.page, bbox=normalized, group=group))

        expected_type = expected_type_for_field_path(data_schema, rule.field_path, rule.expected_value)
        comparison: ValueComparison | None = value_match_by_rule.get(id(rule))
        matched_pred_path = matched_pred_path_by_rule.get(id(rule))
        if not gt_boxes:
            # loc / attr / element / IoU are grounding metrics, defined only for
            # rules whose GT carries a bounding box. A rule with no GT bbox is
            # not counted at all (neither pass nor fail), so it can't drag the
            # pass rates or mean IoU down; a document whose rules carry no GT
            # bbox emits none of these metrics and is excluded from the average.
            rule_results.append(
                {
                    "field_path": rule.field_path,
                    "verified": rule.verified,
                    "has_gt_bbox": False,
                    "graded_for_grounding": False,
                }
            )
            continue
        graded_total += 1
        pred_boxes = citations_by_field_path.get(matched_pred_path, []) if matched_pred_path else []
        selected_pred_boxes = _select_best_bbox_group(
            gt_boxes,
            pred_boxes,
            comparison=comparison,
            strict_iou_threshold=rule_iou_threshold,
        )
        bbox_summary = compute_standard_iou_metrics(gt_boxes, selected_pred_boxes)
        iou = bbox_summary.iou
        max_ioa = field_grounding_max_ioa(bbox_summary)
        loc_pass = field_grounding_localization_passes(
            iou=iou,
            max_ioa=max_ioa,
            comparison=comparison,
            strict_iou_threshold=rule_iou_threshold,
        )
        attr_pass = loc_pass and comparison is not None and comparison.passed

        element_pass = loc_pass and attr_pass

        loc_passes += int(loc_pass)
        attr_passes += int(attr_pass)
        element_passes += int(element_pass)
        if rule.verified:
            verified_total += 1
            verified_loc_passes += int(loc_pass)
            verified_attr_passes += int(attr_pass)
            verified_element_passes += int(element_pass)
        iou_sum += iou
        if loc_pass:
            matched_iou_sum += iou
        else:
            unmatched_iou_sum += iou

        rule_results.append(
            {
                "field_path": rule.field_path,
                "verified": rule.verified,
                "loc_pass": loc_pass,
                "attr_pass": attr_pass,
                "element_pass": element_pass,
                "iou": iou,
                "max_ioa": max_ioa,
                "has_gt_bbox": bool(gt_boxes),
                "matched_pred_field_path": matched_pred_path,
                "matched_pred_bboxes": [list(box.bbox) for box in selected_pred_boxes],
                "expected_type": expected_type,
                "attr_source": "structured_value_index_tolerant" if comparison is not None else "missing",
                "mode": comparison.mode if comparison is not None else "missing",
                "reason": comparison.reason if comparison is not None else "missing_prediction",
                "localization_reason": (
                    field_grounding_localization_reason(
                        iou=iou,
                        max_ioa=max_ioa,
                        comparison=comparison,
                        strict_iou_threshold=rule_iou_threshold,
                    )
                    if selected_pred_boxes or loc_pass
                    else "no_support_match"
                ),
                "canonical_exact": field_grounding_has_canonical_exact_text_match(comparison),
                "comparator_version": COMPARATOR_VERSION,
            }
        )

    total = graded_total
    if total == 0:
        # No rule in this document carries a GT bbox -> localization /
        # attribution / element / IoU are undefined. Emit nothing so the
        # document is excluded from these metrics' averages (rather than
        # counted as an all-fail 0.0).
        return []
    unmatched = total - loc_passes
    base_meta: dict[str, Any] = {
        "total": total,
        "iou_threshold": _PASS_RATE_IOU_THRESHOLD,
        "relaxed_iou_threshold": FIELD_GROUNDING_RELAXED_IOU_THRESHOLD,
        "relaxed_max_ioa_threshold": FIELD_GROUNDING_RELAXED_MAX_IOA_THRESHOLD,
        "canonical_exact_score_threshold": FIELD_GROUNDING_CANONICAL_EXACT_SCORE_THRESHOLD,
        "rule_results": rule_results,
        "skipped_field_paths": sorted(skip_set),
        "verified_total": verified_total,
    }

    return [
        MetricValue(
            metric_name="extract_localization_pass_rate",
            value=loc_passes / total,
            metadata={
                **base_meta,
                "passed": loc_passes,
                "verified_passed": verified_loc_passes,
                "tp": loc_passes,
                "fp": total - loc_passes,
                "fn": 0,
            },
        ),
        MetricValue(
            metric_name="extract_attribution_pass_rate",
            value=attr_passes / total,
            metadata={
                **base_meta,
                "passed": attr_passes,
                "verified_passed": verified_attr_passes,
                "tp": attr_passes,
                "fp": total - attr_passes,
                "fn": 0,
            },
        ),
        MetricValue(
            metric_name="extract_element_pass_rate",
            value=element_passes / total,
            metadata={
                **base_meta,
                "passed": element_passes,
                "verified_passed": verified_element_passes,
                "tp": element_passes,
                "fp": total - element_passes,
                "fn": 0,
            },
        ),
        MetricValue(
            metric_name="extract_avg_iou",
            value=iou_sum / total,
            metadata={
                **base_meta,
                "matched": loc_passes,
                "unmatched": unmatched,
            },
        ),
        MetricValue(
            metric_name="extract_avg_iou_matched",
            value=matched_iou_sum / loc_passes if loc_passes > 0 else 0.0,
            metadata={
                **base_meta,
                "matched": loc_passes,
                "unmatched": unmatched,
            },
        ),
        MetricValue(
            metric_name="extract_avg_iou_unmatched",
            value=unmatched_iou_sum / unmatched if unmatched > 0 else 0.0,
            metadata={
                **base_meta,
                "matched": loc_passes,
                "unmatched": unmatched,
            },
        ),
    ]


@dataclass(frozen=True)
class _V02FamilyCitations:
    """Citation-derived facts shared by every rule of one field pattern.

    All of these depend only on the rule's field pattern, and long-list docs
    repeat the same pattern across thousands of leaf rules — computing them
    per rule made the evidence loop quadratic in family size (40+ minutes on
    a 37k-rule document).
    """

    group: str
    page_qual: bool
    cit_pages: frozenset[int]
    bbox_cit_pages: frozenset[int]
    pred_boxes_by_page: dict[int, list[BBox]]
    exact_count: int
    coarse_count: int
    descendant_count: int


def _build_v02_family_citations(
    rule_pattern: tuple[str | None, ...],
    citations_by_pattern: dict[tuple[str | None, ...], list[Any]],
) -> _V02FamilyCitations:
    exact_cits = citations_by_pattern.get(rule_pattern, [])
    # Coarse parent prefix-walk: take only the NEAREST non-empty parent
    # prefix's citations, not every ancestor. The nearest parent is the
    # most specific cite a pipeline could plausibly mean for this leaf;
    # broader top-level cites would over-credit M2a coverage.
    coarse_cits: list[Any] = []
    for prefix_len in range(len(rule_pattern) - 1, 0, -1):
        candidates = citations_by_pattern.get(rule_pattern[:prefix_len], [])
        if candidates:
            coarse_cits = candidates
            break

    # Descendant walk: for array-shaped rules whose pattern is a strict
    # prefix of any per-leaf citation pattern, include those leaf citations.
    # Without this, an array rule with pattern ``('grants',)`` never sees
    # pipeline citations like ``('grants', None, 'recipient_name')`` even
    # though the pipeline IS correctly grounding every row's leaf — the
    # pipeline just doesn't emit a synthetic citation at the array level.
    # Pipelines that DO emit array-level citations contribute via
    # ``exact_cits`` and additionally via this walk when leaf citations
    # also exist; the two are unioned, not gated.
    descendant_cits: list[Any] = []
    for pattern, cits in citations_by_pattern.items():
        if len(pattern) > len(rule_pattern) and pattern[: len(rule_pattern)] == rule_pattern:
            descendant_cits.extend(cits)

    group = ".".join("[]" if token is None else token for token in rule_pattern)

    cit_pages: set[int | None] = set()
    for cits in (exact_cits, coarse_cits, descendant_cits):
        cit_pages.update(_as_int(getattr(cit, "page", None)) for cit in cits)
    cit_pages.discard(None)

    bbox_cit_pages: set[int] = set()
    for cits in (exact_cits, descendant_cits):
        for cit in cits:
            if getattr(cit, "bbox", None) is None:
                continue
            page = _as_int(getattr(cit, "page", None))
            if page is not None:
                bbox_cit_pages.add(page)

    pred_boxes_by_page: dict[int, list[BBox]] = defaultdict(list)
    for citation in exact_cits:
        page = _as_int(getattr(citation, "page", None))
        bbox = _as_xywh(getattr(citation, "bbox", None))
        if page is not None and bbox is not None:
            pred_boxes_by_page[page].append(BBox(page=page, bbox=bbox, group=group))

    return _V02FamilyCitations(
        group=group,
        page_qual=bool(exact_cits or coarse_cits or descendant_cits),
        cit_pages=frozenset(page for page in cit_pages if page is not None),
        bbox_cit_pages=frozenset(bbox_cit_pages),
        pred_boxes_by_page=dict(pred_boxes_by_page),
        exact_count=len(exact_cits),
        coarse_count=len(coarse_cits),
        descendant_count=len(descendant_cits),
    )


def _compute_v02_evidence_metrics(
    field_rules: list[ExtractFieldTestRule],
    extracted_data: Any,
    field_citations: list[Any],
    *,
    skip_field_paths: Iterable[str] = (),
    data_schema: dict[str, Any] | None = None,
) -> list[MetricValue]:
    """v0.2 evidence-list value and page-grounded metrics.

    Pass/fail does not depend on bbox IoU. For bbox-bearing evidence entries we
    still compute alignment diagnostics so dashboards can analyze localization
    quality separately from value/page correctness.

    Rules with ``evidence_required=False`` are excluded from denominators. If a
    rule has ``evidence=[]`` while evidence is required, emitting no value (or
    null) passes value-only grading; emitting any concrete value fails because
    there is no accepted evidence value to match.
    """
    skip_set = set(skip_field_paths)
    eligible = [
        rule
        for rule in field_rules
        if rule.evidence_required and not _has_stray_tag(rule) and rule.field_path not in skip_set and rule.verified
    ]
    if not eligible:
        return []

    citations_by_pattern: dict[tuple[str | None, ...], list[Any]] = defaultdict(list)
    for citation in field_citations:
        cit_field_path = getattr(citation, "field_path", None)
        if not cit_field_path:
            continue
        if _as_int(getattr(citation, "page", None)) is None:
            continue
        cit_pattern = _field_pattern(cit_field_path)
        if cit_pattern is None:
            continue
        citations_by_pattern[cit_pattern].append(citation)

    # Row identity alignment for match_by array families: per-row leaf rules
    # are graded against the predicted row with the matching identity, not the
    # row at the same position, so reordered or dropped rows fail only
    # themselves. Built from the full rule list — the parent rule contributes
    # alignment context even when it is itself unverified/ungraded.
    row_alignments = _build_match_by_alignments(field_rules, extracted_data)

    value_pass = 0
    page_pass = 0
    page_pass_covered = 0
    page_qualified = 0
    bbox_qualified = 0
    bbox_iou_sum = 0.0
    bbox_iou_count = 0
    bbox_covered_pass = 0
    bbox_iou_pass_sum = 0.0
    # Denominator for the bbox_* metrics: leaves whose GT evidence carries a
    # bounding box. Grounding is undefined for a leaf with no GT bbox, so such
    # leaves are excluded from the bbox metrics (and a doc with none emits no
    # bbox metric at all) instead of being scored 0 and averaged in.
    bbox_gt_total = 0

    rule_results: list[dict[str, Any]] = []
    family_citations: dict[tuple[str | None, ...], _V02FamilyCitations] = {}

    for rule in eligible:
        rule_pattern = _field_pattern(rule.field_path)
        if rule_pattern is None:
            continue
        evidence_entries = iter_rule_evidence(rule)

        family = family_citations.get(rule_pattern)
        if family is None:
            family = _build_v02_family_citations(rule_pattern, citations_by_pattern)
            family_citations[rule_pattern] = family

        page_qual = family.page_qual
        # bbox coverage is only meaningful when the predicted bbox is on a page
        # that matches some evidence page. A bbox emitted on the wrong page is
        # not "the correct bbox at the correct position" — it's a wrong-page
        # bbox, which should not count as covered. We require both:
        #   1. an exact-path or descendant-path citation with a non-null bbox,
        #   2. that citation's page intersects the evidence page set.
        # If no evidence carries a page, we cannot validate; leaf is uncovered.
        ev_pages = {ev.page for ev in evidence_entries if ev.page is not None}
        # A leaf is grounding-gradeable only when its GT evidence carries a bbox.
        gt_has_bbox = any(
            ev.page is not None and ev.bbox is not None and _as_xywh(ev.bbox) is not None for ev in evidence_entries
        )
        if gt_has_bbox:
            bbox_gt_total += 1
        bbox_qual = bool(ev_pages) and not ev_pages.isdisjoint(family.bbox_cit_pages)
        page_qualified += int(page_qual)
        if gt_has_bbox:
            bbox_qualified += int(bbox_qual)

        predictions = _prediction_values_for_v02_rule(extracted_data, rule, row_alignments)
        value_comparisons = [_compare_v02_prediction(rule, prediction) for prediction in predictions]
        best_value_comparison = _best_comparison(value_comparisons)
        empty_evidence_null_pass = not evidence_entries and (
            not predictions or all(prediction is None for prediction in predictions)
        )
        value_pass_for_rule = bool(best_value_comparison and best_value_comparison.passed) or empty_evidence_null_pass
        value_pass += int(value_pass_for_rule)

        # Page-pass logic. The metric "is the value at the right page" is
        # vacuously true when there's no positive page claim to verify.
        # No positive page claim means ``evidence_with_value`` is empty —
        # which covers two cases:
        #   1. Evidence list is itself empty (no entries at all).
        #   2. Evidence list is non-empty but every entry has ``value=None``
        #      (the gold asserts "no value here", and a null assertion
        #      doesn't bind to a page).
        # In both cases, when ``value_pass`` succeeds, page_pass passes
        # vacuously. The positive-evidence path only runs when at least one
        # entry has a non-null value; mixed evidence lists score against
        # the value-bearing entries only.
        evidence_with_value = [ev for ev in evidence_entries if ev.value is not None]
        page_pass_for_rule = False
        if value_pass_for_rule and not evidence_with_value:
            # Vacuous: no positive page claim. Pass on any value_pass.
            page_pass_for_rule = True
        elif page_qual and value_pass_for_rule and evidence_with_value:
            page_pass_for_rule = any(ev.page in family.cit_pages for ev in evidence_with_value)

        if page_pass_for_rule:
            page_pass += 1
            if page_qual:
                page_pass_covered += 1

        bbox_iou = _best_v02_bbox_iou(evidence_entries, family.pred_boxes_by_page, family.group)
        if bbox_iou is not None:
            bbox_iou_sum += bbox_iou
            bbox_iou_count += 1

        # Value-gated bbox metrics. A bbox is only meaningful when the value is
        # also right; a wrong value with a perfectly placed bbox is still a
        # wrong extraction. Both metrics use ``total`` as denominator so wrong
        # values, missing bboxes, and wrong-page bboxes all contribute 0.
        bbox_covered_pass_for_rule = bool(value_pass_for_rule and bbox_qual)
        bbox_iou_pass_for_rule = float(bbox_iou) if (value_pass_for_rule and bbox_iou is not None) else 0.0
        if gt_has_bbox:
            bbox_covered_pass += int(bbox_covered_pass_for_rule)
            bbox_iou_pass_sum += bbox_iou_pass_for_rule

        rule_results.append(
            {
                "field_path": rule.field_path,
                "path": rule.field_path,
                "evidence_count": len(evidence_entries),
                "value_pass": value_pass_for_rule,
                "page_pass": page_pass_for_rule,
                "page_qualified": page_qual,
                "bbox_qualified": bbox_qual,
                "bbox_iou": bbox_iou,
                "bbox_covered_pass": bbox_covered_pass_for_rule,
                "bbox_iou_pass": bbox_iou_pass_for_rule,
                "exact_citation_count": family.exact_count,
                "coarse_citation_count": family.coarse_count,
                "descendant_citation_count": family.descendant_count,
                "mode": getattr(best_value_comparison, "mode", "null_empty" if empty_evidence_null_pass else "missing"),
                "reason": (
                    "pass" if value_pass_for_rule else getattr(best_value_comparison, "reason", "missing_prediction")
                ),
            }
        )

    total = len(eligible)
    base_meta: dict[str, Any] = {
        "total": total,
        "verified_only": True,
        "rule_results": rule_results,
        "skipped_field_paths": sorted(skip_set),
        "match_by_row_alignment": {family: len(family_map) for family, family_map in row_alignments.items()},
    }

    metrics: list[MetricValue] = []

    metrics.append(
        MetricValue(
            metric_name="extract_evidence_value_pass_rate",
            value=value_pass / total if total > 0 else 0.0,
            metadata={
                **base_meta,
                "tp": value_pass,
                "fp": total - value_pass,
                "fn": 0,
                "denominator": "graded_verified_v02_rules",
            },
        )
    )
    metrics.append(
        MetricValue(
            metric_name="extract_evidence_page_pass_rate",
            value=page_pass / total if total > 0 else 0.0,
            metadata={
                **base_meta,
                "tp": page_pass,
                "fp": total - page_pass,
                "fn": 0,
                "denominator": "all_verified_rules",
            },
        )
    )
    metrics.append(
        MetricValue(
            metric_name="extract_evidence_page_covered_pass_rate",
            value=page_pass_covered / page_qualified if page_qualified > 0 else 0.0,
            metadata={
                **base_meta,
                "tp": page_pass_covered,
                "fp": page_qualified - page_pass_covered,
                "fn": 0,
                "denominator": "page_qualified_rules",
                "covered_total": page_qualified,
            },
        )
    )
    # bbox_* metrics are grounding metrics: they are defined only over leaves
    # whose GT evidence carries a bounding box (``bbox_gt_total``). A document
    # with no bbox-bearing GT emits none of them, so it is excluded from their
    # averages instead of being scored 0 and dragging the dataset number down.
    # A leaf whose GT *does* carry a bbox but whose prediction emits none still
    # counts as a 0 within these denominators -- a real grounding miss.
    if bbox_gt_total > 0:
        metrics.append(
            MetricValue(
                metric_name="extract_evidence_bbox_IOU_alignment",
                value=bbox_iou_sum / bbox_iou_count if bbox_iou_count > 0 else 0.0,
                metadata={
                    **base_meta,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "diagnostic": True,
                    "bbox_iou_count": bbox_iou_count,
                    "bbox_gt_total": bbox_gt_total,
                    "definition": "diagnostic_mean_best_iou_for_bbox_bearing_evidence",
                },
            )
        )
        metrics.append(
            MetricValue(
                metric_name="extract_evidence_bbox_coverage",
                value=bbox_qualified / bbox_gt_total,
                metadata={
                    **base_meta,
                    "tp": bbox_qualified,
                    "fp": bbox_gt_total - bbox_qualified,
                    "fn": 0,
                    "denominator": "gt_bbox_bearing_rules",
                    "bbox_gt_total": bbox_gt_total,
                    "definition": "per_leaf_binary_any_matching_citation_with_bbox",
                },
            )
        )
        # Value-gated bbox metrics. ``bbox_covered_pass_rate`` is the binary joint
        # of value-pass AND bbox-coverage (the bbox is on a page that matches some
        # evidence page). ``bbox_iou_pass_rate`` is the IoU score when value passes
        # else 0, averaged over the bbox-bearing leaves. Both penalize wrong
        # values, missing bboxes, and wrong-page bboxes.
        metrics.append(
            MetricValue(
                metric_name="extract_evidence_bbox_covered_pass_rate",
                value=bbox_covered_pass / bbox_gt_total,
                metadata={
                    **base_meta,
                    "tp": bbox_covered_pass,
                    "fp": bbox_gt_total - bbox_covered_pass,
                    "fn": 0,
                    "denominator": "gt_bbox_bearing_rules",
                    "bbox_gt_total": bbox_gt_total,
                    "definition": "per_leaf_binary_value_pass_and_bbox_qualified",
                },
            )
        )
        metrics.append(
            MetricValue(
                metric_name="extract_evidence_bbox_IOU_pass_rate",
                value=bbox_iou_pass_sum / bbox_gt_total,
                metadata={
                    **base_meta,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "denominator": "gt_bbox_bearing_rules",
                    "bbox_gt_total": bbox_gt_total,
                    "bbox_iou_pass_sum": bbox_iou_pass_sum,
                    "definition": "per_leaf_iou_when_value_pass_else_zero",
                },
            )
        )
    return metrics


def _select_best_bbox_group(
    gt_boxes: list[BBox],
    pred_boxes: list[BBox],
    *,
    comparison: ValueComparison | None,
    strict_iou_threshold: float | None = None,
) -> list[BBox]:
    """Select the predicted citation bbox group using field localization semantics."""
    if not gt_boxes or not pred_boxes:
        return []

    candidates = [box for box in pred_boxes if _bbox_near_any_gt_box(box, gt_boxes)]
    if not candidates:
        candidates = [
            box for box in pred_boxes if any(box.page == gt.page and box.group == gt.group for gt in gt_boxes)
        ]

    best_group: list[BBox] = []
    best_key: tuple[float, float, float, float, float, float, float] | None = None
    for group in _candidate_bbox_groups(candidates):
        summary = compute_standard_iou_metrics(gt_boxes, group)
        max_ioa = field_grounding_max_ioa(summary)
        loc_candidate = field_grounding_localization_passes(
            iou=summary.iou,
            max_ioa=max_ioa,
            comparison=comparison,
            strict_iou_threshold=strict_iou_threshold,
        )
        key = (
            float(loc_candidate),
            float(field_grounding_has_canonical_exact_text_match(comparison)),
            float(comparison.passed if comparison is not None else False),
            comparison.score if comparison is not None else 0.0,
            summary.iou,
            max_ioa,
            -abs(summary.pred_area - summary.gt_area),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_group = group
    return best_group


def _candidate_bbox_groups(boxes: list[BBox]) -> Iterable[list[BBox]]:
    ordered = sorted(boxes, key=lambda box: (box.page, box.bbox[1], box.bbox[0], box.bbox[2] * box.bbox[3]))
    for box in ordered:
        yield [box]

    by_page: dict[int, list[BBox]] = defaultdict(list)
    for box in ordered:
        by_page[box.page].append(box)
    for page_boxes in by_page.values():
        for start in range(len(page_boxes)):
            group: list[BBox] = []
            for box in page_boxes[start : start + 20]:
                group.append(box)
                if len(group) > 1:
                    yield list(group)


def _bbox_near_any_gt_box(box: BBox, gt_boxes: list[BBox], *, margin: float = 0.01) -> bool:
    box_xyxy = _xywh_to_xyxy(box.bbox)
    for gt in gt_boxes:
        if box.page != gt.page or box.group != gt.group:
            continue
        gt_xyxy = _expand_xyxy(_xywh_to_xyxy(gt.bbox), margin=margin)
        if _xyxy_intersects(box_xyxy, gt_xyxy):
            return True
        if _xyxy_contains_point(gt_xyxy, _xyxy_center(box_xyxy)):
            return True
        if _xyxy_contains_point(box_xyxy, _xyxy_center(gt_xyxy)):
            return True
    return False


def _get_field_value(extracted_data: Any, field_path: str) -> Any:
    try:
        tokens = parse_field_path(field_path)
    except ValueError:
        return _MISSING
    return get_path(extracted_data, tokens, default=_MISSING)


def _is_stray_rule(rule: ExtractFieldTestRule) -> bool:
    """Identify rules that should not contribute a value comparison.

    Stray rules are bbox-only evidence rules: they assert that some content
    exists at a location without prescribing a value. They are excluded from
    value F1 (already) and from record-level metrics; they remain in bbox
    metrics.

    v0.2: rules with ``expected_value=None`` but evidence carrying non-null
    values are NOT stray — they prescribe values via the evidence list. A rule
    is stray when both ``expected_value`` is None AND no evidence entry
    contributes a value (legacy null-expected behavior preserved).
    """
    tags = {tag.casefold() for tag in rule.tags}
    if "stray" in tags or "no_value" in tags or any(tag.endswith(":stray") for tag in tags):
        return True
    if rule.expected_value is not None:
        return False
    if rule.evidence:
        if any(entry.value is not None for entry in rule.evidence):
            return False
    return True


def _has_stray_tag(rule: ExtractFieldTestRule) -> bool:
    tags = {tag.casefold() for tag in rule.tags}
    return "stray" in tags or "no_value" in tags or any(tag.endswith(":stray") for tag in tags)


def compute_failure_field_denominator(field_rules: Iterable[ExtractFieldTestRule]) -> int:
    """Count rules eligible under the v0.2 evidence-metric filter, matching
    ``_compute_v02_evidence_metrics``.

    Exposed for the evaluation runner: when inference fails entirely on a
    doc, the runner synthesises a zero-scored EvaluationResult so the doc
    still counts in macro averages. But the synthesised metrics have no
    ``tp/fp/fn`` metadata, so the *micro* aggregator silently drops them
    (0/0 contribution) and pipelines that crash on a subset of docs end up
    ranked higher than they deserve. The runner uses this denominator to
    project the failure as ``tp=0, fp=N, fn=0`` for pooled pass-rate
    metrics, making the doc count in micro too.
    """
    return sum(1 for rule in field_rules if rule.evidence_required and rule.verified and not _has_stray_tag(rule))


def _field_pattern(field_path: str) -> tuple[str | None, ...] | None:
    """Return a path pattern with array indices wildcarded.

    Exact index matching is too brittle for table extraction: if a provider
    skips one row, all later rows shift and would falsely fail. The source export's
    text metrics are field-family metrics, so `rows[3].amount` and
    `rows[4].amount` are compared within the same `rows[].amount` pool.
    """
    try:
        tokens = parse_field_path(field_path)
    except ValueError:
        return None
    return tuple(None if isinstance(token, int) else token for token in tokens)


def _pattern_group(field_path: str) -> str:
    """Render the field pattern as a stable group key for bbox scoping.

    Bbox metrics share the same field-family logic as text metrics: skipping
    or reordering one row should not punish all later rows. Boxes at any list
    index of the same field family are scoped together so the IoU / bbox
    recall match is index-insensitive.
    """
    pattern = _field_pattern(field_path)
    if pattern is None:
        return field_path
    return ".".join("[]" if token is None else token for token in pattern)


def _iter_values_for_pattern(source: Any, pattern: Iterable[str | None]) -> Iterable[Any]:
    cursors = [source]
    for token in pattern:
        next_cursors: list[Any] = []
        if token is None:
            for cursor in cursors:
                if isinstance(cursor, list):
                    next_cursors.extend(item for item in cursor if item is not None)
        else:
            for cursor in cursors:
                if isinstance(cursor, dict) and token in cursor:
                    next_cursors.append(cursor[token])
        cursors = next_cursors
        if not cursors:
            return []
    return [cursor for cursor in cursors if cursor is not None and not isinstance(cursor, (dict, list))]


# Pass-2 alignment is quadratic in the worst case; bound the work so a
# pathological doc (thousands of GT rows, none exactly matching) degrades to
# positional fallback instead of stalling evaluation.
_ALIGNMENT_PASS2_MAX_PAIRS = 250_000
_IDENTITY_COMBO_CAP = 16


def _build_match_by_alignments(
    field_rules: list[ExtractFieldTestRule],
    extracted_data: Any,
) -> dict[str, dict[int, int]]:
    """Map GT row indices to predicted row indices for ``match_by`` families.

    Exact-index leaf lookup is too brittle for long lists: one dropped or
    reordered row shifts every later index and falsely fails every subsequent
    per-row leaf rule (the same brittleness ``_field_pattern`` documents for
    the legacy field-family metrics). When an array's parent rule declares
    ``match_by:<keys>`` semantics — rows are identified by key, order is
    irrelevant — leaf rules are graded against the predicted row carrying the
    matching identity rather than the row at the same position.

    Returns ``{family_root_path: {gt_index: pred_index}}``. GT rows absent
    from a family map have no surviving counterpart in the prediction; their
    leaf rules grade as missing predictions — a genuine miss scoped to that
    row instead of an index-shift cascade. Families whose parent rule is not
    ``match_by`` (ordered/set/multiset) are not realigned.
    """
    alignments: dict[str, dict[int, int]] = {}
    for parent in field_rules:
        structural = parent.structural or ""
        if not structural.startswith("match_by:"):
            continue
        keys = parse_match_by_keys(structural.split(":", 1)[1])
        if not keys:
            continue
        pred_rows = _get_field_value(extracted_data, parent.field_path)
        if pred_rows is _MISSING or not isinstance(pred_rows, list):
            continue
        try:
            root_tokens = parse_field_path(parent.field_path)
        except ValueError:
            continue
        identities, row_indices = _collect_gt_row_identities(field_rules, parent, root_tokens, keys)
        if not row_indices:
            continue
        alignments[parent.field_path] = _align_family_rows(identities, row_indices, pred_rows, keys, parent.comparator)
    return alignments


def _collect_gt_row_identities(
    field_rules: list[ExtractFieldTestRule],
    parent: ExtractFieldTestRule,
    root_tokens: list[str | int],
    keys: list[str],
) -> tuple[dict[int, dict[str, list[Any]]], set[int]]:
    """Gather per-row identity-key values for one array family.

    Primary source is the per-leaf rules (``root[i].key``), whose evidence may
    carry multiple OR-acceptable surface forms. Cells with no leaf rule fall
    back to the parent rule's evidence row dict, which is index-aligned with
    ``expected_output`` by construction in the GT builder.
    """
    depth = len(root_tokens)
    identities: dict[int, dict[str, list[Any]]] = {}
    row_indices: set[int] = set()
    for rule in field_rules:
        try:
            tokens = parse_field_path(rule.field_path)
        except ValueError:
            continue
        if len(tokens) <= depth or tokens[:depth] != root_tokens:
            continue
        row_index = tokens[depth]
        if not isinstance(row_index, int):
            continue
        row_indices.add(row_index)
        if len(tokens) != depth + 2:
            continue
        leaf = tokens[depth + 1]
        if not isinstance(leaf, str) or leaf not in keys:
            continue
        variants = identities.setdefault(row_index, {}).setdefault(leaf, [])
        for entry in iter_rule_evidence(rule):
            if entry.value not in variants:
                variants.append(entry.value)
    for row_index, entry in enumerate(iter_rule_evidence(parent)):
        row_value = entry.value
        if not isinstance(row_value, dict):
            continue
        row_identity = identities.setdefault(row_index, {})
        for key in keys:
            if key not in row_identity and key in row_value:
                row_identity[key] = [row_value[key]]
    return identities, row_indices


def _identity_combos(row_identity: dict[str, list[Any]], keys: list[str]) -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = [()]
    for key in keys:
        combos = [combo + (stable_value_key(variant),) for combo in combos for variant in row_identity[key]][
            :_IDENTITY_COMBO_CAP
        ]
    return combos


def _align_family_rows(
    identities: dict[int, dict[str, list[Any]]],
    row_indices: set[int],
    pred_rows: list[Any],
    keys: list[str],
    comparator: Any,
) -> dict[int, int]:
    used: set[int] = set()
    aligned: dict[int, int] = {}

    # Pass 1: exact normalized identity tuples, GT order claims duplicate
    # identities in prediction order so relative order is preserved.
    pred_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for pred_pos, row in enumerate(pred_rows):
        if isinstance(row, dict):
            pred_index[tuple(stable_value_key(row.get(key)) for key in keys)].append(pred_pos)
    pending: list[int] = []
    for row_index in sorted(row_indices):
        row_identity = identities.get(row_index)
        if not row_identity or any(key not in row_identity for key in keys):
            pending.append(row_index)
            continue
        matched = False
        for combo in _identity_combos(row_identity, keys):
            for pred_pos in pred_index.get(combo, []):
                if pred_pos in used:
                    continue
                aligned[row_index] = pred_pos
                used.add(pred_pos)
                matched = True
                break
            if matched:
                break
        if not matched:
            pending.append(row_index)

    # Pass 2: comparator-based matching for leftovers — catches surface-form
    # drift the exact pass misses (case, punctuation, near-miss strings).
    leftover_preds = [
        pred_pos for pred_pos in range(len(pred_rows)) if pred_pos not in used and isinstance(pred_rows[pred_pos], dict)
    ]
    if pending and leftover_preds and len(pending) * len(leftover_preds) <= _ALIGNMENT_PASS2_MAX_PAIRS:
        still_pending: list[int] = []
        for row_index in pending:
            row_identity = identities.get(row_index) or {}
            known_keys = [key for key in keys if key in row_identity]
            matched_pred: int | None = None
            if known_keys:
                for pred_pos in leftover_preds:
                    if pred_pos in used:
                        continue
                    pred_row = pred_rows[pred_pos]
                    if all(
                        any(
                            compare_evidence_value(
                                variant,
                                pred_row.get(key),
                                comparator.get(key) if isinstance(comparator, dict) else None,
                            ).passed
                            for variant in row_identity[key]
                        )
                        for key in known_keys
                    ):
                        matched_pred = pred_pos
                        break
            if matched_pred is None:
                still_pending.append(row_index)
            else:
                aligned[row_index] = matched_pred
                used.add(matched_pred)
        pending = still_pending

    # Pass 3: positional fallback. A GT row whose identity never matched keeps
    # today's exact-index behavior when that position is still unclaimed, so a
    # mis-extracted identity cell degrades to the legacy semantics instead of
    # grading the whole row as missing.
    for row_index in pending:
        if 0 <= row_index < len(pred_rows) and row_index not in used:
            aligned[row_index] = row_index
            used.add(row_index)
    return aligned


def _prediction_values_for_v02_rule(
    source: Any,
    rule: ExtractFieldTestRule,
    alignments: dict[str, dict[int, int]] | None = None,
) -> list[Any]:
    """Return exact-path predictions for v0.2 rules, preserving arrays/objects.

    When ``alignments`` covers the rule's array family, the first array index
    is translated from GT row space to predicted row space; an unaligned GT
    row resolves to a missing prediction.
    """
    if alignments:
        try:
            tokens = parse_field_path(rule.field_path)
        except ValueError:
            return []
        for position, token in enumerate(tokens):
            if not isinstance(token, int):
                continue
            family = ".".join(str(part) for part in tokens[:position])
            family_map = alignments.get(family)
            if family_map is not None:
                mapped = family_map.get(token)
                if mapped is None:
                    return []
                tokens[position] = mapped
            # Only the first array level carries match_by row identity.
            break
        value = get_path(source, tokens, default=_MISSING)
    else:
        value = _get_field_value(source, rule.field_path)
    if value is _MISSING:
        return []
    return [value]


def _compare_v02_prediction(rule: ExtractFieldTestRule, prediction: Any) -> ValueComparison:
    if rule.structural is not None or isinstance(prediction, list):
        return compare_array_against_rule(rule, prediction)
    return compare_value_against_rule(
        rule,
        prediction,
        source_kind="structured_value_no_citation_text",
    )


def _best_comparison(comparisons: Iterable[ValueComparison]) -> ValueComparison | None:
    best: ValueComparison | None = None
    for comparison in comparisons:
        if (
            best is None
            or (comparison.passed and not best.passed)
            or (comparison.passed == best.passed and comparison.score > best.score)
        ):
            best = comparison
    return best


def _best_v02_bbox_iou(
    evidence_entries: list[Any],
    pred_boxes_by_page: dict[int, list[BBox]],
    group: str,
) -> float | None:
    if not pred_boxes_by_page:
        return None
    best: float | None = None
    for ev in evidence_entries:
        if getattr(ev, "page", None) is None or getattr(ev, "bbox", None) is None:
            continue
        normalized = _as_xywh(ev.bbox)
        if normalized is None:
            continue
        # Predictions on other pages always score IoU 0.0 (the metric scopes
        # rectangles by page), so only same-page candidates need the full
        # computation and 0.0 is the floor once any (evidence, prediction)
        # pair exists.
        if best is None:
            best = 0.0
        gt = [BBox(page=ev.page, bbox=normalized, group=group)]
        for pred in pred_boxes_by_page.get(ev.page, ()):
            summary = compute_standard_iou_metrics(gt, [pred])
            if summary.iou > best:
                best = summary.iou
    return best


def _iter_values_for_pattern_with_paths(
    source: Any,
    pattern: Iterable[str | None],
) -> list[tuple[str, Any]]:
    cursors: list[tuple[Any, list[str | int]]] = [(source, [])]
    for token in pattern:
        next_cursors: list[tuple[Any, list[str | int]]] = []
        if token is None:
            for cursor, path in cursors:
                if isinstance(cursor, list):
                    next_cursors.extend((item, [*path, index]) for index, item in enumerate(cursor) if item is not None)
        else:
            for cursor, path in cursors:
                if isinstance(cursor, dict) and token in cursor:
                    next_cursors.append((cursor[token], [*path, token]))
        cursors = next_cursors
        if not cursors:
            return []

    return [
        (_format_field_path(path), cursor)
        for cursor, path in cursors
        if cursor is not None and not isinstance(cursor, (dict, list))
    ]


def _format_field_path(tokens: Iterable[str | int]) -> str:
    rendered = ""
    for token in tokens:
        if isinstance(token, int):
            rendered = f"{rendered}[{token}]"
        elif rendered:
            rendered = f"{rendered}.{token}"
        else:
            rendered = token
    return rendered


def _rules_by_field_pattern(
    rules: list[ExtractFieldTestRule],
) -> dict[tuple[str | None, ...], list[ExtractFieldTestRule]]:
    grouped: dict[tuple[str | None, ...], list[ExtractFieldTestRule]] = defaultdict(list)
    for rule in rules:
        pattern = _field_pattern(rule.field_path)
        if pattern is not None:
            grouped[pattern].append(rule)
    return grouped


def _match_value_group(
    rules: list[ExtractFieldTestRule],
    predictions: list[Any],
    *,
    data_schema: dict[str, Any] | None = None,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    _, _, matches, rule_results = _match_value_group_detailed(rules, predictions, data_schema=data_schema)
    return matches, rule_results


def _match_value_group_detailed(
    rules: list[ExtractFieldTestRule],
    predictions: list[Any],
    *,
    data_schema: dict[str, Any] | None = None,
) -> tuple[set[int], dict[int, ValueComparison], list[tuple[int, int]], list[dict[str, Any]]]:
    candidates: list[tuple[float, int, int, ValueComparison]] = []
    best_by_rule: dict[int, ValueComparison] = {}
    for rule_index, rule in enumerate(rules):
        expected_type = expected_type_for_field_path(data_schema, rule.field_path, rule.expected_value)
        for pred_index, prediction in enumerate(predictions):
            comparison = compare_value_against_rule(
                rule,
                prediction,
                expected_type=expected_type,
                source_kind="structured_value_no_citation_text",
            )
            if comparison.score > getattr(best_by_rule.get(rule_index), "score", -1.0):
                best_by_rule[rule_index] = comparison
            if comparison.passed:
                candidates.append((comparison.score, rule_index, pred_index, comparison))

    candidates.sort(key=lambda item: item[0], reverse=True)
    matched_rules: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[tuple[int, int]] = []
    match_comparisons: dict[int, ValueComparison] = {}
    for _, rule_index, pred_index, comparison in candidates:
        if rule_index in matched_rules or pred_index in matched_predictions:
            continue
        matched_rules.add(rule_index)
        matched_predictions.add(pred_index)
        matches.append((rule_index, pred_index))
        match_comparisons[rule_index] = comparison

    rule_results: list[dict[str, Any]] = []
    for rule_index, rule in enumerate(rules):
        final_comparison = match_comparisons.get(rule_index) or best_by_rule.get(rule_index)
        rule_results.append(
            {
                "field_path": rule.field_path,
                "field_pattern": ".".join(
                    "[]" if token is None else token for token in (_field_pattern(rule.field_path) or ())
                ),
                "passed": rule_index in matched_rules,
                "has_prediction": bool(predictions),
                "score": getattr(final_comparison, "score", 0.0),
                "mode": getattr(final_comparison, "mode", "missing"),
                "expected_type": expected_type_for_field_path(data_schema, rule.field_path, rule.expected_value),
                "attr_source": "structured_value_no_citation_text" if predictions else "missing",
                "comparator_version": COMPARATOR_VERSION,
                "reason": (
                    "pass" if rule_index in matched_rules else getattr(final_comparison, "reason", "missing_prediction")
                ),
            }
        )
    return matched_rules, match_comparisons, matches, rule_results


def _match_value_group_detailed_with_geometry(
    rules: list[ExtractFieldTestRule],
    path_predictions: list[tuple[str, Any]],
    *,
    candidate_pred_paths: list[str],
    citations_by_field_path: dict[str, list[BBox]],
    data_schema: dict[str, Any] | None = None,
) -> tuple[set[int], dict[int, ValueComparison], list[tuple[int, str]]]:
    """Select extract predictions index-tolerantly, using bbox fit first.

    Extract outputs often contain repeated values in record arrays. The
    grounded pass-rate metrics must follow parse semantics: select the
    predicted support by localization geometry, then evaluate attribution from
    the selected prediction's structured value. A value mismatch must not hide a
    valid localization match.
    """
    value_by_path = dict(path_predictions)
    fallback_values = [value for _, value in path_predictions]
    pred_paths = sorted({*candidate_pred_paths, *value_by_path})
    best_by_rule: dict[int, tuple[tuple[float, float, float, float, float, float], str, ValueComparison]] = {}

    for rule_index, rule in enumerate(rules):
        rule_iou_threshold = rule.iou_threshold
        expected_type = expected_type_for_field_path(data_schema, rule.field_path, rule.expected_value)
        group = _pattern_group(rule.field_path)
        if rule.evidence is not None:
            gt_boxes = [
                BBox(page=entry.page, bbox=normalized, group=group)
                for entry in rule.evidence
                if entry.page is not None
                and entry.bbox is not None
                and (normalized := _as_xywh(entry.bbox)) is not None
            ]
        else:
            gt_boxes = [
                BBox(page=bbox.page, bbox=normalized, group=group)
                for bbox in rule.bboxes
                if (normalized := _as_xywh(bbox.bbox)) is not None
            ]
        for pred_path in pred_paths:
            comparison = _compare_prediction_path_value(
                rule,
                pred_path=pred_path,
                value_by_path=value_by_path,
                fallback_values=fallback_values,
                expected_type=expected_type,
            )
            iou = 0.0
            max_ioa = 0.0
            area_delta = 1.0
            loc_candidate = False
            if gt_boxes:
                selected = _select_best_bbox_group(
                    gt_boxes,
                    citations_by_field_path.get(pred_path, []),
                    comparison=comparison,
                    strict_iou_threshold=rule_iou_threshold,
                )
                summary = compute_standard_iou_metrics(gt_boxes, selected)
                iou = summary.iou
                max_ioa = field_grounding_max_ioa(summary)
                area_delta = abs(summary.pred_area - summary.gt_area)
                loc_candidate = field_grounding_localization_passes(
                    iou=iou,
                    max_ioa=max_ioa,
                    comparison=comparison,
                    strict_iou_threshold=rule_iou_threshold,
                )

            key = (
                float(loc_candidate),
                iou,
                max_ioa,
                -area_delta,
                float(comparison.passed),
                comparison.score,
            )
            current = best_by_rule.get(rule_index)
            if current is None or key > current[0]:
                best_by_rule[rule_index] = (key, pred_path, comparison)

    selected_rules: set[int] = set(best_by_rule)
    matches: list[tuple[int, str]] = []
    match_comparisons: dict[int, ValueComparison] = {}
    for rule_index, (_, pred_path, comparison) in best_by_rule.items():
        matches.append((rule_index, pred_path))
        match_comparisons[rule_index] = comparison

    return selected_rules, match_comparisons, matches


def _compare_prediction_path_value(
    rule: ExtractFieldTestRule,
    *,
    pred_path: str,
    value_by_path: dict[str, Any],
    fallback_values: list[Any],
    expected_type: ExpectedType,
) -> ValueComparison:
    if pred_path in value_by_path:
        return compare_value_against_rule(
            rule,
            value_by_path[pred_path],
            expected_type=expected_type,
            source_kind="structured_value_no_citation_text",
        )

    best: ValueComparison | None = None
    for value in fallback_values:
        comparison = compare_value_against_rule(
            rule,
            value,
            expected_type=expected_type,
            source_kind="structured_value_no_citation_text",
        )
        if best is None or comparison.score > best.score:
            best = comparison
    return best or ValueComparison(passed=False, score=0.0, mode="missing", reason="missing_prediction")


def _record_signature(field_path: str) -> tuple[tuple[str | None, ...], int, tuple[str, ...]] | None:
    """Locate the innermost list index in a field path and split around it.

    Returns ``(list_pattern, gt_record_index, subpath)`` where:
    - ``list_pattern`` ends in a wildcard (``None``) standing in for the
      innermost list index — e.g. ``("employees", None)``.
    - ``gt_record_index`` is the integer index of the GT row.
    - ``subpath`` is the chain of string keys after the list index — e.g.
      ``("name",)`` for ``employees[3].name``.

    Returns ``None`` for scalar paths (no list index): those don't define a
    record and are skipped by record-level metrics.
    """
    try:
        tokens = parse_field_path(field_path)
    except ValueError:
        return None
    last_int_idx = -1
    for index, token in enumerate(tokens):
        if isinstance(token, int):
            last_int_idx = index
    if last_int_idx == -1:
        return None
    list_pattern = tuple(None if isinstance(t, int) else t for t in tokens[: last_int_idx + 1])
    gt_index = tokens[last_int_idx]
    if not isinstance(gt_index, int):
        return None
    subpath = tuple(t for t in tokens[last_int_idx + 1 :] if isinstance(t, str))
    return list_pattern, gt_index, subpath


def _iter_records_for_pattern(source: Any, list_pattern: tuple[str | None, ...]) -> list[tuple[int, Any]]:
    """Walk extracted_data to the list under ``list_pattern`` and enumerate dict items.

    Only dict items count as records. ``None`` slots and scalar items are
    silently skipped — they can't carry per-record fields and shouldn't
    contribute to the precision denominator.
    """
    cursors: list[Any] = [source]
    for token in list_pattern[:-1]:
        next_cursors: list[Any] = []
        if token is None:
            for cursor in cursors:
                if isinstance(cursor, list):
                    next_cursors.extend(c for c in cursor if c is not None)
        else:
            for cursor in cursors:
                if isinstance(cursor, dict) and token in cursor:
                    next_cursors.append(cursor[token])
        cursors = next_cursors
        if not cursors:
            return []

    out: list[tuple[int, Any]] = []
    for cursor in cursors:
        if not isinstance(cursor, list):
            continue
        for index, item in enumerate(cursor):
            if isinstance(item, dict):
                out.append((index, item))
    return out


def _record_field_value(record: Any, subpath: tuple[str, ...]) -> Any:
    cursor: Any = record
    for token in subpath:
        if not isinstance(cursor, dict) or token not in cursor:
            return _MISSING
        cursor = cursor[token]
    return cursor


def _xywh_intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = ax1 + a[2], ay1 + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = bx1 + b[2], by1 + b[3]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


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


def _is_field_grounded(
    gt_bboxes: Iterable[Any],
    pred_citations: Iterable[Any],
    *,
    threshold: float,
) -> bool:
    """A field is grounded if every GT bbox is covered by some pred citation.

    "Covered" means ``intersection / GT_area >= threshold`` on the same page —
    same recall-shaped check as the existing IoU metric, just per-field.
    A field with no GT bboxes is treated as N/A (grounded by default).
    """
    gt_list = list(gt_bboxes)
    if not gt_list:
        return True
    cit_list = list(pred_citations)
    if not cit_list:
        return False
    for gt in gt_list:
        gt_xywh = _as_xywh(gt.bbox)
        if gt_xywh is None:
            continue
        gt_area = gt_xywh[2] * gt_xywh[3]
        if gt_area <= 0.0:
            continue
        covered = False
        for citation in cit_list:
            if _as_int(getattr(citation, "page", None)) != gt.page:
                continue
            cit_xywh = _as_xywh(getattr(citation, "bbox", None))
            if cit_xywh is None:
                continue
            if _xywh_intersection_area(gt_xywh, cit_xywh) / gt_area >= threshold:
                covered = True
                break
        if not covered:
            return False
    return True


def _compute_record_metrics(
    field_rules: list[ExtractFieldTestRule],
    extracted_data: Any,
    field_citations: list[Any],
    *,
    bbox_overlap_threshold: float = 0.5,
    data_schema: dict[str, Any] | None = None,
) -> list[MetricValue]:
    """Compute strict record-level precision / recall / F1 plus grounded recall.

    Algorithm
    ---------
    1. Skip stray rules and scalar (non-record) rules.
    2. Group GT rules by ``(list_pattern, gt_record_index)``; group pred dict
       items at the same list pattern by their actual list index.
    3. For each list pattern, build an overlap matrix (number of non-null GT
       fields whose value matches the corresponding pred record's value).
    4. Greedy bipartite alignment by descending overlap, with majority threshold
       (overlap > half the non-null GT field count) — soft alignment.
    5. **Strict TP**: an aligned pair counts as a true positive iff *every*
       non-null GT field passes ``compare_field_value`` against its pred. A
       value-swap row will fail strict even if alignment succeeded.
    6. **Grounded TP** (subset of text TP): also require every non-null GT
       field with bboxes to have a pred citation overlapping by
       ``intersection / GT_area >= threshold`` on the same page.

    Records with no non-null GT fields (all-stray) are excluded from the GT
    denominator. Pred records that are non-dict or empty contribute to the
    pred denominator iff they appear at the relevant list pattern as dict
    items — empty-dict spam thus correctly hurts precision.
    """
    citations_by_record_field: dict[tuple[tuple[str | None, ...], int, tuple[str, ...]], list[Any]] = defaultdict(list)
    for citation in field_citations:
        field_path = getattr(citation, "field_path", None)
        if not field_path:
            continue
        signature = _record_signature(field_path)
        if signature is None:
            continue
        citations_by_record_field[signature].append(citation)

    gt_by_pattern: dict[tuple[str | None, ...], dict[int, list[tuple[tuple[str, ...], ExtractFieldTestRule]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for rule in field_rules:
        if _is_stray_rule(rule):
            continue
        signature = _record_signature(rule.field_path)
        if signature is None:
            continue
        list_pattern, gt_index, subpath = signature
        gt_by_pattern[list_pattern][gt_index].append((subpath, rule))

    if not gt_by_pattern:
        return []

    text_tp = 0
    grounded_tp = 0
    grounded_gt = 0  # eligible GT records with at least one bbox-bearing field
    total_gt = 0
    total_pred = 0

    for list_pattern, gt_records in gt_by_pattern.items():
        pred_records = _iter_records_for_pattern(extracted_data, list_pattern)
        gt_field_counts: dict[int, int] = {}
        passes: dict[tuple[int, int], dict[tuple[str, ...], bool]] = {}
        for gt_index, fields in gt_records.items():
            non_null_fields = [(sub, rule) for sub, rule in fields if rule.expected_value is not None]
            gt_field_counts[gt_index] = len(non_null_fields)
            if not non_null_fields:
                continue
            for pred_index, pred_record in pred_records:
                field_passes: dict[tuple[str, ...], bool] = {}
                for subpath, rule in non_null_fields:
                    actual = _record_field_value(pred_record, subpath)
                    if actual is _MISSING:
                        field_passes[subpath] = False
                        continue
                    expected_type = expected_type_for_field_path(data_schema, rule.field_path, rule.expected_value)
                    field_passes[subpath] = compare_attributed_value(
                        rule.expected_value,
                        actual,
                        expected_type=expected_type,
                        source_kind="structured_value_no_citation_text",
                    ).passed
                passes[(gt_index, pred_index)] = field_passes

        eligible_gt_indices = [g for g, count in gt_field_counts.items() if count > 0]
        total_gt += len(eligible_gt_indices)
        grounded_gt += sum(1 for g in eligible_gt_indices if _record_has_bbox_field(gt_records[g]))
        total_pred += len(pred_records)

        edges = sorted(
            ((gt_index, pred_index, sum(p.values())) for (gt_index, pred_index), p in passes.items()),
            key=lambda item: item[2],
            reverse=True,
        )
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for gt_index, pred_index, overlap in edges:
            if gt_index in used_gt or pred_index in used_pred:
                continue
            field_count = gt_field_counts[gt_index]
            if overlap * 2 <= field_count:
                continue
            used_gt.add(gt_index)
            used_pred.add(pred_index)
            field_passes = passes[(gt_index, pred_index)]
            if not all(field_passes.values()):
                continue
            text_tp += 1
            # record_grounded_recall is defined only over records whose GT
            # carries a bbox. ``_is_record_grounded`` returns True vacuously for
            # a record with no bbox field, so gate on real bbox presence to keep
            # un-gradeable records out of grounded_tp (and grounded_gt below).
            if _record_has_bbox_field(gt_records[gt_index]) and _is_record_grounded(
                gt_records[gt_index],
                list_pattern=list_pattern,
                pred_index=pred_index,
                citations_by_record_field=citations_by_record_field,
                threshold=bbox_overlap_threshold,
            ):
                grounded_tp += 1

    if total_gt == 0 and total_pred == 0:
        return []

    fp = max(total_pred - text_tp, 0)
    fn = max(total_gt - text_tp, 0)
    precision = text_tp / total_pred if total_pred > 0 else 0.0
    recall = text_tp / total_gt if total_gt > 0 else 0.0
    f1 = _harmonic_mean(precision, recall)
    union = text_tp + fp + fn
    accuracy = text_tp / union if union > 0 else 0.0
    grounded_recall = grounded_tp / grounded_gt if grounded_gt > 0 else 0.0

    metadata = {
        "tp": text_tp,
        "fp": fp,
        "fn": fn,
        "total_gt_records": total_gt,
        "total_pred_records": total_pred,
        "grounded_tp": grounded_tp,
        "grounded_gt_records": grounded_gt,
        "alignment_threshold": "majority",
        "bbox_overlap_threshold": bbox_overlap_threshold,
        "accuracy_definition": "tp / (tp + fp + fn)",
    }
    metrics = [
        MetricValue(metric_name="record_precision", value=precision, metadata=metadata),
        MetricValue(metric_name="record_recall", value=recall, metadata=metadata),
        MetricValue(metric_name="record_f1", value=f1, metadata=metadata),
        MetricValue(metric_name="record_accuracy", value=accuracy, metadata=metadata),
    ]
    # record_grounded_recall is a grounding metric: only emit it when the GT
    # actually carries record-level bboxes, so documents with no grounded GT are
    # excluded from its average instead of contributing a vacuous value.
    if grounded_gt > 0:
        metrics.append(MetricValue(metric_name="record_grounded_recall", value=grounded_recall, metadata=metadata))
    return metrics


def _record_has_bbox_field(gt_fields: list[tuple[tuple[str, ...], ExtractFieldTestRule]]) -> bool:
    """True when a GT record has at least one non-null field carrying a bbox.

    Only such records are grounding-gradeable, so they alone form the
    ``record_grounded_recall`` denominator (``_is_record_grounded`` otherwise
    passes vacuously for a record with no bbox field).
    """
    return any(rule.expected_value is not None and rule.bboxes for _subpath, rule in gt_fields)


def _is_record_grounded(
    gt_fields: list[tuple[tuple[str, ...], ExtractFieldTestRule]],
    *,
    list_pattern: tuple[str | None, ...],
    pred_index: int,
    citations_by_record_field: dict[tuple[tuple[str | None, ...], int, tuple[str, ...]], list[Any]],
    threshold: float,
) -> bool:
    for subpath, rule in gt_fields:
        if rule.expected_value is None:
            continue
        if not rule.bboxes:
            continue
        citations = citations_by_record_field.get((list_pattern, pred_index, subpath), [])
        if not _is_field_grounded(rule.bboxes, citations, threshold=threshold):
            return False
    return True


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


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _harmonic_mean(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
