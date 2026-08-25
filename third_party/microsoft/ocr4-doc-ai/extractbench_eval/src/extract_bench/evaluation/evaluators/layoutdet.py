"""Evaluator for LAYOUT_DETECTION product type."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from extract_bench.evaluation.evaluators.base import BaseEvaluator
from extract_bench.evaluation.layout_adapters import create_layout_adapter_for_result
from extract_bench.evaluation.layout_label_mappers import project_layout_predictions
from extract_bench.evaluation.metrics.attribution.constants import (
    ATTRIBUTION_OVERLAP_IOA_THRESHOLD,
    ATTRIBUTION_TOKEN_F1_THRESHOLD,
    LOCALIZATION_IOA_PRED_THRESHOLD,
    LOCALIZATION_IOA_THRESHOLD,
)
from extract_bench.evaluation.metrics.attribution.core import (
    GTElement,
    PredBlock,
    compute_attribution_metrics,
    gt_element_is_explicit,
    gt_element_skips_attribution,
    is_truthy,
    layout_element_is_formula,
    normalize_layout_attributes,
    parse_gt_elements,
)
from extract_bench.evaluation.metrics.attribution.geometry import compute_ioa_matrix
from extract_bench.evaluation.metrics.layoutdet.classification_utils import (
    compute_map_at_thresholds,
    compute_per_class_metrics,
)
from extract_bench.evaluation.metrics.layoutdet.iou import (
    compute_iou_matrix,
)
from extract_bench.evaluation.stats import build_operational_stats
from extract_bench.layout_label_mapping import (
    map_label_to_target_ontology,
    normalize_evaluation_ontology,
)
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue
from extract_bench.schemas.layout_detection_output import LayoutOutput
from extract_bench.schemas.layout_ontology import CORE_LABELS, CanonicalLabel
from extract_bench.schemas.metrics import ConfusionMatrixMetrics
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.schema import LayoutDetectionTestCase, TestCase

# Core11 class names for evaluation
CORE11_CLASS_NAMES = [label.value for label in CORE_LABELS]
_PAGE_FURNITURE_CLASSES: frozenset[str] = frozenset(
    {CanonicalLabel.PAGE_HEADER.value, CanonicalLabel.PAGE_FOOTER.value}
)
_PAGE_FURNITURE_X_SPAN_COVERAGE_THRESHOLD = 0.80
_PAGE_FURNITURE_Y_COVERAGE_THRESHOLD = 0.50


@dataclass
class _PageFurnitureGroup:
    pred_indices: list[int]
    clipped_boxes: list[list[float]]
    representative_pred_idx: int | None
    earliest_order_index: int | None
    x_span_coverage: float = 0.0
    x_fill_coverage: float = 0.0
    y_coverage: float = 0.0
    label_histogram: dict[str, int] = field(default_factory=dict)


@dataclass
class _PageFurnitureAttributionMatch:
    overlapping_indices: list[int]
    selected_indices: list[int]
    representative_pred_idx: int | None
    selected_tokens: list[str]
    selected_text_norm: str | None
    precision: float
    recall: float
    f1: float


def _is_page_furniture(canonical_class: str | None) -> bool:
    """Return True for GT page furniture classes."""
    return str(canonical_class or "").strip() in _PAGE_FURNITURE_CLASSES


def _clip_box_to_box(box: list[float], boundary: list[float]) -> list[float] | None:
    """Return the clipped intersection box, or None when there is no overlap."""
    x1 = max(box[0], boundary[0])
    y1 = max(box[1], boundary[1])
    x2 = min(box[2], boundary[2])
    y2 = min(box[3], boundary[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _interval_union_length(intervals: list[tuple[float, float]]) -> float:
    """Return the total covered length of 1D intervals."""
    merged = sorted((start, end) for start, end in intervals if end > start)
    if not merged:
        return 0.0

    total = 0.0
    curr_start, curr_end = merged[0]
    for start, end in merged[1:]:
        if start <= curr_end:
            curr_end = max(curr_end, end)
            continue
        total += curr_end - curr_start
        curr_start, curr_end = start, end
    total += curr_end - curr_start
    return total


def _compute_page_furniture_band_coverage(
    gt_box: list[float],
    clipped_boxes: list[list[float]],
) -> tuple[float, float, float]:
    """Return normalized horizontal and vertical recovery of a GT furniture band."""
    gt_width = max(gt_box[2] - gt_box[0], 0.0)
    gt_height = max(gt_box[3] - gt_box[1], 0.0)
    if gt_width <= 0.0 or gt_height <= 0.0 or not clipped_boxes:
        return 0.0, 0.0, 0.0

    x_span_coverage = (max(box[2] for box in clipped_boxes) - min(box[0] for box in clipped_boxes)) / gt_width
    x_fill_coverage = _interval_union_length([(box[0], box[2]) for box in clipped_boxes]) / gt_width
    y_coverage = _interval_union_length([(box[1], box[3]) for box in clipped_boxes]) / gt_height
    return min(x_span_coverage, 1.0), min(x_fill_coverage, 1.0), min(y_coverage, 1.0)


def _build_page_furniture_group(
    *,
    gt_box: list[float],
    gt_idx: int,
    pred_boxes: list[list[float]],
    ioa_pred_to_gt: np.ndarray | None,
    iou_row: np.ndarray | None = None,
    pred_order_indices: list[int] | None = None,
    pred_classes: list[str | None] | None = None,
) -> _PageFurnitureGroup:
    """Group predictions that recover a page-header/footer GT band."""
    if ioa_pred_to_gt is None or not pred_boxes:
        return _PageFurnitureGroup([], [], None, None)

    candidate_indices = [
        int(pred_idx) for pred_idx in np.where(ioa_pred_to_gt[:, gt_idx] >= LOCALIZATION_IOA_PRED_THRESHOLD)[0]
    ]

    retained_indices: list[int] = []
    clipped_boxes: list[list[float]] = []
    for pred_idx in candidate_indices:
        clipped = _clip_box_to_box(pred_boxes[pred_idx], gt_box)
        if clipped is None:
            continue
        retained_indices.append(pred_idx)
        clipped_boxes.append(clipped)

    if not retained_indices:
        return _PageFurnitureGroup([], [], None, None)

    representative_pred_idx = retained_indices[0]
    if iou_row is not None:
        representative_pred_idx = int(retained_indices[np.argmax(iou_row[retained_indices])])

    if pred_order_indices is None:
        earliest_order_index = min(retained_indices)
    else:
        earliest_order_index = min(pred_order_indices[pred_idx] for pred_idx in retained_indices)

    label_histogram: dict[str, int] = {}
    if pred_classes is not None:
        label_histogram = dict(
            Counter(str(pred_classes[pred_idx]) for pred_idx in retained_indices if pred_classes[pred_idx] is not None)
        )

    x_span_coverage, x_fill_coverage, y_coverage = _compute_page_furniture_band_coverage(gt_box, clipped_boxes)
    return _PageFurnitureGroup(
        pred_indices=retained_indices,
        clipped_boxes=clipped_boxes,
        representative_pred_idx=representative_pred_idx,
        earliest_order_index=earliest_order_index,
        x_span_coverage=x_span_coverage,
        x_fill_coverage=x_fill_coverage,
        y_coverage=y_coverage,
        label_histogram=label_histogram,
    )


def _multiset_intersection_size(a: list[str], b: list[str]) -> int:
    """Compute the size of the multiset intersection of two token lists."""
    counter_a = Counter(a)
    counter_b = Counter(b)
    return sum((counter_a & counter_b).values())


def _multiset_difference_sample(a: list[str], b: list[str], limit: int) -> list[str]:
    """Return up to `limit` unique tokens from multiset(a - b)."""
    if limit <= 0:
        return []
    counter_a = Counter(a)
    counter_b = Counter(b)
    remaining = counter_a - counter_b
    sample: list[str] = []
    seen: set[str] = set()
    for token in remaining.elements():
        if token in seen:
            continue
        seen.add(token)
        sample.append(token)
        if len(sample) >= limit:
            break
    return sample


def _multiset_difference(a: list[str], b: list[str]) -> list[str]:
    """Return multiset(a - b) as a list."""
    counter_a = Counter(a)
    counter_b = Counter(b)
    return list((counter_a - counter_b).elements())


def _compute_token_f1(gt_tokens: list[str], pred_tokens: list[str]) -> float:
    """Compute token-level F1 for attribution pass/fail."""
    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0
    matched = _multiset_intersection_size(gt_tokens, pred_tokens)
    precision = matched / len(pred_tokens) if pred_tokens else 0.0
    recall = matched / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _compute_token_metrics(gt_tokens: list[str], pred_tokens: list[str]) -> tuple[float, float, float]:
    """Return token precision, recall, and F1 for a GT/pred token pair."""
    if not gt_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0
    if not gt_tokens:
        return 0.0, 1.0, 0.0
    if not pred_tokens:
        return 0.0, 0.0, 0.0

    matched = _multiset_intersection_size(gt_tokens, pred_tokens)
    precision = matched / len(pred_tokens)
    recall = matched / len(gt_tokens)
    return precision, recall, _compute_token_f1(gt_tokens, pred_tokens)


def _coerce_int(value: Any) -> int | None:
    """Return an int value when safely representable, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _score_local_reading_order(rule_results: list[dict[str, Any]], max_neighbor_distance: int = 3) -> tuple[int, int]:
    """Score reading-order correctness with a bounded local neighborhood.

    Eligibility gate intentionally ignores classification:
    - localization must pass
    - attribution must pass

    For each eligible element, compare against up to ``max_neighbor_distance``
    eligible elements before and after in GT reading order (per page).
    """
    if max_neighbor_distance < 1:
        raise ValueError("max_neighbor_distance must be >= 1")

    if not rule_results:
        return 0, 0

    total = 0
    eligible_by_page: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for fallback_index, raw in enumerate(rule_results):
        localization_pass = raw.get("localization_pass") is True
        attribution_pass = raw.get("attribution_pass") is True
        eligible = localization_pass and attribution_pass

        raw["reading_order_eligible"] = eligible
        raw["reading_order_pass"] = False
        if not eligible:
            if not localization_pass:
                raw["reading_order_reason"] = "ineligible_no_localization"
            else:
                attribution_reason = raw.get("attribution_reason")
                if attribution_reason in {"caption_skip", "formula_skip", "no_gt_content"}:
                    raw["reading_order_reason"] = f"ineligible_{attribution_reason}"
                else:
                    raw["reading_order_reason"] = "ineligible_no_attribution"
            continue

        total += 1
        page = _coerce_int(raw.get("page"))
        gt_ro_index = _coerce_int(raw.get("gt_ro_index"))
        pred_order_index = _coerce_int(raw.get("matched_pred_order_index"))
        element_index = _coerce_int(raw.get("element_index"))
        if element_index is None:
            element_index = fallback_index

        if page is None:
            raw["reading_order_reason"] = "missing_page"
            continue
        if gt_ro_index is None:
            raw["reading_order_reason"] = "missing_ro_index"
            continue
        if pred_order_index is None:
            raw["reading_order_reason"] = "missing_pred_order_index"
            continue

        eligible_by_page[page].append((fallback_index, gt_ro_index, element_index, pred_order_index))

    passed = 0
    for page_entries in eligible_by_page.values():
        page_entries.sort(key=lambda item: (item[1], item[2]))
        for curr_pos, curr in enumerate(page_entries):
            curr_idx, _curr_ro, _curr_el_idx, curr_pred_order = curr
            curr_row = rule_results[curr_idx]

            has_neighbors = False
            order_violation = False

            for distance in range(1, max_neighbor_distance + 1):
                before_pos = curr_pos - distance
                if before_pos >= 0:
                    has_neighbors = True
                    _n_idx, _n_ro, _n_el_idx, neighbor_pred_order = page_entries[before_pos]
                    if neighbor_pred_order >= curr_pred_order:
                        order_violation = True
                        curr_row["reading_order_reason"] = "before_not_before"
                        break

                after_pos = curr_pos + distance
                if after_pos < len(page_entries):
                    has_neighbors = True
                    _n_idx, _n_ro, _n_el_idx, neighbor_pred_order = page_entries[after_pos]
                    if curr_pred_order >= neighbor_pred_order:
                        order_violation = True
                        curr_row["reading_order_reason"] = "after_not_after"
                        break

            if order_violation:
                continue

            if not has_neighbors:
                curr_row["reading_order_reason"] = "no_local_neighbors"
                continue

            curr_row["reading_order_pass"] = True
            curr_row["reading_order_reason"] = "pass"
            passed += 1

    return passed, total


def _merge_aware_pred_tokens(
    gt_idx: int,
    pred_idx: int,
    gt_elements: list[GTElement],
    pred_blocks: list[PredBlock],
    ioa_attr: np.ndarray | None,
) -> list[str]:
    """Remove tokens belonging only to other overlapping GT elements."""
    pred_tokens = pred_blocks[pred_idx].tokens
    if not pred_tokens or ioa_attr is None:
        return pred_tokens

    overlapping_gt_indices = np.where(ioa_attr[:, pred_idx] >= ATTRIBUTION_OVERLAP_IOA_THRESHOLD)[0]
    other_tokens: list[str] = []
    for other_gt_idx in overlapping_gt_indices:
        if other_gt_idx == gt_idx:
            continue
        other_tokens.extend(gt_elements[other_gt_idx].tokens)

    if not other_tokens:
        return pred_tokens

    other_only_tokens = _multiset_difference(other_tokens, gt_elements[gt_idx].tokens)
    if not other_only_tokens:
        return pred_tokens
    return _multiset_difference(pred_tokens, other_only_tokens)


def _select_best_attribution_match(
    *,
    gt_idx: int,
    gt_elements: list[GTElement],
    pred_blocks: list[PredBlock],
    ioa_attr: np.ndarray | None,
    iou_attr: np.ndarray | None,
    scoring: Literal["f1", "recall"],
) -> tuple[list[int], int | None, list[str], float, float, float]:
    """Return the best overlapping prediction for attribution scoring."""
    if pred_blocks and ioa_attr is not None:
        overlapping = [int(idx) for idx in np.where(ioa_attr[gt_idx, :] >= ATTRIBUTION_OVERLAP_IOA_THRESHOLD)[0]]
    else:
        overlapping = []

    best_pred_idx = None
    best_tokens: list[str] = []
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_score = -1.0
    best_iou = -1.0

    for pred_idx in overlapping:
        pred_tokens = _merge_aware_pred_tokens(
            gt_idx=gt_idx,
            pred_idx=pred_idx,
            gt_elements=gt_elements,
            pred_blocks=pred_blocks,
            ioa_attr=ioa_attr,
        )
        precision, recall, f1 = _compute_token_metrics(gt_elements[gt_idx].tokens, pred_tokens)
        score = recall if scoring == "recall" else f1
        iou_score = float(iou_attr[gt_idx, pred_idx]) if iou_attr is not None else 0.0
        if score > best_score or (score == best_score and iou_score > best_iou):
            best_pred_idx = pred_idx
            best_tokens = pred_tokens
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_score = score
            best_iou = iou_score

    return overlapping, best_pred_idx, best_tokens, best_precision, best_recall, best_f1


def _select_page_furniture_attribution_match(
    *,
    gt_idx: int,
    gt_elements: list[GTElement],
    pred_blocks: list[PredBlock],
    ioa_attr: np.ndarray | None,
    ioa_attr_pred: np.ndarray | None,
    iou_attr: np.ndarray | None,
    scoring: Literal["f1", "recall"],
) -> _PageFurnitureAttributionMatch:
    """Select the best contiguous ordered span inside a grouped furniture band."""
    pred_boxes = [pred.bbox_xyxy for pred in pred_blocks]
    group = _build_page_furniture_group(
        gt_box=gt_elements[gt_idx].bbox_xyxy,
        gt_idx=gt_idx,
        pred_boxes=pred_boxes,
        ioa_pred_to_gt=ioa_attr_pred,
        iou_row=iou_attr[gt_idx] if iou_attr is not None else None,
        pred_order_indices=[pred.order_index for pred in pred_blocks],
    )

    if not group.pred_indices:
        return _PageFurnitureAttributionMatch([], [], None, [], None, 0.0, 0.0, 0.0)

    ordered_indices = sorted(group.pred_indices, key=lambda pred_idx: pred_blocks[pred_idx].order_index)

    tokens_by_pred_idx = {
        pred_idx: _merge_aware_pred_tokens(
            gt_idx=gt_idx,
            pred_idx=pred_idx,
            gt_elements=gt_elements,
            pred_blocks=pred_blocks,
            ioa_attr=ioa_attr,
        )
        for pred_idx in ordered_indices
    }

    best_selected_indices: list[int] = []
    best_tokens: list[str] = []
    best_text_norm: str | None = None
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_score = -1.0
    best_secondary_score = -1.0
    best_span_length = float("inf")
    best_representative_pred_idx = None
    best_representative_iou = -1.0

    for start_idx in range(len(ordered_indices)):
        span_indices: list[int] = []
        span_tokens: list[str] = []
        span_representative_pred_idx = None
        span_representative_iou = -1.0
        for pred_idx in ordered_indices[start_idx:]:
            span_indices.append(pred_idx)
            span_tokens.extend(tokens_by_pred_idx[pred_idx])
            pred_iou = float(iou_attr[gt_idx, pred_idx]) if iou_attr is not None else 0.0
            if pred_iou > span_representative_iou:
                span_representative_iou = pred_iou
                span_representative_pred_idx = pred_idx

            precision, recall, f1 = _compute_token_metrics(gt_elements[gt_idx].tokens, span_tokens)
            score = recall if scoring == "recall" else f1
            secondary_score = f1 if scoring == "recall" else recall
            span_length = len(span_indices)

            should_update = False
            if score > best_score:
                should_update = True
            elif score == best_score and secondary_score > best_secondary_score:
                should_update = True
            elif score == best_score and secondary_score == best_secondary_score and span_length < best_span_length:
                should_update = True
            elif (
                score == best_score
                and secondary_score == best_secondary_score
                and span_length == best_span_length
                and span_representative_iou > best_representative_iou
            ):
                should_update = True

            if should_update:
                best_selected_indices = list(span_indices)
                best_tokens = list(span_tokens)
                best_text_norm = " ".join(best_tokens).strip() or None
                best_precision = precision
                best_recall = recall
                best_f1 = f1
                best_score = score
                best_secondary_score = secondary_score
                best_span_length = span_length
                best_representative_pred_idx = span_representative_pred_idx
                best_representative_iou = span_representative_iou

    return _PageFurnitureAttributionMatch(
        overlapping_indices=ordered_indices,
        selected_indices=best_selected_indices,
        representative_pred_idx=best_representative_pred_idx,
        selected_tokens=best_tokens,
        selected_text_norm=best_text_norm,
        precision=best_precision,
        recall=best_recall,
        f1=best_f1,
    )


def coco_normalized_to_xyxy_normalized(bbox: list[float]) -> list[float]:
    """Convert normalized COCO bbox to normalized xyxy format.

    :param bbox: Normalized bbox in [x, y, w, h] format
    :return: Normalized bbox in [x1, y1, x2, y2] format
    """
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


class LayoutDetectionEvaluator(BaseEvaluator):
    """
    Evaluator for LAYOUT_DETECTION product type.

    Computes:
    - mAP@[.50:.95], AP50, AP75 (COCO-style)
    - Per-class precision/recall/F1 at IoU=0.5

    Supports two evaluation views:
    - Core11: Required for all models (DocLayNet-compatible)
    - Canonical17: Optional where ground-truth is available
    """

    def __init__(
        self,
        iou_thresholds: list[float] | None = None,
        evaluation_view: Literal["core", "canonical"] = "core",
        default_ontology: str = "basic",
    ):
        """
        Initialize the layout detection evaluator.

        :param iou_thresholds: IoU thresholds for mAP computation
                               (default: [0.5, 0.55, ..., 0.95])
        :param evaluation_view: Label view for evaluation outputs:
                               - "core": Core11 (DocLayNet-compatible)
                               - "canonical": Canonical17
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5 + i * 0.05 for i in range(10)]
        self._iou_thresholds = iou_thresholds
        self._evaluation_view = evaluation_view
        self._default_ontology = normalize_evaluation_ontology(default_ontology)

    def can_evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> bool:
        """
        Check if this evaluator can evaluate the given inference result and test case.

        :param inference_result: The inference result to evaluate
        :param test_case: The test case to evaluate against
        :return: True if this evaluator can handle this case
        """
        # Must be LAYOUT_DETECTION product type
        if inference_result.product_type != ProductType.LAYOUT_DETECTION:
            return False

        # Must have LayoutOutput
        if not isinstance(inference_result.output, LayoutOutput):
            return False

        # Must be LayoutDetectionTestCase
        if not isinstance(test_case, LayoutDetectionTestCase):
            return False

        # Must have layout annotations (from test_rules)
        if not test_case.get_layout_annotations():
            return False

        return True

    def _extract_predictions(
        self,
        inference_result: InferenceResult,
        output: LayoutOutput,
        *,
        target_ontology: str,
        page_filter: int | None = None,
    ) -> list[dict]:
        """
        Extract predictions in evaluation format, normalized to [0,1] space.

        :param inference_result: Source inference result
        :param output: Unified layout output from inference
        :param target_ontology: Target ontology for this evaluation
        :param page_filter: Optional 1-indexed page number to filter predictions.
                           If provided, only predictions from this page are returned.
                           If None, all predictions are returned (for single-page docs).
        :return: List of dicts with 'bbox' (normalized xyxy), 'class_name', 'score'
        """
        effective_view = self._resolve_effective_evaluation_view(target_ontology)
        return project_layout_predictions(
            inference_result,
            output,
            evaluation_view=effective_view,
            target_ontology=target_ontology,
            page_filter=page_filter,
        )

    def _extract_ground_truth(self, test_case: LayoutDetectionTestCase, *, target_ontology: str) -> list[dict]:
        """
        Extract ground truth in evaluation format.

        GT bboxes are in normalized COCO format [x, y, width, height] in [0,1] range.
        Converts to normalized xyxy format [x1, y1, x2, y2] in [0,1] range.

        :param test_case: Test case with layout annotations
        :return: List of dicts with 'bbox' (normalized xyxy), 'class_name'
        """
        ground_truth: list[dict] = []
        effective_view = self._resolve_effective_evaluation_view(target_ontology)

        # Get layout annotations from test_rules
        annotations = test_case.get_layout_annotations()

        for annotation in annotations:
            # Convert normalized COCO format to normalized xyxy format
            bbox_xyxy = coco_normalized_to_xyxy_normalized(annotation.bbox)

            # Map canonical_class to the appropriate view
            class_name = annotation.canonical_class

            # For core view, check if class is in Core11
            if effective_view == "core":
                try:
                    canonical_label = CanonicalLabel(class_name)
                    if canonical_label not in CORE_LABELS:
                        # Skip non-Core11 classes in core evaluation
                        continue
                except ValueError:
                    # Unknown class, skip
                    continue

            ground_truth.append(
                {
                    "bbox": bbox_xyxy,
                    "class_name": class_name,
                }
            )

        return ground_truth

    def _get_class_names(self, ground_truth: list[dict]) -> list[str]:
        """
        Get unique class names from ground truth.

        :param ground_truth: List of ground truth dicts
        :return: Sorted list of unique class names
        """
        return sorted({g["class_name"] for g in ground_truth})

    def _resolve_target_ontology(self, test_case: LayoutDetectionTestCase) -> str:
        """Resolve target ontology with precedence: test_case > runner/CLI > default."""
        return normalize_evaluation_ontology(test_case.ontology or self._default_ontology)

    def _resolve_effective_evaluation_view(self, target_ontology: str) -> Literal["core", "canonical"]:
        """Use canonical view when scoring in the collapsed Basic ontology."""
        if normalize_evaluation_ontology(target_ontology) == "basic":
            return "canonical"
        return self._evaluation_view

    def evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> EvaluationResult:
        """
        Evaluate a layout detection inference result against a test case.

        :param inference_result: The inference result to evaluate
        :param test_case: The test case with layout annotations
        :return: Evaluation result with metrics
        :raises ValueError: If evaluation cannot be performed
        """
        if not self.can_evaluate(inference_result, test_case):
            raise ValueError("Cannot evaluate: missing layout_annotations or invalid product type")

        if not isinstance(inference_result.output, LayoutOutput):
            raise ValueError("Inference result output is not LayoutOutput")

        if not isinstance(test_case, LayoutDetectionTestCase):
            raise ValueError("Test case must be LayoutDetectionTestCase for LAYOUT_DETECTION evaluation")

        adapter = create_layout_adapter_for_result(inference_result)
        layout_output: LayoutOutput = adapter.to_layout_output(inference_result)
        target_ontology = self._resolve_target_ontology(test_case)
        effective_view = self._resolve_effective_evaluation_view(target_ontology)
        predictions = self._extract_predictions(
            inference_result,
            layout_output,
            target_ontology=target_ontology,
        )
        ground_truth = self._extract_ground_truth(test_case, target_ontology=target_ontology)

        normalized_ground_truth = [
            {
                **gt,
                "class_name": map_label_to_target_ontology(
                    gt.get("class_name"),
                    target_ontology,
                ),
            }
            for gt in ground_truth
        ]

        # Get class names from ground truth (ontology-normalized)
        class_names = self._get_class_names(normalized_ground_truth)

        if not class_names:
            # No ground truth classes to evaluate
            early_stats = build_operational_stats(inference_result)
            return EvaluationResult(
                test_id=test_case.test_id,
                example_id=inference_result.request.example_id,
                pipeline_name=inference_result.pipeline_name,
                product_type=inference_result.product_type.value,
                success=True,
                metrics=[],
                error="No ground truth annotations found",
                stats=early_stats,
            )

        metrics: list[MetricValue] = []

        # Compute mAP at multiple thresholds
        map_metrics = compute_map_at_thresholds(predictions, normalized_ground_truth, class_names, self._iou_thresholds)

        metrics.append(
            MetricValue(
                metric_name="mAP@[.50:.95]",
                value=map_metrics["mAP@[.50:.95]"],
                metadata={"evaluation_view": effective_view, "target_ontology": target_ontology},
            )
        )
        metrics.append(
            MetricValue(
                metric_name="AP50",
                value=map_metrics["AP50"],
                metadata={"evaluation_view": effective_view, "target_ontology": target_ontology},
            )
        )
        metrics.append(
            MetricValue(
                metric_name="AP75",
                value=map_metrics["AP75"],
                metadata={"evaluation_view": effective_view, "target_ontology": target_ontology},
            )
        )

        # Compute per-class metrics at IoU=0.5
        per_class_metrics = compute_per_class_metrics(
            predictions, normalized_ground_truth, class_names, iou_threshold=0.5
        )

        # Add per-class F1 scores
        for class_name, class_metrics in per_class_metrics.items():
            metrics.append(
                MetricValue(
                    metric_name=f"f1_{class_name}",
                    value=class_metrics["f1"],
                    metadata={
                        "class_name": class_name,
                        "precision": class_metrics["precision"],
                        "recall": class_metrics["recall"],
                        "ap": class_metrics["ap"],
                        "support": class_metrics["support"],
                    },
                )
            )
            metrics.append(
                MetricValue(
                    metric_name=f"precision_{class_name}",
                    value=class_metrics["precision"],
                    metadata={
                        "class_name": class_name,
                        "f1": class_metrics["f1"],
                        "recall": class_metrics["recall"],
                        "ap": class_metrics["ap"],
                        "support": class_metrics["support"],
                    },
                )
            )
            metrics.append(
                MetricValue(
                    metric_name=f"recall_{class_name}",
                    value=class_metrics["recall"],
                    metadata={
                        "class_name": class_name,
                        "f1": class_metrics["f1"],
                        "precision": class_metrics["precision"],
                        "ap": class_metrics["ap"],
                        "support": class_metrics["support"],
                    },
                )
            )

        # Compute mean F1 across classes
        f1_values = [m["f1"] for m in per_class_metrics.values() if m["support"] > 0]
        mean_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
        metrics.append(
            MetricValue(
                metric_name="mean_f1",
                value=mean_f1,
                metadata={"num_classes": len(f1_values)},
            )
        )

        # Add summary metrics
        metrics.append(
            MetricValue(
                metric_name="num_predictions",
                value=float(len(predictions)),
                metadata={},
            )
        )
        metrics.append(
            MetricValue(
                metric_name="num_ground_truth",
                value=float(len(ground_truth)),
                metadata={},
            )
        )

        # Pass/fail criteria and attribution metrics
        localization_passed = 0
        localization_total = 0
        classification_passed = 0
        classification_total = 0
        attribution_passed = 0
        attribution_total = 0
        unmatched_gt = 0
        unmatched_pred = 0
        rule_passed_count = 0
        rule_total_count = 0
        reading_order_passed = 0
        reading_order_total = 0
        rule_results: list[dict] = []

        has_content_any = any(
            rule.content is not None and not is_truthy(normalize_layout_attributes(rule.attributes).get("ignore"))
            for rule in test_case.get_layout_rules()
        )

        total_lap_num = 0.0
        total_lap_den = 0
        total_lar_num = 0.0
        total_lar_den = 0
        attribution_metrics_available = False

        for page_index in test_case.get_page_indices():
            page_number = page_index + 1
            raw_layout_rules = test_case.get_layout_rules(page=page_number)
            layout_rules: list[Any] = []
            layout_rule_attrs: list[dict[str, str]] = []
            for rule in raw_layout_rules:
                normalized_attrs = normalize_layout_attributes(rule.attributes)
                if is_truthy(normalized_attrs.get("ignore")):
                    continue
                layout_rules.append(rule)
                layout_rule_attrs.append(normalized_attrs)
            if not layout_rules:
                continue

            page_predictions = self._extract_predictions(
                inference_result,
                layout_output,
                target_ontology=target_ontology,
                page_filter=page_number,
            )
            page_prediction_order_indices = [
                raw_order_index if isinstance((raw_order_index := pred.get("order_index")), int) else idx
                for idx, pred in enumerate(page_predictions)
            ]
            page_prediction_classes = [
                str(pred.get("class_name")) if pred.get("class_name") is not None else None for pred in page_predictions
            ]

            gt_boxes = [coco_normalized_to_xyxy_normalized(rule.bbox) for rule in layout_rules]
            pred_boxes = [pred["bbox"] for pred in page_predictions]
            iou_matrix = compute_iou_matrix(
                np.array(gt_boxes, dtype=float) if gt_boxes else np.zeros((0, 4)),
                np.array(pred_boxes, dtype=float) if pred_boxes else np.zeros((0, 4)),
            )

            ioa_matrix = compute_ioa_matrix(
                np.array(gt_boxes, dtype=float) if gt_boxes else np.zeros((0, 4)),
                np.array(pred_boxes, dtype=float) if pred_boxes else np.zeros((0, 4)),
            )
            ioa_matrix_pred = compute_ioa_matrix(
                np.array(pred_boxes, dtype=float) if pred_boxes else np.zeros((0, 4)),
                np.array(gt_boxes, dtype=float) if gt_boxes else np.zeros((0, 4)),
            )

            if gt_boxes:
                if pred_boxes:
                    eligible = (ioa_matrix >= LOCALIZATION_IOA_THRESHOLD) & (
                        ioa_matrix_pred.T >= LOCALIZATION_IOA_PRED_THRESHOLD
                    )
                    for gt_idx, rule in enumerate(layout_rules):
                        if not _is_page_furniture(rule.canonical_class):
                            continue
                        eligible[gt_idx, :] = (ioa_matrix[gt_idx] > 0.0) & (
                            ioa_matrix_pred[:, gt_idx] >= LOCALIZATION_IOA_PRED_THRESHOLD
                        )
                    unmatched_gt += int(np.sum(~np.any(eligible, axis=1)))
                else:
                    unmatched_gt += len(gt_boxes)

            if pred_boxes:
                if gt_boxes:
                    eligible = (ioa_matrix >= LOCALIZATION_IOA_THRESHOLD) & (
                        ioa_matrix_pred.T >= LOCALIZATION_IOA_PRED_THRESHOLD
                    )
                    for gt_idx, rule in enumerate(layout_rules):
                        if not _is_page_furniture(rule.canonical_class):
                            continue
                        eligible[gt_idx, :] = (ioa_matrix[gt_idx] > 0.0) & (
                            ioa_matrix_pred[:, gt_idx] >= LOCALIZATION_IOA_PRED_THRESHOLD
                        )
                    unmatched_pred += int(np.sum(~np.any(eligible, axis=0)))
                else:
                    unmatched_pred += len(pred_boxes)

            gt_elements = None
            pred_blocks = None
            ioa_attr = None
            ioa_attr_pred = None
            iou_attr = None
            gt_has_content = [rule.content is not None for rule in layout_rules]

            if has_content_any:
                gt_elements = parse_gt_elements([rule.model_dump() for rule in layout_rules])

            if has_content_any:
                pred_blocks = adapter.to_attribution_blocks(
                    layout_output,
                    page_number=page_number,
                    test_case=test_case,
                )

                if gt_elements:
                    attr_result = compute_attribution_metrics(
                        gt_elements,
                        pred_blocks,
                        ioa_threshold=ATTRIBUTION_OVERLAP_IOA_THRESHOLD,
                    )
                    attribution_metrics_available = True
                    total_lap_num += attr_result.lap * attr_result.num_pred_tokens
                    total_lap_den += attr_result.num_pred_tokens
                    total_lar_num += attr_result.lar * attr_result.num_gt_tokens
                    total_lar_den += attr_result.num_gt_tokens

                if gt_elements and pred_blocks:
                    gt_boxes_attr = np.array([g.bbox_xyxy for g in gt_elements])
                    pred_boxes_attr = np.array([p.bbox_xyxy for p in pred_blocks])
                    ioa_attr = compute_ioa_matrix(gt_boxes_attr, pred_boxes_attr)
                    ioa_attr_pred = compute_ioa_matrix(pred_boxes_attr, gt_boxes_attr)
                    iou_attr = compute_iou_matrix(gt_boxes_attr, pred_boxes_attr)
                elif gt_elements is not None and pred_blocks is not None:
                    ioa_attr = np.zeros((len(gt_elements), len(pred_blocks)))
                    ioa_attr_pred = np.zeros((len(pred_blocks), len(gt_elements)))
                    iou_attr = np.zeros((len(gt_elements), len(pred_blocks)))

            for gt_idx, rule in enumerate(layout_rules):
                rule_attrs = layout_rule_attrs[gt_idx]
                explicit_mode = is_truthy(rule_attrs.get("explicit"))
                caption_skip = is_truthy(rule_attrs.get("caption"))
                localization_total += 1
                classification_total += 1
                gt_class_raw = rule.canonical_class
                is_page_furniture = _is_page_furniture(gt_class_raw)
                furniture_group = _PageFurnitureGroup([], [], None, None)

                best_ioa = 0.0
                best_pred_idx = None
                if pred_boxes:
                    best_pred_idx = int(np.argmax(ioa_matrix[gt_idx]))
                    best_ioa = float(ioa_matrix[gt_idx, best_pred_idx])

                best_iou = 0.0
                best_ioa_pred = 0.0
                if pred_boxes:
                    if is_page_furniture:
                        furniture_group = _build_page_furniture_group(
                            gt_box=gt_boxes[gt_idx],
                            gt_idx=gt_idx,
                            pred_boxes=pred_boxes,
                            ioa_pred_to_gt=ioa_matrix_pred,
                            iou_row=iou_matrix[gt_idx],
                            pred_order_indices=page_prediction_order_indices,
                            pred_classes=page_prediction_classes,
                        )
                        if furniture_group.representative_pred_idx is not None:
                            best_pred_idx = furniture_group.representative_pred_idx
                            best_ioa = float(ioa_matrix[gt_idx, best_pred_idx])
                            best_iou = float(iou_matrix[gt_idx, best_pred_idx])
                            best_ioa_pred = float(ioa_matrix_pred[best_pred_idx, gt_idx])
                    else:
                        eligible = np.where(  # type: ignore[assignment]
                            (ioa_matrix[gt_idx] >= LOCALIZATION_IOA_THRESHOLD)
                            & (ioa_matrix_pred[:, gt_idx] >= LOCALIZATION_IOA_PRED_THRESHOLD)
                        )[0]
                        if len(eligible) > 0:
                            best_pred_idx = int(eligible[np.argmax(iou_matrix[gt_idx, eligible])])
                            best_ioa = float(ioa_matrix[gt_idx, best_pred_idx])
                            best_iou = float(iou_matrix[gt_idx, best_pred_idx])
                        if best_pred_idx is not None:
                            best_ioa_pred = float(ioa_matrix_pred[best_pred_idx, gt_idx])

                matched_pred_order_index = None
                if is_page_furniture and furniture_group.earliest_order_index is not None:
                    matched_pred_order_index = furniture_group.earliest_order_index
                elif best_pred_idx is not None and best_pred_idx < len(page_predictions):
                    raw_order_index = page_predictions[best_pred_idx].get("order_index")
                    if isinstance(raw_order_index, int):
                        matched_pred_order_index = raw_order_index
                    else:
                        matched_pred_order_index = best_pred_idx

                localization_pass = (
                    (
                        bool(furniture_group.pred_indices)
                        and furniture_group.x_span_coverage >= _PAGE_FURNITURE_X_SPAN_COVERAGE_THRESHOLD
                        and furniture_group.y_coverage >= _PAGE_FURNITURE_Y_COVERAGE_THRESHOLD
                    )
                    if is_page_furniture
                    else (best_ioa >= LOCALIZATION_IOA_THRESHOLD and best_ioa_pred >= LOCALIZATION_IOA_PRED_THRESHOLD)
                )
                if localization_pass:
                    localization_passed += 1

                if is_page_furniture and not furniture_group.pred_indices:
                    localization_reason = "no_overlap"
                elif best_pred_idx is None or best_ioa == 0.0:
                    localization_reason = "no_overlap"
                elif not localization_pass:
                    localization_reason = "below_threshold"
                else:
                    localization_reason = "pass"

                gt_class_norm = map_label_to_target_ontology(
                    gt_class_raw,
                    target_ontology,
                )
                pred_class_raw = None
                pred_class_norm = None
                classification_pass = False
                if localization_pass and best_pred_idx is not None:
                    pred_class_raw = page_predictions[best_pred_idx]["class_name"]
                    pred_class_norm = pred_class_raw
                    classification_pass = pred_class_norm == gt_class_norm
                    if classification_pass:
                        classification_passed += 1

                if not localization_pass:
                    classification_reason = "no_localization"
                elif not classification_pass:
                    classification_reason = "class_mismatch"
                else:
                    classification_reason = "pass"

                # Attribution diagnostics per GT element
                attribution_applicable = False
                attribution_pass = None
                attribution_reason = "no_gt_content"
                attribution_method = "skip"
                attribution_threshold: float | None = None
                overlap_pred_count = 0
                token_precision = None
                token_recall = None
                token_f1 = None
                missing_tokens_sample: list[str] | None = None
                extra_tokens_sample: list[str] | None = None
                gt_text_norm: str | None = None
                pred_text_norm: str | None = None
                extra_tokens_ignored = False
                furniture_selected_span_indices: list[int] | None = [] if is_page_furniture else None

                if layout_element_is_formula(gt_class_raw, rule_attrs):
                    attribution_reason = "formula_skip"
                    missing_tokens_sample = []
                    extra_tokens_sample = []
                elif caption_skip:
                    attribution_reason = "caption_skip"
                    missing_tokens_sample = []
                    extra_tokens_sample = []
                elif not gt_has_content[gt_idx]:
                    attribution_reason = "no_gt_content"
                    missing_tokens_sample = []
                    extra_tokens_sample = []
                elif (
                    gt_elements is None
                    or pred_blocks is None
                    or ioa_attr is None
                    or (is_page_furniture and ioa_attr_pred is None)
                ):
                    attribution_reason = "no_pred_content"
                else:
                    if gt_elements[gt_idx].tokens:
                        if gt_elements[gt_idx].content_type == "text":
                            gt_text_norm = gt_elements[gt_idx].normalized_text
                        attribution_applicable = True
                        attribution_method = "recall" if explicit_mode else "f1"
                        attribution_threshold = ATTRIBUTION_TOKEN_F1_THRESHOLD
                        extra_tokens_ignored = explicit_mode
                        attribution_scoring: Literal["f1", "recall"] = "recall" if explicit_mode else "f1"
                        if is_page_furniture:
                            furniture_match = _select_page_furniture_attribution_match(
                                gt_idx=gt_idx,
                                gt_elements=gt_elements,
                                pred_blocks=pred_blocks,
                                ioa_attr=ioa_attr,
                                ioa_attr_pred=ioa_attr_pred,
                                iou_attr=iou_attr,
                                scoring=attribution_scoring,
                            )
                            overlapping = furniture_match.overlapping_indices
                            best_attr_pred_idx = furniture_match.representative_pred_idx
                            best_pred_tokens = furniture_match.selected_tokens
                            best_precision = furniture_match.precision
                            best_recall = furniture_match.recall
                            best_f1 = furniture_match.f1
                            pred_text_norm = furniture_match.selected_text_norm
                            furniture_selected_span_indices = furniture_match.selected_indices
                        else:
                            (
                                overlapping,
                                best_attr_pred_idx,
                                best_pred_tokens,
                                best_precision,
                                best_recall,
                                best_f1,
                            ) = _select_best_attribution_match(
                                gt_idx=gt_idx,
                                gt_elements=gt_elements,
                                pred_blocks=pred_blocks,
                                ioa_attr=ioa_attr,
                                iou_attr=iou_attr,
                                scoring=attribution_scoring,
                            )
                        overlap_pred_count = len(overlapping)

                        if gt_elements[gt_idx].content_type == "text" and not is_page_furniture:
                            if best_attr_pred_idx is not None:
                                pred_text_norm = pred_blocks[best_attr_pred_idx].normalized_text
                        if overlap_pred_count == 0:
                            attribution_pass = False
                            attribution_reason = "no_overlap_preds"
                            token_precision = 0.0
                            token_recall = 0.0
                            token_f1 = 0.0
                            missing_tokens_sample = _multiset_difference_sample(gt_elements[gt_idx].tokens, [], 5)
                            extra_tokens_sample = []
                        else:
                            if best_attr_pred_idx is None:
                                attribution_pass = False
                                attribution_reason = "no_overlap_preds"
                                token_precision = 0.0
                                token_recall = 0.0
                                token_f1 = 0.0
                                missing_tokens_sample = _multiset_difference_sample(gt_elements[gt_idx].tokens, [], 5)
                                extra_tokens_sample = []
                            else:
                                pred_tokens = best_pred_tokens
                                token_precision = best_precision
                                token_recall = best_recall
                                token_f1 = best_f1
                                missing_tokens_sample = _multiset_difference_sample(
                                    gt_elements[gt_idx].tokens, pred_tokens, 5
                                )
                                extra_tokens_sample = _multiset_difference_sample(
                                    pred_tokens, gt_elements[gt_idx].tokens, 5
                                )
                                if explicit_mode:
                                    if token_recall >= ATTRIBUTION_TOKEN_F1_THRESHOLD:
                                        attribution_pass = True
                                        attribution_reason = "pass"
                                    else:
                                        attribution_pass = False
                                        attribution_reason = "explicit_recall_below_threshold"
                                elif token_f1 >= ATTRIBUTION_TOKEN_F1_THRESHOLD:
                                    attribution_pass = True
                                    attribution_reason = "pass"
                                else:
                                    attribution_pass = False
                                    attribution_reason = "f1_below_threshold"
                    else:
                        attribution_reason = "no_gt_content"
                        missing_tokens_sample = []
                        extra_tokens_sample = []

                rule_passed = localization_pass and classification_reason == "pass"
                if attribution_applicable:
                    rule_passed = rule_passed and bool(attribution_pass)

                rule_total_count += 1
                if rule_passed:
                    rule_passed_count += 1

                rule_results.append(
                    {
                        "element_id": rule.id,
                        "element_index": gt_idx,
                        "page": page_number,
                        "gt_class": gt_class_raw,
                        "gt_class_norm": gt_class_norm,
                        "best_pred_index": best_pred_idx,
                        "best_pred_class": pred_class_raw,
                        "best_pred_class_norm": pred_class_norm,
                        "best_pred_ioa_gt": best_ioa,
                        "best_pred_iou": best_iou,
                        "best_pred_bbox": (
                            page_predictions[best_pred_idx]["bbox"] if best_pred_idx is not None else None
                        ),
                        "gt_ro_index": rule.ro_index,
                        "matched_pred_order_index": matched_pred_order_index,
                        "localization_pass": localization_pass,
                        "localization_reason": localization_reason,
                        "classification_pass": classification_pass,
                        "classification_reason": classification_reason,
                        "attribution_applicable": attribution_applicable,
                        "attribution_pass": attribution_pass,
                        "attribution_reason": attribution_reason,
                        "attribution_method": attribution_method,
                        "attribution_threshold": attribution_threshold,
                        "overlap_pred_count": overlap_pred_count,
                        "token_precision": token_precision,
                        "token_recall": token_recall,
                        "token_f1": token_f1,
                        "extra_tokens_ignored": extra_tokens_ignored,
                        "normalized_attributes": rule_attrs,
                        "gt_text_norm": gt_text_norm,
                        "pred_text_norm": pred_text_norm,
                        "missing_tokens": missing_tokens_sample,
                        "extra_tokens": extra_tokens_sample,
                        "furniture_group_size": len(furniture_group.pred_indices) if is_page_furniture else None,
                        "furniture_x_span_coverage": furniture_group.x_span_coverage if is_page_furniture else None,
                        "furniture_x_fill_coverage": furniture_group.x_fill_coverage if is_page_furniture else None,
                        "furniture_y_coverage": furniture_group.y_coverage if is_page_furniture else None,
                        "furniture_label_histogram": furniture_group.label_histogram if is_page_furniture else None,
                        "furniture_selected_span_size": (
                            len(furniture_selected_span_indices)
                            if furniture_selected_span_indices is not None
                            else None
                        ),
                        "furniture_selected_span_indices": furniture_selected_span_indices,
                        "reading_order_eligible": False,
                        "reading_order_pass": False,
                        "reading_order_reason": "pending",
                    }
                )

            # Attribution pass/fail totals
            if gt_elements is not None and pred_blocks is not None and ioa_attr is not None and gt_has_content:
                for gt_idx, gt in enumerate(gt_elements):
                    if gt_element_skips_attribution(gt):
                        continue
                    if not gt_has_content[gt_idx]:
                        continue
                    if not gt.tokens:
                        continue
                    attribution_total += 1
                    explicit_mode = gt_element_is_explicit(gt)
                    aggregate_attribution_scoring: Literal["f1", "recall"] = "recall" if explicit_mode else "f1"
                    if _is_page_furniture(gt.canonical_class):
                        furniture_match = _select_page_furniture_attribution_match(
                            gt_idx=gt_idx,
                            gt_elements=gt_elements,
                            pred_blocks=pred_blocks,
                            ioa_attr=ioa_attr,
                            ioa_attr_pred=ioa_attr_pred,
                            iou_attr=iou_attr,
                            scoring=aggregate_attribution_scoring,
                        )
                        overlapping = furniture_match.overlapping_indices
                        best_recall = furniture_match.recall
                        best_f1 = furniture_match.f1
                    else:
                        (
                            overlapping,
                            _best_attr_pred_idx,
                            _best_pred_tokens,
                            _best_precision,
                            best_recall,
                            best_f1,
                        ) = _select_best_attribution_match(
                            gt_idx=gt_idx,
                            gt_elements=gt_elements,
                            pred_blocks=pred_blocks,
                            ioa_attr=ioa_attr,
                            iou_attr=iou_attr,
                            scoring=aggregate_attribution_scoring,
                        )
                    if len(overlapping) == 0:
                        continue
                    passes = (
                        best_recall >= ATTRIBUTION_TOKEN_F1_THRESHOLD
                        if explicit_mode
                        else best_f1 >= ATTRIBUTION_TOKEN_F1_THRESHOLD
                    )
                    if passes:
                        attribution_passed += 1

        if rule_results:
            reading_order_passed, reading_order_total = _score_local_reading_order(
                rule_results,
                max_neighbor_distance=3,
            )

        if localization_total > 0:
            metrics.append(
                MetricValue(
                    metric_name="layout_localization_pass_rate",
                    value=localization_passed / localization_total,
                    metadata={"passed": localization_passed, "total": localization_total},
                )
            )
        if classification_total > 0:
            metrics.append(
                MetricValue(
                    metric_name="layout_classification_pass_rate",
                    value=classification_passed / classification_total,
                    metadata={"passed": classification_passed, "total": classification_total},
                )
            )
        if attribution_total > 0:
            metrics.append(
                MetricValue(
                    metric_name="layout_attribution_pass_rate",
                    value=attribution_passed / attribution_total,
                    metadata={"passed": attribution_passed, "total": attribution_total},
                )
            )
        if reading_order_total > 0:
            metrics.append(
                MetricValue(
                    metric_name="layout_reading_order_pass_rate",
                    value=reading_order_passed / reading_order_total,
                    metadata={
                        "passed": reading_order_passed,
                        "total": reading_order_total,
                        "max_neighbor_distance": 3,
                    },
                )
            )
        total_rule_count = localization_total + classification_total + attribution_total
        total_rule_passed = localization_passed + classification_passed + attribution_passed
        if total_rule_count > 0:
            metrics.append(
                MetricValue(
                    metric_name="layout_rule_pass_rate",
                    value=total_rule_passed / total_rule_count,
                    metadata={
                        "passed": total_rule_passed,
                        "total": total_rule_count,
                        "localization_passed": localization_passed,
                        "localization_total": localization_total,
                        "classification_passed": classification_passed,
                        "classification_total": classification_total,
                        "attribution_passed": attribution_passed,
                        "attribution_total": attribution_total,
                    },
                )
            )
            metrics.append(
                MetricValue(
                    metric_name="rule_pass_rate",
                    value=total_rule_passed / total_rule_count,
                    metadata={
                        "passed": total_rule_passed,
                        "total": total_rule_count,
                        "localization_passed": localization_passed,
                        "localization_total": localization_total,
                        "classification_passed": classification_passed,
                        "classification_total": classification_total,
                        "attribution_passed": attribution_passed,
                        "attribution_total": attribution_total,
                    },
                )
            )
        if rule_total_count > 0:
            metrics.append(
                MetricValue(
                    metric_name="layout_element_rule_pass_rate",
                    value=rule_passed_count / rule_total_count,
                    metadata={
                        "passed": rule_passed_count,
                        "total": rule_total_count,
                        "rule_results": rule_results,
                    },
                )
            )

        if attribution_metrics_available and total_lar_den > 0:
            lap = total_lap_num / total_lap_den if total_lap_den > 0 else 1.0
            lar = total_lar_num / total_lar_den if total_lar_den > 0 else 1.0
            af1 = 2.0 * lap * lar / (lap + lar) if (lap + lar) > 0 else 0.0
            metrics.append(
                MetricValue(
                    metric_name="lap",
                    value=lap,
                    metadata={},
                )
            )
            metrics.append(
                MetricValue(
                    metric_name="lar",
                    value=lar,
                    metadata={},
                )
            )
            metrics.append(
                MetricValue(
                    metric_name="af1",
                    value=af1,
                    metadata={},
                )
            )

        metrics.append(
            MetricValue(
                metric_name="unmatched_gt_elements",
                value=float(unmatched_gt),
                metadata={"count": unmatched_gt},
            )
        )
        metrics.append(
            MetricValue(
                metric_name="unmatched_pred_elements",
                value=float(unmatched_pred),
                metadata={"count": unmatched_pred},
            )
        )

        stats = build_operational_stats(inference_result)

        return EvaluationResult(
            test_id=test_case.test_id,
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            product_type=inference_result.product_type.value,
            success=True,
            metrics=metrics,
            error=None,
            stats=stats,
        )

    def compute_confusion_matrix(
        self,
        inference_results: dict[str, InferenceResult],
        test_cases: dict[str, TestCase],
        iou_threshold: float = 0.5,
    ) -> ConfusionMatrixMetrics:
        """
        Compute aggregate confusion matrix across all test cases.

        Uses class-agnostic IoU matching to capture misclassifications.
        Tracks which test case IDs contribute to each confusion cell.

        :param inference_results: Dict mapping example_id → InferenceResult
        :param test_cases: Dict mapping test_id → TestCase
        :param iou_threshold: IoU threshold for matching (default 0.5)
        :return: ConfusionMatrixMetrics with full metadata
        """
        from collections import defaultdict

        import numpy as np

        from extract_bench.evaluation.metrics.layoutdet.iou import compute_iou_matrix
        from extract_bench.schemas.metrics import (
            ConfusionMatrixCell,
            ConfusionMatrixMetrics,
        )

        # Accumulate confusion data
        # Structure: (gt_class, pred_class) → list[test_id]
        confusion_cells_data: dict[tuple[str, str], list[str]] = defaultdict(list)
        false_negatives_data: dict[str, list[str]] = defaultdict(list)
        false_positives_data: dict[str, list[str]] = defaultdict(list)
        gt_totals: dict[str, int] = defaultdict(int)
        pred_totals: dict[str, int] = defaultdict(int)
        all_classes_set: set[str] = set()
        confusion_evaluation_view: Literal["core", "canonical"] = self._evaluation_view

        # Iterate over all test cases
        for test_id, test_case in test_cases.items():
            if not isinstance(test_case, LayoutDetectionTestCase):
                continue
            if not test_case.get_layout_annotations():
                continue

            # Find matching inference result
            # Note: For multi-page PDFs, multiple test_ids map to same example_id
            # Example: test_id="pdfs/doc/page_5", example_id="pdfs/doc"
            inference_result = None
            for example_id, result in inference_results.items():
                # Match if test_id starts with example_id or they're equal
                if test_id == example_id or test_id.startswith(example_id + "/"):
                    inference_result = result
                    break

            if not inference_result:
                continue

            # Extract predictions and GT
            try:
                adapter = create_layout_adapter_for_result(inference_result)
                page_indices = test_case.get_page_indices()
                page_filter = page_indices[0] + 1 if len(page_indices) == 1 else None
                layout_output = adapter.to_layout_output(
                    inference_result,
                    page_filter=page_filter,
                )
                target_ontology = self._resolve_target_ontology(test_case)
                effective_view = self._resolve_effective_evaluation_view(target_ontology)
                if effective_view == "canonical":
                    confusion_evaluation_view = "canonical"
                predictions = self._extract_predictions(
                    inference_result,
                    layout_output,
                    target_ontology=target_ontology,
                    page_filter=page_filter,
                )
                ground_truth = self._extract_ground_truth(test_case, target_ontology=target_ontology)

                ground_truth = [
                    {
                        **gt,
                        "class_name": map_label_to_target_ontology(
                            gt.get("class_name"),
                            target_ontology,
                        ),
                    }
                    for gt in ground_truth
                ]
            except Exception:
                continue

            if not ground_truth:
                continue

            # Convert to arrays for confusion matrix computation
            gt_bboxes_list = [g["bbox"] for g in ground_truth]
            gt_classes_list = [g["class_name"] for g in ground_truth]

            for gt_class in gt_classes_list:
                all_classes_set.add(gt_class)
                gt_totals[gt_class] += 1

            if not predictions:
                # All GT are false negatives
                for gt_class in gt_classes_list:
                    false_negatives_data[gt_class].append(test_id)
                continue

            pred_bboxes_list = [p["bbox"] for p in predictions]
            pred_classes_list = [p["class_name"] for p in predictions]
            pred_scores_list = [p["score"] for p in predictions]

            for pred_class in pred_classes_list:
                all_classes_set.add(pred_class)
                pred_totals[pred_class] += 1

            # Convert to numpy arrays
            pred_bboxes = np.array(pred_bboxes_list, dtype=float)
            pred_scores = np.array(pred_scores_list, dtype=float)
            gt_bboxes = np.array(gt_bboxes_list, dtype=float)

            # Compute IoU matrix
            iou_matrix = compute_iou_matrix(pred_bboxes, gt_bboxes)

            # Class-agnostic greedy matching
            sorted_indices = np.argsort(-pred_scores)
            matched_gt: set[int] = set()
            matched_pred: set[int] = set()

            for pred_idx in sorted_indices:
                pred_class = pred_classes_list[pred_idx]

                # Find best GT by IoU (any class)
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx in range(len(gt_bboxes)):
                    if gt_idx in matched_gt:
                        continue

                    iou = iou_matrix[pred_idx, gt_idx]
                    if iou >= iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0:
                    # Match found - record confusion
                    gt_class = gt_classes_list[best_gt_idx]
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx)

                    confusion_cells_data[(gt_class, pred_class)].append(test_id)

            # Unmatched GT → false negatives
            for gt_idx in range(len(gt_bboxes)):
                if gt_idx not in matched_gt:
                    gt_class = gt_classes_list[gt_idx]
                    false_negatives_data[gt_class].append(test_id)

            # Unmatched predictions → false positives
            for pred_idx in range(len(pred_bboxes)):
                if pred_idx not in matched_pred:
                    pred_class = pred_classes_list[pred_idx]
                    false_positives_data[pred_class].append(test_id)

        # Build ConfusionMatrixCell objects
        all_classes = sorted(all_classes_set)
        cells = []

        for gt_class in all_classes:
            gt_total = gt_totals[gt_class]
            for pred_class in all_classes:
                example_ids = confusion_cells_data.get((gt_class, pred_class), [])
                count = len(example_ids)
                percentage = (count / gt_total * 100) if gt_total > 0 else 0.0

                # Only include cells with non-zero counts (or diagonal)
                if count > 0 or gt_class == pred_class:
                    cells.append(
                        ConfusionMatrixCell(
                            gt_class=gt_class,
                            pred_class=pred_class,
                            count=count,
                            percentage=percentage,
                            example_ids=example_ids,
                        )
                    )

        return ConfusionMatrixMetrics(
            iou_threshold=iou_threshold,
            evaluation_view=confusion_evaluation_view,
            cells=cells,
            false_negatives=dict(false_negatives_data),
            false_positives=dict(false_positives_data),
            gt_totals=dict(gt_totals),
            pred_totals=dict(pred_totals),
            all_classes=all_classes,
        )
