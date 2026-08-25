"""Shared metric display names and tooltip explanations.

Single source of truth for metric metadata used across all report generators
(aggregation dashboard, detailed report, comparison report).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class MetricInfo:
    """Display metadata for a single evaluation metric."""

    display_name: str
    tooltip: str


# ---------------------------------------------------------------------------
# Master metric definitions — superset of all report display-name dicts.
# Tooltip text is derived from the actual evaluation / metric code.
# ---------------------------------------------------------------------------
METRIC_DEFINITIONS: dict[str, MetricInfo] = {
    # ── Parse: TEDS ──
    "teds": MetricInfo(
        "TEDS (All)",
        "Tree Edit Distance Similarity. Compares table HTML as trees using APTED algorithm: "
        "1 \u2212 (edit_distance / max_nodes). Evaluates both structure and cell text content. "
        "Score is averaged across matched table pairs.",
    ),
    "teds_predicted": MetricInfo(
        "TEDS (Predicted)",
        "Same as TEDS, but computed only among examples where tables were actually predicted "
        "(excludes examples with zero predicted tables).",
    ),
    "teds_struct": MetricInfo(
        "TEDS-Struct (All)",
        "TEDS structure-only variant. Compares table HTML tree structure while ignoring cell text content entirely.",
    ),
    "teds_struct_predicted": MetricInfo(
        "TEDS-Struct (Predicted)",
        "Same as TEDS-Struct, but only among examples with predicted tables.",
    ),
    "teds_struct_bool": MetricInfo(
        "TEDS-Struct+Bool (All)",
        "TEDS structure variant with boolean content awareness. Penalizes when one cell is "
        "empty and the other is not, but ignores actual text differences.",
    ),
    "teds_struct_bool_predicted": MetricInfo(
        "TEDS-Struct+Bool (Predicted)",
        "Same as TEDS-Struct+Bool, but only among examples with predicted tables.",
    ),
    # ── Parse: GriTS ──
    "grits_top": MetricInfo(
        "GriTS Top (All)",
        "Grid Table Similarity for topology. F-score from 2D Most-Similar Substructures "
        "algorithm, using IoU on cell spans for structural comparison.",
    ),
    "grits_con": MetricInfo(
        "GriTS Con (All)",
        "Grid Table Similarity for content. F-score from 2D Most-Similar Substructures "
        "algorithm, using Longest Common Subsequence for cell text comparison.",
    ),
    "grits_trm_composite": MetricInfo(
        "GTRM",
        "Composite GriTS score combining topology, recognition, and matching components.",
    ),
    "grits_top_predicted": MetricInfo(
        "GriTS Top (Predicted)",
        "Same as GriTS Top, but only among examples with predicted tables.",
    ),
    "grits_con_predicted": MetricInfo(
        "GriTS Con (Predicted)",
        "Same as GriTS Con, but only among examples with predicted tables.",
    ),
    "grits_precision_top": MetricInfo(
        "GriTS Precision (Topology)",
        "Precision component of GriTS topology score.",
    ),
    "grits_recall_top": MetricInfo(
        "GriTS Recall (Topology)",
        "Recall component of GriTS topology score.",
    ),
    "grits_precision_con": MetricInfo(
        "GriTS Precision (Content)",
        "Precision component of GriTS content score.",
    ),
    "grits_recall_con": MetricInfo(
        "GriTS Recall (Content)",
        "Recall component of GriTS content score.",
    ),
    "grits_top_upper_bound": MetricInfo(
        "GriTS Top Upper Bound",
        "Upper bound for GriTS topology score based on table matching.",
    ),
    "grits_con_upper_bound": MetricInfo(
        "GriTS Con Upper Bound",
        "Upper bound for GriTS content score based on table matching.",
    ),
    # ── Parse: reference GriTS ──
    "ref_grits_top": MetricInfo(
        "Ref GriTS Top (All)",
        "Reference GriTS topology score computed against reference tables.",
    ),
    "ref_grits_con": MetricInfo(
        "Ref GriTS Con (All)",
        "Reference GriTS content score computed against reference tables.",
    ),
    "ref_grits_top_predicted": MetricInfo(
        "Ref GriTS Top (Predicted)",
        "Reference GriTS topology score, only among examples with predicted tables.",
    ),
    "ref_grits_con_predicted": MetricInfo(
        "Ref GriTS Con (Predicted)",
        "Reference GriTS content score, only among examples with predicted tables.",
    ),
    # ── Parse: header accuracy ──
    "header_composite": MetricInfo(
        "Header Composite",
        "Mean of 8 header submetrics: cell count, GriTS content, content bag, perfect match, "
        "structure, block order, block extent, and block relative position.",
    ),
    "header_composite_v3": MetricInfo(
        "Header Composite v3",
        "Version 3 of the header composite metric with updated submetric weights and components.",
    ),
    "header_cell_count": MetricInfo(
        "Header Cell Count",
        "Ratio of predicted header cells to expected. Penalizes both missing and extra header cells symmetrically.",
    ),
    "header_grits": MetricInfo(
        "Header GriTS",
        "GriTS content score applied to contiguous header blocks only.",
    ),
    "header_content_bag": MetricInfo(
        "Header Content Bag",
        "Bag-of-cells exact content overlap: measures how many header cell texts match regardless of position.",
    ),
    "header_perfect": MetricInfo(
        "Header Perfect",
        "Binary metric: 1.0 if the header structure matches the ground truth exactly, 0.0 otherwise.",
    ),
    "header_structure": MetricInfo(
        "Header Structure",
        "GriTS topology score applied to the header region, measuring grid structure accuracy.",
    ),
    "header_block_order": MetricInfo(
        "Header Block Order",
        "Relative position preservation when multiple header blocks exist in a table.",
    ),
    "header_block_extent": MetricInfo(
        "Header Block Extent",
        "Location and size accuracy of each header block within the full table.",
    ),
    "header_block_proximity": MetricInfo(
        "Header Block Proximity",
        "Nearest-edge distance between matched header blocks, measuring spatial closeness.",
    ),
    "header_block_relative_direction": MetricInfo(
        "Header Block Relative Direction",
        "Cosine similarity of relative direction vectors between matched header blocks.",
    ),
    "header_block_relative_position": MetricInfo(
        "Header Block Relative Position",
        "Product of proximity (nearest-edge distance) and direction (cosine similarity) between matched header blocks.",
    ),
    # ── Parse: structural consistency ──
    "structural_consistency": MetricInfo(
        "Structural Consistency",
        "Self-consistency check on predicted tables (no ground truth comparison). "
        "Binary: 1.0 if every row has the same column count and every column has the same "
        "row count after resolving colspan/rowspan.",
    ),
    # ── Parse: table composite ──
    "table_composite": MetricInfo(
        "Table Composite",
        "Weighted combination of table metrics: "
        "0.8 \u00d7 (header_composite \u00d7 grits_con) + 0.2 \u00d7 structural_consistency. "
        "Balances content accuracy with structural integrity.",
    ),
    "table_composite_v3": MetricInfo(
        "Table Composite v3",
        "Version 3 of the table composite metric with updated component weights.",
    ),
    "table_composite_v3_harmonic": MetricInfo(
        "Table Composite v3 Harmonic",
        "Harmonic mean variant of table composite v3, penalizing low outlier scores more heavily.",
    ),
    # ── Parse: experimental composites ──
    "exp_header_composite_v3_generous": MetricInfo(
        "Exp Header Composite v3 Generous",
        "Experimental header composite v3 with generous matching criteria.",
    ),
    "exp_table_composite_v3_generous": MetricInfo(
        "Exp Table Composite v3 Generous",
        "Experimental table composite v3 with generous matching criteria.",
    ),
    "exp_table_composite_v3_generous_harmonic": MetricInfo(
        "Exp Table Composite v3 Generous (Harmonic)",
        "Harmonic mean variant of the experimental generous table composite v3.",
    ),
    # ── Parse: normalized text metrics ──
    "normalized_text_styling": MetricInfo(
        "Normalized Text Styling",
        "Normalized score for text styling accuracy (bold, italic, underline, etc.).",
    ),
    "normalized_text_correctness": MetricInfo(
        "Normalized Text Correctness",
        "Normalized score for text content correctness.",
    ),
    "normalized_order": MetricInfo(
        "Normalized Order",
        "Normalized score for reading order accuracy of text elements.",
    ),
    "normalized_title_accuracy": MetricInfo(
        "Normalized Title Accuracy",
        "Normalized score for title detection and content accuracy.",
    ),
    "normalized_code_block": MetricInfo(
        "Normalized Code Block",
        "Normalized score for code block detection and content accuracy.",
    ),
    "normalized_latex": MetricInfo(
        "Normalized LaTeX",
        "Normalized score for LaTeX equation rendering accuracy.",
    ),
    "normalized_text_score": MetricInfo(
        "Normalized Text Score",
        "Overall normalized text score combining multiple text quality dimensions.",
    ),
    # ── Parse: semantic metrics ──
    "content_faithfulness": MetricInfo(
        "Content Faithfulness",
        "Measures how faithfully the predicted content represents the source document.",
    ),
    "semantic_formatting": MetricInfo(
        "Semantic Formatting",
        "Measures accuracy of semantic formatting elements (headings, lists, emphasis, etc.).",
    ),
    # ── Parse: text similarity ──
    "text_similarity": MetricInfo(
        "Text Similarity",
        "Normalized Levenshtein distance between expected and predicted text, "
        "scaled to 0\u20131 where 1.0 is a perfect match.",
    ),
    # ── Parse: rule-based ──
    "rule_pass_rate": MetricInfo(
        "Rule Pass Rate",
        "Fraction of test rules that pass for each example: passed / total across all rule types.",
    ),
    # ── Parse: rule subtypes ──
    "chart_data_point": MetricInfo(
        "Chart Data Point",
        "Pass rate for chart data point extraction rules.",
    ),
    "form_field": MetricInfo(
        "Form Field",
        "Pass rate for form field rules (key-value, checkbox, signature). "
        "A rule passes when the labeled field is located in the parsed output "
        "and the extracted value matches the expected value (or any acceptable "
        "alternative when value is a list; signature rules pass on presence).",
    ),
    "order": MetricInfo(
        "Order",
        "Pass rate for reading order rules, checking that elements appear in the expected sequence.",
    ),
    "is_bold": MetricInfo(
        "Is Bold",
        "Pass rate for bold formatting detection rules.",
    ),
    "is_footer": MetricInfo(
        "Is Footer",
        "Pass rate for footer section detection rules.",
    ),
    "is_header": MetricInfo(
        "Is Header",
        "Pass rate for header section detection rules.",
    ),
    "is_sup": MetricInfo(
        "Is Sup",
        "Pass rate for superscript formatting detection rules.",
    ),
    "is_underline": MetricInfo(
        "Is Underline",
        "Pass rate for underline formatting detection rules.",
    ),
    "missing_sentence": MetricInfo(
        "Missing Sentence",
        "Pass rate for missing sentence rules. Checks that expected sentences appear in the output.",
    ),
    "missing_specific_sentence": MetricInfo(
        "Missing Specific Sentence",
        "Pass rate for specific required sentence presence rules.",
    ),
    "missing_specific_word": MetricInfo(
        "Missing Specific Word",
        "Pass rate for specific required word presence rules.",
    ),
    "missing_word": MetricInfo(
        "Missing Word",
        "Pass rate for missing word rules. Checks that expected words appear in the output.",
    ),
    "too_many_sentence_occurence": MetricInfo(
        "Too Many Sentence Occurence",
        "Pass rate for sentence frequency rules. Penalizes when sentences appear more times than expected.",
    ),
    "too_many_word_occurence": MetricInfo(
        "Too Many Word Occurence",
        "Pass rate for word frequency rules. Penalizes when words appear more times than expected.",
    ),
    "unexpected_sentence": MetricInfo(
        "Unexpected Sentence",
        "Pass rate for unexpected sentence rules. Penalizes extra sentences not in the ground truth.",
    ),
    "unexpected_word": MetricInfo(
        "Unexpected Word",
        "Pass rate for unexpected word rules. Penalizes extra words not in the ground truth.",
    ),
    "table_adjacent_down": MetricInfo(
        "Table Adjacent Down",
        "Pass rate for table adjacency rules checking cells directly below.",
    ),
    "table_adjacent_right": MetricInfo(
        "Table Adjacent Right",
        "Pass rate for table adjacency rules checking cells directly to the right.",
    ),
    "table_colspan": MetricInfo(
        "Table Colspan",
        "Pass rate for column span detection rules in tables.",
    ),
    "table_rowspan": MetricInfo(
        "Table Rowspan",
        "Pass rate for row span detection rules in tables.",
    ),
    "table_no_above": MetricInfo(
        "Table No Above",
        "Pass rate for rules verifying no content exists above a given table cell.",
    ),
    "table_no_below": MetricInfo(
        "Table No Below",
        "Pass rate for rules verifying no content exists below a given table cell.",
    ),
    "table_no_left": MetricInfo(
        "Table No Left",
        "Pass rate for rules verifying no content exists to the left of a given table cell.",
    ),
    "table_no_right": MetricInfo(
        "Table No Right",
        "Pass rate for rules verifying no content exists to the right of a given table cell.",
    ),
    "table_same_column": MetricInfo(
        "Table Same Column",
        "Pass rate for rules verifying that specific cells share the same column.",
    ),
    "table_same_row": MetricInfo(
        "Table Same Row",
        "Pass rate for rules verifying that specific cells share the same row.",
    ),
    "table_top_header": MetricInfo(
        "Table Top Header",
        "Pass rate for rules checking top header row identification in tables.",
    ),
    # ── Extract: the three headline metrics ──
    # Definitions follow the ExtractBench paper (Metric Details appendix). The
    # three nest by construction on any document where all three are defined:
    # value F1 >= page F1 >= word-level grounding F1, because a cell is only
    # eligible for grounding credit once its value is already accepted.
    "extract_unified_value_f1": MetricInfo(
        "Value F1",
        "The headline metric. Output is flattened into cells (one per scalar field and per "
        "aligned record subfield) and a cell is correct when it matches its expected "
        "counterpart after normalization. Records pair by a globally optimal one-to-one "
        "assignment, so a correct record counts in any output position. Every scalar field "
        "enters both denominators and an omitted key scores as an explicit null, so a "
        "faithful null is credited and a hallucinated value costs both precision and recall. "
        "Per-document micro F1; slice scores are unweighted means over documents.",
    ),
    "extract_unified_grounded_f1": MetricInfo(
        "Bounding Box F1",
        "Word-level grounding. A cell counts only when its value is accepted AND its predicted "
        "citation box overlaps an accepted evidence box for that field at IoU >= 0.5 on the "
        "correct page. Precision counts only gradeable claims (citations on cells aligned to "
        "box-bearing ground truth); recall counts ground-truth cells carrying a verified box. "
        "Documents whose ground truth has no verified boxes emit no score at all rather than a "
        "zero, so this averages over box-bearing documents only.",
    ),
    "extract_unified_page_f1": MetricInfo(
        "Page F1",
        "Page-level grounding: the same rule as Bounding Box F1 with page membership replacing "
        "the box overlap test. A cell counts when its value is accepted and the page it cites "
        "is correct. Like the box metric, documents whose ground truth carries no page evidence "
        "emit no score rather than a zero.",
    ),
    "extract_unified_value_precision": MetricInfo(
        "Value Precision",
        "Precision half of Value F1: correct cells / predicted cells. Surplus or duplicated "
        "records lower it; a truncated list does not.",
    ),
    "extract_unified_value_recall": MetricInfo(
        "Value Recall",
        "Recall half of Value F1: correct cells / expected cells. Missing records lower it, "
        "which is why a precision-recall gap is the long-list truncation signature.",
    ),
    "extract_unified_grounded_precision": MetricInfo(
        "Bounding Box Precision",
        "Precision half of Bounding Box F1, over gradeable citation claims only.",
    ),
    "extract_unified_grounded_recall": MetricInfo(
        "Bounding Box Recall",
        "Recall half of Bounding Box F1, over ground-truth cells carrying a verified box.",
    ),
    "extract_unified_page_precision": MetricInfo(
        "Page Precision",
        "Precision half of Page F1.",
    ),
    "extract_unified_page_recall": MetricInfo(
        "Page Recall",
        "Recall half of Page F1.",
    ),
    # ── Extract: everything else ──
    "accuracy": MetricInfo(
        "Accuracy",
        "JSON subset match comparing expected vs actual extracted data. Supports date "
        "normalization and weighted scoring by leaf node count.",
    ),
    "extract_value_precision": MetricInfo(
        "Extract Value Precision",
        "Native extract value precision using schema-aware typed comparison and "
        "index-tolerant array matching: matched predicted values / predicted values.",
    ),
    "extract_value_recall": MetricInfo(
        "Extract Value Recall",
        "Native extract value recall using schema-aware typed comparison and "
        "index-tolerant array matching: matched expected values / expected values.",
    ),
    "extract_value_f1": MetricInfo(
        "Extract Value F1",
        "Harmonic mean of native extract value precision and recall.",
    ),
    "extract_value_pass_rate": MetricInfo(
        "Extract Value Pass Rate",
        "Per-rule pass rate for native extract values using schema-aware typed comparison "
        "with index-tolerant array matching.",
    ),
    "extract_bbox_iou": MetricInfo(
        "Extract BBox IoU",
        "Per-document metric: mean of per-field-rule intersection-over-union between ground-truth "
        "extract-field bboxes and selected native extract field-citation bboxes.",
    ),
    "extract_bbox_recall": MetricInfo(
        "Extract BBox Recall",
        "Per-document metric: mean of per-field-rule ground-truth bbox area covered by selected "
        "native extract field-citation bboxes.",
    ),
    "extract_element_pass_rate": MetricInfo(
        "Extract Element Pass Rate",
        "Native extract per-field pass rate where localization and typed attribution both pass.",
    ),
    "extract_localization_pass_rate": MetricInfo(
        "Extract Localization Pass Rate",
        "Native extract per-field pass rate where predicted field-citation bboxes meet strict "
        "or relaxed localization criteria.",
    ),
    "extract_attribution_pass_rate": MetricInfo(
        "Extract Attribution Pass Rate",
        "Native extract per-field pass rate where the localized structured prediction matches "
        "the expected value under typed comparison.",
    ),
    "extract_avg_iou": MetricInfo(
        "Extract Avg IoU",
        "Average per-rule IoU for extract field localization candidates.",
    ),
    "extract_avg_iou_matched": MetricInfo(
        "Extract Avg IoU Matched",
        "Average per-rule IoU across native extract fields that passed localization.",
    ),
    "extract_avg_iou_unmatched": MetricInfo(
        "Extract Avg IoU Unmatched",
        "Average per-rule IoU across native extract fields that failed localization.",
    ),
    "parse_field_element_pass_rate": MetricInfo(
        "Parse Field Element Pass Rate",
        "Parse-side field grounding pass rate where localization, trivial classification, "
        "and typed attribution all pass against extract_field rules.",
    ),
    "parse_field_rule_pass_rate": MetricInfo(
        "Parse Field Rule Pass Rate",
        "Parse-side average pass rate across localization, classification, and attribution "
        "checks for extract_field rules.",
    ),
    "parse_field_localization_pass_rate": MetricInfo(
        "Parse Field Localization Pass Rate",
        "Parse-side pass rate where granular parse support bboxes meet strict or relaxed "
        "localization criteria for extract_field rules.",
    ),
    "parse_field_classification_pass_rate": MetricInfo(
        "Parse Field Classification Pass Rate",
        "Parse-side classification pass rate for extract_field rules. This is currently "
        "trivial because field rules do not carry class labels.",
    ),
    "parse_field_attribution_pass_rate": MetricInfo(
        "Parse Field Attribution Pass Rate",
        "Parse-side pass rate where localized support text matches the expected field value under typed comparison.",
    ),
    "parse_field_avg_iou": MetricInfo(
        "Parse Field Avg IoU",
        "Average per-rule IoU for parse-side field grounding candidates.",
    ),
    "parse_field_avg_iou_matched": MetricInfo(
        "Parse Field Avg IoU Matched",
        "Average per-rule IoU across parse-side field rules that passed localization.",
    ),
    "parse_field_avg_iou_unmatched": MetricInfo(
        "Parse Field Avg IoU Unmatched",
        "Average per-rule IoU across parse-side field rules that failed localization.",
    ),
    "parse_field_iou": MetricInfo(
        "Parse Field IoU",
        "Per-document metric: mean of per-field-rule intersection-over-union between ground-truth "
        "extract-field bboxes and selected parse support bboxes.",
    ),
    "parse_field_bbox_recall": MetricInfo(
        "Parse Field BBox Recall",
        "Per-document metric: mean of per-field-rule ground-truth bbox area covered by selected parse support bboxes.",
    ),
    "parse_field_text_similarity": MetricInfo(
        "Parse Field Text Similarity",
        "Average typed text similarity for string extract_field rules matched to parse support text.",
    ),
    "parse_field_gt_count": MetricInfo(
        "Parse Field GT Count",
        "Number of non-stray extract_field rules evaluated against parse output support.",
    ),
    # ── Layout detection: attribution ──
    "af1": MetricInfo(
        "Attribution F1",
        "Harmonic mean of LAP and LAR. Measures overall content attribution accuracy in spatial regions.",
    ),
    "lap": MetricInfo(
        "Local Attribution Precision",
        "For each predicted block, checks whether its text tokens are found in "
        "ground truth elements that spatially overlap with it.",
    ),
    "lar": MetricInfo(
        "Local Attribution Recall",
        "For each ground truth element, checks whether its text tokens are "
        "recovered by predicted blocks that spatially overlap with it.",
    ),
    # ── Layout detection: COCO metrics ──
    "mAP@[.50:.95]": MetricInfo(
        "mAP@[.50:.95]",
        "Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95 "
        "(step 0.05). Standard COCO object detection metric.",
    ),
    "AP50": MetricInfo(
        "AP@50",
        "Average Precision at IoU threshold 0.50. Measures detection accuracy with a lenient overlap requirement.",
    ),
    "AP75": MetricInfo(
        "AP@75",
        "Average Precision at IoU threshold 0.75. Measures detection accuracy with a strict overlap requirement.",
    ),
    "mean_f1": MetricInfo(
        "Mean F1",
        "Average F1 score across all layout element classes.",
    ),
    # ── Layout detection: rule pass rates ──
    "layout_element_rule_pass_rate": MetricInfo(
        "Layout Element Rule Pass Rate",
        "Overall per-element rule pass rate combining localization, classification, "
        "attribution, and reading order checks.",
    ),
    "layout_localization_rule_pass_rate": MetricInfo(
        "Layout Localization Rule Pass Rate",
        "Pass rate for bounding box localization rules. Checks spatial accuracy of predicted element positions.",
    ),
    "layout_classification_rule_pass_rate": MetricInfo(
        "Layout Classification Rule Pass Rate",
        "Pass rate for class label prediction rules. Checks whether predicted element types match ground truth.",
    ),
    "layout_attribution_rule_pass_rate": MetricInfo(
        "Layout Attribution Rule Pass Rate",
        "Pass rate for content attribution rules. Checks whether predicted blocks contain the correct text content.",
    ),
    "layout_reading_order_pass_rate": MetricInfo(
        "Layout Reading Order Pass Rate",
        "Pass rate for reading order rules. Checks whether layout elements are ordered correctly.",
    ),
    # ── QA ──
    "qa_answer_match": MetricInfo(
        "QA Match",
        "Binary pass/fail for each question. Supports single-choice, multiple-choice, "
        "numerical (with tolerance), and free-text answer types.",
    ),
    "qa_anls_star": MetricInfo(
        "QA ANLS*",
        "Average Normalized Levenshtein Similarity for free-text answers. "
        "Ranges from 0 (completely different) to 1 (perfect match).",
    ),
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Headline metric ordering.
# ---------------------------------------------------------------------------
# The three metrics the ExtractBench paper reports, in the order they should be
# shown. Everything else is secondary and sorts after a separator. Without this,
# metric menus sorted alphabetically and defaulted to whatever came first --
# "accuracy", a legacy JSON-subset score the paper does not report.
EXTRACT_HEADLINE_METRICS: tuple[str, ...] = (
    "extract_unified_value_f1",
    "extract_unified_grounded_f1",
    "extract_unified_page_f1",
)

# Sentinel inserted between the headline block and the rest so report UIs can
# draw a divider. It is not a metric and must never be selectable.
METRIC_GROUP_SEPARATOR = "__separator__"

# The metric a report opens on for an extract split.
EXTRACT_DEFAULT_METRIC = "extract_unified_value_f1"


def order_metrics(metric_names: list[str] | set[str], *, separator: bool = True) -> list[str]:
    """Headline metrics first (in paper order), then the rest alphabetically.

    ``separator`` inserts :data:`METRIC_GROUP_SEPARATOR` between the two blocks
    when both are non-empty, so a menu can render a divider. Callers that cannot
    represent a divider pass ``separator=False`` and get a plain ordering.
    """
    present = set(metric_names)
    head = [m for m in EXTRACT_HEADLINE_METRICS if m in present]
    tail = sorted(present - set(head))
    if separator and head and tail:
        return [*head, METRIC_GROUP_SEPARATOR, *tail]
    return [*head, *tail]


def display_name(metric_key: str) -> str:
    """Return human-friendly display name for a metric.

    Falls back to title-cased key with underscores replaced by spaces.
    """
    info = METRIC_DEFINITIONS.get(metric_key)
    if info is not None:
        return info.display_name
    return metric_key.replace("_", " ").title()


def tooltip(metric_key: str) -> str:
    """Return tooltip explanation for a metric.

    Returns empty string when no tooltip is available (dynamic fallbacks
    are handled by the JS ``tooltipIcon()`` function in each report).
    """
    info = METRIC_DEFINITIONS.get(metric_key)
    if info is not None:
        return info.tooltip
    return ""


def display_name_dict() -> dict[str, str]:
    """Return ``{metric_key: display_name}`` for embedding in report JS."""
    return {k: v.display_name for k, v in METRIC_DEFINITIONS.items()}


def tooltip_dict() -> dict[str, str]:
    """Return ``{metric_key: tooltip_text}`` for embedding in report JS."""
    return {k: v.tooltip for k, v in METRIC_DEFINITIONS.items()}
