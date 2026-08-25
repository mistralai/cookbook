from __future__ import annotations

import importlib.util
from datetime import datetime
from typing import Any

import pytest

from extract_bench.evaluation.metrics.field_grounding.parse_adapter import (
    EXTRACT_FIELD_LOCALIZATION_IOU_THRESHOLD,
    FIELD_GROUNDING_STRICT_IOU_THRESHOLD,
    compute_parse_field_grounding_metrics,
)
from extract_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.schema import ExtractFieldBbox, ExtractFieldTestRule

requires_grounded_adapter = pytest.mark.skipif(
    importlib.util.find_spec("llama_cloud") is None,
    reason="dev and runners extras required; run: uv sync --extra dev --extra runners",
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_text_item(
    *,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    type_: str = "text",
) -> LayoutItemIR:
    return LayoutItemIR(
        type=type_,
        value=text,
        layout_segments=[LayoutSegmentIR(x=x, y=y, w=w, h=h)],
    )


def _make_layout_page(
    *,
    page_number: int,
    items: list[LayoutItemIR],
    width: float = 1.0,
    height: float = 1.0,
) -> ParseLayoutPageIR:
    return ParseLayoutPageIR(
        page_number=page_number,
        width=width,
        height=height,
        items=items,
    )


def _make_grounded_page_with_words(
    *,
    page_number: int,
    words: list[tuple[str, float, float, float, float]],  # (text, x, y, w, h)
) -> dict[str, Any]:
    """Build a grounded_pages payload entry with line + word grounding.

    The shape mirrors what ``_build_llamaparse_granular_pages_from_payload``
    consumes (see ``evaluation/layout_adapters/adapters.py``). LlamaParse spans
    are UTF-8 byte ``[start, end]`` integer pairs.
    """
    source_text = " ".join(w[0] for w in words)
    word_entries: list[dict[str, Any]] = []
    offset = 0
    for text, x, y, w, h in words:
        start = offset
        end = offset + len(text.encode("utf-8"))
        word_entries.append(
            {
                "span": [start, end],
                "bbox": {"x": x, "y": y, "w": w, "h": h},
            }
        )
        offset = end + 1  # +1 for the space separator (1 byte in UTF-8)

    line_start = 0
    line_end = len(source_text.encode("utf-8"))
    line_x = min(wrd[1] for wrd in words)
    line_y = min(wrd[2] for wrd in words)
    line_w = max(wrd[1] + wrd[3] for wrd in words) - line_x
    line_h = max(wrd[2] + wrd[4] for wrd in words) - line_y

    return {
        "page_number": page_number,
        "page_width": 1.0,
        "page_height": 1.0,
        "success": True,
        "items": [
            {
                "md": source_text,
                "grounding": {
                    "source": "md",
                    "lines": [
                        {
                            "span": [line_start, line_end],
                            "bbox": {"x": line_x, "y": line_y, "w": line_w, "h": line_h},
                            "words": word_entries,
                        }
                    ],
                },
            }
        ],
    }


def _make_grounded_page_with_byte_spans(
    *,
    page_number: int,
    source_text: str,
    text: str,
    bbox: tuple[float, float, float, float],
) -> dict[str, Any]:
    start = source_text.index(text)
    byte_start = len(source_text[:start].encode("utf-8"))
    byte_end = byte_start + len(text.encode("utf-8"))
    x, y, w, h = bbox
    return {
        "page_number": page_number,
        "page_width": 1.0,
        "page_height": 1.0,
        "success": True,
        "items": [
            {
                "md": source_text,
                "grounding": {
                    "source": "md",
                    "lines": [
                        {
                            "span": [byte_start, byte_end],
                            "bbox": {"x": x, "y": y, "w": w, "h": h},
                            "words": [
                                {
                                    "span": [byte_start, byte_end],
                                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                                }
                            ],
                        }
                    ],
                },
            }
        ],
    }


def _make_grounded_page_with_lines(
    *,
    page_number: int,
    lines: list[tuple[str, float, float, float, float]],  # (text, x, y, w, h)
) -> dict[str, Any]:
    return {
        "page_number": page_number,
        "page_width": 1.0,
        "page_height": 1.0,
        "success": True,
        "items": [
            {
                "md": text,
                "grounding": {
                    "source": "md",
                    "lines": [
                        {
                            "span": [0, len(text.encode("utf-8"))],
                            "bbox": {"x": x, "y": y, "w": w, "h": h},
                            "words": [
                                {
                                    "span": [0, len(text.encode("utf-8"))],
                                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                                }
                            ],
                        }
                    ],
                },
            }
            for text, x, y, w, h in lines
        ],
    }


def _make_string_rule(
    *,
    field_path: str,
    value: str,
    page: int,
    bbox: list[float],
) -> ExtractFieldTestRule:
    return ExtractFieldTestRule(
        field_path=field_path,
        expected_value=value,
        bboxes=[ExtractFieldBbox(page=page, bbox=bbox)],
    )


def _make_bool_rule(
    *,
    field_path: str,
    value: bool,
    page: int,
    bbox: list[float],
) -> ExtractFieldTestRule:
    return ExtractFieldTestRule(
        field_path=field_path,
        expected_value=value,
        bboxes=[ExtractFieldBbox(page=page, bbox=bbox)],
    )


def _make_number_rule(
    *,
    field_path: str,
    value: float | int,
    page: int,
    bbox: list[float],
) -> ExtractFieldTestRule:
    return ExtractFieldTestRule(
        field_path=field_path,
        expected_value=value,
        bboxes=[ExtractFieldBbox(page=page, bbox=bbox)],
    )


def _make_date_rule(
    *,
    field_path: str,
    value: str,
    page: int,
    bbox: list[float],
) -> ExtractFieldTestRule:
    """Date values are carried as strings; compare_field_value parses both sides."""
    return ExtractFieldTestRule(
        field_path=field_path,
        expected_value=value,
        bboxes=[ExtractFieldBbox(page=page, bbox=bbox)],
    )


def _make_parse_inference_result(
    *,
    layout_pages: list[ParseLayoutPageIR],
    grounded_pages: list[dict[str, Any]] | None = None,
    example_id: str = "test_example",
    pipeline_name: str = "test_pipeline",
) -> InferenceResult:
    now = datetime.now()
    output = ParseOutput(
        example_id=example_id,
        pipeline_name=pipeline_name,
        pages=[],
        layout_pages=layout_pages,
        grounded_pages=grounded_pages or [],
        markdown="",
    )
    return InferenceResult(
        product_type=ProductType.PARSE,
        pipeline_name=pipeline_name,
        output=output,
        request=InferenceRequest(
            example_id=example_id,
            source_file_path="/tmp/doc.pdf",
            product_type=ProductType.PARSE,
        ),
        raw_output={},
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


# ---------------------------------------------------------------------------
# Legacy tests (kept, assertions updated to new metric names)
# ---------------------------------------------------------------------------


def test_parse_content_and_bbox_metrics_from_line_subset() -> None:
    rule = ExtractFieldTestRule(
        field_path="employee.name",
        expected_value="Jane Doe",
        bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.2, 0.1])],
    )
    page = ParseLayoutPageIR(
        page_number=1,
        width=1000,
        height=1000,
        items=[
            LayoutItemIR(
                value="Jane Doe",
                bbox=LayoutSegmentIR(x=100, y=100, w=200, h=100),
            )
        ],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[page]),
        field_rules=[rule],
    )
    by_name = {metric.metric_name: metric for metric in metrics}

    assert by_name["extract_field_attribution_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_element_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_bbox_recall"].value == pytest.approx(1.0)
    assert by_name["extract_field_iou"].value == pytest.approx(1.0)


@requires_grounded_adapter
def test_parse_metrics_decode_llamaparse_utf8_byte_spans() -> None:
    rule = ExtractFieldTestRule(
        field_path="tax.box_23",
        expected_value=3243244443,
        bboxes=[ExtractFieldBbox(page=1, bbox=[0.19, 0.11, 0.06, 0.02])],
    )
    grounded = _make_grounded_page_with_byte_spans(
        page_number=1,
        source_text="Montant réel des dividendes déterminés (23): 3243244443",
        text="3243244443",
        bbox=(0.19, 0.11, 0.06, 0.02),
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[], grounded_pages=[grounded]),
        field_rules=[rule],
    )
    by_name = {metric.metric_name: metric for metric in metrics}
    [row] = by_name["extract_field_element_pass_rate"].metadata["rule_results"]

    assert row["matched_pred_text"] == "3243244443"
    assert row["element_pass"] is True
    assert by_name["extract_field_attribution_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_localization_pass_rate"].value == pytest.approx(1.0)


def test_parse_metrics_are_page_scoped() -> None:
    rule = ExtractFieldTestRule(
        field_path="employee.name",
        expected_value="Jane Doe",
        bboxes=[ExtractFieldBbox(page=2, bbox=[0.1, 0.1, 0.2, 0.1])],
    )
    page = ParseLayoutPageIR(
        page_number=1,
        width=1000,
        height=1000,
        items=[LayoutItemIR(value="Jane Doe", bbox=LayoutSegmentIR(x=100, y=100, w=200, h=100))],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[page]),
        field_rules=[rule],
    )
    by_name = {metric.metric_name: metric for metric in metrics}

    assert by_name["extract_field_attribution_pass_rate"].value == 0.0
    assert by_name["extract_field_localization_pass_rate"].value == 0.0
    assert by_name["extract_field_element_pass_rate"].value == 0.0


def test_parse_metrics_marks_table_text_present_but_ungrounded() -> None:
    rule = _make_string_rule(
        field_path="personnel[25].personnel_name",
        value="EWILLIAMS SARAH",
        page=1,
        bbox=[0.1, 0.1, 0.2, 0.04],
    )
    grounded = {
        "page_number": 1,
        "page_width": 1.0,
        "page_height": 1.0,
        "success": True,
        "items": [
            {
                "type": "table",
                "rows": [["EWILLIAMS<br/>SARAH<br/>File: 106036"]],
                "grounding": {"rows": [[{"span": [0, 36]}]]},
            }
        ],
    }

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[], grounded_pages=[grounded]),
        field_rules=[rule],
    )
    by_name = {metric.metric_name: metric for metric in metrics}
    [row] = by_name["extract_field_element_pass_rate"].metadata["rule_results"]

    assert row["element_pass"] is False
    assert row["reason"] == "text_present_but_ungrounded"
    assert row["localization_reason"] == "text_present_but_ungrounded"
    assert row["ungrounded_text_source"] == "EWILLIAMS<br/>SARAH<br/>File: 106036"


# ---------------------------------------------------------------------------
# New 9-metric taxonomy tests
# ---------------------------------------------------------------------------


def test_string_field_single_word_match() -> None:
    """GT single-word field, pred: one word unit. Word granularity wins, all rungs pass."""
    rule = _make_string_rule(
        field_path="invoice_number",
        value="INV-001",
        page=1,
        bbox=[0.10, 0.10, 0.10, 0.02],
    )
    grounded = _make_grounded_page_with_words(
        page_number=1,
        words=[("INV-001", 0.10, 0.10, 0.10, 0.02)],
    )
    layout = _make_layout_page(
        page_number=1,
        items=[_make_text_item(text="INV-001", x=0.10, y=0.10, w=0.10, h=0.02)],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(
            layout_pages=[layout],
            grounded_pages=[grounded],
        ),
        field_rules=[rule],
    )
    by_name = {m.metric_name: m for m in metrics}

    assert by_name["extract_field_element_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_attribution_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_text_similarity"].value == pytest.approx(1.0)
    mix = by_name["extract_field_gt_count"].metadata["granularity_mix"]
    # word or line can win depending on key ordering — the important property
    # is that broad layout/table regions don't get credited.
    assert mix["none"] == 0
    assert mix["word"] + mix["line"] == 1


@requires_grounded_adapter
def test_string_field_multi_word_combination_words_win() -> None:
    """GT phrase with multiple words, word combination captures it."""
    rule = _make_string_rule(
        field_path="vendor",
        value="Acme Corp Ltd",
        page=1,
        bbox=[0.10, 0.10, 0.30, 0.02],
    )
    grounded = _make_grounded_page_with_words(
        page_number=1,
        words=[
            ("Acme", 0.10, 0.10, 0.08, 0.02),
            ("Corp", 0.19, 0.10, 0.10, 0.02),
            ("Ltd", 0.30, 0.10, 0.10, 0.02),
        ],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(
            layout_pages=[_make_layout_page(page_number=1, items=[])],
            grounded_pages=[grounded],
        ),
        field_rules=[rule],
    )
    by_name = {m.metric_name: m for m in metrics}

    assert by_name["extract_field_element_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_attribution_pass_rate"].value == pytest.approx(1.0)


def test_string_field_multi_word_combination_text_line_wins() -> None:
    """GT phrase spans a full line. A text-line support unit can cover the entire phrase."""
    rule = _make_string_rule(
        field_path="vendor",
        value="Acme Corp Ltd",
        page=1,
        bbox=[0.10, 0.10, 0.30, 0.02],
    )
    # Provide only a text layout item (no grounded_pages) — text-like layout
    # items remain eligible as line support, while tables/layout regions do not.
    layout = _make_layout_page(
        page_number=1,
        items=[_make_text_item(text="Acme Corp Ltd", x=0.10, y=0.10, w=0.30, h=0.02)],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[rule],
    )
    by_name = {m.metric_name: m for m in metrics}

    assert by_name["extract_field_element_pass_rate"].value == pytest.approx(1.0)
    mix = by_name["extract_field_gt_count"].metadata["granularity_mix"]
    assert mix["line"] == 1


@requires_grounded_adapter
def test_string_field_multi_line_gt_matches_multiple_line_boxes() -> None:
    rule = ExtractFieldTestRule(
        field_path="electronic_deposits_bank_credits[3].transaction_detail",
        expected_value=(
            "IN Trucks Insura ACH Pmt 231002 11110032239 Down Payment - Brazos - Intercontinental Trucking"
        ),
        bboxes=[
            ExtractFieldBbox(page=1, bbox=[0.10, 0.10, 0.58, 0.02]),
            ExtractFieldBbox(page=1, bbox=[0.10, 0.13, 0.22, 0.02]),
        ],
    )
    grounded = _make_grounded_page_with_lines(
        page_number=1,
        lines=[
            ("IN Trucks Insura ACH Pmt 231002 11110032239 Down Payment - Brazos -", 0.10, 0.10, 0.58, 0.02),
            ("Intercontinental Trucking", 0.10, 0.13, 0.22, 0.02),
        ],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(
            layout_pages=[_make_layout_page(page_number=1, items=[])],
            grounded_pages=[grounded],
        ),
        field_rules=[rule],
    )
    by_name = {m.metric_name: m for m in metrics}
    rule_result = by_name["extract_field_element_pass_rate"].metadata["rule_results"][0]

    assert by_name["extract_field_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_attribution_pass_rate"].value == pytest.approx(1.0)
    assert rule_result["matched_pred_text"] == (
        "IN Trucks Insura ACH Pmt 231002 11110032239 Down Payment - Brazos - Intercontinental Trucking"
    )
    assert len(rule_result["matched_pred_bboxes"]) == 2


@requires_grounded_adapter
def test_string_field_multi_line_gt_can_match_one_union_line_box() -> None:
    rule = ExtractFieldTestRule(
        field_path="personnel[25].personnel_name",
        expected_value="EWILLIAMS SARAH",
        bboxes=[
            ExtractFieldBbox(page=1, bbox=[0.10, 0.10, 0.10, 0.02]),
            ExtractFieldBbox(page=1, bbox=[0.10, 0.13, 0.08, 0.02]),
        ],
    )
    grounded = _make_grounded_page_with_lines(
        page_number=1,
        lines=[("EWILLIAMS SARAH", 0.10, 0.10, 0.10, 0.05)],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(
            layout_pages=[_make_layout_page(page_number=1, items=[])],
            grounded_pages=[grounded],
        ),
        field_rules=[rule],
    )
    by_name = {m.metric_name: m for m in metrics}
    rule_result = by_name["extract_field_element_pass_rate"].metadata["rule_results"][0]

    assert rule_result["iou"] == pytest.approx(0.72)
    assert by_name["extract_field_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_attribution_pass_rate"].value == pytest.approx(1.0)
    assert len(rule_result["matched_pred_bboxes"]) == 1


def test_table_layout_item_does_not_satisfy_extract_field_rule() -> None:
    rule = _make_string_rule(
        field_path="stock_list[0].catalog_number",
        value="A399S-4",
        page=1,
        bbox=[0.50, 0.10, 0.04, 0.02],
    )
    layout = _make_layout_page(
        page_number=1,
        items=[
            _make_text_item(
                text="| Manufacturer | Catalog # |\n| --- | --- |\n| Fisher Chemical | A399S-4 |",
                x=0.0,
                y=0.0,
                w=0.90,
                h=0.90,
                type_="table",
            )
        ],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[rule],
    )
    by_name = {m.metric_name: m for m in metrics}
    rule_result = by_name["extract_field_element_pass_rate"].metadata["rule_results"][0]

    assert by_name["extract_field_localization_pass_rate"].value == pytest.approx(0.0)
    assert by_name["extract_field_attribution_pass_rate"].value == pytest.approx(0.0)
    assert rule_result["granularity"] == "none"


def test_bool_field_with_yes_no_text_attribution() -> None:
    """Bool GT (True) attributes via a nearby 'Yes' text layout item.

    Covers the bool comparator path without relying on Checkbox-Selected /
    Checkbox-Unselected layout region labels (those are detector region
    labels, not glyph-level guarantees — bool-on-checkbox attribution is
    deferred to a follow-up).
    """
    rule = _make_bool_rule(
        field_path="is_approved",
        value=True,
        page=1,
        bbox=[0.50, 0.10, 0.04, 0.02],
    )
    layout = _make_layout_page(
        page_number=1,
        items=[_make_text_item(text="Yes", x=0.50, y=0.10, w=0.04, h=0.02)],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[rule],
    )
    by_name = {m.metric_name: m for m in metrics}

    assert by_name["extract_field_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_attribution_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_element_pass_rate"].value == pytest.approx(1.0)
    # Bool rule is not string-typed → no text_similarity contribution.
    assert "extract_field_text_similarity" not in by_name


def test_number_field_tolerance_pass() -> None:
    """Number GT 1234.56 matches pred text '1234.56' via _parse_number."""
    rule = _make_number_rule(
        field_path="amount",
        value=1234.56,
        page=1,
        bbox=[0.10, 0.10, 0.10, 0.02],
    )
    layout = _make_layout_page(
        page_number=1,
        items=[_make_text_item(text="1234.56", x=0.10, y=0.10, w=0.10, h=0.02)],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[rule],
    )
    by_name = {m.metric_name: m for m in metrics}

    assert by_name["extract_field_attribution_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_element_pass_rate"].value == pytest.approx(1.0)
    # Not a string-typed rule — text_similarity should not be emitted.
    assert "extract_field_text_similarity" not in by_name


def test_date_field_ymd_match() -> None:
    """Date GT '2025-01-15' matches pred text 'January 15, 2025' via _parse_date."""
    rule = _make_date_rule(
        field_path="invoice_date",
        value="2025-01-15",
        page=1,
        bbox=[0.10, 0.10, 0.20, 0.02],
    )
    layout = _make_layout_page(
        page_number=1,
        items=[_make_text_item(text="January 15, 2025", x=0.10, y=0.10, w=0.20, h=0.02)],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[rule],
    )
    by_name = {m.metric_name: m for m in metrics}

    assert by_name["extract_field_attribution_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_element_pass_rate"].value == pytest.approx(1.0)


def test_no_support_all_rungs_fail() -> None:
    """Rule with no support in any support set: all rungs fail; no text_similarity."""
    rule = _make_string_rule(
        field_path="vendor",
        value="Acme",
        page=1,
        bbox=[0.10, 0.10, 0.10, 0.02],
    )
    # Empty layout, empty grounded_pages
    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(
            layout_pages=[_make_layout_page(page_number=1, items=[])],
        ),
        field_rules=[rule],
    )
    by_name = {m.metric_name: m for m in metrics}

    assert by_name["extract_field_localization_pass_rate"].value == pytest.approx(0.0)
    assert by_name["extract_field_attribution_pass_rate"].value == pytest.approx(0.0)
    assert by_name["extract_field_element_pass_rate"].value == pytest.approx(0.0)
    # Classification is trivially 1.0.
    assert by_name["extract_field_classification_pass_rate"].value == pytest.approx(1.0)
    assert "extract_field_text_similarity" not in by_name
    mix = by_name["extract_field_gt_count"].metadata["granularity_mix"]
    assert mix["none"] == 1


def test_rule_pass_rate_formula() -> None:
    """Two rules: rule A passes all 3 rungs, rule B fails loc+attr (cls trivially passes)."""
    rule_pass = _make_string_rule(
        field_path="vendor",
        value="Acme",
        page=1,
        bbox=[0.10, 0.10, 0.10, 0.02],
    )
    rule_fail = _make_string_rule(
        field_path="invoice_number",
        value="INV-001",
        page=1,
        bbox=[0.50, 0.50, 0.10, 0.02],  # no support near this bbox
    )
    layout = _make_layout_page(
        page_number=1,
        items=[_make_text_item(text="Acme", x=0.10, y=0.10, w=0.10, h=0.02)],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[rule_pass, rule_fail],
    )
    by_name = {m.metric_name: m for m in metrics}

    # element_pass_rate = 1/2 = 0.5
    assert by_name["extract_field_element_pass_rate"].value == pytest.approx(0.5)
    # rule_pass_rate = (1 + 1 + 1 + 0 + 1 + 0) / (2*3) = 4/6 ≈ 0.667
    assert by_name["extract_field_rule_pass_rate"].value == pytest.approx(4.0 / 6.0)
    assert by_name["extract_field_rule_pass_rate"].metadata["passed"] == 4
    assert by_name["extract_field_rule_pass_rate"].metadata["total"] == 6
    assert by_name["extract_field_localization_pass_rate"].value == pytest.approx(0.5)
    assert by_name["extract_field_classification_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_field_attribution_pass_rate"].value == pytest.approx(0.5)
    assert by_name["extract_field_avg_iou"].value == pytest.approx(0.5)
    assert by_name["extract_field_avg_iou_matched"].value == pytest.approx(1.0)
    assert by_name["extract_field_avg_iou_unmatched"].value == pytest.approx(0.0)
    assert by_name["extract_field_avg_iou"].metadata["total"] == 2
    assert by_name["extract_field_avg_iou"].metadata["matched"] == 1
    assert by_name["extract_field_avg_iou"].metadata["unmatched"] == 1


def test_text_similarity_excludes_non_string_rules() -> None:
    """3 rules (string, bool, number): only the string contributes to text_similarity."""
    string_rule = _make_string_rule(
        field_path="vendor",
        value="Acme",
        page=1,
        bbox=[0.10, 0.10, 0.10, 0.02],
    )
    bool_rule = _make_bool_rule(
        field_path="is_approved",
        value=True,
        page=1,
        bbox=[0.50, 0.10, 0.04, 0.02],
    )
    number_rule = _make_number_rule(
        field_path="amount",
        value=1234.56,
        page=1,
        bbox=[0.70, 0.10, 0.10, 0.02],
    )
    layout = _make_layout_page(
        page_number=1,
        items=[
            _make_text_item(text="Acme", x=0.10, y=0.10, w=0.10, h=0.02),
            _make_text_item(text="Yes", x=0.50, y=0.10, w=0.04, h=0.02),
            _make_text_item(text="1234.56", x=0.70, y=0.10, w=0.10, h=0.02),
        ],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[string_rule, bool_rule, number_rule],
    )
    by_name = {m.metric_name: m for m in metrics}

    assert "extract_field_text_similarity" in by_name
    assert by_name["extract_field_text_similarity"].metadata["string_rule_count"] == 1
    assert by_name["extract_field_text_similarity"].metadata["total_rule_count"] == 3


def test_rule_results_include_viz_metadata_fields() -> None:
    """Phase 1: rule_results must carry `cls_pass`, `localization_reason`,
    `matched_pred_bboxes`, and `matched_pred_text` so the attribution
    visualization can render per-rule badges, overlays, and the LCS text diff
    without a second lookup.
    """
    pass_rule = _make_string_rule(
        field_path="vendor",
        value="Acme Corp",
        page=1,
        bbox=[0.10, 0.10, 0.20, 0.02],
    )
    fail_rule = _make_string_rule(
        field_path="invoice_number",
        value="INV-001",
        page=1,
        bbox=[0.50, 0.50, 0.10, 0.02],
    )
    layout = _make_layout_page(
        page_number=1,
        items=[_make_text_item(text="Acme Corp", x=0.10, y=0.10, w=0.20, h=0.02)],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[pass_rule, fail_rule],
    )
    by_name = {m.metric_name: m for m in metrics}
    rule_results = by_name["extract_field_element_pass_rate"].metadata["rule_results"]

    assert len(rule_results) == 2

    pass_entry = next(entry for entry in rule_results if entry["field_path"] == "vendor")
    fail_entry = next(entry for entry in rule_results if entry["field_path"] == "invoice_number")

    # Every entry carries the four new fields.
    for entry in rule_results:
        assert "cls_pass" in entry
        assert "localization_reason" in entry
        assert "matched_pred_bboxes" in entry
        assert "matched_pred_text" in entry
        assert entry["cls_pass"] is True  # trivial for text fields

    # Passing rule: localization reason is "pass", matched bboxes non-empty,
    # matched text matches the layout item.
    assert pass_entry["localization_reason"] == "pass"
    assert pass_entry["matched_pred_bboxes"]
    for bbox in pass_entry["matched_pred_bboxes"]:
        assert isinstance(bbox, list)
        assert len(bbox) == 4
    assert "Acme Corp" in pass_entry["matched_pred_text"]

    # Failing rule: no candidate was picked (GT bbox has no support in vicinity).
    assert fail_entry["localization_reason"] == "no_support_match"
    assert fail_entry["matched_pred_bboxes"] == []
    assert fail_entry["matched_pred_text"] == ""


def test_rule_results_localization_reason_iou_below_threshold() -> None:
    """When a candidate group is selected but its standard IoU
    with the GT bbox falls under the extract-field localization threshold,
    the localization_reason should be ``iou_below_threshold``.

    Setup: a wide GT bbox (0.10-0.30) is partially covered by a smaller
    support unit (0.10-0.14) that sits at its left edge. Intersection covers
    roughly 20% of GT area, so IoU is below the threshold and localization
    fails.
    """
    rule = _make_string_rule(
        field_path="vendor",
        value="Acme",
        page=1,
        bbox=[0.10, 0.10, 0.20, 0.02],  # wide GT
    )
    layout = _make_layout_page(
        page_number=1,
        # Small item covering only the leftmost portion of the GT bbox.
        items=[_make_text_item(text="Acme", x=0.10, y=0.10, w=0.04, h=0.02)],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[rule],
    )
    rule_results = metrics[0].metadata["rule_results"]
    entry = rule_results[0]

    # A candidate was selected (matched_pred_text non-empty) but IoU is too low.
    assert entry["matched_pred_text"] != ""
    assert entry["matched_pred_bboxes"]
    assert entry["iou"] < EXTRACT_FIELD_LOCALIZATION_IOU_THRESHOLD
    assert entry["localization_reason"] == "iou_below_threshold"
    assert entry["loc_pass"] is False


@requires_grounded_adapter
def test_exact_numeric_word_support_beats_broader_line_support() -> None:
    """Prefer the loc-valid typed value span over a broader row line.

    This guards against K-1 numeric failures where the line bbox overlaps the
    GT better but includes label text, so the numeric attribution parser fails.
    """
    rule = _make_number_rule(
        field_path="part_iii_line_8_net_short_term_capital_gain_loss",
        value=18809,
        page=1,
        bbox=[0.10, 0.10, 0.30, 0.02],
    )
    grounded_page = _make_grounded_page_with_words(
        page_number=1,
        words=[
            ("8", 0.10, 0.10, 0.02, 0.02),
            ("Net", 0.13, 0.10, 0.04, 0.02),
            ("short-term", 0.18, 0.10, 0.07, 0.02),
            ("18809", 0.255, 0.10, 0.145, 0.02),
        ],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[], grounded_pages=[grounded_page]),
        field_rules=[rule],
    )
    entry = metrics[0].metadata["rule_results"][0]

    assert entry["matched_pred_text"] == "18809"
    assert entry["granularity"] == "word"
    assert entry["iou"] < FIELD_GROUNDING_STRICT_IOU_THRESHOLD
    assert entry["localization_reason"] == "pass_relaxed_iou_canonical_exact"
    assert entry["loc_pass"] is True
    assert entry["attr_pass"] is True
    assert entry["element_pass"] is True


@requires_grounded_adapter
def test_t4_numeric_word_support_beats_label_prefixed_line_support() -> None:
    """Prefer a T4 numeric value word over a label-prefixed line."""
    rule = _make_number_rule(
        field_path="box_18_ei_premiums",
        value=912.53,
        page=1,
        bbox=[0.15, 0.10, 0.12, 0.02],
    )
    grounded_page = _make_grounded_page_with_words(
        page_number=1,
        words=[
            ("18:", 0.10, 0.10, 0.03, 0.02),
            ("912.53", 0.15, 0.10, 0.12, 0.02),
        ],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[], grounded_pages=[grounded_page]),
        field_rules=[rule],
    )
    entry = metrics[0].metadata["rule_results"][0]

    assert entry["matched_pred_text"] == "912.53"
    assert entry["granularity"] == "word"
    assert entry["expected_type"] == "number"
    assert entry["canonical_exact"] is True
    assert entry["loc_pass"] is True
    assert entry["attr_pass"] is True
    assert entry["element_pass"] is True


@requires_grounded_adapter
def test_t4_address_word_group_beats_label_prefixed_line_support() -> None:
    """Prefer the geometry-local address words over the broader address line."""
    rule = _make_string_rule(
        field_path="employee_address",
        value="saint john",
        page=1,
        bbox=[0.21, 0.20, 0.12, 0.02],
    )
    grounded_page = _make_grounded_page_with_words(
        page_number=1,
        words=[
            ("Address:", 0.10, 0.20, 0.10, 0.02),
            ("saint", 0.21, 0.20, 0.06, 0.02),
            ("john", 0.28, 0.20, 0.05, 0.02),
        ],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[], grounded_pages=[grounded_page]),
        field_rules=[rule],
    )
    entry = metrics[0].metadata["rule_results"][0]

    assert entry["matched_pred_text"] == "saint john"
    assert entry["granularity"] == "word"
    assert entry["loc_pass"] is True
    assert entry["attr_pass"] is True
    assert entry["element_pass"] is True


@requires_grounded_adapter
def test_t4_employer_name_word_support_beats_broader_block_support() -> None:
    """Prefer the employer-name word over a larger employer block line."""
    rule = _make_string_rule(
        field_path="employer_name",
        value="testing",
        page=1,
        bbox=[0.19, 0.30, 0.07, 0.02],
    )
    grounded_page = _make_grounded_page_with_words(
        page_number=1,
        words=[
            ("Employer", 0.10, 0.30, 0.08, 0.02),
            ("testing", 0.19, 0.30, 0.07, 0.02),
            ("218-1-218-12", 0.27, 0.30, 0.13, 0.02),
            ("Smythe", 0.41, 0.30, 0.07, 0.02),
            ("Street", 0.49, 0.30, 0.06, 0.02),
        ],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[], grounded_pages=[grounded_page]),
        field_rules=[rule],
    )
    entry = metrics[0].metadata["rule_results"][0]

    assert entry["matched_pred_text"] == "testing"
    assert entry["granularity"] == "word"
    assert entry["loc_pass"] is True
    assert entry["attr_pass"] is True
    assert entry["element_pass"] is True


@requires_grounded_adapter
def test_exact_date_word_group_beats_broader_line_support() -> None:
    """Prefer the date word sequence over the full tax-year sentence line."""
    rule = _make_string_rule(
        field_path="tax_year_beginning",
        value="01 / 01 / 2023",
        page=1,
        bbox=[0.10, 0.20, 0.30, 0.02],
    )
    grounded_page = _make_grounded_page_with_words(
        page_number=1,
        words=[
            ("beginning", 0.10, 0.20, 0.07, 0.02),
            ("01", 0.18, 0.20, 0.03, 0.02),
            ("/", 0.21, 0.20, 0.01, 0.02),
            ("01", 0.22, 0.20, 0.03, 0.02),
            ("/", 0.25, 0.20, 0.01, 0.02),
            ("2023", 0.26, 0.20, 0.05, 0.02),
            ("ending", 0.31, 0.20, 0.06, 0.02),
            ("12", 0.37, 0.20, 0.03, 0.02),
        ],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[], grounded_pages=[grounded_page]),
        field_rules=[rule],
    )
    entry = metrics[0].metadata["rule_results"][0]

    assert entry["matched_pred_text"] == "01 / 01 / 2023"
    assert entry["granularity"] == "word"
    assert entry["loc_pass"] is True
    assert entry["attr_pass"] is True
    assert entry["element_pass"] is True


def test_extract_field_localization_accepts_small_bbox_mismatch_with_exact_text() -> None:
    """Small bbox granularity mismatches just under 0.5 IoU should pass."""
    rule = _make_number_rule(
        field_path="companies[11].current_rank",
        value=12,
        page=1,
        bbox=[0.10, 0.30, 0.20, 0.02],
    )
    layout = _make_layout_page(
        page_number=1,
        items=[_make_text_item(text="12", x=0.10, y=0.30, w=0.097, h=0.02)],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[rule],
    )
    entry = metrics[0].metadata["rule_results"][0]

    assert entry["matched_pred_text"] == "12"
    assert entry["iou"] == pytest.approx(0.485)
    assert entry["localization_reason"] == "pass_relaxed_iou_canonical_exact"
    assert entry["loc_pass"] is True
    assert entry["attr_pass"] is True
    assert entry["element_pass"] is True


def test_extract_field_relaxed_localization_requires_canonical_exact_text() -> None:
    """Fuzzy string attribution should not unlock relaxed localization."""
    rule = _make_string_rule(
        field_path="employees[17].post",
        value="Security Guard",
        page=1,
        bbox=[0.10, 0.30, 0.20, 0.02],
    )
    layout = _make_layout_page(
        page_number=1,
        items=[_make_text_item(text="Secunty Guard", x=0.10, y=0.30, w=0.097, h=0.02)],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[rule],
    )
    entry = metrics[0].metadata["rule_results"][0]

    assert entry["matched_pred_text"] == "Secunty Guard"
    assert entry["iou"] == pytest.approx(0.485)
    assert entry["score"] >= 0.90
    assert entry["canonical_exact"] is False
    assert entry["localization_reason"] == "iou_below_threshold"
    assert entry["loc_pass"] is False
    assert entry["attr_pass"] is False


def test_extract_field_relaxed_localization_accepts_short_exact_rank() -> None:
    """Tiny exact-text fields should tolerate modest word bbox drift."""
    rule = _make_string_rule(
        field_path="companies[0].current_rank",
        value="1",
        page=1,
        bbox=[0.081913, 0.137485, 0.004391, 0.007434],
    )
    layout = _make_layout_page(
        page_number=1,
        items=[
            _make_text_item(text="1", x=0.08062836021505376, y=0.13521765991825874, w=0.0046875, h=0.011488926908088587)
        ],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[rule],
    )
    entry = metrics[0].metadata["rule_results"][0]

    assert entry["matched_pred_text"] == "1"
    assert entry["iou"] == pytest.approx(0.4133462430067865)
    assert entry["max_ioa"] == pytest.approx(0.774962472114269)
    assert entry["localization_reason"] == "pass_relaxed_iou_canonical_exact"
    assert entry["loc_pass"] is True
    assert entry["attr_pass"] is True


def test_extract_field_null_empty_localization_accepts_any_overlap() -> None:
    """Dash/blank placeholders should not require standard field IoU."""
    rule = _make_string_rule(
        field_path="statements[4].amount",
        value="—",
        page=1,
        bbox=[0.10, 0.30, 0.40, 0.10],
    )
    layout = _make_layout_page(
        page_number=1,
        items=[_make_text_item(text="—", x=0.10, y=0.30, w=0.004, h=0.10)],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[rule],
    )
    entry = metrics[0].metadata["rule_results"][0]

    assert entry["matched_pred_text"] == "—"
    assert entry["iou"] == pytest.approx(0.01)
    assert entry["mode"] == "null_empty"
    assert entry["localization_reason"] == "pass_null_empty_overlap"
    assert entry["loc_pass"] is True
    assert entry["attr_pass"] is True
    assert entry["element_pass"] is True


def test_stray_rule_excluded_from_denominator() -> None:
    """Stray rules (expected_value=None or tagged stray) are excluded from denominator."""
    value_rule = _make_string_rule(
        field_path="vendor",
        value="Acme",
        page=1,
        bbox=[0.10, 0.10, 0.10, 0.02],
    )
    stray_rule = ExtractFieldTestRule(
        field_path="noise",
        expected_value=None,  # stray: None value
        bboxes=[ExtractFieldBbox(page=1, bbox=[0.40, 0.40, 0.10, 0.02])],
    )
    tagged_stray = ExtractFieldTestRule(
        field_path="noise2",
        expected_value="whatever",
        tags=["stray"],
        bboxes=[ExtractFieldBbox(page=1, bbox=[0.60, 0.60, 0.10, 0.02])],
    )
    layout = _make_layout_page(
        page_number=1,
        items=[_make_text_item(text="Acme", x=0.10, y=0.10, w=0.10, h=0.02)],
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=_make_parse_inference_result(layout_pages=[layout]),
        field_rules=[value_rule, stray_rule, tagged_stray],
    )
    by_name = {m.metric_name: m for m in metrics}

    # Denominator is 1 (only the value rule).
    assert by_name["extract_field_gt_count"].value == pytest.approx(1.0)
    assert by_name["extract_field_element_pass_rate"].metadata["total"] == 1
    assert by_name["extract_field_element_pass_rate"].value == pytest.approx(1.0)
