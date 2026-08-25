"""End-to-end integration test for the 9-metric extract_field_* taxonomy.

This test drives ``compute_parse_field_grounding_metrics`` through a mix of
string and number rules in one call and asserts the full 9-metric contract.

Bool-field attribution via checkbox glyphs is deferred to a follow-up —
``Checkbox-Selected`` / ``Checkbox-Unselected`` are detector region labels,
not glyph-level guarantees.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from extract_bench.evaluation.metrics.field_grounding.parse_adapter import (
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


def _make_text_item(*, text: str, x: float, y: float, w: float, h: float) -> LayoutItemIR:
    return LayoutItemIR(
        type="text",
        value=text,
        layout_segments=[LayoutSegmentIR(x=x, y=y, w=w, h=h)],
    )


def _make_layout_page(*, page_number: int, items: list[LayoutItemIR]) -> ParseLayoutPageIR:
    return ParseLayoutPageIR(page_number=page_number, width=1.0, height=1.0, items=items)


def _make_grounded_page_with_words(
    *,
    page_number: int,
    words: list[tuple[str, float, float, float, float]],
) -> dict[str, Any]:
    source_text = " ".join(w[0] for w in words)
    word_entries: list[dict[str, Any]] = []
    offset = 0
    for text, x, y, w, h in words:
        start = offset
        end = offset + len(text)
        word_entries.append(
            {
                "span": [start, end],
                "bbox": {"x": x, "y": y, "w": w, "h": h},
            }
        )
        offset = end + 1

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
                            "span": [0, len(source_text)],
                            "bbox": {"x": line_x, "y": line_y, "w": line_w, "h": line_h},
                            "words": word_entries,
                        }
                    ],
                },
            }
        ],
    }


def _make_string_rule(*, field_path: str, value: str, page: int, bbox: list[float]) -> ExtractFieldTestRule:
    return ExtractFieldTestRule(
        field_path=field_path,
        expected_value=value,
        bboxes=[ExtractFieldBbox(page=page, bbox=bbox)],
    )


def _make_number_rule(*, field_path: str, value: float | int, page: int, bbox: list[float]) -> ExtractFieldTestRule:
    return ExtractFieldTestRule(
        field_path=field_path,
        expected_value=value,
        bboxes=[ExtractFieldBbox(page=page, bbox=bbox)],
    )


def _make_parse_inference_result(
    *,
    layout_pages: list[ParseLayoutPageIR],
    grounded_pages: list[dict[str, Any]] | None = None,
) -> InferenceResult:
    now = datetime.now()
    output = ParseOutput(
        example_id="test",
        pipeline_name="test_pipeline",
        pages=[],
        layout_pages=layout_pages,
        grounded_pages=grounded_pages or [],
        markdown="",
    )
    return InferenceResult(
        product_type=ProductType.PARSE,
        pipeline_name="test_pipeline",
        output=output,
        request=InferenceRequest(
            example_id="test",
            source_file_path="/tmp/doc.pdf",
            product_type=ProductType.PARSE,
        ),
        raw_output={},
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def test_nine_metric_taxonomy_emitted_end_to_end() -> None:
    """One inference result with a mix of string and number rules.

    Asserts the full 9-metric contract: all 9 expected metrics are emitted,
    no legacy metric names leak through, and basic invariants hold
    (element_pass_rate <= min(loc, attr), classification trivially 1.0,
    granularity_mix counts sum to gt_count).

    Bool-field coverage is deferred to a follow-up ticket — Checkbox-*
    region labels aren't reliable glyph indicators.
    """
    layout_pages = [
        _make_layout_page(
            page_number=1,
            items=[
                _make_text_item(text="Acme Corp", x=0.10, y=0.10, w=0.20, h=0.02),
                _make_text_item(text="1234.56", x=0.70, y=0.10, w=0.10, h=0.02),
            ],
        ),
    ]
    grounded_pages = [
        _make_grounded_page_with_words(
            page_number=1,
            words=[
                ("Acme", 0.10, 0.10, 0.08, 0.02),
                ("Corp", 0.19, 0.10, 0.11, 0.02),
                ("1234.56", 0.70, 0.10, 0.10, 0.02),
            ],
        ),
    ]
    rules = [
        _make_string_rule(
            field_path="vendor",
            value="Acme Corp",
            page=1,
            bbox=[0.10, 0.10, 0.20, 0.02],
        ),
        _make_number_rule(
            field_path="amount",
            value=1234.56,
            page=1,
            bbox=[0.70, 0.10, 0.10, 0.02],
        ),
    ]
    inference_result = _make_parse_inference_result(
        layout_pages=layout_pages,
        grounded_pages=grounded_pages,
    )

    metrics = compute_parse_field_grounding_metrics(
        inference_result=inference_result,
        field_rules=rules,
    )
    by_name = {m.metric_name: m for m in metrics}

    expected_names = {
        "extract_field_element_pass_rate",
        "extract_field_rule_pass_rate",
        "extract_field_localization_pass_rate",
        "extract_field_classification_pass_rate",
        "extract_field_attribution_pass_rate",
        "extract_field_iou",
        "extract_field_bbox_recall",
        "extract_field_text_similarity",
        "extract_field_gt_count",
    }
    assert expected_names.issubset(by_name.keys()), f"missing: {expected_names - set(by_name.keys())}"

    forbidden = {
        "content_recall",
        "iou",
        "bbox_recall",
        "extract_field_precision@0.5",
        "extract_field_recall@0.5",
        "extract_field_f1@0.5",
    }
    assert not (forbidden & set(by_name.keys())), f"lingering old metrics: {forbidden & set(by_name.keys())}"

    # Contract: element_pass_rate <= min(loc, attr).
    assert (
        by_name["extract_field_element_pass_rate"].value
        <= min(
            by_name["extract_field_localization_pass_rate"].value,
            by_name["extract_field_attribution_pass_rate"].value,
        )
        + 1e-9
    )

    # Classification is trivially 1.0.
    assert by_name["extract_field_classification_pass_rate"].value == 1.0

    # Granularity mix counts sum to gt_count.
    mix = by_name["extract_field_gt_count"].metadata["granularity_mix"]
    assert sum(mix.values()) == int(by_name["extract_field_gt_count"].value)

    # Text similarity is averaged over string rules only.
    assert by_name["extract_field_text_similarity"].metadata["string_rule_count"] == 1
