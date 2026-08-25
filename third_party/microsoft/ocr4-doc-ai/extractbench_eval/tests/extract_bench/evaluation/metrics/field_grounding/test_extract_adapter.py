from __future__ import annotations

from types import SimpleNamespace

import pytest

from extract_bench.evaluation.metrics.field_grounding.extract_adapter import compute_extract_field_grounding_metrics
from extract_bench.test_cases.schema import ExtractFieldBbox, ExtractFieldTestRule


def test_extract_metrics_emit_all_five_scores_with_citations() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="invoice.total",
            expected_value="Service revenue",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.2, 0.1])],
        ),
        ExtractFieldTestRule(
            field_path="invoice.date",
            expected_value="2024-12-25",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.5, 0.1, 0.2, 0.1])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="invoice.total", page=1, bbox=[0.1, 0.1, 0.2, 0.1]),
        SimpleNamespace(field_path="invoice.date", page=1, bbox=[0.5, 0.1, 0.2, 0.1]),
    ]

    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"invoice": {"total": "Service revnue", "date": "Dec 25, 2024"}},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {metric.metric_name: metric for metric in metrics}

    assert {"precision", "recall", "f1", "iou", "bbox_recall"}.issubset(by_name)
    assert by_name["precision"].value == pytest.approx(1.0)
    assert by_name["recall"].value == pytest.approx(1.0)
    assert by_name["f1"].value == pytest.approx(1.0)
    assert by_name["iou"].value == pytest.approx(1.0)
    assert by_name["bbox_recall"].value == pytest.approx(1.0)


def test_extract_element_pass_rate_metadata_includes_verified_counts() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="invoice.total",
            expected_value="Service revenue",
            verified=True,
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.2, 0.1])],
        ),
        ExtractFieldTestRule(
            field_path="invoice.date",
            expected_value="2024-12-25",
            verified=False,
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.5, 0.1, 0.2, 0.1])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="invoice.total", page=1, bbox=[0.1, 0.1, 0.2, 0.1]),
        SimpleNamespace(field_path="invoice.date", page=1, bbox=[0.5, 0.1, 0.2, 0.1]),
    ]

    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"invoice": {"total": "Service revenue", "date": "bad"}},
        field_rules=rules,
        field_citations=citations,
    )
    element = {metric.metric_name: metric for metric in metrics}["extract_element_pass_rate"]

    assert element.metadata["passed"] == 1
    assert element.metadata["total"] == 2
    assert element.metadata["verified_passed"] == 1
    assert element.metadata["verified_total"] == 1
    assert element.metadata["rule_results"][0]["verified"] is True
    assert element.metadata["rule_results"][1]["verified"] is False


def test_extract_element_pass_rate_metadata_includes_all_rule_results() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path=f"rows[{index}].amount",
            expected_value=index,
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.01 * index, 0.1, 0.005])],
        )
        for index in range(25)
    ]
    citations = [
        SimpleNamespace(field_path=f"rows[{index}].amount", page=1, bbox=[0.1, 0.01 * index, 0.1, 0.005])
        for index in range(25)
    ]

    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"amount": index} for index in range(25)]},
        field_rules=rules,
        field_citations=citations,
    )
    element = {metric.metric_name: metric for metric in metrics}["extract_element_pass_rate"]

    assert element.metadata["total"] == 25
    assert len(element.metadata["rule_results"]) == 25
    assert element.metadata["rule_results"][24]["field_path"] == "rows[24].amount"


def test_extract_wrong_present_value_counts_as_fp_and_fn() -> None:
    rules = [ExtractFieldTestRule(field_path="a", expected_value="alpha")]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"a": "omega"},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {metric.metric_name: metric for metric in metrics}
    assert by_name["precision"].metadata["tp"] == 0
    assert by_name["precision"].metadata["fp"] == 1
    assert by_name["precision"].metadata["fn"] == 1
    assert by_name["precision"].value == 0.0
    assert by_name["recall"].value == 0.0


def test_array_fields_match_by_field_family_not_exact_row_index() -> None:
    rules = [
        ExtractFieldTestRule(field_path="rows[0].sku", expected_value="A-1"),
        ExtractFieldTestRule(field_path="rows[1].sku", expected_value="B-2"),
        ExtractFieldTestRule(field_path="rows[2].sku", expected_value="C-3"),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"sku": "B-2"}, {"sku": "C-3"}]},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {metric.metric_name: metric for metric in metrics}

    assert by_name["precision"].metadata["tp"] == 2
    assert by_name["precision"].metadata["fp"] == 0
    assert by_name["precision"].metadata["fn"] == 1
    assert by_name["precision"].value == pytest.approx(1.0)
    assert by_name["recall"].value == pytest.approx(2 / 3)


def test_extract_pass_rate_attribution_is_index_tolerant() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="line_items[0].quantity",
            expected_value=5,
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.1, 0.1])],
        ),
        ExtractFieldTestRule(
            field_path="line_items[1].quantity",
            expected_value=7,
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.3, 0.1, 0.1, 0.1])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="line_items[1].quantity", page=1, bbox=[0.1, 0.1, 0.1, 0.1]),
        SimpleNamespace(field_path="line_items[0].quantity", page=1, bbox=[0.3, 0.1, 0.1, 0.1]),
    ]

    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"line_items": [{"quantity": 7}, {"quantity": 5}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {metric.metric_name: metric for metric in metrics}

    assert by_name["precision"].value == 1.0
    assert by_name["recall"].value == 1.0
    assert by_name["extract_localization_pass_rate"].value == 1.0
    assert by_name["extract_attribution_pass_rate"].value == 1.0
    assert by_name["extract_element_pass_rate"].value == 1.0


def test_extract_pass_rate_uses_geometry_to_break_repeated_value_ties() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="rows[0].posted_date",
            expected_value="10/02",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.10, 0.08, 0.04])],
        ),
        ExtractFieldTestRule(
            field_path="rows[1].posted_date",
            expected_value="10/02",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.20, 0.08, 0.04])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="rows[0].posted_date", page=1, bbox=[0.10, 0.20, 0.08, 0.04]),
        SimpleNamespace(field_path="rows[1].posted_date", page=1, bbox=[0.10, 0.10, 0.08, 0.04]),
    ]

    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"posted_date": "10/02"}, {"posted_date": "10/02"}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {metric.metric_name: metric for metric in metrics}
    rule_results = by_name["extract_localization_pass_rate"].metadata["rule_results"]

    assert by_name["extract_localization_pass_rate"].value == 1.0
    assert by_name["extract_attribution_pass_rate"].value == 1.0
    assert rule_results[0]["matched_pred_field_path"] == "rows[1].posted_date"
    assert rule_results[1]["matched_pred_field_path"] == "rows[0].posted_date"


def test_extract_pass_rate_localizes_before_scoring_attribution() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="rows[0].name",
            expected_value="Alpha",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.10, 0.20, 0.04])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="rows[0].name", page=1, bbox=[0.10, 0.10, 0.20, 0.04]),
        SimpleNamespace(field_path="rows[1].name", page=1, bbox=[0.10, 0.70, 0.20, 0.04]),
    ]

    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"name": "Wrong"}, {"name": "Alpha"}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {metric.metric_name: metric for metric in metrics}
    rule_result = by_name["extract_localization_pass_rate"].metadata["rule_results"][0]

    assert by_name["extract_localization_pass_rate"].value == 1.0
    assert by_name["extract_attribution_pass_rate"].value == 0.0
    assert by_name["extract_element_pass_rate"].value == 0.0
    assert rule_result["matched_pred_field_path"] == "rows[0].name"
    assert rule_result["localization_reason"] == "pass"


def test_extract_pass_rate_multiline_value_uses_matched_prediction_path() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="electronic_deposits_bank_credits[3].transaction_detail",
            expected_value=(
                "IN Trucks Insura ACH Pmt 231002 11110032239 Down Payment - Brazos - Intercontinental Trucking"
            ),
            bboxes=[
                ExtractFieldBbox(page=1, bbox=[0.39, 0.57, 0.50, 0.014]),
                ExtractFieldBbox(page=1, bbox=[0.39, 0.584, 0.17, 0.014]),
            ],
        )
    ]
    citations = [
        SimpleNamespace(
            field_path="electronic_deposits_bank_credits[0].transaction_detail",
            page=1,
            bbox=[0.39, 0.20, 0.50, 0.028],
        ),
        SimpleNamespace(
            field_path="electronic_deposits_bank_credits[3].transaction_detail",
            page=1,
            bbox=[0.39, 0.57, 0.50, 0.028],
        ),
    ]

    metrics = compute_extract_field_grounding_metrics(
        extracted_data={
            "electronic_deposits_bank_credits": [
                {"transaction_detail": "Capital Premium Ins. Pmt 231002"},
                {},
                {},
                {
                    "transaction_detail": (
                        "IN Trucks Insura ACH Pmt 231002 11110032239 Down Payment - Brazos - Intercontinental Trucking"
                    )
                },
            ]
        },
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {metric.metric_name: metric for metric in metrics}
    rule_result = by_name["extract_localization_pass_rate"].metadata["rule_results"][0]

    assert by_name["extract_localization_pass_rate"].value == 1.0
    assert by_name["extract_attribution_pass_rate"].value == 1.0
    assert rule_result["matched_pred_field_path"] == "electronic_deposits_bank_credits[3].transaction_detail"


def test_null_expected_values_are_excluded_from_text_metrics() -> None:
    rules = [
        ExtractFieldTestRule(field_path="rows[0]", expected_value=None),
        ExtractFieldTestRule(field_path="rows[0].sku", expected_value="A-1"),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"sku": "A-1"}]},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {metric.metric_name: metric for metric in metrics}

    assert by_name["precision"].metadata["total_gt"] == 1
    assert by_name["precision"].metadata["tp"] == 1


def test_bbox_metrics_index_insensitive_within_field_family() -> None:
    """A pred at row[5] still credits a GT at row[3] when both share the same row coordinates.

    Providers regularly emit row entries at a different list index than GT
    (e.g., they skip a header row, or order rows differently), so binding
    bbox scope to the exact field_path would punish correct extractions for
    bookkeeping reasons. Bbox scope is the field-family pattern, matching
    the value-metric semantics.
    """
    rules = [
        ExtractFieldTestRule(
            field_path="rows[3].sku",
            expected_value="A-1",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.30, 0.20, 0.05])],
        ),
        ExtractFieldTestRule(
            field_path="rows[4].sku",
            expected_value="B-2",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.40, 0.20, 0.05])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="rows[0].sku", page=1, bbox=[0.10, 0.30, 0.20, 0.05]),
        SimpleNamespace(field_path="rows[1].sku", page=1, bbox=[0.10, 0.40, 0.20, 0.05]),
    ]

    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"sku": "A-1"}, {"sku": "B-2"}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {metric.metric_name: metric for metric in metrics}
    assert by_name["iou"].value == pytest.approx(1.0)
    assert by_name["bbox_recall"].value == pytest.approx(1.0)


def test_bbox_metrics_dont_cross_different_field_families() -> None:
    """Two unrelated fields with overlapping geometry must not credit each other.

    A pred for `rows[].sku` placed on the `rows[].price` GT must not be
    credited: the pattern-based scope still separates distinct field
    families. Otherwise IoU would gain points from a misaligned prediction.
    """
    rules = [
        ExtractFieldTestRule(
            field_path="rows[0].price",
            expected_value="9.99",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.30, 0.20, 0.05])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="rows[0].sku", page=1, bbox=[0.10, 0.30, 0.20, 0.05]),
    ]

    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"price": 9.99, "sku": "A-1"}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {metric.metric_name: metric for metric in metrics}
    assert by_name["iou"].value == pytest.approx(0.0)
    assert by_name["bbox_recall"].value == pytest.approx(0.0)


def test_record_metrics_perfect_extraction() -> None:
    rules = [
        ExtractFieldTestRule(field_path="rows[0].sku", expected_value="A-1"),
        ExtractFieldTestRule(field_path="rows[0].price", expected_value=10.0),
        ExtractFieldTestRule(field_path="rows[1].sku", expected_value="B-2"),
        ExtractFieldTestRule(field_path="rows[1].price", expected_value=20.0),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"sku": "A-1", "price": 10.0}, {"sku": "B-2", "price": 20.0}]},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}
    assert by_name["record_precision"].value == pytest.approx(1.0)
    assert by_name["record_recall"].value == pytest.approx(1.0)
    assert by_name["record_f1"].value == pytest.approx(1.0)
    assert by_name["record_accuracy"].value == pytest.approx(1.0)


def test_record_accuracy_is_jaccard_of_match_set() -> None:
    """Accuracy = TP / (TP + FP + FN). Distinct from precision, recall, F1."""
    # 3 GT records, 4 pred records, 2 strict TPs (rows[0], rows[1] correct).
    # rows[2] GT missing entirely (FN). Two extra pred rows hallucinated (FP).
    rules = [
        ExtractFieldTestRule(field_path="rows[0].sku", expected_value="A"),
        ExtractFieldTestRule(field_path="rows[0].price", expected_value=1.0),
        ExtractFieldTestRule(field_path="rows[1].sku", expected_value="B"),
        ExtractFieldTestRule(field_path="rows[1].price", expected_value=2.0),
        ExtractFieldTestRule(field_path="rows[2].sku", expected_value="C"),
        ExtractFieldTestRule(field_path="rows[2].price", expected_value=3.0),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={
            "rows": [
                {"sku": "A", "price": 1.0},
                {"sku": "B", "price": 2.0},
                {"sku": "X", "price": 99.0},
                {"sku": "Y", "price": 88.0},
            ]
        },
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}
    # tp=2, fp=2, fn=1
    assert by_name["record_recall"].value == pytest.approx(2 / 3)
    assert by_name["record_precision"].value == pytest.approx(2 / 4)
    assert by_name["record_f1"].value == pytest.approx(2 * (1 / 2) * (2 / 3) / ((1 / 2) + (2 / 3)))
    # Accuracy = 2 / (2 + 2 + 1) = 0.4 — strictly below F1 = 0.571
    assert by_name["record_accuracy"].value == pytest.approx(2 / 5)
    assert by_name["record_accuracy"].value < by_name["record_f1"].value


def test_record_metrics_pure_value_swap_is_caught_strictly() -> None:
    """Alice/Bob swap: every field appears under a row, but no row is correct.

    Alignment by majority overlap pairs each GT with its swap-partner pred row
    (since two of three fields match there). Strict TP then requires *every*
    field to match — and the third field does not, so neither record is a TP.
    Result: P=R=F1=0, even though field-level F1 would be 1.0.
    """
    rules = [
        ExtractFieldTestRule(field_path="employees[0].name", expected_value="Alice"),
        ExtractFieldTestRule(field_path="employees[0].dob", expected_value="1990-01-01"),
        ExtractFieldTestRule(field_path="employees[0].salary", expected_value=50000),
        ExtractFieldTestRule(field_path="employees[1].name", expected_value="Bob"),
        ExtractFieldTestRule(field_path="employees[1].dob", expected_value="1985-05-05"),
        ExtractFieldTestRule(field_path="employees[1].salary", expected_value=60000),
    ]
    extracted_data = {
        "employees": [
            {"name": "Alice", "dob": "1985-05-05", "salary": 60000},
            {"name": "Bob", "dob": "1990-01-01", "salary": 50000},
        ]
    }
    metrics = compute_extract_field_grounding_metrics(
        extracted_data=extracted_data,
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}
    assert by_name["f1"].value == pytest.approx(1.0)
    assert by_name["record_precision"].value == pytest.approx(0.0)
    assert by_name["record_recall"].value == pytest.approx(0.0)
    assert by_name["record_f1"].value == pytest.approx(0.0)


def test_record_metrics_partial_extraction_with_missing_rows() -> None:
    """Provider extracts the right values for half the rows. Recall halves; precision stays high."""
    rules = [
        ExtractFieldTestRule(field_path="rows[0].sku", expected_value="A"),
        ExtractFieldTestRule(field_path="rows[0].price", expected_value=1.0),
        ExtractFieldTestRule(field_path="rows[1].sku", expected_value="B"),
        ExtractFieldTestRule(field_path="rows[1].price", expected_value=2.0),
        ExtractFieldTestRule(field_path="rows[2].sku", expected_value="C"),
        ExtractFieldTestRule(field_path="rows[2].price", expected_value=3.0),
        ExtractFieldTestRule(field_path="rows[3].sku", expected_value="D"),
        ExtractFieldTestRule(field_path="rows[3].price", expected_value=4.0),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"sku": "A", "price": 1.0}, {"sku": "C", "price": 3.0}]},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}
    assert by_name["record_precision"].value == pytest.approx(1.0)
    assert by_name["record_recall"].value == pytest.approx(0.5)
    assert by_name["record_f1"].value == pytest.approx(2 * 1.0 * 0.5 / 1.5)


def test_record_metrics_hallucinated_extra_record_drops_precision() -> None:
    rules = [
        ExtractFieldTestRule(field_path="rows[0].sku", expected_value="A"),
        ExtractFieldTestRule(field_path="rows[0].price", expected_value=1.0),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"sku": "A", "price": 1.0}, {"sku": "X", "price": 99.0}]},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}
    assert by_name["record_recall"].value == pytest.approx(1.0)
    assert by_name["record_precision"].value == pytest.approx(0.5)


def test_record_metrics_one_field_off_fails_record_strictly() -> None:
    """An OCR slip on one field of an otherwise-correct row drops that record entirely."""
    rules = [
        ExtractFieldTestRule(field_path="rows[0].sku", expected_value="A"),
        ExtractFieldTestRule(field_path="rows[0].price", expected_value=1.0),
        ExtractFieldTestRule(field_path="rows[0].qty", expected_value=10),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"sku": "A", "price": 999.0, "qty": 10}]},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}
    assert by_name["record_precision"].value == pytest.approx(0.0)
    assert by_name["record_recall"].value == pytest.approx(0.0)


def test_record_metrics_skip_scalar_paths_and_stray_rules() -> None:
    """Scalar fields and stray (null-valued) rules don't define records.

    A doc whose only rules are a scalar path plus one stray rule should emit
    no record-level metrics — there are no records to score.
    """
    rules = [
        ExtractFieldTestRule(field_path="invoice.total", expected_value="alpha"),
        ExtractFieldTestRule(
            field_path="rows[0].sku",
            expected_value=None,
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.0, 0.0, 0.1, 0.1])],
        ),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"invoice": {"total": "alpha"}, "rows": [{"sku": "X"}]},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}
    assert "record_precision" not in by_name
    assert "record_recall" not in by_name
    assert "record_f1" not in by_name
    assert by_name["f1"].value == pytest.approx(1.0)


def test_record_grounded_recall_drops_when_citations_miss_pixels() -> None:
    """Text matches but citations point at wrong pixels — text tp=1, grounded tp=0."""
    rules = [
        ExtractFieldTestRule(
            field_path="rows[0].sku",
            expected_value="A",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.30, 0.20, 0.05])],
        ),
        ExtractFieldTestRule(
            field_path="rows[0].price",
            expected_value=1.0,
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.40, 0.30, 0.10, 0.05])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="rows[0].sku", page=1, bbox=[0.80, 0.80, 0.10, 0.05]),
        SimpleNamespace(field_path="rows[0].price", page=1, bbox=[0.85, 0.85, 0.10, 0.05]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"sku": "A", "price": 1.0}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}
    assert by_name["record_recall"].value == pytest.approx(1.0)
    assert by_name["record_grounded_recall"].value == pytest.approx(0.0)


def test_record_grounded_recall_passes_when_citations_align() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="rows[0].sku",
            expected_value="A",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.30, 0.20, 0.05])],
        ),
        ExtractFieldTestRule(
            field_path="rows[0].price",
            expected_value=1.0,
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.40, 0.30, 0.10, 0.05])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="rows[0].sku", page=1, bbox=[0.10, 0.30, 0.20, 0.05]),
        SimpleNamespace(field_path="rows[0].price", page=1, bbox=[0.40, 0.30, 0.10, 0.05]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"sku": "A", "price": 1.0}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}
    assert by_name["record_recall"].value == pytest.approx(1.0)
    assert by_name["record_grounded_recall"].value == pytest.approx(1.0)


def test_record_grounded_recall_pred_at_different_index_still_aligns() -> None:
    """Pred row 0 is GT row 1 (provider shifted indices). Alignment + grounded check must follow."""
    rules = [
        ExtractFieldTestRule(
            field_path="rows[0].sku",
            expected_value="A",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.10, 0.20, 0.05])],
        ),
        ExtractFieldTestRule(
            field_path="rows[1].sku",
            expected_value="B",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.30, 0.20, 0.05])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="rows[0].sku", page=1, bbox=[0.10, 0.30, 0.20, 0.05]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"sku": "B"}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}
    assert by_name["record_recall"].value == pytest.approx(0.5)
    assert by_name["record_grounded_recall"].value == pytest.approx(0.5)


def test_stray_rule_is_excluded_from_text_metrics_but_included_in_bbox() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="stray[0]",
            expected_value=None,
            tags=["stray"],
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.0, 0.0, 0.1, 0.1])],
        )
    ]
    metrics = compute_extract_field_grounding_metrics(extracted_data={}, field_rules=rules, field_citations=[])
    by_name = {metric.metric_name: metric for metric in metrics}
    assert "precision" not in by_name
    assert by_name["iou"].metadata["gt_count"] == 1
    assert by_name["bbox_recall"].value == 0.0


def test_null_hallucination_metric_absent_when_no_null_rules() -> None:
    rules = [ExtractFieldTestRule(field_path="a", expected_value="alpha")]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"a": "alpha"}, field_rules=rules, field_citations=[]
    )
    by_name = {m.metric_name: m for m in metrics}
    assert "null_hallucination_rate" not in by_name


def test_null_hallucination_metric_zero_when_model_correctly_skips() -> None:
    rules = [
        ExtractFieldTestRule(field_path="employees[0].balance", expected_value=None),
        ExtractFieldTestRule(field_path="employees[1].balance", expected_value=None),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"employees": [{"balance": None}, {}]},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}
    halluc = by_name["null_hallucination_rate"]
    assert halluc.value == pytest.approx(0.0)
    assert halluc.metadata["tp"] == 2
    assert halluc.metadata["fp"] == 0
    assert halluc.metadata["fn"] == 0
    assert halluc.metadata["total_null_rules"] == 2
    assert halluc.metadata["hallucinated_paths"] == []


def test_null_hallucination_metric_one_when_model_emits_for_every_null() -> None:
    rules = [
        ExtractFieldTestRule(field_path="employees[0].balance", expected_value=None),
        ExtractFieldTestRule(field_path="employees[1].balance", expected_value=None),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"employees": [{"balance": "$100"}, {"balance": "$200"}]},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}
    halluc = by_name["null_hallucination_rate"]
    assert halluc.value == pytest.approx(1.0)
    assert halluc.metadata["tp"] == 0
    assert halluc.metadata["fp"] == 2
    assert halluc.metadata["hallucinated_count"] == 2
    paths = {p["field_path"] for p in halluc.metadata["hallucinated_paths"]}
    assert paths == {"employees[0].balance", "employees[1].balance"}


def test_null_hallucination_metric_mixed_outcomes() -> None:
    rules = [
        ExtractFieldTestRule(field_path="employees[0].balance", expected_value=None),
        ExtractFieldTestRule(field_path="employees[1].balance", expected_value=None),
        ExtractFieldTestRule(field_path="employees[2].balance", expected_value=None),
        ExtractFieldTestRule(field_path="employees[3].balance", expected_value=None),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={
            "employees": [
                {"balance": None},
                {"balance": "$200"},
                {},
                {"balance": 0},
            ]
        },
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}
    halluc = by_name["null_hallucination_rate"]
    assert halluc.value == pytest.approx(0.5)
    assert halluc.metadata["tp"] == 2
    assert halluc.metadata["fp"] == 2


def test_null_hallucination_excludes_unverified_rules() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="employees[0].balance",
            expected_value=None,
            verified=False,
            tags=["bronze_expected_null_got_text"],
        ),
        ExtractFieldTestRule(field_path="employees[1].balance", expected_value=None, verified=True),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"employees": [{"balance": "$100"}, {"balance": None}]},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}
    halluc = by_name["null_hallucination_rate"]
    assert halluc.metadata["total_null_rules"] == 1
    assert halluc.metadata["tp"] == 1
    assert halluc.metadata["fp"] == 0
    assert halluc.value == pytest.approx(0.0)


def test_null_hallucination_metric_handles_phantom_record_paths() -> None:
    """Whole-record nulls (``stock_list[5]``) score the same as field-level nulls."""
    rules = [
        ExtractFieldTestRule(field_path="stock_list[0]", expected_value=None),
        ExtractFieldTestRule(field_path="stock_list[1]", expected_value=None),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"stock_list": [{"item": "phantom"}]},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}
    halluc = by_name["null_hallucination_rate"]
    assert halluc.metadata["tp"] == 1
    assert halluc.metadata["fp"] == 1
    assert halluc.value == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# extract_*_pass_rate per-rule metrics (mirrors parse-side localization /
# attribution / element pass rates).
# ---------------------------------------------------------------------------


def test_extract_pass_rate_metrics_absent_when_no_value_rules() -> None:
    """All-stray (or empty) rule sets emit no pass-rate metrics."""
    rules = [
        ExtractFieldTestRule(
            field_path="invoice.total",
            expected_value=None,
            tags=["stray"],
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.2, 0.1])],
        ),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}
    assert "extract_localization_pass_rate" not in by_name
    assert "extract_attribution_pass_rate" not in by_name
    assert "extract_element_pass_rate" not in by_name
    assert "extract_avg_iou" not in by_name
    assert "extract_avg_iou_matched" not in by_name
    assert "extract_avg_iou_unmatched" not in by_name


def test_extract_pass_rate_metrics_perfect_extraction() -> None:
    """Bbox overlaps + value matches => loc / attr / element all 1.0."""
    rules = [
        ExtractFieldTestRule(
            field_path="invoice.total",
            expected_value="42.50",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.2, 0.1])],
        ),
        ExtractFieldTestRule(
            field_path="invoice.date",
            expected_value="2024-12-25",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.5, 0.1, 0.2, 0.1])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="invoice.total", page=1, bbox=[0.1, 0.1, 0.2, 0.1]),
        SimpleNamespace(field_path="invoice.date", page=1, bbox=[0.5, 0.1, 0.2, 0.1]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"invoice": {"total": "42.50", "date": "2024-12-25"}},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}

    loc = by_name["extract_localization_pass_rate"]
    attr = by_name["extract_attribution_pass_rate"]
    element = by_name["extract_element_pass_rate"]

    assert loc.value == pytest.approx(1.0)
    assert attr.value == pytest.approx(1.0)
    assert element.value == pytest.approx(1.0)
    assert loc.metadata["tp"] == 2
    assert loc.metadata["fp"] == 0
    assert loc.metadata["fn"] == 0
    assert attr.metadata["tp"] == 2
    assert element.metadata["tp"] == 2
    assert loc.metadata["iou_threshold"] == 0.5
    assert loc.metadata["total"] == 2
    assert by_name["extract_avg_iou"].value == pytest.approx(1.0)
    assert by_name["extract_avg_iou_matched"].value == pytest.approx(1.0)
    assert by_name["extract_avg_iou_unmatched"].value == pytest.approx(0.0)
    assert by_name["extract_avg_iou"].metadata["total"] == 2
    assert by_name["extract_avg_iou"].metadata["matched"] == 2
    assert by_name["extract_avg_iou"].metadata["unmatched"] == 0


def test_extract_pass_rate_metrics_loc_only_when_value_wrong() -> None:
    """Bbox overlaps but emitted value mismatches => loc=1.0, attr=0.0, element=0.0."""
    rules = [
        ExtractFieldTestRule(
            field_path="invoice.total",
            expected_value="42.50",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.2, 0.1])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="invoice.total", page=1, bbox=[0.1, 0.1, 0.2, 0.1]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"invoice": {"total": "999.99"}},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}

    assert by_name["extract_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_attribution_pass_rate"].value == pytest.approx(0.0)
    assert by_name["extract_element_pass_rate"].value == pytest.approx(0.0)
    # Localization metadata still records 1 tp; attribution/element record 1 fp.
    assert by_name["extract_localization_pass_rate"].metadata["tp"] == 1
    assert by_name["extract_attribution_pass_rate"].metadata["tp"] == 0
    assert by_name["extract_attribution_pass_rate"].metadata["fp"] == 1


def test_extract_pass_rate_metrics_attr_requires_loc_pass() -> None:
    """Bbox misses but value happens to match => loc / attr / element all 0.0.

    Attribution is gated on localization passing, so a "right value, wrong
    place" outcome is treated as a failure across the board. This matches
    the parse-side semantics exactly.
    """
    rules = [
        ExtractFieldTestRule(
            field_path="invoice.total",
            expected_value="42.50",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.2, 0.1])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="invoice.total", page=1, bbox=[0.8, 0.8, 0.1, 0.05]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"invoice": {"total": "42.50"}},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}

    assert by_name["extract_localization_pass_rate"].value == pytest.approx(0.0)
    assert by_name["extract_attribution_pass_rate"].value == pytest.approx(0.0)
    assert by_name["extract_element_pass_rate"].value == pytest.approx(0.0)


def test_extract_pass_rate_metrics_accepts_relaxed_exact_localization() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="companies[11].current_rank",
            expected_value=12,
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.30, 0.20, 0.02])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="companies[11].current_rank", page=1, bbox=[0.10, 0.30, 0.097, 0.02]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"companies": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {"current_rank": 12}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}
    rule_result = by_name["extract_localization_pass_rate"].metadata["rule_results"][0]

    assert rule_result["iou"] == pytest.approx(0.485)
    assert rule_result["localization_reason"] == "pass_relaxed_iou_canonical_exact"
    assert by_name["extract_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_attribution_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_element_pass_rate"].value == pytest.approx(1.0)


def test_extract_bbox_group_selection_prefers_relaxed_localization_pass() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="companies[11].current_rank",
            expected_value=12,
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.10, 0.40, 0.10])],
        ),
    ]
    citations = [
        # Higher raw IoU, but only 67.5% GT coverage and 54% pred coverage, so
        # it fails the relaxed max-IoA gate.
        SimpleNamespace(field_path="companies[11].current_rank", page=1, bbox=[0.23, 0.10, 0.50, 0.10]),
        # Lower raw IoU, but fully inside the GT and an exact typed value match.
        SimpleNamespace(field_path="companies[11].current_rank", page=1, bbox=[0.23, 0.10, 0.13, 0.10]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"companies": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {"current_rank": 12}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}
    rule_result = by_name["extract_localization_pass_rate"].metadata["rule_results"][0]

    assert rule_result["iou"] == pytest.approx(0.325)
    assert rule_result["max_ioa"] == pytest.approx(1.0)
    assert rule_result["matched_pred_bboxes"] == [[0.23, 0.1, 0.13, 0.1]]
    assert rule_result["localization_reason"] == "pass_relaxed_iou_canonical_exact"
    assert by_name["extract_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_attribution_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_element_pass_rate"].value == pytest.approx(1.0)


def test_extract_pass_rate_metrics_relaxed_localization_rejects_fuzzy_text() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="employees[17].post",
            expected_value="Security Guard",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.30, 0.20, 0.02])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="employees[17].post", page=1, bbox=[0.10, 0.30, 0.097, 0.02]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={
            "employees": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {"post": "Secunty Guard"}]
        },
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}
    rule_result = by_name["extract_localization_pass_rate"].metadata["rule_results"][0]

    assert rule_result["iou"] == pytest.approx(0.485)
    assert rule_result["mode"] == "jaro_winkler"
    assert rule_result["canonical_exact"] is False
    assert rule_result["localization_reason"] == "iou_below_threshold"
    assert by_name["extract_localization_pass_rate"].value == pytest.approx(0.0)
    assert by_name["extract_attribution_pass_rate"].value == pytest.approx(0.0)
    assert by_name["extract_element_pass_rate"].value == pytest.approx(0.0)


def test_extract_pass_rate_metrics_relaxed_localization_accepts_short_exact_rank() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="companies[0].current_rank",
            expected_value="1",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.081913, 0.137485, 0.004391, 0.007434])],
        ),
    ]
    citations = [
        SimpleNamespace(
            field_path="companies[0].current_rank",
            page=1,
            bbox=[0.08062836021505376, 0.13521765991825874, 0.0046875, 0.011488926908088587],
        ),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"companies": [{"current_rank": "1"}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}
    rule_result = by_name["extract_localization_pass_rate"].metadata["rule_results"][0]

    assert rule_result["iou"] == pytest.approx(0.4133462430067865)
    assert rule_result["max_ioa"] == pytest.approx(0.774962472114269)
    assert rule_result["localization_reason"] == "pass_relaxed_iou_canonical_exact"
    assert by_name["extract_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_attribution_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_element_pass_rate"].value == pytest.approx(1.0)


def test_extract_pass_rate_metrics_null_empty_localization_accepts_any_overlap() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="statements[4].amount",
            expected_value="—",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.30, 0.40, 0.10])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="statements[4].amount", page=1, bbox=[0.10, 0.30, 0.004, 0.10]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"statements": [{}, {}, {}, {}, {"amount": "—"}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}
    rule_result = by_name["extract_localization_pass_rate"].metadata["rule_results"][0]

    assert rule_result["iou"] == pytest.approx(0.01)
    assert rule_result["mode"] == "null_empty"
    assert rule_result["localization_reason"] == "pass_null_empty_overlap"
    assert by_name["extract_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_attribution_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_element_pass_rate"].value == pytest.approx(1.0)


def test_extract_pass_rate_metrics_numeric_dash_placeholder_accepts_any_overlap() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="statements[4].amount",
            expected_value="—",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.30, 0.40, 0.10])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="statements[4].amount", page=1, bbox=[0.10, 0.30, 0.004, 0.10]),
    ]
    schema = {
        "type": "object",
        "properties": {
            "statements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"amount": {"type": "number"}},
                },
            },
        },
    }
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"statements": [{}, {}, {}, {}, {"amount": "—"}]},
        field_rules=rules,
        field_citations=citations,
        data_schema=schema,
    )
    by_name = {m.metric_name: m for m in metrics}
    rule_result = by_name["extract_localization_pass_rate"].metadata["rule_results"][0]

    assert rule_result["iou"] == pytest.approx(0.01)
    assert rule_result["mode"] == "null_empty"
    assert rule_result["localization_reason"] == "pass_null_empty_overlap"
    assert by_name["extract_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_attribution_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_element_pass_rate"].value == pytest.approx(1.0)


def test_extract_pass_rate_metrics_numeric_dash_placeholder_does_not_require_support() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="statements[4].amount",
            expected_value="—",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.30, 0.40, 0.10])],
        ),
    ]
    schema = {
        "type": "object",
        "properties": {
            "statements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"amount": {"type": "number"}},
                },
            },
        },
    }

    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"statements": [{}, {}, {}, {}, {"amount": "—"}]},
        field_rules=rules,
        field_citations=[],
        data_schema=schema,
    )
    by_name = {m.metric_name: m for m in metrics}
    rule_result = by_name["extract_localization_pass_rate"].metadata["rule_results"][0]

    assert rule_result["mode"] == "null_empty"
    assert rule_result["matched_pred_bboxes"] == []
    assert rule_result["localization_reason"] == "pass_null_empty_no_support"
    assert by_name["extract_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_attribution_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_element_pass_rate"].value == pytest.approx(1.0)


def test_extract_avg_iou_metrics_split_matched_and_unmatched_rules() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="invoice.total",
            expected_value="42.50",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.0, 0.0, 0.2, 0.2])],
        ),
        ExtractFieldTestRule(
            field_path="invoice.date",
            expected_value="2024-12-25",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.4, 0.0, 0.2, 0.2])],
        ),
        ExtractFieldTestRule(
            field_path="invoice.vendor",
            expected_value="Acme",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.8, 0.0, 0.1, 0.1])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="invoice.total", page=1, bbox=[0.0, 0.0, 0.2, 0.2]),
        SimpleNamespace(field_path="invoice.date", page=1, bbox=[0.5, 0.0, 0.2, 0.2]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"invoice": {"total": "42.50", "date": "2024-12-25", "vendor": "Acme"}},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}

    low_overlap_iou = 1 / 3
    assert by_name["extract_avg_iou"].value == pytest.approx((1.0 + low_overlap_iou + 0.0) / 3)
    assert by_name["extract_avg_iou_matched"].value == pytest.approx(1.0)
    assert by_name["extract_avg_iou_unmatched"].value == pytest.approx((low_overlap_iou + 0.0) / 2)
    assert by_name["extract_avg_iou"].metadata["total"] == 3
    assert by_name["extract_avg_iou"].metadata["matched"] == 1
    assert by_name["extract_avg_iou"].metadata["unmatched"] == 2


def test_extract_pass_rate_metrics_use_standard_iou_not_gt_coverage() -> None:
    """A broad table citation covering a tiny GT cell does not localize the field."""
    rules = [
        ExtractFieldTestRule(
            field_path="stock_list[0].catalog_number",
            expected_value="A399S-4",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.50, 0.10, 0.04, 0.02])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="stock_list[0].catalog_number", page=1, bbox=[0.0, 0.0, 0.90, 0.90]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"stock_list": [{"catalog_number": "A399S-4"}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}
    rule_result = by_name["extract_localization_pass_rate"].metadata["rule_results"][0]

    assert by_name["extract_localization_pass_rate"].value == pytest.approx(0.0)
    assert by_name["extract_attribution_pass_rate"].value == pytest.approx(0.0)
    assert rule_result["iou"] < 0.5
    assert by_name["iou"].value < 0.5


def test_extract_pass_rate_metrics_multi_box_gt_matches_multiple_citations() -> None:
    rules = [
        ExtractFieldTestRule(
            field_path="electronic_deposits_bank_credits[3].transaction_detail",
            expected_value=(
                "IN Trucks Insura ACH Pmt 231002 11110032239 Down Payment - Brazos - Intercontinental Trucking"
            ),
            bboxes=[
                ExtractFieldBbox(page=1, bbox=[0.10, 0.10, 0.58, 0.02]),
                ExtractFieldBbox(page=1, bbox=[0.10, 0.13, 0.22, 0.02]),
            ],
        ),
    ]
    citations = [
        SimpleNamespace(
            field_path="electronic_deposits_bank_credits[3].transaction_detail",
            page=1,
            bbox=[0.10, 0.10, 0.58, 0.02],
        ),
        SimpleNamespace(
            field_path="electronic_deposits_bank_credits[3].transaction_detail",
            page=1,
            bbox=[0.10, 0.13, 0.22, 0.02],
        ),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={
            "electronic_deposits_bank_credits": [
                {},
                {},
                {},
                {
                    "transaction_detail": (
                        "IN Trucks Insura ACH Pmt 231002 11110032239 Down Payment - Brazos - Intercontinental Trucking"
                    )
                },
            ]
        },
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}
    rule_result = by_name["extract_localization_pass_rate"].metadata["rule_results"][0]

    assert by_name["extract_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_attribution_pass_rate"].value == pytest.approx(1.0)
    assert len(rule_result["matched_pred_bboxes"]) == 2


def test_extract_pass_rate_metrics_no_citation_emitted_for_rule() -> None:
    """Rule has GT bboxes but the model emitted no citation => loc=0.0."""
    rules = [
        ExtractFieldTestRule(
            field_path="invoice.total",
            expected_value="42.50",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.2, 0.1])],
        ),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"invoice": {"total": "42.50"}},
        field_rules=rules,
        field_citations=[],
    )
    by_name = {m.metric_name: m for m in metrics}

    assert by_name["extract_localization_pass_rate"].value == pytest.approx(0.0)
    assert by_name["extract_attribution_pass_rate"].value == pytest.approx(0.0)
    assert by_name["extract_element_pass_rate"].value == pytest.approx(0.0)
    assert by_name["extract_localization_pass_rate"].metadata["fp"] == 1


def test_extract_pass_rate_metrics_index_drift_within_pattern_group_passes() -> None:
    """GT ``rows[3].sku`` and pred ``rows[5].sku`` share the same coords => loc=1.0.

    Bbox scoping uses the field-family pattern (``rows[].sku``), so an
    arbitrary index drift between GT row index and pred row index does not
    invalidate the localization match.
    """
    rules = [
        ExtractFieldTestRule(
            field_path="rows[3].sku",
            expected_value="A-1",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.30, 0.20, 0.05])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="rows[5].sku", page=1, bbox=[0.10, 0.30, 0.20, 0.05]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"sku": "A-1"}]},
        field_rules=rules,
        field_citations=citations,
    )
    by_name = {m.metric_name: m for m in metrics}
    # Localization and attribution both use the field-family pattern, so
    # index drift does not make an otherwise correct value/grounding fail.
    assert by_name["extract_localization_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_attribution_pass_rate"].value == pytest.approx(1.0)
    assert by_name["extract_element_pass_rate"].value == pytest.approx(1.0)


def test_extract_pass_rate_metrics_skip_field_paths_excluded_from_denominator() -> None:
    """Rules listed in ``skip_field_paths`` don't appear in pass-rate metadata.

    On per_table_row pipelines the list-unwrap normalizer adds scalar fields
    like ``client_id`` to ``skip_field_paths`` because they're structurally
    unreachable in the unwrapped data. Without the skip propagation those
    rules would always fail attribution (``_get_field_value`` returns
    ``_MISSING``), pushing tp/fp denominators downward artificially.
    """
    rules = [
        ExtractFieldTestRule(
            field_path="client_id",
            expected_value="ABC-123",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.10, 0.20, 0.05])],
        ),
        ExtractFieldTestRule(
            field_path="rows[0].sku",
            expected_value="A-1",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.10, 0.30, 0.20, 0.05])],
        ),
    ]
    citations = [
        SimpleNamespace(field_path="client_id", page=1, bbox=[0.10, 0.10, 0.20, 0.05]),
        SimpleNamespace(field_path="rows[0].sku", page=1, bbox=[0.10, 0.30, 0.20, 0.05]),
    ]
    metrics = compute_extract_field_grounding_metrics(
        extracted_data={"rows": [{"sku": "A-1"}]},  # client_id structurally absent
        field_rules=rules,
        field_citations=citations,
        skip_field_paths=["client_id"],
    )
    by_name = {m.metric_name: m for m in metrics}

    # Only one rule survives the skip filter, and it passes both checks.
    loc = by_name["extract_localization_pass_rate"]
    attr = by_name["extract_attribution_pass_rate"]
    element = by_name["extract_element_pass_rate"]

    assert loc.metadata["total"] == 1
    assert loc.metadata["tp"] == 1
    assert loc.metadata["fp"] == 0
    assert attr.metadata["tp"] == 1
    assert attr.metadata["fp"] == 0
    assert element.metadata["tp"] == 1
    assert element.metadata["fp"] == 0
    assert loc.value == pytest.approx(1.0)
    assert attr.value == pytest.approx(1.0)
    assert element.value == pytest.approx(1.0)
    # Skipped path is recorded for downstream provenance.
    assert loc.metadata["skipped_field_paths"] == ["client_id"]
    # The skipped rule must not appear in per-rule samples.
    assert all(rr["field_path"] != "client_id" for rr in loc.metadata["rule_results"])
