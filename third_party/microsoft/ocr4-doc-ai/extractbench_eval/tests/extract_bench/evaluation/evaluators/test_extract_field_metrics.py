"""Tests for per-rule extract_field metrics emitted by ExtractEvaluator."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.schema import ExtractTestCase


def _make_inference_result(
    extracted_data: dict[str, Any],
    *,
    field_citations: list[FieldCitation] | None = None,
) -> InferenceResult:
    now = datetime.now()
    return InferenceResult(
        request=InferenceRequest(
            example_id="doc",
            source_file_path="/tmp/doc.pdf",
            product_type=ProductType.EXTRACT,
        ),
        pipeline_name="fake",
        product_type=ProductType.EXTRACT,
        raw_output={},
        output=ExtractOutput(
            example_id="doc",
            pipeline_name="fake",
            extracted_data=extracted_data,
            field_citations=field_citations or [],
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def _make_test_case(
    *,
    expected_output: dict[str, Any] | None = None,
    test_rules: list[dict[str, Any]] | None = None,
    data_schema: dict[str, Any] | None = None,
) -> ExtractTestCase:
    return ExtractTestCase(
        test_id="group/doc",
        group="group",
        file_path="/tmp/doc.pdf",
        schema=data_schema or {"type": "object"},
        expected_output=expected_output,
        test_rules=test_rules,
    )


def _metrics_by_name(metrics: list[MetricValue]) -> dict[str, MetricValue]:
    return {m.metric_name: m for m in metrics}


def _diagnostic_metrics_by_name(result: EvaluationResult) -> dict[str, MetricValue]:
    return _metrics_by_name(result.diagnostic_metrics)


def test_parse_rules_do_not_enter_extract_rule_metrics() -> None:
    test_case = _make_test_case(
        expected_output={"po_number": "PO-1"},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "po_number",
                "expected_value": "PO-1",
            },
            {
                "type": "table",
                "cell": "[yes]",
                "top_heading": "Vehicle 1",
                "left_heading": "Navigation",
            },
        ],
    )

    result = ExtractEvaluator().evaluate(_make_inference_result({"po_number": "PO-1"}), test_case)

    metrics = _metrics_by_name(result.metrics)
    assert metrics["extract_field_value_pass_rate"].value == 1.0
    assert "rule_pass_rate" not in metrics


# -----------------------------------------------------------------------------
# Per-field metric: scalar pass / fail
# -----------------------------------------------------------------------------


def test_scalar_pass() -> None:
    tc = _make_test_case(
        expected_output={"po_number": "PO-1"},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "po_number",
                "expected_value": "PO-1",
                "bboxes": [],
                "verified": True,
            }
        ],
    )
    ir = _make_inference_result({"po_number": "PO-1"})
    result = ExtractEvaluator().evaluate(ir, tc)
    diagnostics = _diagnostic_metrics_by_name(result)
    assert "field_accuracy[po_number]" not in _metrics_by_name(result.metrics)
    assert "field_accuracy[po_number]" in diagnostics
    assert diagnostics["field_accuracy[po_number]"].value == 1.0
    assert diagnostics["field_accuracy[po_number]"].metadata["verified"] is True
    assert diagnostics["field_accuracy[po_number]"].metadata["field_path"] == "po_number"


def test_scalar_fail() -> None:
    tc = _make_test_case(
        expected_output={"po_number": "PO-1"},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "po_number",
                "expected_value": "PO-1",
                "bboxes": [],
            }
        ],
    )
    ir = _make_inference_result({"po_number": "PO-2"})
    result = ExtractEvaluator().evaluate(ir, tc)
    diagnostics = _diagnostic_metrics_by_name(result)
    assert diagnostics["field_accuracy[po_number]"].value == 0.0


def test_evaluator_emits_confidence_scoped_metrics_from_field_citations() -> None:
    tc = _make_test_case(
        expected_output={"invoice_id": "INV-1", "total": 100},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "invoice_id",
                "expected_value": "INV-1",
            },
            {
                "type": "extract_field",
                "field_path": "total",
                "expected_value": 100,
            },
        ],
    )
    ir = _make_inference_result(
        {"invoice_id": "INV-1", "total": 99},
        field_citations=[
            FieldCitation(field_path="invoice_id", page=1, confidence=0.99),
            FieldCitation(field_path="total", page=1, confidence=0.97),
        ],
    )

    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)

    assert "confidence_scoped_auc" in metrics
    assert metrics["confidence_scoped_precision_at_0_95"].value == 0.5
    assert metrics["confidence_scoped_coverage_at_0_95"].value == 1.0
    assert metrics["confidence_scoped_precision_at_0_99"].value == 1.0
    assert metrics["confidence_scoped_coverage_at_0_99"].value == 0.5
    assert metrics["confidence_scoped_auc"].metadata["fields_with_confidence"] == 2


def test_evaluator_derives_confidence_rows_from_expected_output_without_rules() -> None:
    tc = _make_test_case(expected_output={"invoice_id": "INV-1", "total": 100})
    ir = _make_inference_result(
        {"invoice_id": "INV-1", "total": 99},
        field_citations=[
            FieldCitation(field_path="invoice_id", page=1, confidence=0.99),
            FieldCitation(field_path="total", page=1, confidence=0.97),
        ],
    )

    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)

    assert "confidence_scoped_auc" in metrics
    assert metrics["confidence_scoped_auc"].metadata["total_fields"] == 2
    assert metrics["confidence_scoped_precision_at_0_99"].value == 1.0


def test_confidence_scoped_metrics_expand_structured_array_rules_to_leaves() -> None:
    tc = _make_test_case(
        expected_output={
            "line_items": [
                {"sku": "A", "amount": 100},
                {"sku": "B", "amount": 200},
            ]
        },
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "line_items",
                "comparator": {"sku": "case_insensitive", "amount": "number"},
                "structural": "match_by:sku",
                "evidence": [
                    {"page": 1, "value": {"sku": "A", "amount": 100}},
                    {"page": 1, "value": {"sku": "B", "amount": 200}},
                ],
            }
        ],
    )
    ir = _make_inference_result(
        {
            "line_items": [
                {"sku": "A", "amount": 200},
                {"sku": "B", "amount": 100},
            ]
        },
        field_citations=[
            FieldCitation(field_path="line_items[0].sku", page=1, confidence=0.99),
            FieldCitation(field_path="line_items[0].amount", page=1, confidence=0.99),
            FieldCitation(field_path="line_items[1].sku", page=1, confidence=0.99),
            FieldCitation(field_path="line_items[1].amount", page=1, confidence=0.99),
        ],
    )

    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)

    assert metrics["confidence_scoped_auc"].metadata["total_fields"] == 4
    assert metrics["confidence_scoped_auc"].metadata["fields_with_confidence"] == 4
    assert metrics["confidence_scoped_precision_at_0_95"].value == 0.5
    assert metrics["confidence_scoped_coverage_at_0_95"].value == 1.0
    rows = metrics["confidence_scoped_auc"].metadata["field_rows_sample"]
    assert {row["field_path"] for row in rows} == {
        "line_items[0].sku",
        "line_items[0].amount",
        "line_items[1].sku",
        "line_items[1].amount",
    }


def test_scalar_case_insensitive_match() -> None:
    tc = _make_test_case(
        expected_output={"company": "Acme Corp"},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "company",
                "expected_value": "Acme Corp",
            }
        ],
    )
    ir = _make_inference_result({"company": "ACME CORP"})
    result = ExtractEvaluator().evaluate(ir, tc)
    diagnostics = _diagnostic_metrics_by_name(result)
    assert diagnostics["field_accuracy[company]"].value == 1.0


# -----------------------------------------------------------------------------
# Nested path + missing path
# -----------------------------------------------------------------------------


def test_nested_array_path_pass() -> None:
    tc = _make_test_case(
        expected_output={"line_items": [{"quantity": 5}]},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "line_items[0].quantity",
                "expected_value": 5,
            }
        ],
    )
    ir = _make_inference_result({"line_items": [{"quantity": 5}]})
    result = ExtractEvaluator().evaluate(ir, tc)
    diagnostics = _diagnostic_metrics_by_name(result)
    assert diagnostics["field_accuracy[line_items[0].quantity]"].value == 1.0


def test_array_path_value_metrics_are_index_tolerant() -> None:
    tc = _make_test_case(
        expected_output={"line_items": [{"quantity": 5}, {"quantity": 7}]},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "line_items[0].quantity",
                "expected_value": 5,
            },
            {
                "type": "extract_field",
                "field_path": "line_items[1].quantity",
                "expected_value": 7,
            },
        ],
    )
    ir = _make_inference_result({"line_items": [{"quantity": 7}, {"quantity": 5}]})

    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    diagnostics = _diagnostic_metrics_by_name(result)

    assert diagnostics["field_accuracy[line_items[0].quantity]"].value == 1.0
    assert diagnostics["field_accuracy[line_items[1].quantity]"].value == 1.0
    assert metrics["extract_field_value_pass_rate"].value == 1.0


def test_extract_field_value_metrics_use_typed_date_equivalence() -> None:
    tc = _make_test_case(
        expected_output={"attendance_records": [{"date": "2023-01-02"}]},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "attendance_records[0].date",
                "expected_value": "2023-01-02",
                "verified": True,
            }
        ],
        data_schema={
            "type": "object",
            "properties": {
                "attendance_records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                        },
                    },
                }
            },
        },
    )
    ir = _make_inference_result({"attendance_records": [{"date": "Mon. Jan. 02 2023"}]})

    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    diagnostics = _diagnostic_metrics_by_name(result)

    assert diagnostics["field_accuracy[attendance_records[0].date]"].value == 1.0
    assert metrics["extract_field_value_pass_rate"].value == 1.0


def test_null_expected_field_accuracy_passes_for_explicit_null() -> None:
    tc = _make_test_case(
        expected_output={"a": None},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "a",
                "expected_value": None,
            }
        ],
    )
    ir = _make_inference_result({"a": None})

    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    diagnostics = _diagnostic_metrics_by_name(result)

    assert diagnostics["field_accuracy[a]"].value == 1.0
    assert metrics["extract_field_value_pass_rate"].value == 1.0


def test_null_expected_field_accuracy_passes_for_missing_path() -> None:
    tc = _make_test_case(
        expected_output={"a": None},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "a",
                "expected_value": None,
            }
        ],
    )
    ir = _make_inference_result({})

    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    diagnostics = _diagnostic_metrics_by_name(result)

    assert diagnostics["field_accuracy[a]"].value == 1.0
    assert metrics["extract_field_value_pass_rate"].value == 1.0


def test_missing_path_emits_zero() -> None:
    tc = _make_test_case(
        expected_output={"line_items": [{"quantity": 5}]},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "line_items[0].quantity",
                "expected_value": 5,
            }
        ],
    )
    ir = _make_inference_result({})  # no line_items at all
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    diagnostics = _diagnostic_metrics_by_name(result)
    assert "field_accuracy[line_items[0].quantity]" not in metrics
    assert "field_accuracy[line_items[0].quantity]" in diagnostics
    assert diagnostics["field_accuracy[line_items[0].quantity]"].value == 0.0


# -----------------------------------------------------------------------------
# Aggregate: extract_field_value_pass_rate
# -----------------------------------------------------------------------------


def test_aggregate_pass_rate_half() -> None:
    tc = _make_test_case(
        expected_output={"a": "1", "b": "2"},
        test_rules=[
            {"type": "extract_field", "field_path": "a", "expected_value": "1"},
            {"type": "extract_field", "field_path": "b", "expected_value": "2"},
        ],
    )
    ir = _make_inference_result({"a": "1", "b": "wrong"})
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    aggregate = metrics["extract_field_value_pass_rate"]
    assert aggregate.value == 0.5
    assert aggregate.metadata["total"] == 2
    assert aggregate.metadata["passed"] == 1


# -----------------------------------------------------------------------------
# Backward compat
# -----------------------------------------------------------------------------


def test_backward_compat_no_extract_field_rules() -> None:
    """Datasets without extract_field rules keep field_accuracy_{key} as diagnostics."""
    tc = _make_test_case(
        expected_output={"po_number": "PO-1"},
        test_rules=None,
    )
    ir = _make_inference_result({"po_number": "PO-1"})
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    diagnostics = _diagnostic_metrics_by_name(result)
    # Original per-key metric remains available outside headline metrics.
    assert "field_accuracy_po_number" not in metrics
    assert "field_accuracy_po_number" in diagnostics
    # No new extract_field metrics.
    assert "extract_field_value_pass_rate" not in metrics
    assert "extract_field_localization_pass_rate" not in metrics


def test_both_metric_schemes_emitted_when_both_present() -> None:
    tc = _make_test_case(
        expected_output={"po_number": "PO-1"},
        test_rules=[{"type": "extract_field", "field_path": "po_number", "expected_value": "PO-1"}],
    )
    ir = _make_inference_result({"po_number": "PO-1"})
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    diagnostics = _diagnostic_metrics_by_name(result)
    # legacy name scheme still present as diagnostics
    assert "field_accuracy_po_number" not in metrics
    assert "field_accuracy_po_number" in diagnostics
    # new per-rule name scheme also present as diagnostics
    assert "field_accuracy[po_number]" not in metrics
    assert "field_accuracy[po_number]" in diagnostics


def test_field_grounding_metrics_emitted_from_extract_rules_and_citations() -> None:
    tc = _make_test_case(
        expected_output={"po_number": "PO-1"},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "po_number",
                "expected_value": "PO-1",
                "bboxes": [{"page": 1, "bbox": [0.1, 0.1, 0.2, 0.1]}],
            }
        ],
    )
    ir = _make_inference_result(
        {"po_number": "PO-1"},
        field_citations=[
            FieldCitation(
                field_path="po_number",
                page=1,
                bbox=[0.1, 0.1, 0.2, 0.1],
            )
        ],
    )
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)

    assert metrics["precision"].value == 1.0
    assert metrics["recall"].value == 1.0
    assert metrics["f1"].value == 1.0
    assert metrics["iou"].value == 1.0
    assert metrics["bbox_recall"].value == 1.0


# -----------------------------------------------------------------------------
# Unverified metadata propagates
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# List-unwrap for per_table_row predictions
# -----------------------------------------------------------------------------


def _make_inference_result_list(extracted_data: list[Any]) -> InferenceResult:
    """Helper: build an InferenceResult with a list-rooted ``extracted_data``."""
    now = datetime.now()
    return InferenceResult(
        request=InferenceRequest(
            example_id="doc",
            source_file_path="/tmp/doc.pdf",
            product_type=ProductType.EXTRACT,
        ),
        pipeline_name="fake",
        product_type=ProductType.EXTRACT,
        raw_output={},
        output=ExtractOutput(
            example_id="doc",
            pipeline_name="fake",
            extracted_data=extracted_data,  # type: ignore[arg-type]
            field_citations=[],
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def test_list_unwrap_applied_on_list_rooted_prediction() -> None:
    """List root + single array-prefix → unwrap, scalar rule skipped.

    The unwrap is a pure shape adapter — it does not emit new metrics.
    Its presence is recorded in the metadata of the existing precision /
    recall / f1 metrics via ``list_unwrap_applied`` and
    ``skipped_field_paths``.
    """
    tc = _make_test_case(
        expected_output={"personnel": [{"name": "Alice"}], "client_id": "C-1"},
        test_rules=[
            {"type": "extract_field", "field_path": "personnel[0].name", "expected_value": "Alice"},
            {"type": "extract_field", "field_path": "client_id", "expected_value": "C-1"},
        ],
    )
    ir = _make_inference_result_list([{"name": "Alice"}])
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    diagnostics = _diagnostic_metrics_by_name(result)

    # No new metric names introduced.
    assert "extract_list_unwrap_applied" not in metrics
    assert "extract_list_unwrap_skipped_count" not in metrics

    # Unwrap state is carried as metadata on the existing precision metric.
    assert metrics["precision"].metadata.get("list_unwrap_applied") is True
    assert metrics["precision"].metadata.get("skipped_field_paths") == ["client_id"]

    # Scalar rule excluded — no per-rule metric for client_id, and it doesn't
    # count toward extract_field_value_pass_rate totals.
    assert "field_accuracy[client_id]" not in metrics
    assert "field_accuracy[client_id]" not in diagnostics
    assert "field_accuracy[personnel[0].name]" not in metrics
    assert "field_accuracy[personnel[0].name]" in diagnostics
    assert diagnostics["field_accuracy[personnel[0].name]"].value == 1.0
    assert metrics["extract_field_value_pass_rate"].metadata["total"] == 1
    assert metrics["extract_field_value_pass_rate"].metadata["passed"] == 1

    # Field grounding P/R/F1 denominator excludes client_id.
    assert metrics["precision"].metadata["total_gt"] == 1
    assert metrics["precision"].value == 1.0
    assert metrics["recall"].value == 1.0


def test_list_unwrap_flattens_per_table_row_document_wrappers() -> None:
    """per_table_row + per-doc schema emits wrappers; evaluator scores their array rows."""
    tc = _make_test_case(
        expected_output={
            "personnel": [{"name": "Alice", "net_pay": 100}, {"name": "Bob", "net_pay": 200}],
            "client_id": "C-1",
        },
        test_rules=[
            {"type": "extract_field", "field_path": "personnel[0].name", "expected_value": "Alice"},
            {"type": "extract_field", "field_path": "personnel[0].net_pay", "expected_value": 100},
            {"type": "extract_field", "field_path": "personnel[1].name", "expected_value": "Bob"},
            {"type": "extract_field", "field_path": "personnel[1].net_pay", "expected_value": 200},
            {"type": "extract_field", "field_path": "client_id", "expected_value": "C-1"},
        ],
    )
    ir = _make_inference_result_list(
        [
            {"client_id": "C-1", "personnel": [{"name": "Alice", "net_pay": 100}]},
            {"client_id": "C-1", "personnel": [{"name": "Bob", "net_pay": 200}]},
        ]
    )
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    diagnostics = _diagnostic_metrics_by_name(result)

    assert metrics["precision"].metadata.get("list_unwrap_applied") is True
    assert metrics["precision"].metadata.get("list_unwrap_mode") == "wrapper_merge"
    assert metrics["precision"].metadata.get("skipped_field_paths") == []
    assert "field_accuracy[client_id]" not in metrics
    assert "field_accuracy[client_id]" in diagnostics

    # Array-rooted rules plus the preserved wrapper scalar are scoreable.
    assert metrics["precision"].metadata["total_gt"] == 5
    assert metrics["precision"].metadata["total_pred"] == 5
    assert metrics["precision"].value == 1.0
    assert metrics["recall"].value == 1.0
    assert metrics["extract_field_value_pass_rate"].metadata["total"] == 5
    assert metrics["extract_field_value_pass_rate"].metadata["passed"] == 5


def test_list_unwrap_keeps_duplicate_wrapper_rows_as_precision_loss() -> None:
    """Duplicate wrapper rows remain extra predictions rather than being deduped."""
    tc = _make_test_case(
        expected_output={"personnel": [{"name": "Alice"}]},
        test_rules=[
            {"type": "extract_field", "field_path": "personnel[0].name", "expected_value": "Alice"},
        ],
    )
    ir = _make_inference_result_list(
        [
            {"personnel": [{"name": "Alice"}]},
            {"personnel": [{"name": "Alice"}]},
        ]
    )
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)

    assert metrics["precision"].metadata["tp"] == 1
    assert metrics["precision"].metadata["fp"] == 1
    assert metrics["precision"].metadata["fn"] == 0
    assert metrics["precision"].value == 0.5
    assert metrics["recall"].value == 1.0


def test_list_unwrap_not_applied_on_dict_rooted_prediction() -> None:
    """Regression: per_doc predictions (dict root) are untouched."""
    tc = _make_test_case(
        expected_output={"personnel": [{"name": "Alice"}], "client_id": "C-1"},
        test_rules=[
            {"type": "extract_field", "field_path": "personnel[0].name", "expected_value": "Alice"},
            {"type": "extract_field", "field_path": "client_id", "expected_value": "C-1"},
        ],
    )
    ir = _make_inference_result({"personnel": [{"name": "Alice"}], "client_id": "C-1"})
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    diagnostics = _diagnostic_metrics_by_name(result)

    # No new metric names introduced; unwrap state reported in metadata.
    assert "extract_list_unwrap_applied" not in metrics
    assert "extract_list_unwrap_skipped_count" not in metrics
    assert metrics["precision"].metadata.get("list_unwrap_applied") is False
    assert metrics["precision"].metadata.get("skipped_field_paths") == []

    # All rules score and are counted.
    assert "field_accuracy[personnel[0].name]" not in metrics
    assert "field_accuracy[client_id]" not in metrics
    assert "field_accuracy[personnel[0].name]" in diagnostics
    assert "field_accuracy[client_id]" in diagnostics
    assert metrics["extract_field_value_pass_rate"].metadata["total"] == 2


def test_list_unwrap_ambiguous_multi_array_not_applied() -> None:
    """List root but rules span two arrays → cannot unwrap; no-op."""
    tc = _make_test_case(
        expected_output={},
        test_rules=[
            {"type": "extract_field", "field_path": "personnel[0].name", "expected_value": "Alice"},
            {"type": "extract_field", "field_path": "transactions[0].amount", "expected_value": 50},
        ],
    )
    ir = _make_inference_result_list([{"name": "Alice"}])
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)

    assert "extract_list_unwrap_applied" not in metrics
    assert metrics["precision"].metadata.get("list_unwrap_applied") is False


def test_list_unwrap_scores_multi_array_wrapper_rows() -> None:
    """Multi-table wrappers are merged across all top-level arrays."""
    tc = _make_test_case(
        expected_output={
            "account_number": "123",
            "checks_paid": [{"amount": 10}],
            "electronic_debits_bank_debits": [{"amount": 20}],
        },
        test_rules=[
            {"type": "extract_field", "field_path": "account_number", "expected_value": "123"},
            {"type": "extract_field", "field_path": "checks_paid[0].amount", "expected_value": 10},
            {
                "type": "extract_field",
                "field_path": "electronic_debits_bank_debits[0].amount",
                "expected_value": 20,
            },
        ],
    )
    ir = _make_inference_result_list(
        [
            {"account_number": "123", "checks_paid": [{"amount": 10}], "electronic_debits_bank_debits": []},
            {"account_number": "123", "checks_paid": [], "electronic_debits_bank_debits": [{"amount": 20}]},
        ]
    )

    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)

    assert metrics["precision"].metadata.get("list_unwrap_mode") == "wrapper_merge"
    assert metrics["precision"].metadata["total_gt"] == 3
    assert metrics["precision"].metadata["total_pred"] == 3
    assert metrics["precision"].value == 1.0
    assert metrics["recall"].value == 1.0


def test_list_unwrap_scores_singleton_scalar_document_list() -> None:
    """Scalar-only extraction results can be returned as a singleton list."""
    tc = _make_test_case(
        expected_output={"tax_year": "2024", "employee_name": "Jane Doe"},
        test_rules=[
            {"type": "extract_field", "field_path": "tax_year", "expected_value": "2024"},
            {"type": "extract_field", "field_path": "employee_name", "expected_value": "Jane Doe"},
        ],
    )
    ir = _make_inference_result_list([{"tax_year": "2024", "employee_name": "Jane Doe"}])

    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)

    assert metrics["precision"].metadata.get("list_unwrap_mode") == "singleton_doc"
    assert metrics["precision"].value == 1.0
    assert metrics["recall"].value == 1.0
    assert metrics["extract_field_value_pass_rate"].metadata["total"] == 2
    assert metrics["extract_field_value_pass_rate"].metadata["passed"] == 2


def test_list_unwrap_skips_case_only_alias_rules_from_schema() -> None:
    """Schema canonical key prevents duplicate Employee/employee aliases from blocking unwrap."""
    tc = _make_test_case(
        expected_output={
            "Employee": [{"Name": "Alice"}],
            "employee": [{"Name": "Alice"}],
        },
        data_schema={"type": "object", "properties": {"Employee": {"type": "array"}}},
        test_rules=[
            {"type": "extract_field", "field_path": "Employee[0].Name", "expected_value": "Alice"},
            {"type": "extract_field", "field_path": "employee[0].Name", "expected_value": "Alice"},
        ],
    )
    ir = _make_inference_result_list([{"Employee": [{"Name": "Alice"}]}])

    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    diagnostics = _diagnostic_metrics_by_name(result)

    assert metrics["precision"].metadata.get("list_unwrap_mode") == "wrapper_merge"
    assert metrics["precision"].metadata.get("alias_skipped_field_paths") == ["employee[0].Name"]
    assert "field_accuracy[Employee[0].Name]" not in metrics
    assert "field_accuracy[Employee[0].Name]" in diagnostics
    assert "field_accuracy[employee[0].Name]" not in diagnostics
    assert metrics["precision"].metadata["total_gt"] == 1
    assert metrics["precision"].value == 1.0


def test_unverified_flag_propagates_to_metadata() -> None:
    tc = _make_test_case(
        expected_output={"a": "v"},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "a",
                "expected_value": "v",
                "verified": False,
            }
        ],
    )
    ir = _make_inference_result({"a": "v"})
    result = ExtractEvaluator().evaluate(ir, tc)
    diagnostics = _diagnostic_metrics_by_name(result)
    assert diagnostics["field_accuracy[a]"].metadata["verified"] is False


# -----------------------------------------------------------------------------
# accuracy: match_by identity keys make array pairing order-insensitive
# -----------------------------------------------------------------------------


def test_accuracy_uses_match_by_identity_keys_for_row_pairing() -> None:
    rows = [
        {"item_no": "0001", "description": "ALPHA", "amount": 100},
        {"item_no": "0002", "description": "BRAVO", "amount": 200},
        {"item_no": "0003", "description": "CHARLIE", "amount": 300},
    ]
    test_case = _make_test_case(
        expected_output={"line_items": rows},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "line_items",
                "structural": "match_by:item_no",
                "comparator": {"item_no": "case_insensitive", "amount": "number"},
                "evidence": [{"page": 1, "value": row, "coarse": True} for row in rows],
            }
        ],
    )
    reversed_result = ExtractEvaluator().evaluate(
        _make_inference_result({"line_items": list(reversed(rows))}), test_case
    )
    accuracy = _metrics_by_name(reversed_result.metrics)["accuracy"]
    assert accuracy.value == 1.0
    assert accuracy.metadata["identity_paired_paths"] == ["line_items"]

    dropped_result = ExtractEvaluator().evaluate(_make_inference_result({"line_items": rows[1:]}), test_case)
    # Only the dropped row's leaves are lost — no index-shift cascade.
    assert abs(_metrics_by_name(dropped_result.metrics)["accuracy"].value - 2 / 3) < 1e-9


def test_accuracy_without_match_by_falls_back_to_assignment_pairing() -> None:
    """Without match_by rules, rows pair by optimal assignment: reordered
    but correct rows score 1.0, while wrong values still get no credit."""
    rows = [
        {"description": "ALPHA", "amount": 100},
        {"description": "BRAVO", "amount": 200},
    ]
    test_case = _make_test_case(expected_output={"line_items": rows})
    result = ExtractEvaluator().evaluate(_make_inference_result({"line_items": list(reversed(rows))}), test_case)
    assert _metrics_by_name(result.metrics)["accuracy"].value == 1.0

    wrong_rows = [
        {"description": "CHARLIE", "amount": 700},
        {"description": "DELTA", "amount": 800},
    ]
    result = ExtractEvaluator().evaluate(_make_inference_result({"line_items": wrong_rows}), test_case)
    assert _metrics_by_name(result.metrics)["accuracy"].value < 1.0
