from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.evaluation.evaluators.parse import ParseEvaluator
from extract_bench.evaluation.runner import EvaluationRunner, _evaluate_single_worker
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.parse_output import ParseOutput
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases import filter_verified_test_rules, load_test_cases
from extract_bench.test_cases.schema import ExtractTestCase, ParseTestCase


def _extract_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "invoice": {
                "type": "object",
                "properties": {
                    "number": {"type": "string"},
                    "date": {"type": "string"},
                },
            }
        },
    }


def _legacy_two_rule_case(tmp_path: Path, *, second_rule_verified: bool = True) -> ExtractTestCase:
    return ExtractTestCase(
        test_id="docs/payroll_7",
        group="docs",
        file_path=tmp_path / "payroll_7.pdf",
        schema=_extract_schema(),
        expected_output={"invoice": {"number": "INV-001", "date": "2026-05-01"}},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "invoice.number",
                "expected_value": "INV-001",
                "bboxes": [{"page": 1, "bbox": [0.1, 0.2, 0.3, 0.1]}],
                "verified": True,
            },
            {
                "type": "extract_field",
                "field_path": "invoice.date",
                "expected_value": "2026-05-01",
                "bboxes": [{"page": 1, "bbox": [0.5, 0.2, 0.2, 0.1]}],
                "verified": second_rule_verified,
            },
        ],
    )


def _extract_inference_result(case: ExtractTestCase, *, cite_both: bool = True) -> InferenceResult:
    now = datetime.now()
    citations = [FieldCitation(field_path="invoice.number", page=1, bbox=[0.1, 0.2, 0.3, 0.1])]
    if cite_both:
        citations.append(FieldCitation(field_path="invoice.date", page=1, bbox=[0.5, 0.2, 0.2, 0.1]))
    return InferenceResult(
        request=InferenceRequest(
            example_id=case.test_id,
            source_file_path=str(case.file_path),
            product_type=ProductType.EXTRACT,
            schema_override=case.data_schema,
        ),
        pipeline_name="test_pipeline",
        product_type=ProductType.EXTRACT,
        raw_output={"job_id": "ext-123"},
        output=ExtractOutput(
            example_id=case.test_id,
            pipeline_name="test_pipeline",
            extracted_data={"invoice": {"number": "INV-001", "date": "May 1, 2026"}},
            field_citations=citations,
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )


def test_extract_sidecar_loader_ignores_companion_jsonl(tmp_path: Path) -> None:
    pdf_path = tmp_path / "payroll_7.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    (tmp_path / "payroll_7.v2.raw_words.jsonl").write_text('{"word":"ignored"}\n', encoding="utf-8")
    (tmp_path / "payroll_7.test.json").write_text(
        json.dumps(
            {
                "data_schema": _extract_schema(),
                "expected_output": {"invoice": {"number": "INV-001"}},
                "test_rules": [
                    {
                        "type": "extract_field",
                        "field_path": "invoice.number",
                        "expected_value": "INV-001",
                        "bboxes": [{"page": 1, "bbox": [0.1, 0.2, 0.3, 0.1]}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cases = load_test_cases(tmp_path, product_type="extract")

    assert len(cases) == 1
    assert isinstance(cases[0], ExtractTestCase)
    assert cases[0].test_id == f"{tmp_path.name}/payroll_7"
    assert cases[0].get_extract_field_rules()[0].field_path == "invoice.number"


def test_extract_sidecar_loader_reads_v02_field_rules(tmp_path: Path) -> None:
    """The v0.2 `_field_rules` dict shape loads with comparator + evidence intact."""
    pdf_path = tmp_path / "receipt_1.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    (tmp_path / "receipt_1.test.json").write_text(
        json.dumps(
            {
                "data_schema": _extract_schema(),
                "expected_output": {"invoice": {"number": "INV-001"}},
                "_schema_version": "sample/v0.2",
                "_field_rules": {
                    "invoice.number": {
                        "comparator": "case_insensitive",
                        "evidence": [{"page": 1, "bbox": [0.1, 0.2, 0.3, 0.1], "quote": "INV-001", "value": "INV-001"}],
                        "source_policy": "verbatim",
                        "evidence_required": True,
                        "verified": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cases = load_test_cases(tmp_path, product_type="extract")

    assert len(cases) == 1
    case = cases[0]
    assert isinstance(case, ExtractTestCase)
    assert case.schema_version == "sample/v0.2"
    rules = case.get_extract_field_rules()
    assert len(rules) == 1
    assert rules[0].field_path == "invoice.number"
    assert rules[0].comparator == "case_insensitive"
    assert rules[0].evidence is not None and rules[0].evidence[0].quote == "INV-001"


def test_extract_evaluator_emits_unified_value_metrics(tmp_path: Path) -> None:
    case = _legacy_two_rule_case(tmp_path)
    result = _extract_inference_result(case)

    evaluated = ExtractEvaluator().evaluate(result, case)
    by_name = {metric.metric_name: metric for metric in evaluated.metrics}

    # Headline unified value metrics (the primary ExtractBench scores).
    for metric_name in (
        "extract_unified_value_precision",
        "extract_unified_value_recall",
        "extract_unified_value_f1",
    ):
        assert by_name[metric_name].value == pytest.approx(1.0)

    # Value normalization: "May 1, 2026" matched "2026-05-01" via date canonicalization.
    assert by_name["extract_field_value_pass_rate"].value == pytest.approx(1.0)

    # Grounding diagnostics for legacy bbox rules.
    for metric_name in (
        "extract_localization_pass_rate",
        "extract_attribution_pass_rate",
        "extract_element_pass_rate",
    ):
        assert by_name[metric_name].value == pytest.approx(1.0)

    assert evaluated.job_id == "ext-123"


def test_verified_only_filter_removes_unverified_rules_generically(tmp_path: Path) -> None:
    case = ParseTestCase(
        test_id="docs/payroll_7",
        group="docs",
        file_path=tmp_path / "payroll_7.pdf",
        test_rules=[
            {"type": "present", "text": "keep me"},
            {"type": "present", "text": "drop me", "verified": False},
        ],
    )

    filtered = filter_verified_test_rules(case)

    assert filtered.test_rules is not None
    assert len(filtered.test_rules) == 1
    assert filtered.test_rules[0].get("text") == "keep me"


def test_extract_evaluator_scores_filtered_verified_rules(tmp_path: Path) -> None:
    case = _legacy_two_rule_case(tmp_path, second_rule_verified=False)
    result = _extract_inference_result(case, cite_both=False)

    default_rules = case.get_extract_field_rules()
    verified_case = filter_verified_test_rules(case)
    verified_rules = verified_case.get_extract_field_rules()

    assert len(default_rules) == 2
    assert len(verified_rules) == 1

    default_metrics = {m.metric_name: m for m in ExtractEvaluator().evaluate(result, case).metrics}
    verified_metrics = {m.metric_name: m for m in ExtractEvaluator().evaluate(result, verified_case).metrics}

    # The unverified date rule (uncited) drags grounding down in the default
    # run; the verified-only run scores the cited rule alone.
    assert verified_metrics["extract_element_pass_rate"].value >= default_metrics["extract_element_pass_rate"].value


def test_parse_evaluator_scores_extract_field_grounding_rules(tmp_path: Path) -> None:
    """Parse pipelines are cross-evaluated on extract_field rules (extract_field_* namespace)."""
    case = _legacy_two_rule_case(tmp_path)
    now = datetime.now()
    result = InferenceResult(
        request=InferenceRequest(
            example_id=case.test_id,
            source_file_path=str(case.file_path),
            product_type=ProductType.PARSE,
        ),
        pipeline_name="test_parse_pipeline",
        product_type=ProductType.PARSE,
        raw_output={},
        output=ParseOutput(
            example_id=case.test_id,
            pipeline_name="test_parse_pipeline",
            markdown="Invoice INV-001 2026-05-01",
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )

    evaluated = ParseEvaluator().evaluate(result, case)
    by_name = {metric.metric_name: metric for metric in evaluated.metrics}

    assert "extract_field_element_pass_rate" in by_name
    assert "extract_field_localization_pass_rate" in by_name
    assert by_name["extract_field_gt_count"].value == 2.0


def test_extract_avg_micro_aggregation() -> None:
    runner = EvaluationRunner(output_dir=Path("/tmp/unused"))
    results = [
        EvaluationResult(
            test_id="a",
            example_id="a",
            pipeline_name="p",
            product_type="extract",
            success=True,
            metrics=[
                MetricValue(
                    metric_name="extract_element_pass_rate",
                    value=0.5,
                    metadata={"passed": 1, "total": 2, "tp": 1, "fp": 1, "fn": 0},
                )
            ],
        ),
        EvaluationResult(
            test_id="b",
            example_id="b",
            pipeline_name="p",
            product_type="extract",
            success=True,
            metrics=[
                MetricValue(
                    metric_name="extract_element_pass_rate",
                    value=1.0,
                    metadata={"passed": 3, "total": 3, "tp": 3, "fp": 0, "fn": 0},
                )
            ],
        ),
    ]

    aggregate = runner._aggregate_metrics(results)

    assert aggregate["avg_extract_element_pass_rate"] == 0.75
    assert aggregate["micro_extract_element_pass_rate"] == 0.8
    assert "macro_extract_element_pass_rate" not in aggregate
    assert aggregate["total_extract_element_pass_rate_passed"] == 4.0
    assert aggregate["total_extract_element_pass_rate_evaluated"] == 5.0
    assert aggregate["total_extract_element_pass_rate_tp"] == 4.0
    assert aggregate["total_extract_element_pass_rate_fp"] == 1.0
    assert aggregate["total_extract_element_pass_rate_fn"] == 0.0


def test_public_extract_pipelines_registered() -> None:
    from extract_bench.inference.pipelines import get_pipeline

    llamaextract = get_pipeline("llamaextract_agentic")
    assert llamaextract.product_type == ProductType.EXTRACT
    assert llamaextract.provider_name == "llamaextract_v2"
    assert llamaextract.config["tier"] == "agentic"
    assert llamaextract.config["cite_sources"] is True
    assert "use_staging" not in llamaextract.config

    for tier in ("cost_effective", "agentic"):
        default = get_pipeline(f"llamaextract_{tier}")
        parse_config = default.config["parse_config"]
        assert parse_config["output_options"]["granular_bboxes"] == ["word"]
        assert parse_config["tier"] == tier
        assert default.config["parse_tier"] == tier
        assert "use_staging" not in default.config

        standard = get_pipeline(f"llamaextract_{tier}_standard_bbox")
        assert standard.config["tier"] == tier
        assert "parse_config" not in standard.config
        assert standard.config["cite_sources"] is True

    openai_oneshot = get_pipeline("openai_gpt_5_4_extract_oneshot_structured_output_file")
    assert openai_oneshot.provider_name == "openai_extract"
    assert openai_oneshot.config["model"] == "gpt-5.4"

    gemini_3_5 = get_pipeline("gemini_3_5_flash_extract_oneshot_structured_output_file")
    assert gemini_3_5.provider_name == "gemini_extract"

    gemini_3_6 = get_pipeline("gemini_3_6_flash_extract_oneshot_structured_output_file")
    assert gemini_3_6.provider_name == "gemini_extract"
    assert gemini_3_6.config["model"] == "gemini-3.6-flash"
    assert gemini_3_6.config["thinking_level"] == "medium"

    gemini_3_7 = get_pipeline("gemini_3_7_flash_extract_oneshot_structured_output_file")
    assert gemini_3_7.provider_name == "gemini_extract"
    assert gemini_3_7.config["model"] == "gemini-3.7-flash"
    assert gemini_3_7.config["thinking_level"] == "medium"

    gemini_3_6_twostage = get_pipeline("gemini_3_6_flash_extract_twostage_parse_agentic_structured_output_text")
    assert gemini_3_6_twostage.provider_name == "gemini_extract"
    assert gemini_3_6_twostage.config["model"] == "gemini-3.6-flash"

    gemini_3_7_twostage = get_pipeline("gemini_3_7_flash_extract_twostage_parse_agentic_structured_output_text")
    assert gemini_3_7_twostage.provider_name == "gemini_extract"
    assert gemini_3_7_twostage.config["model"] == "gemini-3.7-flash"

    twostage = get_pipeline("openai_gpt_5_4_extract_twostage_parse_agentic_structured_output_text")
    assert twostage.config["input_mode"] == "parsed_text"
    assert twostage.config["parse_source"]["type"] == "llamaparse"

    extend = get_pipeline("extend_extract")
    assert extend.provider_name == "extend"
    assert extend.config["advancedOptions"]["citationsEnabled"] is True


def test_parallel_worker_respects_verified_only_flag(tmp_path: Path) -> None:
    case = _legacy_two_rule_case(tmp_path, second_rule_verified=False)
    result = _extract_inference_result(case, cite_both=False)

    worker_result = _evaluate_single_worker(
        result.model_dump(),
        case.model_dump(),
        "extract",
        False,
        "extract",
        verified_only=True,
    )
    evaluated = EvaluationResult.model_validate(worker_result)
    by_name = {metric.metric_name: metric for metric in evaluated.metrics}

    # Only the single verified rule is scored.
    assert by_name["extract_field_value_pass_rate"].metadata["total"] == 1
