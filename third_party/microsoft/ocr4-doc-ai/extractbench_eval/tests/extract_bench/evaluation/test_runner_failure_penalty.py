"""Tests for the inference-failure penalty path in EvaluationRunner.

When a pipeline errors on a test case (no .result.json produced; entry in
_errors.json), the runner should synthesize a zero-scored EvaluationResult so
the failure pulls the aggregate down rather than being silently dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

from extract_bench.evaluation.runner import EvaluationRunner
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue
from extract_bench.test_cases.schema import ExtractTestCase


def _runner_with_output(output_dir: Path) -> EvaluationRunner:
    runner = object.__new__(EvaluationRunner)
    runner.output_dir = output_dir  # type: ignore[attr-defined]
    return runner


def _success_result(
    test_id: str,
    example_id: str,
    accuracy: float = 1.0,
    pipeline_name: str = "my-pipeline",
) -> EvaluationResult:
    return EvaluationResult(
        test_id=test_id,
        example_id=example_id,
        pipeline_name=pipeline_name,
        product_type="extract",
        success=True,
        metrics=[MetricValue(metric_name="accuracy", value=accuracy)],
        diagnostic_metrics=[MetricValue(metric_name="field_accuracy_x", value=accuracy)],
    )


def _extract_test_case(test_id: str, group: str = "g") -> ExtractTestCase:
    return ExtractTestCase(
        test_id=test_id,
        group=group,
        file_path=Path("/tmp/x.pdf"),
        schema={"type": "object"},
        expected_output={},
    )


def _write_errors(output_dir: Path, entries: list[dict]) -> None:
    (output_dir / "_errors.json").write_text(json.dumps(entries))


class TestLoadInferenceErrors:
    def test_loads_errors_from_pipeline_dir(self, tmp_path):
        runner = _runner_with_output(tmp_path)
        _write_errors(tmp_path, [{"example_id": "doc-a", "error": "timeout"}])
        errors = runner._load_inference_errors(tmp_path, pipeline_name=None)
        assert len(errors) == 1
        assert errors[0]["example_id"] == "doc-a"

    def test_returns_empty_when_no_errors_file(self, tmp_path):
        runner = _runner_with_output(tmp_path)
        assert runner._load_inference_errors(tmp_path, pipeline_name=None) == []

    def test_pipeline_filter_excludes_other_pipelines(self, tmp_path):
        runner = _runner_with_output(tmp_path)
        (tmp_path / "pipe_a").mkdir()
        (tmp_path / "pipe_b").mkdir()
        _write_errors(tmp_path / "pipe_a", [{"example_id": "doc-1", "error": "x"}])
        _write_errors(tmp_path / "pipe_b", [{"example_id": "doc-2", "error": "y"}])
        errors = runner._load_inference_errors(tmp_path, pipeline_name="pipe_a")
        assert [e["example_id"] for e in errors] == ["doc-1"]


class TestSynthesizeFailureResults:
    def test_synthesizes_zero_row_per_unmatched_error(self, tmp_path):
        runner = _runner_with_output(tmp_path)
        test_cases = {
            "g/doc-a": _extract_test_case("g/doc-a"),
            "g/doc-b": _extract_test_case("g/doc-b"),
        }
        successful = [_success_result("g/doc-a", "doc-a", 0.9)]
        errors = [{"example_id": "doc-b", "error": "OOM", "error_type": "PipelineError"}]

        synthesized = runner._synthesize_failure_results(
            inference_errors=errors,
            evaluated_test_ids={"g/doc-a"},
            test_cases_dict=test_cases,
            successful_results=successful,
            pipeline_name="my-pipeline",
            product_type="extract",
        )

        assert len(synthesized) == 1
        row = synthesized[0]
        assert row.test_id == "g/doc-b"
        assert row.success is True
        # All metric values are 0; metric names match successful results.
        assert {m.metric_name for m in row.metrics} == {"accuracy"}
        assert all(m.value == 0.0 for m in row.metrics)
        assert all(m.metadata.get("inference_failed") for m in row.metrics)
        # Diagnostic metric names are also projected.
        assert {m.metric_name for m in row.diagnostic_metrics} == {"field_accuracy_x"}
        assert "Inference failed: OOM" in (row.error or "")

    def test_skips_when_test_case_already_evaluated(self, tmp_path):
        runner = _runner_with_output(tmp_path)
        test_cases = {"g/doc-a": _extract_test_case("g/doc-a")}
        successful = [_success_result("g/doc-a", "doc-a")]
        errors = [{"example_id": "doc-a", "error": "transient"}]

        synthesized = runner._synthesize_failure_results(
            inference_errors=errors,
            evaluated_test_ids={"g/doc-a"},
            test_cases_dict=test_cases,
            successful_results=successful,
            pipeline_name="my-pipeline",
            product_type="extract",
        )
        assert synthesized == []

    def test_zero_fills_from_fallback_when_nothing_succeeded(self, tmp_path):
        """Total failure: no successful result to template metric names from.

        Upstream skipped here, which made a pipeline that failed on every
        document vanish from the aggregate instead of scoring 0.0. We diverge
        deliberately and project the product's headline metrics as zeros; see
        ``test_runner_total_failure.py`` for the full behaviour.
        """
        runner = _runner_with_output(tmp_path)
        test_cases = {"g/doc-a": _extract_test_case("g/doc-a")}
        errors = [{"example_id": "doc-a", "error": "all_failed"}]

        synthesized = runner._synthesize_failure_results(
            inference_errors=errors,
            evaluated_test_ids=set(),
            test_cases_dict=test_cases,
            successful_results=[],
            pipeline_name="my-pipeline",
            product_type="extract",
        )
        assert len(synthesized) == 1
        assert {m.metric_name for m in synthesized[0].metrics} == {
            "extract_unified_value_precision",
            "extract_unified_value_recall",
            "extract_unified_value_f1",
        }
        assert all(m.value == 0.0 for m in synthesized[0].metrics)

    def test_skips_when_example_id_does_not_match_any_test_case(self, tmp_path):
        runner = _runner_with_output(tmp_path)
        test_cases = {"g/doc-a": _extract_test_case("g/doc-a")}
        successful = [_success_result("g/doc-a", "doc-a")]
        errors = [{"example_id": "doc-zzz-not-in-cases", "error": "x"}]
        synthesized = runner._synthesize_failure_results(
            inference_errors=errors,
            evaluated_test_ids={"g/doc-a"},
            test_cases_dict=test_cases,
            successful_results=successful,
            pipeline_name="my-pipeline",
            product_type="extract",
        )
        assert synthesized == []


class TestAggregationIncludesFailures:
    """End-to-end check: zero-rows produced by synthesis pull the macro
    average down, matching the user-facing 'punish' behavior."""

    def test_avg_metric_drops_when_failure_row_added(self, tmp_path):
        runner = _runner_with_output(tmp_path)
        test_cases = {
            "g/doc-a": _extract_test_case("g/doc-a"),
            "g/doc-b": _extract_test_case("g/doc-b"),
        }
        successful = [_success_result("g/doc-a", "doc-a", 1.0)]
        errors = [{"example_id": "doc-b", "error": "boom"}]
        synthesized = runner._synthesize_failure_results(
            inference_errors=errors,
            evaluated_test_ids={"g/doc-a"},
            test_cases_dict=test_cases,
            successful_results=successful,
            pipeline_name="my-pipeline",
            product_type="extract",
        )
        all_results = successful + synthesized

        # Without the failure row, avg_accuracy = 1.0 (single 1.0).
        # With the failure row, avg_accuracy = 0.5 ((1.0 + 0.0) / 2).
        agg_without = runner._aggregate_metrics(successful)
        agg_with = runner._aggregate_metrics(all_results)
        assert agg_without["avg_accuracy"] == 1.0
        assert agg_with["avg_accuracy"] == 0.5


class TestSynthesizeMissingAttempts:
    def test_synthesizes_zero_row_when_pipeline_skips_test_case(self, tmp_path):
        runner = _runner_with_output(tmp_path)
        pipeline_dir = tmp_path / "my-pipeline"
        pipeline_dir.mkdir()
        (pipeline_dir / "_metadata.json").write_text("{}")
        test_cases = {
            "g/doc-a": _extract_test_case("g/doc-a"),
            "g/doc-b": _extract_test_case("g/doc-b"),
        }
        successful = [_success_result("g/doc-a", "doc-a", 1.0)]

        synthesized = runner._synthesize_missing_attempts(
            output_dir=tmp_path,
            evaluation_results=successful,
            test_cases_dict=test_cases,
            pipeline_name=None,
            product_type="extract",
        )

        assert len(synthesized) == 1
        row = synthesized[0]
        assert row.test_id == "g/doc-b"
        assert row.pipeline_name == "my-pipeline"
        assert row.success is True
        assert row.error == "No prediction emitted"
        assert {m.metric_name for m in row.metrics} == {"accuracy"}
        assert all(m.value == 0.0 for m in row.metrics)
        assert all(m.metadata.get("missing_prediction") for m in row.metrics)
        assert {m.metric_name for m in row.diagnostic_metrics} == {"field_accuracy_x"}

        agg = runner._aggregate_metrics(successful + synthesized)
        assert agg["avg_accuracy"] == 0.5

    def test_zero_fills_each_pipeline_test_case_pair_independently(self, tmp_path):
        runner = _runner_with_output(tmp_path)
        for pipeline_name in ("pipe-a", "pipe-b"):
            pipeline_dir = tmp_path / pipeline_name
            pipeline_dir.mkdir()
            (pipeline_dir / "_metadata.json").write_text("{}")
        test_cases = {
            "g/doc-a": _extract_test_case("g/doc-a"),
            "g/doc-b": _extract_test_case("g/doc-b"),
        }
        successful = [
            _success_result("g/doc-a", "doc-a", pipeline_name="pipe-a"),
            _success_result("g/doc-b", "doc-b", pipeline_name="pipe-b"),
        ]

        synthesized = runner._synthesize_missing_attempts(
            output_dir=tmp_path,
            evaluation_results=successful,
            test_cases_dict=test_cases,
            pipeline_name=None,
            product_type="extract",
        )

        assert {(row.pipeline_name, row.test_id) for row in synthesized} == {
            ("pipe-a", "g/doc-b"),
            ("pipe-b", "g/doc-a"),
        }
        assert all(m.value == 0.0 for row in synthesized for m in row.metrics)

    def test_skips_pipeline_with_no_metric_template(self, tmp_path):
        runner = _runner_with_output(tmp_path)
        pipeline_dir = tmp_path / "never-ran"
        pipeline_dir.mkdir()
        (pipeline_dir / "_metadata.json").write_text("{}")
        test_cases = {"g/doc-a": _extract_test_case("g/doc-a")}

        synthesized = runner._synthesize_missing_attempts(
            output_dir=tmp_path,
            evaluation_results=[],
            test_cases_dict=test_cases,
            pipeline_name=None,
            product_type="extract",
        )

        assert synthesized == []


def _extract_test_case_with_rules(test_id: str, n_eligible: int, group: str = "g") -> ExtractTestCase:
    """Build an ExtractTestCase whose ``get_extract_field_rules()`` returns
    ``n_eligible`` rules all matching the failure-denominator filter
    (``evidence_required and verified and not _has_stray_tag``)."""
    from extract_bench.test_cases.schema import ExtractFieldTestRule, FieldEvidence

    rules = [
        ExtractFieldTestRule(
            field_path=f"f{i}",
            evidence=[FieldEvidence(page=1, value="v")],
            evidence_required=True,
            verified=True,
        )
        for i in range(n_eligible)
    ]
    return ExtractTestCase(
        test_id=test_id,
        group=group,
        file_path=Path("/tmp/x.pdf"),
        schema={"type": "object"},
        expected_output={},
        test_rules=rules,
    )


class TestFailurePooledDenominator:
    """``_failure_pooled_denominator`` returns the eligible-rule count for
    Extract test cases and zero otherwise."""

    def test_extract_test_case_returns_eligible_rule_count(self, tmp_path):
        runner = _runner_with_output(tmp_path)
        tc = _extract_test_case_with_rules("g/doc-a", n_eligible=7)
        assert runner._failure_pooled_denominator(tc) == 7

    def test_extract_test_case_without_rules_returns_zero(self, tmp_path):
        runner = _runner_with_output(tmp_path)
        tc = _extract_test_case("g/doc-a")  # no test_rules
        assert runner._failure_pooled_denominator(tc) == 0


class TestPooledMicroInjection:
    """Inference-failure synthesis projects ``tp=0, fp=N, fn=0`` only for the
    five Extract pass-rate / coverage metrics whose successful-path
    denominator is the per-doc eligible rule count. Other metrics
    (diagnostic, ``page_covered_pass_rate``, non-pooled) keep the bare
    failure marker.
    """

    def _synthesize_one(self, tmp_path, metric_names: set[str], n_eligible: int) -> EvaluationResult:
        runner = _runner_with_output(tmp_path)
        test_cases = {
            "g/doc-a": _extract_test_case_with_rules("g/doc-a", n_eligible=2),
            "g/doc-b": _extract_test_case_with_rules("g/doc-b", n_eligible=n_eligible),
        }
        successful = [
            EvaluationResult(
                test_id="g/doc-a",
                example_id="doc-a",
                pipeline_name="p",
                product_type="extract",
                success=True,
                metrics=[MetricValue(metric_name=n, value=1.0) for n in metric_names],
                diagnostic_metrics=[],
            )
        ]
        errors = [{"example_id": "doc-b", "error": "x"}]
        synthesized = runner._synthesize_failure_results(
            inference_errors=errors,
            evaluated_test_ids={"g/doc-a"},
            test_cases_dict=test_cases,
            successful_results=successful,
            pipeline_name="p",
            product_type="extract",
        )
        assert len(synthesized) == 1
        return synthesized[0]

    def test_pooled_metric_gets_tp_fp_fn_injected(self, tmp_path):
        row = self._synthesize_one(tmp_path, {"extract_evidence_value_pass_rate"}, n_eligible=9)
        m = next(m for m in row.metrics if m.metric_name == "extract_evidence_value_pass_rate")
        assert m.value == 0.0
        assert m.metadata["tp"] == 0
        assert m.metadata["fp"] == 9
        assert m.metadata["fn"] == 0
        assert m.metadata["total"] == 9
        assert m.metadata["denominator"] == "inference_failed_synthesised"

    def test_diagnostic_pass_rate_metric_does_not_inject_counts(self, tmp_path):
        # ``extract_evidence_bbox_IOU_pass_rate`` is diagnostic on the success
        # path (tp=fp=fn=0). Injecting fp=N on failure would surface a new
        # micro_* aggregate that doesn't exist today — keep the bare marker.
        row = self._synthesize_one(tmp_path, {"extract_evidence_bbox_IOU_pass_rate"}, n_eligible=9)
        m = next(m for m in row.metrics if m.metric_name == "extract_evidence_bbox_IOU_pass_rate")
        assert m.metadata.get("inference_failed") is True
        assert "tp" not in m.metadata
        assert "fp" not in m.metadata
        assert "fn" not in m.metadata

    def test_page_covered_pass_rate_does_not_inject_counts(self, tmp_path):
        # Denominator for this metric is ``page_qualified`` on the success
        # path, not the full rule count. Injecting fp=N would mix
        # incompatible denominators when pooling across success + failure.
        row = self._synthesize_one(tmp_path, {"extract_evidence_page_covered_pass_rate"}, n_eligible=9)
        m = next(m for m in row.metrics if m.metric_name == "extract_evidence_page_covered_pass_rate")
        assert "tp" not in m.metadata
        assert "fp" not in m.metadata

    def test_pooled_denom_zero_does_not_inject_even_for_whitelisted_metric(self, tmp_path):
        # Test case has no field rules -> denominator is 0 -> no injection
        # (would otherwise produce a degenerate 0/0 pool).
        runner = _runner_with_output(tmp_path)
        test_cases = {
            "g/doc-a": _extract_test_case("g/doc-a"),
            "g/doc-b": _extract_test_case("g/doc-b"),
        }
        successful = [
            EvaluationResult(
                test_id="g/doc-a",
                example_id="doc-a",
                pipeline_name="p",
                product_type="extract",
                success=True,
                metrics=[MetricValue(metric_name="extract_evidence_value_pass_rate", value=1.0)],
            )
        ]
        synthesized = runner._synthesize_failure_results(
            inference_errors=[{"example_id": "doc-b", "error": "x"}],
            evaluated_test_ids={"g/doc-a"},
            test_cases_dict=test_cases,
            successful_results=successful,
            pipeline_name="p",
            product_type="extract",
        )
        assert len(synthesized) == 1
        m = synthesized[0].metrics[0]
        assert "tp" not in m.metadata


class TestPooledMicroAggregationIntegration:
    """End-to-end: a 1-success-1-failure mix pools to ``micro = 0 / (N1 + N2)``
    for whitelisted metrics; macro averages to 0.5 either way."""

    def test_micro_pools_failure_as_zero_over_n(self, tmp_path):
        runner = _runner_with_output(tmp_path)
        test_cases = {
            "g/doc-a": _extract_test_case_with_rules("g/doc-a", n_eligible=4),
            "g/doc-b": _extract_test_case_with_rules("g/doc-b", n_eligible=6),
        }
        # Success path: 3/4 rules passed on doc-a → tp=3, fp=1, fn=0, total=4
        successful = [
            EvaluationResult(
                test_id="g/doc-a",
                example_id="doc-a",
                pipeline_name="p",
                product_type="extract",
                success=True,
                metrics=[
                    MetricValue(
                        metric_name="extract_evidence_value_pass_rate",
                        value=0.75,
                        metadata={"tp": 3, "fp": 1, "fn": 0, "total": 4},
                    )
                ],
            )
        ]
        synthesized = runner._synthesize_failure_results(
            inference_errors=[{"example_id": "doc-b", "error": "x"}],
            evaluated_test_ids={"g/doc-a"},
            test_cases_dict=test_cases,
            successful_results=successful,
            pipeline_name="p",
            product_type="extract",
        )
        all_results = successful + synthesized

        agg = runner._aggregate_metrics(all_results)
        # macro: (0.75 + 0.0) / 2 = 0.375
        assert agg["avg_extract_evidence_value_pass_rate"] == 0.375
        # micro: tp=3, fp=1+6, fn=0 -> 3/10 = 0.30
        assert agg["micro_extract_evidence_value_pass_rate"] == 0.3
