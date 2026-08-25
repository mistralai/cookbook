"""Total-failure runs must score 0.0 rather than vanish from the report.

The partial-failure path derives metric names from the successful results, so a
run where *nothing* succeeded has no names to project zeros onto. Without a
fallback the pipeline silently disappears from the aggregate and the evaluation
prints an empty, green-looking report.
"""

from __future__ import annotations

import json
from pathlib import Path

from extract_bench.evaluation.runner import _FALLBACK_METRIC_NAMES, EvaluationRunner
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue
from extract_bench.test_cases.schema import ExtractTestCase

EXTRACT_FALLBACK = set(_FALLBACK_METRIC_NAMES["extract"])


def _runner_with_output(output_dir: Path) -> EvaluationRunner:
    runner = object.__new__(EvaluationRunner)
    runner.output_dir = output_dir  # type: ignore[attr-defined]
    return runner


def _extract_test_case(test_id: str) -> ExtractTestCase:
    return ExtractTestCase(
        test_id=test_id,
        group="short",
        file_path=Path("/tmp/x.pdf"),
        schema={"type": "object"},
        expected_output={},
    )


class TestTotalInferenceFailure:
    def test_all_documents_errored_scores_zero(self, tmp_path):
        """Every document in _errors.json, nothing successful -> zero-scored rows."""
        runner = _runner_with_output(tmp_path)
        test_cases = {f"short/doc{i}": _extract_test_case(f"short/doc{i}") for i in range(3)}
        errors = [{"example_id": f"short/doc{i}", "error": "401 unauthorized"} for i in range(3)]

        synthesized = runner._synthesize_failure_results(
            inference_errors=errors,
            evaluated_test_ids=set(),
            test_cases_dict=test_cases,
            successful_results=[],  # nothing succeeded
            pipeline_name="my_pipeline",
            product_type="extract",
        )

        assert len(synthesized) == 3
        for result in synthesized:
            assert {m.metric_name for m in result.metrics} == EXTRACT_FALLBACK
            assert all(m.value == 0.0 for m in result.metrics)
            assert result.error is not None and "401 unauthorized" in result.error

    def test_unknown_product_type_still_skips(self, tmp_path):
        """No fallback defined -> preserve the old skip rather than invent names."""
        runner = _runner_with_output(tmp_path)
        test_cases = {"short/doc0": _extract_test_case("short/doc0")}

        synthesized = runner._synthesize_failure_results(
            inference_errors=[{"example_id": "short/doc0", "error": "boom"}],
            evaluated_test_ids=set(),
            test_cases_dict=test_cases,
            successful_results=[],
            pipeline_name="my_pipeline",
            product_type="parse",
        )

        assert synthesized == []

    def test_partial_failure_still_uses_observed_metric_names(self, tmp_path):
        """The fallback must not displace the richer observed-name path."""
        runner = _runner_with_output(tmp_path)
        test_cases = {
            "short/doc0": _extract_test_case("short/doc0"),
            "short/doc1": _extract_test_case("short/doc1"),
        }
        success = EvaluationResult(
            test_id="short/doc0",
            example_id="short/doc0",
            pipeline_name="my_pipeline",
            product_type="extract",
            success=True,
            metrics=[MetricValue(metric_name="a_bespoke_metric", value=1.0)],
        )

        synthesized = runner._synthesize_failure_results(
            inference_errors=[{"example_id": "short/doc1", "error": "boom"}],
            evaluated_test_ids={"short/doc0"},
            test_cases_dict=test_cases,
            successful_results=[success],
            pipeline_name="my_pipeline",
            product_type="extract",
        )

        assert len(synthesized) == 1
        assert {m.metric_name for m in synthesized[0].metrics} == {"a_bespoke_metric"}


class TestTotalMissingPredictions:
    def _pipeline_dir(self, tmp_path, name="my_pipeline"):
        d = tmp_path / name
        (d / "short").mkdir(parents=True)
        (d / "_metadata.json").write_text("{}")
        (d / "short" / "doc0.result.json").write_text("{}")
        return d

    def test_named_pipeline_with_no_usable_output_scores_zero(self, tmp_path):
        """`--pipeline_name X` + X produced nothing loadable -> zeros, not silence."""
        self._pipeline_dir(tmp_path)
        runner = _runner_with_output(tmp_path)
        test_cases = {f"short/doc{i}": _extract_test_case(f"short/doc{i}") for i in range(2)}

        synthesized = runner._synthesize_missing_attempts(
            output_dir=tmp_path,
            evaluation_results=[],
            test_cases_dict=test_cases,
            pipeline_name="my_pipeline",
            product_type="extract",
        )

        assert len(synthesized) == 2
        for result in synthesized:
            assert {m.metric_name for m in result.metrics} == EXTRACT_FALLBACK
            assert all(m.value == 0.0 for m in result.metrics)

    def test_unnamed_run_does_not_invent_zeros_for_other_products(self, tmp_path):
        """Without an explicit pipeline name a parse dir must not be zero-filled."""
        self._pipeline_dir(tmp_path, "some_parse_pipeline")
        runner = _runner_with_output(tmp_path)
        test_cases = {"short/doc0": _extract_test_case("short/doc0")}

        synthesized = runner._synthesize_missing_attempts(
            output_dir=tmp_path,
            evaluation_results=[],
            test_cases_dict=test_cases,
            pipeline_name=None,
            product_type="extract",
        )

        assert synthesized == []


def test_fallback_metrics_are_the_always_emitted_extract_family():
    """Guard the fallback against drift into conditional (page/grounded) metrics."""
    assert EXTRACT_FALLBACK == {
        "extract_unified_value_precision",
        "extract_unified_value_recall",
        "extract_unified_value_f1",
    }
    assert not any("grounded" in name or "page" in name for name in EXTRACT_FALLBACK)


def test_errors_json_round_trips(tmp_path):
    """Sanity: the runner reads the on-disk _errors.json shape the runner writes."""
    (tmp_path / "my_pipeline").mkdir()
    (tmp_path / "my_pipeline" / "_errors.json").write_text(
        json.dumps([{"example_id": "short/doc0", "error": "boom", "error_type": "Transient"}])
    )
    runner = _runner_with_output(tmp_path)
    errors = runner._load_inference_errors(tmp_path, pipeline_name="my_pipeline")
    assert errors[0]["error_type"] == "Transient"
