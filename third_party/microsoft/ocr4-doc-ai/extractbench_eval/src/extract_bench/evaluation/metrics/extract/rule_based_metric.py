"""Rule-based metric for executing extract test rules."""

from typing import Any

from extract_bench.evaluation.metrics.base import Metric
from extract_bench.evaluation.metrics.extract.test_rules import create_test_rule
from extract_bench.schemas.evaluation import MetricValue


class ExtractRuleBasedMetric(Metric):
    """Metric for executing test rules against extracted JSON data."""

    @property
    def name(self) -> str:
        """Return the name of this metric."""
        return "rule_pass_rate"

    def compute(
        self,
        expected: list[dict[str, Any]] | None,
        actual: dict[str, Any],
        **kwargs: Any,
    ) -> MetricValue:
        """
        Execute test rules against extracted JSON data.

        :param expected: List of test rule definitions (from test_rules)
        :param actual: Actual extracted JSON data to test
        :param kwargs: Additional parameters (not used)
        :return: MetricValue with pass rate and per-rule results
        """
        if not expected:
            return MetricValue(
                metric_name=self.name,
                value=1.0,  # No rules means pass
                metadata={"note": "No test rules provided"},
            )

        if not actual:
            return MetricValue(
                metric_name=self.name,
                value=0.0,
                metadata={"note": "No extracted data provided"},
            )

        # Execute each rule
        passed = 0
        total = len(expected)
        rule_results = []

        for rule_data in expected:
            try:
                rule = create_test_rule(rule_data)
                rule_passed, explanation = rule.run(actual)
                rule_results.append(
                    {
                        "type": rule_data.get("type"),
                        "id": rule_data.get("id"),
                        "name": rule_data.get("name"),
                        "path": rule_data.get("path"),
                        "passed": rule_passed,
                        "explanation": explanation,
                    }
                )
                if rule_passed:
                    passed += 1
            except Exception as e:
                # If rule execution fails, count as failed
                rule_results.append(
                    {
                        "type": rule_data.get("type"),
                        "id": rule_data.get("id"),
                        "name": rule_data.get("name"),
                        "path": rule_data.get("path"),
                        "passed": False,
                        "explanation": f"Error executing rule: {e}",
                    }
                )

        pass_rate = passed / total if total > 0 else 0.0

        return MetricValue(
            metric_name=self.name,
            value=pass_rate,
            metadata={
                "passed": passed,
                "total": total,
                "rule_results": rule_results,
            },
        )
