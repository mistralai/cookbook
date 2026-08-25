"""Rule-based metric for executing parse test rules."""

import signal
import time
from collections import Counter
from typing import Any

from extract_bench.evaluation.metrics.base import Metric
from extract_bench.evaluation.metrics.parse.test_rules import (
    MissingSpecificWordRule,
    RotateCheckRule,
    WordBagRule,
    create_test_rule,
)
from extract_bench.evaluation.metrics.parse.utils import normalize_text
from extract_bench.schemas.evaluation import MetricValue
from extract_bench.schemas.parse_output import ParseOutput
from extract_bench.test_cases.parse_rule_schemas import (
    ParseRuleBase,
    ParseRuleInput,
    get_rule_id,
    get_rule_layout_bindings,
    get_rule_layout_id,
    get_rule_layout_ids,
    get_rule_page,
    get_rule_type,
)

# Per-rule timeout in seconds. Rules that exceed this are marked as failed.
RULE_TIMEOUT_SECONDS = 120


class _RuleTimeoutError(Exception):
    """Raised when a single rule exceeds its time budget."""


def _alarm_handler(signum: int, frame: Any) -> None:
    raise _RuleTimeoutError()


class RuleBasedMetric(Metric):
    """Metric for executing test rules against markdown content."""

    @property
    def name(self) -> str:
        """Return the name of this metric."""
        return "rule_pass_rate"

    def compute(
        self,
        expected: list[ParseRuleInput] | None,
        actual: str,
        page: int | None = None,
        **kwargs: Any,
    ) -> MetricValue:
        """
        Execute test rules against markdown content.

        :param expected: List of test rule definitions (from test_rules)
        :param actual: Actual markdown content to test
        :param page: Optional page number (1-indexed) to filter rules
        :param kwargs: Additional parameters (e.g. raw_output for RotateCheckRule)
        :return: MetricValue with pass rate and per-rule results
        """
        if not expected:
            return MetricValue(
                metric_name=self.name,
                value=1.0,  # No rules means pass
                metadata={"note": "No test rules provided"},
            )

        # Filter rules by page if page is specified
        rules_to_run = expected
        if page is not None:
            # Filter rules that match this page or have no page specified
            rules_to_run = [rule for rule in expected if get_rule_page(rule) is None or get_rule_page(rule) == page]

        if not rules_to_run:
            return MetricValue(
                metric_name=self.name,
                value=1.0,  # No rules for this page means pass
                metadata={"note": f"No test rules for page {page}"},
            )

        if not actual:
            # Blank output fails every rule. Emit full per-rule metadata so the
            # judge metric and per-type pass rates include this doc (otherwise
            # blank-output docs silently drop out of the aggregate averages,
            # inflating scores for tools that fail to parse hard documents).
            rule_results = [
                {
                    "type": get_rule_type(rule_data),
                    "id": rule_data.id if isinstance(rule_data, ParseRuleBase) else get_rule_id(rule_data),
                    "page": get_rule_page(rule_data),
                    "tags": rule_data.tags if isinstance(rule_data, ParseRuleBase) else [],
                    "layout_id": get_rule_layout_id(rule_data),
                    "layout_ids": get_rule_layout_ids(rule_data),
                    "layout_bindings": get_rule_layout_bindings(rule_data),
                    "passed": False,
                    "score": 0.0,
                    "explanation": "No markdown content provided",
                    # rotate_check entries need expected_angle so the per-angle
                    # breakdown includes blank-output docs too.
                    **(
                        {"expected_angle": rule_data.get("value")} if get_rule_type(rule_data) == "rotate_check" else {}
                    ),
                }
                for rule_data in rules_to_run
            ]
            return MetricValue(
                metric_name=self.name,
                value=0.0,
                metadata={
                    "note": "No markdown content provided",
                    "passed": 0,
                    "total": len(rules_to_run),
                    "ambiguous_anchor_failures": 0,
                    "rule_results": rule_results,
                },
            )

        # Pre-normalize content ONCE for all rules (major performance optimization)
        t_normalize_start = time.monotonic()
        normalized_actual = normalize_text(actual)
        t_normalize_elapsed = time.monotonic() - t_normalize_start
        print(f"  Pre-normalized content: {len(actual)} -> {len(normalized_actual)} chars ({t_normalize_elapsed:.1f}s)")

        # Execute each rule
        passed = 0
        ambiguous_anchor_failures = 0
        total = len(rules_to_run)
        rule_results = []
        missing_specific_word_cache: tuple[Counter[str], str] | None = None

        # Timing accumulators
        t_rules_start = time.monotonic()
        slow_rules: list[tuple[int, str, float]] = []  # (index, type, seconds)
        timed_out_rules: list[tuple[int, str]] = []  # (index, type)

        # Use signal.alarm for per-rule timeout (Unix only, main thread of worker process)
        use_alarm = hasattr(signal, "SIGALRM")
        prev_handler = None
        if use_alarm:
            prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)

        # Log every ~100 rules, but at least first and last
        log_interval = max(total // 10, 100) if total > 10 else total
        try:
            for i, rule_data in enumerate(rules_to_run):
                if i == 0 or (i + 1) % log_interval == 0:
                    elapsed = time.monotonic() - t_rules_start
                    print(f"  Processing rule {i + 1}/{total} ({elapsed:.1f}s elapsed)", flush=True)
                rule_id = rule_data.id if isinstance(rule_data, ParseRuleBase) else get_rule_id(rule_data)
                rule_tags = rule_data.tags if isinstance(rule_data, ParseRuleBase) else []
                rule_layout_id = get_rule_layout_id(rule_data)
                rule_layout_ids = get_rule_layout_ids(rule_data)
                rule_layout_bindings = get_rule_layout_bindings(rule_data)
                try:
                    t_rule_start = time.monotonic()
                    rule_type_name = get_rule_type(rule_data) or "unknown"

                    # Arm the alarm before rule creation + execution
                    if use_alarm:
                        signal.alarm(RULE_TIMEOUT_SECONDS)

                    rule = create_test_rule(rule_data)
                    parse_output = kwargs.get("parse_output")
                    if isinstance(parse_output, ParseOutput) and hasattr(rule, "parse_output"):
                        rule.parse_output = parse_output

                    if isinstance(rule, RotateCheckRule):
                        raw_output = kwargs.get("raw_output")
                        if isinstance(raw_output, dict):
                            rule.raw_output = raw_output
                    if isinstance(rule, MissingSpecificWordRule):
                        if missing_specific_word_cache is None:
                            missing_specific_word_cache = (
                                WordBagRule._extract_normalized_words_static(
                                    actual,
                                    include_table_cells=True,
                                ),
                                MissingSpecificWordRule.strip_apostrophes(normalized_actual),
                            )
                        rule.actual_words = missing_specific_word_cache[0]
                        rule.apostrophe_stripped_content = missing_specific_word_cache[1]
                    # Pass pre-normalized content to avoid redundant normalization
                    result = rule.run(actual, normalized_content=normalized_actual)

                    # Disarm the alarm
                    if use_alarm:
                        signal.alarm(0)

                    t_rule_elapsed = time.monotonic() - t_rule_start
                    if t_rule_elapsed > 2.0:
                        slow_rules.append((i, rule_type_name, t_rule_elapsed))
                    rule_passed, explanation = result[0], result[1]
                    score = result[2] if len(result) == 3 else (1.0 if rule_passed else 0.0)
                    rule_result_entry: dict[str, Any] = {
                        "type": get_rule_type(rule_data),
                        "id": rule_id,
                        "page": get_rule_page(rule_data),
                        "tags": rule_tags,
                        "layout_id": rule_layout_id,
                        "layout_ids": rule_layout_ids,
                        "layout_bindings": rule_layout_bindings,
                        "passed": rule_passed,
                        "score": score,
                        "explanation": explanation,
                    }
                    if isinstance(rule, RotateCheckRule):
                        rule_result_entry["expected_angle"] = rule.expected_angle
                    rule_results.append(rule_result_entry)
                    if rule_passed:
                        passed += 1
                    elif explanation.startswith("[AMBIGUOUS ANCHORS]"):
                        ambiguous_anchor_failures += 1
                except _RuleTimeoutError:
                    t_rule_elapsed = time.monotonic() - t_rule_start
                    timed_out_rules.append((i, rule_type_name))
                    print(
                        f"    TIMEOUT rule #{i}: type={rule_type_name}"
                        f" exceeded {RULE_TIMEOUT_SECONDS}s ({t_rule_elapsed:.1f}s)",
                        flush=True,
                    )
                    rule_results.append(
                        {
                            "type": get_rule_type(rule_data),
                            "id": rule_id,
                            "page": get_rule_page(rule_data),
                            "tags": rule_tags,
                            "layout_id": rule_layout_id,
                            "layout_ids": rule_layout_ids,
                            "layout_bindings": rule_layout_bindings,
                            "passed": False,
                            "score": 0.0,
                            "explanation": f"Rule timed out after {RULE_TIMEOUT_SECONDS}s",
                        }
                    )
                except Exception as e:
                    # Disarm the alarm on error
                    if use_alarm:
                        signal.alarm(0)
                    # If rule execution fails, count as failed
                    rule_results.append(
                        {
                            "type": get_rule_type(rule_data),
                            "id": rule_id,
                            "page": get_rule_page(rule_data),
                            "tags": rule_tags,
                            "layout_id": rule_layout_id,
                            "layout_ids": rule_layout_ids,
                            "layout_bindings": rule_layout_bindings,
                            "passed": False,
                            "score": 0.0,
                            "explanation": f"Error executing rule: {e}",
                        }
                    )
        finally:
            # Always disarm alarm and restore previous handler
            if use_alarm:
                signal.alarm(0)
                if prev_handler is not None:
                    signal.signal(signal.SIGALRM, prev_handler)

        total_score = 0.0
        for r in rule_results:
            total_score += float(r["score"])
        pass_rate = total_score / total if total > 0 else 0.0
        t_rules_total = time.monotonic() - t_rules_start
        print(
            f"  Rules: done, {passed}/{total} passed ({pass_rate:.1%}) in {t_rules_total:.1f}s",
            flush=True,
        )
        if timed_out_rules:
            for idx, rtype in timed_out_rules:
                print(f"    TIMED OUT rule #{idx}: type={rtype}", flush=True)
        if slow_rules:
            for idx, rtype, secs in slow_rules:
                print(f"    slow rule #{idx}: type={rtype} took {secs:.1f}s", flush=True)

        return MetricValue(
            metric_name=self.name,
            value=pass_rate,
            metadata={
                "passed": passed,
                "total": total,
                "ambiguous_anchor_failures": ambiguous_anchor_failures,
                "rule_results": rule_results,
            },
        )
