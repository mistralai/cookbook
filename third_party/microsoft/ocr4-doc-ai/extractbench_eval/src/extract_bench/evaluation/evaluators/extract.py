"""Evaluator for EXTRACT product type using annotation-based evaluation."""

import copy
import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from extract_bench.evaluation.evaluators.base import BaseEvaluator
from extract_bench.evaluation.grounded_confidence_payloads import confidence_field_rules
from extract_bench.evaluation.metrics.extract.array_record_match_metric import (
    ArrayRecordMatchMetric,
)
from extract_bench.evaluation.metrics.extract.association_f1_metric import (
    ExtractAssociationF1Metric,
)
from extract_bench.evaluation.metrics.extract.confidence_scoped.summary import (
    compute_confidence_scoped_metrics,
)
from extract_bench.evaluation.metrics.extract.json_subset_match_metric import (
    JsonSubsetMatchMetric,
)
from extract_bench.evaluation.metrics.extract.list_unwrap import normalize_list_prediction
from extract_bench.evaluation.metrics.extract.rule_based_metric import (
    ExtractRuleBasedMetric,
)
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    compute_unified_evidence_metrics,
)
from extract_bench.evaluation.metrics.field_grounding.evidence_comparator import (
    parse_match_by_keys,
)
from extract_bench.evaluation.metrics.field_grounding.extract_adapter import (
    compute_extract_field_grounding_metrics,
)
from extract_bench.evaluation.metrics.field_grounding.value_compare import (
    compare_attributed_value,
    expected_type_for_field_path,
)
from extract_bench.evaluation.stats import build_operational_stats
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.extract_field_paths import parse_field_path
from extract_bench.test_cases.schema import ExtractFieldTestRule, ExtractTestCase, TestCase

logger = logging.getLogger(__name__)
# Parse and Extract annotations may coexist in one sidecar. Only these legacy
# dictionary rules belong to ExtractRuleBasedMetric; typed extract_field rules
# are evaluated by the dedicated field metrics below.
_EXTRACT_NATIVE_RULE_TYPES = frozenset({"array_length", "array_head", "array_tail"})


def _flatten_pd_claims(d: Any) -> Any:
    """Pull `claims` out of every payment_details[*] up to the doc root."""
    if not isinstance(d, dict):
        return d
    pd = d.get("payment_details")
    if not isinstance(pd, list):
        return d
    has_nested = any(isinstance(p, dict) and isinstance(p.get("claims"), list) for p in pd)
    if not has_nested:
        return d
    out = dict(d)
    flat: list[Any] = []
    new_pd: list[Any] = []
    for p in pd:
        if isinstance(p, dict) and isinstance(p.get("claims"), list):
            flat.extend(p["claims"])
            new_pd.append({k: v for k, v in p.items() if k != "claims"})
        else:
            new_pd.append(p)
    out["payment_details"] = new_pd
    existing = out.get("claims") or []
    out["claims"] = list(existing) + flat
    return out


def _normalize_eob_layouts(extract: Any, gt: Any) -> tuple[Any, Any]:
    """Reconcile flat vs nested claims layouts before scoring.

    EOB GTs sometimes nest claims under payment_details[*].claims while
    extracts emit a flat top-level claims list (or vice versa). Without
    reconciliation the JsonSubsetMatchMetric pairs an empty list against
    a populated one and scores near-zero. We flatten the nested side so
    both layouts look the same. No-op when shapes already agree.

    Also reconciles payment_details cardinality: singleton-schema variants
    declare `payment_details` as a single object while GT is authored as
    an array. The subset-match metric short-circuits to 0.0 on a
    dict-vs-list mismatch, so we wrap the singular side as a 1-element
    list before scoring.
    """
    if isinstance(extract, dict) and isinstance(gt, dict):
        ext_pd = extract.get("payment_details")
        gt_pd = gt.get("payment_details")
        if isinstance(ext_pd, dict) and isinstance(gt_pd, list):
            extract = {**extract, "payment_details": [ext_pd]}
        elif isinstance(gt_pd, dict) and isinstance(ext_pd, list):
            gt = {**gt, "payment_details": [gt_pd]}
    ext_top = isinstance(extract, dict) and isinstance(extract.get("claims"), list) and extract["claims"]
    gt_top = isinstance(gt, dict) and isinstance(gt.get("claims"), list) and gt["claims"]
    ext_nested = (
        isinstance(extract, dict)
        and isinstance(extract.get("payment_details"), list)
        and any(isinstance(p, dict) and isinstance(p.get("claims"), list) for p in extract["payment_details"])
    )
    gt_nested = (
        isinstance(gt, dict)
        and isinstance(gt.get("payment_details"), list)
        and any(isinstance(p, dict) and isinstance(p.get("claims"), list) for p in gt["payment_details"])
    )
    if ext_top and gt_nested and not gt_top:
        gt = _flatten_pd_claims(gt)
    elif gt_top and ext_nested and not ext_top:
        extract = _flatten_pd_claims(extract)
    return extract, gt


class ExtractEvaluator(BaseEvaluator):
    """
    Evaluator for EXTRACT product type.

    Supports two evaluation modes:
    1. Annotation-based: Compare extracted_data with expected_output using JsonSubsetMatchMetric
    2. Rule-based: Execute test rules against extracted_data using ExtractRuleBasedMetric
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        cosine_similarity: bool = False,
        normalize_dates: bool = True,
        weighted: bool = True,
        enable_rule_based: bool = True,
    ):
        """
        Initialize the extract evaluator.

        :param case_sensitive: Whether string comparison should be case-sensitive
        :param cosine_similarity: Use embedding similarity for strings (requires OpenAI API key)
        :param normalize_dates: Normalize date strings before comparison
        :param enable_rule_based: Enable rule-based metric evaluation (default: True)
        """
        self._accuracy_metric = JsonSubsetMatchMetric(
            case_sensitive=case_sensitive,
            cosine_similarity=cosine_similarity,
            normalize_dates=normalize_dates,
            weighted=weighted,
        )
        self._enable_rule_based = enable_rule_based
        self._rule_metric = ExtractRuleBasedMetric()
        self._association_f1_metric = ExtractAssociationF1Metric()
        self._array_record_metric = ArrayRecordMatchMetric(normalize_dates=normalize_dates)

    def can_evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> bool:
        """
        Check if this evaluator can evaluate the given inference result and test case.

        :param inference_result: The inference result to evaluate
        :param test_case: The test case to evaluate against
        :return: True if this evaluator can handle this case
        """
        # Must be EXTRACT product type
        if inference_result.product_type != ProductType.EXTRACT:
            return False

        # Must have ExtractOutput
        if not isinstance(inference_result.output, ExtractOutput):
            return False

        # Must be ExtractTestCase
        if not isinstance(test_case, ExtractTestCase):
            return False

        # Need either expected_output (for annotation-based) or test_rules (for rule-based)
        has_expected_output = test_case.expected_output is not None
        has_test_rules = test_case.test_rules is not None and len(test_case.test_rules) > 0

        return has_expected_output or has_test_rules

    def evaluate(self, inference_result: InferenceResult, test_case: TestCase) -> EvaluationResult:
        """
        Evaluate an EXTRACT inference result against a test case.

        :param inference_result: The inference result to evaluate
        :param test_case: The test case with expected output or test rules
        :return: Evaluation result with accuracy metrics
        :raises ValueError: If neither expected_output nor test_rules are provided
        """
        if not self.can_evaluate(inference_result, test_case):
            raise ValueError("Cannot evaluate: missing expected_output or test_rules, or invalid product type")

        if not isinstance(inference_result.output, ExtractOutput):
            raise ValueError("Inference result output is not ExtractOutput")

        if not isinstance(test_case, ExtractTestCase):
            raise ValueError("Test case must be ExtractTestCase for EXTRACT evaluation")

        raw_extracted_data = inference_result.output.extracted_data
        metrics: list[MetricValue] = []
        diagnostic_metrics: list[MetricValue] = []

        # Normalize per_table_row list projections back into the per-doc shape
        # used by extract_field rules. The adapter is a pure shape transform:
        # state is recorded on existing metric metadata, not as standalone
        # dashboard metrics.
        field_rules_for_unwrap = (
            test_case.get_extract_field_rules() if hasattr(test_case, "get_extract_field_rules") else []
        )
        normalization = normalize_list_prediction(
            raw_extracted_data,
            field_rules_for_unwrap,
            data_schema=test_case.data_schema,
        )
        extracted_data = normalization.extracted_data
        unwrap_skipped = [
            *normalization.skipped_field_paths,
            *normalization.alias_skipped_field_paths,
        ]

        # Annotation-based evaluation.
        #
        # Note: the accuracy metric is computed against the *unwrapped*
        # extracted_data vs the full expected_output. On per_table_row runs
        # this honestly drops accuracy because scalar fields the prediction
        # doesn't emit (e.g. ``client_id``) still appear in expected_output.
        # That drop is a correct signal, not noise — if scalar coverage
        # matters, run a per_doc pipeline instead. See list_unwrap.py.
        # TODO: remove this legacy M0 path after the v0.2 migration drops
        # expected_output from all extract test cases.
        # Dataset-declared identity keys (match_by structural rules) let the
        # JSON subset match pair array rows by identity instead of by index,
        # so a reordered or dropped row doesn't cascade into every later row.
        identity_keys_by_path = _identity_keys_by_path(field_rules_for_unwrap)

        if test_case.expected_output is not None:
            expected_output = test_case.expected_output

            # Reconcile flat vs nested EOB claims layouts so the JSON subset
            # match metric does not score zero when only the nesting differs.
            extracted_data, expected_output = _normalize_eob_layouts(extracted_data, expected_output)

            # Calculate overall accuracy using the metric
            accuracy_metric = self._accuracy_metric.compute(
                expected=expected_output,
                actual=extracted_data,
                identity_keys_by_path=identity_keys_by_path,
                data_schema=test_case.data_schema,
            )
            metrics.append(accuracy_metric)

            metrics.extend(
                self._array_record_metric.compute(
                    expected=expected_output,
                    actual=extracted_data,
                    data_schema=test_case.data_schema,
                )
            )

            # Unified evidence metric: array_record's keyless Hungarian alignment
            # (no match_by cascade) plus OR-acceptable evidence values and
            # page/bbox grounding. The *_value_* metrics equal array_record_*
            # when evidence adds no acceptable value; the *_grounded_* metrics
            # are 0 for pipelines that emit no citations.
            metrics.extend(
                compute_unified_evidence_metrics(
                    expected_output=expected_output,
                    extracted_data=extracted_data,
                    field_rules=test_case.get_extract_field_rules(),
                    field_citations=getattr(inference_result.output, "field_citations", []),
                    data_schema=test_case.data_schema,
                )
            )

            # Calculate field-level accuracy if both are dicts
            if isinstance(expected_output, dict) and isinstance(extracted_data, dict):
                schema_props = (
                    test_case.data_schema.get("properties") if isinstance(test_case.data_schema, dict) else None
                )
                for key in expected_output.keys():
                    expected_value = expected_output.get(key)
                    actual_value = extracted_data.get(key)
                    field_schema = (
                        schema_props.get(key) if isinstance(schema_props, dict) and key in schema_props else None
                    )
                    field_result = self._accuracy_metric.compute(
                        expected=expected_value,
                        actual=actual_value,
                        identity_keys_by_path={
                            path[1:]: keys for path, keys in identity_keys_by_path.items() if path and path[0] == key
                        },
                        data_schema=field_schema,
                    )
                    diagnostic_metrics.append(
                        MetricValue(
                            metric_name=f"field_accuracy_{key}",
                            value=field_result.value,
                            metadata={"field": key, **field_result.metadata},
                        )
                    )

            # Dataset-declared metric groups: accuracy over a named subset of
            # the schema (e.g. a client-critical field list, or one heavy array
            # subtree whose errors the headline accuracy would otherwise
            # dominate or dilute). The expected output is projected onto the
            # group's field paths and scored with the same JSON subset match as
            # the headline accuracy; actual stays whole because subset matching
            # ignores undeclared keys.
            for group_name, group_paths in (test_case.schema_field_metric_groups or {}).items():
                projected_expected = _project_expected_to_paths(expected_output, group_paths, identity_keys_by_path)
                if projected_expected is None:
                    continue
                group_result = self._accuracy_metric.compute(
                    expected=projected_expected,
                    actual=extracted_data,
                    identity_keys_by_path=identity_keys_by_path,
                    data_schema=test_case.data_schema,
                )
                metrics.append(
                    MetricValue(
                        # The prefix is a display marker: the dashboard renders
                        # these as "Field Accuracy: <Group>".
                        metric_name=f"schema_field_accuracy_{group_name}",
                        value=group_result.value,
                        metadata={"metric_group": group_name, "field_paths": list(group_paths)},
                    )
                )

            # Per-payment-details claim_numbers association F1 (multi-PD bench).
            # Emits `association_f1` only when GT carries claim_numbers; emits
            # `association_f1_inapplicable` otherwise so `avg_association_f1`
            # reflects honest F1 over applicable docs only.
            assoc_result = self._association_f1_metric.compute(expected=expected_output, actual=extracted_data)
            metrics.append(assoc_result)
            metrics.append(
                MetricValue(
                    metric_name="association_f1_applicable",
                    value=0.0 if assoc_result.metric_name == "association_f1_inapplicable" else 1.0,
                    metadata={"applicable_metric_emitted": assoc_result.metric_name},
                )
            )

        # Per-rule extract_field metrics (separate name scheme: field_accuracy[path])
        self._emit_extract_field_metrics(
            test_case,
            extracted_data,
            metrics,
            diagnostic_metrics,
            skip_field_paths=unwrap_skipped,
        )
        metrics.extend(
            compute_extract_field_grounding_metrics(
                extracted_data=extracted_data,
                field_rules=test_case.get_extract_field_rules(),
                field_citations=getattr(inference_result.output, "field_citations", []),
                data_schema=test_case.data_schema,
                skip_field_paths=unwrap_skipped,
                list_unwrap_applied=normalization.applied,
                list_unwrap_mode=normalization.mode,
                alias_skipped_field_paths=normalization.alias_skipped_field_paths,
                normalized_top_level_keys=normalization.normalized_top_level_keys,
                list_unwrap_warnings=normalization.warnings,
            )
        )
        metrics.extend(
            compute_confidence_scoped_metrics(
                extracted_data=extracted_data,
                field_rules=confidence_field_rules(test_case),
                field_citations=getattr(inference_result.output, "field_citations", []),
                # eval-side view: row identity re-attached from `_eval_row_identity`.
                # Alignment treats a schema-declared identity_key as authoritative,
                # unlike the gated `match_by:` rule path, so the block has to be
                # visible here even though it never ships to a provider.
                data_schema=test_case.eval_data_schema,
                skip_field_paths=set(unwrap_skipped),
                expected_output=getattr(test_case, "expected_output", None),
            )
        )

        # Rule-based evaluation
        if self._enable_rule_based:
            if not test_case.test_rules:
                logger.debug(
                    f"Skipping rule-based metric: test_rules not provided "
                    f"(test_id: {test_case.test_id}, "
                    f"example_id: {inference_result.request.example_id})"
                )
            else:
                extract_rules = [
                    rule
                    for rule in test_case.test_rules
                    if isinstance(rule, dict) and rule.get("type") in _EXTRACT_NATIVE_RULE_TYPES
                ]
                if not extract_rules:
                    logger.debug(
                        f"Skipping extract rule metric: no Extract-native dictionary rules present "
                        f"(test_id: {test_case.test_id}, example_id: {inference_result.request.example_id})"
                    )
                    return_metric = None
                else:
                    # Execute rules
                    rule_result = self._rule_metric.compute(
                        expected=extract_rules,
                        actual=extracted_data,
                    )
                    metrics.append(rule_result)
                    return_metric = rule_result

                # Add per-type pass rates when we actually executed extract rules
                if return_metric and return_metric.metadata and "rule_results" in return_metric.metadata:
                    rule_results = return_metric.metadata["rule_results"]
                    rule_types: dict[str, list[dict[str, Any]]] = {}
                    for result in rule_results:
                        rule_type = result.get("type", "unknown")
                        if rule_type not in rule_types:
                            rule_types[rule_type] = []
                        rule_types[rule_type].append(result)

                    for rule_type, type_results in rule_types.items():
                        passed = sum(1 for r in type_results if r.get("passed", False))
                        total = len(type_results)
                        pass_rate = passed / total if total > 0 else 0.0
                        metrics.append(
                            MetricValue(
                                metric_name=f"rule_{rule_type}_pass_rate",
                                value=pass_rate,
                                metadata={
                                    "passed": passed,
                                    "total": total,
                                    "rule_type": rule_type,
                                },
                            )
                        )

        stats = build_operational_stats(inference_result)

        return EvaluationResult(
            test_id=test_case.test_id,
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            product_type=inference_result.product_type.value,
            success=True,
            metrics=metrics,
            diagnostic_metrics=diagnostic_metrics,
            error=None,
            job_id=inference_result.raw_output.get("job_id"),
            parse_job_id=inference_result.raw_output.get("parse_job_id"),
            stats=stats,
        )

    def _emit_extract_field_metrics(
        self,
        test_case: ExtractTestCase,
        extracted_data: Any,
        metrics: list[MetricValue],
        diagnostic_metrics: list[MetricValue],
        *,
        skip_field_paths: Iterable[str] = (),
    ) -> None:
        """Emit per-rule and doc-level metrics for `extract_field` rules.

        Rules whose ``field_path`` is in ``skip_field_paths`` are dropped
        entirely — no per-rule metric is emitted and they don't count toward
        ``extract_field_value_pass_rate`` totals. This is used by the
        list-unwrap path on per_table_row predictions to avoid penalizing
        pipelines for scalar fields they structurally cannot emit.
        """
        field_rules = [rule for rule in test_case.get_extract_field_rules() if rule.evidence is None]
        if not field_rules:
            return

        skip_set = set(skip_field_paths)
        eligible_rules = [rule for rule in field_rules if rule.field_path not in skip_set]
        matched_rule_ids = _match_extract_field_rules_index_tolerant(
            eligible_rules,
            extracted_data,
            data_schema=test_case.data_schema,
        )
        total = 0
        passed = 0
        for rule in field_rules:
            if rule.field_path in skip_set:
                continue
            try:
                parse_field_path(rule.field_path)
            except ValueError:
                continue
            match = id(rule) in matched_rule_ids
            diagnostic_metrics.append(
                MetricValue(
                    metric_name=f"field_accuracy[{rule.field_path}]",
                    value=float(match),
                    metadata={
                        "verified": rule.verified,
                        "field_path": rule.field_path,
                    },
                )
            )
            total += 1
            passed += int(match)

        if total > 0:
            metrics.append(
                MetricValue(
                    metric_name="extract_field_value_pass_rate",
                    value=passed / total,
                    metadata={"total": total, "passed": passed},
                )
            )


_PROJECTION_MISSING = object()


def _project_node(
    node: Any,
    paths: list[tuple[str, ...]],
    dict_path: tuple[str, ...],
    identity_keys_by_path: dict[tuple[str, ...], list[str]],
) -> Any:
    """Project ``node`` onto the union of relative ``paths`` (dict-key tuples).

    A path that terminates at this node includes the whole subtree. Arrays
    traversed mid-path are projected element-wise; elements keep the
    dataset-declared identity keys for that array path so ``match_by`` row
    pairing on the projection behaves like the full-document metric (the
    identity leaves then carry a small share of the group's weight). Elements
    where no path matches are dropped. Returns ``_PROJECTION_MISSING`` when
    nothing matched.
    """
    if any(len(p) == 0 for p in paths):
        return copy.deepcopy(node)
    if isinstance(node, list):
        identity_keys = identity_keys_by_path.get(dict_path, [])
        projected_elements = []
        for element in node:
            projected = _project_node(element, paths, dict_path, identity_keys_by_path)
            if projected is _PROJECTION_MISSING:
                continue
            if isinstance(element, dict) and isinstance(projected, dict):
                for key in identity_keys:
                    if key in element and key not in projected:
                        projected[key] = copy.deepcopy(element[key])
            projected_elements.append(projected)
        return projected_elements if projected_elements else _PROJECTION_MISSING
    if isinstance(node, dict):
        tails_by_head: dict[str, list[tuple[str, ...]]] = {}
        for path in paths:
            tails_by_head.setdefault(path[0], []).append(path[1:])
        out: dict[str, Any] = {}
        for head, tails in tails_by_head.items():
            if head not in node:
                continue
            projected = _project_node(node[head], tails, dict_path + (head,), identity_keys_by_path)
            if projected is not _PROJECTION_MISSING:
                out[head] = projected
        return out if out else _PROJECTION_MISSING
    return _PROJECTION_MISSING


def _project_expected_to_paths(
    expected: Any,
    field_paths: list[str],
    identity_keys_by_path: dict[tuple[str, ...], list[str]],
) -> dict[str, Any] | None:
    """Project ``expected`` onto dot-separated ``field_paths``.

    Returns ``None`` when no declared path is present in ``expected`` (the
    group is inapplicable for this document, so no metric should be emitted).
    """
    if not isinstance(expected, dict):
        return None
    parsed_paths = [tuple(part for part in path.split(".") if part) for path in field_paths]
    parsed_paths = [path for path in parsed_paths if path]
    if not parsed_paths:
        return None
    projected = _project_node(expected, parsed_paths, (), identity_keys_by_path)
    return projected if isinstance(projected, dict) else None


def _identity_keys_by_path(field_rules: list[ExtractFieldTestRule]) -> dict[tuple[str, ...], list[str]]:
    """Dataset-declared identity keys per array path, from ``match_by`` rules.

    Keys are dict-key path tuples (``("line_items",)``); array parents whose
    own path contains a list index are skipped — a per-row identity scope is
    ambiguous for the path-based pairing in the JSON subset match.
    """
    identity_keys: dict[tuple[str, ...], list[str]] = {}
    for rule in field_rules:
        structural = rule.structural or ""
        if not structural.startswith("match_by:"):
            continue
        keys = parse_match_by_keys(structural.split(":", 1)[1])
        if not keys:
            continue
        try:
            tokens = parse_field_path(rule.field_path)
        except ValueError:
            continue
        if any(isinstance(token, int) for token in tokens):
            continue
        identity_keys[tuple(str(token) for token in tokens)] = keys
    return identity_keys


def _field_value_match(expected: Any, actual: Any) -> bool:
    """Simple per-rule value match.

    * None ≡ None.
    * Booleans and numbers compare by equality (with bool/number cross-typing allowed).
    * Strings compare case-insensitively with whitespace collapsed.
    * Other mismatched types return False.
    """
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) == bool(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(expected) == float(actual)
    if isinstance(expected, str) and isinstance(actual, str):
        return _normalize_str(expected) == _normalize_str(actual)
    # Cross-type fallback: best-effort string compare.
    return _normalize_str(str(expected)) == _normalize_str(str(actual))


def _extract_field_value_match(
    *,
    field_path: str,
    expected: Any,
    actual: Any,
    data_schema: dict[str, Any] | None,
) -> bool:
    expected_type = expected_type_for_field_path(data_schema, field_path, expected)
    comparison = compare_attributed_value(
        expected,
        actual,
        expected_type=expected_type,
        source_kind="structured_value_no_citation_text",
    )
    return comparison.passed


def _normalize_str(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).casefold()


def _extract_field_pattern(field_path: str) -> tuple[str | None, ...] | None:
    try:
        tokens = parse_field_path(field_path)
    except ValueError:
        return None
    return tuple(None if isinstance(token, int) else token for token in tokens)


def _iter_values_for_extract_field_pattern(source: Any, pattern: Iterable[str | None]) -> list[Any]:
    cursors = [source]
    for token in pattern:
        next_cursors: list[Any] = []
        if token is None:
            for cursor in cursors:
                if isinstance(cursor, list):
                    next_cursors.extend(item for item in cursor if item is not None)
        else:
            for cursor in cursors:
                if isinstance(cursor, dict) and token in cursor:
                    next_cursors.append(cursor[token])
        cursors = next_cursors
        if not cursors:
            return []
    return [cursor for cursor in cursors if not isinstance(cursor, (dict, list))]


def _match_extract_field_rules_index_tolerant(
    field_rules: list[Any],
    extracted_data: Any,
    *,
    data_schema: dict[str, Any] | None = None,
) -> set[int]:
    rules_by_pattern: dict[tuple[str | None, ...], list[Any]] = defaultdict(list)
    for rule in field_rules:
        pattern = _extract_field_pattern(rule.field_path)
        if pattern is not None:
            rules_by_pattern[pattern].append(rule)

    matched_rule_ids: set[int] = set()
    for pattern, rules in rules_by_pattern.items():
        predictions = _iter_values_for_extract_field_pattern(extracted_data, pattern)
        used_predictions: set[int] = set()
        for rule in rules:
            if rule.expected_value is None and not predictions:
                matched_rule_ids.add(id(rule))
                continue
            for pred_index, prediction in enumerate(predictions):
                if pred_index in used_predictions:
                    continue
                if not _extract_field_value_match(
                    field_path=rule.field_path,
                    expected=rule.expected_value,
                    actual=prediction,
                    data_schema=data_schema,
                ):
                    continue
                matched_rule_ids.add(id(rule))
                used_predictions.add(pred_index)
                break
    return matched_rule_ids
