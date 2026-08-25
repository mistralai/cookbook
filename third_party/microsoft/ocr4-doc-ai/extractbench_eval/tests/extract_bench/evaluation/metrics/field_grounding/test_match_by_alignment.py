"""Tests for multi-key ``match_by`` semantics and identity-key row alignment.

Two layers:

* ``compare_evidence_array`` — composite (comma-joined) identity keys, per-key
  comparators, and the unverified-cell (null) skip in full-row compares.
* ``_compute_v02_evidence_metrics`` — per-index leaf rules under a ``match_by``
  parent are graded against the predicted row with the matching identity, so
  reordered or dropped rows no longer cascade index-shift false negatives.
"""

from __future__ import annotations

from extract_bench.evaluation.metrics.field_grounding.evidence_comparator import (
    compare_evidence_array,
    parse_match_by_keys,
)
from extract_bench.evaluation.metrics.field_grounding.extract_adapter import (
    compute_extract_field_grounding_metrics,
)
from extract_bench.test_cases.schema import ExtractFieldTestRule, FieldEvidence


def _named(metrics):
    return {m.metric_name: m for m in metrics}


class TestParseMatchByKeys:
    def test_single_key(self):
        assert parse_match_by_keys("sku") == ["sku"]

    def test_multi_key_with_whitespace(self):
        assert parse_match_by_keys("cusip, title_of_class ,put_call") == [
            "cusip",
            "title_of_class",
            "put_call",
        ]

    def test_empty_spec(self):
        assert parse_match_by_keys(" , ") == []


class TestMultiKeyMatchBy:
    EXPECTED = [
        {"item_no": "0001", "sub_no": "A", "amount": 100},
        {"item_no": "0001", "sub_no": "B", "amount": 200},
    ]
    COMPARATOR = {"item_no": "case_insensitive", "sub_no": "case_insensitive", "amount": "number"}

    def _compare(self, actual, *, exhaustive=True):
        return compare_evidence_array(
            self.EXPECTED,
            actual,
            comparator=self.COMPARATOR,
            structural="match_by:item_no,sub_no",
            exhaustive=exhaustive,
        )

    def test_reordered_rows_pass(self):
        actual = [
            {"item_no": "0001", "sub_no": "b", "amount": 200},
            {"item_no": "0001", "sub_no": "a", "amount": 100},
        ]
        assert self._compare(actual).passed

    def test_one_key_mismatch_fails(self):
        actual = [
            {"item_no": "0001", "sub_no": "A", "amount": 100},
            {"item_no": "0002", "sub_no": "B", "amount": 200},
        ]
        result = self._compare(actual)
        assert not result.passed
        assert result.reason == "missing_match:item_no,sub_no"

    def test_threshold_counts_composite_tuples(self):
        # Both expected rows share item_no; only the composite (item_no, sub_no)
        # distinguishes them, so matching just one prediction must not pass.
        actual = [{"item_no": "0001", "sub_no": "A", "amount": 100}]
        result = self._compare(actual, exhaustive=False)
        assert not result.passed

    def test_per_key_comparator_applies(self):
        # amount graded as number: "200.00" == 200 in the full-row compare.
        actual = [
            {"item_no": "0001", "sub_no": "A", "amount": "100"},
            {"item_no": "0001", "sub_no": "B", "amount": "200.00"},
        ]
        assert self._compare(actual).passed

    def test_expected_missing_key_reports_specific_key(self):
        result = compare_evidence_array(
            [{"item_no": "0001"}],
            [{"item_no": "0001"}],
            comparator={"item_no": "case_insensitive"},
            structural="match_by:item_no,sub_no",
        )
        assert not result.passed
        assert result.reason == "expected_missing_key:sub_no"


class TestMatchByUnverifiedCells:
    def test_null_gold_cell_is_skipped_not_asserted(self):
        # The GT builder nulls out cells it could not consensus-verify; a null
        # gold cell must not fail a prediction that has a real value there.
        expected = [{"sku": "A1", "description": None, "amount": 5}]
        actual = [{"sku": "A1", "description": "WIDGET, STEEL", "amount": 5}]
        result = compare_evidence_array(
            expected,
            actual,
            comparator={"sku": "case_insensitive", "description": "case_insensitive", "amount": "number"},
            structural="match_by:sku",
        )
        assert result.passed

    def test_null_gold_identity_matches_absent_actual_key(self):
        # Providers may omit keys whose value is null; absent == null for the
        # identity compare.
        expected = [{"cusip": "X1", "put_call": None, "value": 10}]
        actual = [{"cusip": "X1", "value": 10}]
        result = compare_evidence_array(
            expected,
            actual,
            comparator={"cusip": "case_insensitive", "value": "number"},
            structural="match_by:cusip,put_call",
        )
        assert result.passed

    def test_non_null_cells_still_graded(self):
        expected = [{"sku": "A1", "amount": 5}]
        actual = [{"sku": "A1", "amount": 7}]
        result = compare_evidence_array(
            expected,
            actual,
            comparator={"sku": "case_insensitive", "amount": "number"},
            structural="match_by:sku",
        )
        assert not result.passed


def _family_rules(*, parent_verified: bool = True) -> list[ExtractFieldTestRule]:
    """A 3-row match_by family with per-index leaf rules, GT-builder style."""
    rows = [
        {"item_no": "0001", "description": "ALPHA UNIT", "amount": 100},
        {"item_no": "0002", "description": "BRAVO UNIT", "amount": 200},
        {"item_no": "0003", "description": "CHARLIE UNIT", "amount": 300},
    ]
    rules = [
        ExtractFieldTestRule(
            field_path="items",
            evidence=[FieldEvidence(page=1, value=row, coarse=True) for row in rows],
            comparator={"item_no": "case_insensitive", "description": "case_insensitive", "amount": "number"},
            structural="match_by:item_no",
            verified=parent_verified,
        )
    ]
    for index, row in enumerate(rows):
        for leaf, comparator in (
            ("item_no", "case_insensitive"),
            ("description", "case_insensitive"),
            ("amount", "number"),
        ):
            rules.append(
                ExtractFieldTestRule(
                    field_path=f"items[{index}].{leaf}",
                    evidence=[FieldEvidence(page=1, value=row[leaf], coarse=True)],
                    comparator=comparator,
                )
            )
    return rules


def _rows(*indices: int) -> list[dict]:
    rows = [
        {"item_no": "0001", "description": "ALPHA UNIT", "amount": 100},
        {"item_no": "0002", "description": "BRAVO UNIT", "amount": 200},
        {"item_no": "0003", "description": "CHARLIE UNIT", "amount": 300},
    ]
    return [rows[i] for i in indices]


def _value_pass_metric(rules, extracted_data):
    metrics = _named(
        compute_extract_field_grounding_metrics(
            extracted_data=extracted_data,
            field_rules=rules,
            field_citations=[],
            data_schema={"type": "object"},
        )
    )
    return metrics["extract_evidence_value_pass_rate"]


class TestRowAlignment:
    def test_reordered_rows_all_leaves_pass(self):
        metric = _value_pass_metric(_family_rules(), {"items": _rows(2, 0, 1)})
        assert metric.value == 1.0
        assert metric.metadata["match_by_row_alignment"] == {"items": 3}

    def test_dropped_row_fails_only_that_row(self):
        metric = _value_pass_metric(_family_rules(), {"items": _rows(0, 2)})
        failures = {r["field_path"]: r["reason"] for r in metric.metadata["rule_results"] if not r["value_pass"]}
        # The dropped row's three leaves grade as missing; the parent rule
        # correctly reports the missing row; every other row passes.
        assert set(failures) == {
            "items",
            "items[1].item_no",
            "items[1].description",
            "items[1].amount",
        }
        assert failures["items[1].amount"] == "missing_prediction"

    def test_drop_plus_reorder_does_not_cascade(self):
        metric = _value_pass_metric(_family_rules(), {"items": _rows(2, 0)})
        failures = [r["field_path"] for r in metric.metadata["rule_results"] if not r["value_pass"]]
        assert sorted(failures) == ["items", "items[1].amount", "items[1].description", "items[1].item_no"]

    def test_unverified_parent_still_provides_alignment(self):
        metric = _value_pass_metric(_family_rules(parent_verified=False), {"items": _rows(1, 2, 0)})
        # Parent excluded from grading, but its match_by declaration still
        # realigns the leaf rules.
        assert metric.value == 1.0
        graded_paths = {r["field_path"] for r in metric.metadata["rule_results"]}
        assert "items" not in graded_paths

    def test_ordered_family_is_not_realigned(self):
        # Without a match_by parent, exact-index semantics are preserved:
        # reordering rows fails the leaf rules.
        rules = [rule for rule in _family_rules() if rule.field_path != "items"]
        metric = _value_pass_metric(rules, {"items": _rows(2, 0, 1)})
        assert metric.value == 0.0
        assert metric.metadata["match_by_row_alignment"] == {}

    def test_duplicate_identities_keep_relative_order(self):
        rows = [
            {"item_no": "0001", "amount": 100},
            {"item_no": "0001", "amount": 200},
        ]
        rules = [
            ExtractFieldTestRule(
                field_path="items",
                evidence=[FieldEvidence(page=1, value=row, coarse=True) for row in rows],
                comparator={"item_no": "case_insensitive", "amount": "number"},
                structural="match_by:item_no",
            )
        ]
        for index, row in enumerate(rows):
            for leaf, comparator in (("item_no", "case_insensitive"), ("amount", "number")):
                rules.append(
                    ExtractFieldTestRule(
                        field_path=f"items[{index}].{leaf}",
                        evidence=[FieldEvidence(page=1, value=row[leaf], coarse=True)],
                        comparator=comparator,
                    )
                )
        metric = _value_pass_metric(rules, {"items": rows})
        assert metric.value == 1.0

    def test_positional_fallback_without_identity_info(self):
        # Leaf rules exist only for non-identity cells and the parent has no
        # evidence rows: alignment falls back to today's positional lookup.
        rules = [
            ExtractFieldTestRule(
                field_path="items",
                evidence=[],
                comparator={"amount": "number"},
                structural="match_by:item_no",
                evidence_required=False,
            ),
            ExtractFieldTestRule(
                field_path="items[0].amount",
                evidence=[FieldEvidence(page=1, value=100, coarse=True)],
                comparator="number",
            ),
            ExtractFieldTestRule(
                field_path="items[1].amount",
                evidence=[FieldEvidence(page=1, value=200, coarse=True)],
                comparator="number",
            ),
        ]
        metric = _value_pass_metric(rules, {"items": [{"amount": 100}, {"amount": 200}]})
        assert metric.value == 1.0

    def test_punctuation_case_drift_aligns_via_normalization(self):
        metric = _value_pass_metric(
            _family_rules(),
            {
                "items": [
                    {"item_no": "0002", "description": "BRAVO UNIT", "amount": 200},
                    {"item_no": "0003", "description": "CHARLIE UNIT", "amount": 300},
                    {"item_no": "0001.", "description": "alpha unit", "amount": 100},
                ]
            },
        )
        assert metric.value == 1.0

    def test_typed_identity_drift_aligns_via_comparator_pass(self):
        # Numeric identity emitted in a different surface form: the exact
        # normalized pass misses ("100" vs "100.00"), the per-key comparator
        # pass (number) aligns the rows.
        rows = [{"qty": 100, "name": "ALPHA"}, {"qty": 250, "name": "BRAVO"}]
        rules = [
            ExtractFieldTestRule(
                field_path="items",
                evidence=[FieldEvidence(page=1, value=row, coarse=True) for row in rows],
                comparator={"qty": "number", "name": "case_insensitive"},
                structural="match_by:qty",
            )
        ]
        for index, row in enumerate(rows):
            for leaf, comparator in (("qty", "number"), ("name", "case_insensitive")):
                rules.append(
                    ExtractFieldTestRule(
                        field_path=f"items[{index}].{leaf}",
                        evidence=[FieldEvidence(page=1, value=row[leaf], coarse=True)],
                        comparator=comparator,
                    )
                )
        metric = _value_pass_metric(
            rules,
            {"items": [{"qty": "250.00", "name": "BRAVO"}, {"qty": "100.00", "name": "ALPHA"}]},
        )
        assert metric.value == 1.0


class TestRowAlignmentRegressionGuards:
    """Edges where realignment must not change correct grading outcomes."""

    def test_extra_hallucinated_row_does_not_break_leaf_alignment(self):
        # All GT rows present plus one fabricated row: every leaf still passes;
        # the parent rule keeps penalizing the extra row (exhaustive default).
        rows = _rows(0, 1, 2)
        rows.insert(1, {"item_no": "9999", "description": "FABRICATED", "amount": 999})
        metric = _value_pass_metric(_family_rules(), {"items": rows})
        failures = {r["field_path"]: r["reason"] for r in metric.metadata["rule_results"] if not r["value_pass"]}
        assert failures == {"items": "extra_array_items"}

    def test_wrong_cell_value_at_right_identity_fails_only_that_cell(self):
        rows = _rows(2, 0, 1)
        rows[1] = {**rows[1], "amount": 12345}  # GT row 0, reordered, one bad cell
        metric = _value_pass_metric(_family_rules(), {"items": rows})
        failed_leaves = [
            r["field_path"]
            for r in metric.metadata["rule_results"]
            if not r["value_pass"] and r["field_path"] != "items"
        ]
        assert failed_leaves == ["items[0].amount"]

    def test_empty_prediction_array_fails_all_value_bearing_leaves(self):
        metric = _value_pass_metric(_family_rules(), {"items": []})
        leaf_results = [r for r in metric.metadata["rule_results"] if r["field_path"] != "items"]
        assert all(not r["value_pass"] for r in leaf_results)
        assert all(r["reason"] == "missing_prediction" for r in leaf_results)

    def test_non_list_prediction_keeps_missing_semantics(self):
        metric = _value_pass_metric(_family_rules(), {"items": "not-a-list"})
        leaf_results = [r for r in metric.metadata["rule_results"] if r["field_path"] != "items"]
        assert all(not r["value_pass"] for r in leaf_results)
