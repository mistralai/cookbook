"""Tests for v0.2 schema extensions, comparators, and page-grounded metrics."""

from __future__ import annotations

import pytest

from extract_bench.evaluation.metrics.field_grounding.extract_adapter import (
    compute_extract_field_grounding_metrics,
)
from extract_bench.evaluation.metrics.field_grounding.value_compare import (
    candidate_values_for_rule,
    compare_field_with_rule,
    compare_value_against_rule,
)
from extract_bench.schemas.extract_output import FieldCitation
from extract_bench.test_cases.schema import (
    ExtractFieldBbox,
    ExtractFieldTestRule,
    FieldEvidence,
    iter_rule_evidence,
)


def _named(metrics):
    return {m.metric_name: m for m in metrics}


class TestIterRuleEvidence:
    def test_legacy_bboxes_synthesize_evidence(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.2, 0.2])],
        )
        evidence = iter_rule_evidence(rule)
        assert len(evidence) == 1
        assert evidence[0].page == 1
        assert evidence[0].value == "bar"
        assert evidence[0].coarse is False

    def test_v02_evidence_list_returned_verbatim(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="UNII",
            evidence=[
                FieldEvidence(page=2, bbox=[0.5, 0.5, 0.1, 0.1], value="LABEL"),
                FieldEvidence(page=3, value="UNII", coarse=True),
            ],
        )
        evidence = iter_rule_evidence(rule)
        assert len(evidence) == 2
        assert evidence[1].coarse is True
        assert evidence[1].bbox is None

    def test_empty_rule_returns_empty(self):
        rule = ExtractFieldTestRule(field_path="foo", expected_value=None)
        assert iter_rule_evidence(rule) == []


class TestComparatorShim:
    def test_falls_through_to_attributed_value_when_no_comparator(self):
        rule = ExtractFieldTestRule(field_path="foo", expected_value="bar")
        result = compare_field_with_rule(rule, "bar", "bar", expected_type="string")
        assert result.passed

    def test_rule_none_falls_through(self):
        result = compare_field_with_rule(None, "bar", "bar", expected_type="string")
        assert result.passed

    def test_string_substring_comparator(self):
        rule = ExtractFieldTestRule(
            field_path="hazmat_code",
            evidence=[FieldEvidence(page=1, value="1463")],
            comparator="string_substring",
        )
        result = compare_value_against_rule(rule, "UN1463", expected_type="string")
        assert result.passed

    def test_object_comparator_map(self):
        rule = ExtractFieldTestRule(
            field_path="warnings",
            evidence=[FieldEvidence(page=1, value={"text": "Do not use", "category": "boxed"})],
            comparator={"text": "case_insensitive", "category": "enum"},
        )
        result = compare_value_against_rule(rule, {"text": "do not use", "category": "BOXED"}, expected_type="string")
        assert result.passed


class TestCandidateValuesForRule:
    def test_legacy_returns_single_expected(self):
        rule = ExtractFieldTestRule(field_path="foo", expected_value="bar")
        assert candidate_values_for_rule(rule) == ["bar"]

    def test_v02_evidence_emits_all_unique_values(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="canonical",
            evidence=[
                FieldEvidence(page=1, value="alt-1"),
                FieldEvidence(page=2, value="canonical"),
                FieldEvidence(page=3, value="alt-1"),
            ],
        )
        cands = candidate_values_for_rule(rule)
        assert "alt-1" in cands and "canonical" in cands
        assert len(cands) == 2

    def test_none_rule_returns_none(self):
        assert candidate_values_for_rule(None) == [None]


class TestOrOverEvidence:
    def test_pipeline_value_matches_alternate_evidence_value(self):
        rule = ExtractFieldTestRule(
            field_path="ingredients[0].name",
            expected_value="UNII-CANONICAL",
            evidence=[
                FieldEvidence(page=1, value="LABEL-PRINTED"),
                FieldEvidence(page=2, value="UNII-CANONICAL"),
            ],
        )
        result = compare_value_against_rule(rule, "LABEL-PRINTED", expected_type="string")
        assert result.passed

    def test_no_match_against_any_candidate(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            evidence=[FieldEvidence(page=1, value="baz")],
        )
        result = compare_value_against_rule(rule, "qux", expected_type="string")
        assert not result.passed


class TestPageGroundedMetric:
    def test_passes_when_page_matches_no_bbox(self):
        rule = ExtractFieldTestRule(
            field_path="ndc[0]",
            expected_value="37000-439",
            evidence=[FieldEvidence(page=4, value="37000-439")],
        )
        citations = [FieldCitation(field_path="ndc[0]", page=4, bbox=None, source="extend")]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"ndc": ["37000-439"]},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_evidence_page_pass_rate"].value == 1.0
        # The GT evidence carries no bbox, so bbox grounding is undefined here:
        # the bbox_* metrics are omitted (excluded from the dataset average)
        # rather than emitted as 0.0. Page grounding above is unaffected.
        assert "extract_evidence_bbox_coverage" not in metrics
        assert "extract_evidence_bbox_covered_pass_rate" not in metrics
        assert "extract_evidence_bbox_IOU_pass_rate" not in metrics
        assert "extract_evidence_bbox_IOU_alignment" not in metrics

    def test_fails_when_page_wrong(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            evidence=[FieldEvidence(page=1, value="bar")],
        )
        citations = [FieldCitation(field_path="foo", page=99, bbox=None, source="x")]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"foo": "bar"},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_evidence_page_pass_rate"].value == 0.0


class TestCoarseParentPrefixWalk:
    def test_parent_cite_counts_for_m2a_not_m2b(self):
        rule = ExtractFieldTestRule(
            field_path="warnings[0].text",
            expected_value="Do not use",
            evidence=[FieldEvidence(page=2, value="Do not use")],
        )
        citations = [
            FieldCitation(field_path="warnings", page=2, bbox=[0.1, 0.1, 0.5, 0.5], source="reducto"),
        ]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"warnings": [{"text": "Do not use"}]},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_evidence_page_pass_rate"].value == 1.0
        # GT evidence carries no bbox -> bbox metrics are omitted, not 0.0.
        assert "extract_evidence_bbox_coverage" not in metrics


class TestCoverage:
    def test_zero_when_no_citations(self):
        rule = ExtractFieldTestRule(field_path="foo", evidence=[FieldEvidence(page=1, value="bar")])
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"foo": "bar"},
                field_rules=[rule],
                field_citations=[],
                data_schema={"type": "object"},
            )
        )
        # No citations -> nothing is page-qualified, so value+page cannot pass.
        assert metrics["extract_evidence_page_pass_rate"].value == 0.0
        # GT evidence carries no bbox -> bbox_coverage is undefined and omitted.
        assert "extract_evidence_bbox_coverage" not in metrics

    def test_one_when_all_rules_have_citations(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            evidence=[FieldEvidence(page=1, bbox=[0.1, 0.1, 0.2, 0.2], value="bar")],
        )
        citations = [FieldCitation(field_path="foo", page=1, bbox=[0.1, 0.1, 0.2, 0.2], source="x")]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"foo": "bar"},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_evidence_page_pass_rate"].value == 1.0
        assert metrics["extract_evidence_bbox_coverage"].value == 1.0


class TestPerRuleIouThreshold:
    def test_loose_threshold_lets_imperfect_iou_pass(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.4, 0.4])],
            iou_threshold=0.2,
        )
        citations = [FieldCitation(field_path="foo", page=1, bbox=[0.15, 0.15, 0.4, 0.4], source="x")]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"foo": "bar"},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_attribution_pass_rate"].value == 1.0


class TestBackwardCompat:
    def test_legacy_bboxes_path_unchanged(self):
        """Legacy rules without v0.2 evidence keep producing the same bbox-gated results."""
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.2, 0.2])],
        )
        citations = [FieldCitation(field_path="foo", page=1, bbox=[0.1, 0.1, 0.2, 0.2], source="x")]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"foo": "bar"},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_attribution_pass_rate"].value == 1.0
        assert metrics["f1"].value == 1.0


class TestV02StrayAndNullSemantics:
    """v0.2 rules carry the canonical value in evidence, not expected_value.

    A rule with ``expected_value=None`` but populated ``evidence[].value`` is
    NOT stray and NOT null-expected — it prescribes the value via evidence.
    """

    def test_v02_evidence_only_rule_scores_value_f1(self):
        rule = ExtractFieldTestRule(
            field_path="drug_name",
            expected_value=None,
            evidence=[FieldEvidence(page=1, bbox=[0.1, 0.1, 0.2, 0.2], value="Aspirin")],
            comparator="case_insensitive",
        )
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"drug_name": "Aspirin"},
                field_rules=[rule],
                field_citations=[],
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_evidence_value_pass_rate"].value == 1.0

    def test_v02_evidence_only_rule_not_treated_as_hallucination(self):
        rule = ExtractFieldTestRule(
            field_path="drug_name",
            expected_value=None,
            evidence=[FieldEvidence(page=1, value="Aspirin")],
        )
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"drug_name": "Aspirin"},
                field_rules=[rule],
                field_citations=[],
                data_schema={"type": "object"},
            )
        )
        assert "null_hallucination_rate" not in metrics

    def test_no_evidence_no_expected_value_is_null_expected(self):
        """Legacy null-expected behavior preserved when evidence is also empty."""
        rule = ExtractFieldTestRule(
            field_path="adverse_reactions",
            expected_value=None,
            evidence=None,
        )
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"adverse_reactions": "made up"},
                field_rules=[rule],
                field_citations=[],
                data_schema={"type": "object"},
            )
        )
        assert metrics["null_hallucination_rate"].value == 1.0

    def test_explicit_stray_tag_still_treated_stray(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            evidence=[FieldEvidence(page=1, value="bar")],
            tags=["stray"],
        )
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"foo": "bar"},
                field_rules=[rule],
                field_citations=[],
                data_schema={"type": "object"},
            )
        )
        assert "f1" not in metrics
        assert "extract_evidence_value_pass_rate" not in metrics

    def test_evidence_not_required_excluded(self):
        rule = ExtractFieldTestRule(
            field_path="paper_type",
            evidence=[],
            evidence_required=False,
            source_policy="inferred",
        )
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"paper_type": "research"},
                field_rules=[rule],
                field_citations=[],
                data_schema={"type": "object"},
            )
        )
        assert "extract_evidence_value_pass_rate" not in metrics

    def test_empty_required_evidence_null_passes_value_only(self):
        rule = ExtractFieldTestRule(field_path="unused", evidence=[], evidence_required=True)
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"unused": None},
                field_rules=[rule],
                field_citations=[],
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_evidence_value_pass_rate"].value == 1.0


class TestLoaderDictShape:
    def test_field_rules_dict_converts_to_list(self, tmp_path):
        import json

        from extract_bench.test_cases.loader import load_test_case

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        config = {
            "_schema_version": "sample/v0.2",
            "data_schema": {"type": "object", "properties": {"foo": {"type": "string"}}},
            "expected_output": {"foo": "bar"},
            "_field_rules": {
                "foo": {
                    "expected_value": "bar",
                    "evidence": [{"page": 1, "value": "bar", "bbox": [0.1, 0.1, 0.2, 0.2]}],
                    "iou_threshold": 0.4,
                    "comparator": "exact",
                },
            },
        }
        (tmp_path / "doc.test.json").write_text(json.dumps(config))
        case = load_test_case(pdf)
        assert case is not None
        assert case.schema_version == "sample/v0.2"
        rules = case.test_rules
        assert len(rules) == 1
        rule = rules[0]
        assert rule.field_path == "foo"
        assert rule.iou_threshold == 0.4
        assert rule.comparator == "exact"
        assert rule.evidence is not None and rule.evidence[0].value == "bar"

    def test_conflict_both_shapes_errors(self, tmp_path):
        import json

        from extract_bench.test_cases.loader import load_test_case

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        config = {
            "data_schema": {"type": "object"},
            "expected_output": {},
            "_field_rules": {"foo": {"expected_value": "bar"}},
            "test_rules": [{"type": "extract_field", "field_path": "foo", "expected_value": "bar"}],
        }
        (tmp_path / "doc.test.json").write_text(json.dumps(config))
        with pytest.raises(ValueError, match="cannot specify both"):
            load_test_case(pdf)

    def test_conflict_fires_even_when_test_rules_is_empty_list(self, tmp_path):
        """Empty `test_rules: []` is still presence — conflict policy must fire."""
        import json

        from extract_bench.test_cases.loader import load_test_case

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        config = {
            "data_schema": {"type": "object"},
            "expected_output": {},
            "_field_rules": {"foo": {"expected_value": "bar"}},
            "test_rules": [],
        }
        (tmp_path / "doc.test.json").write_text(json.dumps(config))
        with pytest.raises(ValueError, match="cannot specify both"):
            load_test_case(pdf)


class TestCoarseParentNearestOnly:
    """Coarse parent prefix-walk should only count the NEAREST non-empty
    ancestor, not every ancestor up the path."""

    def test_nearer_parent_wins_over_more_general(self):
        """Citations at warnings[0] (immediate parent) and warnings (root)
        coexist for rule warnings[0].text — only the nearer warnings[0]
        should count as coarse."""
        rule = ExtractFieldTestRule(
            field_path="warnings[0].text",
            expected_value="Do not use",
            evidence=[FieldEvidence(page=2, value="Do not use")],
        )
        # Both citations are coarse parents; the nearer is warnings[0].
        # Both reside on page 2 so M2a passes either way; the test asserts
        # only the nearer ancestor is collected — not both.
        citations = [
            FieldCitation(field_path="warnings", page=2, bbox=None, source="root"),
            FieldCitation(field_path="warnings[0]", page=2, bbox=None, source="immediate"),
        ]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"warnings": [{"text": "Do not use"}]},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        rule_results = metrics["extract_evidence_page_pass_rate"].metadata["rule_results"]
        assert len(rule_results) == 1
        # Nearest parent has 1 cite, root has 1 cite. With nearest-only walk
        # we expect exactly 1 coarse cite, not 2.
        assert rule_results[0]["coarse_citation_count"] == 1


class TestBBoxDiagnostics:
    """Bbox overlap is diagnostic-only for v0.2 evidence metrics."""

    def test_passes_when_one_evidence_pair_aligns_geometrically(self):
        """Two evidence entries with same value at different locations.
        Pipeline cites only ONE of the two locations correctly. Union-IoU
        would dilute; per-pair scoring against the matched evidence
        passes."""
        rule = ExtractFieldTestRule(
            field_path="drug_name",
            expected_value="Aspirin",
            evidence=[
                FieldEvidence(page=1, bbox=[0.05, 0.05, 0.1, 0.05], value="Aspirin"),
                FieldEvidence(page=4, bbox=[0.7, 0.7, 0.1, 0.05], value="Aspirin"),
            ],
        )
        # Pipeline cites the page-4 location precisely.
        citations = [
            FieldCitation(field_path="drug_name", page=4, bbox=[0.7, 0.7, 0.1, 0.05], source="x"),
        ]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"drug_name": "Aspirin"},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_evidence_page_pass_rate"].value == 1.0
        assert metrics["extract_evidence_bbox_IOU_alignment"].value == 1.0

    def test_fails_when_no_pair_aligns(self):
        """Pipeline cites a page neither evidence entry mentions —
        per-pair never finds a passing pair."""
        rule = ExtractFieldTestRule(
            field_path="drug_name",
            expected_value="Aspirin",
            evidence=[
                FieldEvidence(page=1, bbox=[0.05, 0.05, 0.1, 0.05], value="Aspirin"),
                FieldEvidence(page=4, bbox=[0.7, 0.7, 0.1, 0.05], value="Aspirin"),
            ],
        )
        citations = [
            FieldCitation(field_path="drug_name", page=99, bbox=[0.5, 0.5, 0.1, 0.1], source="x"),
        ]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"drug_name": "Aspirin"},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_evidence_page_pass_rate"].value == 0.0
        assert metrics["extract_evidence_bbox_IOU_alignment"].value == 0.0

    def test_value_must_match_same_evidence_as_bbox(self):
        """Two evidence entries with DIFFERENT values at different bboxes.
        Pipeline emits value matching evidence A but cites evidence B's
        location. Per-pair semantics correctly fail (value/bbox cross-pair
        mismatch); union semantics would falsely pass."""
        rule = ExtractFieldTestRule(
            field_path="ingredient",
            expected_value=None,
            evidence=[
                FieldEvidence(page=1, bbox=[0.05, 0.05, 0.1, 0.05], value="LABEL-NAME"),
                FieldEvidence(page=4, bbox=[0.7, 0.7, 0.1, 0.05], value="UNII-NAME"),
            ],
        )
        # Pipeline says value="LABEL-NAME" but cites the UNII bbox at page 4.
        # Per-pair: pair 1 (LABEL-NAME, page1 bbox) fails IoU; pair 2
        # (UNII-NAME, page4 bbox) fails value. No pair passes. Fail.
        citations = [
            FieldCitation(field_path="ingredient", page=4, bbox=[0.7, 0.7, 0.1, 0.05], source="x"),
        ]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"ingredient": "LABEL-NAME"},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_evidence_value_pass_rate"].value == 1.0
        assert metrics["extract_evidence_page_pass_rate"].value == 1.0
        assert metrics["extract_evidence_bbox_IOU_alignment"].value == 1.0


class TestStructuralRules:
    def test_array_of_scalars_set(self):
        rule = ExtractFieldTestRule(
            field_path="codes",
            evidence=[FieldEvidence(page=1, value="UN1463"), FieldEvidence(page=2, value="1463")],
            comparator="case_insensitive",
            structural="set",
            exhaustive=True,
        )
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"codes": ["1463", "UN1463"]},
                field_rules=[rule],
                field_citations=[FieldCitation(field_path="codes", page=1, bbox=None, source="x")],
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_evidence_value_pass_rate"].value == 1.0
        assert metrics["extract_evidence_page_pass_rate"].value == 1.0

    def test_array_of_objects_match_by_text(self):
        rule = ExtractFieldTestRule(
            field_path="warnings",
            evidence=[
                FieldEvidence(page=1, value={"text": "Do not use", "category": "boxed"}),
                FieldEvidence(page=2, value={"text": "Ask a doctor", "category": "general"}),
            ],
            comparator={"text": "case_insensitive", "category": "enum"},
            structural="match_by:text",
            exhaustive=True,
        )
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={
                    "warnings": [
                        {"text": "ask a doctor", "category": "GENERAL"},
                        {"text": "do not use", "category": "BOXED"},
                    ]
                },
                field_rules=[rule],
                field_citations=[FieldCitation(field_path="warnings", page=2, bbox=None, source="x")],
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_evidence_value_pass_rate"].value == 1.0


class TestDescendantCitationPairing:
    """For array-shaped rules, pipelines that emit per-leaf citations
    (``grants[0].recipient_name``, ``grants[0].address_line1``, ...) but no
    synthetic array-level citation should still be paired with the rule
    whose ``field_path`` is the array root (``grants``)."""

    def test_array_rule_paired_with_leaf_citations(self):
        rule = ExtractFieldTestRule(
            field_path="grants",
            evidence=[
                FieldEvidence(
                    page=1,
                    bbox=[0.05, 0.30, 0.20, 0.02],
                    value={"recipient_name": "ACME FOUNDATION", "amount": 5000},
                ),
            ],
            comparator={"recipient_name": "case_insensitive", "amount": "number"},
            structural="match_by:recipient_name",
            exhaustive=True,
            expected_min=1,
        )
        # Pipeline emits per-leaf citations only; no top-level "grants" citation.
        leaf_citations = [
            FieldCitation(
                field_path="grants[0].recipient_name",
                page=1,
                bbox=[0.05, 0.30, 0.10, 0.02],
                source="extract_v2",
            ),
            FieldCitation(
                field_path="grants[0].amount",
                page=1,
                bbox=[0.16, 0.30, 0.04, 0.02],
                source="extract_v2",
            ),
        ]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"grants": [{"recipient_name": "ACME FOUNDATION", "amount": 5000}]},
                field_rules=[rule],
                field_citations=leaf_citations,
                data_schema={"type": "object"},
            )
        )
        rule_results = metrics["extract_evidence_page_pass_rate"].metadata["rule_results"]
        assert len(rule_results) == 1
        # Both leaf citations are descendants of the array rule's pattern.
        assert rule_results[0]["descendant_citation_count"] == 2
        assert rule_results[0]["exact_citation_count"] == 0
        # page_pass kicks in via the descendant walk even though
        # ``exact_cits`` is empty.
        assert rule_results[0]["page_pass"] is True
        assert metrics["extract_evidence_page_pass_rate"].value == 1.0
        assert metrics["extract_evidence_value_pass_rate"].value == 1.0

    def test_array_rule_unions_exact_and_descendant(self):
        """When a pipeline emits BOTH an array-level citation AND per-leaf
        citations, both contribute. The descendant walk is union, not gated
        on absence of exact citations."""
        rule = ExtractFieldTestRule(
            field_path="grants",
            evidence=[
                FieldEvidence(page=2, bbox=[0.1, 0.4, 0.2, 0.02], value={"name": "X"}),
            ],
            comparator={"name": "case_insensitive"},
            structural="match_by:name",
            exhaustive=True,
            expected_min=1,
        )
        citations = [
            # Array-level citation on page 2.
            FieldCitation(field_path="grants", page=2, bbox=None, source="extract_v2"),
            # Per-leaf citation on page 2 (descendant).
            FieldCitation(field_path="grants[0].name", page=2, bbox=[0.1, 0.4, 0.2, 0.02], source="extract_v2"),
        ]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"grants": [{"name": "X"}]},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        rule_results = metrics["extract_evidence_page_pass_rate"].metadata["rule_results"]
        assert rule_results[0]["exact_citation_count"] == 1
        assert rule_results[0]["descendant_citation_count"] == 1
        assert rule_results[0]["page_pass"] is True


class TestVacuousNullPagePass:
    """Rules whose evidence asserts only null values (or have empty
    evidence) make no positive page claim; ``value_pass=True`` should
    imply ``page_pass=True`` without requiring any citation."""

    def test_all_null_evidence_passes_page_when_value_passes(self):
        """Gold has one evidence entry with ``value=None`` (e.g. an
        optional field the filer left blank). Pipeline correctly returns
        null. No citation is emitted. ``page_pass`` should be True."""
        rule = ExtractFieldTestRule(
            field_path="amendment_type",
            evidence=[FieldEvidence(page=1, bbox=None, value=None, coarse=True)],
        )
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"amendment_type": None},
                field_rules=[rule],
                field_citations=[],  # No citation for null fields.
                data_schema={"type": "object"},
            )
        )
        rule_results = metrics["extract_evidence_page_pass_rate"].metadata["rule_results"]
        assert len(rule_results) == 1
        assert rule_results[0]["value_pass"] is True
        # Without the vacuous-pass fix, this would be False because
        # ``page_qual`` is False (no citation) and the rule has no
        # value-bearing evidence to verify a page against.
        assert rule_results[0]["page_pass"] is True
        assert metrics["extract_evidence_page_pass_rate"].value == 1.0


class TestMixedEvidenceNullFiltering:
    """When a rule's evidence list mixes null and value-bearing entries,
    only the value-bearing entries' pages must be matched. Null entries
    don't bind to a page."""

    def test_mixed_evidence_only_value_bearing_pages_matter(self):
        """Two evidence entries: one with value=None on page 1 (no claim),
        one with value="X" on page 3. Pipeline cites page 3 only.
        page_pass should be True — the page-3 value-bearing entry matched."""
        rule = ExtractFieldTestRule(
            field_path="drug_name",
            evidence=[
                # Null entry — bound to page 1 but makes no claim.
                FieldEvidence(page=1, bbox=None, value=None, coarse=True),
                # Value-bearing entry on page 3.
                FieldEvidence(page=3, bbox=[0.1, 0.1, 0.2, 0.05], value="Aspirin"),
            ],
        )
        citations = [
            FieldCitation(field_path="drug_name", page=3, bbox=[0.1, 0.1, 0.2, 0.05], source="x"),
        ]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"drug_name": "Aspirin"},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        rule_results = metrics["extract_evidence_page_pass_rate"].metadata["rule_results"]
        assert rule_results[0]["value_pass"] is True
        assert rule_results[0]["page_pass"] is True
        assert metrics["extract_evidence_page_pass_rate"].value == 1.0

    def test_mixed_evidence_pipeline_cites_only_null_page_fails(self):
        """Same setup but pipeline cites page 1 (where the null entry
        lives) instead of page 3. The null entry doesn't satisfy the
        positive page claim, so page_pass should be False."""
        rule = ExtractFieldTestRule(
            field_path="drug_name",
            evidence=[
                FieldEvidence(page=1, bbox=None, value=None, coarse=True),
                FieldEvidence(page=3, bbox=[0.1, 0.1, 0.2, 0.05], value="Aspirin"),
            ],
        )
        citations = [
            FieldCitation(field_path="drug_name", page=1, bbox=None, source="x"),
        ]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"drug_name": "Aspirin"},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        rule_results = metrics["extract_evidence_page_pass_rate"].metadata["rule_results"]
        assert rule_results[0]["value_pass"] is True
        # page 1 is where the null entry lives; page 3 is the value-bearing
        # entry. Pipeline missed page 3, so page_pass=False.
        assert rule_results[0]["page_pass"] is False
