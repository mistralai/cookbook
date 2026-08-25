"""Tests for the unified evidence metric (array_record + OR-values + grounding)."""

from __future__ import annotations

from typing import Any

from extract_bench.evaluation.metrics.extract import unified_evidence_metric
from extract_bench.evaluation.metrics.extract.array_record_match_metric import (
    ArrayRecordMatchMetric,
)
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    compute_unified_evidence_metrics,
)
from extract_bench.test_cases.schema import ExtractFieldTestRule, FieldEvidence


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "as_of": {"type": ["string", "null"]},
            "holdings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "security": {"type": ["string", "null"]},
                        "coupon": {"type": ["number", "null"]},
                        "note": {"type": ["string", "null"]},
                    },
                },
            },
        },
    }


def _rule(
    path: str,
    *values: Any,
    page: int | None = None,
    bbox: list[float] | None = None,
    normalizers: list[str] | None = None,
) -> ExtractFieldTestRule:
    ev = [FieldEvidence(value=v, page=page, bbox=bbox) for v in values] or [FieldEvidence(value=None)]
    return ExtractFieldTestRule(field_path=path, evidence=ev, normalizers=normalizers or [])


def _leaf_rules_single(expected: dict[str, Any]) -> list[ExtractFieldTestRule]:
    """One single-evidence rule per leaf, value == expected (no alt, no bbox)."""
    rules: list[ExtractFieldTestRule] = [_rule("as_of", expected.get("as_of"))]
    for i, row in enumerate(expected["holdings"]):
        for key in ("security", "coupon", "note"):
            rules.append(_rule(f"holdings[{i}].{key}", row.get(key)))
    return rules


def _val(metrics: list[Any], name: str) -> float | None:
    return next((m.value for m in metrics if m.metric_name == name), None)


# --------------------------------------------------------------- reduction
def test_value_metrics_equal_array_record_on_single_evidence() -> None:
    expected = {
        "as_of": "2024-01-01",
        "holdings": [
            {"security": "AAA", "coupon": 5.0, "note": None},
            {"security": "BBB", "coupon": 6.0, "note": "x"},
        ],
    }
    actual = {  # one cell wrong, one row reordered -> exercises Hungarian + a miss
        "as_of": "2024-01-01",
        "holdings": [
            {"security": "BBB", "coupon": 6.0, "note": "x"},
            {"security": "AAA", "coupon": 9.9, "note": None},
        ],
    }
    arr = ArrayRecordMatchMetric(normalize_dates=True).compute(expected=expected, actual=actual, data_schema=_schema())
    uni = compute_unified_evidence_metrics(expected, actual, _leaf_rules_single(expected), [], _schema())
    assert _val(uni, "extract_unified_value_f1") == _val(arr, "array_record_f1")
    assert _val(uni, "extract_unified_value_recall") == _val(arr, "array_record_recall")
    assert _val(uni, "extract_unified_value_precision") == _val(arr, "array_record_precision")


def test_reserved_provenance_key_is_not_scored() -> None:
    # A reserved _provenance key (top-level + per record) is attribution metadata, not
    # an extracted cell. A perfect prediction carrying it must still score 1.0 — the
    # value metric must not count it in either the precision or recall denominator.
    expected = {
        "as_of": "2024-01-01",
        "holdings": [
            {"security": "AAA", "coupon": 5.0, "note": None},
            {"security": "BBB", "coupon": 6.0, "note": "x"},
        ],
    }
    with_prov = {
        **expected,
        "_provenance": {"page": 1},
        "holdings": [{**r, "_provenance": {"page": 2 + i}} for i, r in enumerate(expected["holdings"])],
    }
    uni = compute_unified_evidence_metrics(expected, with_prov, _leaf_rules_single(expected), [], _schema())
    assert _val(uni, "extract_unified_value_precision") == 1.0
    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_value_f1") == 1.0


# ------------------------------------------------------------ OR-acceptance
def test_or_acceptable_alternate_value_passes() -> None:
    expected = {"as_of": None, "holdings": [{"security": "Acme Inc.", "coupon": 5.0, "note": None}]}
    actual = {"as_of": None, "holdings": [{"security": "Acme Incorporated", "coupon": 5.0, "note": None}]}
    # array_record sees a mismatch on `security`; the unified metric accepts the
    # alternate evidence value, so its recall is strictly higher.
    rules = _leaf_rules_single(expected)
    rules.append(_rule("holdings[0].security", "Acme Inc.", "Acme Incorporated"))
    arr = ArrayRecordMatchMetric().compute(expected=expected, actual=actual, data_schema=_schema())
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], _schema())
    assert _val(uni, "extract_unified_value_recall") > _val(arr, "array_record_recall")
    assert _val(uni, "extract_unified_value_recall") == 1.0


def test_alternate_evidence_values_use_full_assignment() -> None:
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "object", "properties": {"name": {"type": ["string", "null"]}}},
            }
        },
    }
    expected = {"rows": [{"name": "apple"}, {"name": "apples"}]}
    actual = {"rows": [{"name": "apple"}, {"name": "applez"}]}
    rules = [
        _rule("rows[0].name", "apple", "applez"),
        _rule("rows[1].name", "apples", "apple"),
    ]

    uni = compute_unified_evidence_metrics(expected, actual, rules, [], schema)

    assert _val(uni, "extract_unified_value_recall") == 1.0


def test_optional_terminal_punctuation_normalizer_is_field_scoped() -> None:
    expected = {"as_of": "1093' feet"}
    actual = {"as_of": "1093' feet."}
    schema = {"type": "object", "properties": {"as_of": {"type": ["string", "null"]}}}

    strict = compute_unified_evidence_metrics(expected, actual, [_rule("as_of", "1093' feet")], [], schema)
    normalized = compute_unified_evidence_metrics(
        expected,
        actual,
        [_rule("as_of", "1093' feet", normalizers=["optional_terminal_punctuation"])],
        [],
        schema,
    )

    assert _val(strict, "extract_unified_value_f1") == 0.0
    assert _val(normalized, "extract_unified_value_f1") == 1.0


def test_optional_terminal_punctuation_normalizer_does_not_drop_internal_punctuation() -> None:
    expected = {"as_of": "1,000 feet"}
    actual = {"as_of": "1000 feet."}
    schema = {"type": "object", "properties": {"as_of": {"type": ["string", "null"]}}}

    metrics = compute_unified_evidence_metrics(
        expected,
        actual,
        [_rule("as_of", "1,000 feet", normalizers=["optional_terminal_punctuation"])],
        [],
        schema,
    )

    assert _val(metrics, "extract_unified_value_f1") == 0.0


def test_optional_terminal_punctuation_normalizer_participates_in_array_alignment() -> None:
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "depth": {"type": ["string", "null"]},
                        "name": {"type": ["string", "null"]},
                    },
                },
            }
        },
    }
    expected = {"rows": [{"depth": "1093' feet", "name": "A"}, {"depth": "1073 feet", "name": "B"}]}
    actual = {"rows": [{"depth": "1073 feet.", "name": "B"}, {"depth": "1093' feet.", "name": "A"}]}
    rules = [
        _rule("rows[0].depth", "1093' feet", normalizers=["optional_terminal_punctuation"]),
        _rule("rows[0].name", "A"),
        _rule("rows[1].depth", "1073 feet", normalizers=["optional_terminal_punctuation"]),
        _rule("rows[1].name", "B"),
    ]

    metrics = compute_unified_evidence_metrics(expected, actual, rules, [], schema)

    assert _val(metrics, "extract_unified_value_f1") == 1.0


# -------------------------------------------------------- truncation / order
def test_truncation_penalized_no_vacuous_null_pass() -> None:
    # 3 GT rows, 2 of which have a null `note`. Prediction returns only 1 row.
    # The dropped rows' null cells must NOT pass: recall ~= 1/3, not inflated.
    expected = {
        "as_of": None,
        "holdings": [
            {"security": "AAA", "coupon": 1.0, "note": None},
            {"security": "BBB", "coupon": 2.0, "note": None},
            {"security": "CCC", "coupon": 3.0, "note": None},
        ],
    }
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": None}]}
    uni = compute_unified_evidence_metrics(expected, actual, _leaf_rules_single(expected), [], _schema())
    recall = _val(uni, "extract_unified_value_recall")
    assert recall is not None and 0.30 <= recall <= 0.40  # ~ 1 of 3 rows, not vacuously high


def test_reordered_rows_pass_without_match_by() -> None:
    expected = {
        "as_of": None,
        "holdings": [{"security": s, "coupon": float(i), "note": None} for i, s in enumerate("ABCDE")],
    }
    actual = {"as_of": None, "holdings": list(reversed(expected["holdings"]))}
    uni = compute_unified_evidence_metrics(expected, actual, _leaf_rules_single(expected), [], _schema())
    assert _val(uni, "extract_unified_value_recall") == 1.0  # Hungarian re-aligns; no match_by rule used


def test_unified_exact_peel_handles_large_shifted_array() -> None:
    rows = [{"id": str(i), "value": f"value-{i}"} for i in range(30_000)]
    expected = {"rows": rows}
    actual = {"rows": [{"id": "extra", "value": "extra"}, *rows]}
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["string", "null"]},
                        "value": {"type": ["string", "null"]},
                    },
                },
            }
        },
    }

    uni = compute_unified_evidence_metrics(expected, actual, [], [], schema)

    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_value_precision") == 60_000 / 60_002


def test_over_extraction_lowers_precision() -> None:
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": None}]}
    actual = {
        "as_of": None,
        "holdings": [
            {"security": "AAA", "coupon": 1.0, "note": None},
            {"security": "ZZZ", "coupon": 9.0, "note": "junk"},
        ],
    }
    uni = compute_unified_evidence_metrics(expected, actual, _leaf_rules_single(expected), [], _schema())
    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_value_precision") < 1.0  # extra row penalized


# ----------------------------------------------------- nested sub-records
def test_object_array_subfield_gets_per_record_credit() -> None:
    # `tags` is a list-of-objects sub-record. array_record scores it as one
    # opaque cell (whole list mismatches -> 1 of 2 subfields correct -> 0.5);
    # the unified metric recurses, so only tags[1].name is wrong -> 4 of 5
    # cells -> 0.8. This is the depth array_record cannot give.
    schema = {
        "type": "object",
        "properties": {
            "holdings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "security": {"type": ["string", "null"]},
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": ["string", "null"]},
                                    "kind": {"type": ["string", "null"]},
                                },
                            },
                        },
                    },
                },
            }
        },
    }
    expected = {"holdings": [{"security": "AAA", "tags": [{"name": "x", "kind": "a"}, {"name": "y", "kind": "b"}]}]}
    actual = {"holdings": [{"security": "AAA", "tags": [{"name": "x", "kind": "a"}, {"name": "WRONG", "kind": "b"}]}]}
    rules = [
        _rule("holdings[0].security", "AAA"),
        _rule("holdings[0].tags[0].name", "x"),
        _rule("holdings[0].tags[0].kind", "a"),
        _rule("holdings[0].tags[1].name", "y"),
        _rule("holdings[0].tags[1].kind", "b"),
    ]
    arr = ArrayRecordMatchMetric().compute(expected=expected, actual=actual, data_schema=schema)
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], schema)
    assert _val(arr, "array_record_recall") == 0.5  # tags is one opaque cell that mismatches
    assert _val(uni, "extract_unified_value_recall") == 0.8  # recursed: only tags[1].name wrong
    assert _val(uni, "extract_unified_value_recall") > _val(arr, "array_record_recall")


# ------------------------------------------------------------- grounding
def _bbox_rules(expected: dict[str, Any], page: int, bbox: list[float]) -> list[ExtractFieldTestRule]:
    rules = [_rule("as_of", expected.get("as_of"))]
    for i, row in enumerate(expected["holdings"]):
        for key in ("security", "coupon", "note"):
            rules.append(_rule(f"holdings[{i}].{key}", row.get(key), page=page, bbox=bbox))
    return rules


def test_grounded_requires_matching_bbox() -> None:
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    gt_box = [0.1, 0.1, 0.2, 0.05]
    rules = _bbox_rules(expected, page=1, bbox=gt_box)
    # Citation on the right page/box -> grounded passes.
    good_cit = [{"field_path": f"holdings[0].{k}", "page": 1, "bbox": gt_box} for k in ("security", "coupon", "note")]
    uni_good = compute_unified_evidence_metrics(expected, actual, rules, good_cit, _schema())
    assert _val(uni_good, "extract_unified_grounded_recall") == 1.0
    # Citation on the wrong page -> value still right, grounded fails.
    bad_cit = [{"field_path": f"holdings[0].{k}", "page": 2, "bbox": gt_box} for k in ("security", "coupon", "note")]
    uni_bad = compute_unified_evidence_metrics(expected, actual, rules, bad_cit, _schema())
    assert _val(uni_bad, "extract_unified_value_recall") == 1.0
    assert _val(uni_bad, "extract_unified_grounded_recall") == 0.0


def test_no_citations_yields_zero_grounded() -> None:
    # GT carries bboxes (grounding IS applicable) but the prediction emits no
    # citation: a real grounding miss -> grounded F1 is emitted as 0.0.
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    rules = _bbox_rules(expected, page=1, bbox=[0.1, 0.1, 0.2, 0.05])
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], _schema())
    assert _val(uni, "extract_unified_value_f1") == 1.0
    assert _val(uni, "extract_unified_grounded_f1") == 0.0


def test_no_gt_bbox_omits_grounded_metrics() -> None:
    # When the ground truth carries NO evidence bbox, grounding is undefined:
    # the *_grounded_* metrics are omitted entirely (so the runner excludes the
    # document from the grounded average) even if the prediction emits boxes.
    # The *_value_* metrics are unaffected.
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    rules = _leaf_rules_single(expected)  # no bbox on any GT evidence
    cits = [
        {"field_path": f"holdings[0].{k}", "page": 1, "bbox": [0.1, 0.1, 0.2, 0.05]}
        for k in ("security", "coupon", "note")
    ]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    assert _val(uni, "extract_unified_value_f1") == 1.0
    assert _val(uni, "extract_unified_grounded_f1") is None
    assert _val(uni, "extract_unified_grounded_precision") is None
    assert _val(uni, "extract_unified_grounded_recall") is None


def test_sparse_gt_bbox_does_not_punish_unannotated_claims() -> None:
    # Sparsely annotated GT: only ONE cell carries an evidence bbox while the
    # rest are value-only (the sec_13f shape: one cover field annotated, 30k
    # value-only cells). A pipeline that cites EVERY cell must not have its
    # grounded precision divided by all those ungradeable claims -- only the
    # claim on the bbox-bearing cell is gradeable. Correct grounding there
    # means grounded P/R/F1 == 1.0.
    expected = {"as_of": "2024-01-01", "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": "2024-01-01", "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    gt_box = [0.1, 0.1, 0.2, 0.05]
    rules = _leaf_rules_single(expected)  # value-only everywhere...
    rules.append(_rule("as_of", "2024-01-01", page=1, bbox=gt_box))  # ...except the one cover scalar
    cits = [{"field_path": "as_of", "page": 1, "bbox": gt_box}] + [
        {"field_path": f"holdings[0].{k}", "page": 1, "bbox": [0.5, 0.5, 0.1, 0.05]}
        for k in ("security", "coupon", "note")
    ]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    assert _val(uni, "extract_unified_grounded_recall") == 1.0
    assert _val(uni, "extract_unified_grounded_precision") == 1.0  # 3 unannotated claims excluded, not counted wrong
    assert _val(uni, "extract_unified_grounded_f1") == 1.0
    meta = next(m.metadata for m in uni if m.metric_name == "extract_unified_grounded_f1")
    assert meta["grounded_expected_cells"] == 1
    assert meta["grounded_pred_claims"] == 1


def test_extra_predicted_rows_claims_stay_out_of_grounded_precision() -> None:
    # An extra predicted row has no GT counterpart, so its citation bboxes are
    # ungradeable: they must not enter the grounded precision denominator (they
    # still hurt VALUE precision as always).
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {
        "as_of": None,
        "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}, {"security": "ZZZ", "coupon": 9.0, "note": "z"}],
    }
    gt_box = [0.1, 0.1, 0.2, 0.05]
    rules = _bbox_rules(expected, page=1, bbox=gt_box)
    cits = [{"field_path": f"holdings[{j}].{k}", "page": 1, "bbox": gt_box} for j in (0, 1) for k in ("security",)]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    meta = next(m.metadata for m in uni if m.metric_name == "extract_unified_grounded_f1")
    assert meta["grounded_pred_claims"] == 1  # only the claim aligned to GT row 0's bbox-bearing cell
    assert _val(uni, "extract_unified_grounded_precision") == 1.0
    assert _val(uni, "extract_unified_value_precision") < 1.0  # the extra row still costs value precision


# --------------------------------------------------------------- page grounding
def _page_rules(expected: dict[str, Any], page: int) -> list[ExtractFieldTestRule]:
    """One rule per leaf carrying a page but NO bbox (page-only grounding)."""
    rules = [_rule("as_of", expected.get("as_of"))]
    for i, row in enumerate(expected["holdings"]):
        for key in ("security", "coupon", "note"):
            rules.append(_rule(f"holdings[{i}].{key}", row.get(key), page=page))
    return rules


def test_page_correct_but_bbox_wrong_nests_between_value_and_grounded() -> None:
    # The core of the page family: a citation on the RIGHT page but with a
    # non-overlapping bbox is value-correct and page-correct, yet bbox-wrong.
    # So value_f1 == page_f1 == 1.0 while grounded_f1 == 0.0, and the three
    # nest: value_f1 >= page_f1 >= grounded_f1.
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    rules = _bbox_rules(expected, page=1, bbox=[0.1, 0.1, 0.2, 0.05])
    cits = [
        {"field_path": f"holdings[0].{k}", "page": 1, "bbox": [0.8, 0.8, 0.1, 0.05]}  # right page, disjoint box
        for k in ("security", "coupon", "note")
    ]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    vf1 = _val(uni, "extract_unified_value_f1")
    pf1 = _val(uni, "extract_unified_page_f1")
    gf1 = _val(uni, "extract_unified_grounded_f1")
    assert vf1 == 1.0
    assert pf1 == 1.0
    assert gf1 == 0.0
    assert vf1 >= pf1 >= gf1


def test_page_only_gt_emits_page_metrics_but_omits_grounded() -> None:
    # GT evidence carries a page but no bbox: the page family is defined (and
    # passes) while the bbox/grounded family is undefined and omitted -- page is
    # the coarser, more widely-applicable signal.
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    rules = _page_rules(expected, page=3)
    cits = [{"field_path": f"holdings[0].{k}", "page": 3} for k in ("security", "coupon", "note")]  # page-only cites
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    assert _val(uni, "extract_unified_page_recall") == 1.0
    assert _val(uni, "extract_unified_page_precision") == 1.0
    assert _val(uni, "extract_unified_page_f1") == 1.0
    assert _val(uni, "extract_unified_grounded_f1") is None  # no GT bbox -> grounded undefined
    meta = next(m.metadata for m in uni if m.metric_name == "extract_unified_page_f1")
    assert meta["page_expected_cells"] == 3
    assert meta["page_pred_claims"] == 3


def test_wrong_page_fails_page_metric() -> None:
    # Value right, but the citation names a page the GT evidence never claims:
    # page_recall drops to 0 while value_recall stays 1.
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    rules = _page_rules(expected, page=1)
    cits = [{"field_path": f"holdings[0].{k}", "page": 7} for k in ("security", "coupon", "note")]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_page_recall") == 0.0


def test_no_gt_page_omits_page_metrics() -> None:
    # When the GT carries NO page anywhere, page grounding is undefined: the
    # *_page_* metrics are omitted (excluded from the dataset average), mirroring
    # how *_grounded_* is omitted when the GT carries no bbox.
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    rules = _leaf_rules_single(expected)  # no page, no bbox on any GT evidence
    cits = [{"field_path": f"holdings[0].{k}", "page": 1} for k in ("security", "coupon", "note")]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    assert _val(uni, "extract_unified_value_f1") == 1.0
    assert _val(uni, "extract_unified_page_f1") is None
    assert _val(uni, "extract_unified_page_precision") is None
    assert _val(uni, "extract_unified_page_recall") is None


def test_nested_grounding_survives_outer_array_reorder() -> None:
    # Regression for the gt/pred path bug: when the outer object-array is
    # reordered, nested-record citations must be looked up at the *matched
    # predicted* index, not the GT index. With correct citations on the actual
    # prediction paths, grounded recall must stay 1.0.
    schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["string", "null"]},
                        "aliases": {
                            "type": "array",
                            "items": {"type": "object", "properties": {"name": {"type": ["string", "null"]}}},
                        },
                    },
                },
            }
        },
    }
    box_a = [0.1, 0.1, 0.2, 0.05]
    box_b = [0.5, 0.5, 0.2, 0.05]
    expected = {
        "entities": [
            {"id": "E1", "aliases": [{"name": "alpha"}]},
            {"id": "E2", "aliases": [{"name": "beta"}]},
        ]
    }
    actual = {"entities": list(reversed(expected["entities"]))}  # E2 first, E1 second
    rules = [
        ExtractFieldTestRule(field_path="entities[0].id", evidence=[FieldEvidence(value="E1", page=1, bbox=box_a)]),
        ExtractFieldTestRule(
            field_path="entities[0].aliases[0].name", evidence=[FieldEvidence(value="alpha", page=1, bbox=box_a)]
        ),
        ExtractFieldTestRule(field_path="entities[1].id", evidence=[FieldEvidence(value="E2", page=1, bbox=box_b)]),
        ExtractFieldTestRule(
            field_path="entities[1].aliases[0].name", evidence=[FieldEvidence(value="beta", page=1, bbox=box_b)]
        ),
    ]
    # Citations sit at the PREDICTED paths: entities[0]=E2 -> box_b, entities[1]=E1 -> box_a.
    cits = [
        {"field_path": "entities[0].id", "page": 1, "bbox": box_b},
        {"field_path": "entities[0].aliases[0].name", "page": 1, "bbox": box_b},
        {"field_path": "entities[1].id", "page": 1, "bbox": box_a},
        {"field_path": "entities[1].aliases[0].name", "page": 1, "bbox": box_a},
    ]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, schema)
    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_grounded_recall") == 1.0  # would be 0.5 with the gt/pred path bug


# --------------------------------------------------------------- degenerate
def test_non_dict_inputs_return_empty() -> None:
    assert compute_unified_evidence_metrics(["a"], {"x": 1}, [], [], {}) == []
    assert compute_unified_evidence_metrics({"x": 1}, "nope", [], [], {}) == []


# ------------------------------------------------------- evaluator wiring
def test_extract_evaluator_emits_unified_metrics() -> None:
    """End-to-end: the metric flows through ExtractEvaluator and matches array_record."""
    from datetime import datetime
    from pathlib import Path

    from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
    from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
    from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
    from extract_bench.schemas.product import ProductType
    from extract_bench.test_cases.schema import ExtractTestCase

    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    box = [0.1, 0.1, 0.2, 0.05]
    test_case = ExtractTestCase(
        test_id="longarray/example",
        group="longarray",
        file_path=Path("example.pdf"),
        data_schema=_schema(),
        expected_output=expected,
        test_rules=[r.model_dump() for r in _bbox_rules(expected, page=1, bbox=box)],
    )
    inference_result = InferenceResult(
        request=InferenceRequest(example_id="ex", source_file_path="example.pdf", product_type=ProductType.EXTRACT),
        pipeline_name="candidate",
        product_type=ProductType.EXTRACT,
        raw_output={},
        output=ExtractOutput(
            example_id="ex",
            pipeline_name="candidate",
            extracted_data=expected,
            field_citations=[
                FieldCitation(field_path=f"holdings[0].{k}", page=1, bbox=box) for k in ("security", "coupon", "note")
            ],
        ),
        started_at=datetime.now(),
        completed_at=datetime.now(),
        latency_in_ms=1,
    )
    metrics = {m.metric_name: m.value for m in ExtractEvaluator().evaluate(inference_result, test_case).metrics}
    assert metrics["extract_unified_value_f1"] == metrics["array_record_f1"] == 1.0
    assert metrics["extract_unified_grounded_recall"] == 1.0  # citations match the evidence boxes


# ----------------------------------------- exact-row peel is pairing-safe only
# The exact-row peel preserves array_record's value score (pairing-independent:
# correct == n_pairs*k - total_cost). But the unified score is *pairing-sensitive*
# whenever it recurses into nested sub-records or scores per-cell grounding, since
# both key off the matched predicted index. There the peel could select a
# different (equal-cost) pairing than the full assignment and shift grounded /
# nested true positives, so it must fall back to full assignment.
_FLAT_PEEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": ["string", "null"]},
                    "value": {"type": ["string", "null"]},
                },
            },
        }
    },
}
_NESTED_PEEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": ["string", "null"]},
                    "kids": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"v": {"type": ["string", "null"]}}},
                    },
                },
            },
        }
    },
}


def _spy_on_peel(monkeypatch: Any) -> list[list[str]]:
    """Record the cost-subfields each exact-row peel call aligned on."""
    calls: list[list[str]] = []
    original = unified_evidence_metric._peel_exact_row_matches

    def _spy(act_rows: Any, exp_rows: Any, subfields: Any, fuzzy: Any) -> Any:
        calls.append(list(subfields))
        return original(act_rows, exp_rows, subfields, fuzzy)

    monkeypatch.setattr(unified_evidence_metric, "_peel_exact_row_matches", _spy)
    return calls


def test_exact_peel_used_for_flat_ungrounded_arrays(monkeypatch: Any) -> None:
    calls = _spy_on_peel(monkeypatch)
    rows = [{"id": "1", "value": "a"}, {"id": "2", "value": "b"}]
    compute_unified_evidence_metrics({"rows": rows}, {"rows": rows}, [], [], _FLAT_PEEL_SCHEMA)
    assert ["id", "value"] in calls, "flat un-grounded arrays must keep the fast exact-row peel"


def test_exact_peel_skipped_for_nested_record_subfields(monkeypatch: Any) -> None:
    calls = _spy_on_peel(monkeypatch)
    rows = [{"id": "A", "kids": [{"v": "x"}]}, {"id": "B", "kids": [{"v": "y"}]}]
    compute_unified_evidence_metrics({"rows": rows}, {"rows": rows}, [], [], _NESTED_PEEL_SCHEMA)
    # The outer array (identity cell "id") has nested sub-records, so its
    # alignment must use full assignment -- the peel must not see ["id"].
    assert ["id"] not in calls, "outer array with sub-records must use full assignment"
    # The inner flat "kids" arrays are opaque-cell-only, so they still peel.
    assert ["v"] in calls, "flat inner sub-arrays should still use the fast peel"


def test_exact_peel_skipped_when_cell_grounding_present(monkeypatch: Any) -> None:
    calls = _spy_on_peel(monkeypatch)
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "object", "properties": {"id": {"type": ["string", "null"]}}},
            }
        },
    }
    rows = [{"id": "A"}, {"id": "B"}]
    box = [0.0, 0.0, 10.0, 10.0]
    rules = [
        _rule("rows[0].id", "A", page=1, bbox=box),
        _rule("rows[1].id", "B", page=1, bbox=box),
    ]
    citations = [
        {"field_path": "rows[0].id", "page": 1, "bbox": box},
        {"field_path": "rows[1].id", "page": 1, "bbox": box},
    ]
    compute_unified_evidence_metrics({"rows": rows}, {"rows": rows}, rules, citations, schema)
    assert calls == [], "per-cell grounding makes scoring pairing-sensitive; must use full assignment"


def test_nested_record_full_assignment_scores_perfect_match() -> None:
    # Sanity: the full-assignment fallback still scores a perfect nested match.
    rows = [{"id": "A", "kids": [{"v": "x"}]}, {"id": "B", "kids": [{"v": "y"}]}]
    uni = compute_unified_evidence_metrics({"rows": rows}, {"rows": rows}, [], [], _NESTED_PEEL_SCHEMA)
    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_value_precision") == 1.0


def _scalar_schema(field: str, typ: str = "string") -> dict[str, Any]:
    return {"type": "object", "properties": {field: {"type": [typ, "null"]}}}


def test_null_equals_false_normalizer_accepts_blank_checkbox_both_ways() -> None:
    schema = _scalar_schema("swr37_bw_hearing", "boolean")

    # GT False, model omitted the field entirely.
    omitted = compute_unified_evidence_metrics(
        {"swr37_bw_hearing": False},
        {},
        [_rule("swr37_bw_hearing", False, normalizers=["null_equals_false"])],
        [],
        schema,
    )
    # GT null, model asserted False.
    asserted = compute_unified_evidence_metrics(
        {"swr37_bw_hearing": None},
        {"swr37_bw_hearing": False},
        [_rule("swr37_bw_hearing", None, normalizers=["null_equals_false"])],
        [],
        schema,
    )
    strict = compute_unified_evidence_metrics(
        {"swr37_bw_hearing": False},
        {},
        [_rule("swr37_bw_hearing", False)],
        [],
        schema,
    )

    assert _val(omitted, "extract_unified_value_f1") == 1.0
    assert _val(asserted, "extract_unified_value_f1") == 1.0
    assert _val(strict, "extract_unified_value_f1") == 0.0


def test_null_equals_false_normalizer_rejects_true_and_zero() -> None:
    schema = _scalar_schema("swr37_bw_hearing", "boolean")

    true_vs_null = compute_unified_evidence_metrics(
        {"swr37_bw_hearing": True},
        {},
        [_rule("swr37_bw_hearing", True, normalizers=["null_equals_false"])],
        [],
        schema,
    )
    zero_vs_false = compute_unified_evidence_metrics(
        {"swr37_bw_hearing": False},
        {"swr37_bw_hearing": 0},
        [_rule("swr37_bw_hearing", False, normalizers=["null_equals_false"])],
        [],
        schema,
    )

    assert _val(true_vs_null, "extract_unified_value_f1") == 0.0
    # 0 == False in Python; the normalizer must not launder ints, but the base
    # ``==`` cell match already treats 0 as False, so this stays a pass there.
    assert _val(zero_vs_false, "extract_unified_value_f1") == 1.0


def test_case_insensitive_normalizer_is_field_scoped() -> None:
    schema = _scalar_schema("time_point")

    strict = compute_unified_evidence_metrics(
        {"time_point": "INITIAL"},
        {"time_point": "Initial"},
        [_rule("time_point", "INITIAL")],
        [],
        schema,
    )
    normalized = compute_unified_evidence_metrics(
        {"time_point": "INITIAL"},
        {"time_point": "Initial"},
        [_rule("time_point", "INITIAL", normalizers=["case_insensitive"])],
        [],
        schema,
    )

    assert _val(strict, "extract_unified_value_f1") == 0.0
    assert _val(normalized, "extract_unified_value_f1") == 1.0


def test_phone_digits_normalizer_matches_formatting_variants_only() -> None:
    schema = _scalar_schema("phone")
    rule = [_rule("phone", "( 713 ) 372-2430", normalizers=["phone_digits"])]

    same_digits = compute_unified_evidence_metrics(
        {"phone": "( 713 ) 372-2430"}, {"phone": "713-372-2430"}, rule, [], schema
    )
    other_digits = compute_unified_evidence_metrics(
        {"phone": "( 713 ) 372-2430"}, {"phone": "(713) 372-2431"}, rule, [], schema
    )
    area_code_only = compute_unified_evidence_metrics(
        {"phone": "( 512 )"},
        {"phone": "512"},
        [_rule("phone", "( 512 )", normalizers=["phone_digits"])],
        [],
        schema,
    )

    assert _val(same_digits, "extract_unified_value_f1") == 1.0
    assert _val(other_digits, "extract_unified_value_f1") == 0.0
    assert _val(area_code_only, "extract_unified_value_f1") == 1.0


def test_lenient_date_normalizer_joins_split_preprinted_years() -> None:
    schema = _scalar_schema("p2_date_well_plugged")
    rule = [_rule("p2_date_well_plugged", "1955-05-11", normalizers=["lenient_date"])]

    split_year = compute_unified_evidence_metrics(
        {"p2_date_well_plugged": "1955-05-11"}, {"p2_date_well_plugged": "May 11 , 19 55"}, rule, [], schema
    )
    two_digit_year = compute_unified_evidence_metrics(
        {"p2_date_well_plugged": "1955-05-11"}, {"p2_date_well_plugged": "May 11, 55"}, rule, [], schema
    )
    wrong_day = compute_unified_evidence_metrics(
        {"p2_date_well_plugged": "1955-05-11"}, {"p2_date_well_plugged": "May 12, 19 55"}, rule, [], schema
    )

    assert _val(split_year, "extract_unified_value_f1") == 1.0
    assert _val(two_digit_year, "extract_unified_value_f1") == 1.0
    assert _val(wrong_day, "extract_unified_value_f1") == 0.0


def test_lenient_date_normalizer_does_not_equate_different_values() -> None:
    """Rewrites are guarded: no century expansion without a day, no split-year
    join without a preceding day, and unparseable rewrites never match."""
    schema = _scalar_schema("d")

    def _f1(gt: str, pred: str) -> float | None:
        rule = [_rule("d", gt, normalizers=["lenient_date"])]
        return _val(
            compute_unified_evidence_metrics({"d": gt}, {"d": pred}, rule, [], schema), "extract_unified_value_f1"
        )

    # A day-only value must not be read as a 2-digit year ("May 11" != May 2011).
    assert _f1("May 11", "May 2011") == 0.0
    # "19" here is the day, not a split decade: "June 19 44" is June 19, 1944.
    assert _f1("1944-06-19", "June 19 44") == 1.0
    assert _f1("June 19 44", "June 1944") == 0.0
    # Non-date text must never match through a fabricated rewrite.
    assert _f1("Permit 12-34", "Permit 12-19 34") == 0.0


def test_optional_terminal_punctuation_normalizer_strips_one_char_only() -> None:
    schema = _scalar_schema("f")

    def _f1(gt: str, pred: str) -> float | None:
        rule = [_rule("f", gt, normalizers=["optional_terminal_punctuation"])]
        return _val(
            compute_unified_evidence_metrics({"f": gt}, {"f": pred}, rule, [], schema), "extract_unified_value_f1"
        )

    assert _f1("item 5.", "item 5") == 1.0
    # Only ONE terminal char is optional; a punctuation run is real content.
    assert _f1("item 5.", "item 5;,,.") == 0.0
    # Punctuation-only values must not collapse to '' and match each other.
    assert _f1(".", ";") == 0.0


def test_punctuation_spacing_normalizer_is_opt_in() -> None:
    schema = _scalar_schema("address")
    gt = {"address": "P.O. Box 978"}
    pred = {"address": "P.O.Box 978"}

    strict = compute_unified_evidence_metrics(gt, pred, [_rule("address", "P.O. Box 978")], [], schema)
    lenient = compute_unified_evidence_metrics(
        gt, pred, [_rule("address", "P.O. Box 978", normalizers=["punctuation_spacing"])], [], schema
    )
    different = compute_unified_evidence_metrics(
        gt,
        {"address": "P.O. Box 979"},
        [_rule("address", "P.O. Box 978", normalizers=["punctuation_spacing"])],
        [],
        schema,
    )

    assert _val(strict, "extract_unified_value_f1") == 0.0
    assert _val(lenient, "extract_unified_value_f1") == 1.0
    assert _val(different, "extract_unified_value_f1") == 0.0


def test_omitted_scalar_counts_as_implicit_null_prediction_right_or_wrong() -> None:
    """Omission is scored exactly like an explicit null: it always enters the
    precision denominator, so dropping an uncertain field cannot outscore
    asserting it."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": ["string", "null"]}, "b": {"type": ["string", "null"]}},
    }
    rules = [_rule("a", "x"), _rule("b", None)]

    omitted = compute_unified_evidence_metrics({"a": "x", "b": None}, {}, rules, [], schema)
    explicit = compute_unified_evidence_metrics({"a": "x", "b": None}, {"a": None, "b": None}, rules, [], schema)

    assert _val(omitted, "extract_unified_value_precision") == 0.5
    assert _val(omitted, "extract_unified_value_precision") == _val(explicit, "extract_unified_value_precision")
    assert _val(omitted, "extract_unified_value_recall") == _val(explicit, "extract_unified_value_recall")


def test_supported_normalizers_match_schema_vocabulary() -> None:
    from extract_bench.test_cases.schema import EXTRACT_FIELD_NORMALIZERS

    assert unified_evidence_metric.SUPPORTED_NORMALIZERS == EXTRACT_FIELD_NORMALIZERS


def test_giant_grounded_array_skips_grounding_value_exact_and_peels(monkeypatch: Any) -> None:
    """A grounded flat array over the cell threshold: value stays bit-exact, the
    grounded metrics are withheld (grounded_incomplete), and it takes the peel
    instead of building the multi-GB grounded matrix. Nothing below the
    threshold changes. This is the memory escape hatch for oklahoma-scale docs.
    """
    box = [0.0, 0.0, 10.0, 10.0]
    expected = {
        "as_of": None,
        "holdings": [{"security": f"S{i}", "coupon": float(i), "note": f"n{i}"} for i in range(6)],
    }
    actual = {  # one reorder + one wrong cell so value scoring is non-trivial
        "as_of": None,
        "holdings": [{"security": f"S{i}", "coupon": float(i), "note": f"n{i}"} for i in (1, 0, 2, 3, 4, 5)],
    }
    actual["holdings"][2]["coupon"] = 999.0
    rules = _bbox_rules(expected, page=1, bbox=box)
    cits = [
        {"field_path": f"holdings[{i}].{k}", "page": 1, "bbox": box}
        for i in range(6)
        for k in ("security", "coupon", "note")
    ]

    # Full-matrix reference (skip disabled): grounding present, value computed.
    full = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    assert _val(full, "extract_unified_grounded_f1") is not None

    # Force the skip by lowering the threshold below this array's cell count
    # (6 GT x 6 pred = 36), and confirm it takes the peel, not the full matrix.
    monkeypatch.setattr(unified_evidence_metric, "_GROUNDED_MAX_CELLS", 1)
    calls = _spy_on_peel(monkeypatch)
    skipped = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())

    assert ["security", "coupon", "note"] in calls, "giant grounded array must fall back to the peel"
    # Value metrics identical to the full-matrix reference.
    for name in (
        "extract_unified_value_precision",
        "extract_unified_value_recall",
        "extract_unified_value_f1",
    ):
        assert _val(skipped, name) == _val(full, name), name
    # Grounded metrics withheld entirely for the document.
    assert _val(skipped, "extract_unified_grounded_f1") is None
    assert _val(skipped, "extract_unified_grounded_precision") is None
    assert _val(skipped, "extract_unified_grounded_recall") is None
    # And the value metadata records why.
    vf1 = next(m for m in skipped if m.metric_name == "extract_unified_value_f1")
    assert vf1.metadata["grounded_incomplete"] is True


def test_giant_threshold_does_not_touch_nested_or_below_threshold(monkeypatch: Any) -> None:
    """The skip must not fire for arrays with sub-records (peel would shift
    nested TP), nor below the threshold."""
    box = [0.0, 0.0, 10.0, 10.0]
    expected = {"as_of": None, "holdings": [{"security": "A", "coupon": 1.0, "note": "n"}]}
    rules = _bbox_rules(expected, page=1, bbox=box)
    cits = [{"field_path": f"holdings[0].{k}", "page": 1, "bbox": box} for k in ("security", "coupon", "note")]
    # Below threshold (default 100M): grounding kept.
    kept = compute_unified_evidence_metrics(expected, expected, rules, cits, _schema())
    assert _val(kept, "extract_unified_grounded_f1") == 1.0
    assert next(m for m in kept if m.metric_name == "extract_unified_value_f1").metadata["grounded_incomplete"] is False

    # A nested-record array over the threshold must NOT skip (stays grounded via
    # full assignment) because the peel there could shift nested TP.
    monkeypatch.setattr(unified_evidence_metric, "_GROUNDED_MAX_CELLS", 1)
    nested = {"rows": [{"id": "A", "kids": [{"v": "x"}]}]}
    nrules = [
        _rule("rows[0].id", "A", page=1, bbox=box),
        _rule("rows[0].kids[0].v", "x", page=1, bbox=box),
    ]
    ncits = [
        {"field_path": "rows[0].id", "page": 1, "bbox": box},
        {"field_path": "rows[0].kids[0].v", "page": 1, "bbox": box},
    ]
    out = compute_unified_evidence_metrics(nested, nested, nrules, ncits, _NESTED_PEEL_SCHEMA)
    # Outer array has sub-records -> not eligible for the skip -> grounding kept.
    assert _val(out, "extract_unified_grounded_f1") == 1.0
