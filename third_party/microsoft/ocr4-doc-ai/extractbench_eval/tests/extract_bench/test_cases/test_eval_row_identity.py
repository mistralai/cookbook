"""Contract for the eval-only row-identity channel (`_eval_row_identity`).

Row identity used to live in `data_schema.repeated_structure`, which shipped to
every extract provider. It now rides beside the schema and is re-attached for
scoring only. Two properties keep that safe, and both can regress silently:

  1. Absent key  -> `eval_data_schema` IS `data_schema` (unmigrated datasets,
     whose sidecars may carry an inert legacy `_repeated_structure`, cannot be
     affected at all).
  2. Present key -> the block is visible to alignment as if it had never left
     the schema, WITHOUT mutating the schema that goes to providers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from extract_bench.evaluation.metrics.extract.confidence_scoped.summary import (
    compute_confidence_scoped_metrics,
)
from extract_bench.schemas.extract_output import FieldCitation
from extract_bench.test_cases.loader import load_test_cases
from extract_bench.test_cases.schema import ExtractTestCase

ROW = {
    "type": "object",
    "properties": {
        "cusip": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "CUSIP"},
        "amount": {"anyOf": [{"type": "number"}, {"type": "null"}], "description": "Amount"},
    },
}
SCHEMA = {
    "type": "object",
    "title": "T",
    "$defs": {"Row": ROW},
    "properties": {
        "holdings": {
            "anyOf": [{"type": "array", "items": {"$ref": "#/$defs/Row"}}, {"type": "null"}],
            "description": "Holdings",
        }
    },
}
IDENTITY = {"holdings": {"identity_key": "cusip"}}


def _case(**kwargs) -> ExtractTestCase:
    return ExtractTestCase(test_id="t", group="g", file_path=Path("/tmp/x.pdf"), **kwargs)


# --------------------------------------------------------------------------
# eval_data_schema
# --------------------------------------------------------------------------


def test_absent_key_returns_the_same_object():
    """Identity, not equality — an unmigrated dataset must be untouchable."""
    tc = _case(schema=SCHEMA)
    assert tc.eval_data_schema is tc.data_schema


def test_legacy_repeated_structure_sidecar_is_ignored():
    """Legacy sidecars carry this key inertly; honoring it would move scores."""
    tc = _case(schema=SCHEMA, **{"_repeated_structure": IDENTITY})
    assert tc.eval_data_schema is tc.data_schema
    assert "repeated_structure" not in tc.eval_data_schema


def test_present_key_reattaches_without_mutating_shipped_schema():
    tc = _case(schema=SCHEMA, **{"_eval_row_identity": IDENTITY})
    assert tc.eval_data_schema["repeated_structure"] == IDENTITY
    # what providers receive stays plain JSON Schema
    assert "repeated_structure" not in tc.data_schema
    assert tc.eval_data_schema["properties"] == SCHEMA["properties"]


# --------------------------------------------------------------------------
# loader pass-through
# --------------------------------------------------------------------------


def _write_case(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "doc.pdf").touch()
    (directory / "doc.test.json").write_text(json.dumps(payload))


def test_loader_carries_the_key(tmp_path: Path):
    _write_case(
        tmp_path / "grp",
        {
            "data_schema": SCHEMA,
            "expected_output": {"holdings": []},
            "_eval_row_identity": IDENTITY,
        },
    )
    (tc,) = load_test_cases(str(tmp_path), product_type="extract")
    assert tc.row_identity == IDENTITY
    assert tc.eval_data_schema["repeated_structure"] == IDENTITY


def test_loader_warns_on_a_misnamed_legacy_block(tmp_path: Path, caplog):
    """A declaration under the legacy name is ignored, but never in silence.

    Honoring `_repeated_structure` would activate every inert legacy block, so
    the block stays ignored; the warning is what keeps a misnamed key from
    dropping identity without a trace.
    """
    _write_case(
        tmp_path / "grp",
        {"data_schema": SCHEMA, "expected_output": {"holdings": []}, "_repeated_structure": IDENTITY},
    )
    with caplog.at_level("WARNING"):
        (tc,) = load_test_cases(str(tmp_path), product_type="extract")
    assert tc.row_identity is None
    assert "repeated_structure" not in tc.eval_data_schema
    assert "_eval_row_identity" in caplog.text


def test_loader_stays_quiet_when_the_legacy_block_is_empty(tmp_path: Path, caplog):
    _write_case(
        tmp_path / "grp",
        {"data_schema": SCHEMA, "expected_output": {"holdings": []}, "_repeated_structure": {}},
    )
    with caplog.at_level("WARNING"):
        load_test_cases(str(tmp_path), product_type="extract")
    assert "_repeated_structure" not in caplog.text


def test_loader_rejects_a_malformed_block(tmp_path: Path):
    """A typo'd block must not silently drop identity."""
    _write_case(
        tmp_path / "grp",
        {"data_schema": SCHEMA, "expected_output": {}, "_eval_row_identity": ["cusip"]},
    )
    with pytest.raises(ValueError, match="_eval_row_identity must be an object"):
        load_test_cases(str(tmp_path), product_type="extract")


# --------------------------------------------------------------------------
# end-to-end equivalence on a DUPLICATE-identity doc
# --------------------------------------------------------------------------

# Duplicated identity tuples are the case that separates the two channels: a
# schema-declared identity_key is authoritative, while the `match_by:` rule path
# is gated on global uniqueness and refuses to join. This mirrors real GT docs
# whose arrays repeat the same identity tuple across many rows.
DUPE_GT = {
    "holdings": [
        {"cusip": "AAA", "amount": 1},
        {"cusip": "AAA", "amount": 2},
        {"cusip": "BBB", "amount": 3},
    ]
}


def _score(schema: dict, identity: dict | None) -> dict[str, float]:
    tc = _case(
        schema=schema,
        expected_output=DUPE_GT,
        **({"_eval_row_identity": identity} if identity else {}),
    )
    predicted = {"holdings": list(reversed(DUPE_GT["holdings"]))}
    citations = [
        FieldCitation(field_path=f"holdings[{i}].{cell}", page=1, confidence=0.9, source="t")
        for i in range(len(predicted["holdings"]))
        for cell in ("cusip", "amount")
    ]
    metrics = compute_confidence_scoped_metrics(
        extracted_data=predicted,
        field_rules=[],
        field_citations=citations,
        data_schema=tc.eval_data_schema,
        skip_field_paths=set(),
        expected_output=DUPE_GT,
    )
    return {m.metric_name: m.value for m in metrics}


def test_sidecar_scores_identically_to_the_old_inline_block():
    """The relocation must be a no-op for scoring, duplicates included."""
    inline = _score({**SCHEMA, "repeated_structure": IDENTITY}, None)
    sidecar = _score(SCHEMA, IDENTITY)
    assert inline == sidecar
    assert inline, "expected the confidence metrics to actually emit"
