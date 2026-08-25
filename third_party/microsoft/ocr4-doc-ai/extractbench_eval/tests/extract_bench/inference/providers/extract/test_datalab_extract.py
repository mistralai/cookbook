from __future__ import annotations

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")
pytest.importorskip("datalab_sdk", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

from extract_bench.inference.providers.extract.datalab import (
    _build_field_citations,
    _strip_citation_and_score_keys,
)


def _meta() -> dict:
    return {
        "extraction_status": "EXTRACTED",
        "reasoning": None,
        "citations": ["/page/0/Table/7"],
        "verification": {"status": "PASS", "feedback": "ok"},
    }


def test_strip_removes_meta_alongside_citations_and_score() -> None:
    payload = {
        "invoice_number": "NF67652",
        "invoice_number_meta": _meta(),
        "invoice_number_citations": ["/page/0/Table/7"],
        "total": 1382.0,
        "total_meta": _meta(),
        "total_score": {"score": 5},
    }

    assert _strip_citation_and_score_keys(payload) == {
        "invoice_number": "NF67652",
        "total": 1382.0,
    }


def test_strip_removes_nested_meta_keys() -> None:
    payload = {
        "line_items": [
            {
                "description": "widget",
                "description_meta": _meta(),
                "amount": 10.0,
                "amount_meta": _meta(),
            }
        ],
        "vendor": {"name": "ACME", "name_meta": _meta()},
    }

    assert _strip_citation_and_score_keys(payload) == {
        "line_items": [{"description": "widget", "amount": 10.0}],
        "vendor": {"name": "ACME"},
    }


def test_meta_sidecar_produces_no_citations() -> None:
    payload = {"invoice_number": "NF67652", "invoice_number_meta": _meta()}

    assert _build_field_citations(payload, block_lookup={}) == []
