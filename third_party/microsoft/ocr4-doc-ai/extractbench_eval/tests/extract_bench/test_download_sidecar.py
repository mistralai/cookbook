"""`extract-bench download` must land the `repeated_structure` column under
`_eval_row_identity` — the only key the loader reads. Writing it to the legacy
`_repeated_structure` name would leave row identity dormant in every
reconstructed dataset (the loader deliberately ignores that key)."""

from __future__ import annotations

import json
from pathlib import Path

from extract_bench.data.download import _write_sidecar

SCHEMA = {
    "type": "object",
    "properties": {"holdings": {"type": "array", "items": {"type": "object"}}},
}
IDENTITY = {"holdings": {"identity_key": "cusip"}}


def _row(tmp_path: Path, repeated: dict | None) -> tuple[dict, Path]:
    snapshot_dir = tmp_path / "snapshot"
    pdf = snapshot_dir / "docs" / "short" / "doc.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")
    row = {
        "id": "doc",
        "category": "short",
        "pdf": "docs/short/doc.pdf",
        "data_schema": json.dumps(SCHEMA),
        "expected_output": json.dumps({"holdings": []}),
        "field_rules": "{}",
        "repeated_structure": json.dumps(repeated or {}),
        "tags": [],
    }
    return row, snapshot_dir


def test_nonempty_column_lands_as_eval_row_identity(tmp_path: Path):
    row, snapshot_dir = _row(tmp_path, IDENTITY)
    sidecar_path = _write_sidecar(row, tmp_path / "short", snapshot_dir)
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["_eval_row_identity"] == IDENTITY
    assert "_repeated_structure" not in sidecar
    assert "repeated_structure" not in sidecar["data_schema"]


def test_empty_column_writes_no_identity_key(tmp_path: Path):
    row, snapshot_dir = _row(tmp_path, None)
    sidecar_path = _write_sidecar(row, tmp_path / "short", snapshot_dir)
    sidecar = json.loads(sidecar_path.read_text())
    assert "_eval_row_identity" not in sidecar
    assert "_repeated_structure" not in sidecar
