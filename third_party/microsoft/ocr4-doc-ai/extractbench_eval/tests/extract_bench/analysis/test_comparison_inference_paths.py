"""Tests for comparison inference result path resolution."""

import json
from pathlib import Path

from extract_bench.analysis.comparison_core import (
    inference_result_candidate_paths,
    load_inference_result,
)


def _write_result(path: Path, example_id: str = "doc") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "request": {"source_file_path": f"{example_id}.pdf"},
                "pipeline_name": "test",
                "product_type": "extract",
                "raw_output": {},
                "output": {
                    "task_type": "extract",
                    "example_id": example_id,
                    "pipeline_name": "test",
                    "extracted_data": {"field": "value"},
                },
                "started_at": "2026-01-01T00:00:00Z",
                "completed_at": "2026-01-01T00:00:01Z",
                "latency_in_ms": 1,
            }
        ),
        encoding="utf-8",
    )


def test_load_inference_result_from_pipeline_root(tmp_path: Path) -> None:
    result_path = tmp_path / "short" / "bianco-2024.result.json"
    _write_result(result_path, "bianco-2024")

    loaded = load_inference_result(tmp_path, "short/bianco-2024")
    assert loaded is not None
    assert loaded["output"]["extracted_data"] == {"field": "value"}


def test_load_inference_result_from_group_directory(tmp_path: Path) -> None:
    """Per-group compares use .../short as the pipeline dir with test_id short/<stem>."""
    group_dir = tmp_path / "short"
    _write_result(group_dir / "bianco-2024.result.json", "bianco-2024")

    loaded = load_inference_result(group_dir, "short/bianco-2024")
    assert loaded is not None
    assert loaded["output"]["extracted_data"] == {"field": "value"}


def test_inference_result_candidate_paths_include_basename(tmp_path: Path) -> None:
    paths = inference_result_candidate_paths(tmp_path / "short", "short/bianco-2024")
    assert tmp_path / "short" / "bianco-2024.result.json" in paths
    assert tmp_path / "short" / "short" / "bianco-2024.result.json" in paths
