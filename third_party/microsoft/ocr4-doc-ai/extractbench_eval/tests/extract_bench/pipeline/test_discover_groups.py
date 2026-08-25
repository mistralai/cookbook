"""Contract for evaluation-group discovery.

A category whose documents all fail writes no result file. If groups come only
from the results tree, that category never becomes a group, never gets a report,
and reaches the leaderboard as "no data" rather than zero — so a pipeline that
fails an entire split outscores one that merely does badly on it. The dataset is
therefore the authority on which groups should have been scored.
"""

from __future__ import annotations

from pathlib import Path

from extract_bench.pipeline.cli import _discover_groups


def _result(directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.result.json").write_text("{}")


def _case(directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.test.json").write_text("{}")


def test_groups_come_from_results_when_no_dataset_is_given(tmp_path: Path):
    out = tmp_path / "out"
    _result(out / "short", "a")
    _result(out / "medium", "b")
    assert _discover_groups(out) == ["medium", "short"]


def test_a_totally_failed_category_is_still_a_group(tmp_path: Path):
    """The regression: `long` produced no results, so it must come from the dataset."""
    out = tmp_path / "out"
    data = tmp_path / "data"
    _result(out / "short", "a")
    _result(out / "medium", "b")
    _case(data / "short", "a")
    _case(data / "medium", "b")
    _case(data / "long", "c")  # every long document errored -> no result file

    assert _discover_groups(out, data) == ["long", "medium", "short"]


def test_dataset_and_results_are_unioned(tmp_path: Path):
    """A group present only in results is kept — the dataset does not mask it."""
    out = tmp_path / "out"
    data = tmp_path / "data"
    _result(out / "extra", "a")
    _case(data / "short", "b")
    assert _discover_groups(out, data) == ["extra", "short"]


def test_loose_files_at_the_root_are_not_groups(tmp_path: Path):
    out = tmp_path / "out"
    data = tmp_path / "data"
    out.mkdir()
    data.mkdir()
    (out / "x.result.json").write_text("{}")
    (data / "x.test.json").write_text("{}")
    assert _discover_groups(out, data) == []
