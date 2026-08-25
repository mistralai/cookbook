"""Tests for the ``--test`` flag's data-directory routing.

``extract-bench run <pipeline> --test`` and ``extract-bench download --test``
route to ``./data/test``; the full dataset lives at ``./data``. The two
locations must never collide.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from extract_bench.data.cli import DataCLI
from extract_bench.data.download import (
    DEFAULT_DATA_DIR,
    DEFAULT_TEST_DATA_DIR,
    MANIFEST_NAME,
    _prune_orphans,
    _revision_is_stale,
    default_data_dir,
)


class TestDefaultDataDir:
    def test_full_dataset_default(self) -> None:
        assert default_data_dir(False) == Path("./data")
        assert default_data_dir(False) == DEFAULT_DATA_DIR

    def test_test_dataset_default(self) -> None:
        assert default_data_dir(True) == Path("./data/test")
        assert default_data_dir(True) == DEFAULT_TEST_DATA_DIR

    def test_full_and_test_paths_diverge(self) -> None:
        assert default_data_dir(False) != default_data_dir(True)


class TestDownloadRouting:
    def test_download_test_routes_to_test_subdir(self) -> None:
        cli = DataCLI()
        with patch("extract_bench.data.cli.download_dataset") as mock_dl:
            cli.download(test=True)
        mock_dl.assert_called_once()
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["data_dir"] == DEFAULT_TEST_DATA_DIR
        assert kwargs["test"] is True

    def test_download_default_routes_to_full_dir(self) -> None:
        cli = DataCLI()
        with patch("extract_bench.data.cli.download_dataset") as mock_dl:
            cli.download()
        mock_dl.assert_called_once()
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["data_dir"] == DEFAULT_DATA_DIR
        assert kwargs["test"] is False

    def test_explicit_data_dir_is_respected(self, tmp_path: Path) -> None:
        # Explicit --data_dir overrides the --test default.
        cli = DataCLI()
        explicit = tmp_path / "elsewhere"
        with patch("extract_bench.data.cli.download_dataset") as mock_dl:
            cli.download(data_dir=str(explicit), test=True)
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["data_dir"] == explicit


class TestStatusRouting:
    def test_status_test_flag_checks_test_subdir(self) -> None:
        # --test must route status to the test subset directory, matching the
        # relative-default semantics used by download.
        cli = DataCLI()
        with patch("extract_bench.data.cli.is_dataset_ready", return_value=False) as mock_ready:
            rc = cli.status(test=True)
        # Status returns 1 when not ready; we only care about which path it checked.
        assert rc == 1
        checked_path = mock_ready.call_args.args[0]
        assert checked_path == DEFAULT_TEST_DATA_DIR


@pytest.mark.parametrize(
    "test_flag,expected_relative",
    [(False, Path("./data")), (True, Path("./data/test"))],
)
def test_pipeline_run_input_dir_routing(test_flag: bool, expected_relative: Path) -> None:
    """The pipeline runner must default ``input_dir`` based on ``--test``.

    We mock the heavy machinery (download, inference, evaluation, analysis)
    and only inspect the ``input_dir`` that gets propagated to
    ``InferenceCLI.run``.
    """
    from extract_bench.pipeline.cli import PipelineCLI

    cli = PipelineCLI()
    with (
        patch("extract_bench.pipeline.cli.is_dataset_ready", return_value=True),
        patch("extract_bench.pipeline.cli.InferenceCLI") as mock_inf_cls,
        patch.object(cli, "_run_multi_group_evaluation", return_value=0),
    ):
        mock_inf = mock_inf_cls.return_value
        mock_inf.run.return_value = 0
        rc = cli.run(pipeline="dummy", test=test_flag)

    assert rc == 0
    # InferenceCLI.run receives the resolved input_dir; assert routing works.
    inf_kwargs = mock_inf.run.call_args.kwargs
    assert inf_kwargs["input_dir"] == expected_relative


@pytest.mark.parametrize(
    "test_flag,expected_relative",
    [(False, Path("./data")), (True, Path("./data/test"))],
)
def test_pipeline_run_auto_download_routing(test_flag: bool, expected_relative: Path) -> None:
    """When the dataset isn't on disk, ``pipeline.run`` must auto-download to
    the test-routed path — not silently to ``./data``.

    This is the second half of the original bug: even after fixing the
    ``input_dir`` default, a wrong download target would re-introduce the
    overlay/masking problem on a fresh machine.
    """
    from extract_bench.pipeline.cli import PipelineCLI

    cli = PipelineCLI()
    with (
        patch("extract_bench.pipeline.cli.is_dataset_ready", return_value=False),
        patch("extract_bench.pipeline.cli.download_dataset") as mock_dl,
        patch("extract_bench.pipeline.cli.InferenceCLI") as mock_inf_cls,
        patch.object(cli, "_run_multi_group_evaluation", return_value=0),
    ):
        mock_inf = mock_inf_cls.return_value
        mock_inf.run.return_value = 0
        rc = cli.run(pipeline="dummy", test=test_flag)

    assert rc == 0
    mock_dl.assert_called_once()
    dl_kwargs = mock_dl.call_args.kwargs
    assert dl_kwargs["data_dir"] == expected_relative
    assert dl_kwargs["test"] is test_flag


class TestRevisionStaleness:
    """A branch can swap documents without changing per-split counts."""

    @staticmethod
    def _seed(data_dir: Path, revision: str, stems: dict[str, list[str]]) -> None:
        import json

        for split, names in stems.items():
            (data_dir / split).mkdir(parents=True, exist_ok=True)
            for stem in names:
                (data_dir / split / f"{stem}.test.json").write_text("{}", encoding="utf-8")
                (data_dir / split / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n")
        (data_dir / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "repo": "llamaindex/ExtractBench",
                    "revision": revision,
                    "test": True,
                    "cases": sum(len(v) for v in stems.values()),
                    "per_split": {s: len(v) for s, v in stems.items()},
                }
            ),
            encoding="utf-8",
        )

    def test_same_counts_different_revision_is_stale(self, tmp_path: Path) -> None:
        self._seed(tmp_path, "old-sha", {"short": ["a"], "medium": ["b"], "long": ["c"]})
        with patch("extract_bench.data.download._remote_revision", return_value="new-sha"):
            assert _revision_is_stale(tmp_path, "test-data") is True

    def test_matching_revision_is_not_stale(self, tmp_path: Path) -> None:
        self._seed(tmp_path, "same-sha", {"short": ["a"], "medium": ["b"], "long": ["c"]})
        with patch("extract_bench.data.download._remote_revision", return_value="same-sha"):
            assert _revision_is_stale(tmp_path, "test-data") is False

    def test_offline_keeps_existing_download(self, tmp_path: Path) -> None:
        self._seed(tmp_path, "old-sha", {"short": ["a"], "medium": ["b"], "long": ["c"]})
        with patch("extract_bench.data.download._remote_revision", return_value=None):
            assert _revision_is_stale(tmp_path, "test-data") is False

    def test_prune_removes_documents_no_longer_shipped(self, tmp_path: Path) -> None:
        self._seed(tmp_path, "old-sha", {"short": ["keep", "drop"], "medium": ["b"], "long": ["c"]})
        keep = {
            tmp_path / "short" / "keep.test.json",
            tmp_path / "medium" / "b.test.json",
            tmp_path / "long" / "c.test.json",
        }
        removed = _prune_orphans(tmp_path, keep)
        assert removed == [tmp_path / "short" / "drop.test.json"]
        assert not (tmp_path / "short" / "drop.pdf").exists()
        assert (tmp_path / "short" / "keep.pdf").exists()
