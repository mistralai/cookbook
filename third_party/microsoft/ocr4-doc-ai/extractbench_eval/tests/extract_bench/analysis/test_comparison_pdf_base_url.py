"""Tests for comparison report PDF base URL resolution."""

from pathlib import Path

from extract_bench.analysis.comparison_report import resolve_comparison_pdf_base_url


def test_resolve_comparison_pdf_base_url_relative_path(tmp_path: Path) -> None:
    test_dir = tmp_path / "data" / "short"
    test_dir.mkdir(parents=True)
    report = tmp_path / "output" / "comparison.html"
    report.parent.mkdir()

    url = resolve_comparison_pdf_base_url(test_dir, report)
    assert url == "../data/short"


def test_resolve_comparison_pdf_base_url_keeps_test_cases_root(tmp_path: Path) -> None:
    """input_file_rel is relative to test_cases_dir, so do not descend into pdfs/."""
    test_dir = tmp_path / "data" / "short"
    (test_dir / "pdfs").mkdir(parents=True)
    report = tmp_path / "output" / "comparison.html"
    report.parent.mkdir()

    url = resolve_comparison_pdf_base_url(test_dir, report)
    assert url == "../data/short"


def test_resolve_comparison_pdf_base_url_explicit_override(tmp_path: Path) -> None:
    test_dir = tmp_path / "data" / "short"
    test_dir.mkdir(parents=True)
    report = tmp_path / "comparison.html"

    url = resolve_comparison_pdf_base_url(
        test_dir,
        report,
        explicit_url="http://localhost:8080/data/short/",
    )
    assert url == "http://localhost:8080/data/short"


def test_resolve_comparison_pdf_base_url_missing_test_dir(tmp_path: Path) -> None:
    report = tmp_path / "comparison.html"
    url = resolve_comparison_pdf_base_url(None, report)
    assert url == ""
