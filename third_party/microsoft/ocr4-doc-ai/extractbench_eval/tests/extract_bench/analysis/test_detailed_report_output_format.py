"""Tests for detailed report output format handling."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from extract_bench.analysis.detailed_report import _render_output_html

_STATIC_DIR = Path(str(files("extract_bench.analysis").joinpath("static")))


def test_json_panel_does_not_reserialize_the_payload() -> None:
    """The highlighted view and the copy source must show the same bytes.

    The payload already arrives as ``json.dumps(indent=2)``. Re-serializing it in
    the browser reorders integer-like keys, so the JSON on screen would no longer
    match what Copy puts on the clipboard.
    """
    js = (_STATIC_DIR / "detailed_report.js").read_text(encoding="utf-8")
    assert "JSON.stringify(JSON.parse(" not in js


def test_render_output_html_skips_json() -> None:
    payload = '{\n  "invoice_number": "INV_001"\n}'
    assert _render_output_html(payload, "json") == ""


def test_render_output_html_renders_markdown() -> None:
    html = _render_output_html("# Title\n\nSome **bold** text.", "markdown")
    assert "<" in html
    assert "Title" in html


def test_render_output_html_empty_for_missing_text() -> None:
    assert _render_output_html(None, "markdown") == ""
    assert _render_output_html("", "json") == ""
