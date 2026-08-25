"""Tests for HTML report static JS/CSS assets."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path

import pytest

from extract_bench.analysis.report_static_assets import (
    aggregation_report_script,
    aggregation_report_style,
    comparison_report_script,
    comparison_report_style,
    detailed_report_script,
    detailed_report_style,
)

_STATIC_DIR = Path(str(files("extract_bench.analysis").joinpath("static")))
_JS_FILES = sorted(_STATIC_DIR.glob("*.js"))

# Patterns left behind when Python string templates are copied into .js files.
_PYTHON_ARTIFACTS = (
    "json.dumps(",
    '+ """',
    '""" +',
)


def _node_check(script: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
        tmp.write(script)
        tmp_path = tmp.name

    try:
        result = subprocess.run(  # noqa: S603
            [node, "--check", tmp_path],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip()
        raise AssertionError(f"node --check failed:\n{msg}")


def _node_eval(script: str) -> str:
    """Run ``script`` under node and return its stdout."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
        tmp.write(script)
        tmp_path = tmp.name

    try:
        result = subprocess.run(  # noqa: S603
            [node, tmp_path],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip()
        raise AssertionError(f"node failed:\n{msg}")
    return result.stdout.strip()


@pytest.mark.parametrize("js_path", _JS_FILES, ids=lambda p: p.name)
def test_static_js_has_no_python_template_artifacts(js_path: Path) -> None:
    content = js_path.read_text(encoding="utf-8")
    for artifact in _PYTHON_ARTIFACTS:
        assert artifact not in content, f"{js_path.name} contains Python template artifact: {artifact!r}"


@pytest.mark.parametrize("js_path", _JS_FILES, ids=lambda p: p.name)
def test_static_js_passes_node_syntax_check(js_path: Path) -> None:
    body = js_path.read_text(encoding="utf-8")
    preamble = _syntax_check_preamble(js_path.name)
    _node_check(f"{preamble}\n{body}\n")


# ``\\n``, ``\\b``, ``\\w`` … — a doubled backslash in front of a character that
# forms an escape sequence on its own. A bare doubled backslash is fine (e.g.
# ``[^"\\]`` matches a literal backslash), so only these are worth flagging.
_DOUBLE_ESCAPE_RE = re.compile(r"\\\\[nbwdstru]")


@pytest.mark.parametrize("js_path", _JS_FILES, ids=lambda p: p.name)
def test_static_js_has_no_double_escaped_sequences(js_path: Path) -> None:
    """Guard the class of bug that ``\\n`` and ``\\b\\w`` belonged to.

    These files used to be Python string literals, where ``\\\\n`` was evaluated
    down to ``\\n`` before being emitted. Copied verbatim into a ``.js`` file the
    doubling survives, and the code silently matches a literal backslash instead
    of a newline or word boundary.
    """
    content = js_path.read_text(encoding="utf-8")
    offenders = [
        f"line {i}: {line.strip()}" for i, line in enumerate(content.splitlines(), 1) if _DOUBLE_ESCAPE_RE.search(line)
    ]
    assert not offenders, f"{js_path.name} has double-escaped sequences:\n" + "\n".join(offenders)


def test_layout_comparison_overlay_scales_gt_to_coco_dimensions() -> None:
    """GT layout boxes are COCO [x, y, w, h]; the drawn rect must keep w/h."""
    overlay_js = (_STATIC_DIR / "bbox_overlay.js").read_text(encoding="utf-8")
    harness = """
const strokes = [];
const ctx = {
  save() {}, restore() {}, setLineDash() {}, fillRect() {},
  measureText() { return { width: 0 }; },
  strokeRect(x, y, w, h) { strokes.push([x, y, w, h]); },
  set lineWidth(v) {}, set strokeStyle(v) {}, set fillStyle(v) {}, set font(v) {},
};
window.BboxOverlay.drawLayoutComparisonOverlay(
  ctx, [{ bbox: [100, 50, 200, 80] }], [], [], 0.5,
  { showGT: true, showA: false, showB: false },
);
console.log(JSON.stringify(strokes));
"""
    drawn = json.loads(_node_eval("const window = globalThis;\n" + overlay_js + harness))
    assert drawn == [[50.0, 25.0, 100.0, 40.0]]


def test_detailed_report_css_keeps_narrow_viewport_rules() -> None:
    css = (_STATIC_DIR / "detailed_report.css").read_text(encoding="utf-8")
    assert "@media (max-width: 768px)" in css
    narrow = css.split("@media (max-width: 768px)", 1)[1]
    for selector in (".report-container", ".controls-bar", ".col-tags"):
        assert selector in narrow, f"detailed_report.css narrow-viewport block is missing {selector!r}"


def test_detailed_report_css_includes_collapsible_and_output_panel_styles() -> None:
    css = (_STATIC_DIR / "detailed_report.css").read_text(encoding="utf-8")
    for selector in (
        ".detail-collapsible-body.open",
        ".detail-collapsible-toggle.open .chevron",
        ".output-columns",
        ".output-view-btn.active",
    ):
        assert selector in css, f"detailed_report.css is missing {selector!r}"


def test_comparison_report_css_uses_single_backslash_unicode_escape() -> None:
    css = (_STATIC_DIR / "comparison_report.css").read_text(encoding="utf-8")
    assert "content: '\\25B6';" in css
    assert "content: '\\\\25B6';" not in css


def test_comparison_report_js_splits_diff_on_real_newlines() -> None:
    """Python-string leftovers used split('\\\\n'), which diffs whole JSON blobs."""
    js = (_STATIC_DIR / "comparison_report.js").read_text(encoding="utf-8")
    assert ".split('\\n')" in js
    assert ".split('\\\\n')" not in js


def test_comparison_report_has_json_highlight_styles() -> None:
    css = (_STATIC_DIR / "comparison_report.css").read_text(encoding="utf-8")
    js = (_STATIC_DIR / "comparison_report.js").read_text(encoding="utf-8")
    assert "function highlightJson(" in js
    assert ".output-panel-body.output-json .json-key" in css
    assert ".output-copy-btn" in css


def test_tooltip_js_uses_single_backslash_unicode_escapes() -> None:
    js = (_STATIC_DIR / "tooltip.js").read_text(encoding="utf-8")
    assert "\\u201c" in js
    assert "\\u201d" in js
    assert "\\u00d7" in js
    assert "\\\\u201c" not in js
    assert "\\\\u201d" not in js
    assert "\\\\u00d7" not in js


def test_composed_report_bundles_are_html_elements() -> None:
    styles = (detailed_report_style(), aggregation_report_style(), comparison_report_style())
    scripts = (detailed_report_script(), aggregation_report_script(), comparison_report_script())
    for style in styles:
        assert style.startswith("<style>\n")
        assert style.endswith("\n</style>")
    for script in scripts:
        assert script.startswith("<script>\n")
        assert script.endswith("\n</script>")


def test_composed_report_bundles_pass_node_syntax_check() -> None:
    bundles = {
        "detailed_report.js": detailed_report_script(),
        "aggregation_report.js": aggregation_report_script(),
        "comparison_report.js": comparison_report_script(),
    }
    for preamble_name, script in bundles.items():
        script = script.removeprefix("<script>\n").removesuffix("\n</script>")
        preamble = _syntax_check_preamble(preamble_name)
        _node_check(f"{preamble}\n{script}\n")


def _syntax_check_preamble(filename: str) -> str:
    """Minimal browser/global stubs so node --check can parse report scripts."""
    common = """
const window = globalThis;
const document = {
  addEventListener() {},
  getElementById() { return null; },
  createElement() {
    return {
      textContent: '',
      innerHTML: '',
      setAttribute() {},
      appendChild() {},
      classList: { add() {}, remove() {}, toggle() {} },
    };
  },
  body: { appendChild() {} },
  querySelector() { return null; },
  querySelectorAll() { return []; },
};
const localStorage = { getItem() { return null; }, setItem() {} };
const DATA = { categories: [], metricTooltips: {}, examples: [], pdfBaseUrl: '' };
function esc(value) {
  return String(value ?? '');
}
window.BboxOverlay = {
  drawLayoutPredictions() {},
  drawLayoutComparisonOverlay() {},
  drawExtractGroundingOverlay() {},
  drawExtractComparisonOverlay() {},
  syncCanvasToImage() {},
  syncCanvasToCanvas() {},
  clearCanvas() {},
};
function tooltipIcon() { return ''; }
"""
    if filename == "comparison_report.js":
        return (
            common
            + """
const comparisonData = [];
const pipelineAName = 'pipeline_a';
const pipelineBName = 'pipeline_b';
const productType = 'extract';
let comparisonMetric = 'accuracy';
let metricDisplayName = 'Accuracy';
const originalBasePath = '';
const pdfBaseUrl = '';
const metricTooltips = {};
const metricDisplayNames = { accuracy: 'Accuracy' };
"""
        )
    if filename == "detailed_report.js":
        return common
    if filename == "aggregation_report.js":
        return common
    if filename in {"bbox_overlay.js", "tooltip.js"}:
        return common
    return common
