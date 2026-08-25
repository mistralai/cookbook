"""Static asset loaders for self-contained HTML reports.

Loads CSS and JavaScript from ``analysis/static/`` and composes them into
report-specific ``<style>`` / ``<script>`` elements for inline embedding.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

_DEPS_MARKER = "// __REPORT_DEPS__\n"


@lru_cache(maxsize=16)
def _read_static(name: str) -> str:
    path = files("extract_bench.analysis").joinpath("static", name)
    return path.read_text(encoding="utf-8")


def _style_tag(css: str) -> str:
    return "<style>\n" + css + "\n</style>"


def _script_tag(js: str) -> str:
    return "<script>\n" + js + "\n</script>"


def _inject_js_deps(raw: str, deps: str) -> str:
    if _DEPS_MARKER not in raw:
        raise ValueError(f"JS asset is missing {_DEPS_MARKER!r} dependency injection marker")
    return raw.replace(_DEPS_MARKER, deps)


@lru_cache(maxsize=1)
def detailed_report_style() -> str:
    css = _read_static("detailed_report.css") + _read_static("bbox_overlay.css") + _read_static("tooltip.css")
    return _style_tag(css)


@lru_cache(maxsize=1)
def detailed_report_script() -> str:
    deps = _read_static("bbox_overlay.js") + _read_static("tooltip.js")
    body = _inject_js_deps(_read_static("detailed_report.js"), deps)
    return _script_tag(body)


@lru_cache(maxsize=1)
def comparison_report_style() -> str:
    css = _read_static("comparison_report.css") + _read_static("bbox_overlay.css") + _read_static("tooltip.css")
    return _style_tag(css)


@lru_cache(maxsize=1)
def comparison_report_script() -> str:
    deps = _read_static("bbox_overlay.js") + _read_static("tooltip.js")
    body = _inject_js_deps(_read_static("comparison_report.js"), deps)
    return _script_tag(body)


@lru_cache(maxsize=1)
def aggregation_report_style() -> str:
    return _style_tag(_read_static("aggregation_report.css") + _read_static("tooltip.css"))


@lru_cache(maxsize=1)
def aggregation_report_script() -> str:
    body = _inject_js_deps(_read_static("aggregation_report.js"), _read_static("tooltip.js"))
    return _script_tag(body)
