"""Helpers for the ``text_only`` extract bench mode.

``text_only`` baselines extraction quality when the input is only raw markdown
(no pages, no images) — the realistic case where a customer uploads a ``.md``
that skips structured parse. These pure functions derive a single collapsed
markdown blob from a parse-result-response (sidecar or freshly parsed) so the
provider can upload it as a direct text job.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Page markers that a paginated markdown rendering may carry. Collapsing strips
# them so the blob has no page boundaries at all.
_PAGE_MARKER_RE = re.compile(r"^<<<PAGE:\d+>>>\s*$", re.MULTILINE)

# Parse tiers, ordered cheapest -> richest. Used to cap the inferred parse tier.
_PARSE_TIER_RANK = {"fast": 0, "cost_effective": 1, "agentic": 2}
_MAX_TEXT_ONLY_PARSE_TIER = "agentic"


def _normalize_tier(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _page_is_failed(page: dict[str, Any]) -> bool:
    return page.get("success") is False or "error" in page


def _markdown_from_items_page(page: dict[str, Any]) -> str:
    items = page.get("items")
    if not isinstance(items, list):
        return ""
    parts = [str(item["md"]) for item in items if isinstance(item, dict) and item.get("md")]
    return "\n\n".join(parts)


def _pages_field(resp: dict[str, Any], section: str) -> list[dict[str, Any]] | None:
    """Return the ``pages`` list of a result section, accepting the flat shape too."""
    value = resp.get(section)
    if isinstance(value, dict):
        pages = value.get("pages")
        if isinstance(pages, list):
            return [p for p in pages if isinstance(p, dict)]
        # Flat single-string markdown (v1-style ``{"markdown": "..."}``).
        if section == "markdown" and isinstance(value.get("markdown"), str):
            return [{"markdown": value["markdown"]}]
    return None


def markdown_pages_from_parse_response(resp: dict[str, Any]) -> list[str]:
    """Extract per-page markdown from a parse-result-response dict.

    Preference order: ``markdown`` pages -> ``items`` pages (joined item md) ->
    ``text`` pages. Failed pages are skipped. Also accepts a bare top-level
    ``pages`` list or a list payload (legacy flat ``.v2.md.json``).
    """
    markdown_pages = _pages_field(resp, "markdown")
    if markdown_pages is not None:
        return [str(p.get("markdown", "")) for p in markdown_pages if not _page_is_failed(p) and p.get("markdown")]

    items_pages = _pages_field(resp, "items")
    if items_pages is not None:
        out = [_markdown_from_items_page(p) for p in items_pages if not _page_is_failed(p)]
        return [p for p in out if p]

    text_pages = _pages_field(resp, "text")
    if text_pages is not None:
        return [str(p.get("text", "")) for p in text_pages if not _page_is_failed(p) and p.get("text")]

    # Legacy flat shapes: a bare list of pages, or a top-level ``pages`` list.
    flat = resp.get("pages") if isinstance(resp.get("pages"), list) else None
    if flat is not None:
        flat_out: list[str] = []
        for p in flat:
            if not isinstance(p, dict) or _page_is_failed(p):
                continue
            value = p.get("markdown") or p.get("md") or _markdown_from_items_page(p) or p.get("text")
            if value:
                flat_out.append(str(value))
        return flat_out

    return []


def collapse_markdown_pages(pages: list[str]) -> str:
    """Join per-page markdown into one blob with no page boundaries.

    Drops empty pages and defensively strips any embedded ``<<<PAGE:N>>>``
    markers so the result is a single continuous document.
    """
    cleaned = [_PAGE_MARKER_RE.sub("", page).strip() for page in pages]
    return "\n\n".join(part for part in cleaned if part)


def infer_text_only_parse_tier(config: dict[str, Any]) -> str:
    """Parse tier used to produce markdown when no sidecar exists.

    Explicit ``parse_tier`` wins. Otherwise derive from the extract ``tier``,
    capped at agentic: ``agentic_plus -> agentic``, ``agentic -> agentic``,
    ``cost_effective -> cost_effective``.
    """
    explicit = _normalize_tier(config.get("parse_tier"))
    if explicit:
        return explicit

    tier = _normalize_tier(config.get("tier")) or "cost_effective"
    cap = _PARSE_TIER_RANK[_MAX_TEXT_ONLY_PARSE_TIER]
    rank = _PARSE_TIER_RANK.get(tier)
    if rank is None or rank > cap:
        return _MAX_TEXT_ONLY_PARSE_TIER
    return tier


def find_sidecar_parse_response(source: Path) -> dict[str, Any] | None:
    """Load ``<stem>.parse/parse-result-response.json`` next to the source doc."""
    sidecar = source.with_suffix(".parse") / "parse-result-response.json"
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
