"""UI-side rendering of already-fetched OCR data.

These are presentation helpers: they turn OCR blocks (already returned by the backend) into
what a viewer draws, an overlay's scaled boxes and a markdown summary. No OCR, no model, no
network. They live with the UI so the UI process depends on no backend module.
"""
from __future__ import annotations


def overlay_boxes(preview: str, blocks: list[dict], width, height) -> list[tuple]:
    """Scale each OCR block's page coordinates to the rendered preview's pixels and pair it
    with its type label. Returns [((x0, y0, x1, y1), type), ...]."""
    if not preview or not blocks or not width or not height:
        return []
    from PIL import Image

    w, h = Image.open(preview).size
    sx, sy = w / width, h / height
    out = []
    for b in blocks:
        box = (int(b["x0"] * sx), int(b["y0"] * sy), int(b["x1"] * sx), int(b["y1"] * sy))
        out.append((box, b.get("type", "text")))
    return out


def blocks_table(blocks: list[dict]) -> str:
    """Markdown summary of detected layout blocks: type counts plus a per-block content
    snippet."""
    if not blocks:
        return "_Upload a document to see its layout blocks._"
    from collections import Counter

    counts = Counter(b.get("type", "text") for b in blocks)
    lines = [f"**{len(blocks)} layout blocks** detected by Mistral OCR 4.", "",
             "**By type:** " + ", ".join(f"{t} ({n})" for t, n in counts.most_common()), "",
             "| # | Type | Content |", "|---|---|---|"]
    for i, b in enumerate(blocks, 1):
        snippet = (b.get("content") or "").replace("\n", " ").replace("|", "\\|")[:70]
        lines.append(f"| {i} | {b.get('type', 'text')} | {snippet} |")
    return "\n".join(lines)
