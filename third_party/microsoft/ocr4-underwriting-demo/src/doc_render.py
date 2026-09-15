"""Rasterize an uploaded document into a displayable image.

This is document processing, not presentation: it turns a PDF page or an image file into a
PNG path that any front end can show. It lives in the backend so the UI never rasterizes
files itself; the UI just asks for a preview and renders whatever path it gets back.
"""
from __future__ import annotations

import os
import tempfile

# Extensions we can hand back to a viewer unchanged (a PDF is rendered; everything else is
# unsupported and returns None).
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

# Preview render scale, relative to a PDF's native 72 dpi. scale=3 is ~216 dpi, enough that the
# on-screen zoom stays sharp when a reviewer inspects a flagged value; scale=2 (~144 dpi) went
# soft as soon as you zoomed in, and it downsampled higher-resolution uploads. Raising this
# trades memory and render time for sharpness. It is safe to tune: the overlay code derives box
# coordinates from the rendered preview's own pixel size, so highlights follow whatever we emit.
PREVIEW_SCALE = 3.0


def render_pdf_page(path: str, page: int = 0) -> str | None:
    """Render one PDF page to a temp PNG and return its path, or None on failure."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(path)
        page = max(0, min(page, len(pdf) - 1))  # clamp to a valid page
        pil = pdf[page].render(scale=PREVIEW_SCALE).to_pil()
        out = tempfile.mktemp(suffix=".png")
        pil.save(out)
        return out
    except Exception:
        return None


def make_preview(path: str | None, page: int = 0) -> str | None:
    """A displayable image path for an upload: images pass through, PDFs render the given page
    (default page 0). Returns None when the file is gone.

    A case keeps its scans in a temp dir, so a case created before an app restart can outlive
    its images; a caller must degrade to no preview rather than pass a missing path onward.
    """
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return path if page == 0 else None  # a plain image has only one page
    if ext == ".pdf":
        return render_pdf_page(path, page)
    return None
