"""Shared Mistral OCR 4 client — used by the CLI, the workflow, and the Function.

Deterministic REST calls to the deployed OCR endpoint (API-key auth, so it works
without the Entra RBAC role). Documents are sent as base64 data URLs because the
endpoint has no outbound internet access.

OCR 4 also supports `document_annotation_format`: pass a JSON schema and the
endpoint returns a structured `document_annotation` (e.g. document_type,
language, summary) in a single call — i.e. classification without a separate LLM.
Annotations are capped at ~8 pages and can time out on large docs.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path

import requests

from config import load_settings

MAX_BYTES = 30 * 1024 * 1024  # 30 MB Foundry limit
MAX_PAGES = 30  # documented Foundry limit

# Default annotation schema: OCR4 classifies + summarizes the document itself.
DEFAULT_ANNOTATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "title": "DocAnnotation",
    "properties": {
        "document_type": {"type": "string", "title": "Document_Type"},
        "language": {"type": "string", "title": "Language"},
        "summary": {"type": "string", "title": "Summary"},
    },
    "required": ["document_type", "language", "summary"],
}


def _build_document(data: bytes, name: str) -> dict:
    if len(data) > MAX_BYTES:
        raise ValueError(f"File is {len(data) / 1e6:.1f} MB — exceeds the 30 MB limit.")

    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    data_url = f"data:{mime};base64,{base64.b64encode(data).decode()}"

    if mime == "application/pdf":
        return {"type": "document_url", "document_name": Path(name).stem, "document_url": data_url}
    if mime.startswith("image/"):
        return {"type": "image_url", "image_url": data_url}
    raise ValueError(f"Unsupported file type: {mime} (need PDF or image).")


# Transient responses worth retrying: 429 rate limiting, plus gateway/backend hiccups. The OCR
# deployment is capacity-limited, so a burst of uploads legitimately draws 429s; back off and
# retry rather than fail the whole request.
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_OCR_RETRIES = 3


def _retry_after_seconds(resp: requests.Response) -> float | None:
    """Seconds to wait per the Retry-After header, when the service sends an integer one."""
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None  # the HTTP-date form is uncommon here; fall back to our own backoff


def _post_ocr(payload: dict) -> dict:
    """POST to the OCR endpoint, retrying transient 429/5xx with backoff.

    401/404 are configuration errors, so they surface immediately with a clear message. A 429 or
    5xx is retried up to _MAX_OCR_RETRIES times, honoring Retry-After when present; if the last
    attempt still fails, the HTTPError (carrying the status) is raised for the caller to handle.
    """
    settings = load_settings()
    headers = {"Authorization": f"Bearer {settings.ai_key}", "Content-Type": "application/json"}
    backoff = 2.0
    resp = None
    for attempt in range(_MAX_OCR_RETRIES + 1):
        resp = requests.post(settings.ocr_endpoint, headers=headers, json=payload, timeout=300)
        if resp.status_code == 404:
            raise RuntimeError(
                "404 — check the exact Target URI, the deployment name in `model`, "
                "and that you're not hitting a chat/completions route."
            )
        if resp.status_code == 401:
            raise RuntimeError("401 — bad key. Verify AZURE_AI_KEY / AZURE_OCR_KEY.")
        if resp.status_code not in _RETRY_STATUSES:
            resp.raise_for_status()
            return resp.json()
        if attempt < _MAX_OCR_RETRIES:
            time.sleep(_retry_after_seconds(resp) or backoff)
            backoff = min(backoff * 2, 20.0)
    resp.raise_for_status()  # retries exhausted: raise the last transient error (e.g. 429)
    return resp.json()       # unreachable; keeps the return type honest


def _markdown(body: dict) -> str:
    return "\n\n---\n\n".join(p.get("markdown", "") for p in body.get("pages", []))


def _page_layout(page: dict) -> dict:
    """Extract one page's layout blocks (bounding boxes) and dimensions from an OCR page."""
    blocks = []
    for b in page.get("blocks") or []:
        try:
            blocks.append({
                "x0": int(float(b["top_left_x"])), "y0": int(float(b["top_left_y"])),
                "x1": int(float(b["bottom_right_x"])), "y1": int(float(b["bottom_right_y"])),
                "type": b.get("type", "text"),
                "content": (b.get("content") or "")[:300],
            })
        except (KeyError, TypeError, ValueError):
            continue
    dims = page.get("dimensions") or {}
    return {"blocks": blocks, "width": dims.get("width"), "height": dims.get("height")}


def ocr_bytes(data: bytes, name: str, *, pages: str | None = None,
              table_format: str | None = None) -> str:
    """OCR raw bytes → concatenated markdown across pages."""
    settings = load_settings()
    payload: dict = {
        "model": settings.ocr_deployment,
        "document": _build_document(data, name),
        "include_image_base64": False,
    }
    if pages:
        payload["pages"] = pages
    if table_format:
        payload["table_format"] = table_format
    return _markdown(_post_ocr(payload))


def ocr_with_annotation(data: bytes, name: str, *, annotation_schema: dict | None = None,
                        pages: str | None = None) -> dict:
    """OCR + document annotation in one call.

    Returns {"markdown": str, "annotation": dict} where annotation follows the
    provided schema (default: document_type / language / summary).
    """
    settings = load_settings()
    payload: dict = {
        "model": settings.ocr_deployment,
        "document": _build_document(data, name),
        "include_image_base64": False,
        "document_annotation_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "doc_annotation",
                "strict": True,
                "schema": annotation_schema or DEFAULT_ANNOTATION_SCHEMA,
            },
        },
    }
    if pages:
        payload["pages"] = pages

    body = _post_ocr(payload)
    annotation = body.get("document_annotation") or {}
    if isinstance(annotation, str):
        try:
            annotation = json.loads(annotation)
        except json.JSONDecodeError:
            annotation = {}

    # Layout blocks (bounding boxes) per page, for the on-page highlights.
    pages = body.get("pages") or []
    pages_layout = [_page_layout(p) for p in pages]
    first = pages_layout[0] if pages_layout else {"blocks": [], "width": None, "height": None}
    return {
        "markdown": _markdown(body),
        "annotation": annotation,
        "pages_count": len(pages),
        "usage": body.get("usage_info") or {},
        # Per-page layout: [{"blocks": [...], "width": w, "height": h}, ...].
        "pages_layout": pages_layout,
        # Page-1 values kept flat for callers that only handle one page (underwriting overlay).
        "blocks": first["blocks"],
        "width": first["width"],
        "height": first["height"],
    }


def ocr_file(path: str | Path, *, pages: str | None = None,
             table_format: str | None = None) -> str:
    """OCR a file on disk → markdown."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")
    return ocr_bytes(p.read_bytes(), p.name, pages=pages, table_format=table_format)
