"""Chat with any document.

Mistral OCR 4 reads whatever the user uploads (any PDF or image it supports) and returns
the text, the layout blocks with their types, and a short classification. A Foundry agent
(document-chat-agent, consumed through foundry_agents like the mortgage agents) then answers
questions grounded in that text. Both steps emit trace spans, so the runs surface in the
Foundry observability views.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request

from config import load_settings
from foundry_agents import DOC_CHAT_AGENT, _FALLBACK_INSTRUCTIONS, _run_server, agents_enabled
from observability import get_tracer, setup_observability
from ocr_client import ocr_with_annotation

# Cap the context handed to the model. Kept moderate so each call stays well under the Medium
# 3.5 deployment's throughput cap (retries below would otherwise burst into 429s), while still
# covering the first several pages of a long scan such as a multi-page 10-Q.
_MAX_CONTEXT = 9000

# The document-chat-agent's instructions, reused as the system prompt on the /models fallback.
_DOC_CHAT_SYSTEM = _FALLBACK_INSTRUCTIONS[DOC_CHAT_AGENT]

# One money/percentage amount anywhere in the text.
_MONEY = re.compile(r"[$€£]\s?\d|(?<!\w)\d[\d,]*\.\d{2}\b|\b\d+(?:\.\d+)?\s?%")
# A description followed by a trailing amount, i.e. a receipt/invoice line item:
# "Whole Milk 2L    $3.25", "TOTAL   $19.65".
_LINE_ITEM = re.compile(r"^.*\S.*?[\s.]+[$€£]?\s?\d[\d,]*\.\d{2}\s*$")


def _refine_type(block: dict) -> str:
    """A finer label than OCR 4's coarse native type (text/title/header/footer/table), derived
    from the block's own content in the SAME OCR call, no extra model round-trip. Mirrors how
    the underwriting overlay derives region labels from OCR 4's output by inspecting content.

    Heuristic, so it favors clear signals and falls back to paragraph; it will not be perfect
    on every layout."""
    native = (block.get("type") or "text").lower()
    # Trust OCR 4's own strong structural signals.
    if native in ("table", "title", "header", "footer"):
        return native
    content = " ".join((block.get("content") or "").split())
    if not content:
        return "text"
    low = content.lower()
    money_hits = len(_MONEY.findall(content))
    words = content.split()

    # Signature / handwriting cues (a name over a signature line, or an explicit note).
    if any(k in low for k in ("signature", "signed", "/s/")) or low.startswith("note:"):
        return "signature"
    # Totals: a total/subtotal/balance keyword with an amount. Checked before line_item so
    # "TOTAL $19.65" reads as a total, not a generic line.
    if any(k in low for k in ("total", "subtotal", "balance", "amount due")) and money_hits >= 1:
        return "total"
    # A row that ends in a price is a line item (the bulk of receipts/invoices).
    if money_hits >= 1 and _LINE_ITEM.match(content):
        return "line_item"
    # Several amounts and no clear single-line-item shape: a figures block.
    if money_hits >= 3:
        return "amounts"
    # Label: value (form fields, "Date: ...", "Receipt #: ..."). One colon is enough when the
    # colon is early in a short line.
    if ":" in content and len(content) <= 80 and content.split(":", 1)[0].split() and \
       len(content.split(":", 1)[0]) <= 24:
        return "key_value"
    # Bulleted or numbered list.
    if re.match(r"^\s*([\-•*]|\d+[.)])\s", content):
        return "list"
    # Heading: a genuinely short line (<= 5 words) that is all-caps or has no sentence
    # punctuation. Deliberately narrow, so ordinary short content is not mislabeled.
    if len(words) <= 5 and not content.endswith((".", ",")) and \
       (content.isupper() or not any(c.isdigit() for c in content)):
        return "heading"
    return "paragraph"


def analyze(data: bytes, name: str) -> dict:
    """OCR a document once: text, layout blocks, page size, and a short classification
    (document_type / language / summary from OCR 4's default annotation). Each block gets a
    finer `type` than OCR 4's native set, derived in code from the same OCR result, so there is
    no second model call, the same one-OCR-call pattern the underwriting flow uses.

    Layout labeling is a backend step done here, so any front end receives labeled blocks."""
    setup_observability()
    with get_tracer().start_as_current_span("docchat.ocr") as span:
        span.set_attribute("document.name", name)
        result = ocr_with_annotation(data, name)  # default schema: type / language / summary
        # Refine every block on every page (the overlay can show any page).
        total = 0
        for page in result.get("pages_layout") or []:
            for b in page.get("blocks") or []:
                b["type"] = _refine_type(b)
                total += 1
        span.set_attribute("ocr.blocks", total)
    return result


# The mistral-medium-3-5 deployment intermittently returns an empty completion (finish_reason
# "stop", one token) on large grounded prompts — a known Medium 3.5 platform flakiness. The
# failures are independent per call, so we retry a few times, spaced out, until we get a
# non-empty answer (spacing keeps a burst of retries from tripping the deployment's 429 cap).
_MODELS_ATTEMPTS = 4
_MODELS_RETRY_PAUSE = 0.6


def _answer_via_models(context: str, question: str) -> str:
    """Answer via the Azure AI inference /models route, with the doc-chat instructions as the
    system prompt. The hosted document-chat-agent runs through Foundry Agent Service (the
    Responses API), which returns 500 on large or dense OCR Markdown, for example a multi-page
    10-Q, that this /models route handles cleanly. Retries absorb Medium 3.5's intermittent
    empty completions. This keeps the chat answering; it does not emit a per-agent portal
    trace (the primary path above does)."""
    s = load_settings()
    inference = os.environ.get("AZURE_INFERENCE_ENDPOINT", "").rstrip("/")
    if not (inference and s.chat_deployment and s.ai_key):
        raise RuntimeError("no inference endpoint or chat deployment configured")
    user = f"DOCUMENT (Markdown from Mistral OCR 4):\n\n{context}\n\nQUESTION: {question}"
    body = json.dumps({
        "model": s.chat_deployment,
        "messages": [{"role": "system", "content": _DOC_CHAT_SYSTEM},
                     {"role": "user", "content": user}],
        "max_tokens": 512,
        "temperature": 0.2,
    }).encode()
    url = f"{inference}/models/chat/completions?api-version=2024-05-01-preview"
    headers = {"Content-Type": "application/json", "api-key": s.ai_key}
    last_exc: Exception | None = None
    for attempt in range(_MODELS_ATTEMPTS):
        if attempt:
            time.sleep(_MODELS_RETRY_PAUSE)
        try:
            req = urllib.request.Request(url, data=body, method="POST", headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = json.load(resp)
            text = (payload["choices"][0]["message"].get("content") or "").strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001 - retry transient model/network errors
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    return ""  # every attempt returned an empty completion


async def answer(markdown: str, question: str) -> str:
    """Answer a question about the document, grounded in its OCR text.

    Primary path: run the hosted document-chat-agent server-side, so the turn appears in that
    agent's Foundry Traces. On large or dense documents the Agent Service (Responses API)
    returns 500, so we fall back to the inference /models route, which handles them reliably."""
    if not agents_enabled():
        return ("Document questions need the chat agent. Set USE_FOUNDRY_AGENTS=true and "
                "AZURE_AI_PROJECT_ENDPOINT, sign in with `az login`, then reload.")
    if not (markdown or "").strip():
        return "Upload a document first, then ask a question about it."
    # Truncate long docs at a clean line boundary, never mid-table, so the model gets a
    # well-formed document.
    context = markdown[:_MAX_CONTEXT]
    if len(markdown) > _MAX_CONTEXT:
        cut = context.rfind("\n")
        if cut > _MAX_CONTEXT // 2:
            context = context[:cut]
    prompt = f"DOCUMENT (Markdown from Mistral OCR 4):\n\n{context}\n\nQUESTION: {question}"

    # Primary: hosted agent, server-side, for the per-agent portal trace.
    try:
        with get_tracer().start_as_current_span("agent.doc_chat"):
            resp = await _run_server(DOC_CHAT_AGENT, prompt)
        text = (resp.text or "").strip()
        if text:
            return text
    except Exception:  # noqa: BLE001 - large/dense docs 500 here; fall back to /models
        pass

    # Fallback: the inference /models route (handles the docs the Responses route rejects).
    try:
        with get_tracer().start_as_current_span("docchat.models_fallback"):
            text = await asyncio.to_thread(_answer_via_models, context, question)
        return text or "I could not find that in the document."
    except Exception as exc:  # keep the demo alive on any model or auth error
        detail = (str(exc).strip().splitlines() or [""])[0]
        low = str(exc).lower()
        if any(k in low for k in ("nameresolution", "getaddrinfo", "failed to resolve",
                                  "max retries", "connectionerror", "login.microsoftonline.com")):
            return ("Could not reach the model endpoint. Check the network, then ask again. "
                    "The document text and blocks above are still available.")
        if any(k in low for k in ("401", "403", "credential", "token", "authorization", "az login")):
            return "The chat model could not authenticate. Check AZURE_AI_KEY, then ask again."
        return f"Sorry, the chat agent could not answer that ({detail[:160]})."
