"""The user interface: three Gradio views over the backend API.

    uv run uvicorn app:ui --port 8000
    Customer view:  http://localhost:8000/apply
    Reviewer view:  http://localhost:8000/review
    Document chat:  http://localhost:8000/chat

This process is UI only. It talks to the backend (api.py) over HTTP through `service_client`
(imported as `S`); it imports no OCR, agent, or service module. Start both with scripts/up.sh,
which runs the backend first and points this app at it via BACKEND_URL.
"""
from __future__ import annotations

import ast
import html
import os
import tempfile

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

import documents as D
import mortgage_rules as R
import service_client as S
from doc_render import make_preview
from ui_render import blocks_table, overlay_boxes
from underwriting_schema import Outcome, Stage, UnderwritingCase

# A tiny FastAPI shell exists only to host the landing page and mount the Gradio views. It
# carries no business endpoints; all of those live in the backend service (api.py).
ui = FastAPI(title="Underwriting UI")


LANDING_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Document to decision</title>
<style>
  :root {
    --ink: oklch(0.30 0.012 60); --muted: oklch(0.52 0.012 60);
    --line: oklch(0.90 0.008 70); --surface: oklch(0.995 0.004 75);
    --paper: oklch(0.985 0.006 75); --accent: oklch(0.63 0.16 47);
  }
  * { box-sizing: border-box; }
  body { margin: 0; min-height: 100vh; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    color: var(--ink); background: radial-gradient(120% 90% at 50% -10%, oklch(0.97 0.03 70), var(--paper));
    display: flex; align-items: center; justify-content: center; padding: 40px 20px; }
  .wrap { width: 100%; max-width: 920px; }
  .eyebrow { font-size: 12.5px; letter-spacing: 0.09em; text-transform: uppercase;
    color: var(--accent); font-weight: 650; }
  h1 { font-size: 34px; line-height: 1.1; margin: 8px 0 6px; font-weight: 680; letter-spacing: -0.01em; }
  .lede { color: var(--muted); font-size: 15.5px; margin: 0 0 30px; max-width: 60ch; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(232px, 1fr)); gap: 16px; }
  @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } h1 { font-size: 28px; } }
  .tile { display: flex; flex-direction: column; gap: 8px; text-decoration: none; color: inherit;
    background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 20px 18px 18px;
    transition: transform 240ms cubic-bezier(0.22,1,0.36,1), box-shadow 240ms cubic-bezier(0.22,1,0.36,1),
      border-color 240ms; }
  .tile:hover { transform: translateY(-3px); border-color: oklch(0.80 0.08 60);
    box-shadow: 0 14px 30px oklch(0.63 0.16 47 / 0.12); }
  .tile .mark { width: 34px; height: 34px; border-radius: 9px; display: grid; place-items: center;
    background: oklch(0.96 0.045 70); color: var(--accent); font-size: 18px; font-weight: 700; margin-bottom: 4px; }
  .tile h2 { font-size: 17px; margin: 0; font-weight: 640; }
  .tile p { font-size: 13.5px; color: var(--muted); margin: 0; line-height: 1.5; }
  .tile .go { margin-top: auto; padding-top: 12px; font-size: 13px; font-weight: 620; color: var(--accent); }
  .foot { margin-top: 28px; font-size: 12.5px; color: var(--muted); }
  .foot code { background: oklch(0.94 0.008 70); padding: 1px 6px; border-radius: 5px; font-size: 12px; }
</style></head>
<body><main class="wrap">
  <div class="eyebrow">Mistral OCR 4 and Mistral Medium 3.5 on Azure Foundry</div>
  <h1>From a document to a decision</h1>
  <p class="lede">A scan goes in, structured data comes out, and a mortgage case reaches a decision with a
    person in the loop. Pick where to start.</p>
  <div class="grid">
    <a class="tile" href="/apply?__theme=light">
      <div class="mark">1</div>
      <h2>Apply</h2>
      <p>Upload a mortgage document. Mistral OCR 4 reads it, and the intake agent asks for anything missing in chat.</p>
      <div class="go">Open the customer view &rarr;</div>
    </a>
    <a class="tile" href="/review?__theme=light">
      <div class="mark">2</div>
      <h2>Review</h2>
      <p>The underwriter queue. Open a flagged case, read the facts and rationale, then approve, decline, or refer.</p>
      <div class="go">Open the reviewer view &rarr;</div>
    </a>
    <a class="tile" href="/chat?__theme=light">
      <div class="mark">3</div>
      <h2>Chat with your document</h2>
      <p>Upload any document. Mistral OCR 4 reads it and shows its layout blocks, and a chat agent answers questions grounded in the text.</p>
      <div class="go">Open the document chat &rarr;</div>
    </a>
  </div>
  <p class="foot">Three views over one backend service. The service runs as a separate app.</p>
</main></body></html>"""


@ui.get("/", response_class=HTMLResponse)
def landing():
    return LANDING_HTML


# ---- Theme and styling ----
# Warm paper neutrals, one accent, tinted (never pure black or white).
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_sm,
    spacing_size=gr.themes.sizes.spacing_md,
).set(
    body_background_fill="oklch(0.982 0.006 75)",
    body_text_color="oklch(0.30 0.012 60)",
    body_text_color_subdued="oklch(0.52 0.012 60)",
    block_background_fill="oklch(0.997 0.003 75)",
    block_border_width="1px",
    block_border_color="oklch(0.90 0.008 70)",
    block_label_text_color="oklch(0.44 0.012 60)",
    input_background_fill="oklch(0.995 0.004 75)",
    button_primary_background_fill="oklch(0.63 0.16 47)",
    button_primary_background_fill_hover="oklch(0.58 0.16 47)",
    button_primary_text_color="oklch(0.99 0.01 75)",
    button_secondary_background_fill="oklch(0.94 0.008 70)",
    button_secondary_text_color="oklch(0.34 0.012 60)",
)

CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; padding-top: 6px !important; }
.app-header { padding: 8px 4px 2px; }
.app-header .home-link { display: inline-block; font-size: 12px; font-weight: 600;
  color: oklch(0.52 0.012 60); text-decoration: none; margin-bottom: 4px; }
.app-header .home-link:hover { color: oklch(0.63 0.16 47); }
.app-header .eyebrow {
  font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
  color: oklch(0.63 0.16 47); font-weight: 600;
}
.app-header h1 { font-size: 21px; line-height: 1.1; margin: 2px 0 1px; font-weight: 650; }
.app-header p { color: oklch(0.52 0.012 60); font-size: 13px; margin: 0; max-width: 70ch; }
/* two-column compact field grid for the reviewer */
.fields-grid { gap: 8px 12px !important; }
/* collapse empty HTML placeholders (package strip, decision, status) so they add no gap */
.gradio-container .block:has(> .html-container:empty) { display: none !important; }
.gradio-container .html-container:empty { padding: 0 !important; margin: 0 !important; }

/* progress stepper */
.stepper { display: flex; flex-wrap: wrap; gap: 6px; padding: 4px 0 2px; }
.step {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 12.5px; padding: 5px 11px 5px 9px; border-radius: 999px;
  border: 1px solid oklch(0.86 0.01 70); color: oklch(0.42 0.012 60) !important;
  background: oklch(0.985 0.004 75);
}
.step .dot { width: 7px; height: 7px; border-radius: 999px; background: oklch(0.80 0.01 70); }
.step.done { color: oklch(0.38 0.06 150) !important; border-color: oklch(0.84 0.06 150); background: oklch(0.96 0.04 150); }
.step.done .dot { background: oklch(0.58 0.12 150); }
.step.current {
  color: oklch(0.42 0.14 47) !important; border-color: oklch(0.72 0.13 47);
  background: oklch(0.97 0.045 70); font-weight: 600;
}
.step.current .dot { background: oklch(0.63 0.16 47); box-shadow: 0 0 0 3px oklch(0.90 0.06 70); }

/* decision card */
.decision { border: 1px solid oklch(0.88 0.01 70); border-radius: 12px; padding: 14px 16px; background: oklch(0.985 0.004 75); color: oklch(0.30 0.012 60) !important; }
.decision .row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.badge { font-size: 12px; font-weight: 650; letter-spacing: 0.02em; text-transform: uppercase; padding: 4px 10px; border-radius: 999px; }
.badge.ok   { background: oklch(0.95 0.05 150); color: oklch(0.42 0.12 150); }
.badge.warn { background: oklch(0.96 0.07 80);  color: oklch(0.46 0.12 70); }
.badge.bad  { background: oklch(0.95 0.05 25);  color: oklch(0.48 0.15 25); }
.badge.muted{ background: oklch(0.93 0.008 70); color: oklch(0.45 0.012 60); }
.decision .title { font-weight: 600; font-size: 15px; color: oklch(0.28 0.012 60) !important; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.chip { display: inline-flex; gap: 6px; align-items: baseline; font-size: 12.5px;
  border: 1px solid oklch(0.91 0.008 70); border-radius: 8px; padding: 4px 9px; background: oklch(0.99 0.004 75); }
.chip .k { color: oklch(0.48 0.012 60) !important; }
.chip .v { font-family: var(--font-mono); color: oklch(0.28 0.012 60) !important; font-weight: 600; }
.rationale { margin-top: 12px; font-size: 13.5px; color: oklch(0.44 0.012 60) !important; line-height: 1.5; }
.muted-note { color: oklch(0.52 0.012 60); font-size: 13.5px; }

/* motion: purposeful reveals, ease-out, no layout animation */
@keyframes fadeUp { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: none; } }
.decision { animation: fadeUp 320ms cubic-bezier(0.22, 1, 0.36, 1); }
.chip { animation: fadeUp 260ms cubic-bezier(0.22, 1, 0.36, 1) both; }
.chips .chip:nth-child(2) { animation-delay: 40ms; }
.chips .chip:nth-child(3) { animation-delay: 80ms; }
.chips .chip:nth-child(4) { animation-delay: 120ms; }
@keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 oklch(0.90 0.06 70 / 0.7); } 50% { box-shadow: 0 0 0 5px oklch(0.90 0.06 70 / 0); } }
.step.current .dot { animation: pulse 1.5s ease-in-out infinite; }
.sample-tray { display: flex; gap: 8px; align-items: center; padding: 2px 0 6px; }
.sample-tray .lbl { font-size: 12.5px; color: oklch(0.52 0.012 60); margin-right: 2px; }
.tray-actions { justify-content: flex-end !important; margin-top: 4px !important; }
.tray-actions button { flex: 0 0 auto; min-width: 0; }
/* fix thumbnail height so the name caption (pinned to the item bottom) is always visible */
.sample-gallery .thumbnail-item, .sample-gallery .thumbnail-item img { height: 118px !important; }
.sample-gallery .grid-wrap { overflow: hidden !important; }
.sample-gallery .caption-label { font-size: 11px !important; padding: 3px 4px !important; }
.suggestions { gap: 6px; flex-wrap: wrap; margin-top: 4px; }
.suggestions button { flex: 0 0 auto; min-width: 0; }
.doc-hint { font-size: 11.5px; color: oklch(0.55 0.012 60); margin: 6px 2px 0; text-align: right; }
#doc-viewer, #doc-viewer-review, #doc-viewer-chat { overflow: hidden; position: relative; }
#doc-viewer img, #doc-viewer-review img, #doc-viewer-chat img { -webkit-user-drag: none; user-select: none; }
/* zoom control buttons overlaid on the document viewer */
.zoom-ctrl { position: absolute; top: 8px; left: 8px; z-index: 40; display: flex;
  flex-direction: column; gap: 4px; }
.zoom-ctrl button { width: 30px; height: 30px; border-radius: 8px; cursor: pointer;
  border: 1px solid oklch(0.85 0.01 70); background: oklch(0.99 0.004 75 / 0.92);
  color: oklch(0.30 0.012 60); font-size: 17px; line-height: 1; display: grid; place-items: center;
  box-shadow: 0 1px 3px oklch(0.4 0.02 60 / 0.14); }
.zoom-ctrl button:hover { background: oklch(0.96 0.03 70); border-color: oklch(0.75 0.08 60); }
.review-flags { border: 1px solid oklch(0.86 0.09 70); background: oklch(0.975 0.05 80);
  border-radius: 10px; padding: 10px 12px; margin: 8px 0 2px; }
.review-flags .rf-title { font-weight: 650; font-size: 12px; color: oklch(0.46 0.12 70);
  text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.review-flags ul { margin: 0; padding-left: 18px; }
.review-flags li { font-size: 13px; color: oklch(0.42 0.04 70); margin: 2px 0; }
.review-ok { font-size: 13px; color: oklch(0.42 0.06 150); margin: 8px 0 2px; }
/* document package checklist */
.pkg { margin: 6px 0 2px; }
.pkg-title { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em;
  color: oklch(0.50 0.012 60); font-weight: 650; margin-bottom: 6px; }
.pkg-count { color: oklch(0.55 0.012 60); font-weight: 600; text-transform: none; letter-spacing: 0; }
.pkg-row { display: flex; flex-wrap: wrap; gap: 6px; }
.pkg-pill { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px;
  padding: 4px 10px; border-radius: 999px; border: 1px solid oklch(0.88 0.01 70);
  background: oklch(0.985 0.004 75); color: oklch(0.42 0.012 60); }
.pkg-pill .pk-mark { font-weight: 700; }
.pkg-pill.received { border-color: oklch(0.84 0.06 150); background: oklch(0.96 0.04 150);
  color: oklch(0.38 0.08 150); }
.pkg-pill.needs_review { border-color: oklch(0.82 0.10 80); background: oklch(0.97 0.06 85);
  color: oklch(0.46 0.12 70); }
.pkg-pill.missing { border-style: dashed; color: oklch(0.55 0.012 60); }
/* decision ledger (reviewer) */
.ledger { overflow-x: auto; }
.ledger-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.ledger-table th { text-align: left; font-weight: 650; color: oklch(0.46 0.02 260);
  border-bottom: 1px solid oklch(0.88 0.01 70); padding: 6px 10px; white-space: nowrap; }
.ledger-table td { border-bottom: 1px solid oklch(0.93 0.006 70); padding: 6px 10px;
  vertical-align: top; color: oklch(0.34 0.02 260); }
.ledger-table tr:hover td { background: oklch(0.985 0.004 75); }
.led-time { white-space: nowrap; color: oklch(0.5 0.01 260); }
.led-id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: oklch(0.5 0.02 260); }
.led-changes .led-chg { margin: 1px 0; white-space: nowrap; }
.led-f { color: oklch(0.46 0.02 260); }
.led-b { color: oklch(0.5 0.03 30); }
.led-arrow { color: oklch(0.6 0.02 150); padding: 0 4px; }
.led-a { color: oklch(0.42 0.12 150); font-weight: 600; }
.led-metric .led-f { font-weight: 600; }
.led-none { color: oklch(0.62 0.008 70); }
.led-ovr { margin-left: 6px; }
"""

# Zoom controls (buttons), wheel-zoom, drag-pan, double-click reset for the document
# viewer. Runs on load and re-arms whenever Gradio swaps the annotated image.
ZOOM_JS = """
() => {
  const setup = (root) => {
    if (!root || root.__zoomInit) return;
    root.__zoomInit = true;
    let scale = 1, tx = 0, ty = 0, lastImg = null, dragging = false, sx = 0, sy = 0;
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const target = () => { const i = root.querySelector('img'); return i ? i.parentElement : null; };
    const apply = (t) => { if (t) { t.style.transformOrigin = '0 0';
      t.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')'; } };
    const ensure = (t) => { const i = root.querySelector('img');
      if (i && i !== lastImg) { lastImg = i; scale = 1; tx = 0; ty = 0;
        if (t) { t.style.cursor = 'grab'; t.style.willChange = 'transform'; } apply(t); } };
    const zoomAt = (mx, my, ns) => { const t = target(); if (!t) return; ensure(t);
      ns = clamp(ns, 1, 6); const ratio = ns / scale;
      tx += mx * (1 - ratio); ty += my * (1 - ratio); scale = ns;
      if (scale <= 1.001) { scale = 1; tx = 0; ty = 0; } apply(t); };
    const centerZoom = (f) => { const t = target(); if (!t) return;
      const rr = root.getBoundingClientRect(), tr = t.getBoundingClientRect();
      zoomAt((rr.left + rr.width / 2) - tr.left, (rr.top + rr.height / 2) - tr.top, scale * f); };
    const reset = () => { const t = target(); scale = 1; tx = 0; ty = 0; apply(t); };

    root.addEventListener('wheel', (e) => { const t = target(); if (!t) return; ensure(t);
      e.preventDefault(); const r = t.getBoundingClientRect();
      zoomAt(e.clientX - r.left, e.clientY - r.top, scale * (e.deltaY < 0 ? 1.1 : 1 / 1.1));
      }, { passive: false });
    root.addEventListener('pointerdown', (e) => { if (e.target.closest('.zoom-ctrl')) return;
      const t = target(); if (!t || scale <= 1) return;
      dragging = true; sx = e.clientX; sy = e.clientY; t.style.cursor = 'grabbing';
      try { t.setPointerCapture(e.pointerId); } catch (_) {} });
    root.addEventListener('pointermove', (e) => { if (!dragging) return; const t = target(); if (!t) return;
      tx += e.clientX - sx; ty += e.clientY - sy; sx = e.clientX; sy = e.clientY; apply(t); });
    const up = () => { dragging = false; const t = target(); if (t) t.style.cursor = 'grab'; };
    root.addEventListener('pointerup', up); root.addEventListener('pointerleave', up);
    root.addEventListener('dblclick', (e) => { if (e.target.closest('.zoom-ctrl')) return;
      e.preventDefault(); reset(); });

    // Explicit zoom controls.
    const bar = document.createElement('div');
    bar.className = 'zoom-ctrl';
    const mk = (txt, fn, title) => { const b = document.createElement('button');
      b.type = 'button'; b.textContent = txt; b.title = title;
      b.addEventListener('click', (ev) => { ev.preventDefault(); ev.stopPropagation(); fn(); });
      return b; };
    bar.appendChild(mk('\\u2212', () => centerZoom(1 / 1.3), 'Zoom out'));
    bar.appendChild(mk('+', () => centerZoom(1.3), 'Zoom in'));
    bar.appendChild(mk('\\u27f2', () => reset(), 'Reset'));
    root.appendChild(bar);

    new MutationObserver(() => { const t = target(); if (t) ensure(t);
      if (!root.querySelector('.zoom-ctrl')) root.appendChild(bar); })
      .observe(root, { childList: true, subtree: true });
  };
  const boot = () => { const r = document.querySelector('#doc-viewer');
    if (r) setup(r); else setTimeout(boot, 300); };
  boot();
}
"""


def zoom_js(selector: str) -> str:
    """The zoom/pan handler, bound to a specific viewer element id."""
    return ZOOM_JS.replace("#doc-viewer", selector)


# Pin the app to the light theme even when the OS/browser prefers dark.
FORCE_LIGHT_JS = """
() => {
  const light = () => {
    document.documentElement.classList.remove('dark');
    if (document.body) document.body.classList.remove('dark');
  };
  light();
  const obs = new MutationObserver(light);
  obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  if (document.body) obs.observe(document.body, { attributes: true, attributeFilter: ['class'] });
}
"""

_STEPS = [
    ("Received", 0), ("Read", 1), ("Extracted", 2), ("Details", 3),
    ("Ready", 4), ("Underwriting", 5), ("Decision", 6),
]
_RANK = {
    Stage.received: 0, Stage.reading: 1, Stage.extracting: 2, Stage.collecting: 3,
    Stage.ready: 4, Stage.underwriting: 5,
    Stage.auto_decided: 6, Stage.pending_review: 6, Stage.in_review: 6, Stage.finalized: 6,
}


def stepper_html(case: UnderwritingCase | None) -> str:
    cur = _RANK.get(case.stage, 0) if case else -1
    reviewing = bool(case) and case.stage in (Stage.pending_review, Stage.in_review)
    done_last = bool(case) and case.stage == Stage.finalized
    out = ['<div class="stepper">']
    for label, idx in _STEPS:
        if idx == 6 and reviewing:
            label = "In review"
        if idx < cur or (idx == 6 and done_last):
            cls = "step done"
        elif idx == cur and not (idx == 6 and done_last):
            cls = "step current"
        else:
            cls = "step"
        out.append(f'<span class="{cls}"><span class="dot"></span>{label}</span>')
    out.append("</div>")
    return "".join(out)


def _chip(k: str, v) -> str:
    return f'<span class="chip"><span class="k">{html.escape(k)}</span><span class="v">{html.escape(str(v))}</span></span>'


_PKG_MARK = {"received": "✓", "needs_review": "!", "missing": ""}


def checklist_html(case: UnderwritingCase | None, *, title: str = "Document package") -> str:
    """The standard-document checklist for a case, as status pills."""
    if not case or not case.documents:
        return ""
    pills = []
    for row in D.checklist(case):
        if not row["required"] and row["status"] == "missing":
            continue  # do not show optional docs that were never provided
        st = row["status"]
        mark = _PKG_MARK.get(st, "")
        opt = "" if row["required"] else " (optional)"
        pills.append(
            f'<span class="pkg-pill {st}"><span class="pk-mark">{mark}</span>'
            f'{html.escape(row["label"])}{opt}</span>')
    have = sum(1 for r in D.checklist(case) if r["required"] and r["status"] != "missing")
    need = sum(1 for r in D.checklist(case) if r["required"])
    return (f'<div class="pkg"><div class="pkg-title">{html.escape(title)} '
            f'<span class="pkg-count">{have}/{need} required</span></div>'
            f'<div class="pkg-row">{"".join(pills)}</div></div>')


def _outcome_badge(outcome: Outcome | None) -> str:
    cls = {Outcome.approve: "ok", Outcome.refer: "warn", Outcome.decline: "bad"}.get(outcome, "muted")
    text = outcome.value if outcome else "pending"
    return f'<span class="badge {cls}">{html.escape(text)}</span>'


def decision_html(case: UnderwritingCase | None, audience: str = "reviewer") -> str:
    """Render the decision panel. audience='reviewer' shows the full underwriting internals
    (risk score, rule confidence, DTI, and the internal rationale). audience='customer' shows
    only what an applicant should see: the outcome, a plain status line, and their own LTV. The
    reviewer view is a strict superset of the customer view, so internal signals never leak to
    the borrower."""
    if case is None or case.stage in (Stage.received, Stage.reading, Stage.extracting, Stage.collecting):
        return ""
    d, m = case.decision, case.metrics
    if case.stage == Stage.ready:
        return ('<div class="decision"><div class="row"><span class="badge muted">ready</span>'
                '<span class="title">Complete. Type submit to send for underwriting.</span></div></div>')
    outcome = case.review.outcome or d.recommendation
    reviewing = case.stage in (Stage.pending_review, Stage.in_review)

    if audience == "customer":
        # Applicant-facing: a plain status line and their own LTV only. No risk score, rule
        # confidence, DTI (modeled with an assumed rate), or internal rationale. While a case is
        # in review, show a neutral badge, not the model's recommendation; the outcome badge
        # appears only once a decision is final (auto-decided or a reviewer has finalized it).
        if reviewing:
            badge = '<span class="badge muted">in review</span>'
            title = "Your application is under review"
        else:
            badge = _outcome_badge(outcome)
            title = "Decision"
        chips = _chip("Loan-to-value", f"{m.ltv}%" if m.ltv is not None else "n/a")
        return (f'<div class="decision"><div class="row">{badge}'
                f'<span class="title">{html.escape(title)}</span></div>'
                f'<div class="chips">{chips}</div></div>')

    title = "With an underwriter for review" if reviewing else "Decision recorded"
    chips = "".join([
        _chip("Loan-to-value", f"{m.ltv}%" if m.ltv is not None else "n/a"),
        _chip("Debt-to-income", f"{m.dti}%" if m.dti is not None else "n/a"),
        _chip("risk", d.risk_score if d.risk_score is not None else "n/a"),
        _chip("confidence", d.confidence if d.confidence is not None else "n/a"),
    ])
    rationale = f'<div class="rationale">{html.escape(d.rationale)}</div>' if d.rationale else ""
    return (f'<div class="decision"><div class="row">{_outcome_badge(outcome)}'
            f'<span class="title">{html.escape(title)}</span></div>'
            f'<div class="chips">{chips}</div>{rationale}</div>')


def _metrics_help_md() -> str:
    """Reviewer reference: how each headline number is derived. Built from the live
    mortgage_rules thresholds so it never drifts from the policy the code enforces."""
    rate = R.DEFAULT_RATE * 100
    years = R.DEFAULT_TERM_MONTHS // 12
    return (
        f"**Loan-to-value (LTV)**: loan amount divided by property value, as a percent. "
        f"Policy limit {R.MAX_LTV:g}%, auto-approve ceiling {R.AUTO_MAX_LTV:g}%, "
        f"hard decline above {R.DECLINE_LTV:g}%.\n\n"
        f"**Debt-to-income (DTI)**: existing monthly debts plus the estimated new mortgage "
        f"payment, divided by gross monthly income, as a percent. The payment is estimated at "
        f"{rate:g}% over {years} years, not read from the application. "
        f"Policy limit {R.MAX_DTI:g}%, auto-approve ceiling {R.AUTO_MAX_DTI:g}%, "
        f"hard decline above {R.DECLINE_DTI:g}%.\n\n"
        f"**Risk**: a 0 to 100 score, higher is riskier. LTV adds points above 60% (up to 45), "
        f"DTI above 30% (up to 30), and a credit score below 740 (up to 25).\n\n"
        f"**Confidence**: the decision is rule-based code, so it reports 1.0 (certain). A case "
        f"auto-decides only when confidence is at least {R.MIN_CONFIDENCE:g}. The per-field "
        f"confidence behind a review flag is separate: it reflects how clearly a value was read "
        f"from the scan.\n\n"
        f"**Routing**: cases inside every auto ceiling auto-approve, hard-limit breaches "
        f"auto-decline, and everything in between comes here for a decision."
    )


def _header(eyebrow: str, title: str, sub: str) -> str:
    return (f'<div class="app-header"><a class="home-link" href="/">&larr; Home</a>'
            f'<div class="eyebrow">{html.escape(eyebrow)}</div>'
            f'<h1>{html.escape(title)}</h1><p>{html.escape(sub)}</p></div>')


# ---- Document preview + entity highlights ----
# make_preview (rasterization) lives in doc_render; entity-to-region matching lives in
# underwriting_service. Both are backend logic; the UI only draws what they return.
_OCR_PLACEHOLDER = "The text Mistral OCR 4 reads will appear here after you upload."


def doc_value(preview: str | None, case: UnderwritingCase | None, flags: set | None = None, doc=None):
    """gr.AnnotatedImage value. Before extraction: OCR blocks. After: labeled entities.

    `flags` is a set of field names needing review; a region holding any of them is
    marked so it stands out. `doc` selects a specific package document to overlay;
    otherwise the case's primary (most recent) document is used.
    """
    if not preview:
        return None
    blocks = (doc.ocr_blocks if doc else case.ocr_blocks) if (doc or case) else None
    width = doc.ocr_width if doc else (case.ocr_width if case else None)
    height = doc.ocr_height if doc else (case.ocr_height if case else None)
    if not case or not blocks or not width or not height:
        return (preview, [])
    flags = flags or set()
    try:
        from PIL import Image

        w, h = Image.open(preview).size
        sx, sy = w / width, h / height
        entities = S.entity_pairs_from(blocks, case.facts.model_dump())
        if entities:
            pairs = []
            for b, fields in entities:
                label = " / ".join(lbl for _, lbl in fields)
                if any(n in flags for n, _ in fields):
                    label = "⚠ " + label + " (review)"
                pairs.append((b, label))
        else:
            pairs = [(b, b.get("type", "text")) for b in blocks]
        anns = []
        for b, label in pairs:
            box = (int(b["x0"] * sx), int(b["y0"] * sy), int(b["x1"] * sx), int(b["y1"] * sy))
            anns.append((box, label))
        return (preview, anns)
    except Exception:
        return (preview, [])


_ENTITIES_PLACEHOLDER = "The extractor agent's entities appear here once a document is analyzed."


def _parse_entity(s: str) -> tuple[str, str, str]:
    """Best-effort parse of one stored entity into (value, type, context). The extractor stores
    either a plain string or the repr of a {type, value, context} dict; literal_eval is safe —
    it evaluates literals only, never code."""
    try:
        obj = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return (s, "", "")
    if isinstance(obj, dict):
        return (str(obj.get("value") or obj.get("name") or s),
                str(obj.get("type") or ""), str(obj.get("context") or ""))
    return (s, "", "")


def entities_markdown(case: UnderwritingCase | None) -> str:
    """Per-document extractor entities as a readable list, for the Entities tab. Replaces the
    raw dict dump that used to land in the chat."""
    if not case or not case.documents:
        return _ENTITIES_PLACEHOLDER
    sections = []
    for d in case.documents:
        rows = []
        for s in d.entities or []:
            value, typ, ctx = _parse_entity(s)
            meta = " · ".join(x for x in (typ, ctx) if x)
            rows.append(f"- **{value}**" + (f" — {meta}" if meta else ""))
        if rows:
            sections.append(f"### {d.type_label}\n" + "\n".join(rows))
    return "\n\n".join(sections) if sections else _ENTITIES_PLACEHOLDER


# Fields a reviewer should scrutinize: anything extraction flagged, plus the values
# that push a case past policy (the drivers of the referral).
def review_flags(case: UnderwritingCase | None) -> dict[str, str]:
    if not case:
        return {}
    flags: dict[str, str] = {}
    for mf in case.missing:
        flags[mf.name] = mf.reason.replace("_", " ")
    m = case.metrics
    # Hard policy breaches
    if m.ltv is not None and m.ltv > R.MAX_LTV:
        flags.setdefault("loan_amount", f"Loan-to-value {m.ltv}% exceeds the {R.MAX_LTV:g}% policy limit")
        flags.setdefault("property_value", f"Loan-to-value {m.ltv}% exceeds the {R.MAX_LTV:g}% policy limit")
    if m.dti is not None and m.dti > R.MAX_DTI:
        flags.setdefault("monthly_debts", f"Debt-to-income {m.dti}% exceeds the {R.MAX_DTI:g}% policy limit")
        flags.setdefault("annual_income", f"Debt-to-income {m.dti}% exceeds the {R.MAX_DTI:g}% policy limit")
    cs = case.facts.credit_score
    if cs is not None and cs < R.MIN_CREDIT:
        flags.setdefault("credit_score", f"below the {R.MIN_CREDIT} minimum")
    # Auto-approve threshold breaches — why this case needs human eyes
    if m.ltv is not None and R.AUTO_MAX_LTV < m.ltv <= R.MAX_LTV:
        flags.setdefault("loan_amount", f"Loan-to-value {m.ltv}% is above the auto-approve ceiling ({R.AUTO_MAX_LTV:g}%)")
        flags.setdefault("property_value", f"Loan-to-value {m.ltv}% is above the auto-approve ceiling ({R.AUTO_MAX_LTV:g}%)")
    if m.dti is not None and R.AUTO_MAX_DTI < m.dti <= R.MAX_DTI:
        flags.setdefault("monthly_debts", f"Debt-to-income {m.dti}% is above the auto-approve ceiling ({R.AUTO_MAX_DTI:g}%)")
        flags.setdefault("annual_income", f"Debt-to-income {m.dti}% is above the auto-approve ceiling ({R.AUTO_MAX_DTI:g}%)")
    if cs is not None and R.MIN_CREDIT <= cs < R.AUTO_MIN_CREDIT:
        flags.setdefault("credit_score", f"credit {cs} is below the auto-approve floor ({R.AUTO_MIN_CREDIT})")
    return flags


# ---- Sample documents (click to run) ----
_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "inputs")
SAMPLES = [
    ("Loan application (1003)", "loan_application_1003.pdf"),
    ("Pay stub", "paystub.pdf"),
    ("W-2", "w2.pdf"),
    ("Bank statement", "bank_statement.pdf"),
    ("Loan application, thin file", "loan_application_thin.pdf"),
    ("Loan application (clean)", "mortgage_clean.pdf"),
    ("Loan application (borderline)", "mortgage_borderline.pdf"),
]


def sample_thumbs():
    """Thumbnails for the gallery: the image itself, or a rendered PDF first page."""
    thumbs = []
    for label, fname in SAMPLES:
        p = os.path.join(_SAMPLE_DIR, fname)
        if os.path.exists(p):
            thumbs.append((make_preview(p) or p, label))
    return thumbs


# ---- Quick-reply suggestions per missing field (label, value-to-send) ----
_MAX_SUGG = 4
SUGGEST = {
    "loan_purpose": [("Purchase", "Purchase"), ("Refinance", "Refinance")],
    "annual_income": [("$90,000", "90000"), ("$120,000", "120000"), ("$150,000", "150000")],
    "monthly_debts": [("$0", "0"), ("$500", "500"), ("$1,500", "1500")],
    "credit_score": [("640", "640"), ("700", "700"), ("760", "760")],
    "employment_years": [("1 yr", "1"), ("3 yrs", "3"), ("6 yrs", "6")],
    "loan_amount": [("$300,000", "300000"), ("$400,000", "400000")],
    "property_value": [("$400,000", "400000"), ("$500,000", "500000")],
}


def suggestions_for(case) -> list[tuple[str, str]]:
    """Suggested quick replies for the current state. Free-text fields get none."""
    if case is None:
        return []
    if case.stage == Stage.ready:
        return [("Submit for underwriting", "submit")]
    if case.stage == Stage.collecting and case.missing:
        return SUGGEST.get(case.missing[0].name, [])
    return []


# ---- Customer view ----
def build_customer_ui() -> gr.Blocks:
    with gr.Blocks(title="Mortgage intake", theme=THEME, css=CSS) as ui:
        gr.HTML(f"<style>{CSS}</style>")
        gr.HTML(_header("Mortgage application", "Upload your document",
                        "Drop your loan document. Mistral OCR 4 reads it, and if anything is missing we ask you here."))
        gr.HTML('<div class="sample-tray"><span class="lbl">Try samples, click to add each to the '
                'application package</span></div>')
        samples = gr.Gallery(
            value=sample_thumbs(), columns=7, rows=1, height=132, object_fit="cover",
            show_label=False, allow_preview=False, elem_classes="sample-gallery",
        )
        with gr.Row(elem_classes="tray-actions"):
            clear_btn = gr.Button("Start over", size="sm", variant="secondary")

        case_state = gr.State(None)
        preview_state = gr.State(None)
        sugg_state = gr.State([])
        awaiting_state = gr.State(None)  # case_id we are polling a reviewer decision for
        stepper = gr.HTML(stepper_html(None))
        pkg = gr.HTML("", visible=False)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=300):
                chat = gr.Chatbot(height=430, show_label=False)
                box = gr.MultimodalTextbox(
                    file_count="single", file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    placeholder="Type an answer, or upload a document", show_label=False,
                )
                with gr.Row(elem_classes="suggestions"):
                    sugg_btns = [gr.Button(visible=False, size="sm", variant="secondary")
                                 for _ in range(_MAX_SUGG)]
                decision = gr.HTML("")
            with gr.Column(scale=7, min_width=440):
                doc_selector = gr.Dropdown(label="Document in package", choices=[],
                                           interactive=True, visible=False)
                with gr.Tabs():
                    with gr.Tab("Document"):
                        doc_image = gr.AnnotatedImage(
                            show_label=False, height=470, elem_id="doc-viewer")
                        gr.HTML('<div class="doc-hint">Scroll to zoom, drag to pan, '
                                'double-click to reset.</div>')
                    with gr.Tab("Read by Mistral OCR 4"):
                        ocr_md = gr.Markdown(_OCR_PLACEHOLDER)
                    with gr.Tab("Extracted entities"):
                        entities_md = gr.Markdown(_ENTITIES_PLACEHOLDER)

        outputs = [chat, case_state, preview_state, stepper, decision, box, doc_selector,
                   doc_image, ocr_md, entities_md, pkg]

        def _frame(history, case, preview, clear_overlay=False):
            ocr = case.markdown if (case and case.markdown) else _OCR_PLACEHOLDER
            has_docs = bool(case and case.documents)
            # While a freshly uploaded document is still being analyzed, show its page with no
            # overlay. The case-level blocks still describe the previous document, so drawing
            # them here would leave the old colored layers mis-scaled over the new scan.
            doc_val = (preview, []) if (clear_overlay and preview) else doc_value(preview, case)
            docs = case.documents if case else []
            # The selector always defaults to the most recent document, which is the one `preview`
            # (and the case-level overlay blocks) describe, so the two never diverge on a re-frame.
            sel = gr.update(
                choices=[(f"{d.type_label} — {d.status.replace('_', ' ')}", d.id) for d in docs],
                value=(docs[-1].id if docs else None), visible=bool(docs))
            return (history, (case.case_id if case else None), preview,
                    stepper_html(case), decision_html(case, audience="customer"),
                    gr.update(value=None), sel, gr.update(value=doc_val), ocr,
                    entities_markdown(case),
                    gr.update(value=checklist_html(case), visible=has_docs))

        async def _run_new(history, path):
            """Stream a fresh document through the pipeline, one frame per stage."""
            preview = make_preview(path)
            with open(path, "rb") as fh:
                data = fh.read()
            case = None
            try:
                async for case in S.create_case_stream(data, os.path.basename(path), owner="customer"):
                    yield _frame(history, case, preview, clear_overlay=True)
            except S.BackendError as exc:  # e.g. OCR rate limit: report it, keep the app alive
                history = history + [{"role": "assistant", "content": f"⚠ {exc}"}]
                yield _frame(history, case, preview, clear_overlay=True)
                return
            try:
                reply = await S.reply_for_agent(case)
            except S.BackendError as exc:  # the doc already streamed; a blip here must not crash the turn
                reply = f"⚠ {exc}"
            history = history + [{"role": "assistant", "content": reply}]
            yield _frame(history, case, preview)  # analysis done: paint the new document's boxes

        async def _add_doc(history, case_id, path):
            """Append another document to the current case, streamed."""
            history = history + [{"role": "user", "content": f"Uploaded {os.path.basename(path)}"}]
            preview = make_preview(path)
            with open(path, "rb") as fh:
                data = fh.read()
            # show the new page while it reads, with the prior document's boxes cleared
            yield _frame(history, S.get_case(case_id), preview, clear_overlay=True)
            case = None
            try:
                async for case in S.create_case_stream_add(case_id, data, os.path.basename(path)):
                    yield _frame(history, case, preview, clear_overlay=True)
            except S.BackendError as exc:  # e.g. OCR rate limit: report it, keep the app alive
                case = case or S.get_case(case_id)
                history = history + [{"role": "assistant", "content": f"⚠ {exc}"}]
                yield _frame(history, case, preview, clear_overlay=True)
                return
            case = case or S.get_case(case_id)
            try:
                reply = await S.reply_for_agent(case)
            except S.BackendError as exc:  # the doc already streamed; a blip here must not crash the turn
                reply = f"⚠ {exc}"
            history = history + [{"role": "assistant", "content": reply}]
            yield _frame(history, case, preview)  # analysis done: paint the new document's boxes

        async def _handle_text(text, history, case_id, preview):
            history = history + [{"role": "user", "content": text}]
            base = S.get_case(case_id) if case_id else None
            yield _frame(history, base, preview)
            try:
                if case_id is None:
                    reply, case = "Please upload your mortgage document to begin.", None
                elif text.strip().lower() in ("submit", "done", "go"):
                    case = None
                    async for case in S.submit_stream(case_id):
                        if case is not None:
                            yield _frame(history, case, preview)
                    case = case or S.get_case(case_id)
                    if case and case.stage == Stage.finalized:
                        reply = "Decision is in. See the summary below."
                    elif case and case.stage in (Stage.pending_review, Stage.in_review):
                        reply = f"Your application is complete and now with an underwriter. Case {case_id}."
                    else:
                        reply = await S.reply_for_agent(case)
                else:
                    case = await S.add_message(case_id, text, owner="customer")
                    reply = await S.reply_for_agent(case)
            except Exception as exc:  # keep the demo alive
                case = S.get_case(case_id) if case_id else None
                reply = f"Something went wrong: {exc}"
            history = history + [{"role": "assistant", "content": reply}]
            yield _frame(history, case, preview)

        _EDITABLE = (Stage.reading, Stage.extracting, Stage.collecting, Stage.ready)

        async def turn(message, history, case_id, preview):
            history = history or []
            text = (message or {}).get("text", "").strip()
            files = (message or {}).get("files", []) or []
            if not text and not files:
                return
            if files:
                case = S.get_case(case_id) if case_id else None
                if case and case.stage in _EDITABLE:  # add to the open package
                    async for frame in _add_doc(history, case_id, files[0]):
                        yield frame
                else:  # start a fresh application
                    fresh = [{"role": "user", "content": f"Uploaded {os.path.basename(files[0])}"}]
                    async for frame in _run_new(fresh, files[0]):
                        yield frame
                return
            async for frame in _handle_text(text, history, case_id, preview):
                yield frame

        async def on_sample(history, case_id, preview, evt: gr.SelectData):
            _, fname = SAMPLES[evt.index]
            path = os.path.join(_SAMPLE_DIR, fname)
            case = S.get_case(case_id) if case_id else None
            if case and case.stage in _EDITABLE:  # add the sample to the open package
                async for frame in _add_doc(history or [], case_id, path):
                    yield frame
            else:
                fresh = [{"role": "user", "content": f"Uploaded {fname}"}]
                async for frame in _run_new(fresh, path):
                    yield frame

        sugg_outputs = [sugg_state, *sugg_btns]

        def update_suggestions(case_id):
            case = S.get_case(case_id) if case_id else None
            pairs = suggestions_for(case)[:_MAX_SUGG]
            updates = []
            for i in range(_MAX_SUGG):
                if i < len(pairs):
                    updates.append(gr.update(value=pairs[i][0], visible=True))
                else:
                    updates.append(gr.update(value="", visible=False))
            return ([v for _, v in pairs], *updates)

        def make_click(index):
            async def handler(history, case_id, preview, values):
                if not values or index >= len(values):
                    return
                async for frame in _handle_text(values[index], history or [], case_id, preview):
                    yield frame
            return handler

        def set_awaiting(case_id):
            """Start polling once a case is handed to an underwriter."""
            case = S.get_case(case_id) if case_id else None
            if case and case.stage in (Stage.pending_review, Stage.in_review):
                return case_id
            return None

        def poll_decision(awaiting, history):
            """The return path: when the reviewer decides, surface it here and stop polling."""
            if not awaiting:
                return gr.update(), gr.update(), gr.update(), None
            case = S.get_case(awaiting)
            if case is None:
                return gr.update(), gr.update(), gr.update(), None
            if case.stage == Stage.finalized:
                outcome = case.review.outcome or case.decision.recommendation
                label = outcome.value if outcome else "decided"
                extra = f" Reviewer note: {case.review.note}" if case.review.note else ""
                msg = f"**An underwriter reviewed your application. Decision: {label}.**{extra}"
                history = (history or []) + [{"role": "assistant", "content": msg}]
                return history, stepper_html(case), decision_html(case, audience="customer"), None
            return gr.update(), stepper_html(case), gr.update(), awaiting  # still in review

        def on_doc(case_id, doc_id):
            """Switch the Document and OCR tabs to another document in the package. Transient:
            the next chat action re-frames to the most recent document (like the reviewer's
            reset-to-primary), so the viewer never diverges from the case's overlay blocks."""
            case = S.get_case(case_id) if case_id else None
            if not case or not case.documents:
                return gr.update(), gr.update()
            doc = next((d for d in case.documents if d.id == doc_id), None) or case.documents[-1]
            preview = make_preview(doc.path) if doc.path else None
            return gr.update(value=doc_value(preview, case, doc=doc)), (doc.markdown or _OCR_PLACEHOLDER)

        box.submit(turn, [box, chat, case_state, preview_state], outputs).then(
            update_suggestions, case_state, sugg_outputs).then(
            set_awaiting, case_state, awaiting_state)
        samples.select(on_sample, [chat, case_state, preview_state], outputs).then(
            update_suggestions, case_state, sugg_outputs).then(
            set_awaiting, case_state, awaiting_state)
        for i, sbtn in enumerate(sugg_btns):
            sbtn.click(make_click(i), [chat, case_state, preview_state, sugg_state], outputs).then(
                update_suggestions, case_state, sugg_outputs).then(
                set_awaiting, case_state, awaiting_state)

        def clear_all():
            """Reset the view to a blank application."""
            return _frame([], None, None)

        clear_btn.click(clear_all, None, outputs).then(
            update_suggestions, case_state, sugg_outputs).then(
            set_awaiting, case_state, awaiting_state)

        doc_selector.change(on_doc, [case_state, doc_selector], [doc_image, ocr_md])

        poll = gr.Timer(4.0)
        poll.tick(poll_decision, [awaiting_state, chat], [chat, stepper, decision, awaiting_state])
        ui.load(None, None, None, js=zoom_js("#doc-viewer"))
        ui.load(None, None, None, js=FORCE_LIGHT_JS)
    return ui


# ---- Reviewer view ----
# Editable fields in the reviewer, in display order. type: "text" or "num".
_REVIEW_FIELDS = [
    ("borrower_name", "Borrower", "text"), ("property_address", "Property address", "text"),
    ("loan_purpose", "Loan purpose", "text"), ("annual_income", "Annual income", "num"),
    ("loan_amount", "Loan amount", "num"), ("property_value", "Property value", "num"),
    ("credit_score", "Credit score", "num"), ("monthly_debts", "Monthly debts", "num"),
    ("employment_years", "Employment years", "num"),
]
_REVIEW_LABELS = {name: label for name, label, _ in _REVIEW_FIELDS}


def _flags_note_html(flags: dict[str, str]) -> str:
    if not flags:
        return ('<p class="review-ok">No values flagged. Verify against the document, '
                'edit any that look wrong, then decide.</p>')
    items = "".join(
        f"<li><b>{html.escape(_REVIEW_LABELS.get(n, n))}</b>: {html.escape(r)}</li>"
        for n, r in flags.items())
    return (f'<div class="review-flags"><div class="rf-title">Values to check</div>'
            f'<ul>{items}</ul></div>')


# ---- Decision ledger (finalized cases; before/after and outcome) ----
_LEDGER_MONEY = {"annual_income", "loan_amount", "property_value", "monthly_debts"}


def _ledger_fmt(name: str, value) -> str:
    if value is None or value == "":
        return "—"
    try:
        if name in _LEDGER_MONEY:
            return f"${float(value):,.0f}"
        if name == "credit_score":
            return str(int(float(value)))
        if name == "employment_years":
            return f"{float(value):g} yrs"
    except (TypeError, ValueError):
        pass
    return html.escape(str(value))


def _ledger_time(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts).astimezone().strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return ""


def _ledger_changes(case: UnderwritingCase) -> str:
    """The before/after cell: fields a reviewer corrected, plus any recomputed metric shift.

    'Before' is the snapshot taken at underwriting time (submitted_facts/metrics); 'after' is the
    live, corrected value. Auto decisions and clean approvals show no corrections.
    """
    before = case.submitted_facts
    if before is None:
        return '<span class="led-none">— not captured</span>'
    b, a = before.model_dump(), case.facts.model_dump()
    rows = []
    for name, label, _typ in _REVIEW_FIELDS:
        bv, av = b.get(name), a.get(name)
        if bv != av:
            rows.append(
                f'<div class="led-chg"><span class="led-f">{html.escape(label)}:</span> '
                f'<span class="led-b">{_ledger_fmt(name, bv)}</span>'
                f'<span class="led-arrow">→</span>'
                f'<span class="led-a">{_ledger_fmt(name, av)}</span></div>')
    mb = case.submitted_metrics
    if mb is not None:
        for mlabel, attr in (("LTV", "ltv"), ("DTI", "dti")):
            ov, nv = getattr(mb, attr, None), getattr(case.metrics, attr, None)
            if ov != nv:
                pct = lambda v: f"{v}%" if v is not None else "n/a"
                rows.append(
                    f'<div class="led-chg led-metric"><span class="led-f">{mlabel}:</span> '
                    f'<span class="led-b">{pct(ov)}</span>'
                    f'<span class="led-arrow">→</span>'
                    f'<span class="led-a">{pct(nv)}</span></div>')
    if not rows:
        return '<span class="led-none">— no corrections</span>'
    return "".join(rows)


def ledger_html(cases: list[UnderwritingCase]) -> str:
    """A read-only table of finalized decisions (manual and automatic), newest first, with the
    values before and after any reviewer correction. Kept separate from the single-case panel."""
    if not cases:
        return '<p class="muted-note">No decisions yet.</p>'
    head = ('<tr><th>Time</th><th>Case</th><th>Borrower</th><th>Outcome</th>'
            '<th>Decided by</th><th>Before &rarr; after</th></tr>')
    rows = []
    for c in cases:
        outcome = c.review.outcome or (c.decision.recommendation if c.decision else None)
        auto = (c.review.reviewer or "") == "auto"
        who = "Auto" if auto else f"Manual · {html.escape(c.review.reviewer or 'reviewer')}"
        if c.review.overrode:
            who += '<span class="badge warn led-ovr">override</span>'
        sid = c.case_id.replace("case-", "")[:6]
        rows.append(
            f'<tr><td class="led-time">{_ledger_time(c.review.at or c.updated_at)}</td>'
            f'<td class="led-id">{html.escape(sid)}</td>'
            f'<td>{html.escape(c.facts.borrower_name or "—")}</td>'
            f'<td>{_outcome_badge(outcome)}</td>'
            f'<td>{who}</td>'
            f'<td class="led-changes">{_ledger_changes(c)}</td></tr>')
    return f'<div class="ledger"><table class="ledger-table">{head}{"".join(rows)}</table></div>'


def build_reviewer_ui() -> gr.Blocks:
    with gr.Blocks(title="Underwriter review", theme=THEME, css=CSS) as ui:
        gr.HTML(f"<style>{CSS}</style>")
        gr.HTML(_header("Underwriter review", "Cases awaiting a decision",
                        "Open a flagged case, check the values against the source document, "
                        "correct anything wrong, then decide."))
        # Two top-level tabs: the single-case workspace, and a separate decision ledger. The
        # ledger never shares screen with the case being reviewed.
        with gr.Tabs():
            with gr.Tab("Review case"):
                with gr.Row():
                    queue = gr.Dropdown(label="Pending cases", choices=[], interactive=True, scale=3)
                    refresh = gr.Button("Refresh", scale=1)
                stepper = gr.HTML("")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=340):
                        detail = gr.HTML('<p class="muted-note">Select a case to review.</p>')
                        pkg_review = gr.HTML("")
                        flags_note = gr.HTML("")
                        with gr.Accordion("How these numbers are calculated", open=False):
                            gr.Markdown(_metrics_help_md())
                        with gr.Accordion("Extracted values (edit to correct)", open=True):
                            def _field(name, label, typ):
                                if typ == "num":
                                    return gr.Number(label=label, precision=0 if name == "credit_score" else None)
                                return gr.Textbox(label=label)

                            field_inputs = []
                            for i in range(0, len(_REVIEW_FIELDS), 2):  # two fields per row
                                with gr.Row(elem_classes="fields-grid"):
                                    for name, label, typ in _REVIEW_FIELDS[i:i + 2]:
                                        field_inputs.append(_field(name, label, typ))
                            save_btn = gr.Button("Save corrections", variant="secondary")
                        with gr.Row():
                            outcome = gr.Radio([o.value for o in Outcome], label="Decision", value="approve", scale=2)
                            note = gr.Textbox(label="Note", placeholder="Reason or compensating factors", scale=3)
                        decide_btn = gr.Button("Submit decision", variant="primary")
                        status = gr.HTML("")
                    with gr.Column(scale=7, min_width=440):
                        doc_selector = gr.Dropdown(label="Document in package", choices=[], interactive=True)
                        with gr.Tabs():
                            with gr.Tab("Document"):
                                doc_image = gr.AnnotatedImage(
                                    show_label=False, height=470, elem_id="doc-viewer-review")
                                gr.HTML('<div class="doc-hint">Flagged regions are marked. '
                                        'Scroll to zoom, drag to pan, double-click to reset.</div>')
                            with gr.Tab("Read by Mistral OCR 4"):
                                ocr_md = gr.Markdown("")

            with gr.Tab("Decision ledger"):
                gr.HTML('<div class="doc-hint">Finalized decisions, manual and automatic, newest '
                        'first, with the values before and after any reviewer correction.</div>')
                ledger = gr.HTML("")

        # One render path everything reuses. Output order is fixed:
        #   detail, pkg_review, flags_note, stepper, doc_selector, doc_image, ocr_md, *field_inputs
        render_outputs = [detail, pkg_review, flags_note, stepper, doc_selector,
                          doc_image, ocr_md, *field_inputs]

        def _doc_view(case: UnderwritingCase, doc_id):
            """Overlay + OCR text for one document in the package (default: the primary)."""
            if not case or not case.documents:
                return None, "No document uploaded."
            doc = next((d for d in case.documents if d.id == doc_id), None) or case.documents[-1]
            preview = make_preview(doc.path) if doc.path else None
            val = doc_value(preview, case, set(review_flags(case)), doc=doc) if preview else None
            return val, (doc.markdown or "No text read for this document.")

        def _render(case: UnderwritingCase | None):
            if not case:
                blanks = [gr.update(value=None) for _ in _REVIEW_FIELDS]
                return ['<p class="muted-note">Select a case to review.</p>', "", "", "",
                        gr.update(choices=[], value=None), None, "", *blanks]
            flags = review_flags(case)
            facts = case.facts.model_dump()
            updates = [
                gr.update(value=facts.get(name), label=label + ("  ⚠ review" if name in flags else ""))
                for name, label, _ in _REVIEW_FIELDS
            ]
            primary = case.documents[-1].id if case.documents else None
            sel = gr.update(
                choices=[(f"{d.type_label} — {d.status.replace('_', ' ')}", d.id) for d in case.documents],
                value=primary)
            doc_val, ocr = _doc_view(case, primary)
            return [decision_html(case), checklist_html(case, title="Application package"),
                    _flags_note_html(flags), stepper_html(case), sel, doc_val, ocr, *updates]

        def on_doc(case_id, doc_id):
            case = S.get_case(case_id) if case_id else None
            no_label_updates = [gr.update() for _ in _REVIEW_FIELDS]
            if not case:
                return None, "", *no_label_updates
            flags = review_flags(case)
            doc = next((d for d in case.documents if d.id == doc_id), None) or (
                case.documents[-1] if case.documents else None)
            if not doc:
                return None, "", *no_label_updates
            preview = make_preview(doc.path) if doc.path else None
            val = doc_value(preview, case, set(flags), doc=doc) if preview else None
            ocr = doc.markdown or "No text read for this document."
            pairs = S.entity_pairs_from(doc.ocr_blocks or [], case.facts.model_dump())
            in_doc = {name for _, fields in pairs for name, _ in fields}
            label_updates = [
                gr.update(label=label
                          + (" (here)" if name in in_doc else "")
                          + ("  ⚠ review" if name in flags else ""))
                for name, label, _ in _REVIEW_FIELDS
            ]
            return val, ocr, *label_updates

        def _queue_label(c) -> str:
            who = c.facts.borrower_name or "Applicant"
            loan = c.facts.loan_amount
            amt = f"${loan:,.0f}" if isinstance(loan, (int, float)) else "loan n/a"
            rec = (c.decision.recommendation.value
                   if c.decision and c.decision.recommendation else c.stage.value)
            sid = c.case_id.replace("case-", "")[:6]
            when = ""
            ts = (c.audit[-1].at if c.audit else None) or c.updated_at
            if ts:
                try:  # show local submit time so duplicates are easy to tell apart
                    from datetime import datetime
                    when = " · " + datetime.fromisoformat(ts).astimezone().strftime("%H:%M")
                except (ValueError, TypeError):
                    when = ""
            return f"{who} · {amt} · {rec}{when} · {sid}"

        def _queue_choices():
            # Newest submission first so a just-submitted case sits at the top. Each entry shows
            # borrower, loan amount, recommendation, and a short id; the value stays the case_id.
            cases = sorted(S.list_pending(),
                           key=lambda c: (c.audit[-1].at if c.audit else ""), reverse=True)
            return [(_queue_label(c), c.case_id) for c in cases]

        def load_queue():
            choices = _queue_choices()
            return gr.update(choices=choices, value=choices[0][1] if choices else None)

        def load_ledger():
            return ledger_html(S.list_finalized())

        def poll_queue(current):
            """Live queue: pull new pending cases in. Only auto-select when idle, so an
            in-progress review (and any unsaved edits) is never disturbed."""
            choices = _queue_choices()
            if not current and choices:
                return gr.update(choices=choices, value=choices[0][1])
            return gr.update(choices=choices)

        def show(case_id):
            return _render(S.get_case(case_id) if case_id else None)

        async def save_corrections(case_id, *values):
            if not case_id:
                return [*_render(None), '<p class="muted-note">No case selected.</p>']
            updates = {}
            for (name, _, typ), v in zip(_REVIEW_FIELDS, values):
                updates[name] = v if typ == "num" else (v or None)
            case = await S.apply_corrections(case_id, updates, reviewer="reviewer")
            note = ('<p class="review-ok">Saved. Metrics and the recommendation were recomputed '
                    'from the corrected values.</p>')
            return [*_render(case), note]

        async def submit_decision(case_id, outcome_val, note_val, *values):
            if not case_id:
                return ['<p class="muted-note">No case selected.</p>', load_queue(), *_render(None)]
            # Persist any pending edits in the correction form before deciding, so a reviewer who
            # changes a field and clicks Submit (without first clicking Save) does not lose the
            # edit — and the decision ledger's before/after reflects it. Only re-run corrections
            # when a value actually changed, to avoid a needless recompute.
            edited = False
            current = S.get_case(case_id)
            if current is not None:
                cf = current.facts.model_dump()
                updates = {}
                for (name, _, typ), v in zip(_REVIEW_FIELDS, values):
                    updates[name] = v if typ == "num" else (v or None)
                # An intentional clear (a blanked field becomes None) is a real change too, so
                # compare every field to the current value rather than skipping None.
                if any(cf.get(name) != updates[name] for name, _, _ in _REVIEW_FIELDS):
                    await S.apply_corrections(case_id, updates, reviewer="reviewer")
                    edited = True
            case = S.decide(case_id, reviewer="reviewer", outcome=Outcome(outcome_val), note=note_val or "")
            tag = " (override)" if case.review.overrode else ""
            saved = " Applied your edits first." if edited else ""
            msg = (f'<p class="review-ok">Recorded {html.escape(case.review.outcome.value)}{tag} '
                   f'on {html.escape(case.case_id)}.{saved}</p>')
            return [msg, load_queue(), *_render(case)]

        refresh.click(load_queue, None, queue).then(load_ledger, None, ledger)
        queue.change(show, queue, render_outputs)
        doc_selector.change(on_doc, [queue, doc_selector], [doc_image, ocr_md, *field_inputs])
        save_btn.click(save_corrections, [queue, *field_inputs], [*render_outputs, status])
        decide_btn.click(submit_decision, [queue, outcome, note, *field_inputs],
                         [status, queue, *render_outputs]).then(load_ledger, None, ledger)
        queue_timer = gr.Timer(5.0)
        queue_timer.tick(poll_queue, queue, queue).then(load_ledger, None, ledger)
        ui.load(load_queue, None, queue).then(show, queue, render_outputs).then(load_ledger, None, ledger)
        ui.load(None, None, None, js=zoom_js("#doc-viewer-review"))
        ui.load(None, None, None, js=FORCE_LIGHT_JS)
    return ui


# ---- Chat with your document (any file Mistral OCR 4 supports) ----
DOC_SAMPLES = [
    ("NVIDIA 10-Q (financials)", "nvidia_10q_financials.pdf"),
    ("Receipt (scan)", "scanned_receipt.png"),
    ("Invoice (PDF)", "sample_invoice.pdf"),
    ("Pay stub", "paystub.pdf"),
    ("Bank statement", "bank_statement.pdf"),
    ("W-2", "w2.pdf"),
]


def doc_sample_thumbs():
    return [(make_preview(os.path.join(_SAMPLE_DIR, f)), label) for label, f in DOC_SAMPLES]


def blocks_overlay(preview, blocks, width, height):
    """gr.AnnotatedImage value: every OCR 4 layout block boxed and labeled by its type.
    The coordinate scaling is done in ui_render.overlay_boxes; this only shapes the result
    into the (image, annotations) tuple the component expects."""
    if not preview:
        return None
    return (preview, overlay_boxes(preview, blocks, width, height))


def doc_info_html(annotation):
    if not annotation:
        return ""
    dt = annotation.get("document_type") or "document"
    lang = annotation.get("language") or ""
    summ = annotation.get("summary") or ""
    lang_bit = f" ({html.escape(lang)})" if lang else ""
    return (f'<div class="doc-hint"><b>Mistral OCR 4 read a {html.escape(dt)}</b>{lang_bit}. '
            f'{html.escape(summ)}</div>')


def build_docchat_ui() -> gr.Blocks:
    with gr.Blocks(title="Chat with your document", theme=THEME, css=CSS) as ui:
        gr.HTML(f"<style>{CSS}</style>")
        gr.HTML(_header("Document chat", "Chat with your document",
                        "Upload any document. Mistral OCR 4 reads it and shows its layout blocks "
                        "by type, and a chat agent answers questions grounded in the text."))
        gr.HTML('<div class="sample-tray"><span class="lbl">Try a sample, or upload your own '
                'PDF or image</span></div>')
        samples = gr.Gallery(value=doc_sample_thumbs(), columns=5, rows=1, height=132,
                             object_fit="cover", show_label=False, allow_preview=False,
                             elem_classes="sample-gallery")
        with gr.Row(elem_classes="tray-actions"):
            clear_btn = gr.Button("Start over", size="sm", variant="secondary")

        md_state = gr.State("")
        preview_state = gr.State(None)
        # (page_images, pages_layout): page_images are pre-rendered PNG paths (one per page);
        # pages_layout = [{"blocks","width","height"}, ...]. Paging is a lookup, no re-render.
        blocks_state = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=300):
                chat = gr.Chatbot(height=470, show_label=False)
                box = gr.MultimodalTextbox(
                    file_count="single",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"],
                    placeholder="Upload a document, or ask a question about it", show_label=False)
                info = gr.HTML("")
            with gr.Column(scale=7, min_width=440):
                # Page selector for multi-page documents (hidden until there is more than one page).
                # Starts with a valid range (min<max); actual page count is set after OCR.
                page_sel = gr.Slider(minimum=1, maximum=2, step=1, value=1, label="Page",
                                     interactive=True, visible=False)
                with gr.Tabs():
                    with gr.Tab("Document blocks"):
                        doc_image = gr.AnnotatedImage(show_label=False, height=470,
                                                      elem_id="doc-viewer-chat")
                        gr.HTML('<div class="doc-hint">Boxes are Mistral OCR 4 layout blocks, '
                                'colored by type. Use the Page slider for multi-page documents. '
                                'Scroll to zoom, drag to pan.</div>')
                    with gr.Tab("Blocks detail"):
                        blocks_md = gr.Markdown("_Upload a document to see its layout blocks._")
                    with gr.Tab("Read by Mistral OCR 4"):
                        ocr_md = gr.Markdown("_The text Mistral OCR 4 reads will appear here._")

        outputs = [chat, md_state, preview_state, blocks_state, doc_image, blocks_md, ocr_md,
                   info, box, page_sel]

        def _page_frame(page_images, pages_layout, page_idx):
            """Overlay + summary for one page (0-based), using the already-rendered page image
            so paging is a fast lookup, not a re-render."""
            layout = (pages_layout or [{}])[page_idx] if pages_layout else {}
            blocks = layout.get("blocks") or []
            preview = page_images[page_idx] if page_images and page_idx < len(page_images) else None
            overlay = blocks_overlay(preview, blocks, layout.get("width"), layout.get("height"))
            return overlay, blocks_table(blocks)

        def _frame(history, md, preview, bt, annotation):
            page_images = bt[0] if bt else None
            pages_layout = bt[1] if bt else None
            npages = len(pages_layout) if pages_layout else 0
            overlay, summary = (_page_frame(page_images, pages_layout, 0) if npages
                                else (((preview, []) if preview else None),
                                      "_Upload a document to see its layout blocks._"))
            # maximum stays >= 2 so the Slider never has min==max (Gradio rejects that); the
            # slider is simply hidden when there is only one page.
            page_update = gr.update(visible=npages > 1, maximum=max(npages, 2), value=1)
            return (history, md, preview, bt,
                    gr.update(value=overlay), summary, md or "_No text yet._",
                    doc_info_html(annotation), gr.update(value=None), page_update)

        def on_page(bt, page_num):
            """Swap the overlay + detail to a different page (images are pre-rendered)."""
            if not bt:
                return gr.update(), gr.update()
            page_images, pages_layout = bt
            idx = int(page_num) - 1
            if not pages_layout or idx < 0 or idx >= len(pages_layout):
                return gr.update(), gr.update()
            overlay, summary = _page_frame(page_images, pages_layout, idx)
            return gr.update(value=overlay), summary

        async def _read(history, path):
            preview = make_preview(path)
            with open(path, "rb") as fh:
                data = fh.read()
            # interim: show page 1 while OCR runs
            yield (history, "", preview, None,
                   gr.update(value=(preview, []) if preview else None),
                   "_Reading with Mistral OCR 4..._", "_Reading..._", "", gr.update(value=None),
                   gr.update(visible=False))
            try:
                r = S.analyze(data, os.path.basename(path))
            except Exception as exc:
                history = history + [{"role": "assistant", "content": f"Could not read that document: {exc}"}]
                yield _frame(history, "", preview, None, None)
                return
            md = r.get("markdown", "")
            pages_layout = r.get("pages_layout") or []
            total_blocks = sum(len(p.get("blocks") or []) for p in pages_layout)
            npages = len(pages_layout)
            # Pre-render every page image once, so paging is an instant lookup and does not
            # depend on the (possibly temporary) uploaded file surviving.
            page_images = [make_preview(path, i) for i in range(npages)] if npages else []
            ann = r.get("annotation") or {}
            dt = ann.get("document_type") or "document"
            pages_bit = f" across {npages} pages" if npages > 1 else ""
            history = history + [{"role": "assistant", "content":
                f"Mistral OCR 4 read your **{dt}** and found {total_blocks} layout blocks"
                f"{pages_bit}. Ask me anything about it."}]
            yield _frame(history, md, preview, (page_images, pages_layout), ann)

        async def turn(message, history, md, preview, bt):
            history = history or []
            text = (message or {}).get("text", "").strip()
            files = (message or {}).get("files", []) or []
            if not text and not files:
                return
            if files:
                fresh = history + [{"role": "user", "content": f"Uploaded {os.path.basename(files[0])}"}]
                async for fr in _read(fresh, files[0]):
                    yield fr
                return
            history = history + [{"role": "user", "content": text}]
            # keep: doc_image, blocks_md, ocr_md, info, page_sel unchanged during a Q&A turn
            keep = (gr.update(), gr.update(), gr.update(), gr.update())
            yield (history, md, preview, bt, *keep, gr.update(value=None), gr.update())
            try:
                reply = await S.answer(md, text)
            except S.BackendError as exc:  # a backend blip or timeout must not crash the chat turn
                reply = f"⚠ {exc}"
            history = history + [{"role": "assistant", "content": reply}]
            yield (history, md, preview, bt, *keep, gr.update(value=None), gr.update())

        async def on_sample(history, evt: gr.SelectData):
            _, fname = DOC_SAMPLES[evt.index]
            fresh = (history or []) + [{"role": "user", "content": f"Uploaded {fname}"}]
            async for fr in _read(fresh, os.path.join(_SAMPLE_DIR, fname)):
                yield fr

        def clear_all():
            return _frame([], "", None, None, None)

        box.submit(turn, [box, chat, md_state, preview_state, blocks_state], outputs)
        samples.select(on_sample, [chat], outputs)
        # `.release` fires once when the drag ends, not on every intermediate value, so paging
        # swaps to one page instead of flooding updates (and racing) mid-drag.
        page_sel.release(on_page, [blocks_state, page_sel], [doc_image, blocks_md])
        clear_btn.click(clear_all, None, outputs)
        ui.load(None, None, None, js=zoom_js("#doc-viewer-chat"))
        ui.load(None, None, None, js=FORCE_LIGHT_JS)
    return ui


# Gradio only serves files under allowed paths; include the samples dir and temp
# (rendered PDF previews and uploads) so thumbnails and document images show.
_ALLOWED = [os.path.abspath(_SAMPLE_DIR), tempfile.gettempdir()]
ui = gr.mount_gradio_app(ui, build_customer_ui(), path="/apply", allowed_paths=_ALLOWED)
ui = gr.mount_gradio_app(ui, build_reviewer_ui(), path="/review", allowed_paths=_ALLOWED)
ui = gr.mount_gradio_app(ui, build_docchat_ui(), path="/chat", allowed_paths=_ALLOWED)


if __name__ == "__main__":
    # Convenience launcher so `python app.py` serves the UI (landing at /, customer view at
    # /apply, reviewer at /review, document chat at /chat). Same as `uvicorn app:ui`.
    # The backend must already be running (see scripts/up.sh); BACKEND_URL points at it.
    # Override host/port with APP_HOST / APP_PORT.
    import uvicorn

    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "8000"))
    print(f"Serving UI on http://{host}:{port}  (apply: /apply, review: /review, chat: /chat)")
    print(f"Backend: {S.BACKEND_URL}")
    uvicorn.run(ui, host=host, port=port)
