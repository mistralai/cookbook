"""
Gradio web application for Invoice OCR Analyzer.

Features:
  - Upload PDF → OCR with Mistral Document AI
  - Page-by-page verification view (per-page text + tables)
  - Side-by-side PDF viewer + extracted text
  - Invoice line item extraction with math verification
  - Financial totals verification and discrepancy detection
  - OCR correction log with quality metrics
  - Invoice field summary table
"""

import base64
import json
import logging
import os
import re
import shutil
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from openai import AzureOpenAI

import config
from ocr_engine import OCRResult, process_pdf
from ocr_correction import (
    CorrectionReport,
    CompletenessReport,
    correct_ocr_pages,
    check_completeness,
)
from invoice_analyzer import (
    InvoiceAnalysisReport,
    FieldStatus,
    analyse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Globals for session state ─────────────────────────────────────────────────
_last_ocr: Optional[OCRResult] = None
_last_correction: Optional[CorrectionReport] = None
_last_completeness: Optional[CompletenessReport] = None
_last_analysis: Optional[InvoiceAnalysisReport] = None
_last_pdf_path: Optional[str] = None


# ── Chat helpers ──────────────────────────────────────────────────────────────

_chat_client: Optional[AzureOpenAI] = None


def _get_chat_client() -> Optional[AzureOpenAI]:
    """Return a cached AzureOpenAI client for the chat deployment."""
    global _chat_client
    if _chat_client is not None:
        return _chat_client
    if not config.CHAT_AZURE_ENDPOINT:
        return None
    if config.CHAT_API_KEY:
        _chat_client = AzureOpenAI(
            azure_endpoint=config.CHAT_AZURE_ENDPOINT,
            api_key=config.CHAT_API_KEY,
            api_version=config.CHAT_API_VERSION,
        )
        return _chat_client
    try:
        from ocr_engine import _get_azure_ad_token
        token = _get_azure_ad_token()
        if token:
            _chat_client = AzureOpenAI(
                azure_endpoint=config.CHAT_AZURE_ENDPOINT,
                api_key=token,
                api_version=config.CHAT_API_VERSION,
            )
            return _chat_client
    except Exception as exc:
        logger.warning("Azure AD token acquisition failed: %s", exc)
    return None


def _build_system_prompt() -> str:
    """Build a system prompt that includes invoice document context if available."""
    base = (
        "You are a helpful invoice and financial document assistant. You help users "
        "understand their invoices, verify line items, identify discrepancies, and "
        "answer questions about financial documents that have been scanned and analyzed.\n"
        "Be clear, accurate, and concise. When discussing totals or line items, "
        "highlight any discrepancies found. If you're unsure about something, say so.\n"
    )
    if _last_analysis and _last_analysis.fields:
        lines = []
        for f in _last_analysis.fields:
            status_str = f.status.value if f.status else "unknown"
            note = f" ({f.note})" if f.note else ""
            lines.append(f"  {f.display_name}: {f.value_str}{note} -> {status_str}")
        base += "\n--- EXTRACTED INVOICE FIELDS ---\n" + "\n".join(lines) + "\n--- END FIELDS ---\n"
        if _last_analysis.line_items:
            base += f"\n--- LINE ITEMS ({len(_last_analysis.line_items)}) ---\n"
            for li in _last_analysis.line_items[:20]:
                base += f"  {li.description}: qty={li.quantity}, price={li.unit_price}, amount={li.amount} [{li.status.value}]\n"
            base += "--- END LINE ITEMS ---\n"
        if _last_analysis.discrepancies:
            base += "\n--- DISCREPANCIES ---\n"
            for d in _last_analysis.discrepancies:
                base += f"  ⚠️  {d}\n"
            base += "--- END DISCREPANCIES ---\n"
    if _last_correction and _last_correction.corrected_text:
        doc_text = _last_correction.corrected_text[:3000]
        base += f"\n--- DOCUMENT EXCERPT ---\n{doc_text}\n--- END EXCERPT ---\n"
    return base


def _chat_completion(messages: list) -> str:
    """Call the chat model via OpenAI SDK."""
    client = _get_chat_client()
    if client is None:
        return "Chat is not configured. Set CHAT_AZURE_ENDPOINT and CHAT_API_KEY in .env."
    try:
        resp = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=1024,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        logger.error("Chat request failed: %s", exc)
        return f"Sorry, the chat service returned an error: {exc}"


def chat_respond(message: str, history: list) -> Tuple[str, list]:
    """Handle a user message and return (cleared input, updated history)."""
    if not message.strip():
        return "", history
    api_messages = [{"role": "system", "content": _build_system_prompt()}]
    for msg in history:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    api_messages.append({"role": "user", "content": message})
    reply = _chat_completion(api_messages)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history


# ── Style constants (Mistral brand palette) ───────────────────────────────────
S = {
    "bg": "#0A0A0A", "bg2": "#1E1E1E", "bg3": "#2A2A2A",
    "border": "#3A3A3A", "border_light": "#2A2A2A",
    "text": "#F5F5F5", "text2": "#E8E8E8", "text3": "#CCCCCC", "text_dim": "#999999",
    "green": "#22C55E", "green_bg": "#052A14", "green_border": "#0A4020", "green_text": "#86EFAC",
    "amber": "#FFD800", "amber_bg": "#2A1F00", "amber_border": "#4A3500", "amber_text": "#FFE980",
    "red": "#FF5555", "red_bg": "#2A0000", "red_border": "#5C0000", "red_text": "#FFAAAA",
    "blue": "#FF8205", "blue_bg": "#2A1500", "blue_border": "#5C2E00", "blue_text": "#FFD5A0",
    "purple": "#FFAF00",
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _pdf_to_images_html(pdf_path: str) -> str:
    """Render each PDF page as a base64 <img> (works in all browsers / iframe contexts)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        imgs = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode()
            imgs.append(
                f'<div style="margin-bottom:8px;">'
                f'<div style="padding:4px 8px;background:{S["bg3"]};color:{S["text_dim"]};font-size:0.8em;">Page {i+1}</div>'
                f'<img src="data:image/png;base64,{img_b64}" style="width:100%;display:block;"/>'
                f'</div>'
            )
        doc.close()
        return "\n".join(imgs)
    except Exception as exc:
        try:
            b64 = base64.b64encode(Path(pdf_path).read_bytes()).decode()
            return (
                f'<iframe src="data:application/pdf;base64,{b64}" '
                f'width="100%" height="800px" style="border:none;border-radius:8px;"></iframe>'
            )
        except Exception:
            return f"<p style='color:{S['red']};'>Error loading PDF: {exc}</p>"


def _status_badge(status: FieldStatus) -> str:
    colors = {
        FieldStatus.MATCHED: S["green"],
        FieldStatus.EXTRACTED: "#6b7280",
        FieldStatus.FLAGGED: "#f97316",
        FieldStatus.CRITICAL: "#ef4444",
        FieldStatus.MISSING: "#ef4444",
        FieldStatus.UNKNOWN: "#6b7280",
    }
    labels = {
        FieldStatus.MATCHED: "✓ VERIFIED",
        FieldStatus.EXTRACTED: "EXTRACTED",
        FieldStatus.FLAGGED: "⚠ FLAGGED",
        FieldStatus.CRITICAL: "⚠ DISCREPANCY",
        FieldStatus.MISSING: "MISSING",
        FieldStatus.UNKNOWN: "—",
    }
    color = colors.get(status, "#6b7280")
    label = labels.get(status, "?")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600;">{label}</span>'
    )


def _extract_markdown_tables(text: str) -> List[str]:
    """Extract markdown tables from text."""
    tables = []
    lines = text.split("\n")
    current_table: List[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            current_table.append(stripped)
        else:
            if in_table and current_table:
                data_rows = [r for r in current_table if not re.match(r'^\|[\s\-:|]+\|$', r)]
                if len(data_rows) >= 1:
                    tables.append("\n".join(current_table))
                current_table = []
            in_table = False

    if current_table:
        data_rows = [r for r in current_table if not re.match(r'^\|[\s\-:|]+\|$', r)]
        if len(data_rows) >= 1:
            tables.append("\n".join(current_table))

    return tables


def _md_table_to_html(md_table: str) -> str:
    """Convert a markdown table to styled HTML table, filtering empty cells."""
    rows = md_table.strip().split("\n")
    html_rows = []

    for i, row in enumerate(rows):
        raw_cells = [c.strip() for c in row.split("|")[1:-1]]
        if not raw_cells:
            continue
        if all(re.match(r'^[\-:]+$', c.strip()) for c in raw_cells if c.strip()):
            continue

        cells = [c for c in raw_cells if c]
        if not cells:
            continue

        tag = "th" if i == 0 else "td"
        bg = S["bg3"] if i == 0 else ""
        style = f'background:{bg};' if bg else ""
        cell_html = "".join(
            f'<{tag} style="padding:6px 10px;color:{S["text"]};border:1px solid {S["border"]};{style}">{c}</{tag}>'
            for c in cells
        )
        html_rows.append(f"<tr>{cell_html}</tr>")

    return f'<table style="width:100%;border-collapse:collapse;font-size:0.9em;margin:8px 0;">{"".join(html_rows)}</table>'


def _page_summary(page_md: str) -> str:
    """Short one-line summary of what is on a page."""
    tables = _extract_markdown_tables(page_md)
    headings = re.findall(r'^#{1,4}\s+(.+)$', page_md, re.MULTILINE)
    sections = [h for h in headings if h.strip()]
    parts = []
    if sections:
        parts.append(", ".join(h.strip()[:40] for h in sections[:4]))
    if tables:
        parts.append(f"{len(tables)} table(s)")
    if not parts:
        first_line = page_md.strip().split("\n")[0][:60] if page_md.strip() else "Empty"
        parts.append(first_line)
    return " · ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _render_page_text(corrected_md: str) -> str:
    """Render page markdown to styled HTML with tables rendered inline."""
    display_text = corrected_md
    display_text = re.sub(
        r'^### (.+)$',
        rf'<h4 style="color:{S["purple"]};margin:12px 0 4px;">\1</h4>',
        display_text, flags=re.MULTILINE,
    )
    display_text = re.sub(
        r'^## (.+)$',
        rf'<h3 style="color:{S["blue"]};margin:14px 0 6px;">\1</h3>',
        display_text, flags=re.MULTILINE,
    )
    display_text = re.sub(
        r'^# (.+)$',
        rf'<h2 style="color:{S["text"]};margin:16px 0 8px;">\1</h2>',
        display_text, flags=re.MULTILINE,
    )
    display_text = re.sub(
        r'\*\*(.+?)\*\*',
        rf'<strong style="color:{S["text"]};">\1</strong>',
        display_text,
    )

    lines = display_text.split("\n")
    processed_lines = []
    table_lines = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                table_lines = [stripped]
                in_table = True
            else:
                table_lines.append(stripped)
        else:
            if in_table:
                processed_lines.append(_md_table_to_html("\n".join(table_lines)))
                in_table = False
                table_lines = []
            if stripped.startswith("<h") or stripped.startswith("<strong"):
                processed_lines.append(stripped)
            elif stripped:
                processed_lines.append(
                    f'<p style="margin:2px 0;color:{S["text3"]};line-height:1.5;">{stripped}</p>'
                )

    if in_table:
        processed_lines.append(_md_table_to_html("\n".join(table_lines)))

    return "\n".join(processed_lines)


def _build_page_view_html(
    ocr_result: OCRResult,
    correction_report: CorrectionReport,
    corrected_pages: List[str],
) -> str:
    """Build per-page verification view."""
    html = f"""<div style="font-family:'Inter',sans-serif;color:{S['text']};">
    <h2 style="color:{S['text']};margin:0 0 16px;font-size:1.3em;">
        Page-by-Page Extraction ({ocr_result.page_count} pages)
    </h2>
    <p style="color:{S['text_dim']};font-size:0.9em;margin-bottom:16px;">
        Expand each page to verify extracted text. Tables are rendered inline.
        Corrections on that page are highlighted below its content.
    </p>"""

    for idx, page in enumerate(ocr_result.pages):
        corrected_md = corrected_pages[idx] if idx < len(corrected_pages) else page.markdown
        tables = _extract_markdown_tables(corrected_md)
        page_corrections = [c for c in correction_report.corrections if c.page == idx]
        meaningful_corrections = [c for c in page_corrections
                                  if not c.original.isspace() and c.original.strip()]
        summary = _page_summary(corrected_md)

        badges = ""
        if tables:
            badges += (
                f'<span style="background:{S["blue_bg"]};color:{S["blue"]};'
                f'padding:2px 8px;border-radius:4px;font-size:0.8em;margin-left:8px;">'
                f'{len(tables)} table(s)</span>'
            )
        if meaningful_corrections:
            badges += (
                f'<span style="background:{S["amber_bg"]};color:{S["amber"]};'
                f'padding:2px 8px;border-radius:4px;font-size:0.8em;margin-left:8px;">'
                f'{len(meaningful_corrections)} fix(es)</span>'
            )

        text_html = _render_page_text(corrected_md)

        corrections_html = ""
        if meaningful_corrections:
            corr_items = ""
            for c in meaningful_corrections[:10]:
                corr_items += (
                    f'<div style="display:inline-flex;gap:4px;align-items:center;'
                    f'background:{S["bg"]};padding:3px 8px;border-radius:4px;'
                    f'margin:2px;font-size:0.85em;">'
                    f'<span style="color:{S["red"]};text-decoration:line-through;">{c.original}</span>'
                    f'<span style="color:{S["text_dim"]};">→</span>'
                    f'<span style="color:{S["green"]};font-weight:600;">{c.corrected}</span>'
                    f'</div>'
                )
            corrections_html = (
                f'<div style="margin-top:8px;padding:8px;background:{S["amber_bg"]};'
                f'border:1px solid {S["amber_border"]};border-radius:6px;">'
                f'<div style="font-size:0.85em;color:{S["amber"]};font-weight:600;'
                f'margin-bottom:4px;">OCR Corrections:</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{corr_items}</div>'
                f'</div>'
            )

        is_open = "open" if idx == 0 or tables else ""
        html += f"""
        <details {is_open} style="margin-bottom:12px;border:1px solid {S['border_light']};border-radius:8px;overflow:hidden;">
            <summary style="cursor:pointer;padding:12px 16px;background:{S['bg2']};font-weight:600;color:{S['text']};font-size:1em;display:flex;align-items:center;">
                <span style="background:{S['blue']};color:white;padding:2px 10px;border-radius:4px;font-size:0.85em;margin-right:12px;">Page {idx + 1}</span>
                <span style="flex:1;color:{S['text_dim']};font-weight:400;font-size:0.9em;">{summary}</span>
                {badges}
            </summary>
            <div style="padding:16px;background:{S['bg']};">
                {text_html}
                {corrections_html}
            </div>
        </details>"""

    html += "</div>"
    return html


def _build_invoice_data_html(
    ocr_result: OCRResult,
    corrected_pages: List[str],
    analysis_report: InvoiceAnalysisReport,
) -> str:
    """
    Build the invoice data view:
      1. Summary cards (fields, line items, grand total, discrepancies)
      2. Line items table with math verification
      3. Extracted financial fields by category
      4. All raw extracted tables for reference
    """
    html = f"""<div style="font-family:'Inter',sans-serif;color:{S['text']};">"""

    currency = analysis_report.currency or "$"

    # ── Summary cards ─────────────────────────────────────────────────────────
    field_count = len(analysis_report.fields)
    line_count = len(analysis_report.line_items)
    discrepancy_count = len(analysis_report.discrepancies) + len(analysis_report.flagged_items)

    grand_total_str = "—"
    if analysis_report.grand_total is not None:
        grand_total_str = f"{currency}{analysis_report.grand_total:,.2f}"
    elif analysis_report.subtotal is not None:
        grand_total_str = f"{currency}{analysis_report.subtotal:,.2f}"

    tax_str = "—"
    if analysis_report.tax_amount is not None:
        tax_str = f"{currency}{analysis_report.tax_amount:,.2f}"

    disc_color = S["red"] if discrepancy_count > 0 else S["green"]
    disc_bg = S["red_bg"] if discrepancy_count > 0 else S["green_bg"]
    disc_border = S["red_border"] if discrepancy_count > 0 else S["green_border"]

    html += f"""
    <h2 style="color:{S['text']};margin:0 0 12px;font-size:1.3em;">Invoice Overview</h2>
    <div style="display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;">
      <div style="flex:1;min-width:100px;background:{S['blue_bg']};border:1px solid {S['blue_border']};border-radius:8px;padding:14px;text-align:center;">
        <div style="font-size:1.8em;font-weight:700;color:{S['blue']};">{field_count}</div>
        <div style="color:{S['blue_text']};font-size:0.9em;">Fields Extracted</div>
      </div>
      <div style="flex:1;min-width:100px;background:{S['bg2']};border:1px solid {S['border']};border-radius:8px;padding:14px;text-align:center;">
        <div style="font-size:1.8em;font-weight:700;color:{S['purple']};">{line_count}</div>
        <div style="color:{S['text_dim']};font-size:0.9em;">Line Items</div>
      </div>
      <div style="flex:1;min-width:100px;background:{S['green_bg']};border:1px solid {S['green_border']};border-radius:8px;padding:14px;text-align:center;">
        <div style="font-size:1.4em;font-weight:700;color:{S['green']};">{grand_total_str}</div>
        <div style="color:{S['green_text']};font-size:0.9em;">Grand Total</div>
      </div>
      <div style="flex:1;min-width:100px;background:{S['amber_bg']};border:1px solid {S['amber_border']};border-radius:8px;padding:14px;text-align:center;">
        <div style="font-size:1.4em;font-weight:700;color:{S['amber']};">{tax_str}</div>
        <div style="color:{S['amber_text']};font-size:0.9em;">Tax Amount</div>
      </div>
      <div style="flex:1;min-width:100px;background:{disc_bg};border:1px solid {disc_border};border-radius:8px;padding:14px;text-align:center;">
        <div style="font-size:1.8em;font-weight:700;color:{disc_color};">{discrepancy_count}</div>
        <div style="color:{disc_color};font-size:0.9em;">Discrepancies</div>
      </div>
    </div>"""

    # ── Discrepancy alerts ────────────────────────────────────────────────────
    all_discrepancies = list(analysis_report.discrepancies)
    for li in analysis_report.flagged_items:
        all_discrepancies.append(f"Line item '{li.description}': {li.note}")

    if all_discrepancies:
        alert_items = ""
        for d in all_discrepancies:
            alert_items += (
                f'<div style="padding:8px 12px;background:{S["red_bg"]};border-radius:4px;'
                f'margin-bottom:4px;color:{S["red_text"]};">⚠ {d}</div>'
            )
        html += f"""
        <div style="margin-bottom:16px;padding:16px;background:{S['red_bg']};border:2px solid {S['red_border']};border-radius:8px;">
          <h3 style="color:{S['red']};margin:0 0 10px;font-size:1.1em;">⚠ Discrepancies Detected</h3>
          {alert_items}
        </div>"""

    # ── Line Items table ──────────────────────────────────────────────────────
    if analysis_report.line_items:
        rows_html = ""
        for li in analysis_report.line_items:
            row_bg = {
                FieldStatus.FLAGGED: S["amber_bg"],
                FieldStatus.CRITICAL: S["red_bg"],
            }.get(li.status, "")
            style = f'background:{row_bg};' if row_bg else ""
            status_html = _status_badge(li.status)
            qty_str = f"{li.quantity:g}" if li.quantity is not None else "—"
            price_str = f"{currency}{li.unit_price:,.2f}" if li.unit_price is not None else "—"
            amount_str = f"{currency}{li.amount:,.2f}" if li.amount is not None else "—"
            note_html = (
                f'<span style="font-size:0.8em;color:{S["text_dim"]};">{li.note}</span>'
                if li.note else ""
            )
            rows_html += (
                f'<tr style="{style}">'
                f'<td style="padding:8px 10px;color:{S["text"]};font-weight:500;">{li.description}</td>'
                f'<td style="padding:8px 10px;color:{S["text3"]};text-align:center;">{qty_str}</td>'
                f'<td style="padding:8px 10px;color:{S["text3"]};text-align:right;">{price_str}</td>'
                f'<td style="padding:8px 10px;font-weight:700;color:white;text-align:right;">{amount_str}</td>'
                f'<td style="padding:8px 10px;">{status_html}{note_html}</td>'
                f'</tr>'
            )

        html += f"""
        <div style="margin-bottom:20px;">
          <h3 style="margin:0 0 8px;color:{S['text']};border-bottom:2px solid {S['border']};padding-bottom:4px;">
            Line Items ({len(analysis_report.line_items)})
          </h3>
          <div style="overflow-x:auto;">
          <table style="width:100%;border-collapse:collapse;font-size:0.92em;">
            <thead><tr style="background:{S['bg3']};">
              <th style="padding:8px 10px;text-align:left;color:{S['text']};">Description</th>
              <th style="padding:8px 10px;text-align:center;color:{S['text']};">Qty</th>
              <th style="padding:8px 10px;text-align:right;color:{S['text']};">Unit Price</th>
              <th style="padding:8px 10px;text-align:right;color:{S['text']};">Amount</th>
              <th style="padding:8px 10px;text-align:left;color:{S['text']};">Status</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
          </div>
        </div>"""

    # ── Extracted fields by category ──────────────────────────────────────────
    categories = analysis_report.values_by_category()
    for cat_name, cat_fields in categories.items():
        if not cat_fields:
            continue
        rows_html = ""
        for f in cat_fields:
            row_bg = {
                FieldStatus.CRITICAL: S["red_bg"],
                FieldStatus.FLAGGED: S["amber_bg"],
            }.get(f.status, "")
            style = f'background:{row_bg};' if row_bg else ""
            note_html = (
                f'<br><span style="font-size:0.8em;color:{S["text_dim"]};">{f.note}</span>'
                if f.note else ""
            )
            rows_html += (
                f'<tr style="{style}">'
                f'<td style="padding:6px 10px;font-weight:500;color:{S["text"]};">{f.display_name}</td>'
                f'<td style="padding:6px 10px;font-weight:600;color:white;">{f.value_str}{note_html}</td>'
                f'<td style="padding:6px 10px;">{_status_badge(f.status)}</td>'
                f'<td style="padding:6px 10px;color:{S["text_dim"]};font-size:0.8em;">p.{f.page + 1}</td>'
                f'</tr>'
            )

        html += f"""
        <div style="margin-bottom:16px;">
          <h3 style="margin:0 0 8px;color:{S['text']};border-bottom:2px solid {S['border']};padding-bottom:4px;">{cat_name}</h3>
          <table style="width:100%;border-collapse:collapse;font-size:0.92em;">
            <thead><tr style="background:{S['bg3']};">
              <th style="padding:8px 10px;text-align:left;color:{S['text']};">Field</th>
              <th style="padding:8px 10px;text-align:left;color:{S['text']};">Value</th>
              <th style="padding:8px 10px;text-align:left;color:{S['text']};">Status</th>
              <th style="padding:8px 10px;text-align:left;color:{S['text']};">Page</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>"""

    if not analysis_report.fields and not analysis_report.line_items:
        html += f"""
        <div style="padding:20px;background:{S['bg2']};border:1px solid {S['border_light']};border-radius:8px;text-align:center;color:{S['text_dim']};">
          No structured invoice data detected. Check the raw tables below for extracted content.
        </div>"""

    # ── All Extracted Tables (raw) ────────────────────────────────────────────
    all_tables = []
    for idx, cpage in enumerate(corrected_pages):
        tables = _extract_markdown_tables(cpage)
        for ti, t in enumerate(tables):
            all_tables.append((idx, ti, t))

    html += f"""
    <h2 style="color:{S['text']};margin:24px 0 12px;font-size:1.3em;">All Extracted Tables</h2>
    <p style="color:{S['text_dim']};font-size:0.9em;margin-bottom:12px;">
        Raw tables found across all pages. Verify against the original document.
    </p>"""

    if all_tables:
        for page_idx, table_idx, table_md in all_tables:
            table_html = _md_table_to_html(table_md)
            html += f"""
            <details style="margin-bottom:12px;border:1px solid {S['border_light']};border-radius:8px;overflow:hidden;">
                <summary style="cursor:pointer;padding:8px 12px;background:{S['bg2']};display:flex;align-items:center;gap:8px;">
                    <span style="background:{S['blue']};color:white;padding:2px 8px;border-radius:4px;font-size:0.8em;">Page {page_idx + 1}</span>
                    <span style="color:{S['text_dim']};font-size:0.9em;">Table {table_idx + 1}</span>
                </summary>
                <div style="padding:8px 12px;background:{S['bg']};">{table_html}</div>
            </details>"""
    else:
        html += f"""
        <div style="padding:16px;background:{S['bg2']};border:1px solid {S['border_light']};border-radius:8px;text-align:center;color:{S['text_dim']};">
          No tables detected in the document.
        </div>"""

    html += "</div>"
    return html


def _build_quality_html(
    ocr_result: OCRResult,
    correction_report: CorrectionReport,
    completeness_report: CompletenessReport,
) -> str:
    """Build OCR quality metrics view."""
    html = f"""<div style="font-family:'Inter',sans-serif;color:{S['text']};">
    <h2 style="color:{S['text']};margin:0 0 16px;font-size:1.3em;">OCR Quality Report</h2>"""

    html += f"""
    <div style="margin-bottom:16px;padding:16px;background:{S['bg2']};border:1px solid {S['border_light']};border-radius:8px;">
        <h3 style="color:{S['text']};margin:0 0 8px;font-size:1.1em;">Processing Info</h3>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;">
            <div><span style="color:{S['text_dim']};">Model:</span> <strong style="color:{S['text']};">{ocr_result.model}</strong></div>
            <div><span style="color:{S['text_dim']};">Pages:</span> <strong style="color:{S['text']};">{ocr_result.page_count}</strong></div>
            <div><span style="color:{S['text_dim']};">Time:</span> <strong style="color:{S['text']};">{ocr_result.processing_time_s}s</strong></div>
            <div><span style="color:{S['text_dim']};">Total chars:</span> <strong style="color:{S['text']};">{sum(len(p.markdown) for p in ocr_result.pages):,}</strong></div>
        </div>
    </div>"""

    meaningful = [c for c in correction_report.corrections
                  if not c.original.isspace() and c.original.strip()]

    if meaningful:
        corr_rows = ""
        for c in meaningful[:50]:
            corr_rows += (
                f'<tr>'
                f'<td style="padding:4px 8px;color:{S["text3"]};">Page {c.page + 1}</td>'
                f'<td style="padding:4px 8px;color:{S["red"]};text-decoration:line-through;">{c.original}</td>'
                f'<td style="padding:4px 8px;color:{S["green"]};font-weight:600;">{c.corrected}</td>'
                f'<td style="padding:4px 8px;font-size:0.8em;color:{S["text_dim"]};">{c.rule[:30]}</td>'
                f'</tr>'
            )
        html += f"""
        <div style="margin-bottom:16px;padding:16px;background:{S['amber_bg']};border:1px solid {S['amber_border']};border-radius:8px;">
            <h3 style="color:{S['amber']};margin:0 0 8px;font-size:1.1em;">OCR Corrections ({len(meaningful)})</h3>
            <table style="width:100%;border-collapse:collapse;font-size:0.9em;">
                <thead><tr style="background:{S['bg3']};">
                    <th style="padding:6px 8px;text-align:left;color:{S['text']};">Page</th>
                    <th style="padding:6px 8px;text-align:left;color:{S['text']};">Original</th>
                    <th style="padding:6px 8px;text-align:left;color:{S['text']};">Corrected</th>
                    <th style="padding:6px 8px;text-align:left;color:{S['text']};">Rule</th>
                </tr></thead>
                <tbody>{corr_rows}</tbody>
            </table>
        </div>"""
    else:
        html += f"""
        <div style="margin-bottom:16px;padding:12px 16px;background:{S['green_bg']};border:1px solid {S['green_border']};border-radius:8px;color:{S['green_text']};">
            No OCR character corrections needed — text looks clean.
        </div>"""

    if completeness_report.has_issues:
        issue_items = ""
        for iss in completeness_report.issues:
            icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(iss.severity, "•")
            bg = {"error": S["red_bg"], "warning": S["amber_bg"]}.get(iss.severity, S["bg2"])
            issue_items += (
                f'<div style="padding:8px 12px;background:{bg};border-radius:4px;'
                f'margin-bottom:4px;color:{S["amber_text"]};">'
                f'{icon} <strong style="color:white;">[{iss.issue_type}]</strong> '
                f'Page {iss.page + 1}: {iss.description}</div>'
            )
        html += f"""
        <div style="margin-bottom:16px;padding:16px;background:{S['bg2']};border:1px solid {S['red_border']};border-radius:8px;">
            <h3 style="color:{S['red']};margin:0 0 8px;font-size:1.1em;">Completeness Issues ({len(completeness_report.issues)})</h3>
            {issue_items}
            <p style="font-size:0.85em;color:{S['text_dim']};margin-top:8px;">
                Tables detected: {completeness_report.table_count} | Rows: {completeness_report.total_rows_detected}
            </p>
        </div>"""
    else:
        html += f"""
        <div style="margin-bottom:16px;padding:12px 16px;background:{S['green_bg']};border:1px solid {S['green_border']};border-radius:8px;color:{S['green_text']};">
            Completeness check passed — {completeness_report.table_count} table(s),
            {completeness_report.total_rows_detected} row(s), no skips.
        </div>"""

    html += "</div>"
    return html


def _build_financial_summary_html(analysis_report: InvoiceAnalysisReport) -> str:
    """Build a financial summary view of all extracted invoice fields."""
    if not analysis_report.fields and not analysis_report.line_items:
        return (
            f"<div style='padding:30px;text-align:center;color:{S['text_dim']};"
            f"font-family:Inter,sans-serif;'>No invoice data found in this document.</div>"
        )

    currency = analysis_report.currency or "$"

    html = f"""<div style="font-family:'Inter',sans-serif;color:{S['text']};">
    <h2 style="color:{S['text']};margin:0 0 12px;font-size:1.3em;">Financial Summary</h2>
    <p style="color:{S['text_dim']};font-size:0.9em;margin-bottom:16px;">
        All extracted invoice fields and financial data from the scanned document.
    </p>"""

    # Financial totals panel
    totals_html = ""
    total_fields = [
        ("Subtotal", analysis_report.subtotal),
        ("Tax", analysis_report.tax_amount),
        ("Discount", analysis_report.discount),
        ("Grand Total", analysis_report.grand_total),
    ]
    has_totals = any(v is not None for _, v in total_fields)
    if has_totals:
        for label, val in total_fields:
            if val is not None:
                color = S["green"] if label == "Grand Total" else S["text3"]
                weight = "700" if label == "Grand Total" else "400"
                border = f'border-top:2px solid {S["border"]};' if label == "Grand Total" else ""
                totals_html += (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:8px 0;{border}">'
                    f'<span style="color:{S["text_dim"]};">{label}</span>'
                    f'<span style="color:{color};font-weight:{weight};font-size:1.05em;">'
                    f'{currency}{val:,.2f}</span></div>'
                )
        html += f"""
        <div style="margin-bottom:20px;padding:16px;background:{S['bg2']};border:1px solid {S['border']};border-radius:8px;">
          <h3 style="margin:0 0 12px;color:{S['text']};font-size:1.05em;">Totals Breakdown</h3>
          {totals_html}
        </div>"""

    # Line items summary
    if analysis_report.line_items:
        verified = sum(1 for li in analysis_report.line_items if li.status == FieldStatus.MATCHED)
        flagged = sum(1 for li in analysis_report.line_items if li.status == FieldStatus.FLAGGED)
        line_sum = sum(li.amount for li in analysis_report.line_items if li.amount is not None)

        html += f"""
        <div style="margin-bottom:20px;padding:16px;background:{S['bg2']};border:1px solid {S['border']};border-radius:8px;">
          <h3 style="margin:0 0 12px;color:{S['text']};font-size:1.05em;">Line Items Summary</h3>
          <div style="display:flex;gap:16px;flex-wrap:wrap;">
            <div><span style="color:{S['text_dim']};">Total items:</span> <strong style="color:{S['text']};">{len(analysis_report.line_items)}</strong></div>
            <div><span style="color:{S['text_dim']};">Verified:</span> <strong style="color:{S['green']};">{verified}</strong></div>
            <div><span style="color:{S['text_dim']};">Flagged:</span> <strong style="color:{S['amber']};">{flagged}</strong></div>
            <div><span style="color:{S['text_dim']};">Sum:</span> <strong style="color:{S['text']};">{currency}{line_sum:,.2f}</strong></div>
          </div>
        </div>"""

    # Fields by category
    categories = analysis_report.values_by_category()
    for cat_name, cat_fields in categories.items():
        rows = ""
        for f in cat_fields:
            status_icon = {
                FieldStatus.MATCHED: "✅",
                FieldStatus.FLAGGED: "⚠️",
                FieldStatus.CRITICAL: "🚨",
                FieldStatus.EXTRACTED: "📋",
                FieldStatus.MISSING: "❌",
            }.get(f.status, "—")
            rows += (
                f'<tr>'
                f'<td style="padding:6px 10px;color:{S["text"]};font-weight:500;">{f.display_name}</td>'
                f'<td style="padding:6px 10px;color:{S["text3"]};">{f.value_str}</td>'
                f'<td style="padding:6px 10px;color:{S["text"]};">{status_icon} {f.status.value}</td>'
                f'</tr>'
            )
        html += f"""
        <div style="margin-bottom:16px;">
          <h3 style="margin:0 0 8px;color:{S['text']};border-bottom:2px solid {S['border']};padding-bottom:4px;">{cat_name}</h3>
          <table style="width:100%;border-collapse:collapse;font-size:0.92em;">
            <thead><tr style="background:{S['bg3']};">
              <th style="padding:8px 10px;text-align:left;color:{S['text']};">Field</th>
              <th style="padding:8px 10px;text-align:left;color:{S['text']};">Value</th>
              <th style="padding:8px 10px;text-align:left;color:{S['text']};">Status</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    html += "</div>"
    return html


def _build_sidebyside_html(pdf_path: str, corrected_text: str) -> str:
    """Build side-by-side PDF (as images) + text view for human verification."""
    pdf_images = _pdf_to_images_html(pdf_path)
    escaped = corrected_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
    <div style="display:flex;gap:12px;height:85vh;font-family:'Inter',sans-serif;">
        <div style="flex:1;min-width:0;border:1px solid {S['border_light']};border-radius:8px;overflow:hidden;display:flex;flex-direction:column;">
            <div style="padding:8px 12px;background:{S['bg2']};color:{S['text']};font-weight:600;font-size:0.9em;">
                Original Document
            </div>
            <div style="flex:1;overflow-y:auto;padding:4px;background:{S['bg']};">{pdf_images}</div>
        </div>
        <div style="flex:1;min-width:0;border:1px solid {S['border_light']};border-radius:8px;overflow:hidden;display:flex;flex-direction:column;">
            <div style="padding:8px 12px;background:{S['bg2']};color:{S['text']};font-weight:600;font-size:0.9em;">
                Extracted Text (compare with original)
            </div>
            <pre style="flex:1;margin:0;padding:12px;background:{S['bg']};color:{S['text3']};overflow:auto;font-size:0.85em;font-family:'JetBrains Mono','Fira Code',monospace;line-height:1.6;white-space:pre-wrap;word-wrap:break-word;">{escaped}</pre>
        </div>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _chat_status_html(doc_loaded: bool, num_fields: int = 0) -> str:
    if doc_loaded:
        return (
            f'<div style="padding:14px 20px;background:{S["bg2"]};'
            f'border:1px solid {S["blue_border"]};border-radius:12px;margin-bottom:8px;">'
            f'<div style="display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:1.6em;">📄</span>'
            f'<div style="flex:1;">'
            f'<div style="font-weight:700;color:{S["text"]};font-size:1.15em;">Invoice Assistant</div>'
            f'<div style="color:{S["text_dim"]};font-size:0.9em;margin-top:2px;">'
            f'Ask questions about line items, totals, discrepancies, or any invoice data.</div></div>'
            f'<div style="padding:6px 14px;border-radius:8px;font-size:0.85em;font-weight:600;'
            f'background:{S["green_bg"]};color:{S["green"]};border:1px solid {S["green_border"]};">'
            f'✅ Invoice loaded — {num_fields} fields extracted</div></div></div>'
        )
    return (
        f'<div style="padding:14px 20px;background:{S["bg2"]};'
        f'border:1px solid {S["blue_border"]};border-radius:12px;margin-bottom:8px;">'
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<span style="font-size:1.6em;">📄</span>'
        f'<div style="flex:1;">'
        f'<div style="font-weight:700;color:{S["text"]};font-size:1.15em;">Invoice Assistant</div>'
        f'<div style="color:{S["text_dim"]};font-size:0.9em;margin-top:2px;">'
        f'Ask questions about line items, totals, discrepancies, or any invoice data.</div></div>'
        f'<div style="padding:6px 14px;border-radius:8px;font-size:0.85em;font-weight:600;'
        f'background:{S["amber_bg"]};color:{S["amber"]};border:1px solid {S["amber_border"]};">'
        f'⚠ No invoice analyzed yet</div></div></div>'
    )


def process_document(pdf_file) -> Tuple[str, str, str, str, str, str, str]:
    """
    Main pipeline: PDF → OCR → Correction → Completeness → Analysis → Display.

    Returns:
      (page_view_html, invoice_data_html, sidebyside_html, quality_html, full_text, financial_summary_html, chat_status_html)
    """
    global _last_ocr, _last_correction, _last_completeness, _last_analysis, _last_pdf_path

    placeholder = (
        f"<p style='color:{S['text_dim']};text-align:center;padding:40px;'>Processing…</p>"
    )

    if pdf_file is None:
        empty = (
            f"<p style='color:{S['text_dim']};text-align:center;padding:40px;'>"
            f"Upload a PDF to begin.</p>"
        )
        return (empty, empty, empty, empty, "", empty, _chat_status_html(False))

    pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
    if not os.path.isfile(pdf_path):
        err = f"<p style='color:{S['red']};'>File not found: {pdf_path}</p>"
        return (err, err, err, err, "", err, _chat_status_html(False))

    dest = os.path.join(config.UPLOAD_DIR, os.path.basename(pdf_path))
    shutil.copy2(pdf_path, dest)
    _last_pdf_path = dest

    try:
        # ── Step 1: OCR ───────────────────────────────────────────────────────
        logger.info("Starting OCR for %s", pdf_path)
        ocr_result = process_pdf(pdf_path=pdf_path, include_images=False)
        _last_ocr = ocr_result
        pages_md = [p.markdown for p in ocr_result.pages]

        # ── Step 2: OCR error correction ──────────────────────────────────────
        logger.info("Running OCR corrections …")
        correction_report = correct_ocr_pages(pages_md)
        _last_correction = correction_report
        corrected_pages = correction_report.corrected_text.split("\n\n---\n\n")

        # ── Step 3: Completeness check ────────────────────────────────────────
        logger.info("Checking completeness …")
        completeness_report = check_completeness(corrected_pages)
        _last_completeness = completeness_report

        # ── Step 4: Invoice data extraction & analysis ────────────────────────
        logger.info("Extracting and analysing invoice data …")
        analysis_report = analyse(corrected_pages)
        _last_analysis = analysis_report

        # ── Build outputs ─────────────────────────────────────────────────────
        page_view_html = _build_page_view_html(ocr_result, correction_report, corrected_pages)
        invoice_data_html = _build_invoice_data_html(ocr_result, corrected_pages, analysis_report)
        sidebyside_html = _build_sidebyside_html(dest, correction_report.corrected_text)
        quality_html = _build_quality_html(ocr_result, correction_report, completeness_report)
        full_text = correction_report.corrected_text
        financial_summary_html = _build_financial_summary_html(analysis_report)
        status_html = _chat_status_html(True, len(analysis_report.fields))

        return (page_view_html, invoice_data_html, sidebyside_html, quality_html, full_text, financial_summary_html, status_html)

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Processing failed: %s\n%s", exc, tb)

        error_str = str(exc)
        if "AuthenticationTypeDisabled" in error_str or "az login" in error_str:
            error_html = (
                f"<div style='padding:20px;background:{S['red_bg']};border:2px solid {S['red_border']};"
                f"border-radius:12px;color:{S['text']};'>"
                f"<h3 style='color:{S['red']};margin-top:0;'>Azure Authentication Required</h3>"
                f"<p>Key-based auth is <strong>disabled</strong>. "
                f"Run <code>az login --use-device-code</code> then retry.</p></div>"
            )
        else:
            error_html = (
                f"<div style='color:{S['red']};padding:20px;'><h3>Error</h3>"
                f"<pre style='color:{S['text3']};white-space:pre-wrap;'>{tb}</pre></div>"
            )

        pdf_html = _build_sidebyside_html(dest, "") if os.path.isfile(dest) else ""
        return (error_html, error_html, pdf_html, error_html, "", error_html, _chat_status_html(False))


def load_sample(sample_name: str) -> Optional[str]:
    sample_path = os.path.join(config.DATA_DIR, sample_name)
    if os.path.isfile(sample_path):
        return sample_path
    return None


def get_sample_list() -> List[str]:
    if not os.path.isdir(config.DATA_DIR):
        return []
    return sorted([f for f in os.listdir(config.DATA_DIR) if f.lower().endswith(".pdf")])


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
* { font-family: Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, sans-serif !important; }

.gradio-container { background: #0A0A0A !important; color: #F5F5F5 !important; }
.main-header {
    background: #FF8205;
    color: white; padding: 24px 28px; border-radius: 12px; margin-bottom: 16px;
}
.main-header h1 { margin: 0; font-size: 1.5em; color: #000000; font-weight: 700; letter-spacing: -0.01em; }
.main-header p { margin: 4px 0 0; font-size: 0.9em; color: #1E1E1E; opacity: 0.85; }

.block { background: #1E1E1E !important; border-color: #2A2A2A !important; }
label, .label-wrap, .tab-nav button { color: #E8E8E8 !important; }
.tab-nav button.selected { color: #FF8205 !important; border-color: #FF8205 !important; }
textarea, input[type="text"], input[type="search"] { background: #0A0A0A !important; color: #F5F5F5 !important; border-color: #3A3A3A !important; }
.markdown-text, .prose { color: #E8E8E8 !important; }
.prose h1, .prose h2, .prose h3, .prose h4 { color: #F5F5F5 !important; }
.prose table th { background: #2A2A2A !important; color: #F5F5F5 !important; }
.prose table td { color: #E8E8E8 !important; border-color: #3A3A3A !important; }
.prose strong { color: #F5F5F5 !important; }

/* File upload component */
#pdf_upload label, #pdf_upload .label-wrap { color: #000000 !important; }
.upload-btn, button[aria-label="Upload file"], .file-preview-title { color: #F5F5F5 !important; }
.file-preview { background: #1E1E1E !important; border-color: #3A3A3A !important; color: #F5F5F5 !important; }
.upload-btn-wrap button, .upload-container button {
    background: #FF8205 !important; color: #000000 !important;
    border: none !important; border-radius: 6px !important;
    font-weight: 600 !important;
}
.upload-btn-wrap button:hover, .upload-container button:hover { background: #FA500F !important; }
.drop-input { background: #1E1E1E !important; border: 2px dashed #3A3A3A !important; color: #999999 !important; }
.drop-input:hover { border-color: #FF8205 !important; }
.file-name, .file-size { color: #F5F5F5 !important; }
.clear-button { color: #999999 !important; }
.clear-button:hover { color: #FF5555 !important; }
/* Gradio primary button */
button.primary, .btn-primary { background: #FF8205 !important; color: #000000 !important; border: none !important; }
button.primary:hover, .btn-primary:hover { background: #FA500F !important; }

/* Chat UI — force the entire chatbot container dark first */
#chat_window,
#chat_window > *,
#chat_window .wrap,
#chat_window .scroll-hide,
#chat_window .message-wrap,
#chat_window [class*="wrap"],
#chat_window [class*="scroll"] { background: #0A0A0A !important; background-color: #0A0A0A !important; }

.chatbot, .chatbot > *, .chatbot .wrap { font-size: 1.05em !important; background: #0A0A0A !important; background-color: #0A0A0A !important; }
.chatbot .message-row .message { font-size: 1.05em !important; line-height: 1.6 !important; }

/* User bubble */
#chat_window .bubble-wrap.user,
#chat_window .message.user,
.chatbot .message-row .user { background: #FF8205 !important; color: #000000 !important; border-radius: 14px !important; }
#chat_window .bubble-wrap.user *,
#chat_window .message.user *,
.chatbot .message-row .user * { color: #000000 !important; background: transparent !important; }

/* Bot/assistant bubble */
#chat_window .bubble-wrap.bot,
#chat_window .bubble-wrap.assistant,
#chat_window .message.bot,
#chat_window .message.assistant,
.chatbot .message-row .bot,
.chatbot .message-row .assistant { background: #1E1E1E !important; background-color: #1E1E1E !important; border: 1px solid #3A3A3A !important; border-radius: 14px !important; color: #E8E8E8 !important; }

/* All elements inside bot bubble */
#chat_window .bubble-wrap.bot *,
#chat_window .bubble-wrap.assistant *,
#chat_window .message.bot *,
#chat_window .message.assistant *,
.chatbot .message-row .bot *,
.chatbot .message-row .assistant * { background: transparent !important; background-color: transparent !important; color: #E8E8E8 !important; }

.chatbot .bubble-wrap { padding: 8px 16px !important; }
.chatbot .message-row .message-content { padding: 14px 18px !important; }
"""

EMPTY_MSG = (
    "<p style='color:#999999;text-align:center;padding:40px;"
    "font-family:Arial,sans-serif;'>Upload an invoice PDF to begin analysis.</p>"
)


def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks application."""

    with gr.Blocks(title="Mistral Invoice OCR Analyzer") as app:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="main-header">
            <h1>&#x2588; Mistral Invoice OCR Analyzer</h1>
            <p>Powered by Mistral Document AI &mdash; extract, verify, and analyse financial invoice documents</p>
        </div>
        """)

        with gr.Row():
            # ── Left sidebar: Upload & Controls ───────────────────────────────
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Upload Invoice")
                pdf_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath",
                    elem_id="pdf_upload",
                )

                samples = get_sample_list()
                sample_dropdown = gr.Dropdown(
                    choices=samples if samples else [],
                    label="Sample Invoices",
                    interactive=True,
                    visible=bool(samples),
                )
                if samples:
                    gr.Markdown("**Or pick a sample above**", visible=True)

                process_btn = gr.Button("Analyse Invoice", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("""
                **Verification workflow:**
                1. **Pages** — check extracted text per page
                2. **Side-by-Side** — compare PDF vs extracted text
                3. **Invoice Data** — review line items & totals
                4. **Quality** — see OCR corrections & issues
                5. **Financial Summary** — full field breakdown
                """)

            # ── Main content area ─────────────────────────────────────────────
            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("Side-by-Side"):
                        sidebyside_view = gr.HTML(value=EMPTY_MSG)

                    with gr.Tab("Invoice Data"):
                        invoice_data_view = gr.HTML(value=EMPTY_MSG)

                    with gr.Tab("OCR Quality"):
                        quality_view = gr.HTML(value=EMPTY_MSG)

                    with gr.Tab("Full Text"):
                        full_text = gr.Textbox(
                            label="Complete Extracted Text (corrected)",
                            lines=35,
                            interactive=False,
                        )

                    with gr.Tab("Financial Summary"):
                        financial_summary_view = gr.HTML(value=EMPTY_MSG)

                    with gr.Tab("💬 Chat"):
                        chat_status = gr.HTML(value=f"""
                        <div style="padding:14px 20px;background:{S['bg2']};
                                    border:1px solid {S['blue_border']};border-radius:12px;margin-bottom:8px;">
                            <div style="display:flex;align-items:center;gap:12px;">
                                <span style="font-size:1.6em;">📄</span>
                                <div style="flex:1;">
                                    <div style="font-weight:700;color:{S['text']};font-size:1.15em;">Invoice Assistant</div>
                                    <div style="color:{S['text_dim']};font-size:0.9em;margin-top:2px;">
                                        Ask questions about line items, totals, discrepancies, or any invoice data.
                                    </div>
                                </div>
                                <div style="padding:6px 14px;border-radius:8px;font-size:0.85em;font-weight:600;
                                            background:{S['amber_bg']};color:{S['amber']};border:1px solid {S['amber_border']};">
                                    ⚠ No invoice analyzed yet
                                </div>
                            </div>
                        </div>
                        """)
                        chatbot = gr.Chatbot(
                            value=[],
                            height=620,
                            show_label=False,
                            layout="bubble",
                            placeholder="Upload &amp; analyze an invoice first, then ask me anything about the data…",
                            elem_id="chat_window",
                        )
                        with gr.Row(equal_height=True):
                            chat_input = gr.Textbox(
                                placeholder="e.g. Do the line items add up to the subtotal?",
                                show_label=False,
                                scale=8,
                                container=False,
                                lines=2,
                                max_lines=4,
                            )
                            chat_send = gr.Button("Send ➤", variant="primary", scale=1, min_width=100)
                        with gr.Row():
                            chat_clear = gr.Button("🗑️ Clear Chat", size="sm", scale=1)
                            gr.HTML(f'<div style="flex:1;text-align:right;color:{S["text_dim"]};font-size:0.8em;padding:6px 0;">Uses existing OCR analysis — no re-scanning</div>')

                    with gr.Tab("Pages"):
                        page_view = gr.HTML(value=EMPTY_MSG)

        # ── Event handlers ────────────────────────────────────────────────────
        outputs = [page_view, invoice_data_view, sidebyside_view, quality_view, full_text, financial_summary_view, chat_status]

        def smart_process(pdf_file, sample_name):
            """Process uploaded file or selected sample."""
            if pdf_file:
                return process_document(pdf_file)
            if sample_name:
                path = load_sample(sample_name)
                if path:
                    return process_document(path)
                err = f"<p style='color:{S['red']}'>Sample not found: {sample_name}</p>"
                return (err,) * 5 + ("",) + (_chat_status_html(False),)
            empty = (
                f"<p style='color:{S['text_dim']};text-align:center;padding:40px;'>"
                f"Upload an invoice PDF or pick a sample first.</p>"
            )
            return (empty,) * 5 + ("",) + (_chat_status_html(False),)

        process_btn.click(
            fn=smart_process,
            inputs=[pdf_input, sample_dropdown],
            outputs=outputs,
        )

        def load_and_process(sample_name):
            logger.info("=== load_and_process called with: %r ===", sample_name)
            if not sample_name:
                return (EMPTY_MSG,) * 5 + ("",) + (_chat_status_html(False),)
            path = load_sample(sample_name)
            logger.info("load_sample returned: %r", path)
            if not path:
                err = f"<p style='color:{S['red']}'>Sample not found: {sample_name}</p>"
                return (err,) * 5 + ("",) + (_chat_status_html(False),)
            return process_document(path)

        sample_dropdown.change(
            fn=load_and_process,
            inputs=[sample_dropdown],
            outputs=outputs,
        )

        # ── Chat event handlers ───────────────────────────────────────────────
        chat_send.click(
            fn=chat_respond,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot],
        )
        chat_input.submit(
            fn=chat_respond,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot],
        )
        chat_clear.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, chat_input],
        )

    return app


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
    )
