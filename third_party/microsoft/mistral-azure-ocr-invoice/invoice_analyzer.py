"""
Invoice financial data analyser.

Provides:
  - Structured extraction of invoice fields and line items from OCR-scanned tables
  - Mathematical verification of line item amounts and grand totals
  - Discrepancy detection and flagging
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  FIELD STATUS
# ═══════════════════════════════════════════════════════════════════════════════

class FieldStatus(str, Enum):
    MATCHED = "matched"        # Verified — math checks out
    FLAGGED = "flagged"        # Discrepancy detected
    CRITICAL = "critical"      # Major discrepancy (e.g., grand total mismatch)
    EXTRACTED = "extracted"    # Successfully extracted, no verification possible
    MISSING = "missing"        # Expected field not found
    UNKNOWN = "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedField:
    """A single field extracted from an invoice document."""
    field_name: str
    display_name: str
    value_str: str                      # Raw string value
    numeric_value: Optional[float]      # Parsed number if applicable
    unit: str = ""                      # Currency symbol or unit
    status: FieldStatus = FieldStatus.UNKNOWN
    category: str = "Invoice"
    page: int = 0
    note: str = ""
    source_text: str = ""

    # Compatibility shims so app.py can use .value, .reference_range, .test_name
    @property
    def value(self) -> str:
        return self.value_str

    @property
    def reference_range(self) -> str:
        return self.note

    @property
    def test_name(self) -> str:
        return self.field_name


@dataclass
class LineItem:
    """A single line item from an invoice."""
    description: str
    quantity: Optional[float]
    unit_price: Optional[float]
    amount: Optional[float]
    page: int = 0
    status: FieldStatus = FieldStatus.UNKNOWN
    note: str = ""


@dataclass
class InvoiceAnalysisReport:
    """Summary of invoice data extraction and verification."""
    fields: List[ExtractedField] = field(default_factory=list)
    line_items: List[LineItem] = field(default_factory=list)
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    tax_rate: Optional[float] = None
    discount: Optional[float] = None
    grand_total: Optional[float] = None
    currency: str = "$"
    discrepancies: List[str] = field(default_factory=list)

    # Compatibility properties for app.py
    @property
    def values(self) -> List[ExtractedField]:
        return self.fields

    @property
    def normal_values(self) -> List[ExtractedField]:
        return [f for f in self.fields if f.status in (FieldStatus.MATCHED, FieldStatus.EXTRACTED)]

    @property
    def abnormal_values(self) -> List[ExtractedField]:
        return [f for f in self.fields if f.status == FieldStatus.FLAGGED]

    @property
    def critical_values(self) -> List[ExtractedField]:
        return [f for f in self.fields if f.status == FieldStatus.CRITICAL]

    @property
    def flagged_items(self) -> List[LineItem]:
        return [li for li in self.line_items if li.status == FieldStatus.FLAGGED]

    @property
    def has_discrepancies(self) -> bool:
        return bool(self.discrepancies) or bool(self.flagged_items)

    def values_by_category(self) -> Dict[str, List[ExtractedField]]:
        cats: Dict[str, List[ExtractedField]] = {}
        for f in self.fields:
            cats.setdefault(f.category, []).append(f)
        return cats

    def summary_text(self) -> str:
        lines = [f"Extracted {len(self.fields)} invoice fields:"]
        lines.append(f"  ✅ Verified: {len(self.normal_values)}")
        lines.append(f"  ⚠️  Flagged: {len(self.abnormal_values)}")
        lines.append(f"  🚨 Critical: {len(self.critical_values)}")
        if self.discrepancies:
            lines.append("\nDISCREPANCIES FOUND:")
            for d in self.discrepancies:
                lines.append(f"  ⚠️  {d}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  NUMBER / CURRENCY PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def _try_parse_number(s: str) -> Optional[float]:
    """Parse a number from OCR text, handling currency symbols and formatting."""
    if not s:
        return None
    s = s.strip()
    # Remove currency symbols
    s = re.sub(r'[\$€£¥₹]', '', s)
    s = s.strip()
    # European format: 1.234,56
    if re.match(r'^\d{1,3}(\.\d{3})+(,\d+)?$', s):
        s = s.replace('.', '').replace(',', '.')
    # American format: 1,234.56
    elif re.match(r'^\d{1,3}(,\d{3})+(\.\d+)?$', s):
        s = s.replace(',', '')
    else:
        # Simple comma-decimal: 3,5 → 3.5
        if re.match(r'^\d+,\d{1,2}$', s):
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    # Strip percentage
    s = s.rstrip('%')
    # Remove trailing non-numeric
    s = re.sub(r'[^\d.\-]', '', s)
    try:
        return float(s)
    except ValueError:
        return None


def _detect_currency(text: str) -> str:
    """Detect the primary currency symbol used in the document."""
    counts = {sym: text.count(sym) for sym in ('$', '€', '£', '¥', '₹')}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "$"


def _approx_equal(a: float, b: float, tol: float = 0.02) -> bool:
    """Check if two financial values are approximately equal (within tol absolute)."""
    return abs(a - b) <= tol


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  METADATA EXTRACTION (key-value patterns from text)
# ═══════════════════════════════════════════════════════════════════════════════

_METADATA_PATTERNS: List[Tuple[str, str, re.Pattern]] = [
    ("Invoice Number", "Invoice Header", re.compile(
        r'invoice\s*(?:number|no\.?|#|num\.?)[:\s]+([A-Z0-9\-/]+)',
        re.IGNORECASE,
    )),
    ("Invoice Date", "Invoice Header", re.compile(
        r'(?:invoice\s*)?date[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+\.?\s+\d{1,2},?\s*\d{4})',
        re.IGNORECASE,
    )),
    ("Due Date", "Invoice Header", re.compile(
        r'(?:due|payment|pay\s*by)\s*date[:\s]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+\.?\s+\d{1,2},?\s*\d{4})',
        re.IGNORECASE,
    )),
    ("PO Number", "Invoice Header", re.compile(
        r'(?:purchase\s*order|P\.?O\.?)\s*(?:number|no\.?|#)?[:\s]+([A-Z0-9\-/]+)',
        re.IGNORECASE,
    )),
    ("Vendor", "Parties", re.compile(
        r'(?:from|vendor|supplier|bill(?:ed)?\s*from|sold\s*by)[:\s]+([^\n|]{3,60})',
        re.IGNORECASE,
    )),
    ("Bill To", "Parties", re.compile(
        r'(?:bill(?:ed)?\s*to|ship(?:ped)?\s*to|customer|client|sold\s*to)[:\s]+([^\n|]{3,60})',
        re.IGNORECASE,
    )),
    ("Payment Terms", "Invoice Header", re.compile(
        r'(?:payment\s*terms?|terms?)[:\s]+([^\n|]{3,40})',
        re.IGNORECASE,
    )),
    ("Subtotal", "Financial Totals", re.compile(
        r'sub[-\s]?total[:\s]+([\$€£¥₹]?\s*[\d,]+\.?\d*)',
        re.IGNORECASE,
    )),
    ("Tax Amount", "Financial Totals", re.compile(
        r'(?:tax|vat|gst|hst)\s*(?:\(?\d+\.?\d*%?\)?)?\s*[:\s]+([\$€£¥₹]?\s*[\d,]+\.?\d*)',
        re.IGNORECASE,
    )),
    ("Tax Rate", "Financial Totals", re.compile(
        r'(?:tax|vat|gst|hst)\s*(?:rate|@)\s*[:\s]?(\d+\.?\d*\s*%)',
        re.IGNORECASE,
    )),
    ("Discount", "Financial Totals", re.compile(
        r'discount[:\s]+([\$€£¥₹]?\s*[\d,]+\.?\d*)',
        re.IGNORECASE,
    )),
    ("Grand Total", "Financial Totals", re.compile(
        r'(?:grand\s*)?total\s*(?:due|amount|payable)?\s*[:\s]+([\$€£¥₹]?\s*[\d,]+\.?\d*)',
        re.IGNORECASE,
    )),
]


def _extract_metadata(text: str, page: int) -> List[ExtractedField]:
    """Extract invoice metadata from key-value text patterns."""
    fields: List[ExtractedField] = []
    seen: set = set()

    for field_name, category, pattern in _METADATA_PATTERNS:
        if field_name in seen:
            continue
        m = pattern.search(text)
        if not m:
            continue
        raw_value = m.group(1).strip()
        numeric = _try_parse_number(raw_value)
        fields.append(ExtractedField(
            field_name=field_name,
            display_name=field_name,
            value_str=raw_value,
            numeric_value=numeric,
            category=category,
            status=FieldStatus.EXTRACTED,
            page=page,
            source_text=m.group(0).strip(),
        ))
        seen.add(field_name)

    return fields


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  LINE ITEM TABLE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

_LINE_ITEM_KEYWORDS: Dict[str, set] = {
    "description": {
        "description", "item", "items", "service", "services", "product", "products",
        "details", "work", "labor", "labour", "désignation", "particulars", "task",
    },
    "quantity": {"quantity", "qty", "units", "unit", "hours", "hrs", "days", "qté", "q"},
    "unit_price": {
        "unit price", "unit_price", "price", "rate", "unit cost", "per unit", "each",
        "prix unitaire", "unit rate", "rate/unit", "cost",
    },
    "amount": {
        "amount", "total", "line total", "ext price", "extended", "ext. price",
        "line amount", "montant", "price", "net",
    },
}


def _is_line_item_table(headers: Optional[List[str]], rows: List[List[str]]) -> bool:
    """Check if a table looks like an invoice line items table."""
    if not rows:
        return False
    if headers:
        lower = [h.lower().strip() for h in headers]
        all_kw = set()
        for kws in _LINE_ITEM_KEYWORDS.values():
            all_kw.update(kws)
        matches = sum(1 for h in lower if h in all_kw)
        if matches >= 2:
            return True
    # Heuristic: multiple rows, last column is numeric (amounts)
    numeric_last = sum(
        1 for row in rows
        if row and _try_parse_number(row[-1]) is not None
    )
    return len(rows) >= 2 and numeric_last >= len(rows) * 0.5


def _detect_line_item_columns(
    headers: Optional[List[str]], rows: List[List[str]]
) -> Dict[str, int]:
    """Detect column roles for a line items table."""
    roles: Dict[str, int] = {}

    if headers:
        for idx, h in enumerate(headers):
            h_lower = h.lower().strip()
            for role, keywords in _LINE_ITEM_KEYWORDS.items():
                if h_lower in keywords:
                    if role not in roles:
                        roles[role] = idx
                    break
        if len(roles) >= 2:
            return roles

    # Fallback heuristics
    if not rows:
        return roles
    ncols = max(len(r) for r in rows)
    if ncols >= 2:
        roles["description"] = 0
        roles["amount"] = ncols - 1
    if ncols >= 3:
        roles["unit_price"] = ncols - 2
    if ncols >= 4:
        roles["quantity"] = 1
    return roles


def _extract_line_items(
    headers: Optional[List[str]], rows: List[List[str]], page: int
) -> List[LineItem]:
    """Extract and verify line items from a table."""
    cols = _detect_line_item_columns(headers, rows)
    desc_col = cols.get("description", 0)
    qty_col = cols.get("quantity")
    price_col = cols.get("unit_price")
    amt_col = cols.get("amount")

    items: List[LineItem] = []
    for row in rows:
        if not row:
            continue
        description = row[desc_col].strip() if desc_col < len(row) else ""
        if not description or len(description) < 2:
            continue
        # Skip if description looks like a header or totals row
        desc_lower = description.lower()
        if any(kw in desc_lower for kw in ("subtotal", "total", "tax", "discount", "vat", "gst")):
            continue

        quantity = _try_parse_number(row[qty_col]) if qty_col is not None and qty_col < len(row) else None
        unit_price = _try_parse_number(row[price_col]) if price_col is not None and price_col < len(row) else None
        amount = _try_parse_number(row[amt_col]) if amt_col is not None and amt_col < len(row) else None

        # Verify math if we have all three
        status = FieldStatus.EXTRACTED
        note = ""
        if quantity is not None and unit_price is not None and amount is not None:
            expected = round(quantity * unit_price, 2)
            if _approx_equal(expected, amount):
                status = FieldStatus.MATCHED
                note = f"{quantity} × {unit_price} = {amount:.2f} ✓"
            else:
                status = FieldStatus.FLAGGED
                note = f"Expected {quantity} × {unit_price} = {expected:.2f}, found {amount:.2f}"

        items.append(LineItem(
            description=description,
            quantity=quantity,
            unit_price=unit_price,
            amount=amount,
            page=page,
            status=status,
            note=note,
        ))

    return items


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TABLE PARSING (markdown)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_markdown_tables(text: str) -> List[Dict]:
    """Parse all markdown tables, returning list of {heading, headers, rows} dicts."""
    tables: List[Dict] = []
    lines = text.split("\n")
    current_lines: List[str] = []
    current_heading = "Invoice Data"
    in_table = False

    for line in lines:
        stripped = line.strip()
        heading_m = re.match(r'^#{1,4}\s+(.+)$', stripped)
        if heading_m:
            current_heading = heading_m.group(1).strip()

        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                current_lines = [stripped]
                in_table = True
            else:
                current_lines.append(stripped)
        else:
            if in_table and current_lines:
                t = _process_table(current_lines, current_heading)
                if t:
                    tables.append(t)
                current_lines = []
            in_table = False

    if current_lines:
        t = _process_table(current_lines, current_heading)
        if t:
            tables.append(t)

    return tables


def _process_table(lines: List[str], heading: str) -> Optional[Dict]:
    """Process raw table lines into structured {heading, headers, rows}."""
    header_row: Optional[List[str]] = None
    data_rows: List[List[str]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^\|[\s\-:|]+\|$', stripped):
            continue
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        cells = [c for c in cells if c]
        if not cells:
            continue
        if i == 0 and _looks_like_header(cells):
            header_row = cells
        else:
            data_rows.append(cells)

    if not data_rows:
        return None
    return {"heading": heading, "headers": header_row, "rows": data_rows}


def _looks_like_header(cells: List[str]) -> bool:
    """Check if cells look like a header row (mostly text, no numbers)."""
    lower = [c.lower().strip() for c in cells]
    all_kw = set()
    for kws in _LINE_ITEM_KEYWORDS.values():
        all_kw.update(kws)
    text_kw_matches = sum(1 for c in lower if c in all_kw)
    numeric = sum(1 for c in lower if re.match(r'^[\d.,\$€£]+$', c))
    return text_kw_matches >= 2 or (len(cells) > 0 and numeric == 0 and text_kw_matches >= 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  TOTAL VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def _verify_totals(report: InvoiceAnalysisReport) -> None:
    """Verify financial totals and update field statuses + discrepancies."""
    # Update financial fields from extracted metadata
    for f in report.fields:
        if f.field_name == "Subtotal":
            report.subtotal = f.numeric_value
        elif f.field_name == "Tax Amount":
            report.tax_amount = f.numeric_value
        elif f.field_name == "Tax Rate":
            report.tax_rate = f.numeric_value
        elif f.field_name == "Discount":
            report.discount = f.numeric_value
        elif f.field_name == "Grand Total":
            report.grand_total = f.numeric_value

    # Verify line item sum vs subtotal
    if report.line_items:
        line_sum = sum(li.amount for li in report.line_items if li.amount is not None)
        if line_sum > 0:
            # Find and update Subtotal field status
            for f in report.fields:
                if f.field_name == "Subtotal" and f.numeric_value is not None:
                    if _approx_equal(line_sum, f.numeric_value):
                        f.status = FieldStatus.MATCHED
                        f.note = f"Line items sum {line_sum:.2f} matches subtotal ✓"
                    else:
                        f.status = FieldStatus.FLAGGED
                        f.note = f"Line items sum {line_sum:.2f} ≠ subtotal {f.numeric_value:.2f}"
                        report.discrepancies.append(
                            f"Subtotal mismatch: line items sum to {line_sum:.2f} "
                            f"but subtotal is {f.numeric_value:.2f}"
                        )
                    break
            else:
                # No subtotal field found — infer it
                report.subtotal = line_sum

    # Verify subtotal + tax - discount = grand total
    sub = report.subtotal
    tax = report.tax_amount or 0.0
    disc = report.discount or 0.0
    grand = report.grand_total

    if sub is not None and grand is not None:
        expected_grand = round(sub + tax - disc, 2)
        for f in report.fields:
            if f.field_name == "Grand Total":
                if _approx_equal(expected_grand, grand):
                    f.status = FieldStatus.MATCHED
                    f.note = f"{sub:.2f} + {tax:.2f} tax - {disc:.2f} discount = {grand:.2f} ✓"
                else:
                    f.status = FieldStatus.CRITICAL
                    f.note = (
                        f"Expected {sub:.2f} + {tax:.2f} - {disc:.2f} = {expected_grand:.2f}, "
                        f"found {grand:.2f}"
                    )
                    report.discrepancies.append(
                        f"Grand total mismatch: expected {expected_grand:.2f}, "
                        f"document states {grand:.2f}"
                    )
                break


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════════

def analyse(pages_markdown: List[str]) -> InvoiceAnalysisReport:
    """Run full invoice analysis on OCR-extracted pages."""
    report = InvoiceAnalysisReport()

    all_fields: List[ExtractedField] = []
    all_line_items: List[LineItem] = []
    seen_fields: set = set()

    for page_idx, md in enumerate(pages_markdown):
        report.currency = _detect_currency(md) or report.currency

        # Extract metadata
        meta_fields = _extract_metadata(md, page_idx)
        for f in meta_fields:
            if f.field_name not in seen_fields:
                all_fields.append(f)
                seen_fields.add(f.field_name)

        # Extract tables
        tables = _parse_markdown_tables(md)
        for tbl in tables:
            if _is_line_item_table(tbl.get("headers"), tbl.get("rows", [])):
                items = _extract_line_items(
                    tbl.get("headers"), tbl.get("rows", []), page_idx
                )
                all_line_items.extend(items)
            else:
                # Extract as generic financial fields from non-line-item tables
                rows = tbl.get("rows", [])
                headers = tbl.get("headers")
                for row in rows:
                    if len(row) == 2:
                        name, val = row[0].strip(), row[1].strip()
                        numeric = _try_parse_number(val)
                        if name and val and name not in seen_fields:
                            status = FieldStatus.EXTRACTED
                            # Check if it's a financial total field
                            name_lower = name.lower()
                            if any(kw in name_lower for kw in ("total", "subtotal", "tax", "vat", "gst", "discount")):
                                category = "Financial Totals"
                            else:
                                category = tbl.get("heading", "Invoice Data")
                            all_fields.append(ExtractedField(
                                field_name=name,
                                display_name=name,
                                value_str=val,
                                numeric_value=numeric,
                                category=category,
                                status=status,
                                page=page_idx,
                            ))
                            seen_fields.add(name)

    report.fields = all_fields
    report.line_items = all_line_items

    _verify_totals(report)

    # Mark remaining extracted fields as EXTRACTED (no further verification)
    for f in report.fields:
        if f.status == FieldStatus.UNKNOWN:
            f.status = FieldStatus.EXTRACTED

    logger.info(
        "Invoice analysis: %d fields, %d line items, %d discrepancies",
        len(report.fields), len(report.line_items), len(report.discrepancies),
    )
    return report


def get_invoice_summary_table(report: InvoiceAnalysisReport) -> str:
    """Return a markdown table of extracted invoice fields."""
    if not report.fields:
        return "*No invoice fields extracted from this document.*"
    lines = [
        "| Field | Value | Status | Category |",
        "|-------|-------|--------|----------|",
    ]
    status_icons = {
        FieldStatus.MATCHED: "✅",
        FieldStatus.FLAGGED: "⚠️",
        FieldStatus.CRITICAL: "🚨",
        FieldStatus.EXTRACTED: "📋",
        FieldStatus.MISSING: "❌",
        FieldStatus.UNKNOWN: "—",
    }
    for f in report.fields:
        icon = status_icons.get(f.status, "—")
        lines.append(
            f"| {f.display_name} | {f.value_str} | "
            f"{icon} {f.status.value.upper()} | {f.category} |"
        )
    return "\n".join(lines)
