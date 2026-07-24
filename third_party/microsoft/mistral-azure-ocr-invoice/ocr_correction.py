"""
OCR post-processing: error correction and completeness checking.

Handles two main problems:
  1. Character-level OCR errors common in medical documents
     (e.g. "Gig/1" → "Gig/l", "0.5 mg/d1" → "0.5 mg/dl")
  2. Skipped rows / missing data in tables
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CHARACTER-LEVEL OCR CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# --- 1a. Medical unit corrections (most common OCR errors) ---------------------
# These map regex patterns to their correct replacements.
# The digit "1" is commonly misread for lowercase "l" (and vice versa) in units.

UNIT_CORRECTIONS: List[Tuple[str, str]] = [
    # l ↔ 1 confusion in units  (covers Giga/1 → Giga/l, Téra/1 → Téra/l, etc.)
    (r"\b(g|mg|µg|ug|ng|pg|mEq|mmol|µmol|umol|IU|U|k?U|Gig|gig|Giga|giga|Tera|Téra|tera|téra)\s*/\s*1\b",
     r"\1/l"),                                                # e.g. mg/1 → mg/l
    (r"\b(g|mg|µg|ug|ng|pg|mEq|mmol|µmol|umol|IU|U|k?U|Gig|gig|Giga|giga|Tera|Téra)\s*/\s*d1\b",
     r"\1/dl"),                                               # e.g. mg/d1 → mg/dl
    (r"\b(g|mg|µg|ug|ng|pg|mEq|mmol|µmol|umol|IU|U|k?U|Gig|gig|Giga|giga|Tera|Téra)\s*/\s*m1\b",
     r"\1/ml"),                                               # e.g. IU/m1 → IU/ml
    (r"\bce11s?\b", "cells"),                                 # ce11s → cells
    (r"\bcel1s?\b", "cells"),                                 # cel1s → cells
    (r"\bp1ate1ets?\b", "platelets"),                         # p1ate1ets → platelets
    (r"\bmmH9\b", "mmHg"),                                    # mmH9 → mmHg
    (r"\bmm0l\b", "mmol"),                                    # mm0l → mmol
    (r"\bum0l\b", "µmol"),                                    # um0l → µmol
    (r"\bµm0l\b", "µmol"),
    (r"\bf1/\b", "fl/"),                                      # f1/ → fl/
    (r"\b(\d+\.?\d*)\s*f1\b", r"\1 fl"),                     # 90 f1 → 90 fl
    (r"(?<=\|\s)f1(?=\s*\|)", "fl"),                           # | f1 | → | fl | (table cell)
    (r"\bf1\b(?=\s*\|)", "fl"),                               # f1 at end of table cell
    (r"\bcel(?:ls?|1s?)\s*/\s*m1\b", "cells/ml"),           # cells/m1 → cells/ml
    (r"\b/m1\b", "/ml"),                                     # generic /m1 → /ml
    (r"\b/d1\b", "/dl"),                                     # generic /d1 → /dl
    (r"\b10\^(\d)\s*/\s*1\b", r"10^\1/l"),                   # 10^9/1 → 10^9/l
    (r"\b1(?:0|O)\^(\d)\s*/\s*(?:1|l)\b", r"10^\1/l"),      # 1O^9/l → 10^9/l

    # O ↔ 0 confusion
    (r"\bO\.(\d)", r"0.\1"),                                  # O.5 → 0.5
    (r"(\d)O(\d)", r"\g<1>0\2"),                              # 1O5 → 105

    # Common medical term corruption
    (r"\bhemog1obin\b", "hemoglobin"),
    (r"\bhem0globin\b", "hemoglobin"),
    (r"\bg1ucose\b", "glucose"),
    (r"\bcreatinine\b", "creatinine"),                        # already correct – anchor
    (r"\bcreat1nine\b", "creatinine"),
    (r"\bbi1irubin\b", "bilirubin"),
    (r"\bbil1rubin\b", "bilirubin"),
    (r"\bcholestero1\b", "cholesterol"),
    (r"\btrig1ycerides?\b", "triglycerides"),
    (r"\btr1glycerides?\b", "triglycerides"),
    (r"\ba1bumin\b", "albumin"),
    (r"\bp1asma\b", "plasma"),
    (r"\bserum\b", "serum"),                                  # anchor
    (r"\bl(?:eu|ue)kocytes?\b", "leukocytes"),
    (r"\berythrocyte\b", "erythrocyte"),                      # anchor
    (r"\bthrombocytes?\b", "thrombocytes"),                   # anchor
]

# --- 1b. Whitespace / formatting fixes ----------------------------------------
FORMAT_CORRECTIONS: List[Tuple[str, str]] = [
    (r"(\d)\s*,\s*(\d{3})\b", r"\1,\2"),                    # 1 , 000 → 1,000
    (r"(\d)\s+(\.\s*\d)", r"\1\2"),                          # 3 .5 → 3.5
    (r"[^\S\n]{2,}", " "),                                   # collapse multi-spaces (preserve newlines)
]


@dataclass
class CorrectionRecord:
    """Tracks a single correction applied."""
    original: str
    corrected: str
    rule: str
    page: int
    position: int  # char offset


@dataclass
class CorrectionReport:
    corrections: List[CorrectionRecord] = field(default_factory=list)
    original_text: str = ""
    corrected_text: str = ""

    @property
    def correction_count(self) -> int:
        return len(self.corrections)

    def summary(self) -> str:
        if not self.corrections:
            return "No OCR corrections needed."
        lines = [f"Applied {self.correction_count} corrections:"]
        for c in self.corrections:
            lines.append(f"  Page {c.page}: '{c.original}' → '{c.corrected}' [{c.rule}]")
        return "\n".join(lines)


def correct_ocr_text(text: str, page_index: int = 0) -> Tuple[str, List[CorrectionRecord]]:
    """
    Apply all OCR corrections to a block of text.
    Returns (corrected_text, list_of_corrections).
    """
    records: List[CorrectionRecord] = []
    corrected = text

    all_rules = UNIT_CORRECTIONS + FORMAT_CORRECTIONS
    for pattern, replacement in all_rules:
        for m in re.finditer(pattern, corrected, re.IGNORECASE):
            original_match = m.group(0)
            new_match = re.sub(pattern, replacement, original_match, flags=re.IGNORECASE)
            if new_match != original_match:
                records.append(CorrectionRecord(
                    original=original_match,
                    corrected=new_match,
                    rule=pattern[:40],
                    page=page_index,
                    position=m.start(),
                ))
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

    return corrected, records


def _clean_table_row(row: str) -> str:
    """
    Clean a markdown table row by:
      - Removing empty cells created by double-pipe OCR artefacts
      - Merging standalone flag cells (-, +) with the adjacent reference cell
      - Collapsing {...} / (…) placeholder cells
    """
    if not row.strip().startswith("|"):
        return row

    # Split into cells (skip outer empty strings)
    parts = row.split("|")
    # parts[0] and parts[-1] are empty strings from leading/trailing |
    cells = [p.strip() for p in parts[1:-1]]

    # Remove empty cells
    cells = [c for c in cells if c]

    if not cells:
        return row

    # Merge standalone flag cells (-, +) into the next cell (reference range)
    merged: List[str] = []
    i = 0
    while i < len(cells):
        cell = cells[i]
        # If cell is just a flag marker (-, +) and there's a next cell, merge
        if cell in ("-", "+", "- ", "+ ") and i + 1 < len(cells):
            flag = cell.strip()
            merged.append(f"{flag} {cells[i + 1]}")
            i += 2
        else:
            merged.append(cell)
            i += 1

    return "| " + " | ".join(merged) + " |"


def _is_separator_row(row: str) -> bool:
    """Check if a markdown table row is a separator (e.g. | --- | --- |)."""
    return bool(re.match(r'^\|[\s\-:|]+\|$', row.strip()))


def _clean_tables_in_text(text: str) -> str:
    """Find all markdown tables in text and clean their row structure."""
    lines = text.split("\n")
    result: List[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            if _is_separator_row(stripped):
                result.append(stripped)
            else:
                result.append(_clean_table_row(stripped))
        else:
            in_table = False
            result.append(line)

    return "\n".join(result)


# ── Section title patterns (lines that look like headings but aren't marked) ──
# Match UPPERCASE lines (possibly with dashes/spaces) that are standalone
_SECTION_PATTERNS = [
    # Pure uppercase lines, >= 3 chars, possibly with hyphens/dashes/spaces
    # e.g. "HEMATOLOGIE - CYTOLOGIE", "BIOCHIMIE", "EXAMENS COMPLÉMENTAIRES..."
    # Excludes: repeating page headers/footers, page numbers, short codes
    re.compile(
        r'^(?!CRH |Pat\.|Imprimé|ASSISTANCE PUBLIQUE|UN STIMULATEUR|'
        r'\d+/\d+$)'
        r'([A-ZÀ-ÖÙ-Ý][A-ZÀ-ÖÙ-Ý0-9\s\-–—/,\'().]{2,})$'
    ),
]

# Lines to never promote to headings (repeated headers/footers/contact info)
_HEADER_IGNORE = {
    "ASSISTANCE PUBLIQUE HÔPITAUX DE PARIS",
    "ASSISTANCE PUBLIQUE",
    "HÔPITAUX DE PARIS",
    "AP-HP",
    "SOS ENDOCARDITE",
    "SOS EMBOLIE",
    "PULMONAIRE",
}

# Lines too short to be meaningful sections
_MIN_SECTION_LEN = 4

# Known section title keywords (case-insensitive matching for mixed-case)
_SECTION_KEYWORDS = {
    "numération", "examens sanguins", "examens urinaires",
    "coagulation", "immunologie", "sérologie", "hémostase",
    "enzymologie", "marqueurs", "gaz du sang", "bilan hépatique",
    "bilan lipidique", "bilan rénal", "bilan thyroïdien",
    "bilan martial", "bilan phosphocalcique",
}


def _detect_sections(text: str) -> str:
    """
    Detect standalone section-title lines and promote them to markdown headings.
    Only promotes lines that are:
      - Not already markdown headings (don't start with #)
      - Not inside a table (don't start with |)
      - Match section patterns (uppercase or known keywords)
      - Relatively short (< 80 chars, so we don't catch paragraph text)
    """
    lines = text.split("\n")
    result: List[str] = []
    in_table = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track table state
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            result.append(line)
            continue
        else:
            in_table = False

        # Skip if already a heading
        if stripped.startswith("#"):
            result.append(line)
            continue

        # Skip empty lines
        if not stripped:
            result.append(line)
            continue

        # Skip long lines (not section titles)
        if len(stripped) > 80:
            result.append(line)
            continue

        # Skip short lines that are likely abbreviations, not sections
        if len(stripped) < _MIN_SECTION_LEN:
            result.append(line)
            continue

        # Check uppercase pattern
        is_section = False
        if stripped not in _HEADER_IGNORE:
            # Also skip lines that look like addresses / contact info
            # (contain digits mixed with letters in ways sections don't)
            has_address_pattern = bool(
                re.search(r'\d{2,}\s+(AVENUE|RUE|BOULEVARD|PLACE)\b', stripped, re.IGNORECASE)
                or re.search(r'^\d+\s+', stripped)  # starts with number
                or re.search(r'\b\d{5}\b', stripped)  # postal code
            )
            if not has_address_pattern:
                # Skip lines that look like proper names (CDS, Dr, Pr, etc.)
                has_name_pattern = bool(
                    re.match(r'^(CDS|DR|PR|MME?|M\.) ', stripped, re.IGNORECASE)
                )
                if not has_name_pattern:
                    for pattern in _SECTION_PATTERNS:
                        if pattern.match(stripped):
                            is_section = True
                            break

        # Check known keywords
        if not is_section and stripped.lower() in _SECTION_KEYWORDS:
            is_section = True

        # Also catch mixed-case known section-like patterns
        if not is_section:
            lower = stripped.lower()
            for kw in _SECTION_KEYWORDS:
                if lower == kw:
                    is_section = True
                    break

        if is_section:
            # Determine heading level based on context
            # If there's already a ## heading nearby, use ### for subsections
            level = "###"
            result.append(f"{level} {stripped}")
        else:
            result.append(line)

    return "\n".join(result)


def structure_document(text: str) -> str:
    """
    Apply structural improvements to OCR output:
      1. Clean table row structure (remove empty cells, merge flags)
      2. Restructure lab tables (extract sub-headers, add column labels)
      3. Detect section titles and promote to headings
    """
    text = _clean_tables_in_text(text)
    text = _restructure_lab_tables(text)
    text = _detect_sections(text)
    return text


def _restructure_lab_tables(text: str) -> str:
    """
    Restructure lab/biology markdown tables to be human-readable:
      - Extract single-cell title rows (e.g. "| Numération |") and promote
        them to #### headings above the table
      - Remove machine/equipment reference rows (e.g. "| Compte globules | {...} | DXH2401 |")
      - Insert a proper column header row when a data-table has no header
    """
    lines = text.split("\n")
    result: List[str] = []
    table_lines: List[str] = []
    in_table = False

    def _flush_table(tbl: List[str]) -> List[str]:
        """Process a collected table block and return output lines."""
        if not tbl:
            return []

        output: List[str] = []
        data_rows: List[str] = []
        sep_row: Optional[str] = None
        title: Optional[str] = None

        for row in tbl:
            stripped = row.strip()

            # Separator row
            if _is_separator_row(stripped):
                sep_row = stripped
                continue

            # Extract cells
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            non_empty = [c for c in cells if c]

            # Single-cell row → table sub-heading (e.g. "| Numération |")
            if len(non_empty) == 1 and not re.search(r'\d', non_empty[0]):
                title = non_empty[0]
                continue

            # Machine reference rows (contain {...}, (…), or equipment model IDs
            # like DXH2401, but no numeric test values)
            if any(c in ("{...}", "(…)", "{…}") for c in non_empty):
                continue
            # Rows where first cell looks like "Compte globules" and has equipment ref
            if len(non_empty) >= 2 and re.search(r'^[A-Z]{2,}\d{2,}$', non_empty[-1]):
                continue

            data_rows.append(stripped)

        # Emit title as #### heading
        if title:
            output.append(f"#### {title}")
            output.append("")

        # If we have data rows, build a proper table
        if data_rows:
            # Determine column count from data rows
            max_cols = 0
            for r in data_rows:
                cols = [c.strip() for c in r.split("|")[1:-1]]
                cols = [c for c in cols if c]
                max_cols = max(max_cols, len(cols))

            # Add column header based on detected structure
            if max_cols == 4:
                output.append("| Test | Unité | Valeur | Référence |")
                output.append("| --- | --- | --- | --- |")
            elif max_cols == 3:
                output.append("| Test | Unité | Valeur |")
                output.append("| --- | --- | --- |")
            else:
                # Unknown structure — use separator only
                if sep_row:
                    output.append(sep_row)

            for r in data_rows:
                output.append(r)

        return output

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            table_lines.append(line)
        else:
            if in_table:
                # Is this a lab/biology table? Check if it's near a biology section
                # We'll restructure all tables that have the typical lab format
                # (rows with 3-4 cells where cell 2 looks like a unit)
                has_lab_pattern = False
                for tl in table_lines:
                    cells = [c.strip() for c in tl.strip().split("|")[1:-1]]
                    non_empty = [c for c in cells if c]
                    if len(non_empty) >= 3:
                        # Check if second cell looks like a unit
                        unit_cell = non_empty[1] if len(non_empty) > 1 else ""
                        if re.match(
                            r'^(g/[dlL]|mg/[dlL]|mmol/l|µmol/l|μmol/l|%|fl|pg|'
                            r'Giga/l|Téra/l|mL/min|mm/h|UI/[lL]|ng/[mlL]|'
                            r'µg/[dlL]|mEq/[lL]|10\^)',
                            unit_cell, re.IGNORECASE
                        ):
                            has_lab_pattern = True
                            break

                if has_lab_pattern:
                    result.extend(_flush_table(table_lines))
                else:
                    # Keep non-lab tables unchanged
                    result.extend(table_lines)

                table_lines = []
                in_table = False

            result.append(line)

    # Flush any trailing table
    if table_lines:
        result.extend(_flush_table(table_lines))

    return "\n".join(result)


def correct_ocr_pages(pages_markdown: List[str]) -> CorrectionReport:
    """Correct all pages and return a full report."""
    all_records: List[CorrectionRecord] = []
    corrected_pages: List[str] = []

    for idx, md in enumerate(pages_markdown):
        # Step 1: Character-level OCR corrections
        corrected, records = correct_ocr_text(md, page_index=idx)
        # Step 2: Structural improvements (section headers, table cleanup)
        corrected = structure_document(corrected)
        corrected_pages.append(corrected)
        all_records.extend(records)

    return CorrectionReport(
        corrections=all_records,
        original_text="\n\n---\n\n".join(pages_markdown),
        corrected_text="\n\n---\n\n".join(corrected_pages),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  COMPLETENESS / SKIP DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompletenessIssue:
    page: int
    issue_type: str          # "empty_row", "row_count_low", "missing_values", "suspicious_gap"
    description: str
    severity: str = "warning"  # "info", "warning", "error"


@dataclass
class CompletenessReport:
    issues: List[CompletenessIssue] = field(default_factory=list)
    table_count: int = 0
    total_rows_detected: int = 0

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    def summary(self) -> str:
        if not self.issues:
            return f"Completeness OK — {self.table_count} table(s), {self.total_rows_detected} row(s) detected."
        lines = [f"Found {len(self.issues)} completeness issue(s):"]
        for issue in self.issues:
            icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(issue.severity, "•")
            lines.append(f"  {icon} Page {issue.page}: [{issue.issue_type}] {issue.description}")
        return "\n".join(lines)


def _parse_markdown_tables(md: str) -> List[List[List[str]]]:
    """
    Parse markdown tables into list of tables, each a list of rows,
    each row a list of cell values.
    """
    tables: List[List[List[str]]] = []
    current_table: List[List[str]] = []
    in_table = False

    for line in md.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            # Skip separator lines like |---|---|
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            current_table.append(cells)
            in_table = True
        else:
            if in_table and current_table:
                tables.append(current_table)
                current_table = []
            in_table = False

    if current_table:
        tables.append(current_table)

    return tables


def _parse_html_tables(md: str) -> List[List[List[str]]]:
    """Parse HTML tables from the content."""
    tables: List[List[List[str]]] = []
    table_blocks = re.findall(r"<table[\s\S]*?</table>", md, re.IGNORECASE)

    for table_html in table_blocks:
        rows: List[List[str]] = []
        for tr_match in re.finditer(r"<tr[\s\S]*?</tr>", table_html, re.IGNORECASE):
            tr = tr_match.group(0)
            cells = re.findall(r"<t[dh][^>]*>([\s\S]*?)</t[dh]>", tr, re.IGNORECASE)
            # Strip inner HTML tags
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)

    return tables


def check_completeness(pages_markdown: List[str]) -> CompletenessReport:
    """
    Analyse OCR output for signs of skipped / missing data.
    Checks:
      - Empty cells in table rows
      - Tables with suspiciously few rows (< 2 data rows)
      - Large numeric gaps in sequential data (e.g. row numbering)
      - Completely empty pages
    """
    issues: List[CompletenessIssue] = []
    total_tables = 0
    total_rows = 0

    for page_idx, md in enumerate(pages_markdown):
        # Check for empty pages
        if not md.strip():
            issues.append(CompletenessIssue(
                page=page_idx,
                issue_type="empty_page",
                description="Page returned empty content — possible OCR skip.",
                severity="error",
            ))
            continue

        # Parse tables (both markdown and HTML)
        md_tables = _parse_markdown_tables(md)
        html_tables = _parse_html_tables(md)
        all_tables = md_tables + html_tables
        total_tables += len(all_tables)

        for t_idx, table in enumerate(all_tables):
            total_rows += len(table)

            if len(table) < 2:
                issues.append(CompletenessIssue(
                    page=page_idx,
                    issue_type="row_count_low",
                    description=f"Table {t_idx + 1} has only {len(table)} row(s) — possible skipped rows.",
                    severity="warning",
                ))

            # Check for empty cells
            for r_idx, row in enumerate(table):
                empty_cells = sum(1 for c in row if not c.strip())
                if empty_cells > 0 and empty_cells >= len(row) * 0.5:
                    issues.append(CompletenessIssue(
                        page=page_idx,
                        issue_type="empty_row",
                        description=f"Table {t_idx + 1}, row {r_idx + 1}: {empty_cells}/{len(row)} cells empty.",
                        severity="warning",
                    ))

            # Check for sequential numbering gaps
            first_col_nums: List[Optional[int]] = []
            for row in table:
                if row:
                    m = re.match(r"^\s*(\d+)\s*$", row[0])
                    first_col_nums.append(int(m.group(1)) if m else None)

            nums_only = [n for n in first_col_nums if n is not None]
            if len(nums_only) >= 3:
                for i in range(1, len(nums_only)):
                    gap = nums_only[i] - nums_only[i - 1]
                    if gap > 1:
                        issues.append(CompletenessIssue(
                            page=page_idx,
                            issue_type="suspicious_gap",
                            description=(
                                f"Table {t_idx + 1}: row numbering jumps from "
                                f"{nums_only[i - 1]} to {nums_only[i]} — {gap - 1} row(s) may be skipped."
                            ),
                            severity="error",
                        ))

    return CompletenessReport(
        issues=issues,
        table_count=total_tables,
        total_rows_detected=total_rows,
    )
