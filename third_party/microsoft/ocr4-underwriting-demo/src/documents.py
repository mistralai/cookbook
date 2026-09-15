"""The standard mortgage document package: the checklist and the classifier.

A mortgage application is not one file; it is a set of standard documents. OCR 4
already returns a `document_type` in its annotation, so each upload is mapped to a
checklist slot here, and the case tracks which standard documents are present.
"""
from __future__ import annotations

from underwriting_schema import UnderwritingCase

# Canonical slots. (key, label, required) — required drives "what's still missing".
REQUIRED_DOCS: list[tuple[str, str, bool]] = [
    ("application_1003", "Loan application (1003)", True),
    ("paystub", "Pay stub", True),
    ("w2", "W-2", True),
    ("bank_statement", "Bank statement", True),
]
OPTIONAL_DOCS: list[tuple[str, str, bool]] = [
    ("tax_return", "Tax return", False),
    ("id", "Government ID", False),
    ("purchase_agreement", "Purchase agreement", False),
    ("appraisal", "Appraisal", False),
]
_ALL = REQUIRED_DOCS + OPTIONAL_DOCS
LABELS: dict[str, str] = {key: label for key, label, _ in _ALL}
_REQUIRED_KEYS = {key for key, _, req in REQUIRED_DOCS if req}

# Ordered so the most specific patterns win. Bank statements mention "payroll deposit",
# so they are checked before pay stubs, and pay-stub keywords stay narrow.
_PATTERNS: list[tuple[str, list[str]]] = [
    ("application_1003", ["uniform residential", "form 1003", "loan application", "1003", "urla"]),
    ("w2", ["w-2", "w2", "wage and tax"]),
    ("bank_statement", ["bank statement", "account statement", "statement of account",
                        "beginning balance", "ending balance"]),
    ("tax_return", ["form 1040", "1040", "tax return", "schedule c"]),
    ("paystub", ["pay stub", "paystub", "earnings statement", "net pay this period"]),
    ("id", ["driver license", "driver's license", "passport", "identification card", "state id"]),
    ("purchase_agreement", ["purchase agreement", "purchase and sale", "sales contract"]),
    ("appraisal", ["appraisal report", "appraised value", "uniform residential appraisal"]),
]


def classify(doc_type_raw: str = "", summary: str = "", markdown: str = "") -> tuple[str, str]:
    """Map OCR 4's annotation (and a little text) to a checklist slot. Returns (key, label)."""
    hay = " ".join([doc_type_raw or "", summary or "", (markdown or "")[:800]]).lower()
    for key, pats in _PATTERNS:
        if any(p in hay for p in pats):
            return key, LABELS[key]
    return "other", "Other document"


def checklist(case: UnderwritingCase) -> list[dict]:
    """The package state: one row per standard slot, plus any extra classified docs."""
    have: dict[str, object] = {}
    for d in case.documents:
        have.setdefault(d.type, d)  # first upload wins a slot
    rows: list[dict] = []
    for key, label, required in _ALL:
        d = have.get(key)
        status = "missing" if d is None else getattr(d, "status", "received")
        rows.append({"type": key, "label": label, "required": required,
                     "status": status, "doc_id": getattr(d, "id", None)})
    for key, d in have.items():  # anything classified as "other"
        if key not in LABELS:
            rows.append({"type": key, "label": getattr(d, "type_label", "Other document"),
                         "required": False, "status": getattr(d, "status", "received"),
                         "doc_id": getattr(d, "id", None)})
    return rows


def missing_required(case: UnderwritingCase) -> list[dict]:
    return [r for r in checklist(case) if r["required"] and r["status"] == "missing"]
