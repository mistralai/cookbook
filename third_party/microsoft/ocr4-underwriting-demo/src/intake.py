"""Intake: build mortgage facts from OCR 4's annotation, then run the conversational
missing-value loop.

OCR-4-only design: document field extraction comes from OCR 4's custom
`document_annotation` (see agents.FIELDS_ANNOTATION_SCHEMA), and the applicant's typed
answers are parsed deterministically here. No chat/reasoning model is used.
"""
from __future__ import annotations

import re

import mortgage_rules as R
from underwriting_schema import MissingField, MortgageFacts

_FIELD_KEYS = list(R.REQUIRED_FIELDS.keys()) + ["loan_purpose"]


def _coerce(name: str, value):
    if value is None:
        return None
    if name in R.NUMERIC_FIELDS:
        num = R.parse_number(value)
        if num is None:
            return None
        return int(num) if name == "credit_score" else num
    text = str(value).strip()
    return text or None


def _merge(facts: MortgageFacts, updates: dict) -> MortgageFacts:
    data = facts.model_dump()
    for name in _FIELD_KEYS:
        if name in updates:
            coerced = _coerce(name, updates[name])
            if coerced is not None:
                data[name] = coerced
    return MortgageFacts(**data)


def facts_from_annotation(ann: dict) -> MortgageFacts:
    """Map OCR 4's annotation payload straight into mortgage facts (no LLM)."""
    return _merge(MortgageFacts(), ann or {})


# ---- Deterministic free-text parser for the intake chat loop ----
# Matches a number that may carry $ , and a k/m magnitude suffix (e.g. "300k", "$1,250").
# The suffix must attach to the digits (no space) so "720, monthly" doesn't read the
# "m" of "monthly" as a magnitude.
_NUM = r"\$?\s*(\d[\d,]*(?:\.\d+)?)([kKmM])?\b"


def _to_number(digits: str, suffix: str | None) -> float:
    n = float(digits.replace(",", ""))
    if suffix:
        n *= 1_000_000 if suffix.lower() == "m" else 1_000
    return n


# Keyword -> field. Order matters: more specific phrases first so "monthly debts"
# is not swallowed by a bare "debts", and "property value/home worth" beats "loan".
_FIELD_PATTERNS: list[tuple[str, str]] = [
    ("annual_income", r"(?:annual\s*income|income|salary|earns?|makes?)"),
    ("property_value", r"(?:home\s*worth|property\s*value|purchase\s*price|worth|value)"),
    ("loan_amount", r"(?:loan(?:\s*amount)?)"),
    ("credit_score", r"(?:credit(?:\s*score)?|fico)"),
    ("monthly_debts", r"(?:monthly\s*debts?|debts?)"),
    ("employment_years", r"(?:employ(?:ed|ment)?|tenure)"),
]


def _parse_message(text: str) -> dict:
    updates: dict = {}
    low = text.lower()

    # loan_purpose is a keyword, not a number.
    if re.search(r"\brefinanc|refi\b", low):
        updates["loan_purpose"] = "Refinance"
    elif re.search(r"\bpurchase|buy(?:ing)?\b", low):
        updates["loan_purpose"] = "Purchase"

    for field, kw in _FIELD_PATTERNS:
        m = re.search(kw + r"\s*(?:of|is|:|=|at)?\s*" + _NUM, low)
        if m:
            updates[field] = _to_number(m.group(1), m.group(2))

    # "6 years" / "6 yrs" -> employment_years when not already captured.
    if "employment_years" not in updates:
        m = re.search(_NUM + r"\s*(?:years?|yrs?)", low)
        if m:
            updates["employment_years"] = _to_number(m.group(1), m.group(2))

    # property_address: free text after "property"/"address" (demo puts it last).
    m = re.search(r"(?:property\s*address|property|address)\s*(?:is|:|=)?\s*(.+)", text, re.IGNORECASE)
    if m:
        addr = m.group(1).strip().rstrip(".")
        # Drop a trailing clause that is actually another field (e.g. "..., credit 720").
        addr = re.split(r",\s*(?:credit|loan|income|debts?|worth|value|purchase|refinanc)", addr, flags=re.IGNORECASE)[0].strip()
        if addr:
            updates["property_address"] = addr

    return updates


async def apply_message(facts: MortgageFacts, missing: list[MissingField], user_text: str) -> MortgageFacts:
    """Parse the applicant's free-text answer deterministically and merge it into the facts."""
    updates = _parse_message(user_text)

    # Single-value fallback: a bare answer (e.g. a suggestion-button value) maps to the
    # one outstanding field when the message has no explicit field keywords.
    if not updates and len(missing) == 1:
        field = missing[0].name
        stripped = user_text.strip()
        if field in R.NUMERIC_FIELDS:
            m = re.search(_NUM, stripped)
            if m:
                updates[field] = _to_number(m.group(1), m.group(2))
        elif field == "loan_purpose":
            updates[field] = "Refinance" if re.search(r"refinanc|refi", stripped, re.IGNORECASE) else "Purchase"
        elif stripped:
            updates[field] = stripped

    return _merge(facts, updates)


def next_prompt(missing: list[MissingField]) -> str:
    """One friendly question covering the outstanding fields."""
    if not missing:
        return "Everything looks complete. Ready to submit for underwriting."
    if len(missing) == 1:
        return missing[0].prompt
    first = missing[0].prompt
    rest = ", ".join(R.REQUIRED_FIELDS.get(m.name, m.name) for m in missing[1:])
    return f"{first} After that I still need: {rest}."
