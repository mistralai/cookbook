"""Deterministic mortgage rules: parsing, metrics, validation, and hard limits.

Policy lives here as plain code, not in a prompt. The underwriter agent reasons on
top of these numbers; these functions decide what is missing and what breaks a limit.
All thresholds are illustrative and configurable.
"""
from __future__ import annotations

import re

from underwriting_schema import Metrics, MissingField, MortgageFacts

# Policy limits — flagged and shown to reviewer above these.
MAX_LTV = 80.0
MAX_DTI = 43.0
MIN_CREDIT = 620
CONFORMING_LIMIT = 806_500.0  # illustrative 2025 baseline conforming limit

# Tighter auto-approve bounds — cases that breach these go to human review.
AUTO_MAX_LTV = 65.0
AUTO_MAX_DTI = 35.0
AUTO_MIN_CREDIT = 700
MIN_CONFIDENCE = 0.85

# Wider bounds beyond which a case is a clear hard fail.
DECLINE_LTV = 95.0
DECLINE_DTI = 50.0
DECLINE_CREDIT = 580

# Estimated payment assumptions for the DTI calculation.
DEFAULT_RATE = 0.065
DEFAULT_TERM_MONTHS = 360

REQUIRED_FIELDS: dict[str, str] = {
    "borrower_name": "the borrower's full name",
    "annual_income": "the borrower's gross annual income",
    "loan_amount": "the loan amount requested",
    "property_value": "the property value or purchase price",
    "credit_score": "the borrower's credit score",
    "monthly_debts": "the borrower's total monthly debt payments",
    "employment_years": "years at current employment",
    "property_address": "the property address",
}

NUMERIC_FIELDS = {
    "annual_income", "loan_amount", "property_value",
    "credit_score", "monthly_debts", "employment_years",
}

# These are valid at zero (no other debt, brand-new job); the rest must be positive.
NONNEGATIVE_FIELDS = {"monthly_debts", "employment_years"}


def parse_number(value) -> float | None:
    """Turn '$216,000', '43%', '720' into a float; None if it cannot be read."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def estimated_monthly_payment(loan_amount: float | None,
                              rate: float = DEFAULT_RATE,
                              term: int = DEFAULT_TERM_MONTHS) -> float | None:
    if not loan_amount or loan_amount <= 0:
        return None
    r = rate / 12
    if r == 0:
        return loan_amount / term
    factor = (1 + r) ** term
    return loan_amount * r * factor / (factor - 1)


def compute_metrics(f: MortgageFacts) -> Metrics:
    m = Metrics()
    if f.loan_amount and f.property_value and f.property_value > 0:
        m.ltv = round(f.loan_amount / f.property_value * 100, 1)
    pmt = estimated_monthly_payment(f.loan_amount)
    m.estimated_monthly_payment = round(pmt, 2) if pmt else None
    if f.annual_income and f.annual_income > 0:
        monthly_income = f.annual_income / 12
        total_debt = (f.monthly_debts or 0) + (pmt or 0)
        m.dti = round(total_debt / monthly_income * 100, 1)
    return m


def validate(f: MortgageFacts) -> list[MissingField]:
    """Find fields that are missing, unreadable, or out of range."""
    missing: list[MissingField] = []
    for name, label in REQUIRED_FIELDS.items():
        val = getattr(f, name)
        if val is None or (isinstance(val, str) and not val.strip()):
            missing.append(MissingField(name=name, prompt=f"What is {label}?", reason="missing"))
            continue
        if name in NUMERIC_FIELDS:
            num = parse_number(val)
            floor_bad = num is not None and (num < 0 if name in NONNEGATIVE_FIELDS else num <= 0)
            if num is None or floor_bad:
                missing.append(MissingField(
                    name=name, prompt=f"I could not read {label}. What is it?", reason="unreadable"))
            elif name == "credit_score" and not (300 <= num <= 850):
                missing.append(MissingField(
                    name=name, prompt=f"That credit score looks off. What is {label}?", reason="out_of_range"))
    return missing


def hard_fails(f: MortgageFacts, m: Metrics) -> list[str]:
    fails: list[str] = []
    if m.ltv is not None and m.ltv > DECLINE_LTV:
        fails.append(f"LTV {m.ltv}% exceeds {DECLINE_LTV}%")
    if m.dti is not None and m.dti > DECLINE_DTI:
        fails.append(f"DTI {m.dti}% exceeds {DECLINE_DTI}%")
    if f.credit_score is not None and f.credit_score < DECLINE_CREDIT:
        fails.append(f"Credit score {f.credit_score} below {DECLINE_CREDIT}")
    if f.loan_amount and f.loan_amount > CONFORMING_LIMIT:
        fails.append(f"Loan amount {f.loan_amount:.0f} exceeds conforming limit {CONFORMING_LIMIT:.0f}")
    return fails


def within_auto_limits(f: MortgageFacts, m: Metrics) -> bool:
    """True only for genuinely clean cases that can skip human review."""
    return (
        m.ltv is not None and m.ltv <= AUTO_MAX_LTV
        and m.dti is not None and m.dti <= AUTO_MAX_DTI
        and f.credit_score is not None and f.credit_score >= AUTO_MIN_CREDIT
        and f.loan_amount is not None and f.loan_amount <= CONFORMING_LIMIT
    )
