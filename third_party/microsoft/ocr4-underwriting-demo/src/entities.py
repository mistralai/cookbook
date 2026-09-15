"""Entity-to-region reconciliation: which extracted field maps onto which OCR block.

Pure functions over plain dicts (blocks + facts), no case store, no models, no network. Both
the backend service and the UI's HTTP client use this same copy: matching a value to its
region is deterministic given the OCR blocks and the facts, so there is no reason to make it a
round-trip. The UI feeds these pairs straight into its overlay.
"""
from __future__ import annotations

# Order sets how labels read within a region (distinctive fields first).
_FIELD_LABEL = [
    ("borrower_name", "Borrower"), ("property_address", "Property"),
    ("loan_amount", "Loan amount"), ("property_value", "Property value"),
    ("annual_income", "Annual income"), ("monthly_debts", "Monthly debts"),
    ("credit_score", "Credit score"), ("employment_years", "Employment"),
    ("loan_purpose", "Purpose"),
]
# Keywords per field, ordered specific -> generic. The fallback tries them in this order and
# the first that hits any block wins, so a precise label ("annual salary") beats a generic
# one ("gross", which also appears in a pay stub's "Gross Pay" earnings row).
_KEYWORDS = {
    "loan_amount": ["loan amount", "loan"],
    "property_value": ["property value", "estimated property", "purchase price", "appraised"],
    "annual_income": ["annual salary", "gross annual", "annual income", "salary", "income"],
    "monthly_debts": ["monthly debt", "debt"],
    "credit_score": ["credit score", "credit", "fico"],
    "employment_years": ["years of service", "years employed", "employ", "tenure", "years"],
    "loan_purpose": ["loan purpose", "purpose"],
    "property_address": ["property address", "address"],
    "borrower_name": ["borrower name", "employee name", "name"],
}
_MONEY_FIELDS = {"annual_income", "loan_amount", "property_value", "monthly_debts"}


def _value_variants(name, value) -> list[str]:
    out = [str(value).lower()]
    if name in _MONEY_FIELDS:
        try:
            n = float(value)
            out += [f"{n:,.0f}", f"{int(n)}"]
        except (TypeError, ValueError):
            pass
    if name == "credit_score":
        try:
            out.append(str(int(float(value))))
        except (TypeError, ValueError):
            pass
    return [v for v in out if v]


def entity_pairs_from(blocks: list[dict], facts: dict):
    """Map extracted fields onto the OCR region that holds them.

    OCR 4 returns section-level layout blocks (each section is one region), so we label each
    region with every entity we pulled from it, e.g. a Borrower section box reading
    "Borrower / Credit score / Employment / Annual income".
    Returns [(block, [(field, label), ...])].
    """
    blocks = blocks or []
    assign: dict[int, list[tuple[str, str]]] = {}
    for name, label in _FIELD_LABEL:
        val = facts.get(name)
        if val is None:
            continue
        variants = _value_variants(name, val)
        keywords = _KEYWORDS.get(name, [])
        idx = None
        # Value match pins the region. If the value appears in several blocks, prefer the one
        # that also carries a field keyword (annual income next to "annual salary", not a
        # stray copy of the number in another section).
        val_hits = [i for i, b in enumerate(blocks)
                    if (b.get("content") or "").lower()
                    and any(v in (b.get("content") or "").lower() for v in variants)]
        if val_hits:
            idx = next((i for i in val_hits
                        if any(k in (blocks[i].get("content") or "").lower() for k in keywords)),
                       val_hits[0])
        # Fallback: try keywords in specificity order so a precise label wins over a generic
        # one regardless of block position (e.g. "annual salary" beats a pay stub's "gross").
        if idx is None:
            for k in keywords:
                hit = next((i for i, b in enumerate(blocks)
                            if k in (b.get("content") or "").lower()), None)
                if hit is not None:
                    idx = hit
                    break
        if idx is not None:
            assign.setdefault(idx, []).append((name, label))
    return [(blocks[i], fields) for i, fields in assign.items()]
