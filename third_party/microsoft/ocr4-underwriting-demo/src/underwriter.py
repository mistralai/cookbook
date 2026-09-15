"""Underwriter: deterministic scoring and routing (OCR-4-only, no chat model).

Policy lives in mortgage_rules: metrics, hard limits, and the auto-versus-human bounds.
The recommendation, risk score, and rationale are derived from those numbers here, so no
reasoning model is required. Code owns the whole decision.
"""
from __future__ import annotations

import mortgage_rules as R
from underwriting_schema import Decision, Metrics, MortgageFacts, Outcome, Stage


def _risk_score(facts: MortgageFacts, m: Metrics) -> int:
    """A transparent 0-100 risk score: higher LTV, DTI, and lower credit add risk."""
    score = 0.0
    if m.ltv is not None:
        score += min(45.0, max(0.0, m.ltv - 60.0) * 1.2)
    if m.dti is not None:
        score += min(30.0, max(0.0, m.dti - 30.0) * 1.5)
    if facts.credit_score is not None:
        score += min(25.0, max(0.0, 740 - facts.credit_score) * 0.15)
    return int(round(min(100.0, score)))


def _explain(facts: MortgageFacts, m: Metrics, rec: Outcome, fails: list[str]) -> tuple[list[str], str]:
    """Human-readable factors + rationale built from the metrics and thresholds."""
    factors: list[str] = []
    if m.ltv is not None:
        factors.append(f"LTV {m.ltv}% vs {R.MAX_LTV}% limit (auto ceiling {R.AUTO_MAX_LTV}%)")
    if m.dti is not None:
        factors.append(f"DTI {m.dti}% vs {R.MAX_DTI}% limit (auto ceiling {R.AUTO_MAX_DTI}%)")
    if facts.credit_score is not None:
        factors.append(f"credit {facts.credit_score} vs {R.MIN_CREDIT} min (auto floor {R.AUTO_MIN_CREDIT})")
    factors.extend(fails)

    if rec == Outcome.decline:
        rationale = "Declined: " + ("; ".join(fails) if fails else "fails policy limits") + "."
    elif rec == Outcome.approve:
        rationale = ("Auto-approved: LTV, DTI, and credit are all within the conservative "
                     "auto-approval bounds and no hard limit is breached.")
    else:
        rationale = ("Referred to a human: within overall policy but outside the conservative "
                     "auto-approval bounds (LTV, DTI, credit, or loan size), so it warrants review.")
    return factors, rationale


async def assess(facts: MortgageFacts) -> tuple[Decision, Metrics]:
    """Compute metrics and a deterministic recommendation. Async to match callers."""
    metrics = R.compute_metrics(facts)
    fails = R.hard_fails(facts, metrics)

    if fails:
        rec = Outcome.decline
    elif R.within_auto_limits(facts, metrics):
        rec = Outcome.approve
    else:
        rec = Outcome.refer

    factors, rationale = _explain(facts, metrics, rec, fails)
    decision = Decision(
        recommendation=rec,
        risk_score=_risk_score(facts, metrics),
        confidence=1.0,  # deterministic: the rules are certain
        rationale=rationale,
        factors=factors,
        hard_fails=fails,
    )
    return decision, metrics


def route(facts: MortgageFacts, metrics: Metrics, decision: Decision) -> Stage:
    """Auto-decide clear cases; send the rest to a human. Returns the next stage."""
    conf = decision.confidence or 0.0
    clean_approve = (
        not decision.hard_fails
        and R.within_auto_limits(facts, metrics)
        and decision.recommendation == Outcome.approve
        and conf >= R.MIN_CONFIDENCE
    )
    clear_decline = (
        bool(decision.hard_fails)
        and decision.recommendation == Outcome.decline
        and conf >= R.MIN_CONFIDENCE
    )
    if clean_approve or clear_decline:
        return Stage.auto_decided
    return Stage.pending_review


def auto_outcome(stage: Stage, decision: Decision) -> Outcome | None:
    """The outcome for an auto-decided case; None when a human still owns it."""
    return decision.recommendation if stage == Stage.auto_decided else None
