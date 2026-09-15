"""Orchestration used by both the API and the Gradio views.

Ties OCR, extraction, the intake loop, underwriting, and review together over the
case store, with stage guards, an append-only audit, and trace spans.
"""
from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime, timezone

import documents as D
import mortgage_rules as R
from case_store import get_store, new_case_id
from agents import FIELDS_ANNOTATION_SCHEMA, parse_json_object
from foundry_agents import (
    EXTRACTOR_AGENT, INTAKE_AGENT, UNDERWRITING_AGENT, agents_enabled, build_agent,
)
from intake import apply_message, facts_from_annotation, next_prompt
from observability import get_tracer, record_ocr, setup_observability
from ocr_client import ocr_with_annotation
from underwriter import assess, auto_outcome, route
from underwriting_schema import (
    AuditEntry, Document, MortgageFacts, Outcome, ReviewRecord, Stage, UnderwritingCase,
    can_transition,
)

# Persisted originals so the reviewer can show the source document under the overlay.
_DOC_DIR = os.path.join(tempfile.gettempdir(), "mortgage_case_docs")
os.makedirs(_DOC_DIR, exist_ok=True)


def _save_document(case_id: str, data: bytes, name: str) -> str:
    ext = os.path.splitext(name)[1] or ".bin"
    path = os.path.join(_DOC_DIR, f"{case_id}{ext}")
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _merge_facts(current: MortgageFacts, incoming: MortgageFacts) -> MortgageFacts:
    """Assemble facts across the package: fill any field a later document supplies."""
    data = current.model_dump()
    for key, val in incoming.model_dump().items():
        if val is not None and data.get(key) in (None, ""):
            data[key] = val
    return MortgageFacts(**data)


async def _ingest_document(case: UnderwritingCase, data: bytes, name: str) -> Document:
    """OCR + classify one file, add it to the package, and merge its facts into the case.

    The most recent document becomes the primary (the default overlay); facts are
    accumulated across every document uploaded so far.
    """
    # OCR 4 is the perception step. Wrap it in an `ocr.process` span (with page count, document
    # type, and token metrics) so the document read is visible in the trace, right before the
    # extractor agent span. This is a code step, not an agent call, so it is traced regardless of
    # whether the agents run hosted or via the client fallback.
    with get_tracer().start_as_current_span("ocr.process") as span:
        span.set_attribute("document.name", name)
        result = ocr_with_annotation(data, name, annotation_schema=FIELDS_ANNOTATION_SCHEMA)
        ann = result.get("annotation") or {}
        ocr_doc_type = ann.get("document_type", "unknown") if isinstance(ann, dict) else "unknown"
        span.set_attribute("document.type", ocr_doc_type)
        span.set_attribute("ocr.pages", result.get("pages_count", 0))
        record_ocr(result.get("pages_count", 0), result.get("usage", {}), ocr_doc_type)
    md = result.get("markdown", "")
    doc_type_raw = ann.get("document_type", "") if isinstance(ann, dict) else ""
    ann_summary = ann.get("summary", "") if isinstance(ann, dict) else ""
    key, label = D.classify(doc_type_raw, ann_summary, md)
    doc_id = f"doc-{len(case.documents) + 1:02d}"
    path = _save_document(f"{case.case_id}_{doc_id}", data, name)
    doc = Document(
        id=doc_id, type=key, type_label=label, name=name, path=path,
        doc_type_raw=doc_type_raw, summary=ann_summary, markdown=md,
        ocr_blocks=result.get("blocks") or [], ocr_width=result.get("width"),
        ocr_height=result.get("height"),
        status="received" if (md and key != "other") else "needs_review", added_at=_now(),
    )
    case.documents.append(doc)
    # Primary (most recent) document powers the default overlay + backward-compat fields.
    case.markdown, case.document_path = md, path
    case.ocr_blocks, case.ocr_width, case.ocr_height = doc.ocr_blocks, doc.ocr_width, doc.ocr_height
    incoming = facts_from_annotation(ann)
    if key == "bank_statement":
        # A bank statement shows balances and deposits, not the borrower's stated income, loan,
        # credit, or employment. Keep only identity so a misread figure cannot set those fields.
        incoming = incoming.model_copy(update={
            "annual_income": None, "loan_amount": None, "property_value": None,
            "credit_score": None, "monthly_debts": None, "employment_years": None,
            "loan_purpose": None,
        })
    case.facts = _merge_facts(case.facts, incoming)
    await _run_extractor(case, doc)
    return doc


def _readiness_stage(case: UnderwritingCase) -> Stage:
    """Ready only when the fields are complete AND the required documents are present."""
    complete = not case.missing and not D.missing_required(case)
    return Stage.ready if complete else Stage.collecting


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit(case: UnderwritingCase, actor: str, event: str, detail: str = "") -> None:
    case.audit.append(AuditEntry(at=_now(), actor=actor, event=event, detail=detail))


def _advance(case: UnderwritingCase, nxt: Stage) -> None:
    if case.stage == nxt:
        return
    if not can_transition(case.stage, nxt):
        raise ValueError(f"illegal transition {case.stage.value} -> {nxt.value}")
    case.stage = nxt


# ---- helpers for the chat UIs ----
_LABELS = {
    "borrower_name": "Borrower", "property_address": "Property", "loan_purpose": "Purpose",
    "annual_income": "Annual income", "loan_amount": "Loan amount", "property_value": "Property value",
    "credit_score": "Credit score", "monthly_debts": "Monthly debts", "employment_years": "Employment",
}
_MONEY = {"annual_income", "loan_amount", "property_value", "monthly_debts"}


def _fmt(name: str, value) -> str:
    try:
        if name in _MONEY:
            return f"${float(value):,.0f}"
        if name == "credit_score":
            return str(int(float(value)))
        if name == "employment_years":
            return f"{float(value):g} yrs"
    except (TypeError, ValueError):
        pass
    return str(value)


def _facts_markdown(case: UnderwritingCase) -> str:
    items = [(k, v) for k, v in case.facts.model_dump().items() if v is not None]
    if not items:
        return "_Nothing read yet._"
    return "\n".join(f"- **{_LABELS.get(k, k)}:** {_fmt(k, v)}" for k, v in items)


def summary(case: UnderwritingCase) -> str:
    return _facts_markdown(case)


def _docs_markdown(case: UnderwritingCase) -> str:
    """A compact package checklist for the chat."""
    marks = {"received": "✓", "needs_review": "!", "missing": "○"}
    lines = []
    for row in D.checklist(case):
        if not row["required"] and row["status"] == "missing":
            continue  # do not nag about optional docs that were never provided
        lines.append(f"- {marks.get(row['status'], '○')} {row['label']}")
    return "\n".join(lines) if lines else "_No documents yet._"


def reply_for(case: UnderwritingCase) -> str:
    if case.stage == Stage.collecting:
        parts = [f"**Read by Mistral OCR 4**\n{_facts_markdown(case)}",
                 f"**Documents**\n{_docs_markdown(case)}"]
        missing_docs = D.missing_required(case)
        if case.missing:
            parts.append(next_prompt(case.missing))
        if missing_docs:
            names = ", ".join(r["label"] for r in missing_docs)
            parts.append(f"Please also upload: **{names}**. Attach the next document below.")
        return "\n\n".join(parts)
    if case.stage == Stage.ready:
        return (f"**Read by Mistral OCR 4**\n{_facts_markdown(case)}\n\n"
                f"**Documents**\n{_docs_markdown(case)}\n\n"
                "**All set.** Type `submit` to send for underwriting.")
    return _facts_markdown(case)


# ---- Foundry agents (optional, behind USE_FOUNDRY_AGENTS; deterministic fallback on error) ----
_AGENT_CACHE: dict = {}


def _get_agent(name: str):
    if name not in _AGENT_CACHE:
        _AGENT_CACHE[name] = build_agent(name)
    return _AGENT_CACHE[name]


def _usage_tokens(resp) -> int | None:
    """Total token count from an agent response. UsageDetails is a TypedDict (a dict at runtime)."""
    u = getattr(resp, "usage_details", None)
    if not isinstance(u, dict):
        return None
    total = u.get("total_token_count")
    if isinstance(total, (int, float)):
        return int(total)
    it = u.get("input_token_count") or 0
    ot = u.get("output_token_count") or 0
    return (int(it) + int(ot)) or None


def _add_usage(case: UnderwritingCase, agent: str, resp) -> None:
    tok = _usage_tokens(resp)
    if tok:
        case.agent_usage[agent] = case.agent_usage.get(agent, 0) + tok


async def _run_extractor(case: UnderwritingCase, doc: Document) -> None:
    """Extractor agent: pull named entities from THIS document's OCR markdown (display only).

    Entities are stored on the document, so each file's people, organizations, and ids stay
    separate (a pay stub's employer is not commingled with the loan application's borrower).
    The case-level list is kept as the deduped union across the package.
    """
    md = doc.markdown or ""
    if not agents_enabled() or not md.strip():
        return
    try:
        with get_tracer().start_as_current_span("agent.extractor") as span:
            span.set_attribute("document.type", doc.type_label)
            resp = await _get_agent(EXTRACTOR_AGENT).run(md)
        doc_seen, case_seen = set(doc.entities), set(case.entities)
        for e in parse_json_object(resp.text).get("entities") or []:
            s = e if isinstance(e, str) else str(e)
            if not s:
                continue
            if s not in doc_seen:
                doc.entities.append(s)
                doc_seen.add(s)
            if s not in case_seen:
                case.entities.append(s)
                case_seen.add(s)
        _add_usage(case, "extractor", resp)
    except Exception:
        pass  # never break ingest on an agent error


async def _intake_question(case: UnderwritingCase) -> str:
    """Intake agent: phrase the next question over the outstanding fields."""
    need = ", ".join(R.REQUIRED_FIELDS.get(m.name, m.name) for m in case.missing)
    try:
        with get_tracer().start_as_current_span("agent.intake"):
            resp = await _get_agent(INTAKE_AGENT).run(
                f"Known so far:\n{_facts_markdown(case)}\nStill missing: {need}. "
                "Ask the applicant for the missing items in one short, friendly message.")
        _add_usage(case, "intake", resp)
        return (resp.text or "").strip() or next_prompt(case.missing)
    except Exception:
        return next_prompt(case.missing)


def _rationale_only(text: str) -> str:
    """Drop any 'Recommendation:'/'Risk score:' lines and a leading 'Rationale:' label."""
    if not text:
        return ""
    kept = [ln for ln in text.splitlines()
            if not ln.strip().lower().startswith(("recommendation:", "risk score:"))]
    out = "\n".join(kept).strip()
    return re.sub(r"^rationale\s*:\s*", "", out, flags=re.IGNORECASE).strip()


async def _run_underwriting_rationale(case: UnderwritingCase) -> None:
    """Underwriting agent: explain the code's decision (it does not make its own).

    Policy stays in code: underwriter.assess/route own the recommendation, risk, and routing.
    The agent is anchored to that recommendation so its rationale cannot contradict the card.
    """
    if not agents_enabled():
        return
    m = case.metrics
    rec = case.decision.recommendation.value if case.decision.recommendation else "refer"
    try:
        with get_tracer().start_as_current_span("agent.underwriter"):
            resp = await _get_agent(UNDERWRITING_AGENT).run(
                f"The underwriting decision has already been made: {rec}. "
                f"Metrics: loan-to-value {m.ltv}%, debt-to-income {m.dti}%, credit score {case.facts.credit_score}, "
                f"loan amount {case.facts.loan_amount}. Auto-approval ceilings are loan-to-value 65%, "
                f"debt-to-income 35%, credit score 700; overall limits are loan-to-value 80%, debt-to-income 43%, "
                f"credit score 620. In two sentences, explain why this case is '{rec}', citing loan-to-value, "
                f"debt-to-income, and credit score against those thresholds. Write every term out in full with no "
                "abbreviations. Output only the explanation, with no 'Recommendation:' or 'Risk score:' line.")
        txt = _rationale_only(resp.text)
        if txt:
            case.decision.rationale = txt
        _add_usage(case, "underwriter", resp)
    except Exception:
        pass  # keep the deterministic rationale on any agent error


def _with_agent_footer(case: UnderwritingCase, text: str) -> str:
    """Append per-agent token usage. The extractor's entities are no longer dumped into the chat;
    they render in the app's 'Extracted entities' tab, built from the case object."""
    if case and case.agent_usage:
        text += "\n\n_Agent tokens: " + ", ".join(f"{k} {v}" for k, v in case.agent_usage.items()) + "_"
    return text


async def reply_for_agent(case: UnderwritingCase) -> str:
    """Agent-aware chat reply: the intake agent phrases the ask; falls back to reply_for."""
    if case is None or not agents_enabled():
        return _with_agent_footer(case, reply_for(case))
    if case.stage == Stage.collecting:
        parts = [f"**Read by Mistral OCR 4**\n{_facts_markdown(case)}",
                 f"**Documents**\n{_docs_markdown(case)}"]
        if case.missing:
            parts.append(await _intake_question(case))
        missing_docs = D.missing_required(case)
        if missing_docs:
            names = ", ".join(r["label"] for r in missing_docs)
            parts.append(f"Please also upload: **{names}**. Attach the next document below.")
        return _with_agent_footer(case, "\n\n".join(parts))
    return _with_agent_footer(case, reply_for(case))


def decision_text(case: UnderwritingCase) -> str:
    d = case.decision
    out = case.review.outcome or d.recommendation
    lines = [
        f"Stage: {case.stage.value}",
        f"Outcome: {out.value if out else 'pending'}",
        f"LTV: {case.metrics.ltv}  DTI: {case.metrics.dti}  risk: {d.risk_score}  confidence: {d.confidence}",
    ]
    if d.rationale:
        lines.append(f"Rationale: {d.rationale}")
    if d.hard_fails:
        lines.append("Hard fails: " + "; ".join(d.hard_fails))
    return "\n".join(lines)


# ---- lifecycle ----
async def create_case(data: bytes, name: str, owner: str = "demo") -> UnderwritingCase:
    setup_observability()
    store = get_store()
    case = UnderwritingCase(case_id=new_case_id(), owner=owner, stage=Stage.received)
    _audit(case, owner, "received", name)
    store.create(case)

    with get_tracer().start_as_current_span("intake.create") as span:
        span.set_attribute("case.id", case.case_id)
        _advance(case, Stage.reading)
        doc = await _ingest_document(case, data, name)
        _advance(case, Stage.extracting)
        case.missing = R.validate(case.facts)
        _advance(case, _readiness_stage(case))
        _audit(case, "system", "extracting",
               f"{doc.type_label}; {len(case.missing)} field(s), "
               f"{len(D.missing_required(case))} document(s) missing")
        span.set_attribute("case.stage", case.stage.value)
        return store.put(case)


async def create_case_stream(data: bytes, name: str, owner: str = "customer"):
    """Same as create_case, but yields the case at each stage for a live UI."""
    setup_observability()
    store = get_store()
    case = UnderwritingCase(case_id=new_case_id(), owner=owner, stage=Stage.received)
    _audit(case, owner, "received", name)
    store.create(case)
    with get_tracer().start_as_current_span("intake.create") as span:
        span.set_attribute("case.id", case.case_id)
        _advance(case, Stage.reading)
        yield store.put(case)

        doc = await _ingest_document(case, data, name)
        _advance(case, Stage.extracting)
        _audit(case, "system", "reading", f"ocr complete: {doc.type_label}")
        yield store.put(case)

        case.missing = R.validate(case.facts)
        _advance(case, _readiness_stage(case))
        _audit(case, "system", "extracting",
               f"{len(case.missing)} field(s), {len(D.missing_required(case))} document(s) missing")
        span.set_attribute("case.stage", case.stage.value)
        yield store.put(case)


async def add_document(case_id: str, data: bytes, name: str, owner: str = "customer") -> UnderwritingCase:
    """Append another document to an existing case, re-classify, and re-assess readiness."""
    store = get_store()
    case = store.get(case_id)
    if case is None:
        raise KeyError(case_id)
    with get_tracer().start_as_current_span("intake.add_document") as span:
        span.set_attribute("case.id", case_id)
        if case.stage == Stage.received:
            _advance(case, Stage.reading)
        doc = await _ingest_document(case, data, name)
        case.missing = R.validate(case.facts)
        if case.stage in (Stage.reading, Stage.extracting, Stage.collecting, Stage.ready):
            case.stage = _readiness_stage(case)  # direct set: ready<->collecting is not a guarded move
        _audit(case, owner, "document_added", f"{doc.type_label} ({doc.status})")
        span.set_attribute("case.stage", case.stage.value)
        return store.put(case)


async def create_case_stream_add(case_id: str, data: bytes, name: str, owner: str = "customer"):
    """Streaming append for the live UI: yield reading, then the classified result."""
    store = get_store()
    case = store.get(case_id)
    if case is None:
        return
    with get_tracer().start_as_current_span("intake.add_document") as span:
        span.set_attribute("case.id", case_id)
        doc = await _ingest_document(case, data, name)
        case.missing = R.validate(case.facts)
        if case.stage in (Stage.reading, Stage.extracting, Stage.collecting, Stage.ready):
            case.stage = _readiness_stage(case)
        _audit(case, owner, "document_added", f"{doc.type_label} ({doc.status})")
        span.set_attribute("case.stage", case.stage.value)
        yield store.put(case)


async def submit_stream(case_id: str):
    """Same as submit, but yields the case at underwriting and at the decision."""
    store = get_store()
    case = store.get(case_id)
    if case is None or case.missing or case.stage != Stage.ready:
        yield case
        return
    with get_tracer().start_as_current_span("underwrite.submit") as span:
        span.set_attribute("case.id", case_id)
        _advance(case, Stage.underwriting)
        yield store.put(case)

        decision, metrics = await assess(case.facts)
        case.decision, case.metrics = decision, metrics
        if case.submitted_facts is None:  # capture the pre-review values once, for the ledger
            case.submitted_facts = case.facts.model_copy(deep=True)
            case.submitted_metrics = case.metrics.model_copy(deep=True)
        await _run_underwriting_rationale(case)
        nxt = route(case.facts, metrics, decision)
        _advance(case, nxt)
        if nxt == Stage.auto_decided:
            outcome = auto_outcome(nxt, decision)
            case.review = ReviewRecord(reviewer="auto", outcome=outcome, note="auto decision", at=_now())
            _audit(case, "system", "auto_decided", outcome.value if outcome else "")
            _advance(case, Stage.finalized)
        else:
            _audit(case, "system", "pending_review", "routed to an underwriter")
        span.set_attribute("case.stage", case.stage.value)
        yield store.put(case)


async def add_message(case_id: str, text: str, owner: str = "demo") -> UnderwritingCase:
    store = get_store()
    case = store.get(case_id)
    if case is None:
        raise KeyError(case_id)
    with get_tracer().start_as_current_span("intake.message") as span:
        span.set_attribute("case.id", case_id)
        case.facts = await apply_message(case.facts, case.missing, text)
        case.missing = R.validate(case.facts)
        if case.stage in (Stage.collecting, Stage.ready, Stage.extracting):
            case.stage = _readiness_stage(case)
        _audit(case, owner, "message", f"{len(case.missing)} field(s) still missing")
        span.set_attribute("case.stage", case.stage.value)
        return store.put(case)


async def submit(case_id: str) -> UnderwritingCase:
    store = get_store()
    case = store.get(case_id)
    if case is None:
        raise KeyError(case_id)
    if case.missing or case.stage != Stage.ready:
        return case  # not ready to submit
    with get_tracer().start_as_current_span("underwrite.submit") as span:
        span.set_attribute("case.id", case_id)
        _advance(case, Stage.underwriting)
        decision, metrics = await assess(case.facts)
        case.decision, case.metrics = decision, metrics
        if case.submitted_facts is None:  # capture the pre-review values once, for the ledger
            case.submitted_facts = case.facts.model_copy(deep=True)
            case.submitted_metrics = case.metrics.model_copy(deep=True)
        await _run_underwriting_rationale(case)
        nxt = route(case.facts, metrics, decision)
        _advance(case, nxt)
        if nxt == Stage.auto_decided:
            outcome = auto_outcome(nxt, decision)
            case.review = ReviewRecord(reviewer="auto", outcome=outcome, note="auto decision", at=_now())
            _audit(case, "system", "auto_decided", outcome.value if outcome else "")
            _advance(case, Stage.finalized)
        else:
            _audit(case, "system", "pending_review", "routed to an underwriter")
        span.set_attribute("case.stage", case.stage.value)
        return store.put(case)


async def apply_corrections(case_id: str, updates: dict, reviewer: str = "reviewer") -> UnderwritingCase:
    """Reviewer edits extracted values; re-validate and recompute metrics and the recommendation.

    Stage is untouched (the case stays in review); only the facts, metrics, and the
    model recommendation refresh, with an audit entry naming what changed.
    """
    store = get_store()
    case = store.get(case_id)
    if case is None:
        raise KeyError(case_id)
    with get_tracer().start_as_current_span("review.correct") as span:
        span.set_attribute("case.id", case_id)
        current = case.facts.model_dump()
        # The reviewer form is pre-populated with the current values and submits every field, so a
        # blanked field is an intentional clear, not an unfilled one. Normalize "" to None and apply
        # each provided field; `changed` still limits the audit and recompute to real differences.
        clean = {k: (None if isinstance(v, str) and not v.strip() else v) for k, v in updates.items()}
        changed = [k for k, v in clean.items() if current.get(k) != v]
        merged = {**current, **clean}
        case.facts = MortgageFacts(**merged)
        case.missing = R.validate(case.facts)
        decision, metrics = await assess(case.facts)
        case.decision, case.metrics = decision, metrics
        await _run_underwriting_rationale(case)  # re-explain after a reviewer edit
        _audit(case, reviewer, "reviewer_edit", ", ".join(changed) if changed else "no change")
        span.set_attribute("review.edited_fields", ",".join(changed))
        return store.put(case)


def decide(case_id: str, reviewer: str, outcome: Outcome, note: str = "") -> UnderwritingCase:
    store = get_store()
    case = store.get(case_id)
    if case is None:
        raise KeyError(case_id)
    if case.stage not in (Stage.pending_review, Stage.in_review):
        raise ValueError(f"case {case_id} is not awaiting review")
    with get_tracer().start_as_current_span("review.decide") as span:
        span.set_attribute("case.id", case_id)
        span.set_attribute("review.reviewer", reviewer)
        _advance(case, Stage.in_review)
        overrode = case.decision.recommendation is not None and outcome != case.decision.recommendation
        case.review = ReviewRecord(reviewer=reviewer, outcome=outcome, note=note, at=_now(), overrode=overrode)
        _audit(case, reviewer, "human_decision", f"{outcome.value}{' (override)' if overrode else ''}: {note}")
        _advance(case, Stage.finalized)
        span.set_attribute("review.outcome", outcome.value)
        return store.put(case)


def get_case(case_id: str) -> UnderwritingCase | None:
    return get_store().get(case_id)


def list_pending() -> list[UnderwritingCase]:
    return get_store().list_pending()


def list_finalized() -> list[UnderwritingCase]:
    """Finalized cases (manual and automatic), newest first, for the decision ledger."""
    return get_store().list_finalized()


def list_by_owner(owner: str) -> list[UnderwritingCase]:
    return get_store().list_by_owner(owner)


# ---- Entity-to-region reconciliation --------------------------------------------------
# The matching itself lives in `entities` (pure functions over blocks + facts) so the UI's
# HTTP client can reuse the exact same logic without a round-trip. Re-exported here for the
# existing callers; entity_pairs() wraps it for a case object.
from entities import entity_pairs_from  # noqa: E402  (kept beside its case-aware wrapper)


def entity_pairs(case: UnderwritingCase):
    """Entity-to-region pairs for a case's primary document."""
    return entity_pairs_from(case.ocr_blocks, case.facts.model_dump())
