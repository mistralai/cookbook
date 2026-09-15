"""Schema and stage model for the mortgage underwriting extension.

One `Stage` enum is the source of truth for the progress panels, the case record,
and the trace. `UnderwritingCase` is the durable record that both Gradio views and
the API read and write.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Stage(str, Enum):
    received = "received"
    reading = "reading"          # OCR 4
    extracting = "extracting"    # field extraction
    collecting = "collecting"    # missing-value conversation
    ready = "ready"              # complete, awaiting submit
    underwriting = "underwriting"
    auto_decided = "auto_decided"
    pending_review = "pending_review"
    in_review = "in_review"
    finalized = "finalized"


# Forward transitions only. Guards reject out-of-order moves.
_ALLOWED: dict[Stage, set[Stage]] = {
    Stage.received: {Stage.reading},
    Stage.reading: {Stage.extracting},
    Stage.extracting: {Stage.collecting, Stage.ready},
    Stage.collecting: {Stage.collecting, Stage.ready},
    Stage.ready: {Stage.underwriting},
    Stage.underwriting: {Stage.auto_decided, Stage.pending_review},
    Stage.auto_decided: {Stage.finalized},
    Stage.pending_review: {Stage.in_review},
    Stage.in_review: {Stage.pending_review, Stage.finalized},
    Stage.finalized: set(),
}


def can_transition(current: Stage, nxt: Stage) -> bool:
    return nxt in _ALLOWED.get(current, set())


class Outcome(str, Enum):
    approve = "approve"
    decline = "decline"
    refer = "refer"


class MissingField(BaseModel):
    name: str
    prompt: str  # plain-language question the intake agent asks
    reason: str  # missing | unreadable | out_of_range | low_confidence


class MortgageFacts(BaseModel):
    borrower_name: Optional[str] = None
    property_address: Optional[str] = None
    loan_purpose: Optional[str] = None  # purchase | refinance
    annual_income: Optional[float] = None
    loan_amount: Optional[float] = None
    property_value: Optional[float] = None
    credit_score: Optional[int] = None
    monthly_debts: Optional[float] = None
    employment_years: Optional[float] = None


class Metrics(BaseModel):
    ltv: Optional[float] = None
    dti: Optional[float] = None
    estimated_monthly_payment: Optional[float] = None


class Decision(BaseModel):
    recommendation: Optional[Outcome] = None
    risk_score: Optional[int] = None  # 0 low risk .. 100 high risk
    confidence: Optional[float] = None
    rationale: str = ""
    factors: list[str] = Field(default_factory=list)
    hard_fails: list[str] = Field(default_factory=list)


class ReviewRecord(BaseModel):
    reviewer: Optional[str] = None
    outcome: Optional[Outcome] = None
    note: str = ""
    at: Optional[str] = None
    overrode: bool = False  # reviewer differed from the model recommendation


class AuditEntry(BaseModel):
    at: str
    actor: str  # system | <customer id> | <reviewer id>
    event: str
    detail: str = ""


class Document(BaseModel):
    """One uploaded file in the application package, classified into a checklist slot."""
    id: str
    type: str = "other"            # canonical slot key (see documents.REQUIRED_DOCS)
    type_label: str = "Other document"
    name: str = ""
    path: Optional[str] = None     # persisted original, for the overlay
    doc_type_raw: str = ""         # OCR 4's own document_type string
    summary: str = ""              # OCR 4's one-line summary
    markdown: str = ""
    ocr_blocks: list[dict] = Field(default_factory=list)
    ocr_width: Optional[int] = None
    ocr_height: Optional[int] = None
    entities: list[str] = Field(default_factory=list)  # named entities the extractor pulled from THIS doc
    status: str = "received"       # received | needs_review
    added_at: Optional[str] = None


class UnderwritingCase(BaseModel):
    case_id: str
    doc_id: Optional[str] = None
    owner: Optional[str] = None  # customer identity, for per-user scoping
    stage: Stage = Stage.received
    facts: MortgageFacts = Field(default_factory=MortgageFacts)
    metrics: Metrics = Field(default_factory=Metrics)
    # Snapshot of the extracted values as they stood at underwriting time, before any reviewer
    # correction. Written once at submit; never overwritten. The ledger diffs this ("before")
    # against the live facts/metrics ("after"). None for cases decided before this was added.
    submitted_facts: Optional[MortgageFacts] = None
    submitted_metrics: Optional[Metrics] = None
    missing: list[MissingField] = Field(default_factory=list)
    decision: Decision = Field(default_factory=Decision)
    review: ReviewRecord = Field(default_factory=ReviewRecord)
    audit: list[AuditEntry] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)  # the application package
    entities: list[str] = Field(default_factory=list)  # named entities from the extractor agent
    agent_usage: dict = Field(default_factory=dict)    # per-agent token totals (extractor/intake/underwriter)
    markdown: str = ""                    # primary (most recent) document, for backward compat
    document_path: Optional[str] = None  # persisted original file, for the reviewer overlay
    ocr_blocks: list[dict] = Field(default_factory=list)  # first-page layout boxes for highlights
    ocr_width: Optional[int] = None
    ocr_height: Optional[int] = None
    etag: Optional[str] = None  # optimistic concurrency token (set by the store)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
