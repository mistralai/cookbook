"""Pydantic models for the insurance claims triage workflow.

Defines input/output shapes for the workflow entrypoint and every
structured-output LLM call.  All intermediate types live here so
activity signatures stay clean and the JSON schemas are easy to read
in AI Studio.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Workflow input
# ---------------------------------------------------------------------------


class ClaimInput(BaseModel):
    """Top-level input submitted when a new claim arrives.

    The ``photos`` field is a list of URIs understood by the photo
    analysis activity (``file://`` paths in local dev, ``s3://`` / HTTPS
    in production).
    """

    claim_id: str
    claimant_name: str
    description: str
    photos: list[str] = Field(default_factory=list, min_length=1)


# ---------------------------------------------------------------------------
# Vision activity output
# ---------------------------------------------------------------------------


class DamageLabel(str, Enum):
    """Coarse damage categories returned by the photo analysis stub."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    TOTALED = "totaled"


class PhotoAnalysisResult(BaseModel):
    """Structured result from analysing a single claim photo via Pixtral."""

    photo_uri: str
    damage_label: DamageLabel
    damage_description: str
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Consistency-check activity output
# ---------------------------------------------------------------------------


class ConsistencyCheck(BaseModel):
    """Output of cross-referencing photo findings with the claimant description."""

    consistent: bool
    discrepancies: list[str] = Field(default_factory=list)
    summary: str


# ---------------------------------------------------------------------------
# Severity classification (pure Python, no LLM)
# ---------------------------------------------------------------------------


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SeverityClassification(BaseModel):
    """Result of the deterministic branching step.

    Routing is a hard rule, not an LLM call — see workflow.py for the
    reasoning behind this design decision.
    """

    severity: SeverityLevel
    routing_queue: str
    rationale: str


# ---------------------------------------------------------------------------
# Fraud-risk activity output
# ---------------------------------------------------------------------------


class FraudRiskAssessment(BaseModel):
    """Structured fraud-risk score from a Pixtral + text reasoning call."""

    fraud_risk_score: float = Field(ge=0.0, le=1.0)
    fraud_indicators: list[str] = Field(default_factory=list)
    reasoning: str


# ---------------------------------------------------------------------------
# Triage report (final structured output)
# ---------------------------------------------------------------------------


class TriageReport(BaseModel):
    """The complete triage report returned by the workflow."""

    claim_id: str
    claimant_name: str
    severity: SeverityLevel
    routing_queue: str
    fraud_risk_score: float
    fraud_indicators: list[str]
    photo_findings: list[dict[str, Any]]
    consistency_check: dict[str, Any]
    reasoning: str
    cited_evidence: list[str]


# ---------------------------------------------------------------------------
# Final workflow output
# ---------------------------------------------------------------------------


class ClaimTriageOutput(BaseModel):
    """Everything the workflow returns to its caller."""

    claim_id: str
    routing_decision: str
    severity: SeverityLevel
    fraud_risk_score: float
    report: TriageReport
