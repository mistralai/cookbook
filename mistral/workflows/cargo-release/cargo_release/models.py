"""Pydantic models for the cargo release workflow.

Every LLM call in this workflow uses structured output via these schemas.
All data that crosses workflow/activity boundaries must be JSON-serialisable
and typed: the SDK enforces this at registration time.
"""

from typing import Literal

from pydantic import BaseModel, Field


class CargoReleaseInput(BaseModel):
    """Single entrypoint model: fields become top-level JSON input keys."""

    document_uri: str = Field(description="Local file path or URL of the shipping document to process")
    shipment_id: str = Field(description="Unique identifier for this shipment")


class CargoClassification(BaseModel):
    """Structured output from the cargo-type classifier LLM call."""

    cargo_type: Literal["general", "dangerous_goods", "perishable"]
    reasoning: str = Field(description="Why the LLM chose this classification")


class DangerousGoodsParams(BaseModel):
    """Input to the DangerousGoodsValidationWorkflow sub-workflow."""

    extracted_text: str
    shipment_id: str


class DangerousGoodsResult(BaseModel):
    """Structured output from the dangerous-goods validation sub-workflow."""

    un_number: str | None = Field(default=None, description="UN number if present, e.g. 'UN 1203'")
    hazard_class: str | None = Field(
        default=None,
        description="IMO/ADR hazard class, e.g. 'Class 3 - Flammable liquids'",
    )
    anomalies: list[str] = Field(
        default_factory=list,
        description="List of compliance anomalies found (e.g. missing emergency contact)",
    )
    has_anomaly: bool = Field(description="True if any anomaly was detected that requires human review")


class ComplianceCheck(BaseModel):
    """Structured output from the customs compliance validator LLM call."""

    passed: bool = Field(description="True if all customs rules are satisfied")
    failed_rules: list[str] = Field(
        default_factory=list,
        description="Names of rules that failed, empty when passed=True",
    )
    notes: str = Field(description="Short free-text summary of the compliance review")


class CargoReleaseResult(BaseModel):
    """Final output returned by the top-level workflow."""

    shipment_id: str
    status: Literal["released", "blocked"]
    cargo_type: str
    release_certificate: str | None = Field(
        default=None,
        description="LLM-generated release certificate text; None when blocked",
    )
    block_reason: str | None = Field(
        default=None,
        description="Human-readable reason for blocking; None when released",
    )
