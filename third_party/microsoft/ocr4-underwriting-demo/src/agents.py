"""Reasoning agent (Mistral Medium 3.5 on Azure Foundry).

Design (Option B): OCR 4 itself classifies + summarizes the document via its
`document_annotation` feature, so the LLM layer is a single Extractor agent that
pulls deep structured fields + entities from the OCR markdown.

Auth is key-based. Note: the Foundry Mistral deployment currently mis-serializes
function/tool calls, so we call agents directly (no tool-calling) — which is all
this design needs.
"""
from __future__ import annotations

import json
import re
from typing import Any

from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, Field, field_validator

from config import load_settings


# ---- Structured output schema (what the pipeline ultimately returns) ----
class DocumentAnalysis(BaseModel):
    doc_type: str = Field(default="unknown", description="Document type (from OCR4 annotation).")
    language: str = Field(default="unknown", description="Primary language (from OCR4 annotation).")
    summary: str = Field(default="", description="Summary (from OCR4 annotation).")
    entities: list[str] = Field(default_factory=list, description="Named entities (from Extractor agent).")
    fields: dict[str, Any] = Field(default_factory=dict, description="Key/value fields (from Extractor agent).")

    @field_validator("entities", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            v = [v]
        return [x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in v]


# OCR 4 custom annotation schema: read + classify + extract the mortgage fields in
# ONE OCR call, so no separate reasoning model is needed for document extraction.
# All fields are strings (blanks come back as ""); numeric coercion happens downstream.
FIELDS_ANNOTATION_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "title": "MortgageDocAnnotation",
    "properties": {
        "document_type": {"type": "string", "title": "Document_Type"},
        "language": {"type": "string", "title": "Language"},
        "summary": {"type": "string", "title": "Summary"},
        "borrower_name": {"type": "string", "title": "Borrower_Name"},
        "property_address": {"type": "string", "title": "Property_Address"},
        "loan_purpose": {"type": "string", "title": "Loan_Purpose"},
        "annual_income": {"type": "string", "title": "Annual_Income"},
        "loan_amount": {"type": "string", "title": "Loan_Amount"},
        "property_value": {"type": "string", "title": "Property_Value"},
        "credit_score": {"type": "string", "title": "Credit_Score"},
        "monthly_debts": {"type": "string", "title": "Monthly_Debts"},
        "employment_years": {"type": "string", "title": "Employment_Years"},
    },
    "required": [
        "document_type", "language", "summary", "borrower_name", "property_address",
        "loan_purpose", "annual_income", "loan_amount", "property_value",
        "credit_score", "monthly_debts", "employment_years",
    ],
}


def parse_json_object(text: str) -> dict:
    """Tolerant parse: strip code fences, grab the outermost JSON object."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def build_chat_client() -> OpenAIChatCompletionClient:
    """OpenAI-compatible client pointed at the Foundry inference endpoint."""
    s = load_settings()
    return OpenAIChatCompletionClient(model=s.chat_deployment, api_key=s.ai_key, base_url=s.chat_base_url)


def build_extractor(client: OpenAIChatCompletionClient) -> Agent:
    return Agent(
        client=client,
        name="extractor",
        description="Extracts structured key/value fields and named entities from a document.",
        instructions=(
            "You extract structured data from document OCR markdown. Respond with JSON only: "
            '{"fields": {<key>: <value>, ...}, "entities": [<entity>, ...]}. '
            "Fields should capture dates, totals/amounts, identifiers, parties, line items, and other "
            "salient key/value pairs. Values may be strings or nested objects/arrays. "
            "If nothing applies, use empty objects/arrays."
        ),
    )
