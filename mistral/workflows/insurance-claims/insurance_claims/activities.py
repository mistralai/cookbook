"""Activity functions for the insurance claims triage workflow.

Each activity is a discrete, idempotent unit of work:

- ``analyze_photo``        — Mistral Small vision call for photo damage assessment
- ``check_consistency``    — LLM cross-reference of photo findings vs. description
- ``score_fraud_risk``     — LLM fraud-risk scoring with structured output
- ``generate_triage_report`` — LLM final report with cited evidence

All LLM activities use ``mistralai_chat_parse`` (structured output via
Pydantic) so the response schema is always validated before the workflow
sees it.

Why are these activities and not inline workflow code?
  Activities are the correct boundary for I/O: they get automatic retries,
  distributed scheduling, and per-call observability in AI Studio.  Putting
  an API call directly in workflow code would break deterministic replay.
"""

from __future__ import annotations

import base64
import mimetypes
from datetime import timedelta
from pathlib import Path
from urllib.parse import unquote
from urllib.request import url2pathname

from mistralai import workflows
from mistralai.client import models as mistralai_models
from mistralai.workflows.plugins.mistralai.activities import (
    chat_parse_to_model,
)

from .models import (
    ConsistencyCheck,
    FraudRiskAssessment,
    PhotoAnalysisResult,
    TriageReport,
)

_MODEL = "mistral-small-latest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _confined_local_path(raw_path: str) -> Path:
    """Resolve a caller-supplied local path, confined to the project directory.

    ``photo_uri`` comes from workflow input, which is untrusted. Reading an
    arbitrary path (e.g. ``/etc/passwd`` or ``file:///…``) and shipping its
    bytes to the vision API would let a caller exfiltrate host files. We
    therefore resolve the path under the project working directory and reject
    anything that escapes it.
    """
    root = Path.cwd()
    candidate = Path(raw_path)
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Refusing to read '{candidate}': outside the project directory '{root}'.")
    return candidate


def _resolve_image_uri(photo_uri: str) -> str:
    """Convert a ``file://`` URI to a base64 data URI for Pixtral.

    HTTP(S) URLs and existing data URIs are returned as-is. ``file://`` URIs and
    plain paths are read from disk (confined to the project directory, see
    :func:`_confined_local_path`) and encoded as ``data:image/<ext>;base64,...``
    so Pixtral can consume them without network access to the filesystem.
    """
    if photo_uri.startswith(("http://", "https://", "data:")):
        return photo_uri

    if photo_uri.startswith("file://"):
        raw_path = url2pathname(unquote(photo_uri[7:]))
    else:
        raw_path = photo_uri

    path = _confined_local_path(raw_path)
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


# ---------------------------------------------------------------------------
# Vision analysis — Mistral Small vision call
# ---------------------------------------------------------------------------

_VISION_PROMPT = (
    "You are an expert vehicle damage assessor for an insurance company. "
    "Analyse the photo and return a structured damage assessment.\n\n"
    "Classify the damage into one of these categories:\n"
    "- none: No visible damage\n"
    "- minor: Surface scratches, small paint chips\n"
    "- moderate: Dents, cracked lights, cosmetic panel damage — vehicle is driveable\n"
    "- severe: Structural damage to panels, frame, or mechanical components\n"
    "- totaled: Vehicle is a total loss\n\n"
    "Return JSON with fields:\n"
    "- damage_label: one of none/minor/moderate/severe/totaled\n"
    "- damage_description: brief description of the damage observed\n"
    "- confidence: float 0-1 indicating your confidence in the assessment"
)


@workflows.activity(
    # Vision API calls can be slow; aggressive retries are intentional.
    # 3 attempts with 2x backoff; each attempt must finish within 60 s.
    retry_policy_max_attempts=3,
    retry_policy_backoff_coefficient=2.0,
    start_to_close_timeout=timedelta(seconds=60),
)
async def analyze_photo(photo_uri: str) -> PhotoAnalysisResult:
    """Analyse a single claim photo and return a structured damage assessment.

    Calls Mistral Small with the image URI and a structured-output prompt.
    The response is validated against ``PhotoAnalysisResult`` via
    ``chat_parse_to_model``.
    """
    image_url = _resolve_image_uri(photo_uri)
    request = mistralai_models.ChatCompletionRequest(
        model=_MODEL,
        messages=[
            mistralai_models.UserMessage(
                content=[
                    mistralai_models.ImageURLChunk(image_url=image_url),
                    mistralai_models.TextChunk(text=_VISION_PROMPT),
                ]
            ),
        ],
    )
    result = await chat_parse_to_model(PhotoAnalysisResult, request)
    # Inject the photo_uri since the model won't know it from the image alone.
    result.photo_uri = photo_uri
    return result


# ---------------------------------------------------------------------------
# Consistency check — LLM with structured output
# ---------------------------------------------------------------------------


@workflows.activity(
    retry_policy_max_attempts=3,
    retry_policy_backoff_coefficient=2.0,
    start_to_close_timeout=timedelta(seconds=60),
)
async def check_consistency(
    photo_findings_json: str,
    claimant_description: str,
) -> dict:
    """Cross-reference aggregated photo findings against the claimant's description.

    Returns a dict (not a Pydantic model) so Temporal's data converter round-trips
    it as JSON without class-identity issues across the workflow sandbox boundary.
    The workflow reconstructs ``ConsistencyCheck`` via ``model_validate``.
    """
    prompt = (
        "You are an insurance claims analyst. "
        "Compare the following photo analysis findings with the claimant's written description. "
        "Identify any inconsistencies — e.g. the claimant describes minor scratches but photos "
        "show total loss.\n\n"
        f"Photo findings (JSON):\n{photo_findings_json}\n\n"
        f"Claimant description:\n{claimant_description}\n\n"
        "Return a JSON object with fields: consistent (bool), discrepancies (list[str]), summary (str)."
    )
    request = mistralai_models.ChatCompletionRequest(
        model=_MODEL,
        messages=[mistralai_models.UserMessage(content=prompt)],
    )
    result = await chat_parse_to_model(ConsistencyCheck, request)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Fraud-risk scoring — LLM with structured output
# ---------------------------------------------------------------------------


@workflows.activity(
    retry_policy_max_attempts=3,
    retry_policy_backoff_coefficient=2.0,
    start_to_close_timeout=timedelta(seconds=60),
)
async def score_fraud_risk(
    photo_findings_json: str,
    claimant_description: str,
    consistency_json: str,
    severity: str,
) -> dict:
    """Produce a calibrated fraud-risk score with cited indicators.

    Returns a dict so Temporal's data converter round-trips it as JSON
    without class-identity issues across the workflow sandbox boundary.
    Uses Mistral Small for structured reasoning.
    """
    prompt = (
        "You are a fraud detection specialist at an insurance company. "
        "Analyse the claim details below and output a fraud risk assessment.\n\n"
        f"Photo findings:\n{photo_findings_json}\n\n"
        f"Claimant description:\n{claimant_description}\n\n"
        f"Consistency check:\n{consistency_json}\n\n"
        f"Assessed severity: {severity}\n\n"
        "Return JSON with: fraud_risk_score (float 0-1), "
        "fraud_indicators (list[str]), reasoning (str)."
    )
    request = mistralai_models.ChatCompletionRequest(
        model=_MODEL,
        messages=[mistralai_models.UserMessage(content=prompt)],
    )
    result = await chat_parse_to_model(FraudRiskAssessment, request)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Triage report generation — LLM with structured output
# ---------------------------------------------------------------------------


@workflows.activity(
    retry_policy_max_attempts=3,
    retry_policy_backoff_coefficient=2.0,
    start_to_close_timeout=timedelta(seconds=90),
)
async def generate_triage_report(
    claim_id: str,
    claimant_name: str,
    photo_findings_json: str,
    consistency_json: str,
    severity: str,
    routing_queue: str,
    fraud_risk_score: float,
    fraud_indicators_json: str,
) -> dict:
    """Write the final triage report with reasoning and cited evidence.

    Returns a dict so Temporal's data converter round-trips it as JSON
    without class-identity issues across the workflow sandbox boundary.
    The workflow reconstructs ``TriageReport`` via ``model_validate``.
    """
    prompt = (
        "You are writing the official triage report for an insurance claim. "
        "Synthesise the information below into a final report.\n\n"
        f"Claim ID: {claim_id}\n"
        f"Claimant: {claimant_name}\n"
        f"Photo findings: {photo_findings_json}\n"
        f"Consistency check: {consistency_json}\n"
        f"Severity: {severity}\n"
        f"Routing queue: {routing_queue}\n"
        f"Fraud risk score: {fraud_risk_score:.2f}\n"
        f"Fraud indicators: {fraud_indicators_json}\n\n"
        "Return JSON matching the TriageReport schema: "
        "claim_id, claimant_name, severity, routing_queue, fraud_risk_score, "
        "fraud_indicators (list[str]), photo_findings (list[dict]), "
        "consistency_check (dict), reasoning (str), cited_evidence (list[str])."
    )
    request = mistralai_models.ChatCompletionRequest(
        model=_MODEL,
        messages=[mistralai_models.UserMessage(content=prompt)],
    )
    result = await chat_parse_to_model(TriageReport, request)
    return result.model_dump()
