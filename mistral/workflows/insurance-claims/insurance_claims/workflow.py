"""Insurance Claims Triage workflow definition.

This workflow demonstrates five Mistral Workflows primitives on a realistic
insurance-industry use case:

  1. Parallel activities   — all claim photos are analysed concurrently via
                             ``execute_activities_in_parallel``, not sequentially.
  2. Retry policies        — the flaky vision activity retries up to 3×.
  3. Deterministic branch  — severity routing is a hard Python if/elif/else
                             (not an LLM call) so every decision is auditable.
  4. Structured output     — every LLM call returns a typed Pydantic model.
  5. Observability         — Studio shows each parallel photo task, the
                             retry timeline, and the branch taken.

Workflow input shape (single Pydantic model → top-level JSON keys):
  {
    "claim_id": "CLM-001",
    "claimant_name": "Jane Smith",
    "description": "...",
    "photos": ["apps/worker/src/worker/workflows/insurance_claims/sample_data/photos/claim_low_scratch_door.jpg", ...]
  }
"""

from __future__ import annotations

import json

from mistralai import workflows
from mistralai.workflows import workflow

# Activities are imported via imports_passed_through so the Temporal sandbox
# does not try to re-import mistralai.client (which imports httpx, which
# triggers a sandbox restriction on urllib.request.Request.__mro_entries__).
# The activity functions themselves are safe to call from workflow code —
# the decorator intercepts the call and dispatches it via Temporal's task
# queue rather than executing the function body in the workflow thread.
with workflow.unsafe.imports_passed_through():
    from .activities import (
        analyze_photo,
        check_consistency,
        generate_triage_report,
        score_fraud_risk,
    )

from .models import (  # pydantic models are sandbox-safe; no imports_passed_through needed
    ClaimInput,
    ClaimTriageOutput,
    DamageLabel,
    PhotoAnalysisResult,
    SeverityLevel,
    TriageReport,
)


def _aggregate_severity(results: list[PhotoAnalysisResult]) -> SeverityLevel:
    """Derive the worst-case severity from all photo results.

    This is a pure function — no randomness, no I/O — so it is safe to call
    directly inside workflow code without wrapping it in an activity.
    """
    labels = [r.damage_label for r in results]
    if DamageLabel.TOTALED in labels or DamageLabel.SEVERE in labels:
        return SeverityLevel.HIGH
    if DamageLabel.MODERATE in labels:
        return SeverityLevel.MEDIUM
    return SeverityLevel.LOW


def _route(severity: SeverityLevel) -> str:
    """Map severity to a downstream queue name.

    This branching is deterministic (no LLM call) because:
      - Routing decisions must be auditable — regulators need a hard rule,
        not a probabilistic model output.
      - An LLM could hallucinate an unknown queue name, breaking downstream
        systems silently.
      - The classification step already used an LLM; routing is purely
        mechanical.  Keeping it in Python makes the decision inspectable in
        the Studio timeline without needing to decode a model response.
    """
    if severity == SeverityLevel.HIGH:
        return "full-investigation"
    if severity == SeverityLevel.MEDIUM:
        return "adjuster-review"
    return "fast-track"


@workflows.workflow.define(
    name="insurance-claims-triage",
)
class InsuranceClaimsTriageWorkflow:
    """Orchestrates the full insurance claims triage pipeline.

    The workflow body contains zero non-deterministic code — all I/O,
    API calls, and side effects happen inside activities.  This satisfies
    Temporal's replay requirement: if a worker restarts mid-execution,
    Temporal replays this code and every activity result is replayed from
    the event history rather than re-executed.
    """

    @workflows.workflow.entrypoint
    async def run(self, claim: ClaimInput) -> ClaimTriageOutput:
        """Execute the full triage pipeline for a single claim.

        Steps:
          1. Parallel photo analysis (all photos concurrently).
          2. Consistency check (photos vs. text description).
          3. Severity classification (deterministic Python branch).
          4. Fraud risk scoring.
          5. Triage report generation.
          6. Return routing decision + full report.
        """

        # ------------------------------------------------------------------
        # Step 1: Parallel vision analysis
        #
        # execute_activities_in_parallel beats asyncio.gather here because:
        #   • Each photo activity gets independent retries (automatic, per
        #     the retry_policy on analyze_photo — 3 attempts, 2× backoff).
        #   • Temporal tracks each parallel task separately in the event
        #     history, so Studio shows them as distinct timeline entries.
        #   • If the worker crashes mid-batch, only the incomplete tasks are
        #     re-dispatched — not the whole batch.
        # asyncio.gather would give you none of these guarantees.
        #
        # Items are passed as positional values — each str URI maps directly
        # to analyze_photo's first (and only) parameter, photo_uri: str.
        # ------------------------------------------------------------------
        # execute_activities_in_parallel returns list[PhotoAnalysisResult] already
        # validated via TypeAdapter — no second model_validate needed.
        photo_results: list[PhotoAnalysisResult] = await workflows.execute_activities_in_parallel(
            activity=analyze_photo,
            items=claim.photos,  # list[str] — each element is one photo_uri
            max_concurrent_scheduled_tasks=10,  # cap for API rate limits
        )

        # ------------------------------------------------------------------
        # Step 2: Consistency check
        #
        # check_consistency returns a dict (not a Pydantic model) so the
        # result survives the Temporal sandbox boundary without class-identity
        # issues.  We pass consistency_json directly to the next activities.
        # ------------------------------------------------------------------
        findings_json = json.dumps([r.model_dump() for r in photo_results], default=str)
        consistency_dict: dict = await check_consistency(
            photo_findings_json=findings_json,
            claimant_description=claim.description,
        )
        consistency_json = json.dumps(consistency_dict, default=str)

        # ------------------------------------------------------------------
        # Step 3: Deterministic severity classification
        #
        # This branch is NOT an LLM call.  The rationale is in _route() above.
        # Studio will show the value of `severity` in the workflow timeline
        # without any model inference — pure, auditable Python.
        # ------------------------------------------------------------------
        severity = _aggregate_severity(photo_results)
        routing_queue = _route(severity)

        # ------------------------------------------------------------------
        # Step 4: Fraud risk scoring
        # ------------------------------------------------------------------
        fraud_dict: dict = await score_fraud_risk(
            photo_findings_json=findings_json,
            claimant_description=claim.description,
            consistency_json=consistency_json,
            severity=severity.value,
        )

        # ------------------------------------------------------------------
        # Step 5: Generate the final triage report
        # ------------------------------------------------------------------
        report_dict: dict = await generate_triage_report(
            claim_id=claim.claim_id,
            claimant_name=claim.claimant_name,
            photo_findings_json=findings_json,
            consistency_json=consistency_json,
            severity=severity.value,
            routing_queue=routing_queue,
            fraud_risk_score=fraud_dict["fraud_risk_score"],
            fraud_indicators_json=json.dumps(fraud_dict.get("fraud_indicators", [])),
        )

        # ------------------------------------------------------------------
        # Step 6: Return the routing decision and full report
        # ------------------------------------------------------------------
        return ClaimTriageOutput(
            claim_id=claim.claim_id,
            routing_decision=routing_queue,
            severity=severity,
            fraud_risk_score=fraud_dict["fraud_risk_score"],
            report=TriageReport.model_validate(report_dict),
        )
