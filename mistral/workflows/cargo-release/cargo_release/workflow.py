"""Cargo Release & Dangerous Goods Compliance workflow.

Demonstrates Mistral Workflows' key primitives through a realistic maritime
logistics use case: OCR extraction → cargo classification → dangerous-goods
sub-workflow → customs compliance → HITL review (when anomaly detected) →
release certificate generation.

Key SDK concepts shown:
  - InteractiveWorkflow + wait_for_input()  (AcceptDeclineConfirmation)
  - Child / sub-workflows via workflow.execute_workflow()
  - Activity retry policies and typed timeouts
  - Deterministic branching in workflow code
  - Structured LLM output via Pydantic models
"""

from datetime import timedelta

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai

from .activities import (
    classify_cargo_type,
    generate_release_certificate,
    ocr_extract_shipping_doc,
    validate_customs_compliance,
    validate_dangerous_goods,
)
from .models import (
    CargoClassification,
    CargoReleaseInput,
    CargoReleaseResult,
    DangerousGoodsParams,
    DangerousGoodsResult,
)

_DG_NAME = "dangerous-goods-validation"
_CARGO_NAME = "cargo-release-compliance"


@workflows.workflow.define(
    name=_DG_NAME,
)
class DangerousGoodsValidationWorkflow:
    """Sub-workflow that gives dangerous-goods validation its own retry surface.

    Keeping this as a separate @workflow.define class means:
    - It has its own execution timeline in AI Studio.
    - If the validate_dangerous_goods activity fails exhaustively, only this
      sub-workflow is marked failed, so the parent can decide how to proceed.
    - The parent triggers it with execute_workflow(), which is a durable call:
      if the parent worker restarts, Temporal replays up to the awaited child
      result without re-executing completed activities.
    """

    @workflows.workflow.entrypoint
    async def run(self, params: DangerousGoodsParams) -> DangerousGoodsResult:
        """Run dangerous-goods validation as an isolated, durable unit."""
        return await validate_dangerous_goods(
            extracted_text=params.extracted_text,
            shipment_id=params.shipment_id,
        )


@workflows.workflow.define(
    name=_CARGO_NAME,
    # Workflows default to a 1-hour execution timeout. Most cargo releases
    # complete in minutes, but HITL review can take hours: set a comfortable
    # upper bound so the workflow doesn't time out while waiting for a reviewer.
    execution_timeout=timedelta(hours=24),
)
class CargoReleaseWorkflow(workflows.InteractiveWorkflow):
    """Main cargo release workflow.

    Extends InteractiveWorkflow (not plain Workflow) because it may call
    wait_for_input(), which suspends the workflow in Le Chat and Mistral AI
    Studio, consuming zero compute while waiting for a human reviewer.

    All branching in this method is deterministic: it depends only on the
    structured outputs of activities (which are recorded in event history),
    never on real-time data like datetime.now() or random().
    """

    @workflows.workflow.entrypoint
    async def run(self, params: CargoReleaseInput) -> CargoReleaseResult:
        """Orchestrate the full cargo release pipeline.

        The workflow body contains zero non-deterministic code: every side
        effect (file read, LLM call, certificate generation) happens inside an
        activity so Temporal can safely replay this function on worker restarts.
        """

        # ── Step 1: OCR extraction ─────────────────────────────────────────
        # retry_policy_max_attempts=3 is set on the activity decorator because
        # real OCR APIs are flaky (transient network errors, PDF parsing hiccups).
        # If all attempts fail, the exception propagates here and the workflow
        # fails with a descriptive error visible in AI Studio.
        extracted_text: str = await ocr_extract_shipping_doc(params.document_uri)

        # ── Step 2: Cargo classification ──────────────────────────────────
        # mistralai_chat_parse returns a validated CargoClassification Pydantic
        # model: no raw JSON parsing in the workflow body.
        classification: CargoClassification = await classify_cargo_type(extracted_text)

        # ── Step 3 (conditional): Dangerous goods sub-workflow ────────────
        # Branching on classification.cargo_type is deterministic because the
        # value comes from a recorded activity result, not live data.
        dg_result: DangerousGoodsResult | None = None
        if classification.cargo_type == "dangerous_goods":
            # execute_workflow() is a durable call: the SDK persists the child
            # workflow ID in event history. If this worker restarts mid-flight,
            # Temporal resumes from this await without re-launching the child.
            dg_result = await workflows.workflow.execute_workflow(
                DangerousGoodsValidationWorkflow,
                params=DangerousGoodsParams(
                    extracted_text=extracted_text,
                    shipment_id=params.shipment_id,
                ),
                execution_timeout=timedelta(minutes=10),
            )

        # ── Step 4: Customs compliance validation ─────────────────────────
        compliance = await validate_customs_compliance(
            extracted_text=extracted_text,
            cargo_type=classification.cargo_type,
            shipment_id=params.shipment_id,
        )

        # ── Step 5 (conditional): HITL review ────────────────────────────
        # An anomaly is flagged when either:
        #   a) the dangerous-goods sub-workflow found something wrong, OR
        #   b) customs compliance failed.
        # This boolean is computed from recorded activity results → deterministic.
        anomaly_detected = (dg_result is not None and dg_result.has_anomaly) or (not compliance.passed)

        if anomaly_detected:
            # Build a human-readable summary for the reviewer.
            anomaly_summary_parts: list[str] = []
            if dg_result and dg_result.anomalies:
                anomaly_summary_parts.append("Dangerous goods anomalies: " + "; ".join(dg_result.anomalies))
            if not compliance.passed:
                anomaly_summary_parts.append("Customs compliance failures: " + ", ".join(compliance.failed_rules))
            anomaly_summary = " | ".join(anomaly_summary_parts)

            await workflows_mistralai.send_assistant_message(
                f"Shipment {params.shipment_id} requires review.\n\n"
                f"Cargo type: {classification.cargo_type}\n"
                f"Issues detected: {anomaly_summary}\n\n"
                "Please approve or block this shipment."
            )

            # wait_for_input() is the headline primitive.
            # The workflow is durably suspended here: no polling, no compute,
            # no heartbeats required. Temporal holds the state in its event log
            # and resumes the workflow the instant the reviewer submits a decision
            # in Le Chat or AI Studio.
            confirmation = await self.wait_for_input(
                workflows_mistralai.AcceptDeclineConfirmation(
                    description=(
                        f"Shipment {params.shipment_id}: {anomaly_summary}\n\n"
                        "Approve to release the cargo. Decline to block and escalate."
                    ),
                    accept_label="Approve release",
                    decline_label="Block & escalate",
                )
            )

            if not workflows_mistralai.is_accepted(confirmation):
                return CargoReleaseResult(
                    shipment_id=params.shipment_id,
                    status="blocked",
                    cargo_type=classification.cargo_type,
                    block_reason=f"Reviewer blocked shipment. Issues: {anomaly_summary}",
                )

        # ── Step 6: Generate release certificate ─────────────────────────
        # Reached either when no anomaly was detected, or after the reviewer
        # approved the flagged shipment.
        certificate = await generate_release_certificate(
            shipment_id=params.shipment_id,
            cargo_type=classification.cargo_type,
            compliance_notes=compliance.notes,
        )

        return CargoReleaseResult(
            shipment_id=params.shipment_id,
            status="released",
            cargo_type=classification.cargo_type,
            release_certificate=certificate,
        )
