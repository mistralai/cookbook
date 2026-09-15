"""Mortgage underwriting as an Agent Framework workflow with human-in-the-loop.

Flow (Ingest is OCR 4 perception; the other reasoning nodes are Medium 3.5 agents):

    doc/text ─▶ Ingest(OCR 4) ─▶ Extractor ─▶ Intake ⇄ human ─▶ Underwrite ─▶ Review ⇄ human ─▶ output

Ingest takes a single string: a document path (OCR 4 reads it and seeds the facts from its
annotation) or a free-text case description (OCR skipped). The Extractor, Intake, and Underwrite nodes are
the three Foundry portal agents (Medium 3.5). Intake and Review pause for a human via
`ctx.request_info(...)` + a paired `@response_handler`; the host (Agent Framework DevUI)
surfaces the request and resumes the run with the reply. The underwriting agent decides,
fed code-computed metrics so the numbers are correct, with a hard-fail safety net. Served
by devui_app.py; the Gradio app is untouched and stays OCR-4-only.
"""
# NOTE: no `from __future__ import annotations` here. The @response_handler signature
# validator inspects the real WorkflowContext[...] generic, which stringified annotations
# would hide.
import json
import os
import re
from dataclasses import dataclass

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler, response_handler
from typing_extensions import Never

import mortgage_rules as R
from agents import FIELDS_ANNOTATION_SCHEMA, parse_json_object
from foundry_agents import (
    EXTRACTOR_AGENT, INTAKE_AGENT, UNDERWRITING_AGENT, build_agent,
)
from intake import apply_message, facts_from_annotation
from observability import get_tracer
from ocr_client import ocr_with_annotation
from underwriting_schema import MortgageFacts

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- request payloads (carry their own state so resume needs no shared store) ----
@dataclass
class MissingInfoRequest:
    prompt: str
    field: str
    facts_json: str
    entities: list
    source: str


@dataclass
class ReviewRequest:
    summary: str
    packet_json: str


# ---- helpers ----
def _parse_rec(text: str) -> str:
    m = re.search(r"recommendation\s*[:\-]\s*(approve|decline|refer)", text, re.IGNORECASE)
    return m.group(1).lower() if m else "refer"


def _metrics_prompt(facts: MortgageFacts, metrics) -> str:
    return (
        f"Case metrics: LTV {metrics.ltv}%, DTI {metrics.dti}%, credit {facts.credit_score}, "
        f"loan {facts.loan_amount}. Borrower {facts.borrower_name or 'unknown'}. Apply policy and decide."
    )


def _review_summary(p: dict) -> str:
    return (
        f"Review needed for {p.get('borrower') or 'applicant'}: LTV {p.get('ltv')}%, "
        f"DTI {p.get('dti')}%, credit {p.get('credit')}, loan {p.get('loan')}. "
        f"The underwriting agent said '{p.get('rec')}'. Reply approve or decline (a note is fine)."
    )


def _final_text(p: dict, human: str | None) -> str:
    lines = [
        f"Source: {p.get('source') or 'n/a'}",
        f"Borrower: {p.get('borrower') or 'unknown'}",
        f"LTV {p.get('ltv')}%  DTI {p.get('dti')}%  credit {p.get('credit')}  loan {p.get('loan')}",
        f"Agent recommendation: {p.get('rec')}",
    ]
    if human:
        lines.append(f"Human decision: {human}")
    if p.get("hard"):
        lines.append("Hard-fail safety net applied (agent could not auto-approve).")
    lines += ["", (p.get("rationale") or "").strip()]
    ents = p.get("entities") or []
    if ents:
        lines.append("\nEntities (extractor agent): " + ", ".join(str(e) for e in ents[:12]))
    return "\n".join(lines)


# ---- executors ----
def _load_path(text: str):
    """If text names a readable file (absolute, relative, or under repo/inputs), read it."""
    cand = (text or "").strip().strip('"').strip("'")
    if not cand or len(cand) > 400 or "\n" in cand:
        return None, ""
    for p in (cand, os.path.join(_REPO_ROOT, cand), os.path.join(_REPO_ROOT, "inputs", cand)):
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                return fh.read(), os.path.basename(p)
    return None, ""


class IngestExecutor(Executor):
    """Perception node. The workflow input is a single string:

      - a document path (e.g. `inputs/loan_application_1003.pdf`) -> read from disk and sent
        to OCR 4, whose annotation seeds the facts, or
      - a free-text case description -> parsed directly, OCR skipped.

    A single str keeps the DevUI input box simple (DevUI renders a workflow's input form from
    the start handler's declared type; a structured type would render as a raw field form)."""

    def __init__(self) -> None:
        super().__init__(id="ingest")

    @handler
    async def run(self, text: str, ctx: WorkflowContext[dict]) -> None:
        data, name = _load_path(text)
        await self._ingest(data, name, None if data else text, ctx)

    async def _ingest(self, data, name, free_text, ctx) -> None:
        if data:
            with get_tracer().start_as_current_span("ocr.process") as span:
                span.set_attribute("document.name", name)
                result = ocr_with_annotation(data, name, annotation_schema=FIELDS_ANNOTATION_SCHEMA)
            md = result.get("markdown", "")
            ann = result.get("annotation") if isinstance(result.get("annotation"), dict) else {}
            facts = facts_from_annotation(ann)
            source = f"OCR 4 read {name} ({ann.get('document_type') or 'document'})"
        else:
            md = free_text or ""
            facts = await apply_message(MortgageFacts(), [], md)
            source = "text input (no document; OCR 4 skipped)"
        await ctx.send_message({"markdown": md, "facts": facts.model_dump_json(), "source": source})


class ExtractorExecutor(Executor):
    """Reasoning node 1: the extractor agent pulls named entities from the markdown."""

    def __init__(self, agent) -> None:
        super().__init__(id="extractor")
        self._agent = agent

    @handler
    async def run_extract(self, p: dict, ctx: WorkflowContext[dict]) -> None:
        entities: list = []
        md = p.get("markdown", "")
        if md.strip():
            try:
                resp = await self._agent.run(md)
                entities = parse_json_object(resp.text).get("entities", []) or []
            except Exception:
                entities = []
        await ctx.send_message({"facts": p["facts"], "entities": entities, "source": p.get("source", "")})


class IntakeExecutor(Executor):
    """Intake agent phrases the ask; deterministic parse fills facts; loops until complete."""

    def __init__(self, agent) -> None:
        super().__init__(id="intake")
        self._agent = agent

    @handler
    async def receive(self, p: dict, ctx: WorkflowContext[dict]) -> None:
        facts = MortgageFacts.model_validate_json(p["facts"])
        await self._advance(facts, p.get("entities", []), p.get("source", ""), ctx)

    @response_handler
    async def on_answer(self, req: MissingInfoRequest, answer: str, ctx: WorkflowContext[dict, str]) -> None:
        facts = MortgageFacts.model_validate_json(req.facts_json)
        # Apply against just the field we asked for, so apply_message's single-value
        # fallback can fill any type (including free-text fields with no keyword parser).
        target = [m for m in R.validate(facts) if m.name == req.field]
        facts = await apply_message(facts, target or R.validate(facts)[:1], answer)
        await self._advance(facts, req.entities, req.source, ctx)

    async def _advance(self, facts: MortgageFacts, entities: list, source: str, ctx) -> None:
        missing = R.validate(facts)
        if missing:
            target = missing[0]  # one field at a time: keeps the deterministic parse reliable
            prompt = await self._phrase(facts, target)
            await ctx.request_info(
                MissingInfoRequest(prompt=prompt, field=target.name,
                                   facts_json=facts.model_dump_json(), entities=entities, source=source),
                response_type=str,
            )
        else:
            await ctx.send_message({"facts": facts.model_dump_json(), "entities": entities, "source": source})

    async def _phrase(self, facts: MortgageFacts, target) -> str:
        label = R.REQUIRED_FIELDS.get(target.name, target.name)
        try:
            resp = await self._agent.run(
                f"Ask the mortgage applicant for {label} in one short, friendly sentence. "
                "Ask only for that one item."
            )
            return (resp.text or "").strip() or target.prompt
        except Exception:
            return target.prompt


class UnderwriteExecutor(Executor):
    """Underwriting agent decides against code-computed metrics, with a hard-fail safety net."""

    def __init__(self, agent) -> None:
        super().__init__(id="underwrite")
        self._agent = agent

    @handler
    async def run_underwrite(self, p: dict, ctx: WorkflowContext[dict]) -> None:
        facts = MortgageFacts.model_validate_json(p["facts"])
        metrics = R.compute_metrics(facts)
        try:
            resp = await self._agent.run(_metrics_prompt(facts, metrics))
            text = resp.text or ""
        except Exception as exc:  # keep the demo alive if the model call fails
            text = f"Recommendation: refer\nRisk score: 50\nRationale: underwriting agent unavailable ({exc})."
        rec = _parse_rec(text)
        hard = R.hard_fails(facts, metrics)
        if hard and rec == "approve":
            rec = "refer"  # safety net: never auto-approve a hard-fail case
        auto = (not hard) and R.within_auto_limits(facts, metrics) and rec in ("approve", "decline")
        await ctx.send_message({
            "facts": p["facts"], "entities": p.get("entities", []), "source": p.get("source", ""),
            "ltv": metrics.ltv, "dti": metrics.dti,
            "credit": facts.credit_score, "loan": facts.loan_amount,
            "borrower": facts.borrower_name, "rec": rec, "rationale": text,
            "auto": auto, "hard": bool(hard),
        })


class ReviewExecutor(Executor):
    """Auto-decide clean cases; otherwise pause for a human reviewer, then finalize."""

    def __init__(self) -> None:
        super().__init__(id="review")

    @handler
    async def gate(self, packet: dict, ctx: WorkflowContext[Never, str]) -> None:
        if packet.get("auto"):
            await ctx.yield_output(_final_text(packet, human=None))
        else:
            await ctx.request_info(
                ReviewRequest(summary=_review_summary(packet), packet_json=json.dumps(packet)),
                response_type=str,
            )

    @response_handler
    async def on_decision(self, req: ReviewRequest, decision: str, ctx: WorkflowContext[Never, str]) -> None:
        packet = json.loads(req.packet_json)
        await ctx.yield_output(_final_text(packet, human=decision))


def build_underwriting_workflow():
    ingest = IngestExecutor()
    extractor = ExtractorExecutor(build_agent(EXTRACTOR_AGENT))
    intake = IntakeExecutor(build_agent(INTAKE_AGENT))
    underwrite = UnderwriteExecutor(build_agent(UNDERWRITING_AGENT))
    review = ReviewExecutor()
    return (
        WorkflowBuilder(start_executor=ingest)
        .add_edge(ingest, extractor)
        .add_edge(extractor, intake)
        .add_edge(intake, underwrite)
        .add_edge(underwrite, review)
        .build()
    )
