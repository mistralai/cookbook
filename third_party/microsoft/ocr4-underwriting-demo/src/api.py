"""Backend service: the JSON + streaming API over the underwriting core.

    uv run uvicorn api:api --port 8001

This is the only process that touches the business logic (underwriting_service, docchat,
agents, OCR, the case store). The Gradio UI (app.py) is a separate process that talks to this
API over HTTP; it imports none of these modules. Run both with scripts/up.sh.

Streaming endpoints use Server-Sent Events: one `data:` line per pipeline stage, each a full
`UnderwritingCase` JSON, so the UI can animate the stages as they happen. A final
`data: {"done": true}` closes the stream.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import docchat as DC
import underwriting_service as S
from observability import setup_observability
from underwriting_schema import Outcome

# Initialize Azure Monitor at process start, not lazily inside the first request. On a cold
# process configure_azure_monitor finishes installing the tracer provider just after the first
# span is created, so that first request's spans (docchat.ocr and the OCR-4 POST) attach to the
# not-yet-installed provider and never export. Warming it here means the very first OCR call in
# a demo shows its end-to-end trace. Idempotent and self-swallowing, so import stays safe.
setup_observability()

api = FastAPI(title="Mortgage underwriting API")


# ---- request bodies ----
class MessageIn(BaseModel):
    text: str
    owner: str = "customer"


class DecisionIn(BaseModel):
    reviewer: str
    outcome: Outcome
    note: str = ""


class CorrectionsIn(BaseModel):
    updates: dict
    reviewer: str = "reviewer"


class DocAnswerIn(BaseModel):
    markdown: str
    question: str


def _sse(payload: dict) -> str:
    """One Server-Sent Events frame."""
    return f"data: {json.dumps(payload)}\n\n"


def _error_frame(exc: Exception) -> str:
    """A terminal SSE frame describing a mid-stream failure, so the client closes the stream
    cleanly instead of seeing a truncated connection. Rate limiting is flagged explicitly."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        msg = "The document service is rate-limited (429). Please wait a few seconds and try again."
    else:
        msg = f"The document service failed: {exc}"
    return _sse({"error": msg})


# ---- health + landing ----
@api.get("/health")
def health():
    return {"ok": True}


@api.get("/", response_class=HTMLResponse)
def landing():
    return (
        "<!doctype html><meta charset=utf-8><title>Underwriting API</title>"
        "<body style='font-family:system-ui;max-width:40rem;margin:4rem auto;color:#333'>"
        "<h1>Underwriting API</h1><p>This is the backend service. The user interface runs "
        "as a separate app. See <a href='/docs'>/docs</a> for the interface.</p></body>"
    )


# ---- cases: terminal (non-streaming) ----
@api.post("/cases")
async def create_case(file: UploadFile, owner: str = "customer") -> dict:
    data = await file.read()
    case = await S.create_case(data, file.filename or "document", owner=owner)
    return case.model_dump()


@api.post("/cases/{case_id}/documents")
async def add_document(case_id: str, file: UploadFile) -> dict:
    data = await file.read()
    case = await S.add_document(case_id, data, file.filename or "document")
    return case.model_dump()


@api.post("/cases/{case_id}/messages")
async def add_message(case_id: str, body: MessageIn) -> dict:
    case = await S.add_message(case_id, body.text, owner=body.owner)
    return case.model_dump()


@api.post("/cases/{case_id}/submit")
async def submit(case_id: str) -> dict:
    case = await S.submit(case_id)
    return case.model_dump()


@api.get("/cases/{case_id}")
def get_case(case_id: str) -> dict:
    case = S.get_case(case_id)
    return case.model_dump() if case else {"error": "not found"}


@api.get("/cases")
def list_pending() -> list[dict]:
    return [c.model_dump() for c in S.list_pending()]


@api.get("/ledger")
def ledger() -> list[dict]:
    """Finalized decisions (manual and automatic) for the reviewer's decision ledger."""
    return [c.model_dump() for c in S.list_finalized()]


@api.post("/cases/{case_id}/decide")
def decide(case_id: str, body: DecisionIn) -> dict:
    case = S.decide(case_id, body.reviewer, body.outcome, body.note)
    return case.model_dump()


@api.post("/cases/{case_id}/corrections")
async def corrections(case_id: str, body: CorrectionsIn) -> dict:
    case = await S.apply_corrections(case_id, body.updates, reviewer=body.reviewer)
    return case.model_dump()


@api.post("/cases/{case_id}/agent-reply")
async def agent_reply(case_id: str) -> dict:
    """The intake agent's chat reply for a case's current state."""
    case = S.get_case(case_id)
    if case is None:
        return {"text": ""}
    return {"text": await S.reply_for_agent(case)}


# ---- cases: streaming (one case frame per pipeline stage) ----
@api.post("/cases/stream")
async def create_case_stream(file: UploadFile, owner: str = "customer") -> StreamingResponse:
    data = await file.read()
    name = file.filename or "document"

    async def gen():
        try:
            async for case in S.create_case_stream(data, name, owner=owner):
                yield _sse(case.model_dump())
        except Exception as exc:  # keep the stream well-formed on OCR/model failures (e.g. 429)
            yield _error_frame(exc)
            return
        yield _sse({"done": True})

    return StreamingResponse(gen(), media_type="text/event-stream")


@api.post("/cases/{case_id}/documents/stream")
async def add_document_stream(case_id: str, file: UploadFile) -> StreamingResponse:
    data = await file.read()
    name = file.filename or "document"

    async def gen():
        try:
            async for case in S.create_case_stream_add(case_id, data, name):
                yield _sse(case.model_dump())
        except Exception as exc:  # keep the stream well-formed on OCR/model failures (e.g. 429)
            yield _error_frame(exc)
            return
        yield _sse({"done": True})

    return StreamingResponse(gen(), media_type="text/event-stream")


@api.post("/cases/{case_id}/submit/stream")
async def submit_stream(case_id: str) -> StreamingResponse:
    async def gen():
        try:
            async for case in S.submit_stream(case_id):
                if case is not None:
                    yield _sse(case.model_dump())
        except Exception as exc:  # keep the stream well-formed on OCR/model failures (e.g. 429)
            yield _error_frame(exc)
            return
        yield _sse({"done": True})

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---- document chat (stateless: no case store) ----
@api.post("/doc/analyze")
async def doc_analyze(file: UploadFile) -> dict:
    """OCR a standalone document: markdown, per-page layout blocks, and a short classification.
    Rendering the page images is the UI's job (it holds the upload); this returns only data."""
    data = await file.read()
    return DC.analyze(data, file.filename or "document")


@api.post("/doc/answer")
async def doc_answer(body: DocAnswerIn) -> dict:
    return {"text": await DC.answer(body.markdown, body.question)}
