"""HTTP client for the backend API, shaped like the old in-process service.

The Gradio UI imports this as `S` in place of `underwriting_service`, so its handlers keep the
same calls (`S.create_case_stream(...)`, `S.get_case(...)`, `S.decide(...)`), but each now goes
over HTTP to the backend (api.py) instead of running in the UI process. Responses are rebuilt
into real `UnderwritingCase` objects so the handlers see the same types as before.

Streaming methods are async generators that consume Server-Sent Events and yield a case per
stage, matching the `async for case in S.<stream>(...)` shape the handlers already use.

`entity_pairs_from` / `entity_pairs` are pure and stay local (imported from `entities`): they
need no case store and no network. Document chat (`analyze` / `answer`) also routes through the
backend, so the UI process imports no OCR or agent code.

Base URL comes from BACKEND_URL (default http://127.0.0.1:8001).
"""
from __future__ import annotations

import json
import os

import httpx
import requests

from entities import entity_pairs_from  # pure, no network
from underwriting_schema import Outcome, UnderwritingCase

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8001").rstrip("/")
# Longer than the backend's worst-case OCR budget (300s plus retries), so a large multi-page
# document does not time out on the client while the backend is still working.
_TIMEOUT = float(os.environ.get("BACKEND_TIMEOUT", "360"))


class BackendError(RuntimeError):
    """The backend could not be reached or returned an error. Handlers already wrap service
    calls in try/except and surface the message, so a clear text here becomes a clear UI note
    rather than a stack trace."""


def _url(path: str) -> str:
    return f"{BACKEND_URL}{path}"


def _case(payload: dict) -> UnderwritingCase:
    return UnderwritingCase.model_validate(payload)


def _get(path: str) -> dict:
    try:
        r = requests.get(_url(path), timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        raise BackendError(f"backend unreachable: {exc}") from exc


def _post(path: str, *, json_body: dict | None = None, files=None, data=None) -> dict:
    try:
        r = requests.post(_url(path), json=json_body, files=files, data=data, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        raise BackendError(f"backend unreachable: {exc}") from exc


async def _stream_cases(path: str, *, files=None, data=None):
    """Consume an SSE stream of case frames and yield rebuilt UnderwritingCase objects.
    Stops at the terminal {"done": true} frame."""
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", _url(path), files=files, data=data) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = json.loads(line[len("data:"):].strip())
                    if payload.get("done"):
                        return
                    if payload.get("error"):  # a clean, terminal error frame from the backend
                        raise BackendError(payload["error"])
                    yield _case(payload)
    except httpx.HTTPError as exc:
        raise BackendError(f"backend unreachable during streaming: {exc}") from exc


# ---- health ----
def health() -> bool:
    try:
        return bool(_get("/health").get("ok"))
    except BackendError:
        return False


# ---- cases: terminal ----
async def create_case(data: bytes, name: str, owner: str = "customer") -> UnderwritingCase:
    return _case(_post("/cases", files={"file": (name, data)}, data={"owner": owner}))


async def add_document(case_id: str, data: bytes, name: str, owner: str = "customer") -> UnderwritingCase:
    return _case(_post(f"/cases/{case_id}/documents", files={"file": (name, data)}))


async def add_message(case_id: str, text: str, owner: str = "customer") -> UnderwritingCase:
    return _case(_post(f"/cases/{case_id}/messages", json_body={"text": text, "owner": owner}))


async def submit(case_id: str) -> UnderwritingCase:
    return _case(_post(f"/cases/{case_id}/submit"))


def get_case(case_id: str) -> UnderwritingCase | None:
    payload = _get(f"/cases/{case_id}")
    if not payload or payload.get("error"):
        return None
    return _case(payload)


def list_pending() -> list[UnderwritingCase]:
    return [_case(p) for p in _get("/cases")]


def list_finalized() -> list[UnderwritingCase]:
    return [_case(p) for p in _get("/ledger")]


def decide(case_id: str, reviewer: str, outcome: Outcome, note: str = "") -> UnderwritingCase:
    body = {"reviewer": reviewer, "outcome": Outcome(outcome).value, "note": note}
    return _case(_post(f"/cases/{case_id}/decide", json_body=body))


async def apply_corrections(case_id: str, updates: dict, reviewer: str = "reviewer") -> UnderwritingCase:
    return _case(_post(f"/cases/{case_id}/corrections",
                       json_body={"updates": updates, "reviewer": reviewer}))


async def reply_for_agent(case: UnderwritingCase) -> str:
    """Intake agent reply for the case's current state. The handler always holds the case, so
    we send its id; the backend reads the freshest state and asks the agent."""
    if case is None:
        return ""
    return _post(f"/cases/{case.case_id}/agent-reply").get("text", "")


# ---- cases: streaming ----
async def create_case_stream(data: bytes, name: str, owner: str = "customer"):
    async for case in _stream_cases("/cases/stream",
                                    files={"file": (name, data)}, data={"owner": owner}):
        yield case


async def create_case_stream_add(case_id: str, data: bytes, name: str, owner: str = "customer"):
    async for case in _stream_cases(f"/cases/{case_id}/documents/stream",
                                    files={"file": (name, data)}):
        yield case


async def submit_stream(case_id: str):
    async for case in _stream_cases(f"/cases/{case_id}/submit/stream"):
        yield case


# ---- entity reconciliation (pure, local) ----
def entity_pairs(case: UnderwritingCase):
    return entity_pairs_from(case.ocr_blocks, case.facts.model_dump())


# ---- document chat (routed through the backend) ----
def analyze(data: bytes, name: str) -> dict:
    return _post("/doc/analyze", files={"file": (name, data)})


async def answer(markdown: str, question: str) -> str:
    return _post("/doc/answer", json_body={"markdown": markdown, "question": question}).get("text", "")
