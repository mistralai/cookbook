"""Consume the Foundry Agent Service agents from code.

Each portal agent (intake, underwriting, extractor, document-chat) runs SERVER-SIDE through
Foundry Agent Service: `build_agent(name).run(prompt)` calls the project's Responses API with
an `agent_reference` (name + id), so the run executes on the platform and shows up in that
agent's Traces and Monitor tabs in the Foundry portal. The portal is the single source of
truth for instructions: edit an agent there and the next run uses it.

If a server-side call fails, it falls back to a client-side chat (FoundryChatClient with the
portal's instructions) so a demo never loses an answer, though that fallback path does not
produce per-agent portal traces.

Auth is Microsoft Entra via DefaultAzureCredential: the Function App's managed identity in
Azure (granted the Azure AI User role on the account), or the `az login` identity locally. No
account key is used for the agent calls.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential

from config import load_settings

# Portal agent names (the /agents store on proj-mortgage-doc-pipeline). Their ids equal their
# names in this project, so an agent_reference uses the name for both fields.
INTAKE_AGENT = "mortgage-intake-agent"
UNDERWRITING_AGENT = "mortgage-underwriting-agent"
EXTRACTOR_AGENT = "mortgage-extractor-agent"
DOC_CHAT_AGENT = "document-chat-agent"

# Foundry Agent Service versioned-agent (prompt agent) data-plane version.
_AGENTS_API = "2025-11-15-preview"
_TOKEN_SCOPE = "https://ai.azure.com/.default"

# Local mirrors of the portal instructions. Used only by the client-side fallback, so a
# transient read error does not break the demo.
_FALLBACK_INSTRUCTIONS: dict[str, str] = {
    INTAKE_AGENT: (
        "You are the mortgage intake assistant. Given what is known so far and the list of "
        "still-missing items, ask the applicant for the missing items in one short, friendly "
        "message. Do not invent values. Monthly debts and years employed need an explicit "
        "number (zero is valid)."
    ),
    UNDERWRITING_AGENT: (
        "You are the mortgage underwriting assistant. You are given pre-computed metrics; do "
        "not recompute them. Policy: auto-approve only when LTV <= 65%, DTI <= 35%, credit >= "
        "700, loan <= 806500. Hard decline when LTV > 95%, DTI > 50%, credit < 580, or loan > "
        "806500. Otherwise refer to a human. Respond exactly as:\n"
        "Recommendation: <approve|decline|refer>\nRisk score: <0-100>\nRationale: <two sentences>."
    ),
    EXTRACTOR_AGENT: (
        "You extract structured data from document text. Respond with JSON only: "
        '{"fields": {...}, "entities": [...]}. No commentary outside the JSON.'
    ),
    DOC_CHAT_AGENT: (
        "You answer questions about a single document that Mistral OCR 4 has read, supplied to "
        "you as Markdown. Answer only from that content. Quote figures, dates, and labels exactly "
        "as written. If the answer is not in the document, say so plainly. Keep answers short and "
        "specific."
    ),
}


@dataclass
class _Reply:
    """Minimal agent result. Callers read only `.text` and `.usage_details`."""
    text: str
    usage_details: dict


def agents_enabled() -> bool:
    """True when the app should run through the Foundry agents."""
    s = load_settings()
    return bool(s.use_foundry_agents and s.project_endpoint)


def _token() -> str:
    return DefaultAzureCredential().get_token(_TOKEN_SCOPE).token


def fetch_instructions(name: str) -> str:
    """Read an agent's current instructions from the portal /agents store (client fallback)."""
    s = load_settings()
    url = f"{s.project_endpoint}/agents/{name}?api-version={_AGENTS_API}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_token()}"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        return data["versions"]["latest"]["definition"]["instructions"]
    except (urllib.error.URLError, KeyError, ValueError):
        return _FALLBACK_INSTRUCTIONS[name]


# ---- server-side path: run the registered agent through Foundry Agent Service ----
_PROJECT_OAI = None       # process-wide AsyncOpenAI client bound to the project
_PROJECT_REFS: tuple = ()  # keep the project client + credential alive (avoid GC closing them)


async def _openai_client():
    global _PROJECT_OAI, _PROJECT_REFS
    if _PROJECT_OAI is None:
        from azure.ai.projects.aio import AIProjectClient
        from azure.identity.aio import DefaultAzureCredential as AioDefaultAzureCredential

        s = load_settings()
        cred = AioDefaultAzureCredential()
        proj = AIProjectClient(endpoint=s.project_endpoint, credential=cred)
        _PROJECT_OAI = proj.get_openai_client()
        _PROJECT_REFS = (cred, proj)
    return _PROJECT_OAI


async def aclose() -> None:
    """Close the process-wide project client + credential. The long-running app leaves these
    open for its lifetime; short-lived callers (scripts/probe_agent.py) call this to avoid an
    'Unclosed client session' warning on exit."""
    global _PROJECT_OAI, _PROJECT_REFS
    for ref in _PROJECT_REFS:
        close = getattr(ref, "close", None) or getattr(ref, "aclose", None)
        if close:
            try:
                result = close()
                if result is not None:
                    await result
            except Exception:  # cleanup is best-effort
                pass
    _PROJECT_OAI = None
    _PROJECT_REFS = ()


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    it = int(getattr(u, "input_tokens", 0) or 0)
    ot = int(getattr(u, "output_tokens", 0) or 0)
    return {"input_token_count": it, "output_token_count": ot, "total_token_count": it + ot}


async def _run_server(name: str, prompt: str) -> _Reply:
    """Run the agent server-side. Passing both name and id in the agent_reference is what makes
    the run correlate to the agent in the portal's Traces and Monitor."""
    client = await _openai_client()
    ref = {"name": name, "id": name, "type": "agent_reference"}
    resp = await client.responses.create(
        extra_body={"agent_reference": ref}, input=prompt, store=False)
    return _Reply(text=(resp.output_text or ""), usage_details=_usage(resp))


# ---- client-side fallback: a chat with the portal instructions (no per-agent portal traces) ----
@lru_cache(maxsize=1)
def _client() -> FoundryChatClient:
    s = load_settings()
    return FoundryChatClient(
        project_endpoint=s.project_endpoint,
        model=s.chat_deployment,
        credential=DefaultAzureCredential(),
        allow_preview=True,
    )


@lru_cache(maxsize=None)
def _client_agent(name: str) -> Agent:
    return Agent(client=_client(), name=name, instructions=fetch_instructions(name))


async def _run_client(name: str, prompt: str) -> _Reply:
    resp = await _client_agent(name).run(prompt)
    usage = getattr(resp, "usage_details", None)
    return _Reply(text=(resp.text or ""), usage_details=usage if isinstance(usage, dict) else {})


class _Agent:
    """Runs a registered portal agent server-side, falling back to a client-side chat on error."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def run(self, prompt: str) -> _Reply:
        try:
            return await _run_server(self.name, prompt)
        except Exception as exc:
            first = str(exc).splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            # Rehearsal safety: strict mode re-raises so a missing portal trace is a loud
            # failure, not a silent fallback that leaves the Traces tab empty on stage.
            if load_settings().foundry_agents_strict:
                logging.error("Agent %s: server-side run failed (%s); strict mode, not "
                              "falling back", self.name, first)
                raise
            logging.warning("Agent %s: server-side run failed (%s); using client fallback",
                            self.name, first)
            return await _run_client(self.name, prompt)


def build_agent(name: str) -> _Agent:
    """A callable agent that runs the named portal agent server-side (Agent Service)."""
    return _Agent(name)
