"""Central configuration, loaded from environment (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv optional; real env vars still work
    pass


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required env var {name}. Copy .env.example to .env and fill it in.")
    return value


@dataclass(frozen=True)
class Settings:
    # Shared Foundry account key (OCR + chat use the same account)
    ai_key: str
    # OCR (Mistral OCR 4)
    ocr_endpoint: str
    ocr_deployment: str
    # Chat / reasoning (Mistral Medium 3.5) via the OpenAI-compatible route
    chat_base_url: str
    chat_deployment: str
    # Foundry project endpoint (evaluation + Projects SDK). Optional: empty until wired.
    project_endpoint: str
    # Route intake/underwriting/extraction through the Foundry Agent Service workflow.
    use_foundry_agents: bool
    # Rehearsal safety: when true, a failed server-side agent run raises instead of falling
    # back to /chat/completions, so a missing portal trace surfaces as a loud error rather
    # than a silent success. Leave false in production so a demo never loses an answer.
    foundry_agents_strict: bool
    # Storage
    storage_connection_string: str
    inbox_container: str
    results_container: str


def load_settings() -> Settings:
    # Chat/inference are optional: the OCR-4-only pipeline extracts fields from OCR 4's
    # annotation and parses intake answers deterministically, so no chat model is required.
    inference = os.environ.get("AZURE_INFERENCE_ENDPOINT", "").rstrip("/")
    return Settings(
        ai_key=_require("AZURE_AI_KEY"),
        ocr_endpoint=_require("AZURE_OCR_ENDPOINT").rstrip("/"),
        ocr_deployment=_require("AZURE_OCR_DEPLOYMENT"),
        # If an inference endpoint is set, expose the OpenAI-style base_url the chat
        # client would use; empty when running OCR-only.
        chat_base_url=f"{inference}/openai/v1" if inference else "",
        chat_deployment=os.environ.get("AZURE_CHAT_DEPLOYMENT", ""),
        project_endpoint=os.environ.get("AZURE_AI_PROJECT_ENDPOINT", ""),
        use_foundry_agents=os.environ.get("USE_FOUNDRY_AGENTS", "").lower() in ("1", "true", "yes"),
        foundry_agents_strict=os.environ.get("FOUNDRY_AGENTS_STRICT", "").lower() in ("1", "true", "yes"),
        storage_connection_string=os.environ.get("AZURE_STORAGE_CONNECTION_STRING", ""),
        inbox_container=os.environ.get("AZURE_STORAGE_INBOX_CONTAINER", "inbox"),
        results_container=os.environ.get("AZURE_STORAGE_RESULTS_CONTAINER", "results"),
    )
