"""Register the pipeline's Foundry agents as code, run by `azd up` (postprovision).

Foundry agents are a data-plane resource, so Bicep cannot create them. This registers them
through the current azure-ai-projects SDK: `agents.create_version(...)` with a
`PromptAgentDefinition` (model deployment + instructions). Agents are immutable and versioned,
so re-running simply adds a new version and the app resolves `latest` — the call is safe to
repeat on every deploy.

The extractor is the only agent the event pipeline needs. Pass --all to also register the
intake, underwriting, and document-chat agents (used by the Gradio app, not the Function).

Instructions come from foundry_agents._FALLBACK_INSTRUCTIONS so the portal agent and the
in-code fallback share one source of truth. Agents are prompt-only: no tools (the Mistral
serving layer has guided decoding off, so tool calling is unavailable).

Env:
  AZURE_AI_PROJECT_ENDPOINT   Foundry project endpoint (azd exports the Bicep output).
  AZURE_CHAT_DEPLOYMENT       Model deployment the agents run on (e.g. mistral-medium-3-5).

Auth is DefaultAzureCredential: the deployer's `az login` during `azd up`.

This never fails the deploy: on any error it logs and exits 0, because the pipeline still
works through the /chat/completions fallback even if registration did not complete.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Reuse the agent names + instruction text from the app so there is one source of truth.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from foundry_agents import (  # noqa: E402
    DOC_CHAT_AGENT,
    EXTRACTOR_AGENT,
    INTAKE_AGENT,
    UNDERWRITING_AGENT,
    _FALLBACK_INSTRUCTIONS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("provision_agents")


def _register(client, model: str, name: str) -> None:
    from azure.ai.projects.models import PromptAgentDefinition

    client.agents.create_version(
        agent_name=name,
        definition=PromptAgentDefinition(
            model=model,
            instructions=_FALLBACK_INSTRUCTIONS[name],
        ),
    )
    log.info("registered agent %s (model %s)", name, model)


def main() -> int:
    parser = argparse.ArgumentParser(description="Register Foundry agents for the pipeline.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also register the intake, underwriting, and document-chat agents (Gradio app).",
    )
    args = parser.parse_args()

    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "").rstrip("/")
    model = os.environ.get("AZURE_CHAT_DEPLOYMENT", "")
    if not endpoint or not model:
        log.warning(
            "skipping agent registration: AZURE_AI_PROJECT_ENDPOINT and AZURE_CHAT_DEPLOYMENT "
            "must both be set (endpoint=%r, model=%r). The pipeline still runs via the "
            "/chat/completions fallback.",
            endpoint,
            model,
        )
        return 0

    names = [EXTRACTOR_AGENT]
    if args.all:
        names += [INTAKE_AGENT, UNDERWRITING_AGENT, DOC_CHAT_AGENT]

    try:
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential

        with (
            DefaultAzureCredential() as credential,
            AIProjectClient(endpoint=endpoint, credential=credential) as client,
        ):
            for name in names:
                _register(client, model, name)
    except Exception as exc:  # never fail azd up over agent registration
        log.warning(
            "agent registration did not complete (%s). The pipeline still runs via the "
            "/chat/completions fallback; register later by re-running `azd up` or this script.",
            str(exc).splitlines()[0] if str(exc).strip() else exc.__class__.__name__,
        )
        return 0

    log.info("agent registration complete: %s", ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
