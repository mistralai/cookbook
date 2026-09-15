"""Probe the hosted-agent path before the session, so you know whether portal Traces will
populate. Run this after `azd up` (or any deploy) to confirm the extractor agent runs
server-side through Foundry Agent Service, not just via the /chat/completions fallback.

It reports one of three outcomes:
  - HOSTED OK      the server-side run succeeded. A per-agent trace will show in the portal.
  - FALLBACK ONLY  the server-side run errored. The pipeline still answers, but the Traces
                   tab stays empty. The error is printed so you can fix it before presenting.
  - NOT REGISTERED the agent is not on the project. Run `azd up` or scripts/provision_agents.py.

Env (same as the app): AZURE_AI_PROJECT_ENDPOINT, AZURE_CHAT_DEPLOYMENT, and `az login` (or a
managed identity) with the Azure AI User role on the account.

Usage:
  uv run python scripts/probe_agent.py            # probes the extractor agent
  uv run python scripts/probe_agent.py --name mortgage-intake-agent
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from foundry_agents import EXTRACTOR_AGENT, _run_server, aclose  # noqa: E402


async def _probe(name: str) -> int:
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "").rstrip("/")
    if not endpoint:
        print("AZURE_AI_PROJECT_ENDPOINT is not set. Nothing to probe.")
        return 2

    print(f"Probing hosted agent '{name}' server-side at {endpoint} ...")
    try:
        reply = await _run_server(name, "Reply with the single word: ready.")
    except Exception as exc:  # noqa: BLE001 - we classify and report every failure
        msg = str(exc)
        first = msg.splitlines()[0] if msg.strip() else exc.__class__.__name__
        not_found = "404" in msg or "not found" in msg.lower() or "does not exist" in msg.lower()
        if not_found:
            print(f"NOT REGISTERED: {name} is not on the project ({first}).")
            print("  Fix: run `azd up`, or `uv run python scripts/provision_agents.py`.")
            return 3
        print(f"FALLBACK ONLY: server-side run failed ({first}).")
        print("  The pipeline still answers via /chat/completions, but the portal Traces tab")
        print("  will stay empty for this agent. Resolve this before the session.")
        return 1
    finally:
        await aclose()

    text = (reply.text or "").strip()
    print(f"HOSTED OK: server-side run succeeded. Model replied: {text!r}")
    print("  A per-agent run will appear under this agent's Traces tab in the Foundry portal.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the hosted Foundry agent path.")
    parser.add_argument("--name", default=EXTRACTOR_AGENT, help="Agent name to probe.")
    args = parser.parse_args()
    return asyncio.run(_probe(args.name))


if __name__ == "__main__":
    raise SystemExit(main())
