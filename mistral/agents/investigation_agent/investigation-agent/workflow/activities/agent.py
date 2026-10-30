"""Agent Session activities: start a session on the agent runtime and run one turn in it."""

import os
from datetime import timedelta

import mistralai.workflows as workflows
from mistralai.agents import agents

# The deployed agent's name, which must match agent/main.py. Set INVESTIGATION_AGENT_NAME to reach a local `make serve` agent instead.
DEPLOYED_AGENT_NAME = "investigation-agent-runtime"
AGENT_NAME = os.environ.get("INVESTIGATION_AGENT_NAME", "").strip() or DEPLOYED_AGENT_NAME
# Shorter than the activity timeout, so a slow turn ends the wait cleanly and the retry waits on the same turn.
TURN_WAIT_SECONDS = 10 * 60


def agents_client() -> agents.Client:
    """Build a client with the Mistral API key the workflow opens Agent Sessions with. The SDK finds the Agents control plane itself."""
    return agents.Client(api_key=os.environ["INVESTIGATION_AGENT_MISTRAL_API_KEY"].strip())


@workflows.activity(start_to_close_timeout=timedelta(seconds=45), retry_policy_max_attempts=3)
async def create_session(idempotency_key: str) -> str:
    """Start a new Agent Session and return its ID. A retry with the same key returns the same session."""
    async with agents_client() as client:
        session = await agents.Agent.load_from(agent_name=AGENT_NAME, client=client).start_session(idempotency_key=idempotency_key)
        return session.id


@workflows.activity(start_to_close_timeout=timedelta(minutes=12), retry_policy_max_attempts=3)
async def run_session_turn(session_id: str, turn_id: str, prompt: str) -> str:
    """Run one turn in an existing session and return the final assistant text of that turn.

    `turn_id` is the idempotency key, so a retried activity waits on the same turn instead of starting a second one.
    """
    async with agents_client() as client:
        session = agents.Agent.load_from(agent_name=AGENT_NAME, client=client).session(session_id)
        match await session.run(prompt, idempotency_key=turn_id, timeout=TURN_WAIT_SECONDS):
            case agents.Completed(output):
                return str(output or "").strip()
            case agents.Running():
                # The turn goes on. Fail this attempt so the retry waits on the same turn again.
                raise RuntimeError("Agent turn is still running")
            case state:
                # A workflow cannot answer callbacks, so a blocked, interrupted or failed turn fails.
                raise RuntimeError(f"Agent turn did not complete: {type(state).__name__}")
