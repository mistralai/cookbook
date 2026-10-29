"""Agent Session activities: create a session on the agent runtime and run one turn in it."""

import os
from datetime import timedelta

import mistralai.workflows as workflows
from mistralai.agents import agents
from mistralai.vibe.harness.app_server.session import CallbackOpenedEvent, MessageHistoryEntry, SessionTerminatedEvent, TextContentBlock, TurnTerminatedEvent

# The deployed agent's name, which must match agent/main.py. Set INVESTIGATION_AGENT_NAME to reach a local `make serve` agent instead.
DEPLOYED_AGENT_NAME = "investigation-agent-runtime"
AGENT_NAME = os.environ.get("INVESTIGATION_AGENT_NAME", "").strip() or DEPLOYED_AGENT_NAME


def api_key() -> str:
    """Return the Mistral API key the workflow opens Agent Sessions with. The SDK finds the Agents control plane itself."""
    return os.environ["INVESTIGATION_AGENT_MISTRAL_API_KEY"].strip()


@workflows.activity(start_to_close_timeout=timedelta(seconds=45), retry_policy_max_attempts=1)
async def create_session() -> str:
    """Create a new Agent Session and return its ID. The thread workflow reuses it for every turn."""
    async with await agents.Agent.load_from(agent_name=AGENT_NAME, api_key=api_key()).create_session() as session:
        return session.session_id


@workflows.activity(start_to_close_timeout=timedelta(minutes=12), retry_policy_max_attempts=2)
async def run_session_turn(session_id: str, turn_id: str, prompt: str) -> str:
    """Run one turn in an existing session and return the final assistant text of that turn.

    `turn_id` is the idempotency key, so a replayed activity does not start a second turn.
    """
    async with await agents.Agent.load_session(session_id, api_key=api_key()) as session:
        async for event in session.run_turn([TextContentBlock(text=prompt)], idempotency_key=turn_id):
            # A workflow cannot answer callbacks, so a callback or a dead session fails the turn.
            if isinstance(event.payload, CallbackOpenedEvent | SessionTerminatedEvent):
                raise RuntimeError(f"Agent turn stopped: {type(event.payload).__name__}")
            if isinstance(event.payload, TurnTerminatedEvent):
                done = event.payload.turn
                if done.termination is None or done.termination.outcome != "completed":
                    raise RuntimeError("Agent turn did not complete")
                # Keep only the last assistant message of this turn: earlier ones are intermediate tool-calling steps.
                items = session.state.history.items if session.state.history else []
                mine = [e for e in items if isinstance(e, MessageHistoryEntry) and e.role == "assistant" and e.turn_id == done.id]
                last = max(mine, key=lambda e: e.index, default=None)
                return "".join(b.text for b in last.content if isinstance(b, TextContentBlock)).strip() if last else ""
    raise RuntimeError("Agent event stream ended before the turn terminated")
