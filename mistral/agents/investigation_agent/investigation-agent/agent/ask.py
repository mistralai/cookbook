"""Ask the agent one question over the Session API and print the reply. Run it while `make serve` is up."""

import asyncio
import os
import sys

from mistralai.agents import agents
from mistralai.vibe.harness.app_server.session import HistoryEntryCompletedEvent, MessageHistoryEntry, TextContentBlock

DEFAULT_QUESTION = "What's causing the latest ValueError in Sentry, and which code is responsible?"


async def main(question: str) -> None:
    # Same name `make serve` used, falling back to the deployed agent.
    agent = agents.Agent.load_from(agent_name=os.environ.get("INVESTIGATION_AGENT_NAME", "").strip() or "investigation-agent-runtime")
    async with await agent.create_session() as session:
        async for event in session.run_turn([TextContentBlock(text=question)]):
            if not isinstance(event.payload, HistoryEntryCompletedEvent):
                continue
            entry = event.payload.entry
            # Tool calls and results have no role; show their kind so you can follow the investigation.
            if isinstance(entry, MessageHistoryEntry) and entry.role == "assistant":
                print("\n" + "".join(block.text for block in entry.content if isinstance(block, TextContentBlock)))
            elif not isinstance(entry, MessageHistoryEntry):
                print(f"[{type(entry).__name__}]", flush=True)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or DEFAULT_QUESTION))
