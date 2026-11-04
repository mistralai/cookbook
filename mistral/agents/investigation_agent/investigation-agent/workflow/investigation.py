"""Investigation workflows: a router per Slack mention and one long-lived workflow per Slack thread."""

from contextlib import suppress
from datetime import timedelta
from typing import Any

import mistralai.workflows as workflows
from mistralai.workflows import workflow
from pydantic import BaseModel

with workflow.unsafe.imports_passed_through():
    import structlog

    from activities.agent import create_session, run_session_turn
    from activities.routing import route_message
    from activities.slack import add_reaction, expand_permalinks, is_public_channel, post_reply, read_context, remove_reaction
    from formatting import agent_reply_text, request_from_trigger, ts_key, turn_prompt
    from models import SlackRequest, ThreadState

logger = structlog.get_logger(__name__)

FAILED_TURN_REPLY = "I hit an error while investigating this. Tag me again to retry."
FOOTER = "Tag me to interact in this thread."
# Number of recent message timestamps kept for deduplication.
KNOWN_TS_LIMIT = 200
# A served or deployed Agent Session ends after 4 hours idle or 24 hours in total, so start a new one shortly before either limit.
SESSION_MAX_IDLE = timedelta(hours=3, minutes=30)
SESSION_MAX_AGE = timedelta(hours=23)


class TriggerRequest(BaseModel, extra="allow"):
    """Payload sent by the Slack webhook trigger. Only `body` is read; other fields are accepted and ignored."""

    body: Any = None


@workflow.define(name="investigation-agent-router", workflow_description="Routes a Slack mention to its thread.", execution_timeout=timedelta(minutes=2))
class InvestigationRouter:
    """Short workflow started by the Slack trigger for every mention. It filters the event and hands it to the thread workflow."""

    @workflow.entrypoint
    async def run(self, trigger: TriggerRequest) -> str:
        """Ignore unsupported events and non-public channels, then route the message to its thread."""
        request = request_from_trigger(trigger.body)
        if request is None:
            return "done: unsupported_event"
        if not await is_public_channel(request.channel):
            return "done: non_public_channel"
        outcome = await route_message(request)
        logger.info("investigation_agent_routed", outcome=outcome, channel=request.channel, thread_ts=request.thread_ts)
        return f"done: {outcome}"


@workflow.define(name="investigation-agent-thread", workflow_description="Runs one thread's tags as Agent turns.", execution_timeout=timedelta(days=90))
class InvestigationThread:
    """One workflow per Slack thread. It keeps the Agent Session and answers tags one turn at a time."""

    def __init__(self) -> None:
        self.state = ThreadState()

    @workflow.update(name="message", description="Queue a tagged Slack message for the next turn")
    async def message(self, request: SlackRequest) -> str:
        """Queue a new tag for the next turn and drop duplicates."""
        if request.message_ts in self.state.known_ts:
            return "duplicate"
        self.remember(request.message_ts)
        self.state.pending.append(request)
        return "queued"

    @workflow.entrypoint
    async def run(self, state: ThreadState) -> str:
        """Answer pending tags in batches until the thread has been idle for a day."""
        # The incoming state was already deduped by the router or the previous run; only updates that beat run() are re-accepted.
        early, self.state = self.state.pending, state
        for request in early:
            await self.message(request)
        while True:
            try:
                await workflow.wait_condition(lambda: bool(self.state.pending), timeout=timedelta(days=1))
            except TimeoutError:
                # wait_condition raises on timeout; a tag may still have arrived at the last moment.
                if not self.state.pending:
                    return "done: idle"
            batch, self.state.pending = self.state.pending, []
            await self.answer(batch)
            if workflow.should_continue_as_new():
                workflow.continue_as_new(self.state)

    def remember(self, ts: str) -> None:
        """Record a message or reply timestamp, keeping only the most recent ones."""
        self.state.known_ts = [*self.state.known_ts, ts][-KNOWN_TS_LIMIT:]

    async def answer(self, batch: list[SlackRequest]) -> None:
        """Run one agent turn for the batch and post a single reply under the latest tag."""
        latest = max(batch, key=lambda r: ts_key(r.message_ts))
        turn_id = f"{latest.channel}-{latest.message_ts}"
        # Reactions are cosmetic: a failure must never block the answer.
        for request in batch:
            with suppress(workflows.ActivityError):
                await add_reaction(request)
        text = await self.run_turn(batch, latest, turn_id)
        # Explain how to use the agent the first time it is tagged on a thread parent.
        if any(r.message_ts == r.thread_ts for r in batch):
            text += f"\n\n---\n{FOOTER}"
        try:
            if ts := await post_reply(latest.model_copy(update={"text": text})):
                self.remember(ts)
        except workflows.ActivityError:
            # Keep the thread workflow alive so the next tag can still be answered.
            logger.warning("investigation_agent_reply_failed", channel=latest.channel, thread_ts=latest.thread_ts)
        for request in batch:
            with suppress(workflows.ActivityError):
                await remove_reaction(request)

    async def run_turn(self, batch: list[SlackRequest], latest: SlackRequest, turn_id: str) -> str:
        """Run the agent turn and return the Slack-formatted answer, or an error reply if any step failed."""
        try:
            session_id = await self.session_for(turn_id)
            # The batch tags are already in the prompt; an existing session has also seen every known tag and reply.
            skip = self.state.known_ts if self.state.context_cursor_ts else [r.message_ts for r in batch]
            context = await read_context(self.state, skip)
            texts = [await expand_permalinks(r.text) for r in batch]
            text = await run_session_turn(session_id, turn_id, turn_prompt(texts, context))
        except workflows.ActivityError as error:
            # Drop the session so the next tag starts clean, and tell the user to retry.
            logger.warning("investigation_agent_turn_failed", error_type=type(error.__cause__).__name__)
            self.state.session_id = None
            return FAILED_TURN_REPLY
        self.state.context_cursor_ts = max([m.ts for m in context] + [r.message_ts for r in batch], key=ts_key)
        self.state.last_turn_at = workflow.now()
        return agent_reply_text(text)

    async def session_for(self, turn_id: str) -> str:
        """Return the thread's Agent Session, or start a new one if there is none yet or it is close to its idle or lifetime limit."""
        state, now = self.state, workflow.now()
        last_active = state.last_turn_at or state.session_at
        if state.session_id and state.session_at and last_active and now - state.session_at <= SESSION_MAX_AGE and now - last_active <= SESSION_MAX_IDLE:
            return state.session_id
        session_id = await create_session(f"investigation-session-{turn_id}")
        # The new session has seen nothing, so the next turn re-reads the whole thread.
        state.session_id, state.session_at, state.last_turn_at, state.context_cursor_ts = session_id, now, None, None
        return session_id
