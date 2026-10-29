"""Pydantic models passed between workflows and activities."""

from datetime import datetime

from pydantic import BaseModel, Field


class SlackRequest(BaseModel):
    """One tagged Slack message, or a reply to post when `text` holds the agent answer."""

    team_id: str = ""
    channel: str
    thread_ts: str
    message_ts: str
    text: str


class ThreadMessage(BaseModel):
    """A message read from a Slack thread, used as conversation context for the agent."""

    user: str
    text: str
    ts: str


class ThreadState(BaseModel):
    """State of one thread workflow, carried over on continue-as-new."""

    channel: str = ""
    thread_ts: str = ""
    # Agent Session reused for every turn of the thread, when it was created, and when its last turn ended.
    session_id: str | None = None
    session_at: datetime | None = None
    last_turn_at: datetime | None = None
    # Tags received while a turn was running; they are answered together in the next turn.
    pending: list[SlackRequest] = Field(default_factory=list)
    # Recent message and reply timestamps, used to drop duplicate tags and to keep them out of the context.
    known_ts: list[str] = Field(default_factory=list)
    # Latest message the current session has seen; the next turn only reads newer thread messages.
    context_cursor_ts: str | None = None
