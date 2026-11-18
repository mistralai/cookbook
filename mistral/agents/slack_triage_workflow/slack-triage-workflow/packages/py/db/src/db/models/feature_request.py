"""A feature request filed in the tracker."""

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(UTC)


class FeatureRequest(SQLModel, table=True):
    __tablename__ = "feature_request"

    id: int | None = Field(default=None, primary_key=True)
    summary: str
    description: str
    customer: str
    source_channel: str
    slack_thread_url: str
    urgency: str
    status: str = "open"
    created_at: datetime = Field(default_factory=_now)
