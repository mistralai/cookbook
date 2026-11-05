"""An uploaded profile PDF, the structured profile read from it, and the role it was matched to."""

from datetime import UTC, datetime

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(UTC)


class Expert(SQLModel, table=True):
    __tablename__ = "expert"

    id: int | None = Field(default=None, primary_key=True)
    file_name: str
    file_id: str
    execution_id: str | None = None
    # processing, then matched or failed
    status: str = "processing"
    name: str | None = None
    headline: str | None = None
    years_of_experience: int | None = None
    skills: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    summary: str | None = None
    role_id: int | None = None
    reason: str | None = None
    # pending, approved or declined
    review: str = "pending"
    created_at: datetime = Field(default_factory=_now)
