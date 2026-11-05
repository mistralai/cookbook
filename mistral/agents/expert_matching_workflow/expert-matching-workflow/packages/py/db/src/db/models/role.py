"""A project role an expert can be matched to."""

from sqlmodel import Field, SQLModel


class Role(SQLModel, table=True):
    __tablename__ = "role"

    id: int | None = Field(default=None, primary_key=True)
    title: str
    project: str
    description: str
