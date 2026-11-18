"""Reads and writes of the ``expert`` table."""

from sqlmodel import col, select

from db.engine import get_session_maker
from db.models.expert import Expert


async def save_expert(expert: Expert) -> Expert:
    """Insert a new expert, or write the changes to one read earlier."""
    async with get_session_maker()() as session:
        expert = await session.merge(expert)
        await session.commit()
        await session.refresh(expert)
    return expert


async def get_expert(expert_id: int) -> Expert | None:
    async with get_session_maker()() as session:
        return await session.get(Expert, expert_id)


async def list_experts(limit: int = 50) -> list[Expert]:
    async with get_session_maker()() as session:
        rows = await session.execute(select(Expert).order_by(col(Expert.id).desc()).limit(limit))
        return list(rows.scalars())
