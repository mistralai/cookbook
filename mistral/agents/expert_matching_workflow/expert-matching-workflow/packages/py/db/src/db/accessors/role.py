"""Reads and writes of the ``role`` table."""

from sqlmodel import col, select

from db.engine import get_session_maker
from db.models.role import Role


async def create_role(role: Role) -> Role:
    async with get_session_maker()() as session:
        session.add(role)
        await session.commit()
        await session.refresh(role)
    return role


async def list_roles() -> list[Role]:
    async with get_session_maker()() as session:
        rows = await session.execute(select(Role).order_by(col(Role.id)))
        return list(rows.scalars())
