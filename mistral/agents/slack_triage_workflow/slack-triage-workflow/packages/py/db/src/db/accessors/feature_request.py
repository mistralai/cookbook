"""Reads and writes of the ``feature_request`` table."""

from sqlmodel import col, select

from db.engine import get_session_maker
from db.models.feature_request import FeatureRequest


async def create_feature_request(request: FeatureRequest) -> FeatureRequest:
    async with get_session_maker()() as session:
        session.add(request)
        await session.commit()
        await session.refresh(request)
    return request


async def get_feature_request(request_id: int) -> FeatureRequest | None:
    async with get_session_maker()() as session:
        return await session.get(FeatureRequest, request_id)


async def list_feature_requests(limit: int = 50) -> list[FeatureRequest]:
    async with get_session_maker()() as session:
        rows = await session.execute(
            select(FeatureRequest).order_by(col(FeatureRequest.id).desc()).limit(limit)
        )
        return list(rows.scalars())
