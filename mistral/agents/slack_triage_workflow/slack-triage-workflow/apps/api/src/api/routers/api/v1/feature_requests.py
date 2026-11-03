from datetime import datetime
from typing import Literal

from db.accessors.feature_request import (
    create_feature_request,
    get_feature_request,
    list_feature_requests,
)
from db.models.feature_request import FeatureRequest
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

router = APIRouter(tags=["feature_requests"])

Urgency = Literal["low", "medium", "high"]


class NewFeatureRequest(BaseModel):
    summary: str
    description: str
    customer: str
    source_channel: str
    slack_thread_url: str
    urgency: Urgency


class FeatureRequestOut(NewFeatureRequest):
    id: int
    status: str
    created_at: datetime


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: NewFeatureRequest) -> FeatureRequestOut:
    """File a feature request and return it with its ID."""
    record = await create_feature_request(FeatureRequest(**body.model_dump()))
    return FeatureRequestOut.model_validate(record, from_attributes=True)


@router.get("")
async def list_all(limit: int = 50) -> list[FeatureRequestOut]:
    """List the most recent feature requests, newest first."""
    return [
        FeatureRequestOut.model_validate(r, from_attributes=True)
        for r in await list_feature_requests(limit)
    ]


@router.get("/{request_id}")
async def get(request_id: int) -> FeatureRequestOut:
    """Get one feature request by ID."""
    record = await get_feature_request(request_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such feature request")
    return FeatureRequestOut.model_validate(record, from_attributes=True)
