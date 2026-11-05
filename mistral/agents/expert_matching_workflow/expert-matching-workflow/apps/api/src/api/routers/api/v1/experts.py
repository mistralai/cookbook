from datetime import datetime
from typing import Literal

from db.accessors.expert import get_expert, list_experts, save_expert
from db.accessors.role import list_roles
from db.models.expert import Expert
from fastapi import APIRouter, HTTPException, UploadFile, status
from mistralai_capabilities.workflows.client import get_client, start_workflow
from pydantic import BaseModel
from utils.mistral import mistral_client
from worker.workflows.match_expert import (
    MatchExpert,
    MatchRequest,
    MatchResult,
    RoleBrief,
)

router = APIRouter(tags=["experts"])

FAILED = {"FAILED", "CANCELED", "TERMINATED", "TIMED_OUT"}


class ExpertOut(BaseModel):
    id: int
    file_name: str
    status: str
    name: str | None
    headline: str | None
    years_of_experience: int | None
    skills: list[str]
    summary: str | None
    role_id: int | None
    reason: str | None
    review: str
    created_at: datetime


class Review(BaseModel):
    review: Literal["approved", "declined"]


async def _refresh(expert: Expert) -> Expert:
    """Store the workflow's result once it has finished."""
    client = await get_client()
    execution = await client.workflows.executions.get_workflow_execution_async(execution_id=expert.execution_id)
    if execution.status == "COMPLETED":
        result = MatchResult.model_validate(execution.result)
        expert.sqlmodel_update(result.profile.model_dump())
        expert.role_id = result.match.role_id
        expert.reason = result.match.reason
        expert.status = "matched"
    elif execution.status in FAILED:
        expert.status = "failed"
    else:
        return expert
    await mistral_client().files.delete_async(file_id=expert.file_id)
    return await save_expert(expert)


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload(file: UploadFile) -> ExpertOut:
    """Upload an expert's CV as a PDF and start matching it to a role."""
    roles = await list_roles()
    if not roles:
        raise HTTPException(status.HTTP_409_CONFLICT, "Add a role first")
    uploaded = await mistral_client().files.upload_async(
        file={"file_name": file.filename or "cv.pdf", "content": await file.read()},
        purpose="ocr",
    )
    run = await start_workflow(
        MatchExpert,
        MatchRequest(
            file_id=uploaded.id,
            roles=[RoleBrief.model_validate(r, from_attributes=True) for r in roles],
        ),
        wait_for_result=False,
    )
    expert = await save_expert(
        Expert(
            file_name=uploaded.filename,
            file_id=uploaded.id,
            execution_id=run.execution_id,
        )
    )
    return ExpertOut.model_validate(expert, from_attributes=True)


@router.get("")
async def list_all(limit: int = 50) -> list[ExpertOut]:
    """List the uploaded experts, newest first, with the role each was matched to."""
    experts = [await _refresh(e) if e.status == "processing" else e for e in await list_experts(limit)]
    return [ExpertOut.model_validate(e, from_attributes=True) for e in experts]


@router.post("/{expert_id}/review")
async def review(expert_id: int, body: Review) -> ExpertOut:
    """Approve or decline the role an expert was matched to."""
    expert = await get_expert(expert_id)
    if expert is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such expert")
    if expert.status != "matched":
        raise HTTPException(status.HTTP_409_CONFLICT, "This expert has no match to review yet")
    expert.review = body.review
    return ExpertOut.model_validate(await save_expert(expert), from_attributes=True)
