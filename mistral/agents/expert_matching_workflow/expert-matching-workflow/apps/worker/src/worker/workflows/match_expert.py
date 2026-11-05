import json

from mistralai import workflows
from mistralai.workflows import workflow
from pydantic import BaseModel, Field

with workflow.unsafe.imports_passed_through():
    from env.matching import env
    from mistralai.client.models import (
        ChatCompletionRequest,
        FileChunk,
        OCRRequest,
        ResponseFormat,
        SystemMessage,
        UserMessage,
    )
    from mistralai.extra import response_format_from_pydantic_model
    from mistralai.workflows.plugins.mistralai import chat_parse_to_model, mistralai_ocr

PROMPT = """You staff experts on client projects. Given an expert's profile and a list of open roles,
pick the one role the expert fits best. Explain why in two or three sentences, naming the skills
or experience that match, and anything the expert lacks for the role."""


class RoleBrief(BaseModel):
    id: int
    title: str
    project: str
    description: str


class MatchRequest(BaseModel):
    file_id: str
    roles: list[RoleBrief]


class ExpertProfile(BaseModel):
    name: str = Field(description="The expert's full name")
    headline: str = Field(description="Current or most recent job title, with the employer")
    years_of_experience: int = Field(description="Total years of professional experience")
    skills: list[str] = Field(description="Up to ten key skills, tools or domains")
    summary: str = Field(description="Two sentences on what the expert has done")


class RoleMatch(BaseModel):
    role_id: int
    reason: str


class MatchResult(BaseModel):
    profile: ExpertProfile
    match: RoleMatch


@workflows.workflow.define(
    name="match-expert",
    workflow_display_name="Match expert",
    workflow_description="Read an expert's profile and pick the project role that suits the expert best.",
)
class MatchExpert:
    @workflows.workflow.entrypoint
    async def run(self, request: MatchRequest) -> MatchResult:
        ocr = await mistralai_ocr(
            OCRRequest(
                model=env.ocr_model,
                document=FileChunk(file_id=request.file_id),
                document_annotation_format=ResponseFormat.model_validate(
                    response_format_from_pydantic_model(ExpertProfile)
                ),
            )
        )
        profile = ExpertProfile.model_validate_json(ocr.document_annotation)

        roles = json.dumps([role.model_dump() for role in request.roles], indent=2)
        match = await chat_parse_to_model(
            RoleMatch,
            ChatCompletionRequest(
                model=env.matching_model,
                messages=[
                    SystemMessage(content=PROMPT),
                    UserMessage(content=f"Expert profile:\n{profile.model_dump_json(indent=2)}\n\nRoles:\n{roles}"),
                ],
            ),
        )
        return MatchResult(profile=profile, match=match)
