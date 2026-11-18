from db.accessors.role import create_role, list_roles
from db.models.role import Role
from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter(tags=["roles"])


class NewRole(BaseModel):
    title: str
    project: str
    description: str


class RoleOut(NewRole):
    id: int


@router.get("")
async def list_all() -> list[RoleOut]:
    """List the open roles experts are matched against."""
    return [RoleOut.model_validate(r, from_attributes=True) for r in await list_roles()]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: NewRole) -> RoleOut:
    """Add an open role."""
    role = await create_role(Role(**body.model_dump()))
    return RoleOut.model_validate(role, from_attributes=True)
