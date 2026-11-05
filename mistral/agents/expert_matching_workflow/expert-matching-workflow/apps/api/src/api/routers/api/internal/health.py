from pathlib import Path
from typing import Annotated, Literal

from env.mistral import env as mistral_env
from fastapi import APIRouter, Depends, Request, Response
from mistralai_capabilities.fastapi.hooks import FastAPIHooks
from pydantic import BaseModel
from utils.mistral import caller_credentials_installed

DRAIN_SENTINEL = Path("/tmp/drain")


class DependencyStatus(BaseModel):
    configured: bool | None = None
    reachable: bool | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    deployment: str | None = None
    deps: dict[str, DependencyStatus]


class ProbeResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready", "draining", "not_ready"]
    deps: dict[str, DependencyStatus]


router = APIRouter()


def sentinel_present(sentinel: Path = DRAIN_SENTINEL) -> bool:
    return sentinel.exists()


async def draining() -> bool:
    return sentinel_present()


Draining = Annotated[bool, Depends(draining)]


def _manager(request: Request) -> FastAPIHooks:
    return request.app.state.fastapi_hooks


def mistral_configured() -> bool:
    """Whether the app can reach the Mistral API at all, as itself or on a caller's behalf."""
    return bool(mistral_env.mistral_api_key) or caller_credentials_installed()


def _dependency_statuses(checks: dict[str, bool]) -> dict[str, DependencyStatus]:
    return {
        "mistral": DependencyStatus(configured=mistral_configured()),
        **{name: DependencyStatus(reachable=reachable) for name, reachable in checks.items()},
    }


@router.get(
    "",
    operation_id="health",
    summary="Report dependency status; always answers 200",
    response_model_exclude_unset=True,
)
async def health(request: Request) -> HealthResponse:
    """The app's status and each dependency's: whether Mistral is configured, whether each checked
    service answers. ``degraded`` when one does not; the response is 200 either way."""
    manager = _manager(request)
    checks = await manager.report()
    deps = _dependency_statuses(checks)
    return HealthResponse(
        status="degraded" if any(not reachable for reachable in checks.values()) else "ok",
        deployment=manager.metadata_values.get("deployment"),
        deps=deps,
    )


@router.get("/live", operation_id="health_live", summary="Liveness: is the process answering", include_in_schema=False)
async def health_live() -> ProbeResponse:
    return ProbeResponse(status="ok")


@router.get(
    "/startup", operation_id="health_startup", summary="Startup: has the process booted", include_in_schema=False
)
async def health_startup() -> ProbeResponse:
    return ProbeResponse(status="ok")


@router.get(
    "/ready",
    operation_id="health_ready",
    include_in_schema=False,
    summary="Readiness: should this replica receive traffic",
    responses={
        503: {
            "model": ReadinessResponse,
            "description": "Draining, or a hard dependency is down",
        }
    },
    response_model_exclude_unset=True,
)
async def health_ready(request: Request, response: Response, draining: Draining) -> ReadinessResponse:
    manager = _manager(request)
    checks = await manager.ready()
    if draining:
        status: Literal["ready", "draining", "not_ready"] = "draining"
    elif any(not reachable for reachable in checks.values()):
        status = "not_ready"
    else:
        status = "ready"
    if status != "ready":
        response.status_code = 503
    deps = _dependency_statuses(checks)
    return ReadinessResponse(status=status, deps=deps)
