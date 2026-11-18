"""Shared fixtures for the generated API tests."""

import pytest
from api.routers.api.internal.health import draining
from env.telemetry import env as telemetry_env
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mistralai_capabilities.fastapi.hooks import FastAPIHooks

telemetry_env.telemetry_enabled = False


def build_app(
    *,
    hooks: FastAPIHooks | None = None,
    routers: str = "api.routers",
    prefix: str = "",
) -> FastAPI:
    # Importing api.main constructs its module-level app, so do it only after telemetry is disabled.
    from api.main import create_app

    app = create_app(hooks=hooks if hooks is not None else FastAPIHooks.empty())
    if routers != "api.routers" or prefix:
        from mistralai_capabilities.fastapi.routing import create_api_router

        app.router.routes.clear()
        app.include_router(create_api_router(routers, prefix=prefix))
    app.dependency_overrides[draining] = lambda: False
    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(build_app())
