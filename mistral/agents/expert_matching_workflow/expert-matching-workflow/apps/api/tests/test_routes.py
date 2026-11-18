from collections import Counter
from pathlib import Path

import pytest
from api.routers.api.internal import health as health_module
from api.routers.api.internal.health import mistral_configured, sentinel_present
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from mistralai_capabilities.fastapi.openapi import HTTP_METHODS, operation_id_problems, untyped_json_responses
from starlette.routing import Route
from utils.mistral import CallerCredentialProvider, CallerCredentials, install_caller_credentials

from conftest import build_app


def _caller_credentials() -> CallerCredentials:
    """Stand in for a deployment that can act as its callers."""
    return CallerCredentials(server_url="https://gateway.test/v1/proxy", token="tok-caller")


@pytest.fixture
def app() -> FastAPI:
    return build_app()


def test_health_reports_status_and_dependencies(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert set(data["deps"]) == {"mistral"}


@pytest.mark.parametrize(
    ("api_key", "provider", "expected"),
    [
        pytest.param("key", None, True, id="own_key"),
        pytest.param(None, _caller_credentials, True, id="caller_credentials_only"),
        pytest.param("key", _caller_credentials, True, id="both"),
        pytest.param(None, None, False, id="neither"),
    ],
)
def test_health_counts_a_caller_credential_as_a_way_to_reach_mistral(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str | None,
    provider: CallerCredentialProvider | None,
    expected: bool,
) -> None:
    monkeypatch.setattr(health_module.mistral_env, "mistral_api_key", api_key)
    install_caller_credentials(provider)
    try:
        assert mistral_configured() is expected
    finally:
        install_caller_credentials(None)


def test_the_drain_signal_flips_when_the_file_appears(tmp_path: Path) -> None:
    """Presence alone is the signal — no content, no parsing, nothing that can half-succeed.

    This covers the read side only. That the path here is the path the `preStop` hook writes is a
    separate contract, owned by the deployment that installs the hook.
    """
    sentinel = tmp_path / "drain"

    assert sentinel_present(sentinel) is False
    sentinel.touch()
    assert sentinel_present(sentinel) is True


def test_unprefixed_routes_are_not_exposed(client: TestClient) -> None:
    assert client.get("/health").status_code == 404


def test_every_json_route_declares_a_typed_response(app: FastAPI) -> None:
    """Codegen guard: every JSON response must resolve to a concrete schema.

    The OpenAPI spec feeds spec-driven TS generation; an untyped ``-> Any`` route
    silently drops out of the generated types. A model, a primitive (``-> list[str]``)
    or a union with a typed member all count; see ``mistralai_capabilities.fastapi.openapi``.
    """
    offenders = untyped_json_responses(app.openapi())
    assert not offenders, "untyped JSON responses (give the route a concrete return type): " + ", ".join(offenders)


def test_every_operation_id_is_connector_safe(app: FastAPI) -> None:
    """A Studio HTTP connector imports this spec and names each MCP tool after its operationId.

    It rejects a missing, shared, over-64-character or non ``[A-Za-z0-9_-]`` id.
    """
    problems = operation_id_problems(app.openapi())
    assert not problems, "operationIds a connector rejects: " + ", ".join(map(str, problems))


def test_the_probes_answer_but_stay_out_of_the_spec(client: TestClient, app: FastAPI) -> None:
    """Only the kubelet calls the probes, so neither the generated client nor a connector gets them."""
    paths = app.openapi()["paths"]

    for probe in ("/api/health/live", "/api/health/startup", "/api/health/ready"):
        assert probe not in paths
        assert client.get(probe).status_code != 404
    assert "/api/health" in paths


def test_no_route_path_is_registered_twice(app: FastAPI) -> None:
    """Read from the spec rather than from ``app.routes``.

    ``app.routes`` looks equivalent but is not: FastAPI's lazy ``include_router`` leaves
    ``_IncludedRouter`` objects there, so an ``isinstance`` filter discards every API route and the
    guard would inspect only ``/docs``, ``/redoc``, ``/openapi.json`` .
    """
    served = [route.path for route in app.routes if isinstance(route, Route | APIRoute)]
    paths = list(app.openapi()["paths"]) + served

    duplicates = {path: count for path, count in Counter(paths).items() if count > 1}

    assert duplicates == {}, f"duplicate routes shadow each other: {duplicates}"
    assert "/api/health" in paths, "the guard is not seeing API routes again"


def test_every_operation_id_is_unique(app: FastAPI) -> None:
    """Operation ids are the generated client's function names, so a collision is a broken client.

    Distinct paths cannot collide here, but a router factory mounted twice under the same
    caller-supplied name can: the paths differ while the ids do not. FastAPI only warns, the
    spec becomes invalid, and the TS generator emits two functions with one name.
    """
    operations = [
        operation["operationId"]
        for methods in app.openapi()["paths"].values()
        for method, operation in methods.items()
        if method in HTTP_METHODS and "operationId" in operation
    ]

    duplicates = {name: count for name, count in Counter(operations).items() if count > 1}

    assert duplicates == {}, f"colliding operation ids generate colliding client functions: {duplicates}"
    assert operations, "the guard is not seeing any operations"
