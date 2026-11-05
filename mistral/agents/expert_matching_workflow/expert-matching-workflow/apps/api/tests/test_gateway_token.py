"""The caller token the gateway sends reaches the route serving that request.

The handover runs on a probe app, because this capability ships no route that calls the Mistral API
on the caller's behalf. The last test checks that the app the server actually serves installs the
provider, so dropping the configure hook fails here rather than passing on the probe's own wiring.
"""

from collections.abc import Iterator, Mapping

import pytest
from api import main
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mistralai_capabilities.apps import credentials as gateway_credentials
from mistralai_capabilities.apps.credentials import install_gateway_credentials
from utils import mistral as mistral_clients
from utils.mistral import (
    MistralNotConfiguredError,
    caller_credentials_installed,
    caller_mistral_client,
    install_caller_credentials,
)

PROXY_URL = "https://gateway.test/v1/proxy"


@pytest.fixture(autouse=True)
def clean_module_state() -> Iterator[None]:
    yield
    install_caller_credentials(None)
    mistral_clients._caller_client.cache_clear()


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(gateway_credentials.env, "apps_proxy_url", PROXY_URL)
    app = FastAPI()
    install_gateway_credentials(app)

    @app.get("/probe")
    async def _probe() -> dict[str, str]:
        client = caller_mistral_client()
        return {
            "api_key": client.sdk_configuration.security().api_key,
            "server_url": client.sdk_configuration.server_url,
        }

    return TestClient(app)


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        pytest.param({"x-apps-token": "tok-apps"}, "tok-apps", id="apps"),
        pytest.param({"x-space-token": "tok-space"}, "tok-space", id="space"),
        pytest.param({"x-apps-token": "tok-apps", "x-space-token": "tok-space"}, "tok-apps", id="apps_first"),
    ],
)
def test_a_route_calls_as_the_caller_it_is_serving(
    probe: TestClient, headers: Mapping[str, str], expected: str
) -> None:
    assert probe.get("/probe", headers=dict(headers)).json() == {"api_key": expected, "server_url": PROXY_URL}


def test_two_callers_in_a_row_do_not_share_a_token(probe: TestClient) -> None:
    # A token left behind by one request would hand the next caller someone else's credentials.
    assert probe.get("/probe", headers={"x-apps-token": "tok-first"}).json()["api_key"] == "tok-first"
    assert probe.get("/probe", headers={"x-apps-token": "tok-second"}).json()["api_key"] == "tok-second"


def test_a_request_with_no_token_has_no_caller_to_act_for(probe: TestClient) -> None:
    with pytest.raises(MistralNotConfiguredError):
        probe.get("/probe")


def test_without_a_gateway_nothing_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A deployment that serves the app directly gets no proxy URL. Installing a provider there
    # would refuse every call the app's own key would have served.
    monkeypatch.setattr(gateway_credentials.env, "apps_proxy_url", None)
    app = FastAPI()
    install_gateway_credentials(app)
    assert caller_credentials_installed() is False
    assert app.user_middleware == []


def test_the_app_the_server_runs_installs_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # Through `create_app`, so dropping `api/configure/gateway_token.py` fails here.
    monkeypatch.setattr(gateway_credentials.env, "apps_proxy_url", PROXY_URL)
    main.create_app()
    assert caller_credentials_installed() is True
