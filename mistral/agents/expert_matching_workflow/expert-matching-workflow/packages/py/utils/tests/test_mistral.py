"""Which credential a Mistral call spends, and where its traffic goes.

Background work spends the app's own credential: a deployed worker's service-account token, else
the key. A call made for a caller must spend that caller's credential, whatever supplies it. Where
nothing supplies one the call falls back to the app's own credential, and a caller call with nothing
to authenticate it is refused rather than sent unauthenticated.
Every client trusts the environment's proxy and CA settings.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from utils import mistral as mistral_clients
from utils.mistral import (
    CallerCredentials,
    MistralNotConfiguredError,
    caller_mistral_client,
    install_caller_credentials,
    mistral_api_url,
    mistral_app_token,
    mistral_client,
    mistral_credential_configured,
)

PROXY_URL = "https://gateway.test/v1/proxy"
APP_KEY = "app-key"


@pytest.fixture(autouse=True)
def clean_module_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(mistral_clients.env, "mistral_api_key", APP_KEY)
    monkeypatch.delenv("MISTRAL_SA_TOKEN_PATH", raising=False)
    monkeypatch.setattr(mistral_clients.env, "mistral_base_url", None)
    install_caller_credentials(None)
    mistral_clients._caller_client.cache_clear()
    yield
    install_caller_credentials(None)
    mistral_clients._caller_client.cache_clear()


def install_token(token: str | None) -> None:
    """Stand in for a deployment that can act as its callers, holding one token at a time."""

    def provider() -> CallerCredentials:
        if token is None:
            raise MistralNotConfiguredError("this call carries no caller credential")
        return CallerCredentials(server_url=PROXY_URL, token=token)

    install_caller_credentials(provider)


def test_background_work_spends_the_apps_own_key_against_its_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mistral_clients.env, "mistral_base_url", "https://dedicated.test")
    client = mistral_client()
    assert client.sdk_configuration.security.api_key == APP_KEY
    assert client.sdk_configuration.server_url == "https://dedicated.test"


def test_background_work_can_target_another_mistral_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mistral_clients.env, "mistral_base_url", "https://dedicated.test")
    assert mistral_client(server_url="https://workflows.test").sdk_configuration.server_url == "https://workflows.test"


@pytest.mark.parametrize(
    "api_key",
    [pytest.param(APP_KEY, id="token_wins_over_a_key"), pytest.param(None, id="token_and_no_key")],
)
def test_a_deployed_worker_spends_its_service_account_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_key: str | None
) -> None:
    # The platform reserves MISTRAL_API_KEY on a worker and mounts the token file instead.
    token_file = tmp_path / "sa-token"
    token_file.write_text("tok-worker\n", encoding="utf-8")
    monkeypatch.setattr(mistral_clients.env, "mistral_api_key", api_key)
    monkeypatch.setenv("MISTRAL_SA_TOKEN_PATH", str(token_file))
    for client in (mistral_client(), caller_mistral_client()):
        assert client.sdk_configuration.security().api_key == "tok-worker"


def test_a_rotated_token_is_spent_from_the_next_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    token_file = tmp_path / "sa-token"
    token_file.write_text("tok-before", encoding="utf-8")
    monkeypatch.setenv("MISTRAL_SA_TOKEN_PATH", str(token_file))
    client = mistral_client()
    token_file.write_text("tok-after", encoding="utf-8")
    assert client.sdk_configuration.security().api_key == "tok-after"


@pytest.mark.parametrize(
    ("api_key", "token", "spent"),
    [
        pytest.param(APP_KEY, None, APP_KEY, id="key_only"),
        pytest.param(None, "tok-worker", "tok-worker", id="token_file_only_as_on_a_deployed_worker"),
        pytest.param(APP_KEY, "tok-worker", "tok-worker", id="token_file_wins_over_a_key"),
    ],
)
def test_work_gated_on_the_apps_credential_runs_on_a_key_or_a_token_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_key: str | None, token: str | None, spent: str
) -> None:
    # Optional work checks for a credential before building a client; checking the key alone would
    # turn it off on every deployed worker, where the platform reserves MISTRAL_API_KEY.
    monkeypatch.setattr(mistral_clients.env, "mistral_api_key", api_key)
    if token is not None:
        token_file = tmp_path / "sa-token"
        token_file.write_text(f"{token}\n", encoding="utf-8")
        monkeypatch.setenv("MISTRAL_SA_TOKEN_PATH", str(token_file))
    assert mistral_credential_configured()
    assert mistral_app_token() == spent


def test_with_no_credential_gated_work_is_off_and_a_plain_token_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mistral_clients.env, "mistral_api_key", None)
    assert not mistral_credential_configured()
    with pytest.raises(MistralNotConfiguredError):
        mistral_app_token()


def test_an_unreadable_token_file_is_refused_as_not_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MISTRAL_SA_TOKEN_PATH", str(tmp_path / "not-mounted"))
    with pytest.raises(MistralNotConfiguredError):
        mistral_app_token()


def test_a_plain_token_is_read_again_after_a_rotation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    token_file = tmp_path / "sa-token"
    token_file.write_text("tok-before", encoding="utf-8")
    monkeypatch.setenv("MISTRAL_SA_TOKEN_PATH", str(token_file))
    assert mistral_app_token() == "tok-before"
    token_file.write_text("tok-after", encoding="utf-8")
    assert mistral_app_token() == "tok-after"


def test_without_a_key_background_work_still_builds_a_client_the_api_will_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An empty key, not None: the API answers 401, where None would make the SDK's telemetry hook
    # raise instead and the SDK fall back to reading MISTRAL_API_KEY on its own.
    monkeypatch.setattr(mistral_clients.env, "mistral_api_key", None)
    assert mistral_client().sdk_configuration.security.api_key == ""


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        pytest.param(None, "https://api.mistral.ai", id="public_api_by_default"),
        pytest.param("https://dedicated.test/", "https://dedicated.test", id="configured_without_trailing_slash"),
    ],
)
def test_raw_requests_reach_the_same_api_root_as_the_sdk(
    monkeypatch: pytest.MonkeyPatch, base_url: str | None, expected: str
) -> None:
    monkeypatch.setattr(mistral_clients.env, "mistral_base_url", base_url)
    assert mistral_api_url() == expected


def test_clients_keep_http_clients_that_trust_the_environment() -> None:
    # A pin, not a behaviour of ours: HTTPS_PROXY, NO_PROXY and SSL_CERT_FILE are the documented way
    # to route and verify Mistral traffic, and they stop working if a factory ever injects HTTP
    # clients that ignore the environment.
    install_token("tok-caller")
    for client in (mistral_client(), caller_mistral_client()):
        assert client.sdk_configuration.client.trust_env
        assert client.sdk_configuration.async_client.trust_env


def test_with_nothing_installed_a_caller_call_spends_the_apps_own_key() -> None:
    client = caller_mistral_client()
    assert client.sdk_configuration.server_url is None
    assert client.sdk_configuration.security.api_key == APP_KEY


def test_a_caller_call_spends_the_installed_credential() -> None:
    install_token("tok-caller")
    client = caller_mistral_client()
    assert client.sdk_configuration.server_url == PROXY_URL
    assert client.sdk_configuration.security().api_key == "tok-caller"


def test_one_pooled_client_authenticates_each_call_as_its_own_caller() -> None:
    # The SDK resolves a callable api_key per request, so callers share a client without
    # ever sharing a credential.
    install_token("tok-first")
    first = caller_mistral_client()
    install_token("tok-second")
    second = caller_mistral_client()
    assert first is second
    assert second.sdk_configuration.security().api_key == "tok-second"


def test_a_dedicated_deployment_keeps_its_own_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mistral_clients.env, "mistral_base_url", "https://dedicated.test")
    assert caller_mistral_client().sdk_configuration.server_url == "https://dedicated.test"


@pytest.mark.parametrize(
    ("api_key", "installed"),
    [
        pytest.param(APP_KEY, True, id="installed_but_no_credential_for_this_call"),
        pytest.param(None, True, id="installed_but_no_credential_and_no_key"),
        pytest.param(None, False, id="nothing_installed_and_no_key"),
    ],
)
def test_a_call_with_nothing_to_authenticate_it_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str | None,
    installed: bool,
) -> None:
    monkeypatch.setattr(mistral_clients.env, "mistral_api_key", api_key)
    if installed:
        install_token(None)
    with pytest.raises(MistralNotConfiguredError):
        caller_mistral_client()
