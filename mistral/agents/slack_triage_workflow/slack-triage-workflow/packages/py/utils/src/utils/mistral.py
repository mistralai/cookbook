"""The app's Mistral SDK clients: one acting as the app itself, one acting for the caller served.

Background work, a deployed worker's included, uses ``mistral_client()``. It authenticates with the
service-account token in ``MISTRAL_SA_TOKEN_PATH`` where the platform mounts one, and with
``MISTRAL_API_KEY`` otherwise. Request-scoped work may instead use a credential installed by its host
and falls back to the application's own credential when no provider is installed. The provider slot
keeps this shared module independent of any host implementation.

Every client keeps the SDK's own HTTP clients, so the standard ``HTTPS_PROXY``, ``NO_PROXY`` and
``SSL_CERT_FILE`` environment variables route and verify its traffic.
"""

import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from env.mistral import env
from mistralai.client import Mistral
from mistralai.client.types import UNSET, OptionalNullable
from mistralai.client.utils.retries import RetryConfig

# The public API, where the SDK goes when ``MISTRAL_BASE_URL`` is unset.
_PUBLIC_API = "https://api.mistral.ai"


class MistralNotConfiguredError(RuntimeError):
    """Raised when a Mistral client is requested but nothing can authenticate its calls."""


class CallerCredentials(NamedTuple):
    """Where to reach the Mistral API as the caller being served, and the token to do it with."""

    server_url: str
    token: str


CallerCredentialProvider = Callable[[], CallerCredentials]

_provider: CallerCredentialProvider | None = None


def install_caller_credentials(provider: CallerCredentialProvider | None) -> None:
    """Make ``provider`` the source of the credential that work on a caller's behalf spends.

    A host able to act for callers installs one during application setup. ``None`` removes it.
    """
    global _provider
    _provider = provider


def caller_credentials_installed() -> bool:
    """Whether anything can supply a caller credential, without asking it for one."""
    return _provider is not None


def caller_credentials() -> CallerCredentials | None:
    """The credential for the caller being served, or None when the app can act only as itself.

    A provider raises rather than returning None when it is installed but cannot act for this
    particular call, such as a request that arrived without the credential it needs.
    """
    return None if _provider is None else _provider()


def _sa_token_path() -> str | None:
    # Only the platform sets this, on a deployed worker. It stays out of `env` so the CLI never asks
    # the developer for a value.
    return os.environ.get("MISTRAL_SA_TOKEN_PATH") or None


def _token_file(path: str) -> Callable[[], str]:
    # Read on every request: the platform rotates the token file under a running worker.
    return lambda: Path(path).read_text(encoding="utf-8").strip()


def _caller_token() -> str:
    credentials = caller_credentials()
    if credentials is None:
        raise MistralNotConfiguredError(
            "Nothing supplies a caller credential, so there is no caller to act for"
        )
    return credentials.token


def mistral_api_url() -> str:
    """The Mistral API root the app calls as itself: ``MISTRAL_BASE_URL``, or the public API.

    For requests the SDK client does not cover (raw HTTP, other Mistral SDKs). Never ends in ``/``.
    """
    return (env.mistral_base_url or _PUBLIC_API).rstrip("/")


def _new_client(
    api_key: str | Callable[[], str],
    server_url: str | None,
    retry_config: OptionalNullable[RetryConfig] = UNSET,
) -> Mistral:
    # The one construction of the SDK client, so every client the app builds is built the same way.
    return Mistral(api_key=api_key, server_url=server_url, retry_config=retry_config)


def mistral_client(
    *,
    server_url: str | None = None,
    retry_config: OptionalNullable[RetryConfig] = UNSET,
) -> Mistral:
    """A client acting as the app itself, against ``MISTRAL_BASE_URL``.

    It authenticates with the token in ``MISTRAL_SA_TOKEN_PATH`` when that is set, and with
    ``MISTRAL_API_KEY`` otherwise. ``server_url`` targets another Mistral API host. A missing key is
    not an error here: building a client stays safe wherever the key is optional, and the API rejects
    the first call with a 401. The empty string is deliberate; ``None`` would make the SDK's telemetry
    hook raise on that call instead, and the SDK would fall back to reading ``MISTRAL_API_KEY`` itself.
    """
    token_path = _sa_token_path()
    api_key = _token_file(token_path) if token_path else env.mistral_api_key or ""
    return _new_client(
        api_key,
        env.mistral_base_url if server_url is None else server_url,
        retry_config,
    )


@lru_cache(maxsize=1)
def _caller_client(server_url: str) -> Mistral:
    # One client serves every caller: the api_key callable runs per request, so each call
    # authenticates with the credential belonging to the request in flight.
    return _new_client(_caller_token, server_url)


def caller_mistral_client() -> Mistral:
    """A client that acts for whoever made the request being served.

    Use while handling caller-scoped work. Where no caller credential is installed, this returns the
    client acting as the app itself (see :func:`mistral_client`).
    """
    credentials = caller_credentials()
    if credentials is None:
        if not env.mistral_api_key and not _sa_token_path():
            raise MistralNotConfiguredError(
                "Neither MISTRAL_API_KEY nor MISTRAL_SA_TOKEN_PATH is configured"
            )
        return mistral_client()
    return _caller_client(credentials.server_url)
