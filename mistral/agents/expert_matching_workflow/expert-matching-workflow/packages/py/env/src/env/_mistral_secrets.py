"""A settings source reading the workspace secrets managed at https://console.mistral.ai/secrets.

With ``MISTRAL_SECRETS_ENABLED=true`` and ``MISTRAL_API_KEY`` set, a setting marked
:class:`WorkspaceSecret` that nothing else sets is read from the workspace secret whose name is its
environment variable, compared case-insensitively: ``OPENAI_API_KEY`` fills ``openai_api_key``. An
unmarked setting is never read from the workspace, so apps sharing one cannot pick up each other's
values for a name they both use. The environment and the ``.env`` always win, so a local value
overrides the shared one. The source is off unless ``MISTRAL_SECRETS_ENABLED`` is true; the
generated ``.env`` turns it on.

The key's workspace and the names of its secrets are read once per process; a settings class then
fetches only the values of its fields that the sources above it left unset, at most 100 per request
as the API allows. A key the API rejects, or an API out of reach, logs a warning and leaves the other
sources to load the settings: a required setting still fails validation by name. After a failed
request the process stops asking for 30 seconds, so the settings classes built at startup do not each
wait out their own timeout, then asks again. A secret listed but not readable, such as one whose
latest version was deleted, is skipped. A key that can list the secrets but not read their values
raises :class:`WorkspaceSecretsForbiddenError`, since the settings it would fill silently fall back.

The bootstrap reads ``os.environ`` directly rather than ``env.mistral``, which is itself a settings
class built through this source. Requests use ``urllib`` so the ``env`` package stays a leaf; it
honours ``HTTPS_PROXY``, ``NO_PROXY`` and ``SSL_CERT_FILE`` as the Mistral SDK clients do.
"""

import json
import logging
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from http import HTTPStatus
from itertools import batched
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from pydantic import AliasChoices, AliasPath
from pydantic.fields import FieldInfo
from pydantic_settings import EnvSettingsSource

logger = logging.getLogger(__name__)

# The public API, where the SDK goes when ``MISTRAL_BASE_URL`` is unset.
_PUBLIC_API = "https://api.mistral.ai"
_TIMEOUT_SECONDS = 10.0
_REFERENCE_PREFIX = "secret:workspace:"
# The most names ``/v1/secrets/values`` accepts in one request.
_VALUES_PER_REQUEST = 100
# How long the process stops asking after a failed request.
_RETRY_AFTER_SECONDS = 30.0
_FAILURES = (OSError, ValueError, KeyError, TypeError)


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


class WorkspaceSecretsForbiddenError(RuntimeError):
    """The API key can list the workspace secrets but is not allowed to read their values."""


class _Workspace:
    """The secrets of the workspace an API key belongs to: every name, and the values read so far."""

    def __init__(self, api_url: str, api_key: str) -> None:
        self._api_url = api_url
        self._api_key = api_key
        self._names: frozenset[str] | None = None
        self._values: dict[str, str] = {}
        # Listed names whose value the API would not return, so no later settings class asks again.
        self._unreadable: set[str] = set()
        self._failed_at: float | None = None

    def _paused(self) -> bool:
        return self._failed_at is not None and time.monotonic() - self._failed_at < _RETRY_AFTER_SECONDS

    def _known_names(self) -> frozenset[str]:
        if self._names is None and not self._paused():
            try:
                self._names = self._list_names()
            except _FAILURES as error:
                logger.warning(
                    "Could not list the workspace secrets at %s, so none are loaded for %d seconds: %s",
                    self._api_url,
                    _RETRY_AFTER_SECONDS,
                    error,
                )
                self._failed_at = time.monotonic()
        return self._names or frozenset()

    def _get(self, path: str, query: Mapping[str, str] | None = None) -> Any:
        url = f"{self._api_url}{path}" + (f"?{urlencode(query, safe=':,')}" if query else "")
        request = Request(url, headers={"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"})
        with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            return json.load(response)

    def _list_names(self) -> frozenset[str]:
        workspace_id = self._get("/v1/users/me")["workspace"]["id"]
        names: set[str] = set()
        query = {"limit": "100"}
        while True:
            page = self._get(f"/v1/secrets/workspaces/{quote(workspace_id, safe='')}", query)
            names.update(item["name"].lower() for item in page["items"])
            if not page.get("next_cursor"):
                return frozenset(names)
            query = {"limit": "100", "cursor": page["next_cursor"]}

    def read(self, names: Iterable[str]) -> dict[str, str]:
        """The value of each of ``names`` that the workspace holds, keyed as given."""
        names = list(names)
        unread = sorted({name.lower() for name in names} & self._known_names() - self._values.keys() - self._unreadable)
        for batch in batched(unread, _VALUES_PER_REQUEST):
            if self._paused():
                break
            try:
                self._values |= self._read_values(batch)
            except _FAILURES as error:
                logger.warning(
                    "Could not read the workspace secrets, so no more are loaded for %d seconds: %s",
                    _RETRY_AFTER_SECONDS,
                    error,
                )
                self._failed_at = time.monotonic()
        return {name: self._values[name.lower()] for name in names if name.lower() in self._values}

    def _read_values(self, batch: Sequence[str]) -> dict[str, str]:
        refs = ",".join(_REFERENCE_PREFIX + name for name in batch)
        try:
            secrets = self._get("/v1/secrets/values", {"names": refs})
        except HTTPError as error:
            if error.code == HTTPStatus.FORBIDDEN:
                raise WorkspaceSecretsForbiddenError(
                    "MISTRAL_API_KEY can list the workspace secrets but not read them. Use a key allowed to read "
                    "them, or set MISTRAL_SECRETS_ENABLED=false and set the settings locally."
                ) from error
            if error.code != HTTPStatus.NOT_FOUND:
                raise
            # One unreadable name fails the whole request, so halve the batch until it is isolated.
            if len(batch) == 1:
                logger.warning("The workspace secret %s is listed but cannot be read, so it is skipped", batch[0])
                self._unreadable.add(batch[0])
                return {}
            middle = len(batch) // 2
            return self._read_values(batch[:middle]) | self._read_values(batch[middle:])
        return {secret["name"].lower().removeprefix(_REFERENCE_PREFIX): secret["value"] for secret in secrets}


class WorkspaceSecret:
    """Marks a setting the workspace secrets may fill when nothing local sets it.

    ``openai_api_key: Annotated[str | None, WorkspaceSecret()] = None``
    """


def _field_env_names(
    field_name: str, field: FieldInfo, config: Mapping[str, Any], env_prefix: str, prefix_target: str
) -> dict[str, str]:
    """The environment names that can set a field, keyed by the field key each one sets.

    Follows the public Pydantic rules: a validation alias replaces the field name unless
    ``validate_by_name`` (or ``populate_by_name``) is on, ``validate_by_alias=False`` ignores the
    aliases, only the first segment of an ``AliasPath`` names a variable, and ``env_prefix`` applies
    to the field name or the aliases as ``env_prefix_target`` says.
    """
    by_alias = config.get("validate_by_alias") is not False
    alias = field.validation_alias if by_alias else None
    choices = alias.choices if isinstance(alias, AliasChoices) else [alias] if alias else []
    heads = [choice.path[0] if isinstance(choice, AliasPath) else choice for choice in choices]
    alias_prefix = env_prefix if prefix_target in {"alias", "all"} else ""
    names = {head: alias_prefix + head for head in heads if isinstance(head, str)}
    by_name = config.get("validate_by_name", config.get("populate_by_name")) or not by_alias
    if not names or by_name:
        names[field_name] = (env_prefix if prefix_target in {"variable", "all"} else "") + field_name
    return names


@cache
def _workspace(api_url: str, api_key: str) -> _Workspace:
    return _Workspace(api_url, api_key)


class MistralSecretsSettingsSource(EnvSettingsSource):
    """Workspace secrets as if they were environment variables, for the marked fields left unset.

    Field names, aliases, ``env_prefix`` and JSON-encoded values resolve exactly as they do for the
    environment, because the secrets go through the same parsing. Secrets are read when the source
    is called, once ``current_state`` holds what the sources above it set. The names fetched for a
    field follow the public alias and prefix rules rather than the parser's private field resolution.
    """

    def _load_env_vars(self) -> Mapping[str, str | None]:
        return {}

    def __call__(self) -> dict[str, Any]:
        api_key = _env("MISTRAL_API_KEY")
        enabled = (_env("MISTRAL_SECRETS_ENABLED") or "").lower() in {"1", "true", "yes", "on"}
        if api_key is None or not enabled:
            return {}
        already_set = {key.lower() for key in self.current_state}
        unset: set[str] = set()
        prefix_target = getattr(self, "env_prefix_target", "variable")
        for field_name, field in self.settings_cls.model_fields.items():
            if not any(isinstance(marker, WorkspaceSecret) for marker in field.metadata):
                continue
            names = _field_env_names(field_name, field, self.config, self.env_prefix, prefix_target)
            if not {key.lower() for key in names} & already_set:
                unset |= set(names.values())
        if not self.case_sensitive:
            unset = {name.lower() for name in unset}
        if not unset:
            return {}
        api_url = (_env("MISTRAL_BASE_URL") or _PUBLIC_API).rstrip("/")
        self.env_vars = _workspace(api_url, api_key).read(unset)
        return super().__call__()
