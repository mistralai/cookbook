"""The worker's SDKs send through two HTTP libraries: `mistralai` through `httpx2`, `mistralai-workflows`
and any plain client a capability builds through `httpx`. A test stands in for both here, the one place
that changes when either SDK moves its client boundary.
"""

from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import httpx2
import pytest


def patch_http_send(monkeypatch: pytest.MonkeyPatch, send: Callable[..., Awaitable[Any]]) -> None:
    """Answer every request on any `httpx` or `httpx2` async client with `send`: nothing leaves the process."""
    for client in (httpx.AsyncClient, httpx2.AsyncClient):
        monkeypatch.setattr(client, "send", send)
