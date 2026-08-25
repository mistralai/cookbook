"""Reusable per-``example_id`` cancellation registry for HTTP/SDK providers.

When the runner's per-file timeout fires, ``ThreadPoolExecutor.shutdown(wait=False)``
only releases the calling thread - any in-flight HTTP request the worker
thread spawned (httpx polling, requests session, LlamaCloud SDK call) keeps
running, and the next retry attempt sends a duplicate request to the hosted API.
Closing the underlying client breaks the provider's polling loop on its
next iteration so the worker thread unwinds with a transient error and the
retry loop can submit a fresh request without piling on parallel duplicates.

Important caveat - what closing a client does and does not abort:

* It DOES break a polling loop that calls ``client.get(...)`` repeatedly
  on a long-running job. The next call after ``close()`` raises immediately
  (httpx: ``RuntimeError: Cannot send a request, as the client has been closed.``;
  requests: ``ConnectionError`` on the next ``session.get``).

* It does NOT interrupt a thread already blocked inside a single socket
  read on another thread - Python threads are not OS-cancellable, and
  closing the client object only marks it closed; the kernel ``recv`` call
  finishes only when the server responds or the read timeout expires.

For the bench bug - duplicate requests to the hosted API during per-file timeout
retries - the polling-loop case is the one that matters. Long-running
parse / extract jobs are polled in tight loops; closing the client makes
the next poll raise within milliseconds. Per-request read timeouts on the
underlying client cap the worst-case stalled-read tail.

This module provides a tiny helper that:
  * registers a closeable handle (httpx.Client, requests.Session,
    llama_cloud.LlamaCloud, ...) keyed by ``example_id``;
  * exposes a ``cancel(example_id)`` that pops the handle and calls
    ``.close()`` (best-effort - providers swallow secondary errors so the
    cancel path can never break the runner).

Each provider holds one ``CancellableClientRegistry`` instance, registers
its client at the start of a request, and unregisters in a ``finally``.
The registry is thread-safe so concurrent ``run_inference`` calls (one per
``example_id``) do not collide.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class _Closeable(Protocol):
    """Anything with a no-arg ``close()``: ``httpx.Client``, ``requests.Session``,
    ``llama_cloud.LlamaCloud`` (closes its underlying ``httpx.Client``), ..."""

    def close(self) -> None: ...


class CancellableClientRegistry:
    """Thread-safe per-``example_id`` mapping of in-flight HTTP/SDK clients.

    Providers should:

        # In __init__:
        self._inflight = CancellableClientRegistry(provider_name="...")

        # At the start of run_inference (after the client is built):
        self._inflight.register(request.example_id, client)
        try:
            ...
        finally:
            self._inflight.unregister(request.example_id, client)

        # In cancel(example_id):
        return self._inflight.cancel(example_id)

    The registry never raises from ``cancel`` - a broken cancel must not
    break the runner's retry loop.
    """

    def __init__(self, *, provider_name: str) -> None:
        self._provider_name = provider_name
        self._lock = threading.Lock()
        self._inflight: dict[str, _Closeable] = {}

    def register(self, example_id: str, client: _Closeable) -> None:
        """Track ``client`` so a later ``cancel(example_id)`` can close it.

        If the slot is already occupied (e.g. because the previous attempt's
        cleanup raced the next attempt's submit), the new client wins - the
        old one was either already cancelled or about to be unregistered.
        """
        with self._lock:
            self._inflight[example_id] = client

    def unregister(self, example_id: str, client: _Closeable) -> None:
        """Remove ``client`` from the registry if it is the live entry.

        Compares by identity so we never clobber a registration from a
        concurrent retry attempt. Idempotent - safe to call from a
        ``finally`` even when ``cancel`` already popped the entry.
        """
        with self._lock:
            current = self._inflight.get(example_id)
            if current is client:
                self._inflight.pop(example_id, None)

    def cancel(self, example_id: str) -> bool:
        """Pop and close any registered client for ``example_id``.

        :return: True if a matching client was found and ``close()`` was
            attempted (regardless of whether close itself succeeded), False
            if no client was registered.
        """
        with self._lock:
            client = self._inflight.pop(example_id, None)
        if client is None:
            return False

        logger.info(
            "%s.cancel: closing in-flight client for example_id=%s",
            self._provider_name,
            example_id,
        )
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001 - cancel must never raise
            # Closing a client mid-request can surface various provider-
            # specific exceptions (httpx connection state, broken pipe,
            # SDK wrappers raising their own types). None of them should
            # break the runner's retry loop.
            logger.debug(
                "%s.cancel: client.close() raised for example_id=%s: %s",
                self._provider_name,
                example_id,
                exc,
            )
        return True
