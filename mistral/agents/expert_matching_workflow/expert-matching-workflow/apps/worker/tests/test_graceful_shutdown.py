"""SIGTERM stops the worker through the SDK's own teardown, not by killing the process."""

import asyncio
import contextlib
import os
import signal

import pytest
from mistralai_capabilities.workflows import client as workflows_client
from worker.entrypoints import worker as entrypoint


async def test_sigterm_cancels_the_worker_so_its_teardown_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    torn_down: list[bool] = []

    async def _serve_forever(_classes: list[type]) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            torn_down.append(True)

    monkeypatch.setattr(entrypoint, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(entrypoint.workflows, "discover_all_workflows_in_package", lambda _package: [])
    monkeypatch.setattr(workflows_client, "run_worker", _serve_forever)

    served = asyncio.create_task(entrypoint.main())
    ready = asyncio.create_task(started.wait())
    # main() failing before run_worker never sets `started`: fail with its error, don't hang.
    await asyncio.wait({served, ready}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
    ready.cancel()
    if served.done():
        served.result()
    if not started.is_set():
        served.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await served
        pytest.fail("main() did not reach run_worker")
    os.kill(os.getpid(), signal.SIGTERM)

    await asyncio.wait_for(served, timeout=5)
    assert torn_down == [True]
    asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)
