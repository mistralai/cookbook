"""A second stop signal, which Ctrl+C or `mistral apps dev` plus the dev entrypoint send, leaves the
teardown the first one started to finish."""

import asyncio
import os
import signal

import pytest
from mistralai_capabilities.workflows import client as workflows_client
from worker.entrypoints import worker as entrypoint


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(signal.SIGINT, signal.SIGINT, id="Ctrl+C, then dev"),
        pytest.param(signal.SIGTERM, signal.SIGINT, id="mistral apps dev, then dev"),
        pytest.param(signal.SIGINT, signal.SIGTERM, id="SIGINT, then SIGTERM"),
    ],
)
async def test_a_second_stop_signal_does_not_cut_the_teardown_short(
    monkeypatch: pytest.MonkeyPatch, first: signal.Signals, second: signal.Signals
) -> None:
    started = asyncio.Event()
    tearing_down = asyncio.Event()
    torn_down: list[bool] = []

    async def _serve_forever(_classes: list[type]) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            tearing_down.set()
            # The drain: a second cancel would raise here.
            await asyncio.sleep(0.2)
            torn_down.append(True)

    monkeypatch.setattr(entrypoint, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(
        entrypoint.workflows, "discover_all_workflows_in_package", lambda _package: []
    )
    monkeypatch.setattr(workflows_client, "run_worker", _serve_forever)

    loop = asyncio.get_running_loop()
    stop_signals = (signal.SIGTERM, signal.SIGINT)
    # `remove_signal_handler` resets to the default, not to what the test runner had installed.
    previous = {
        stop_signal: signal.getsignal(stop_signal) for stop_signal in stop_signals
    }
    served = asyncio.create_task(entrypoint.main())
    try:
        await started.wait()
        os.kill(os.getpid(), first)
        await tearing_down.wait()
        os.kill(os.getpid(), second)
        await asyncio.wait_for(served, timeout=5)
    finally:
        for stop_signal in stop_signals:
            loop.remove_signal_handler(stop_signal)
            signal.signal(stop_signal, previous[stop_signal])
    assert torn_down == [True]
