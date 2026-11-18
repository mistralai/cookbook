"""Discover every workflow class in the `worker.workflows` package and serve them.

Registers no schedules: those are declared in the ``schedules`` command and applied by that
deployment init step, so a cadence change needs no worker restart and replicas cannot register
conflicting copies.
"""

import asyncio
import signal

import structlog
from env.logging import env
from mistralai import workflows
from mistralai_capabilities.workflows import client as workflows_client
from utils.logging import configure_logging

logger = structlog.get_logger("workflows")


async def main() -> None:
    configure_logging(log_level=env.log_level, log_format=env.log_format)
    workflow_classes = workflows.discover_all_workflows_in_package("worker.workflows")
    names = [workflows.get_workflow_definition(c).name for c in workflow_classes]
    logger.info("workflow discovery complete", total=len(names), workflows=names)

    # Docker, Kubernetes and Koyeb stop a container with SIGTERM, which Python does not handle, so
    # the process would die mid-activity. The SDK tears the worker down (drain, health server,
    # telemetry flush) only when its task is cancelled, which is what SIGTERM and SIGINT are turned
    # into here. Stops often come in pairs: Ctrl+C and `mistral apps dev` signal the whole process
    # group, then the dev entrypoint stops the worker as well. Only the first cancels; another cancel,
    # or the KeyboardInterrupt asyncio makes of a second SIGINT, would cut the teardown short.
    served = asyncio.current_task()
    stopped_by: signal.Signals | None = None

    def _stop(stop_signal: signal.Signals) -> None:
        nonlocal stopped_by
        if stopped_by is not None:
            return
        stopped_by = stop_signal
        if served is not None:
            served.cancel()

    loop = asyncio.get_running_loop()
    for stop_signal in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(stop_signal, _stop, stop_signal)
    try:
        await workflows_client.run_worker(workflow_classes)
    except asyncio.CancelledError:
        if stopped_by is None:
            raise
        logger.info("worker stopped", signal=stopped_by.name)


if __name__ == "__main__":
    asyncio.run(main())
