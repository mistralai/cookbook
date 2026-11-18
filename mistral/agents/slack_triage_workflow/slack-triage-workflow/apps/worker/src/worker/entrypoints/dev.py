"""Serve the worker, restarting it when the code it serves changes."""

import os
import sys
from pathlib import Path

from mistralai_capabilities.workflows.dev import serve

ROOT = Path(__file__).resolve().parents[5]
WATCH = (ROOT / "apps/worker/src", ROOT / "packages/py")
# This interpreter is already the workspace venv, so the worker runs directly under it. A `uv run`
# in between would receive the SIGINT a reload sends instead of the worker, and leave it orphaned.
SERVE = (sys.executable, "-m", "worker.entrypoints.worker")


def main() -> None:
    # The worker runs from the app root wherever this is launched (the image starts it from apps/worker).
    os.chdir(ROOT)
    sys.exit(serve(SERVE, WATCH))


if __name__ == "__main__":
    main()
