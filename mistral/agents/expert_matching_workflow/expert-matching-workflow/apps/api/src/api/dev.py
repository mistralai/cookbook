"""Serve the API, reloading it when the code it serves changes."""

import os
from pathlib import Path

import uvicorn

# <app root>/apps/api/src/api/dev.py
ROOT = Path(__file__).resolve().parents[4]


def reload_dirs() -> list[str]:
    """The API's own sources and every workspace package's, never a `tests/` tree."""
    return [
        str(ROOT / "apps/api/src"),
        *sorted(str(src) for src in ROOT.glob("packages/py/*/src")),
    ]


def main() -> None:
    uvicorn.run(
        "api.main:app",
        # Loopback unless told otherwise: on the host no gateway strips forged identity headers.
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "3000")),
        reload=True,
        reload_dirs=reload_dirs(),
        # An open chat stream otherwise keeps the old worker alive, and the API unanswered, forever.
        timeout_graceful_shutdown=3,
    )


if __name__ == "__main__":
    main()
