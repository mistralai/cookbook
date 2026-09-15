"""Vendor the shared src/ modules into function_app/ for packaging, or remove them.

Azure Functions (and azd) package only the service folder, but function_app.py imports the
shared pipeline modules by bare name. So we copy them next to function_app.py before packaging
and remove them after. This is the OS-neutral replacement for the inline shell loop in
function_app/deploy.sh (azd runs hooks under sh on macOS/Linux and pwsh on Windows).

Usage:
  python scripts/vendor_src.py copy     # before packaging (azd prepackage hook)
  python scripts/vendor_src.py clean    # after deploy   (azd postdeploy hook)

The module list includes foundry_agents so the deployed Function can run the hosted-agent
extractor path (deploy.sh historically omitted it).
"""
from __future__ import annotations

import os
import shutil
import sys

MODULES = [
    "config",
    "observability",
    "ocr_client",
    "agents",
    "foundry_agents",
    "workflow",
    "pipeline",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DEST = os.path.join(ROOT, "function_app")


def copy() -> None:
    for m in MODULES:
        shutil.copyfile(os.path.join(SRC, f"{m}.py"), os.path.join(DEST, f"{m}.py"))
    print(f"vendored {len(MODULES)} modules into function_app/")


def clean() -> None:
    for m in MODULES:
        path = os.path.join(DEST, f"{m}.py")
        if os.path.exists(path):
            os.remove(path)
    print("removed vendored modules from function_app/")


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("copy", "clean"):
        print("usage: vendor_src.py [copy|clean]", file=sys.stderr)
        return 2
    (copy if argv[1] == "copy" else clean)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
