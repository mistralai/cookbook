"""Run a model-authored extractor script in an isolated subprocess.

Trust model: the script comes from an LLM whose prompt includes arbitrary
parsed-document content (an untrusted-input injection vector), so it must not
run in the bench process. Isolation here is **process-level**: a fresh
interpreter, a scrubbed environment (no API keys), and a hard timeout (a
generated infinite loop must not wedge a runner worker thread). Residual risk,
accepted and documented: the child still runs as the same OS user with
filesystem and network access — container/seccomp isolation is out of scope.

The boundary is JSON-in → JSON-out: the parent serializes the document's raw
pages (``[{page_num, text}]`` — exactly what ``Document.__init__`` takes), the
shim rebuilds the ``Document`` and runs the script, and the output comes back
as JSON (which the gate requires anyway — ``extracted_data`` must be JSON).

The shim loads the value modules (``common``/``table_parser``/``document``) by
file path under their bare top-level names, so the generated script's
``from common import money`` / ``from table_parser import Table`` resolve to
the same module objects the ``Document`` was built with (``isinstance`` checks
hold) — without importing the heavy ``extract_bench`` package in the child
(~1.6 s vs ~0.03 s startup).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_PKG_DIR = str(Path(__file__).resolve().parent)

# The child-side shim. Placeholders (__PKG_DIR__/__FUNCTION_NAME__) are replaced
# with repr()'d values at write time (plain replace — no brace escaping games).
_RUNNER = '''\
"""Sandbox shim: rebuild the Document and run the generated extractor."""
import importlib.util
import json
import sys
import traceback

_PKG_DIR = __PKG_DIR__
_FUNCTION_NAME = __FUNCTION_NAME__
_DOC_MODULE = __DOC_MODULE__

# Sentinel: the interpreter came up and the shim is running. Its absence after
# a nonzero exit tells the parent the failure was environmental (loader/env),
# not the generated script's fault.
open("_started", "w").close()


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _PKG_DIR + "/" + name + ".py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register BEFORE exec so document's fallback import resolves
    spec.loader.exec_module(mod)
    return mod


def main():
    _load("common")
    if _DOC_MODULE == "document":
        _load("table_parser")  # before document: its top-level import fallback needs it
    document_mod = _load(_DOC_MODULE)  # every doc module exposes `Document`
    with open("_pages.json") as f:
        document = document_mod.Document(json.load(f))
    import _script
    fn = getattr(_script, _FUNCTION_NAME, None)
    if not callable(fn):
        print("script does not define " + _FUNCTION_NAME, file=sys.stderr)
        return 2
    out = fn(document)
    try:
        payload = json.dumps(out)
    except (TypeError, ValueError) as e:
        print("the returned output is not JSON-serializable: " + str(e), file=sys.stderr)
        return 3
    with open("_out.json", "w") as f:
        f.write(payload)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
'''


# Loader-critical, non-secret variables the child interpreter may need just to
# START. Notably: GitHub setup-python's Linux toolcache builds are
# shared-linked — the `python` binary locates libpythonX.Y.so via
# LD_LIBRARY_PATH, so dropping it makes every sandbox run die at exec on CI
# runners (macOS framework builds resolve their dylib by absolute path, which
# is why a PATH-only scrub worked locally). None of these carry secrets.
_ENV_ALLOWLIST = (
    "PATH",
    "LD_LIBRARY_PATH",  # Linux: shared-linked CPython finds libpython
    "DYLD_LIBRARY_PATH",  # macOS parity
    "DYLD_FALLBACK_LIBRARY_PATH",
    "PYTHONHOME",  # relocated toolchain layouts
    "TMPDIR",
    "LANG",
    "LC_ALL",
)


def _scrubbed_env() -> dict[str, str]:
    """Minimal child environment: interpreter-loader essentials only (see
    ``_ENV_ALLOWLIST``) — no API keys or other secrets."""
    return {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}


def run_extractor_subprocess(
    script: str,
    pages: list[dict[str, Any]],
    *,
    function_name: str = "extract_direct",
    timeout_s: float = 60.0,
    document_module: str = "document",
) -> tuple[bool, str, Any]:
    """Run ``script``'s ``function_name`` over a Document rebuilt from ``pages``
    in an isolated subprocess. ``document_module`` names the sibling module
    whose ``Document`` class rebuilds ``pages`` (``document`` for markdown-page
    documents, ``items_document`` for items-based ones — the pages list must be
    the matching ``to_pages()`` shape). Returns ``(ok, error_message, output)``;
    the error message is phrased for the agent (it is fed back as gate
    feedback)."""
    runner_src = (
        _RUNNER.replace("__PKG_DIR__", repr(_PKG_DIR))
        .replace("__FUNCTION_NAME__", repr(function_name))
        .replace("__DOC_MODULE__", repr(document_module))
    )
    with tempfile.TemporaryDirectory(prefix="table_codegen_sandbox_") as td:
        tmp = Path(td)
        (tmp / "_script.py").write_text(script)
        (tmp / "_pages.json").write_text(json.dumps(pages))
        runner = tmp / "_runner.py"
        runner.write_text(runner_src)
        try:
            proc = subprocess.run(
                [sys.executable, str(runner)],
                cwd=str(tmp),
                env=_scrubbed_env(),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return (
                False,
                f"your script did not finish within {timeout_s:.0f}s — likely an infinite loop or "
                "unbounded scan; make it terminate over the real document.",
                None,
            )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
            if not (tmp / "_started").exists():
                # The interpreter never came up (loader/env breakage, missing
                # libpython, …): an infrastructure error, not a script error.
                # Raise instead of returning gate feedback — otherwise the
                # model burns max_turns "fixing" a script that never ran.
                raise RuntimeError(f"sandbox interpreter failed to start (not a script error): {err[-800:]}")
            return False, err[-1500:], None
        out_file = tmp / "_out.json"
        if not out_file.exists():
            return False, "the script run produced no output.", None
        return True, "", json.loads(out_file.read_text())
