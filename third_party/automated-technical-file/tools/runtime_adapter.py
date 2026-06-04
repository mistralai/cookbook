#!/usr/bin/env python3
"""
ATF Runtime Adapter — hosted Mistral plus local model invocation.

This module is the stable interface layer between callers (atf_qa.py, future
CLI tools) and the local model runtime.  Callers import this module and call
``query(prompt)``; they never need to know whether the backend is hosted
Mistral, local Ministral, aichat, Ollama, or corpus-only fallback.

Public interface
----------------
    query(prompt, model=None, timeout=120) -> str
    list_models() -> list[str]

CLI usage
---------
    python3 ATF/tools/runtime_adapter.py --check
    python3 ATF/tools/runtime_adapter.py --list-models
    python3 ATF/tools/runtime_adapter.py "Your prompt here"
    python3 ATF/tools/runtime_adapter.py --backend local --model ministral "Explain X."
    echo "Hello" | python3 ATF/tools/runtime_adapter.py --stdin

Machine / runtime prerequisites
--------------------------------
    Hosted Mistral requires:
        MISTRAL_API_KEY=<key>            # read from env only
    Local Ministral / Ollama requires:
        ollama serve                     # starts the daemon on :11434
    At least one of these models must be pulled (pull once, reuse forever):
        ollama pull ministral-3b-latest                              # preferred
        ollama pull MichelRosselli/apertus:8b-instruct-2509-q4_k_m   # generic fallback
        ollama pull gemma4:e4b                                         # alt
        ollama pull gemma:latest                                       # baseline
    Python >= 3.9, standard library only — no pip install required.
    Optional: aichat in PATH (secondary fallback if Ollama is unreachable).

Model selection
---------------
    Default fallback chain:
        1. Mistral hosted   — mistral-medium-2604, then mistral-large-latest
        2. Ministral local  — local Ministral 3B tags via Ollama
        3. aichat           — subprocess fallback
        4. Ollama generic   — Apertus/Gemma/first pulled model
        5. corpus-only      — callers catch RuntimeError and answer from corpus
    Set BACKEND or --backend to auto, hosted, local, aichat, ollama, or
    corpus-only.  Set OLLAMA_HOST to override localhost:11434.

Context handling
----------------
    Prompts are passed verbatim to the model.  Callers (e.g. atf_qa.py) are
    responsible for injecting corpus context before calling query().

Output conventions
------------------
    Returns the model's response text, stripped of surrounding whitespace.
    Raises RuntimeError if no backend is reachable or the response is empty.
    Never returns an empty string; always raises instead.

Failure modes
-------------
    Hosted unavailable      -> falls through to local Ministral unless forced
    Local Ministral missing -> falls through to aichat / generic Ollama
    All model backends miss -> RuntimeError so caller can use corpus-only
    Model not found         -> warning printed; auto-selection used instead
    Empty model response    -> RuntimeError (not silent empty string)
    Timeout                 -> RuntimeError wrapping TimeoutExpired
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from typing import List, Optional

# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MISTRAL_API_BASE: str = os.environ.get("MISTRAL_API_BASE", "https://api.mistral.ai")
DEFAULT_TIMEOUT: int = 120  # seconds

# Backend values accepted by BACKEND env var and --backend.
BACKEND_ALIASES = {
    "auto": "auto",
    "hosted": "hosted",
    "mistral": "hosted",
    "mistral-hosted": "hosted",
    "local": "local",
    "ministral": "local",
    "ministral-local": "local",
    "aichat": "aichat",
    "ollama": "ollama",
    "corpus": "corpus-only",
    "corpus-only": "corpus-only",
}

# Model IDs from the Mistral docs / CB-01 contract.
MISTRAL_HOSTED_MODELS: List[str] = ["mistral-medium-2604", "mistral-large-latest"]

# Substring keywords in priority order — first match in available Ollama models wins.
MINISTRAL_MODEL_PRIORITY: List[str] = [
    "ministral-3b-latest",
    "ministral-3b-2512",
    "ministral:3b",
    "ministral-3b",
    "ministral",
]
GENERIC_OLLAMA_MODEL_PRIORITY: List[str] = ["apertus", "gemma4", "gemma"]
MODEL_PRIORITY: List[str] = MINISTRAL_MODEL_PRIORITY + GENERIC_OLLAMA_MODEL_PRIORITY


def _normalize_backend(value: Optional[str] = None) -> str:
    """Return a canonical backend name from env/CLI input."""
    raw = (value or os.environ.get("BACKEND") or "auto").strip().lower()
    backend = BACKEND_ALIASES.get(raw)
    if not backend:
        raise RuntimeError(
            f"Unsupported BACKEND value '{raw}'. "
            f"Use one of: {', '.join(sorted(BACKEND_ALIASES))}."
        )
    return backend


def _backend_plan(backend: Optional[str] = None) -> List[str]:
    """Return the ordered backends to try for the requested mode."""
    selected = _normalize_backend(backend)
    plans = {
        "auto": ["hosted", "local", "aichat", "ollama"],
        "hosted": ["hosted", "local", "aichat", "ollama"],
        "local": ["local", "aichat", "ollama"],
        "aichat": ["aichat", "ollama"],
        "ollama": ["ollama"],
        "corpus-only": [],
    }
    return plans[selected]


def _backend_label(model_name: str) -> str:
    """Return the public backend label for a query_with_model model marker."""
    if model_name.startswith("mistral-hosted:"):
        return "mistral-hosted"
    if model_name.startswith("ministral-local:"):
        return "ministral-local"
    if model_name == "aichat":
        return "aichat"
    return "ollama"


# ---------------------------------------------------------------------------
# Mistral hosted backend (highest priority)
# ---------------------------------------------------------------------------

def _select_hosted_model(hint: Optional[str] = None) -> str:
    """Select the hosted Mistral model, defaulting to Medium then Large."""
    candidates = []
    env_model = os.environ.get("MISTRAL_MODEL")
    if env_model:
        candidates.append(env_model)
    candidates.extend(MISTRAL_HOSTED_MODELS)

    deduped: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)

    if hint:
        lowered = hint.lower()
        for model_name in deduped:
            if lowered in model_name.lower():
                return model_name
        if lowered.startswith("mistral-"):
            return hint

    return deduped[0]


def _extract_mistral_content(body: dict) -> str:
    """Extract assistant text from a Mistral chat completion response."""
    choices = body.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or ""
                if text:
                    parts.append(str(text))
            elif chunk:
                parts.append(str(chunk))
        return "".join(parts).strip()
    return str(content).strip() if content else ""


def _mistral_hosted_generate(prompt: str, model: Optional[str], timeout: int) -> tuple:
    """POST to Mistral chat completions and return ``(text, model_id)``."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set.")

    selected = _select_hosted_model(model)
    url = f"{MISTRAL_API_BASE.rstrip('/')}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": selected,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"Mistral hosted HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Mistral hosted connection error: {exc}") from exc

    text = _extract_mistral_content(body)
    if not text:
        raise RuntimeError(f"Mistral hosted returned an empty response for '{selected}'.")
    return text, selected


# ---------------------------------------------------------------------------
# Ollama HTTP backend (primary)
# ---------------------------------------------------------------------------

def _ollama_list_models() -> List[str]:
    """Return names of all models currently pulled in Ollama, or [] on error."""
    url = f"{OLLAMA_BASE}/api/tags"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _select_model(models: List[str], hint: Optional[str] = None) -> Optional[str]:
    """
    Choose the best model from *models*.

    If *hint* matches any model name (case-insensitive substring), that model
    is returned.  Otherwise, MODEL_PRIORITY is walked and the first keyword
    match is returned.  Falls back to models[0] if nothing matches.
    """
    if not models:
        return None

    if hint:
        for m in models:
            if hint.lower() in m.lower():
                return m
        print(
            f"[warn] runtime_adapter: requested model '{hint}' not found in Ollama; "
            "auto-selecting from priority list.",
            file=sys.stderr,
        )

    for keyword in MODEL_PRIORITY:
        for m in models:
            if keyword.lower() in m.lower():
                return m

    return models[0]


def _select_ollama_model(
    models: List[str],
    hint: Optional[str] = None,
    priority: Optional[List[str]] = None,
    fallback_first: bool = True,
) -> Optional[str]:
    """
    Choose an Ollama model using a specific priority list.

    When ``fallback_first`` is false, returns None instead of an unrelated
    first model.  That is how the local Ministral stage avoids stealing the
    generic Ollama fallback.
    """
    if not models:
        return None

    if hint:
        for m in models:
            if hint.lower() in m.lower():
                return m
        if hint.lower().startswith("ministral"):
            print(
                f"[warn] runtime_adapter: requested local model '{hint}' not found.",
                file=sys.stderr,
            )

    for keyword in priority or MODEL_PRIORITY:
        for m in models:
            if keyword.lower() in m.lower():
                return m

    return models[0] if fallback_first else None


def _ollama_generate(prompt: str, model: str, timeout: int) -> str:
    """
    POST to Ollama /api/generate (stream=false) and return the response text.
    Raises RuntimeError on HTTP error or empty output.
    """
    url = f"{OLLAMA_BASE}/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except (socket.timeout, TimeoutError) as exc:
        raise RuntimeError(f"Ollama timed out after {timeout}s.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc}") from exc

    text = body.get("response", "").strip()
    if not text:
        raise RuntimeError(
            f"Ollama returned an empty response for model '{model}'. "
            "The model may still be loading; try again in a few seconds."
        )
    return text


def _ministral_local_generate(prompt: str, model: Optional[str], timeout: int) -> tuple:
    """Generate with a local Ministral model served by Ollama."""
    available = _ollama_list_models()
    chosen = _select_ollama_model(
        available,
        hint=model,
        priority=MINISTRAL_MODEL_PRIORITY,
        fallback_first=False,
    )
    if not chosen:
        raise RuntimeError("No local Ministral model was found in Ollama.")
    return _ollama_generate(prompt, chosen, timeout), chosen


def _ollama_generic_generate(prompt: str, model: Optional[str], timeout: int) -> tuple:
    """Generate with the generic Ollama fallback model list."""
    available = _ollama_list_models()
    chosen = _select_ollama_model(
        available,
        hint=model,
        priority=GENERIC_OLLAMA_MODEL_PRIORITY,
        fallback_first=True,
    )
    if not chosen:
        raise RuntimeError("No Ollama model was found.")
    return _ollama_generate(prompt, chosen, timeout), chosen


# ---------------------------------------------------------------------------
# aichat subprocess backend (secondary fallback)
# ---------------------------------------------------------------------------

def _aichat_query(prompt: str, model: Optional[str], timeout: int) -> Optional[str]:
    """
    Send *prompt* to aichat via subprocess.  Returns output text or None.
    Used only when Ollama is unreachable.
    """
    cmd = ["aichat"]
    if model:
        cmd += ["-m", model]
    cmd.append("-")  # read prompt from stdin

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        if result.stderr.strip():
            print(
                f"[warn] runtime_adapter: aichat stderr: {result.stderr.strip()[:200]}",
                file=sys.stderr,
            )
    except FileNotFoundError:
        pass  # aichat not installed — silent
    except subprocess.TimeoutExpired:
        print("[warn] runtime_adapter: aichat timed out.", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_models() -> List[str]:
    """Return hosted Mistral IDs plus locally available Ollama models."""
    models = []
    if os.environ.get("MISTRAL_API_KEY"):
        models.extend([f"mistral-hosted:{m}" for m in MISTRAL_HOSTED_MODELS])
    models.extend(_ollama_list_models())
    return models


def generate(
    prompt: str,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    backend: Optional[str] = None,
) -> str:
    """
    Alias for :func:`query`.  Provided for callers that expect the ATF-8
    ticket-spec interface name (``generate``) rather than ``query``.
    """
    return query(prompt, model=model, timeout=timeout, backend=backend)


def query(
    prompt: str,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    backend: Optional[str] = None,
) -> str:
    """
    Invoke a local model and return its response.

    Parameters
    ----------
    prompt:
        The full prompt text.  Callers are responsible for injecting context.
    model:
        Optional model name or substring hint (e.g. ``"mistral-large"``,
        ``"ministral"``, ``"apertus"``, ``"gemma:latest"``).
    timeout:
        Seconds to wait for the model response before raising RuntimeError.

    Returns
    -------
    str
        Response text, stripped of leading/trailing whitespace.

    Raises
    ------
    RuntimeError
        When no local model backend is reachable or the response is empty.
    """
    text, _ = query_with_model(prompt, model=model, timeout=timeout, backend=backend)
    return text


def query_with_model(
    prompt: str,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    backend: Optional[str] = None,
) -> tuple:
    """
    Like :func:`query` but returns ``(response_text, model_name)`` so callers
    can surface which model was actually used.

    ``model_name`` is prefixed for Mistral backends:
    ``mistral-hosted:<model-id>`` or ``ministral-local:<ollama-tag>``.
    Generic Ollama returns the full model tag and aichat returns ``"aichat"``.
    """
    errors = []
    for candidate in _backend_plan(backend):
        try:
            if candidate == "hosted":
                text, used_model = _mistral_hosted_generate(prompt, model, timeout)
                return text, f"mistral-hosted:{used_model}"

            if candidate == "local":
                text, used_model = _ministral_local_generate(prompt, model, timeout)
                return text, f"ministral-local:{used_model}"

            if candidate == "aichat":
                answer = _aichat_query(prompt, model, timeout)
                if answer:
                    return answer, "aichat"
                raise RuntimeError("aichat unavailable or empty.")

            if candidate == "ollama":
                text, used_model = _ollama_generic_generate(prompt, model, timeout)
                return text, used_model
        except RuntimeError as exc:
            errors.append(f"{candidate}: {exc}")
            print(f"[warn] runtime_adapter: {candidate} failed: {exc}", file=sys.stderr)

    raise RuntimeError(
        "No model backend is reachable; caller should use corpus-only fallback.\n"
        "  Hosted: set MISTRAL_API_KEY for Mistral Medium/Large.\n"
        "  Local: start Ollama and pull a Ministral 3B model.\n"
        "  Fallback: install/configure aichat or pull an Ollama fallback model.\n"
        f"  BACKEND={_normalize_backend(backend)} OLLAMA_HOST={OLLAMA_BASE}\n"
        f"  Attempts: {'; '.join(errors) if errors else 'corpus-only requested'}"
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _cmd_check() -> None:
    """Print backend and model status."""
    try:
        backend = _normalize_backend()
    except RuntimeError as exc:
        print(f"BACKEND: invalid ({exc})")
        backend = "auto"

    key_status = "set" if os.environ.get("MISTRAL_API_KEY") else "not set"
    hosted_default = _select_hosted_model(None)
    print(f"BACKEND: {backend}")
    print(f"Mistral hosted: MISTRAL_API_KEY {key_status}")
    print(f"  Hosted priority: {hosted_default} -> mistral-large-latest")

    models = _ollama_list_models()
    if models:
        print(f"Ollama: reachable at {OLLAMA_BASE}")
        ministral = _select_ollama_model(
            models,
            priority=MINISTRAL_MODEL_PRIORITY,
            fallback_first=False,
        )
        chosen = ministral or _select_ollama_model(
            models,
            priority=GENERIC_OLLAMA_MODEL_PRIORITY,
            fallback_first=True,
        )
        print(f"  Models pulled ({len(models)}):")
        for m in models:
            if m == ministral:
                tag = "  <-- local Ministral"
            elif m == chosen:
                tag = "  <-- generic fallback"
            else:
                tag = ""
            print(f"    {m}{tag}")
    else:
        print(f"Ollama: NOT reachable at {OLLAMA_BASE}")
        print("  Run: ollama serve")

    # aichat probe
    try:
        result = subprocess.run(
            ["aichat", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"aichat: {result.stdout.strip()}")
        else:
            print("aichat: installed but --version failed")
    except FileNotFoundError:
        print("aichat: not in PATH (optional fallback unavailable)")


def _cmd_list_models() -> None:
    models = _ollama_list_models()
    print("Hosted Mistral models:")
    for m in MISTRAL_HOSTED_MODELS:
        marker = "[*]" if m == _select_hosted_model(None) else "   "
        print(f"  {marker} mistral-hosted:{m}")

    if not models:
        print(
            "No local models available.  Ollama may not be running.\n"
            "  Run: ollama serve"
        )
        return
    ministral = _select_ollama_model(
        models,
        priority=MINISTRAL_MODEL_PRIORITY,
        fallback_first=False,
    )
    chosen = ministral or _select_ollama_model(
        models,
        priority=GENERIC_OLLAMA_MODEL_PRIORITY,
        fallback_first=True,
    )
    print(f"Local Ollama models ({len(models)}) -- [*] = would be auto-selected:")
    for m in models:
        mark = "[*]" if m == chosen else "   "
        print(f"  {mark} {m}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ATF runtime adapter: invoke hosted Mistral, local Ministral, "
            "aichat, or Ollama."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
            Prerequisites (one-time setup):
              export MISTRAL_API_KEY=...       # hosted Mistral; env only
              ollama serve
              ollama pull ministral-3b-latest
              ollama pull MichelRosselli/apertus:8b-instruct-2509-q4_k_m
              ollama pull gemma:latest     # alternative

            Backend fallback priority:
              Mistral hosted -> Ministral local -> aichat -> Ollama -> corpus-only

            Environment:
              BACKEND         auto|hosted|local|aichat|ollama|corpus-only
              MISTRAL_API_KEY hosted Mistral API key
              OLLAMA_HOST     override Ollama endpoint (default: {OLLAMA_BASE})

            Examples:
              python3 ATF/tools/runtime_adapter.py --check
              python3 ATF/tools/runtime_adapter.py --list-models
              python3 ATF/tools/runtime_adapter.py "What is the Wall of Fame?"
              python3 ATF/tools/runtime_adapter.py --backend hosted --model mistral-large "Explain X."
              python3 ATF/tools/runtime_adapter.py --backend local --model ministral "Explain X."
              echo "Hello" | python3 ATF/tools/runtime_adapter.py --stdin
        """),
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt text (inline; omit if using --stdin)",
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        help="Model name or substring hint (e.g. 'mistral-large', 'ministral', 'apertus')",
    )
    parser.add_argument(
        "--backend",
        choices=sorted({"auto", "hosted", "local", "aichat", "ollama", "corpus-only"}),
        help="Override BACKEND for this invocation.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        metavar="SEC",
        help=f"Response timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Show backend / model status and exit",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all locally available models and exit",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read prompt from stdin (also triggered automatically when stdin is a pipe)",
    )

    args = parser.parse_args()

    if args.check:
        _cmd_check()
        return

    if args.list_models:
        _cmd_list_models()
        return

    # Resolve prompt: explicit flag, piped stdin, or positional arg
    prompt: Optional[str] = None
    if args.stdin:
        prompt = sys.stdin.read().strip()
    elif args.prompt:
        prompt = args.prompt
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    else:
        parser.print_help()
        sys.exit(0)

    if not prompt:
        print("Error: empty prompt.", file=sys.stderr)
        sys.exit(1)

    try:
        answer = query(prompt, model=args.model, timeout=args.timeout, backend=args.backend)
        print(answer)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
