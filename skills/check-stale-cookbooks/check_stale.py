#!/usr/bin/env python3
"""
check_stale.py — Scan Mistral AI cookbooks for outdated content.

Checks against:
  - /v1/models API (valid model IDs)
  - README from mistralai/client-python (current Python SDK patterns)
  - README from mistralai/client-ts (current TypeScript SDK patterns)
  - Selected pages from mistralai/platform-docs-public (deprecation notices)

Usage:
    python check_stale.py [options]

Options:
    --dir DIR        Directory to scan (default: mistral)
    --file FILE      Scan a single file instead of a directory
    --output FILE    Write JSON report to this file path
    --format FMT     Output format: markdown (default) or json
    --no-fetch       Skip fetching reference data (offline/pattern-only mode)
    --use-llm        Use Mistral API for deeper semantic analysis
    --no-llm         Disable LLM analysis
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ─── Dependency checks ─────────────────────────────────────────────────────────

def _require(package: str, install: str) -> None:
    try:
        __import__(package)
    except ImportError:
        print(f"Error: '{package}' not installed. Run: pip install {install}", file=sys.stderr)
        sys.exit(1)


_require("requests", "requests")
_require("nbformat", "nbformat")

import requests  # noqa: E402
import nbformat  # noqa: E402


# ─── Reference URLs ────────────────────────────────────────────────────────────

PYTHON_README_RAW_URL = (
    "https://raw.githubusercontent.com/mistralai/client-python/main/README.md"
)
TS_README_RAW_URL = (
    "https://raw.githubusercontent.com/mistralai/client-ts/main/README.md"
)
PLATFORM_DOCS_RAW_BASE = (
    "https://raw.githubusercontent.com/mistralai/platform-docs-public/main"
)

# Human-readable links used in issue bodies
OPENAPI_URL = "https://github.com/mistralai/platform-docs-public/blob/main/openapi.yaml"
PYTHON_SDK_URL = "https://github.com/mistralai/client-python/blob/main/README.md"
TS_SDK_URL = "https://github.com/mistralai/client-ts/blob/main/README.md"
DOCS_FINE_TUNING_URL = "https://docs.mistral.ai/capabilities/fine-tuning/"
DOCS_MODELS_URL = "https://docs.mistral.ai/getting-started/models/models_overview/"


# ─── Static deprecated patterns ────────────────────────────────────────────────
# Each entry: pattern (regex), type, detail, reference_url,
#             reference_search (optional text to find in ref doc for #L anchor)
# Add # stale-check: ignore on any source line to suppress a match.

DEPRECATED_PATTERNS: list[dict] = [
    # ── Python SDK v0 imports ──────────────────────────────────────────────────
    {
        "pattern": r"from mistralai\.client import MistralClient",
        "type": "deprecated_import",
        "detail": (
            "Deprecated v0 Python SDK import. "
            "Use `from mistralai import Mistral` instead."
        ),
        "reference_url": PYTHON_SDK_URL,
        "reference_search": "from mistralai import Mistral",
    },
    {
        "pattern": r"from mistralai import MistralClient",
        "type": "deprecated_import",
        "detail": (
            "Deprecated v0 Python SDK import. "
            "Use `from mistralai import Mistral` instead."
        ),
        "reference_url": PYTHON_SDK_URL,
        "reference_search": "from mistralai import Mistral",
    },
    {
        "pattern": r"from mistralai\.models",
        "type": "deprecated_import",
        "detail": (
            "Deprecated model class import from `mistralai.models`. "
            "These classes were removed in SDK v1. "
            "Use plain dicts — `{\"role\": \"user\", \"content\": \"...\"}` — or new SDK types."
        ),
        "reference_url": PYTHON_SDK_URL,
        "reference_search": "from mistralai import Mistral",
    },
    # ── Python SDK v0 client class ─────────────────────────────────────────────
    {
        "pattern": r"\bMistralClient\s*\(",
        "type": "deprecated_class",
        "detail": (
            "Deprecated `MistralClient` class (SDK v0). "
            "Use `Mistral(api_key=...)` from the `mistralai` package instead."
        ),
        "reference_url": PYTHON_SDK_URL,
        "reference_search": "Mistral(api_key",
    },
    {
        "pattern": r"\bChatMessage\s*\(",
        "type": "deprecated_class",
        "detail": (
            "Deprecated `ChatMessage` class removed in SDK v1. "
            "Use plain dicts: `{\"role\": \"user\", \"content\": \"...\"}`."
        ),
        "reference_url": PYTHON_SDK_URL,
        "reference_search": "role",
    },
    # ── Python SDK v0 method signatures ───────────────────────────────────────
    # Matches  client.chat(  but NOT  client.chat.complete(
    {
        "pattern": r"\.chat\s*\(\s*(?!.*\.complete)",
        "type": "deprecated_method",
        "detail": (
            "Deprecated `client.chat()` call (SDK v0). "
            "Use `client.chat.complete(...)` instead."
        ),
        "reference_url": PYTHON_SDK_URL,
        "reference_search": "chat.complete(",
    },
    {
        "pattern": r"\.embeddings\s*\(\s*(?!.*\.create)",
        "type": "deprecated_method",
        "detail": (
            "Deprecated `client.embeddings()` call (SDK v0). "
            "Use `client.embeddings.create(...)` instead."
        ),
        "reference_url": PYTHON_SDK_URL,
        "reference_search": "embeddings.create(",
    },
    # ── Fine-tuning API (deprecated) ───────────────────────────────────────────
    {
        "pattern": r"client\.fine_tuning\b",
        "type": "deprecated_api",
        "detail": (
            "The fine-tuning jobs API (`client.fine_tuning`) is deprecated. "
            "See the current fine-tuning documentation for the supported approach."
        ),
        "reference_url": DOCS_FINE_TUNING_URL,
    },
    {
        "pattern": r"/v1/fine[_-]tuning/",
        "type": "deprecated_api",
        "detail": (
            "The `/v1/fine_tuning/` REST endpoint is deprecated. "
            "See the current fine-tuning documentation for the supported approach."
        ),
        "reference_url": DOCS_FINE_TUNING_URL,
    },
    # ── Pinned old package versions ────────────────────────────────────────────
    {
        "pattern": r"pip install mistralai==0\.",
        "type": "outdated_version",
        "detail": (
            "Pinned to SDK v0. "
            "Update to the current version: `pip install mistralai` (or `uv add mistralai`)."
        ),
        "reference_url": PYTHON_SDK_URL,
        "reference_search": "pip install mistralai",
    },
    {
        "pattern": r"@mistralai/mistralai@0\.",
        "type": "outdated_version",
        "detail": (
            "Pinned to TypeScript SDK v0. "
            "Update to the current version: `npm add @mistralai/mistralai`."
        ),
        "reference_url": TS_SDK_URL,
        "reference_search": "npm add @mistralai/mistralai",
    },
    # ── Deprecated model names ─────────────────────────────────────────────────
    {
        "pattern": r"""['"](mistral-tiny)['"]""",
        "type": "deprecated_model",
        "detail": (
            "`mistral-tiny` is deprecated. "
            "Check the current model list for a replacement."
        ),
        "reference_url": DOCS_MODELS_URL,
    },
    {
        "pattern": r"""['"](mistral-medium)['"]""",
        "type": "deprecated_model",
        "detail": (
            "`mistral-medium` is deprecated. "
            "Use `mistral-medium-latest` or a current equivalent."
        ),
        "reference_url": DOCS_MODELS_URL,
    },
    {
        "pattern": r"""['"](open-mistral-7b)['"]""",
        "type": "verify_model",
        "detail": (
            "`open-mistral-7b` may be deprecated. "
            "Verify against the current model list."
        ),
        "reference_url": DOCS_MODELS_URL,
    },
    {
        "pattern": r"""['"](open-mixtral-8x7b)['"]""",
        "type": "verify_model",
        "detail": (
            "`open-mixtral-8x7b` may be deprecated. "
            "Verify against the current model list."
        ),
        "reference_url": DOCS_MODELS_URL,
    },
    {
        "pattern": r"""['"](open-mixtral-8x22b)['"]""",
        "type": "verify_model",
        "detail": (
            "`open-mixtral-8x22b` may be deprecated. "
            "Verify against the current model list."
        ),
        "reference_url": DOCS_MODELS_URL,
    },
    # ── Pinned model versions (prefer -latest aliases) ─────────────────────────
    {
        "pattern": r"""['"][a-z][a-z0-9\-]+-20\d\d-\d\d[a-z0-9\-]*['"]""",
        "type": "pinned_model_version",
        "detail": (
            "References a pinned dated model version (e.g. `-2309`, `-2402`). "
            "Consider using a `-latest` alias so cookbooks stay current automatically."
        ),
        "reference_url": DOCS_MODELS_URL,
    },
]


# ─── Fetch reference data ──────────────────────────────────────────────────────

def fetch_url(url: str, label: str) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        print(f"Warning: could not fetch {label}: {exc}", file=sys.stderr)
        return None


def fetch_reference_content() -> dict[str, str]:
    """Fetch all reference docs and return as a dict of label → text.

    Content is passed to the LLM so it can verify claims against actual source,
    and used to compute #L anchors for static pattern reference URLs.
    """
    content: dict[str, str] = {}

    print("Fetching Python SDK README ...", file=sys.stderr)
    python_readme = fetch_url(PYTHON_README_RAW_URL, "client-python README")
    if python_readme:
        content["python"] = python_readme

    print("Fetching TypeScript SDK README ...", file=sys.stderr)
    ts_readme = fetch_url(TS_README_RAW_URL, "client-ts README")
    if ts_readme:
        content["typescript"] = ts_readme

    # Fetch platform-docs-public pages that cover deprecations / API changes
    docs_pages = [
        ("fine_tuning", "docs/capabilities/fine-tuning.mdx"),
        ("fine_tuning_alt", "docs/api/fine_tuning.mdx"),
        ("changelog", "CHANGELOG.md"),
        ("deprecations", "docs/deprecations.md"),
        ("migration", "docs/migration.md"),
    ]
    for label, path in docs_pages:
        url = f"{PLATFORM_DOCS_RAW_BASE}/{path}"
        doc = fetch_url(url, f"platform-docs-public/{path}")
        if doc:
            content[label] = doc
            print(f"  Fetched platform-docs-public/{path}", file=sys.stderr)

    return content


def fetch_valid_models(api_key: Optional[str]) -> set[str]:
    """Fetch the current model list from the Mistral /v1/models API."""
    if not api_key:
        print("  No MISTRAL_API_KEY available — skipping dynamic model validation.",
              file=sys.stderr)
        return set()
    try:
        resp = requests.get(
            "https://api.mistral.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        ids = {m["id"] for m in data.get("data", []) if isinstance(m.get("id"), str)}
        print(f"  Found {len(ids)} valid model IDs from /v1/models.", file=sys.stderr)
        return ids
    except Exception as exc:
        print(f"Warning: could not fetch model list from /v1/models: {exc}", file=sys.stderr)
        return set()


# ─── Reference URL enrichment ──────────────────────────────────────────────────

def line_anchored_url(base_url: str, content: str, search_text: str) -> str:
    """Return a GitHub URL with a #L<n> anchor if search_text is found in content."""
    for i, line in enumerate(content.splitlines(), 1):
        if search_text in line:
            return f"{base_url}#L{i}"
    return base_url


def enrich_reference_url(issue: dict, ref_content: dict[str, str]) -> dict:
    """Try to add a #L anchor to the issue's reference_url using fetched content."""
    search = issue.get("reference_search", "")
    if not search:
        return issue

    ref_url = issue.get("reference_url", "")
    if "client-python" in ref_url and "python" in ref_content:
        enriched = line_anchored_url(PYTHON_SDK_URL, ref_content["python"], search)
        if enriched != PYTHON_SDK_URL:
            issue = {**issue, "reference_url": enriched}
    elif "client-ts" in ref_url and "typescript" in ref_content:
        enriched = line_anchored_url(TS_SDK_URL, ref_content["typescript"], search)
        if enriched != TS_SDK_URL:
            issue = {**issue, "reference_url": enriched}

    return issue


# ─── Code extraction ───────────────────────────────────────────────────────────

def extract_notebook_cells(path: Path) -> list[tuple[int, str]]:
    """Return (cell_number, source) for every code cell in a notebook."""
    try:
        nb = nbformat.read(str(path), as_version=4)
    except Exception as exc:
        print(f"Warning: could not parse notebook {path}: {exc}", file=sys.stderr)
        return []
    return [
        (i + 1, cell.source)
        for i, cell in enumerate(nb.cells)
        if cell.cell_type == "code" and cell.source.strip()
    ]


def extract_md_code_blocks(text: str) -> list[tuple[int, str, str]]:
    """Return (block_number, language, code) for fenced code blocks in markdown."""
    blocks = []
    for i, m in enumerate(re.finditer(r"```(\w+)?\n(.*?)```", text, re.DOTALL), 1):
        lang = m.group(1) or "unknown"
        code = m.group(2)
        if code.strip():
            blocks.append((i, lang, code))
    return blocks


# ─── Pattern checking ──────────────────────────────────────────────────────────

_CODE_LANGS = {"python", "typescript", "javascript", "bash", "shell", "sh", "unknown"}

# Matches model IDs in API call context: model="..." or "model": "..."
_MODEL_IN_CALL = re.compile(
    r"""(?:model\s*=\s*|"model"\s*:\s*)['"]([\w\-\.]+)['"]"""
)
_MODEL_KEYWORDS = ("mistral", "codestral", "pixtral", "mixtral", "ministral")


def check_patterns(
    text: str, location: str, ref_content: dict[str, str]
) -> list[dict]:
    """Run all static deprecated patterns against a block of text."""
    issues: list[dict] = []
    lines = text.splitlines()
    for info in DEPRECATED_PATTERNS:
        compiled = re.compile(info["pattern"])
        for lineno, line in enumerate(lines, 1):
            if "stale-check: ignore" in line:
                continue
            if compiled.search(line):
                issue = {
                    "type": info["type"],
                    "detail": info["detail"],
                    "location": f"{location}, line {lineno}" if location else f"line {lineno}",
                    "matched_line": line.strip()[:200],
                    "reference_url": info["reference_url"],
                    "reference_search": info.get("reference_search", ""),
                }
                issue = enrich_reference_url(issue, ref_content)
                # Don't expose reference_search in the output
                issue.pop("reference_search", None)
                issues.append(issue)
    return issues


def check_unknown_models(text: str, valid_models: set[str], location: str) -> list[dict]:
    """Flag model IDs used in API calls that aren't in the valid set."""
    if not valid_models:
        return []
    issues: list[dict] = []
    lines = text.splitlines()
    for lineno, line in enumerate(lines, 1):
        if "stale-check: ignore" in line:
            continue
        for m in _MODEL_IN_CALL.finditer(line):
            model_id = m.group(1)
            if not any(kw in model_id.lower() for kw in _MODEL_KEYWORDS):
                continue
            if model_id in valid_models:
                continue
            issues.append({
                "type": "unknown_model",
                "detail": (
                    f"`{model_id}` is not in the current model list from `/v1/models`. "
                    "It may be deprecated or renamed."
                ),
                "location": f"{location}, line {lineno}" if location else f"line {lineno}",
                "matched_line": line.strip()[:200],
                "reference_url": DOCS_MODELS_URL,
            })
    return issues


# ─── LLM-assisted analysis ─────────────────────────────────────────────────────

_LLM_SYSTEM = """\
You are a technical reviewer checking Mistral AI cookbook files for outdated content.
Your job is to find patterns that are genuinely deprecated or removed — not things that
look unfamiliar to you.

## Current SDK facts

Python SDK (v1+):
- Import: `from mistralai import Mistral`
- Client init: `client = Mistral(api_key=...)`
- Sync methods: `client.chat.complete(...)`, `client.embeddings.create(...)`,
  `client.agents.complete(...)`, `client.beta.connectors.create(...)`, etc.
- Async methods: append `_async` — e.g. `client.chat.complete_async(...)`,
  `client.beta.connectors.create_async(...)`, `client.beta.agents.list_async(...)`, etc.

  ⚠️  CRITICAL: `_async` suffix methods are VALID and NOT deprecated.
  Do NOT flag `create_async`, `list_async`, `start_async`, `delete_async`,
  `get_async`, `update_async`, `run_async`, `complete_async`, or any other
  `_async` variant as deprecated. They are the correct async API.

TypeScript SDK (v1+):
- Import: `import Mistral from "@mistralai/mistralai"` (ESM default export)
- Client init: `new Mistral({ apiKey: ... })`

## Known deprecated features

Flag these if you see them:
- `MistralClient(` class → use `Mistral(api_key=...)`
- `ChatMessage(` class → use plain dicts `{"role": "user", "content": "..."}`
- `client.chat(` without `.complete` → use `client.chat.complete(...)`
- `client.embeddings(` without `.create` → use `client.embeddings.create(...)`
- `from mistralai.models import ...` → removed in SDK v1
- `from mistralai.client import ...` → removed in SDK v1
- `client.fine_tuning` (fine-tuning jobs API) → deprecated
- `/v1/fine_tuning/` REST endpoint → deprecated
- `pip install mistralai==0.` → outdated SDK v0

## Rules for flagging

1. Only flag things you are CERTAIN are deprecated based on the provided reference content
   or the known-deprecated list above.
2. NEVER flag `_async` suffix methods. They are valid.
3. Do not fabricate reference URLs. Only use URLs from the list provided in the user message.
4. For each issue, include a `quote` field: copy the exact line from the provided reference
   content that proves the deprecation. Use empty string if no direct quote is available.
5. If you are uncertain, return [].

## Response format

Respond ONLY with a JSON array. Each element must have:
  type: string (snake_case label)
  detail: string (one sentence: what is wrong and what to use instead)
  reference_url: string (most specific URL from those provided, including a #L anchor if known)
  quote: string (exact text from the reference that confirms the deprecation, or "")

Return [] if you find nothing beyond what was already flagged.\
"""


def _build_ref_context(ref_content: dict[str, str]) -> str:
    """Build a reference context block to include in the LLM user message."""
    parts: list[str] = []

    if "python" in ref_content:
        # Include the first 3 000 chars of the Python README — enough to cover
        # the client init, basic method signatures, and any migration notes.
        excerpt = ref_content["python"][:3000]
        parts.append(f"Python SDK README (excerpt):\n```\n{excerpt}\n```")

    if "typescript" in ref_content:
        excerpt = ref_content["typescript"][:1500]
        parts.append(f"TypeScript SDK README (excerpt):\n```\n{excerpt}\n```")

    # Include any deprecation/fine-tuning docs we managed to fetch
    for label in ("fine_tuning", "fine_tuning_alt", "changelog", "deprecations", "migration"):
        if label in ref_content:
            excerpt = ref_content[label][:2000]
            parts.append(f"platform-docs-public/{label} (excerpt):\n```\n{excerpt}\n```")

    return "\n\n".join(parts)


def llm_analyze(
    file_path: str,
    content: str,
    known_issues: list[dict],
    api_key: str,
    ref_content: dict[str, str],
) -> list[dict]:
    """Call mistral-medium-latest to find issues that pattern matching missed."""
    known_summary = ""
    if known_issues:
        known_summary = "\n\nAlready flagged by pattern matching:\n" + "\n".join(
            f"- {i['type']}: {i['detail']}" for i in known_issues
        )

    ref_context = _build_ref_context(ref_content)

    user_msg = (
        f"File: {file_path}\n\n"
        f"Code content (truncated to 5 000 chars):\n```\n{content[:5000]}\n```"
        f"{known_summary}\n\n"
        f"Reference URLs:\n"
        f"- Python SDK README: {PYTHON_SDK_URL}\n"
        f"- TypeScript SDK README: {TS_SDK_URL}\n"
        f"- Mistral model list: {DOCS_MODELS_URL}\n"
        f"- Fine-tuning docs: {DOCS_FINE_TUNING_URL}\n\n"
    )

    if ref_context:
        user_msg += (
            f"Reference content (use this to verify and quote from):\n\n"
            f"{ref_context}\n\n"
        )

    user_msg += "List any additional stale patterns NOT already flagged above. Return [] if none."

    try:
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistral-medium-latest",
                "messages": [
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.1,
                "max_tokens": 1000,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"Warning: LLM call failed for {file_path}: {exc}", file=sys.stderr)
        return []

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    # Strip any accidental markdown fences
    raw = re.sub(r"^```(?:json)?\n?", "", raw).strip()
    raw = re.sub(r"\n?```$", "", raw).strip()

    # response_format: json_object may wrap the array in an object key
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            # Find the first list value
            result = next((v for v in parsed.values() if isinstance(v, list)), [])
        elif isinstance(parsed, list):
            result = parsed
        else:
            return []

        for item in result:
            item.setdefault("location", "LLM analysis (no specific line)")
            item.setdefault("quote", "")
            # Strip any _async false positives that slipped through
            detail = item.get("detail", "")
            if re.search(r"\b\w+_async\b", item.get("type", "") + detail):
                continue
        return result
    except json.JSONDecodeError as exc:
        print(f"Warning: could not parse LLM response for {file_path}: {exc}", file=sys.stderr)
        return []


# ─── File scanning ─────────────────────────────────────────────────────────────

def scan_file(
    path: Path,
    valid_models: set[str],
    use_llm: bool,
    api_key: Optional[str],
    ref_content: dict[str, str],
) -> Optional[dict]:
    """Scan a single file. Returns issue dict or None if file is clean."""
    suffix = path.suffix.lower()
    all_issues: list[dict] = []
    llm_content = ""

    if suffix == ".ipynb":
        for cell_idx, source in extract_notebook_cells(path):
            loc = f"Cell {cell_idx}"
            all_issues += check_patterns(source, loc, ref_content)
            all_issues += check_unknown_models(source, valid_models, loc)
            llm_content += f"\n# Cell {cell_idx}\n{source}\n"

    elif suffix == ".md":
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
            return None
        for block_idx, lang, code in extract_md_code_blocks(text):
            if lang not in _CODE_LANGS:
                continue
            loc = f"Code block {block_idx} ({lang})"
            all_issues += check_patterns(code, loc, ref_content)
            all_issues += check_unknown_models(code, valid_models, loc)
            llm_content += f"\n# Code block {block_idx} ({lang})\n{code}\n"

    elif suffix == ".py":
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
            return None
        all_issues += check_patterns(text, "", ref_content)
        all_issues += check_unknown_models(text, valid_models, "")
        llm_content = text

    else:
        return None  # unsupported extension

    # LLM pass — only runs if requested and there's code to analyze
    llm_issues: list[dict] = []
    if use_llm and api_key and llm_content.strip():
        llm_issues = llm_analyze(str(path), llm_content, all_issues, api_key, ref_content)

    combined = all_issues + llm_issues
    if not combined:
        return None

    return {"path": str(path), "issues": combined}


_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints", ".venv", "venv"}
_SCAN_EXTS = {".ipynb", ".md", ".py"}


def scan_directory(
    directory: Path,
    valid_models: set[str],
    use_llm: bool,
    api_key: Optional[str],
    ref_content: dict[str, str],
) -> list[dict]:
    """Recursively scan all cookbooks in a directory."""
    results: list[dict] = []
    for path in sorted(directory.rglob("*")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in _SCAN_EXTS or not path.is_file():
            continue
        print(f"  Scanning {path} ...", file=sys.stderr)
        result = scan_file(path, valid_models, use_llm, api_key, ref_content)
        if result:
            results.append(result)
    return results


# ─── Output formatting ─────────────────────────────────────────────────────────

def to_markdown(report: dict) -> str:
    files = report["files"]
    lines: list[str] = [
        "# Stale Cookbook Report",
        "",
        f"**Scanned at:** {report['scanned_at']}  ",
        f"**Files with issues:** {len(files)}  ",
        f"**Valid models found:** {report['valid_models_count']}  ",
        f"**References checked:** {', '.join(report['references_checked']) or 'none (offline mode)'}",
        "",
    ]
    if not files:
        lines.append("✅ No stale content found.")
        return "\n".join(lines)

    for file_info in files:
        lines += [f"---", "", f"## `{file_info['path']}`", ""]
        by_type: dict[str, list[dict]] = {}
        for issue in file_info["issues"]:
            by_type.setdefault(issue["type"], []).append(issue)
        for issue_type, issues in by_type.items():
            label = issue_type.replace("_", " ").title()
            lines.append(f"### {label}")
            for issue in issues:
                lines.append(f"- {issue['detail']}")
                if issue.get("location"):
                    lines.append(f"  - **Location:** `{issue['location']}`")
                if issue.get("matched_line"):
                    lines.append(f"  - **Found:** `{issue['matched_line']}`")
                if issue.get("quote"):
                    lines.append(f"  - **Reference quote:** _{issue['quote']}_")
                lines.append(f"  - **Reference:** [{issue['reference_url']}]({issue['reference_url']})")
            lines.append("")
    return "\n".join(lines)


# ─── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scan Mistral AI cookbooks for stale content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dir", default="mistral", metavar="DIR",
                   help="Directory to scan (default: mistral)")
    p.add_argument("--file", metavar="FILE",
                   help="Scan a single file instead of a directory")
    p.add_argument("--output", metavar="FILE",
                   help="Write JSON report to this path")
    p.add_argument("--format", choices=["markdown", "json"], default="markdown",
                   help="stdout format: markdown (default) or json")
    p.add_argument("--no-fetch", action="store_true",
                   help="Skip fetching reference data (pattern-only, works offline)")
    p.add_argument("--use-llm", action="store_true",
                   help="Use Mistral API for deeper semantic analysis (requires MISTRAL_API_KEY)")
    p.add_argument("--no-llm", action="store_true",
                   help="Disable LLM analysis even if MISTRAL_API_KEY is set")
    return p


def main() -> None:
    args = build_parser().parse_args()

    use_llm = args.use_llm and not args.no_llm
    api_key: Optional[str] = os.environ.get("MISTRAL_API_KEY") if use_llm else None
    if use_llm and not api_key:
        print("Warning: --use-llm set but MISTRAL_API_KEY is not in environment. Skipping LLM pass.",
              file=sys.stderr)
        use_llm = False

    # ── Fetch reference data ───────────────────────────────────────────────────
    valid_models: set[str] = set()
    references_checked: list[str] = []
    ref_content: dict[str, str] = {}

    if not args.no_fetch:
        ref_content = fetch_reference_content()

        if "python" in ref_content:
            references_checked.append("client-python/README.md")
        if "typescript" in ref_content:
            references_checked.append("client-ts/README.md")
        for label in ("fine_tuning", "fine_tuning_alt", "changelog", "deprecations", "migration"):
            if label in ref_content:
                references_checked.append(f"platform-docs-public/{label}")

        print("Fetching current model list from /v1/models ...", file=sys.stderr)
        fetch_key = api_key or os.environ.get("MISTRAL_API_KEY")
        valid_models = fetch_valid_models(fetch_key)
        if valid_models:
            references_checked.append("api.mistral.ai/v1/models")
    else:
        print("Skipping reference data fetch (--no-fetch).", file=sys.stderr)

    # ── Scan files ────────────────────────────────────────────────────────────
    print("", file=sys.stderr)
    stale_files: list[dict] = []

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: {path} does not exist.", file=sys.stderr)
            sys.exit(2)
        print(f"Scanning {path} ...", file=sys.stderr)
        result = scan_file(path, valid_models, use_llm, api_key, ref_content)
        if result:
            stale_files.append(result)
    else:
        directory = Path(args.dir)
        if not directory.exists():
            print(f"Error: directory '{directory}' does not exist.", file=sys.stderr)
            sys.exit(2)
        print(f"Scanning {directory}/ ...", file=sys.stderr)
        stale_files = scan_directory(directory, valid_models, use_llm, api_key, ref_content)

    # ── Build report ──────────────────────────────────────────────────────────
    report = {
        "scanned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "references_checked": references_checked,
        "valid_models_count": len(valid_models),
        "files": stale_files,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.output}", file=sys.stderr)

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(to_markdown(report))

    sys.exit(1 if stale_files else 0)


if __name__ == "__main__":
    main()
