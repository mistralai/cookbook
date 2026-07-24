#!/usr/bin/env python3
"""
check_stale.py — Scan Mistral AI cookbooks for outdated content.

Reference sources:
  - /v1/models API (deprecation field per model)
  - mistralai/client-python README (current Python SDK patterns)
  - mistralai/client-ts README (current TypeScript SDK patterns)
  - platform-docs-public models overview page (legacy/deprecated model list)
  - platform-docs-public deprecation/migration pages

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

# Human-readable links
PYTHON_SDK_URL     = "https://github.com/mistralai/client-python/blob/main/README.md"
TS_SDK_URL         = "https://github.com/mistralai/client-ts/blob/main/README.md"
DOCS_MODELS_URL    = "https://docs.mistral.ai/getting-started/models/models_overview/"
DOCS_LEGACY_URL    = "https://docs.mistral.ai/getting-started/models/models_overview/#legacy-models"
DOCS_FINE_TUNING_URL = "https://docs.mistral.ai/capabilities/fine-tuning/"
PLATFORM_DOCS_URL  = "https://github.com/mistralai/platform-docs-public"

# Keys for the ref_content dict
_KEY_PYTHON = "python"
_KEY_TS     = "typescript"

# Deprecation signal words — used to validate LLM evidence quotes
_DEPRECATION_SIGNALS = frozenset({
    "deprecated", "deprecation", "legacy", "removed", "no longer supported",
    "end of life", "end-of-life", "will be removed", "not recommended",
    "use instead", "replaced by", "superseded",
})

# Model ID prefixes we care about
_MODEL_KEYWORDS = ("mistral", "codestral", "pixtral", "mixtral", "ministral")


# ─── Static deprecated patterns ────────────────────────────────────────────────
# reference_search: text to look for in the fetched reference to compute a #L anchor.

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
        "reference_key": _KEY_PYTHON,
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
        "reference_key": _KEY_PYTHON,
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
        "reference_key": _KEY_PYTHON,
        "reference_search": "from mistralai import Mistral",
    },
    # ── Python SDK v0 client class ─────────────────────────────────────────────
    {
        "pattern": r"\bMistralClient\s*\(",
        "type": "deprecated_class",
        "detail": (
            "Deprecated `MistralClient` class (SDK v0). "
            "Use `Mistral(api_key=...)` instead."
        ),
        "reference_url": PYTHON_SDK_URL,
        "reference_key": _KEY_PYTHON,
        "reference_search": "Mistral(api_key",
    },
    {
        "pattern": r"\bChatMessage\s*\(",
        "type": "deprecated_class",
        "detail": (
            "Deprecated `ChatMessage` class (removed in SDK v1). "
            "Use plain dicts: `{\"role\": \"user\", \"content\": \"...\"}`."
        ),
        "reference_url": PYTHON_SDK_URL,
        "reference_key": _KEY_PYTHON,
        "reference_search": "role",
    },
    # ── Python SDK v0 method signatures ───────────────────────────────────────
    {
        "pattern": r"\.chat\s*\(\s*(?!.*\.complete)",
        "type": "deprecated_method",
        "detail": (
            "Deprecated `client.chat()` call (SDK v0). "
            "Use `client.chat.complete(...)` instead."
        ),
        "reference_url": PYTHON_SDK_URL,
        "reference_key": _KEY_PYTHON,
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
        "reference_key": _KEY_PYTHON,
        "reference_search": "embeddings.create(",
    },
    # ── Fine-tuning API (deprecated) ───────────────────────────────────────────
    {
        "pattern": r"client\.fine_tuning\b",
        "type": "deprecated_api",
        "detail": (
            "The fine-tuning jobs API (`client.fine_tuning`) is deprecated. "
            "See the fine-tuning documentation for the current approach."
        ),
        "reference_url": DOCS_FINE_TUNING_URL,
    },
    {
        "pattern": r"/v1/fine[_-]tuning/",
        "type": "deprecated_api",
        "detail": (
            "The `/v1/fine_tuning/` REST endpoint is deprecated. "
            "See the fine-tuning documentation for the current approach."
        ),
        "reference_url": DOCS_FINE_TUNING_URL,
    },
    # ── Pinned old package versions ────────────────────────────────────────────
    {
        "pattern": r"pip install mistralai==0\.",
        "type": "outdated_version",
        "detail": (
            "Pinned to SDK v0. "
            "Update to the current version: `pip install mistralai`."
        ),
        "reference_url": PYTHON_SDK_URL,
        "reference_key": _KEY_PYTHON,
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
        "reference_key": _KEY_TS,
        "reference_search": "npm add @mistralai/mistralai",
    },
    # ── Pinned model versions (prefer -latest aliases) ─────────────────────────
    {
        "pattern": r"""['"][a-z][a-z0-9\-]+-20\d\d-\d\d[a-z0-9\-]*['"]""",
        "type": "pinned_model_version",
        "detail": (
            "References a pinned dated model version (e.g. `-2309`, `-2402`). "
            "Consider using a `-latest` alias so the cookbook stays current automatically."
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
    """Fetch all reference docs. Returns label → text.

    Content is sent to the LLM so it can ground claims in real source text,
    and used to compute #L anchors for reference URLs.
    """
    content: dict[str, str] = {}

    print("Fetching Python SDK README ...", file=sys.stderr)
    if text := fetch_url(PYTHON_README_RAW_URL, "client-python README"):
        content[_KEY_PYTHON] = text

    print("Fetching TypeScript SDK README ...", file=sys.stderr)
    if text := fetch_url(TS_README_RAW_URL, "client-ts README"):
        content[_KEY_TS] = text

    # Docs pages that contain legacy model lists and deprecation notices.
    # Try several candidate paths — platform-docs-public uses different extensions.
    docs_candidates: list[tuple[str, str]] = [
        ("models_overview", "docs/getting-started/models/models_overview.mdx"),
        ("models_overview", "docs/getting-started/models/models_overview.md"),
        ("fine_tuning",     "docs/capabilities/fine-tuning.mdx"),
        ("fine_tuning",     "docs/capabilities/fine-tuning.md"),
        ("changelog",       "CHANGELOG.md"),
        ("deprecations",    "docs/deprecations.md"),
        ("deprecations",    "docs/deprecations.mdx"),
    ]
    seen_keys: set[str] = set()
    for label, path in docs_candidates:
        if label in seen_keys:
            continue  # already fetched a variant for this key
        url = f"{PLATFORM_DOCS_RAW_BASE}/{path}"
        if text := fetch_url(url, f"platform-docs-public/{path}"):
            content[label] = text
            seen_keys.add(label)
            print(f"  Fetched platform-docs-public/{path}", file=sys.stderr)

    return content


# ─── Deprecated model detection ────────────────────────────────────────────────

def _model_id_pattern() -> re.Pattern[str]:
    kw = "|".join(_MODEL_KEYWORDS)
    return re.compile(rf"\b(?:{kw})[\w\-\.]+\b", re.IGNORECASE)

_MODEL_ID_RE = _model_id_pattern()
_MODEL_IN_CALL = re.compile(
    r"""(?:model\s*=\s*|"model"\s*:\s*)['"]([\w\-\.]+)['"]"""
)


def parse_deprecated_model_ids(content: str) -> dict[str, str]:
    """Extract model IDs from legacy/deprecated sections of a docs page.

    Returns a dict of model_id → section_anchor for use in reference URLs.
    Looks for headings containing 'legacy' or 'deprecated', then extracts
    Mistral model IDs from the text beneath them.
    """
    deprecated: dict[str, str] = {}

    # Split the document on headings (##, ###, etc.)
    sections = re.split(r"(?m)^#{1,4}\s+(.+)$", content)
    # sections alternates: [text_before_first_heading, heading1, body1, heading2, body2, ...]
    i = 0
    while i < len(sections):
        chunk = sections[i]
        if i + 1 < len(sections):
            heading = sections[i]       # might be body text before first heading
            # The actual heading text is at odd indices after the split
        i += 1

    # Simpler approach: walk line by line, track whether we're inside a legacy section
    in_legacy_section = False
    anchor = ""
    for line in content.splitlines():
        # Detect heading lines
        m = re.match(r"^#{1,4}\s+(.+)$", line)
        if m:
            heading_text = m.group(1).strip().lower()
            if any(w in heading_text for w in ("legacy", "deprecated", "deprecat")):
                in_legacy_section = True
                # Build a GitHub-style anchor from the heading
                anchor = "#" + re.sub(r"[^a-z0-9\-]", "", heading_text.replace(" ", "-"))
            else:
                in_legacy_section = False
        elif in_legacy_section:
            for model_id in _MODEL_ID_RE.findall(line):
                deprecated[model_id] = anchor

    return deprecated


def fetch_model_data(
    api_key: Optional[str],
    ref_content: dict[str, str],
) -> tuple[set[str], dict[str, str]]:
    """Return (all_valid_ids, deprecated_id_to_ref_url).

    Deprecated IDs come from two sources:
    - `deprecated`/`deprecation` field in the /v1/models API response
    - The legacy/deprecated sections of the models overview docs page
    """
    all_ids: set[str] = set()
    deprecated: dict[str, str] = {}

    # ── /v1/models API ────────────────────────────────────────────────────────
    if api_key:
        try:
            resp = requests.get(
                "https://api.mistral.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=20,
            )
            resp.raise_for_status()
            for m in resp.json().get("data", []):
                mid = m.get("id")
                if not isinstance(mid, str):
                    continue
                all_ids.add(mid)
                # The API may expose a deprecation date or boolean
                if m.get("deprecated") or m.get("deprecation"):
                    deprecated[mid] = DOCS_LEGACY_URL
            print(f"  Found {len(all_ids)} model IDs from /v1/models "
                  f"({len(deprecated)} deprecated).", file=sys.stderr)
        except Exception as exc:
            print(f"Warning: could not fetch /v1/models: {exc}", file=sys.stderr)

    # ── Docs legacy section ───────────────────────────────────────────────────
    if "models_overview" in ref_content:
        docs_deprecated = parse_deprecated_model_ids(ref_content["models_overview"])
        for mid, anchor in docs_deprecated.items():
            ref_url = DOCS_LEGACY_URL if not anchor else f"{DOCS_MODELS_URL}{anchor}"
            deprecated.setdefault(mid, ref_url)  # API deprecation field takes precedence
        if docs_deprecated:
            print(f"  Found {len(docs_deprecated)} deprecated model ID(s) in docs.",
                  file=sys.stderr)

    return all_ids, deprecated


def check_deprecated_models(
    text: str,
    deprecated_models: dict[str, str],
    location: str,
) -> list[dict]:
    """Flag model IDs that are explicitly listed as deprecated in official sources.

    Only flags models that appear in `deprecated_models` — a dict populated from
    the /v1/models API deprecation field and/or the docs legacy-models section.
    Does NOT flag unrecognized models: absence from the list is not evidence of
    deprecation.
    """
    if not deprecated_models:
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
            if model_id in deprecated_models:
                issues.append({
                    "type": "deprecated_model",
                    "detail": (
                        f"`{model_id}` is listed as a deprecated/legacy model. "
                        "Replace it with a current model."
                    ),
                    "location": f"{location}, line {lineno}" if location else f"line {lineno}",
                    "matched_line": line.strip()[:200],
                    "reference_url": deprecated_models[model_id],
                })
    return issues


# ─── Reference URL enrichment ──────────────────────────────────────────────────

def line_anchored_url(base_url: str, content: str, search_text: str) -> str:
    """Return a GitHub #L<n> URL if search_text is found in content."""
    if not content or not search_text:
        return base_url
    for i, line in enumerate(content.splitlines(), 1):
        if search_text in line:
            return f"{base_url}#L{i}"
    return base_url


def enrich_reference_url(issue: dict, ref_content: dict[str, str]) -> dict:
    """Add a #L anchor to the issue's reference_url using fetched content."""
    search = issue.get("reference_search", "")
    key = issue.get("reference_key", "")
    if not search or key not in ref_content:
        return issue
    base_url = issue.get("reference_url", "")
    enriched = line_anchored_url(base_url, ref_content[key], search)
    if enriched != base_url:
        return {**issue, "reference_url": enriched}
    return issue


# ─── Code extraction ───────────────────────────────────────────────────────────

def extract_notebook_cells(path: Path) -> list[tuple[int, str]]:
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
    blocks = []
    for i, m in enumerate(re.finditer(r"```(\w+)?\n(.*?)```", text, re.DOTALL), 1):
        lang = m.group(1) or "unknown"
        code = m.group(2)
        if code.strip():
            blocks.append((i, lang, code))
    return blocks


# ─── Pattern checking ──────────────────────────────────────────────────────────

_CODE_LANGS = {"python", "typescript", "javascript", "bash", "shell", "sh", "unknown"}


def check_patterns(
    text: str, location: str, ref_content: dict[str, str]
) -> list[dict]:
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
                    "reference_key": info.get("reference_key", ""),
                }
                issue = enrich_reference_url(issue, ref_content)
                issue.pop("reference_search", None)
                issue.pop("reference_key", None)
                issues.append(issue)
    return issues


# ─── LLM-assisted analysis ─────────────────────────────────────────────────────

_LLM_SYSTEM = """\
You are a technical reviewer. Your job is to find deprecated patterns in Mistral AI
cookbook code — but ONLY when you have explicit evidence of the deprecation in the
provided reference content.

## Your process

1. Read the reference content provided in the user message.
2. Find explicit deprecation notices in that content: text that says "deprecated",
   "legacy", "removed", "no longer supported", "replaced by", or similar.
3. Check whether the cookbook code uses any of those deprecated things.
4. Flag ONLY issues where BOTH are true:
   a. The reference content explicitly says the thing is deprecated.
   b. The cookbook code uses that deprecated thing.

## What NOT to flag

- Do NOT flag things based on what looks unusual or unfamiliar to you.
- Do NOT flag a pattern just because you don't see it in the reference — absence
  of a pattern in the reference is NOT evidence of deprecation.
- The `_async` suffix (e.g. `create_async`, `list_async`, `start_async`,
  `complete_async`, `delete_async`) is a valid part of the Mistral Python SDK.
  If you see these used in the reference content, that is proof they are NOT
  deprecated. Do not flag them.

## Required response format

Respond ONLY with a JSON object with a single key "issues" whose value is an array.
Each element must have:

  type: string (snake_case label for the issue category)
  detail: string (one sentence: what is wrong and what to use instead)
  reference_url: string (the most specific URL from those listed in the user message,
    including a #L anchor or section anchor if you can determine one)
  quote: string (the EXACT sentence or phrase from the provided reference content that
    explicitly states the deprecation — not an example of the method being used,
    but the actual deprecation notice)

If you cannot provide a quote from the reference content that explicitly states the
deprecation, do not include that issue.

Return {"issues": []} if you find nothing beyond what was already flagged.\
"""


def _build_ref_context(ref_content: dict[str, str]) -> str:
    """Build the reference content block to include in the LLM user message."""
    parts: list[str] = []
    if _KEY_PYTHON in ref_content:
        excerpt = ref_content[_KEY_PYTHON][:3000]
        parts.append(f"Python SDK README (first 3 000 chars):\n```\n{excerpt}\n```")
    if _KEY_TS in ref_content:
        excerpt = ref_content[_KEY_TS][:1500]
        parts.append(f"TypeScript SDK README (first 1 500 chars):\n```\n{excerpt}\n```")
    for label in ("models_overview", "fine_tuning", "changelog", "deprecations"):
        if label in ref_content:
            excerpt = ref_content[label][:2000]
            parts.append(f"platform-docs-public/{label} (first 2 000 chars):\n```\n{excerpt}\n```")
    return "\n\n".join(parts)


def _is_deprecation_evidence(quote: str, ref_content: dict[str, str]) -> bool:
    """Return True only if the quote appears in the reference content AND is surrounded
    by deprecation language.

    This validates LLM claims: if the model quotes text that exists in the reference
    but isn't actually a deprecation notice, we discard the issue.
    """
    if not quote or not quote.strip():
        return False  # no quote = no evidence

    all_ref = "\n".join(ref_content.values()).lower()
    q = quote.strip().lower()

    if q not in all_ref:
        return False  # quote not found in any reference — fabricated

    # Find the quote's position and check for deprecation language in the surrounding
    # ~300 characters on either side.
    idx = all_ref.find(q)
    window = all_ref[max(0, idx - 300): idx + len(q) + 300]
    return any(signal in window for signal in _DEPRECATION_SIGNALS)


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
        f"Reference URLs (use the most specific one you can determine):\n"
        f"- Python SDK README: {PYTHON_SDK_URL}\n"
        f"- TypeScript SDK README: {TS_SDK_URL}\n"
        f"- Mistral model overview: {DOCS_MODELS_URL}\n"
        f"- Mistral legacy models: {DOCS_LEGACY_URL}\n"
        f"- Fine-tuning docs: {DOCS_FINE_TUNING_URL}\n\n"
    )

    if ref_context:
        user_msg += (
            "Reference content — search this for explicit deprecation notices:\n\n"
            f"{ref_context}\n\n"
        )

    user_msg += (
        "List any additional stale patterns NOT already flagged. "
        "Every issue must include a `quote` that is the exact deprecation notice "
        "from the reference content above. Return {\"issues\": []} if none."
    )

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
    raw = re.sub(r"^```(?:json)?\n?", "", raw).strip()
    raw = re.sub(r"\n?```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            result = parsed.get("issues", [])
            if not isinstance(result, list):
                # Fallback: find first list value
                result = next((v for v in parsed.values() if isinstance(v, list)), [])
        elif isinstance(parsed, list):
            result = parsed
        else:
            return []
    except json.JSONDecodeError as exc:
        print(f"Warning: could not parse LLM response for {file_path}: {exc}", file=sys.stderr)
        return []

    # ── Evidence filter ────────────────────────────────────────────────────────
    # Discard any issue whose quote is not found in the reference content in a
    # deprecation context. This catches hallucinated references and cases where
    # the LLM quotes a usage example instead of an actual deprecation notice.
    validated: list[dict] = []
    for item in result:
        item.setdefault("location", "LLM analysis (no specific line)")
        item.setdefault("quote", "")
        quote = item.get("quote", "")
        if not _is_deprecation_evidence(quote, ref_content):
            print(
                f"  LLM issue discarded (no deprecation evidence in reference): "
                f"{item.get('type', '?')} — quote: {quote[:80]!r}",
                file=sys.stderr,
            )
            continue
        validated.append(item)

    return validated


# ─── File scanning ─────────────────────────────────────────────────────────────

def scan_file(
    path: Path,
    deprecated_models: dict[str, str],
    use_llm: bool,
    api_key: Optional[str],
    ref_content: dict[str, str],
) -> Optional[dict]:
    suffix = path.suffix.lower()
    all_issues: list[dict] = []
    llm_content = ""

    if suffix == ".ipynb":
        for cell_idx, source in extract_notebook_cells(path):
            loc = f"Cell {cell_idx}"
            all_issues += check_patterns(source, loc, ref_content)
            all_issues += check_deprecated_models(source, deprecated_models, loc)
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
            all_issues += check_deprecated_models(code, deprecated_models, loc)
            llm_content += f"\n# Code block {block_idx} ({lang})\n{code}\n"

    elif suffix == ".py":
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
            return None
        all_issues += check_patterns(text, "", ref_content)
        all_issues += check_deprecated_models(text, deprecated_models, "")
        llm_content = text

    else:
        return None

    llm_issues: list[dict] = []
    if use_llm and api_key and llm_content.strip():
        llm_issues = llm_analyze(str(path), llm_content, all_issues, api_key, ref_content)

    combined = all_issues + llm_issues
    return {"path": str(path), "issues": combined} if combined else None


_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints", ".venv", "venv"}
_SCAN_EXTS = {".ipynb", ".md", ".py"}


def scan_directory(
    directory: Path,
    deprecated_models: dict[str, str],
    use_llm: bool,
    api_key: Optional[str],
    ref_content: dict[str, str],
) -> list[dict]:
    results: list[dict] = []
    for path in sorted(directory.rglob("*")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in _SCAN_EXTS or not path.is_file():
            continue
        print(f"  Scanning {path} ...", file=sys.stderr)
        result = scan_file(path, deprecated_models, use_llm, api_key, ref_content)
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
        f"**Deprecated models found in docs/API:** {report['deprecated_models_count']}  ",
        f"**References checked:** {', '.join(report['references_checked']) or 'none (offline mode)'}",
        "",
    ]
    if not files:
        lines.append("✅ No stale content found.")
        return "\n".join(lines)

    for file_info in files:
        lines += ["---", "", f"## `{file_info['path']}`", ""]
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
                    lines.append(f"  - **Evidence:** _{issue['quote']}_")
                lines.append(f"  - **Reference:** [{issue['reference_url']}]({issue['reference_url']})")
            lines.append("")
    return "\n".join(lines)


# ─── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scan Mistral AI cookbooks for stale content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dir", default="mistral", metavar="DIR")
    p.add_argument("--file", metavar="FILE")
    p.add_argument("--output", metavar="FILE")
    p.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--no-llm", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    use_llm = args.use_llm and not args.no_llm
    api_key: Optional[str] = os.environ.get("MISTRAL_API_KEY") if use_llm else None
    if use_llm and not api_key:
        print("Warning: --use-llm set but MISTRAL_API_KEY not in environment. Skipping LLM pass.",
              file=sys.stderr)
        use_llm = False

    ref_content: dict[str, str] = {}
    deprecated_models: dict[str, str] = {}
    references_checked: list[str] = []

    if not args.no_fetch:
        ref_content = fetch_reference_content()

        if _KEY_PYTHON in ref_content:
            references_checked.append("client-python/README.md")
        if _KEY_TS in ref_content:
            references_checked.append("client-ts/README.md")
        for label in ("models_overview", "fine_tuning", "changelog", "deprecations"):
            if label in ref_content:
                references_checked.append(f"platform-docs-public/{label}")

        print("Fetching model data from /v1/models ...", file=sys.stderr)
        fetch_key = api_key or os.environ.get("MISTRAL_API_KEY")
        _all_ids, deprecated_models = fetch_model_data(fetch_key, ref_content)
        if _all_ids:
            references_checked.append("api.mistral.ai/v1/models")
    else:
        print("Skipping reference data fetch (--no-fetch).", file=sys.stderr)

    print("", file=sys.stderr)
    stale_files: list[dict] = []

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: {path} does not exist.", file=sys.stderr)
            sys.exit(2)
        print(f"Scanning {path} ...", file=sys.stderr)
        result = scan_file(path, deprecated_models, use_llm, api_key, ref_content)
        if result:
            stale_files.append(result)
    else:
        directory = Path(args.dir)
        if not directory.exists():
            print(f"Error: directory '{directory}' does not exist.", file=sys.stderr)
            sys.exit(2)
        print(f"Scanning {directory}/ ...", file=sys.stderr)
        stale_files = scan_directory(directory, deprecated_models, use_llm, api_key, ref_content)

    report = {
        "scanned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "references_checked": references_checked,
        "deprecated_models_count": len(deprecated_models),
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
