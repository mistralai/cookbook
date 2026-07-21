#!/usr/bin/env python3
"""
Review Mistral cookbook files added or modified in a pull request.

Reads the Mistral Writing Style Guide from skills/cookbook-review/,
calls the Mistral API for each changed file, then posts one GitHub comment
per issue — each tied to the specific line or cell it pertains to.

For new files (A), the full content is reviewed.
For modified files (M), only the changed cells/lines are reviewed.

Required environment variables:
  GITHUB_TOKEN         GitHub Actions token (pull-requests: write)
  MISTRAL_API_KEY      Mistral API key
  GITHUB_REPOSITORY    owner/repo (e.g. mistralai/cookbook)
  PR_NUMBER            Pull request number
  HEAD_SHA             Commit SHA at the tip of the PR branch
  BASE_SHA             Commit SHA of the base branch

Usage:
  python review_pr.py <path-to-changedfiles-list>

The changedfiles list is a plain-text file with one file path per line,
produced by `git diff --diff-filter=AM --name-only`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
REPO = os.environ["GITHUB_REPOSITORY"]
PR_NUMBER = int(os.environ["PR_NUMBER"])
HEAD_SHA = os.environ["HEAD_SHA"]
BASE_SHA = os.environ["BASE_SHA"]

GITHUB_API = "https://api.github.com"
GH_HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json",
}

SKILLS_DIR = Path("skills/cookbook-review")
REVIEW_MODEL = "mistral-medium-latest"

# Truncate very long files so the review stays focused and within context limits.
MAX_REVIEW_LINES = 600

# ── Style guide ───────────────────────────────────────────────────────────────

STYLE_GUIDE_FILES = [
    "SKILL.md",
    "voice-and-tone.md",
    "checklists.md",
    "ai-terms.md",
    "developer-content.md",
    "inclusive-language.md",
]


def load_style_guide() -> str:
    """Concatenate all style guide resource files into one context block."""
    parts: list[str] = []
    for name in STYLE_GUIDE_FILES:
        path = SKILLS_DIR / name
        if path.exists():
            parts.append(f"### {name}\n\n{path.read_text().strip()}")
        else:
            print(f"  WARNING: style guide file not found: {path}")
    return "\n\n---\n\n".join(parts)


# ── File reading ──────────────────────────────────────────────────────────────

MAX_REVIEW_CELLS = 40  # truncate notebooks at this many cells


def read_numbered(filepath: str) -> tuple[str, int]:
    """
    Return (numbered_content, total_line_count).

    Lines are prepended with their 1-based line number so the model can
    reference exact locations. Content is truncated at MAX_REVIEW_LINES.
    """
    lines = Path(filepath).read_text().splitlines()
    total = len(lines)
    visible = lines[:MAX_REVIEW_LINES]

    numbered = "\n".join(f"{i + 1:4}: {line}" for i, line in enumerate(visible))

    if total > MAX_REVIEW_LINES:
        numbered += (
            f"\n\n[Content truncated: showing lines 1–{MAX_REVIEW_LINES} of {total}. "
            "Flag issues only for the visible lines.]"
        )

    return numbered, total


def read_notebook(filepath: str) -> tuple[str, int]:
    """
    Return (cell_content, total_cell_count) for a Jupyter notebook.

    Extracts markdown and code cells into readable text so the model can
    review prose quality and code style. Cell numbers are used instead of
    line numbers since raw JSON line positions aren't meaningful in a PR diff.
    """
    nb = json.loads(Path(filepath).read_text())
    cells = nb.get("cells", [])
    total = len(cells)
    visible = cells[:MAX_REVIEW_CELLS]

    parts: list[str] = []
    for i, cell in enumerate(visible, 1):
        cell_type = cell.get("cell_type", "unknown")
        source = "".join(cell.get("source", []))
        if source.strip():
            parts.append(f"[Cell {i} — {cell_type}]\n{source}")

    content = "\n\n".join(parts)
    if total > MAX_REVIEW_CELLS:
        content += f"\n\n[Truncated: showing cells 1–{MAX_REVIEW_CELLS} of {total}.]"

    return content, total


# ── Diff-aware reading (modified files only) ──────────────────────────────────


def get_file_status(filepath: str) -> str:
    """Return 'A' (added) or 'M' (modified) for this file in the current PR."""
    result = subprocess.run(
        ["git", "diff", "--diff-filter=AM", "--name-status", BASE_SHA, HEAD_SHA, "--", filepath],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[1].strip() == filepath:
            return parts[0].strip()  # 'A' or 'M'
    return "A"  # default to new-file behaviour if status is unclear


def read_changed_cells(filepath: str) -> tuple[str, int]:
    """
    For a modified notebook, return only the cells whose source changed.

    Compares the notebook at BASE_SHA against the current version and surfaces
    each changed or added cell labelled with its index and change type.
    """
    base_result = subprocess.run(
        ["git", "show", f"{BASE_SHA}:{filepath}"],
        capture_output=True, text=True,
    )
    if base_result.returncode != 0:
        return read_notebook(filepath)

    try:
        base_cells = json.loads(base_result.stdout).get("cells", [])
    except json.JSONDecodeError:
        return read_notebook(filepath)

    curr_cells = json.loads(Path(filepath).read_text()).get("cells", [])

    parts: list[str] = []
    for i in range(min(len(base_cells), len(curr_cells))):
        base_src = "".join(base_cells[i].get("source", []))
        curr_src = "".join(curr_cells[i].get("source", []))
        if base_src != curr_src:
            cell_type = curr_cells[i].get("cell_type", "unknown")
            parts.append(f"[Cell {i + 1} — {cell_type} — MODIFIED]\n{curr_src}")

    for i in range(len(base_cells), len(curr_cells)):
        src = "".join(curr_cells[i].get("source", []))
        if src.strip():
            cell_type = curr_cells[i].get("cell_type", "unknown")
            parts.append(f"[Cell {i + 1} — {cell_type} — ADDED]\n{src}")

    if not parts:
        return "(No cell content changed — only notebook metadata or output was modified.)", 0

    return "\n\n".join(parts), len(parts)


def read_changed_lines(filepath: str) -> tuple[str, int]:
    """
    For a modified markdown file, return the unified diff of changed lines.

    Shows added (+) and removed (-) lines with surrounding context so the
    reviewer understands what changed without re-reading the entire file.
    """
    result = subprocess.run(
        ["git", "diff", "-U5", BASE_SHA, HEAD_SHA, "--", filepath],
        capture_output=True, text=True,
    )
    diff = result.stdout.strip()
    if not diff:
        return "(No line-level changes detected.)", 0

    lines = diff.splitlines()
    hunk_start = next((i for i, l in enumerate(lines) if l.startswith("@@")), 0)
    hunks = "\n".join(lines[hunk_start:MAX_REVIEW_LINES])
    return hunks, len(lines)


# ── Mistral API call ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT_MD = """\
You are a technical documentation reviewer for Mistral AI cookbooks.
Review the provided cookbook Markdown file against the Mistral Writing Style Guide below.

{style_guide}

---

OUTPUT RULES
Respond with a single, valid JSON object — no text before or after. Use this exact schema:

{{
  "summary": "<2–4 sentence overall assessment of the file quality and main issues>",
  "verdict": "approve" | "comment" | "request_changes",
  "line_comments": [
    {{
      "line": <integer — exact line number shown in the numbered content>,
      "severity": "critical" | "moderate" | "minor",
      "issue": "<concise label, max 10 words>",
      "reasoning": "<1–2 sentences explaining exactly which style guide rule is violated and why>",
      "suggestion": "<see SUGGESTION RULES below>"
    }}
  ],
  "file_comments": [
    {{
      "severity": "critical" | "moderate" | "minor",
      "issue": "<concise label, max 10 words>",
      "reasoning": "<1–2 sentences explaining which rule is violated and why it matters>",
      "body": "<description of the structural problem — quote the relevant style guide rule>"
    }}
  ]
}}

RULES
- verdict: "request_changes" if any critical issue exists; "comment" for moderate/minor only; "approve" if the file looks good.
- line_comments[].line: must be an integer matching a line number in the numbered input. Do not guess.
- Use line_comments for every issue you can tie to a specific line. Use file_comments ONLY for issues with no applicable line — for example, an entire required section is completely absent from the file.
- Limit the total issues across both arrays to the 8 most impactful.
- Do not invent problems. Flag only genuine violations of the style guide.

SUGGESTION RULES
The suggestion field replaces the full content of that one line. Be SURGICAL:
- Change ONLY the specific word, phrase, or value that violates the rule.
- Preserve all surrounding text exactly as it appears on that line.
- Do NOT rewrite an entire sentence when only one word needs changing.
- Do NOT include surrounding lines, backticks, or code fences in the suggestion value.
- Omit the "suggestion" key entirely if the fix requires changing multiple lines, or if no minimal single-line replacement is possible.
"""

_SYSTEM_PROMPT_IPYNB = """\
You are a technical documentation reviewer for Mistral AI cookbooks.
Review the provided Jupyter notebook against the Mistral Writing Style Guide below.

{style_guide}

---

OUTPUT RULES
Respond with a single, valid JSON object — no text before or after. Use this exact schema:

{{
  "summary": "<2–4 sentence overall assessment of the notebook quality and main issues>",
  "verdict": "approve" | "comment" | "request_changes",
  "line_comments": [],
  "file_comments": [
    {{
      "severity": "critical" | "moderate" | "minor",
      "cell": <integer cell number from the input, or null for file-wide issues>,
      "issue": "<concise label, max 10 words>",
      "reasoning": "<1–2 sentences explaining exactly which style guide rule is violated and why>",
      "body": "<description referencing the specific cell and quoting the problematic text>",
      "suggestion": "<the corrected replacement for the specific phrase or sentence only — omit if structural or multi-line>"
    }}
  ]
}}

RULES
- verdict: "request_changes" if any critical issue exists; "comment" for moderate/minor only; "approve" if the notebook looks good.
- line_comments must always be an empty array — notebooks use file_comments only.
- file_comments[].cell: the cell number shown in the input (e.g. 3 for "[Cell 3 — markdown]"). Use null only for issues that apply to the entire notebook with no specific cell.
- Focus on markdown cells for prose style and structure; focus on code cells for security (no hard-coded credentials), clarity, and completeness of example output.
- Limit to the 8 most impactful issues.
- Do not invent problems. Flag only genuine violations of the style guide.

SUGGESTION RULES
- The suggestion contains ONLY the corrected version of the specific phrase or sentence that is wrong.
- Do not rewrite the entire cell when only a few words need changing.
- Omit the "suggestion" key if the fix is structural (e.g. adding a missing section) or spans multiple sentences.
"""

_SYSTEM_PROMPT_MD_DIFF = """\
You are a technical documentation reviewer for Mistral AI cookbooks.
Review only the CHANGED lines of the provided Markdown file, shown as a unified diff.
Lines beginning with '+' are additions; lines beginning with '-' are removals;
context lines (no prefix) are unchanged — do not flag them.

{style_guide}

---

OUTPUT RULES
Respond with a single, valid JSON object — no text before or after. Use this exact schema:

{{
  "summary": "<2–4 sentence assessment of the changed content only>",
  "verdict": "approve" | "comment" | "request_changes",
  "line_comments": [],
  "file_comments": [
    {{
      "severity": "critical" | "moderate" | "minor",
      "issue": "<concise label, max 10 words>",
      "reasoning": "<1–2 sentences explaining exactly which style guide rule is violated and why>",
      "body": "<description quoting the specific '+' line being flagged>",
      "suggestion": "<the corrected replacement for the specific phrase or value only — omit if structural or multi-line>"
    }}
  ]
}}

RULES
- Only comment on added ('+') lines. Ignore removed ('-') and context lines.
- verdict: "request_changes" if any critical issue exists; "comment" for moderate/minor only; "approve" if the changes look good.
- line_comments must always be an empty array.
- Limit to the 8 most impactful issues.
- Do not invent problems. Flag only genuine violations of the style guide.

SUGGESTION RULES
- The suggestion contains ONLY the corrected version of the specific phrase or value that is wrong.
- Do not rewrite the full line when only a word or phrase needs changing.
- Omit the "suggestion" key if the fix spans multiple lines.
"""

_SYSTEM_PROMPT_IPYNB_DIFF = """\
You are a technical documentation reviewer for Mistral AI cookbooks.
Review only the CHANGED cells of the provided Jupyter notebook.
Each cell shown is labelled MODIFIED or ADDED — these are the only cells that changed in this PR.
Apply the Mistral Writing Style Guide below to the changed cell content only.

{style_guide}

---

OUTPUT RULES
Respond with a single, valid JSON object — no text before or after. Use this exact schema:

{{
  "summary": "<2–4 sentence assessment of the changed cells only>",
  "verdict": "approve" | "comment" | "request_changes",
  "line_comments": [],
  "file_comments": [
    {{
      "severity": "critical" | "moderate" | "minor",
      "cell": <integer cell number from the input, or null for notebook-wide issues>,
      "issue": "<concise label, max 10 words>",
      "reasoning": "<1–2 sentences explaining exactly which style guide rule is violated and why>",
      "body": "<description referencing the specific cell and quoting the problematic text>",
      "suggestion": "<the corrected replacement for the specific phrase or sentence only — omit if structural or multi-line>"
    }}
  ]
}}

RULES
- Only evaluate the cells shown — do not speculate about unchanged cells.
- verdict: "request_changes" if any critical issue exists; "comment" for moderate/minor only; "approve" if the changes look good.
- line_comments must always be an empty array — notebooks use file_comments only.
- file_comments[].cell: the cell number from the label (e.g. 3 for "[Cell 3 — markdown — MODIFIED]"). Use null only for notebook-wide issues with no specific cell.
- Limit to the 8 most impactful issues.
- Do not invent problems. Flag only genuine violations of the style guide.

SUGGESTION RULES
- The suggestion contains ONLY the corrected version of the specific phrase or sentence that is wrong.
- Do not rewrite the entire cell when only a few words need changing.
- Omit the "suggestion" key if the fix is structural or spans multiple sentences.
"""


def call_mistral(filepath: str, content: str, style_guide: str, is_diff: bool = False) -> dict:
    """Call the Mistral API and return the parsed review as a Python dict."""
    is_notebook = filepath.endswith(".ipynb")
    if is_diff:
        template = _SYSTEM_PROMPT_IPYNB_DIFF if is_notebook else _SYSTEM_PROMPT_MD_DIFF
    else:
        template = _SYSTEM_PROMPT_IPYNB if is_notebook else _SYSTEM_PROMPT_MD
    system = template.format(style_guide=style_guide)

    if is_diff:
        content_label = "Changed notebook cells" if is_notebook else "Unified diff of changes"
    else:
        content_label = "Notebook cells" if is_notebook else "File content with line numbers"
    user = (
        f"Review this file: `{filepath}`\n\n"
        f"{content_label}:\n```\n{content}\n```"
    )

    payload = {
        "model": REVIEW_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }

    resp = requests.post(
        MISTRAL_API_URL, headers=MISTRAL_HEADERS, json=payload, timeout=180
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return json.loads(raw)


# ── Comment body builders ─────────────────────────────────────────────────────

_SEVERITY_PREFIX = {
    "critical": "**Critical**",
    "moderate": "**Moderate**",
    "minor": "Minor",
}


def _build_line_comment_body(lc: dict) -> str:
    """Build the body for a single inline line comment."""
    prefix = _SEVERITY_PREFIX.get(lc.get("severity", "moderate"), "**Moderate**")
    issue = lc.get("issue", "")
    reasoning = lc.get("reasoning", "")

    parts = [f"{prefix} — {issue}", "", reasoning]

    suggestion = lc.get("suggestion")
    if suggestion is not None:
        parts += ["", "```suggestion", suggestion, "```"]

    return "\n".join(parts)


def _build_file_comment_body(filepath: str, fc: dict) -> str:
    """Build the body for a file-level or cell-level PR comment."""
    prefix = _SEVERITY_PREFIX.get(fc.get("severity", "moderate"), "**Moderate**")
    issue = fc.get("issue", "")
    cell = fc.get("cell")
    reasoning = fc.get("reasoning", "")
    body = fc.get("body", "")

    location = f"Cell {cell} — " if cell is not None else ""
    parts = [f"`{filepath}` — {prefix} — {location}{issue}"]

    if reasoning:
        parts += ["", reasoning]
    if body and body != reasoning:
        parts += ["", body]

    suggestion = fc.get("suggestion")
    if suggestion:
        parts += ["", "**Suggested fix:**", "```", suggestion, "```"]

    return "\n".join(parts)


# ── GitHub posting ────────────────────────────────────────────────────────────


def post_verdict_review(filepath: str, summary: str, verdict: str) -> None:
    """Post a lightweight review event with just the summary — no bundled inline comments."""
    event_map = {
        "approve": "APPROVE",
        "request_changes": "REQUEST_CHANGES",
        "comment": "COMMENT",
    }
    gh_event = event_map.get(verdict, "COMMENT")
    body = (
        f"## Cookbook review: `{filepath}`\n\n"
        f"{summary}\n\n"
        "---\n"
        "_Reviewed against the "
        "[Mistral Writing Style Guide](../skills/cookbook-review/SKILL.md)._"
    )
    payload = {
        "commit_id": HEAD_SHA,
        "body": body,
        "event": gh_event,
        "comments": [],
    }
    url = f"{GITHUB_API}/repos/{REPO}/pulls/{PR_NUMBER}/reviews"
    resp = requests.post(url, headers=GH_HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    print(f"  Posted {gh_event} verdict review.")


def post_line_comment(path: str, line: int, body: str) -> bool:
    """
    Post a single inline comment on a specific diff line.
    Returns True on success, False if GitHub rejects the line (not in diff).
    """
    payload = {
        "body": body,
        "commit_id": HEAD_SHA,
        "path": path,
        "line": line,
        "side": "RIGHT",
    }
    url = f"{GITHUB_API}/repos/{REPO}/pulls/{PR_NUMBER}/comments"
    resp = requests.post(url, headers=GH_HEADERS, json=payload, timeout=30)
    if resp.status_code == 422:
        print(f"    Line {line} not in diff (HTTP 422) — will post as PR comment.")
        return False
    resp.raise_for_status()
    return True


def post_pr_comment(body: str) -> None:
    """Post a general PR conversation comment (not attached to a specific line)."""
    payload = {"body": body}
    url = f"{GITHUB_API}/repos/{REPO}/issues/{PR_NUMBER}/comments"
    resp = requests.post(url, headers=GH_HEADERS, json=payload, timeout=30)
    resp.raise_for_status()


def post_review(filepath: str, review: dict, total_lines: int) -> None:
    """
    Post the review as separate comments:
    1. One lightweight verdict review (summary only, no bundled comments).
    2. One separate inline comment per line_comment issue.
    3. One separate PR comment per file_comment issue.
    """
    verdict = review.get("verdict", "comment")
    summary = review.get("summary", "")

    post_verdict_review(filepath, summary, verdict)

    n_inline = 0
    n_fallback = 0
    for lc in review.get("line_comments", []):
        line = lc.get("line")
        body = _build_line_comment_body(lc)
        if isinstance(line, int) and 1 <= line <= total_lines:
            if post_line_comment(filepath, line, body):
                n_inline += 1
                continue
        else:
            print(f"    Out-of-range line={line!r} — posting as PR comment.")
        post_pr_comment(body)
        n_fallback += 1

    n_file = 0
    for fc in review.get("file_comments", []):
        post_pr_comment(_build_file_comment_body(filepath, fc))
        n_file += 1

    print(
        f"  {n_inline} inline comment(s), "
        f"{n_fallback} fallback PR comment(s), "
        f"{n_file} file-level PR comment(s)."
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: review_pr.py <newfiles-list>")

    files_list = Path(sys.argv[1])
    files = [p for p in files_list.read_text().splitlines() if p.strip()]

    if not files:
        print("No new cookbook files to review.")
        return

    print(f"Loading Mistral Writing Style Guide from {SKILLS_DIR}/ ...")
    style_guide = load_style_guide()
    if not style_guide:
        sys.exit(
            f"ERROR: No style guide files found in {SKILLS_DIR}/. "
            "Ensure skills/cookbook-review/ exists and contains the guide files."
        )

    exit_code = 0

    for filepath in files:
        print(f"\nReviewing: {filepath}")

        if not Path(filepath).exists():
            print("  File not found — skipping.")
            continue

        is_notebook = filepath.endswith(".ipynb")
        status = get_file_status(filepath)
        is_diff = status == "M"

        if is_diff:
            kind = "cells" if is_notebook else "lines"
            print(f"  Modified file — reviewing only changed {kind}.")
            if is_notebook:
                content, total = read_changed_cells(filepath)
            else:
                content, total = read_changed_lines(filepath)
            if total == 0:
                print("  No content changes detected — skipping review.")
                continue
            print(f"  {total} changed {kind}.")
        elif is_notebook:
            content, total = read_notebook(filepath)
            print(f"  New file — {total} cell(s), reviewing up to {MAX_REVIEW_CELLS}.")
        else:
            content, total = read_numbered(filepath)
            print(f"  New file — {total} total line(s), reviewing up to {MAX_REVIEW_LINES}.")

        print("  Calling Mistral API ...")
        try:
            review = call_mistral(filepath, content, style_guide, is_diff=is_diff)
        except Exception as exc:
            print(f"  Mistral API call failed: {exc}")
            exit_code = 1
            continue

        verdict = review.get("verdict", "?")
        n_line = len(review.get("line_comments", []))
        n_file = len(review.get("file_comments", []))
        print(
            f"  Verdict: {verdict} | "
            f"{n_line} line comment(s), {n_file} file-level comment(s)."
        )

        print("  Posting GitHub PR review ...")
        try:
            post_review(filepath, review, 0 if (is_notebook or is_diff) else total)
        except requests.HTTPError as exc:
            print(f"  Failed to post review: {exc}")
            print(f"  Response body: {exc.response.text[:500]}")
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
