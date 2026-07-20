#!/usr/bin/env python3
"""
Review new Mistral cookbook files added in a pull request.

Reads the Mistral Writing Style Guide from skills/cookbook-review/,
calls the Mistral API for each new file, then posts a GitHub PR review
with inline suggestions and comments.

Required environment variables:
  GITHUB_TOKEN         GitHub Actions token (pull-requests: write)
  MISTRAL_API_KEY      Mistral API key
  GITHUB_REPOSITORY    owner/repo (e.g. mistralai/cookbook)
  PR_NUMBER            Pull request number
  HEAD_SHA             Commit SHA at the tip of the PR branch
  BASE_SHA             Commit SHA of the base branch

Usage:
  python review_pr.py <path-to-newfiles-list>

The newfiles list is a plain-text file with one file path per line,
produced by `git diff --diff-filter=A --name-only`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
REPO = os.environ["GITHUB_REPOSITORY"]
PR_NUMBER = int(os.environ["PR_NUMBER"])
HEAD_SHA = os.environ["HEAD_SHA"]

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
      "body": "<full explanation for the GitHub review comment>",
      "suggestion": "<replacement text for this single line — omit key entirely if no single-line fix applies>"
    }}
  ],
  "file_comments": [
    {{
      "severity": "critical" | "moderate" | "minor",
      "body": "<comment about a file-level issue: missing required section, wrong structure, etc.>"
    }}
  ]
}}

RULES FOR EACH FIELD
- verdict: "request_changes" if any critical issue exists; "comment" for moderate/minor only; "approve" if the file looks good.
- line_comments[].line: must be an integer matching a line number visible in the numbered input. Do not guess.
- line_comments[].suggestion: when present, contains ONLY the replacement text for that one line — no backticks, no fences. If the fix requires touching multiple lines, use file_comments instead.
- Limit the total issues across both arrays to the 8 most impactful.
- Do not invent problems. Flag only genuine violations of the style guide.
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
      "body": "<comment referencing the specific cell and quoting the problematic text>"
    }}
  ]
}}

RULES FOR EACH FIELD
- verdict: "request_changes" if any critical issue exists; "comment" for moderate/minor only; "approve" if the notebook looks good.
- line_comments must always be an empty array — notebooks use file_comments only.
- file_comments[].cell: the cell number shown in the input (e.g. 3 for "[Cell 3 — markdown]"). Use null for issues that apply to the whole notebook.
- Focus on markdown cells for prose style and structure; focus on code cells for security (no hard-coded credentials), clarity, and completeness of example output.
- Limit to the 8 most impactful issues.
- Do not invent problems. Flag only genuine violations of the style guide.
"""


def call_mistral(filepath: str, content: str, style_guide: str) -> dict:
    """Call the Mistral API and return the parsed review as a Python dict."""
    is_notebook = filepath.endswith(".ipynb")
    template = _SYSTEM_PROMPT_IPYNB if is_notebook else _SYSTEM_PROMPT_MD
    system = template.format(style_guide=style_guide)

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


# ── GitHub review posting ─────────────────────────────────────────────────────

_SEVERITY_PREFIX = {
    "critical": "**Critical**",
    "moderate": "**Moderate**",
    "minor": "Minor",
}


def _inline_body(lc: dict) -> str:
    """Build the body for a single inline comment, adding a suggestion block if present."""
    prefix = _SEVERITY_PREFIX.get(lc.get("severity", "moderate"), "**Moderate**")
    issue = lc.get("issue", "")
    explanation = lc.get("body", "")

    parts = [f"{prefix} — {issue}", "", explanation]

    suggestion = lc.get("suggestion")
    if suggestion is not None:
        # GitHub renders this as a one-click "Apply suggestion" button in the PR.
        parts += ["", "```suggestion", suggestion, "```"]

    return "\n".join(parts)


def _review_body(filepath: str, review: dict) -> str:
    """Build the top-level review body (shown above the diff in the PR timeline)."""
    summary = review.get("summary", "")
    file_comments = review.get("file_comments", [])

    lines = [f"## Cookbook review: `{filepath}`", "", summary]

    if file_comments:
        lines += ["", "### Issues", ""]
        for fc in file_comments:
            prefix = _SEVERITY_PREFIX.get(fc.get("severity", "moderate"), "**Moderate**")
            cell = fc.get("cell")
            location = f"Cell {cell} — " if cell is not None else ""
            lines.append(f"- {prefix}: {location}{fc['body']}")

    lines += [
        "",
        "---",
        "_Reviewed against the "
        "[Mistral Writing Style Guide](../skills/cookbook-review/SKILL.md)._",
    ]

    return "\n".join(lines)


def _fallback_review_body(filepath: str, review: dict, rejected_comments: list[dict]) -> str:
    """
    Extended review body used when GitHub rejects inline comments.

    Appends the rejected inline issues as a plain list so no feedback is lost.
    """
    body = _review_body(filepath, review)

    if not rejected_comments:
        return body

    body += "\n\n### Inline issues (applied as file-level comments)\n\n"
    for lc in rejected_comments:
        prefix = _SEVERITY_PREFIX.get(lc.get("severity", "moderate"), "**Moderate**")
        line = lc.get("line", "?")
        body += f"- Line {line} — {prefix}: {lc.get('body', '')}\n"
        suggestion = lc.get("suggestion")
        if suggestion:
            body += f"\n  Suggested replacement:\n  ```\n  {suggestion}\n  ```\n"

    return body


def post_review(filepath: str, review: dict, total_lines: int) -> None:
    """Post the review to the GitHub PR."""
    event_map = {
        "approve": "APPROVE",
        "request_changes": "REQUEST_CHANGES",
        "comment": "COMMENT",
    }
    gh_event = event_map.get(review.get("verdict", "comment"), "COMMENT")

    # Validate line numbers — GitHub will reject the entire review if any comment
    # references a line that isn't part of the diff.
    valid_comments: list[dict] = []
    invalid_comments: list[dict] = []

    for lc in review.get("line_comments", []):
        line = lc.get("line")
        if isinstance(line, int) and 1 <= line <= total_lines:
            valid_comments.append(lc)
        else:
            print(f"  Skipping line comment with out-of-range line={line!r}")
            invalid_comments.append(lc)

    inline = [
        {
            "path": filepath,
            "line": lc["line"],
            "side": "RIGHT",
            "body": _inline_body(lc),
        }
        for lc in valid_comments
    ]

    payload = {
        "commit_id": HEAD_SHA,
        "body": _review_body(filepath, review),
        "event": gh_event,
        "comments": inline,
    }

    url = f"{GITHUB_API}/repos/{REPO}/pulls/{PR_NUMBER}/reviews"
    resp = requests.post(url, headers=GH_HEADERS, json=payload, timeout=30)

    # 422 means one or more inline comments was rejected (line not in the diff).
    # Retry without inline comments so the review still lands.
    if resp.status_code == 422 and inline:
        print(
            f"  GitHub rejected inline comments (HTTP 422). "
            "Retrying without them — all issues will appear in the review body."
        )
        payload["comments"] = []
        payload["body"] = _fallback_review_body(filepath, review, valid_comments)
        resp = requests.post(url, headers=GH_HEADERS, json=payload, timeout=30)

    resp.raise_for_status()
    review_id = resp.json().get("id", "?")
    n_inline = len(inline) if resp.status_code == 200 else 0
    print(f"  Posted review #{review_id} ({gh_event}, {n_inline} inline comment(s)).")


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

        if is_notebook:
            content, total = read_notebook(filepath)
            print(f"  {total} cell(s), reviewing up to {MAX_REVIEW_CELLS}.")
        else:
            content, total = read_numbered(filepath)
            print(f"  {total} total line(s), reviewing up to {MAX_REVIEW_LINES}.")

        print("  Calling Mistral API ...")
        try:
            review = call_mistral(filepath, content, style_guide)
        except Exception as exc:
            print(f"  Mistral API call failed: {exc}")
            exit_code = 1
            continue

        verdict = review.get("verdict", "?")
        n_line = len(review.get("line_comments", []))
        n_file = len(review.get("file_comments", []))
        print(
            f"  Verdict: {verdict} | "
            f"{n_line} inline comment(s), {n_file} file-level comment(s)."
        )

        print("  Posting GitHub PR review ...")
        try:
            # Notebooks skip inline comments — all feedback lands in the review body.
            post_review(filepath, review, 0 if is_notebook else total)
        except requests.HTTPError as exc:
            print(f"  Failed to post review: {exc}")
            print(f"  Response body: {exc.response.text[:500]}")
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
