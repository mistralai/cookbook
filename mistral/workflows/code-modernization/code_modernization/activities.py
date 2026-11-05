"""Activities for the Code Modernization workflow.

All I/O, file reads, subprocess calls, and LLM interactions live here: never
inside the workflow body. Workflow code must be deterministic (Temporal replays
it); activities are where side effects belong.

Activity design notes:
- Each @workflows.activity() function is independently retryable.
- start_to_close_timeout caps wall-clock time per attempt.
- modernize_file_llm has retry_policy_max_attempts=2 to handle transient
  LLM/network failures within a single attempt. The validate-and-retry loop
  (up to MAX_VALIDATION_ATTEMPTS) lives in sub_workflow.py, not here.
"""

import os
import subprocess
from datetime import timedelta
from pathlib import Path

from mistralai import workflows
from mistralai.client.models import ChatCompletionRequest, UserMessage
from mistralai.workflows.plugins.mistralai.activities import chat_parse_to_model

from .models import FileModernizationResult, FileToModernize, ModernizedFile


def _confined_path(raw_path: str) -> Path:
    """Resolve a caller-supplied path, confined to the project directory.

    ``repo_path`` and the file paths derived from it come from workflow input,
    which is untrusted. Walking or reading an arbitrary location (e.g. ``/etc``)
    and sending its contents to the LLM: or writing ``PR_PROPOSAL.md`` outside
    the project: would let a caller exfiltrate or clobber host files. We resolve
    the path under the project working directory and reject anything that escapes
    it.
    """
    root = Path.cwd()
    candidate = Path(raw_path)
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Refusing to access '{candidate}': outside the project directory '{root}'.")
    return candidate


@workflows.activity(
    name="scan_files_to_modernize",
    start_to_close_timeout=timedelta(seconds=30),
)
async def scan_files_to_modernize(repo_path: str) -> list[FileToModernize]:
    """Scan repo_path for Python source files and return them with their content.

    Rule-based scan: finds all *.py files (excluding __pycache__ and .venv).
    In production this would run against a cloned GitHub repo.
    """
    base = _confined_path(repo_path)
    files: list[FileToModernize] = []
    for root, dirs, filenames in os.walk(base):
        # Skip common noise directories
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".venv", ".git", "node_modules")]
        for fname in filenames:
            # Skip empty __init__.py files: they carry no modernizable content
            if fname == "__init__.py":
                continue
            if fname.endswith(".py"):
                abs_path = _confined_path(os.path.join(root, fname))
                rel_path = abs_path.relative_to(base).as_posix()
                source = abs_path.read_text()
                files.append(FileToModernize(path=str(abs_path), relative_path=rel_path, source=source))
    return files


@workflows.activity(
    name="read_file",
    start_to_close_timeout=timedelta(seconds=10),
)
async def read_file(file: FileToModernize) -> FileToModernize:
    """Re-read a file from disk to guarantee the sub-workflow has the latest content.

    Idempotent: safe to retry if disk I/O transiently fails.
    """
    source = _confined_path(file.path).read_text()
    return FileToModernize(path=file.path, relative_path=file.relative_path, source=source)


@workflows.activity(
    name="modernize_file_llm",
    start_to_close_timeout=timedelta(minutes=2),
    # retry_policy_max_attempts=2: covers transient LLM/network failures within
    # a single attempt (e.g. API timeout, malformed JSON response). This is
    # separate from the validate-and-retry loop in sub_workflow.py, which drives
    # fresh LLM calls when generated code fails syntax validation.
    retry_policy_max_attempts=2,
)
async def modernize_file_llm(source: str, relative_path: str, target: str) -> ModernizedFile:
    """Call the Mistral LLM to produce a modernized version of a Python file.

    Uses chat_parse_to_model so the response is validated as a ModernizedFile
    Pydantic model. If the API call fails or the response is malformed, the
    activity raises: Temporal retries it up to retry_policy_max_attempts times.
    """
    prompt = f"""You are a Python modernization assistant.

Modernize the following Python source file from {target}.

Return a JSON object with exactly these fields:
- original: the original source code (copy verbatim)
- modernized: the fully modernized source code
- changes_summary: a list of short strings, one per change made
- confidence: a float between 0.0 and 1.0 rating your confidence

File: {relative_path}

Source:
```python
{source}
```"""

    request = ChatCompletionRequest(
        model="mistral-small-latest",
        messages=[UserMessage(content=prompt)],
    )
    return await chat_parse_to_model(ModernizedFile, request)


@workflows.activity(
    name="validate_python_syntax",
    start_to_close_timeout=timedelta(seconds=15),
    # No Temporal retry on this activity: SyntaxError is an expected signal
    # that the caller (sub_workflow.py) uses to drive a fresh LLM call.
    # Retrying the validator with the same broken source would always fail.
)
async def validate_python_syntax(source: str, relative_path: str) -> bool:
    """Run syntax validation in a subprocess: sandboxed from the worker process.

    Shells out to a fresh Python interpreter so a SyntaxError in the generated
    code cannot crash the worker. The subprocess timeout (10s) is explicit and
    independent of the Temporal activity timeout.

    Raises:
        SyntaxError: If the modernized source has invalid Python syntax. The
            sub_workflow.py caller catches this and loops back for a fresh LLM
            call (up to MAX_VALIDATION_ATTEMPTS times).
        subprocess.TimeoutExpired: If the subprocess hangs beyond 10s: an
            infrastructure problem that should not be silently swallowed.
    """
    # Blocking on purpose: the activity already runs in the worker's async loop.
    result = subprocess.run(  # noqa: ASYNC221
        ["python3", "-c", f"compile({source!r}, {relative_path!r}, 'exec')"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,  # Subprocess-level guard, independent of Temporal's timeout
    )
    if result.returncode != 0:
        raise SyntaxError(f"Generated code for {relative_path} failed syntax check:\n{result.stderr}")
    return True


@workflows.activity(
    name="write_pr_proposal",
    start_to_close_timeout=timedelta(seconds=30),
)
async def write_pr_proposal(repo_path: str, target: str, files: list[FileModernizationResult]) -> str:
    """Write a PR_PROPOSAL.md summarising what would be opened as a GitHub PR.

    In production this is `await open_github_pr(...)` using the GitHub Connector.
    For the demo we write a local markdown file that describes the proposed changes.
    """
    lines = [
        f"# PR Proposal: {target}",
        "",
        "## Summary",
        "",
        f"Modernized {len(files)} file(s).",
        "",
        "## Files Changed",
        "",
    ]
    for f in files:
        lines.append(f"### `{f.relative_path}`")
        lines.append(f"Confidence: {f.confidence:.0%}")
        for change in f.changes_summary:
            lines.append(f"- {change}")
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- In production this file would not be written; instead a GitHub PR would be opened",
        "  via the GitHub Connector using the modernized file contents.",
        "- Each file was validated with `python3 -c compile(...)` before inclusion.",
    ]

    output_path = _confined_path(repo_path) / "PR_PROPOSAL.md"
    output_path.write_text("\n".join(lines))
    return str(output_path)
