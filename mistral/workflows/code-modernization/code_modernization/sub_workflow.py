"""Sub-workflow: ModernizeFileWorkflow: one per source file.

WHY SUB-WORKFLOWS INSTEAD OF ACTIVITIES?
Each file modernization involves multiple steps (read → LLM → validate) that
can fail independently and need isolated retry surfaces. If we put all three
steps in a single activity, a syntax-validation failure would require re-running
the file read AND the LLM call from scratch. Instead, each file is its own
workflow:

  Parent workflow
    ├── ModernizeFileWorkflow (file A)  ← own history, own retry budget
    ├── ModernizeFileWorkflow (file B)  ← independent: B's retries don't block A
    └── ModernizeFileWorkflow (file C)

This is the "Workflows of Workflows" pattern. Each child appears as a collapsed
node in Studio's timeline, clickable for its own step-by-step drilldown.
"""

from datetime import timedelta

from mistralai import workflows

from .activities import modernize_file_llm, read_file, validate_python_syntax
from .models import FileModernizationResult, ModernizeFileParams

MAX_VALIDATION_ATTEMPTS = 3

_CHILD_NAME = "code-modernization-file"


@workflows.workflow.define(
    name=_CHILD_NAME,
    execution_timeout=timedelta(minutes=5),
)
class ModernizeFileWorkflow:
    """Child workflow that handles one file end-to-end.

    Steps:
    1. Re-read the file (idempotent, safe to replay).
    2. LLM modernization via mistralai_chat_parse with structured output.
    3. Syntax validation in a sandboxed subprocess.
    4. If validation fails, loop back to step 2 (up to MAX_VALIDATION_ATTEMPTS times).
    5. Return a FileModernizationResult.

    RETRY MECHANISM: HOW IT ACTUALLY WORKS
    Temporal retries an activity when *that activity itself* raises. Once
    modernize_file_llm returns successfully, its retry budget is spent: a
    downstream SyntaxError from validate_python_syntax does not cause Temporal
    to re-run the LLM call automatically.

    Instead, the retry loop is explicit in this workflow body: we call
    modernize_file_llm → validate_python_syntax in a for-loop. On SyntaxError
    we continue to the next iteration (a fresh LLM call). This makes each
    attempt visible as a distinct event pair in the Studio timeline, which
    is a better demo than hidden automatic retries.

    modernize_file_llm still has retry_policy_max_attempts=2 on the activity
    decorator to handle transient LLM/network failures within a single attempt.
    That's a separate concern from the validate-and-retry loop here.
    """

    @workflows.workflow.entrypoint
    async def run(self, params: ModernizeFileParams) -> FileModernizationResult:
        """Orchestrate read → (modernize → validate) loop for a single file.

        The validation retry loop is explicit: up to MAX_VALIDATION_ATTEMPTS
        calls to modernize_file_llm, each followed by validate_python_syntax.
        If validation passes, we break early. If it fails on every attempt,
        we record syntax_valid=False and let the parent aggregate the result.

        Infrastructure failures from validate_python_syntax (TimeoutExpired,
        missing binary, etc.) are NOT caught here: they propagate so real
        problems surface rather than being silently swallowed.
        """
        # Step 1: Read: idempotent, light I/O
        refreshed = await read_file(params.file)

        # Steps 2-4: Explicit modernize → validate loop.
        # Each iteration is a fresh LLM call. All attempts appear as separate
        # activity events in the Studio timeline, making the retry behaviour
        # observable without needing to drill into Temporal internals.
        modernized = None
        syntax_valid = False

        for attempt in range(MAX_VALIDATION_ATTEMPTS):
            modernized = await modernize_file_llm(
                source=refreshed.source,
                relative_path=refreshed.relative_path,
                target=params.target,
            )
            try:
                await validate_python_syntax(
                    source=modernized.modernized,
                    relative_path=refreshed.relative_path,
                )
                # Validation passed: exit the loop
                syntax_valid = True
                break
            except SyntaxError:
                # Generated code is syntactically invalid. If we have attempts
                # left, loop back for a fresh LLM call. On the final attempt,
                # fall through with syntax_valid=False so the parent can still
                # aggregate the result rather than losing the whole file.
                if attempt < MAX_VALIDATION_ATTEMPTS - 1:
                    continue
                # Final attempt exhausted: syntax_valid stays False

        assert modernized is not None  # loop always runs at least once

        return FileModernizationResult(
            relative_path=refreshed.relative_path,
            original=modernized.original,
            modernized=modernized.modernized,
            changes_summary=modernized.changes_summary,
            confidence=modernized.confidence,
            syntax_valid=syntax_valid,
        )
