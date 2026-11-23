"""Sub-workflow: ModernizeFileWorkflow: one per source file.

WHY SUB-WORKFLOWS INSTEAD OF ACTIVITIES?
Each file modernization involves multiple steps (read -> LLM -> validate) that
can fail independently and need isolated retry surfaces. If we put all three
steps in a single activity, a syntax-validation failure would require re-running
the file read AND the LLM call from scratch. Instead, each file is its own
workflow:

  Parent workflow
    +-- ModernizeFileWorkflow (file A)  <- own history, own retry budget
    +-- ModernizeFileWorkflow (file B)  <- independent: B's retries don't block A
    +-- ModernizeFileWorkflow (file C)

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
    2. LLM modernization via chat_parse_to_model with structured output.
    3. Syntax validation in a sandboxed subprocess.
    4. If validation fails, loop back to step 2 (up to MAX_VALIDATION_ATTEMPTS times).
    5. Return a FileModernizationResult.

    RETRY MECHANISM: HOW IT ACTUALLY WORKS
    validate_python_syntax does not raise on invalid code. It returns an
    (ok, stderr) pair. An exception raised inside an activity reaches the
    workflow as a Temporal ActivityError, not as the original exception type,
    so an "except SyntaxError" here would never fire. Returning the result
    keeps the decision in the workflow body.

    The retry loop is therefore explicit here: we call modernize_file_llm and
    validate_python_syntax in a for-loop. While ok is False, we continue to
    the next iteration with a fresh LLM call. This makes each attempt visible
    as a distinct event pair in the Studio timeline, which is a better demo
    than hidden automatic retries. Real infrastructure failures (for example
    a subprocess timeout) still raise and surface as activity failures.

    modernize_file_llm still has retry_policy_max_attempts=2 on the activity
    decorator to handle transient LLM/network failures within a single attempt.
    That's a separate concern from the validate-and-retry loop here.
    """

    @workflows.workflow.entrypoint
    async def run(self, params: ModernizeFileParams) -> FileModernizationResult:
        """Orchestrate read -> (modernize -> validate) loop for a single file.

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

        # Steps 2-4: Explicit modernize -> validate loop.
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
            ok, stderr = await validate_python_syntax(
                source=modernized.modernized,
                relative_path=refreshed.relative_path,
            )
            if ok:
                # Validation passed: exit the loop
                syntax_valid = True
                break
            # Generated code is syntactically invalid. If we have attempts
            # left, loop back for a fresh LLM call. On the final attempt,
            # fall through with syntax_valid=False so the parent can still
            # aggregate the result rather than losing the whole file.
            if stderr:
                print(f"Syntax check failed for {refreshed.relative_path} (attempt {attempt + 1}): {stderr}")

        assert modernized is not None  # loop always runs at least once

        return FileModernizationResult(
            relative_path=refreshed.relative_path,
            original=modernized.original,
            modernized=modernized.modernized,
            changes_summary=modernized.changes_summary,
            confidence=modernized.confidence,
            syntax_valid=syntax_valid,
        )
