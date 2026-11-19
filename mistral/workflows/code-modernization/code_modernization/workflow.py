"""Parent workflow: CodeModernizationWorkflow.

Orchestrates a fan-out over N source files, aggregates results, and pauses for
human review before opening a PR. This is the entry point a developer triggers
from Studio or the CLI.

RELIABILITY PRIMITIVES DEMONSTRATED
====================================
1. Sub-workflows (headline primitive)
   Each file is its own sub-workflow: own retry surface, own observability,
   parent stays clean. See sub_workflow.py for the per-file logic.

2. Parallel sub-workflow execution via asyncio.gather
   All files are dispatched simultaneously. A 10-file repo runs in the time
   of the slowest single file, not 10× that.

3. wait_for_input(): durable HITL pause
   After aggregation the workflow suspends at zero compute cost. The reviewer
   sees the change set in Studio, clicks Accept/Decline, and the
   workflow resumes, even if hours have passed. No polling, no timeouts.

4. Sandboxed validation
   The validate_python_syntax activity shells out to a subprocess, preventing
   generated code from crashing the worker process.
"""

import asyncio
from datetime import timedelta

import mistralai.workflows.plugins.mistralai as wm
from mistralai import workflows

from .activities import scan_files_to_modernize, write_pr_proposal
from .models import (
    ChangeSet,
    FileModernizationResult,
    ModernizationParams,
    ModernizeFileParams,
    WorkflowResult,
)
from .sub_workflow import ModernizeFileWorkflow

_PARENT_NAME = "code-modernization"


@workflows.workflow.define(
    name=_PARENT_NAME,
    execution_timeout=timedelta(hours=2),
)
class CodeModernizationWorkflow(workflows.InteractiveWorkflow):
    """Parent workflow: orchestrates the full modernization pipeline.

    Extends InteractiveWorkflow to gain wait_for_input(). The HITL pause at
    step 5 is what makes this a *reliable* modernization tool: no AI output
    reaches production without a human in the loop.
    """

    @workflows.workflow.entrypoint
    async def run(self, params: ModernizationParams) -> WorkflowResult:
        """Execute the full modernization pipeline.

        Args:
            params: repo_path and target (e.g. "Python 2.7 → 3.12").

        Returns:
            WorkflowResult with status='approved' and a PR proposal path, or
            status='declined' if the reviewer discards the change set.
        """
        # ------------------------------------------------------------------
        # Step 1: Clone / locate the repo
        # In production: await clone_repo(github_url) via GitHub Connector.
        # For the demo we use the local fixture at sample_data/legacy_repo/.
        # ------------------------------------------------------------------
        repo_path = params.repo_path

        # ------------------------------------------------------------------
        # Step 2: Identify files to modernize: rule-based scan activity
        # ------------------------------------------------------------------
        files_to_modernize = await scan_files_to_modernize(repo_path)

        if not files_to_modernize:
            return WorkflowResult(
                status="skipped",
                change_set=ChangeSet(repo_path=repo_path, target=params.target, files=[]),
            )

        # ------------------------------------------------------------------
        # Step 3: Fan out: one sub-workflow per file, run in parallel.
        #
        # Each file is its own sub-workflow (not just an activity) because:
        #   - Sub-workflows have their own event history → independent retries.
        #   - If the LLM produces invalid Python, that file retries up to 3×
        #     without affecting other files in flight.
        #   - Studio shows each file as a collapsible child execution.
        #
        # asyncio.gather() dispatches all children simultaneously so a 10-file
        # repo completes in ~max(individual_file_times), not their sum.
        # ------------------------------------------------------------------
        child_tasks = [
            workflows.workflow.execute_workflow(
                ModernizeFileWorkflow,
                params=ModernizeFileParams(file=f, target=params.target),
                execution_timeout=timedelta(minutes=5),
            )
            for f in files_to_modernize
        ]
        results: list[FileModernizationResult] = await asyncio.gather(*child_tasks)

        # ------------------------------------------------------------------
        # Step 4: Aggregate results into a ChangeSet
        # ------------------------------------------------------------------
        change_set = ChangeSet(
            repo_path=repo_path,
            target=params.target,
            files=list(results),
        )

        # ------------------------------------------------------------------
        # Step 5: HITL review: durable pause via wait_for_input().
        #
        # WHY wait_for_input() here?
        # The change set may touch dozens of files. Automated approval would
        # risk shipping broken migrations. wait_for_input() suspends execution
        # at zero compute cost: the worker process is freed, the workflow
        # history is durable, and it resumes the instant the reviewer responds.
        # This is 1 line of Python; the equivalent in raw Temporal is ~50 lines.
        # ------------------------------------------------------------------
        summary_lines = [
            f"Repo: {repo_path}",
            f"Target: {params.target}",
            f"Files modernized: {len(change_set.files)}",
            f"Syntax-valid: {change_set.success_count} / {len(change_set.files)}",
        ]
        for f in change_set.files:
            status = "OK" if f.syntax_valid else "WARN"
            summary_lines.append(f"  [{status}] {f.relative_path} (confidence: {f.confidence:.0%})")

        await wm.send_assistant_message("\n".join(summary_lines))

        confirmation = await self.wait_for_input(
            wm.AcceptDeclineConfirmation(
                description="Reviewer: approve this change set?",
                accept_label="Open PR",
                decline_label="Discard",
            ),
            label="Change Set Review",
        )

        # ------------------------------------------------------------------
        # Step 6/7: On approval, write PR proposal; on decline, return early.
        # ------------------------------------------------------------------
        if wm.is_accepted(confirmation):
            # In production: await open_github_pr(...) via GitHub Connector.
            # For the demo, write a PR_PROPOSAL.md describing the changes.
            pr_path = await write_pr_proposal(
                repo_path=repo_path,
                target=params.target,
                files=change_set.files,
            )
            return WorkflowResult(
                status="approved",
                change_set=change_set,
                pr_proposal_path=pr_path,
            )
        else:
            return WorkflowResult(
                status="declined",
                change_set=change_set,
            )
