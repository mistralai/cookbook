"""Linear Weekly Summary workflow with InteractiveWorkflow HITL.

The workflow declares the **linear** connector via ``@uses_connectors``
for team/project/issue activities. Summary generation uses Mistral's
built-in async plugin activity (``mistralai_chat_complete``).

``on_behalf_of=True`` on ``@workflow.define`` propagates the executor's
identity so connector lookup resolves the correct stored Linear auth.
"""

import re
from datetime import timedelta

import mistralai.workflows.plugins.mistralai as workflows_mistralai
from mistralai import workflows
from mistralai.workflows.plugins.mistralai.connectors import uses_connectors
from pydantic import BaseModel

from .linear_activities import (
    fetch_linear_projects,
    fetch_linear_teams,
    fetch_recent_issues,
    get_project_id_by_name,
    get_team_id_by_name,
    linear_connector,
)
from .summary_activity import generate_weekly_summary


class LinearWorkflowParams(BaseModel):
    """Input parameters for Linear Weekly Summary workflow."""

    team: str | None = None  # name or ID
    project: str | None = None  # name or ID


# Timeout for each HITL step. 10 minutes is enough for an interactive demo;
# raise to hours for production scenarios where reviewers may not be at
# their desk. Workflow execution_timeout (defaults to 1 hour) caps the total
# lifetime independently.
_HITL_TIMEOUT = timedelta(minutes=10)
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@workflows.workflow.define(
    name="linear-weekly-summary",
    # Required so the Linear connector can resolve the executor's stored
    # authentication at activity-execution time — without it, the connector
    # has no user identity to look up credentials for.
    on_behalf_of=True,
)
@uses_connectors(linear_connector)
class LinearWeeklySummaryWorkflow(workflows.InteractiveWorkflow):
    """Linear weekly summary with InteractiveWorkflow HITL.

    Extends InteractiveWorkflow to use ``wait_for_input(ConfirmationInput)``
    for team/project selection. The selection renders natively in Studio.
    """

    @workflows.workflow.entrypoint
    async def run(self, params: LinearWorkflowParams) -> str:
        """Resolve team + project (direct args or HITL), then fetch + summarise."""
        team_id: str | None = None
        project_id: str | None = None

        async def _resolve_project_id(
            selected_team_id: str,
            raw_project: str,
        ) -> str | None:
            """Resolve project input to UUID (or None for '*')."""
            project_input = raw_project.strip()
            if project_input == "*":
                return None
            if _UUID_PATTERN.match(project_input):
                return project_input
            project_result = await get_project_id_by_name(
                selected_team_id, project_input
            )
            return project_result["project_id"]

        # 1. Resolve team input first (team name or team UUID).
        if params.team:
            try:
                team_input = params.team.strip()
                if _UUID_PATTERN.match(team_input):
                    team_id = team_input
                else:
                    team_result = await get_team_id_by_name(team_input)
                    team_id = team_result["team_id"]
            except ValueError:
                # Fall through to HITL
                team_id = None

        # 2. If project provided and team is resolved, route by project ID vs name.
        if team_id and params.project:
            try:
                project_id = await _resolve_project_id(team_id, params.project)
            except ValueError:
                project_id = None

        # 3. HITL fallback for unresolved team and/or unresolved project.
        # project_id is None is valid when project was "*" (all projects in team).
        needs_team_hitl = team_id is None
        needs_project_hitl = project_id is None and (
            not params.project or params.project.strip() != "*"
        )

        if needs_team_hitl:
            # 3a. Pick a team via wait_for_input — the activity result drives the options.
            teams = await fetch_linear_teams()
            if not teams:
                raise ValueError("No teams returned by Linear API.")

            team_choice = await self.wait_for_input(
                workflows_mistralai.ConfirmationInput(
                    description="Pick a Linear team to summarise.",
                    options=[(t["id"], t["name"]) for t in teams],
                ),
                label="Select team",
                timeout=_HITL_TIMEOUT,
            )
            team_id = team_choice.choice

        # If team is now resolved by HITL and project was provided, retry direct
        # project resolution before asking for project via HITL.
        if (
            team_id
            and project_id is None
            and params.project
            and params.project.strip() != "*"
        ):
            try:
                project_id = await _resolve_project_id(team_id, params.project)
            except ValueError:
                project_id = None

        needs_project_hitl = project_id is None and (
            not params.project or params.project.strip() != "*"
        )

        if needs_project_hitl:
            # 3b. Pick a project. "*" means all projects in the team.
            if team_id is None:
                raise ValueError("Team selection failed before project selection.")
            projects = await fetch_linear_projects(team_id)
            project_options: list[tuple[str, str]] = [
                ("*", "All projects in this team")
            ]
            project_options.extend((p["id"], p["name"]) for p in projects)

            project_choice = await self.wait_for_input(
                workflows_mistralai.ConfirmationInput(
                    description="Pick a project (or all projects in the team).",
                    options=project_options,
                ),
                label="Select project",
                timeout=_HITL_TIMEOUT,
            )
            project_id = None if project_choice.choice == "*" else project_choice.choice

        # 4. Fetch recent issues and generate summary
        issues = await fetch_recent_issues(team_id, project_id, 7)
        summary = await generate_weekly_summary(issues)
        return summary
