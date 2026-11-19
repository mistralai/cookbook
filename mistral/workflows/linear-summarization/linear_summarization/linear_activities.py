"""Linear activities using Mistral Connectors.

Why this is useful:
- Mistral handles authentication for connector calls, so no API keys need to
    be managed or exposed in workflow code.
- Linear is available as a built-in Mistral Connector. Enable it in Studio
    and start calling tools without extra integration configuration.

Implementation details:
- Each activity declares ``linear: ToolCallClient = Depends(linear_connector)``.
- The activity calls ``await linear.call_tool(tool_name, arguments)`` with
    MCP-style tool names (``list_teams``, ``list_projects``, ``list_issues``,
    ``get_team``, ``get_project``).
"""

from __future__ import annotations

import json
import re
from typing import Any

from mistralai import workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
)

# Module-level slot. Imported by workflow.py and passed to
# @uses_connectors() on the workflow class.
linear_connector = connector("linear")

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _unwrap(response: Any) -> dict | list:
    """Decode the MCP-style ``content[0].text`` JSON payload from a tool response.

    Connector tool responses follow the MCP spec: ``{"content": [{"type":
    "text", "text": "<json>"}]}``. The shape we receive depends on the call
    site: at SDK-client level it's a Pydantic ``ConnectorToolCallResponse``;
    inside a workflow activity, ``connector_tool_call`` returns
    ``result.model_dump()`` (a plain dict). Handle both.
    """
    if isinstance(response, dict):
        content = response.get("content") or []
    else:
        content = getattr(response, "content", None) or []
    if not content:
        return {}
    first = content[0]
    if isinstance(first, dict):
        text = first.get("text")
    else:
        text = getattr(first, "text", None)
    if text is None:
        return {}
    return json.loads(text)


# ---------------------------------------------------------------------------
# Read-only listing activities — used by the HITL path
# ---------------------------------------------------------------------------


@workflows.activity()
async def fetch_linear_teams(
    linear: ToolCallClient = Depends(linear_connector),
) -> list[dict]:
    """Return all teams visible to the executor (id + name)."""
    payload = _unwrap(
        await linear.call_tool(
            tool_name="list_teams",
            arguments={"limit": 250},
        )
    )
    return [{"id": t["id"], "name": t["name"]} for t in payload.get("teams", [])]


@workflows.activity()
async def fetch_linear_projects(
    team_id: str,
    linear: ToolCallClient = Depends(linear_connector),
) -> list[dict]:
    """Return all projects in the given team (id + name).

    NOTE: ``limit=50`` (not 250) — the Linear connector's ``list_projects``
    tool returns HTTP 400 ``invalid_request`` for any ``limit`` above 50,
    despite its declared schema saying ``maximum: 250``. Other list_*
    tools (``list_teams``, ``list_issues``) honour 250 fine; only
    ``list_projects`` has this hidden cap. Once fixed, raise the limit and
    optionally paginate via ``cursor`` for teams with > 50 projects.
    """
    payload = _unwrap(
        await linear.call_tool(
            tool_name="list_projects",
            arguments={"team": team_id, "limit": 50},
        )
    )
    return [{"id": p["id"], "name": p["name"]} for p in payload.get("projects", [])]


# ---------------------------------------------------------------------------
# Validation activities — used by the direct-input path
# ---------------------------------------------------------------------------


@workflows.activity(retry_policy_max_attempts=0)
async def get_team_id_by_name(
    team_name: str,
    linear: ToolCallClient = Depends(linear_connector),
) -> dict:
    """Resolve a Linear team name to team UUID.

    Returns ``{"team_id": str}``.
    Raises ``ValueError`` if no team can be resolved.
    """
    payload = _unwrap(
        await linear.call_tool(
            tool_name="get_team",
            arguments={"query": team_name.strip()},
        )
    )
    team_obj = payload.get("team") or payload
    team_id = team_obj.get("id")
    if not team_id:
        raise ValueError(f"Team '{team_name}' not found")
    return {"team_id": team_id}


@workflows.activity(retry_policy_max_attempts=0)
async def get_project_id_by_name(
    team_id: str,
    project_name: str,
    linear: ToolCallClient = Depends(linear_connector),
) -> dict:
    """Resolve a Linear project name to project UUID within a team.

    Returns ``{"project_id": str}``.
    Raises ``ValueError`` if the project can't be found or is not in the team.
    """
    payload = _unwrap(
        await linear.call_tool(
            tool_name="get_project",
            arguments={"query": project_name.strip()},
        )
    )
    project_obj = payload.get("project") or payload
    project_id = project_obj.get("id")
    if not project_id:
        raise ValueError(f"Project '{project_name}' not found")

    # Ensure the project belongs to the requested team — the connector does
    # not filter ``get_project`` by team, so we do the membership check here.
    teams_block = project_obj.get("teams") or {}
    team_nodes = (
        teams_block.get("nodes") if isinstance(teams_block, dict) else teams_block
    )
    if team_nodes is None:
        team_nodes = []
    member_ids = {t.get("id") for t in team_nodes if isinstance(t, dict)}
    if member_ids and team_id not in member_ids:
        raise ValueError(f"Project '{project_name}' is not in the selected team")
    return {"project_id": project_id}


# ---------------------------------------------------------------------------
# Issue-listing activity — the data source for the weekly summary
# ---------------------------------------------------------------------------


@workflows.activity()
async def fetch_recent_issues(
    team_id: str,
    project_id: str | None,
    days: int = 7,
    linear: ToolCallClient = Depends(linear_connector),
) -> list[dict]:
    """Return Linear issues updated within the past ``days`` days.

    When ``project_id`` is ``None`` the query is scoped to the team only
    (= all projects in the team, matches the ``"*"`` HITL convention).

    Uses the ``list_issues`` tool's ISO-8601 duration syntax (``-P{days}D``)
    instead of computing a since-timestamp client-side — this keeps the
    workflow body deterministic without needing ``temporal_wf.now()``.
    """
    arguments: dict[str, Any] = {
        "team": team_id,
        "updatedAt": f"-P{days}D",
        "limit": 250,
        "orderBy": "updatedAt",
    }
    if project_id:
        arguments["project"] = project_id

    payload = _unwrap(
        await linear.call_tool(
            tool_name="list_issues",
            arguments=arguments,
        )
    )
    issues = payload.get("issues", payload if isinstance(payload, list) else [])

    # Normalise to the dict shape the summary activity expects (preserves the
    # contract from the previous GraphQL version: identifier, title, state.name,
    # priority, updatedAt, assignee.name, url).
    normalised: list[dict] = []
    for issue in issues:
        state_name = (
            issue.get("state", {}).get("name")
            if isinstance(issue.get("state"), dict)
            else issue.get("state")
        )
        assignee = issue.get("assignee")
        if assignee and not isinstance(assignee, dict):
            assignee = {"name": str(assignee)}
        normalised.append(
            {
                "id": issue.get("id"),
                "identifier": issue.get("identifier"),
                "title": issue.get("title", ""),
                "description": issue.get("description") or "",
                "state": {"name": state_name or "Unknown"},
                "priority": issue.get("priority", 0),
                "updatedAt": issue.get("updatedAt"),
                "assignee": assignee,
                "url": issue.get("url"),
            }
        )
    return normalised
