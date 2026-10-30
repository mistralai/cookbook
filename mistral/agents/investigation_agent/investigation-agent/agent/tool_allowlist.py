"""Fail-closed tool policy: the agent may only call reviewed read-only tools, each from its own connector."""

import os

from mistralai.agents import agents


def connector_name(env_var: str) -> str:
    """Read a connector name from the environment, failing loudly so a typo cannot silently drop a connector."""
    name = os.environ.get(env_var, "").strip()
    if not name:
        raise RuntimeError(f"{env_var} is not set. Set it to the name of your connector in the Studio workspace (see README).")
    return name


# Studio gives every connector a unique name, so each workspace supplies its own: from .env locally, from the workspace secrets declared in main.py on a deployed worker.
SENTRY_CONNECTOR = connector_name("SENTRY_CONNECTOR_NAME")
GITHUB_CONNECTOR = connector_name("GITHUB_CONNECTOR_NAME")

SENTRY_ALLOWED_TOOLS = frozenset({"find_organizations", "find_projects", "get_sentry_resource", "search_errors", "search_issues"})
GITHUB_ALLOWED_TOOLS = frozenset(
    {
        "get_commit",
        "get_file_contents",
        "get_latest_release",
        "get_release_by_tag",
        "get_tag",
        "issue_read",
        "list_branches",
        "list_commits",
        "list_issues",
        "list_pull_requests",
        "list_releases",
        "list_tags",
        "pull_request_read",
        "search_code",
        "search_commits",
        "search_pull_requests",
        "search_repositories",
    }
)
CONNECTOR_ALLOWED_TOOLS = {SENTRY_CONNECTOR: SENTRY_ALLOWED_TOOLS, GITHUB_CONNECTOR: GITHUB_ALLOWED_TOOLS}


def connector_tool_patterns(connector: str, tools: frozenset[str]) -> list[str]:
    """Return the permission patterns that match these tools of one connector, and of no other connector.

    A connector tool's stable ID is `connector_<connector>.<connector>_<tool>`. The namespace binds each tool to the
    connector it was reviewed for; `*_` matches the connector prefix the model sees in the tool name.
    """
    return [f"connector_{connector}.*_{tool}" for tool in sorted(tools)]


READ_ONLY_POLICY = agents.PermissionPolicy(
    default="deny",
    rules=[
        # The skill tool loads the bundled skills, and the policy checks it like any other tool.
        agents.PermissionRule(tools=["skill.read"], decision="allow"),
        agents.PermissionRule(tools=[p for c, tools in CONNECTOR_ALLOWED_TOOLS.items() for p in connector_tool_patterns(c, tools)], decision="allow"),
    ],
)
