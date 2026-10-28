"""Fail-closed tool policy: the agent may only call reviewed read-only tools, each from its own connector."""

import os

from mistralai.agents import agents
from mistralai.vibe.harness.core.protocol import ProvidedToolCall, RuntimeBuiltinToolCall, TextContentBlock
from mistralai.vibe.harness.runtime.capabilities.identifiers import normalize_core_tool_identifier
from mistralai.vibe.harness.runtime.capabilities.models import PreToolCallHookInvocation


def connector_name(env_var: str) -> str:
    """Read a connector name from the environment, failing loudly so a typo cannot silently drop a connector."""
    name = os.environ.get(env_var, "").strip()
    if not name:
        raise RuntimeError(f"{env_var} is not set. Set it to the name of your connector in the Studio workspace (see README).")
    return name


# Studio gives every connector a unique name, so each workspace supplies its own: from .env when deploying, from a bound workspace secret at runtime.
SENTRY_CONNECTOR = connector_name("SENTRY_CONNECTOR_NAME")
GITHUB_CONNECTOR = connector_name("GITHUB_CONNECTOR_NAME")

SENTRY_ALLOWED_TOOLS = frozenset({"find_organizations", "find_projects", "get_sentry_resource", "search_events", "search_issues"})
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
# Order resolves tool-name collisions: Sentry wins over GitHub.
CONNECTOR_ALLOWED_TOOLS = {SENTRY_CONNECTOR: SENTRY_ALLOWED_TOOLS, GITHUB_CONNECTOR: GITHUB_ALLOWED_TOOLS}
ALLOWED_TOOLS = frozenset().union(*CONNECTOR_ALLOWED_TOOLS.values())
# Reading the bundled skills is the only runtime builtin the agent needs.
ALLOWED_BUILTINS = frozenset({"skill.read"})


def connector_tool_namespace(connector_name: str) -> str:
    """Return the tool group the SDK assigns to a connector's MCP tools."""
    safe_name = "".join(c if c.isalnum() else "_" for c in connector_name).strip("_")
    return normalize_core_tool_identifier(f"connector_{safe_name or 'connector'}")


# Allowed tool names keyed by tool group, normalized the way the SDK reports them in hook calls.
ALLOWED_BY_TOOL_GROUP = {connector_tool_namespace(c): frozenset(normalize_core_tool_identifier(t) for t in tools) for c, tools in CONNECTOR_ALLOWED_TOOLS.items()}


class ConnectorToolAllowlist(agents.Hook):
    """Refuse every tool call outside the reviewed per-connector read-only set.

    `Agent(enabled_tools=...)` only filters runtime builtins; connector MCP tools reach the model unfiltered, so this hook is the enforcement point.
    Workflow sessions reject hooks with instance state, so the allowlist lives at module level.
    """

    async def pre_tool_call(self, ctx: agents.Context, call: PreToolCallHookInvocation) -> agents.PreToolCallHookContinue | agents.PreToolCallHookSkip:
        """Let an allowed call run with its original arguments, and answer any other call with a refusal."""
        tool = call.tool_call.call
        if isinstance(tool, ProvidedToolCall) and tool.tool_name in ALLOWED_BY_TOOL_GROUP.get(tool.group_name, ()):
            return agents.PreToolCallHookContinue(effective_arguments=tool.arguments)
        if isinstance(tool, RuntimeBuiltinToolCall) and tool.name in ALLOWED_BUILTINS:
            return agents.PreToolCallHookContinue(effective_arguments=tool.arguments)
        name = tool.tool_name if isinstance(tool, ProvidedToolCall) else tool.name
        return agents.PreToolCallHookSkip(reason=[TextContentBlock(text=f"[refused] {name} is not available.")])
