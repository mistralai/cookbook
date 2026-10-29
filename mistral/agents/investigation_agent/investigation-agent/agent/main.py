"""Agent definition deployed as the investigation agent runtime that the workflow opens sessions on."""

from pathlib import Path

from mistralai.agents import agents

from tool_allowlist import CONNECTOR_ALLOWED_TOOLS, READ_ONLY_POLICY

ROOT = Path(__file__).parent

agent = agents.Agent(
    # The workflow loads the agent by this name (workflow/activities/agent.py).
    name="investigation-agent-runtime",
    model="mistral-large-4",
    description="Answers Slack engineering investigation questions with read-only Sentry and GitHub tools.",
    max_iterations=32,
    instructions=(ROOT / "instructions.md").read_text(),
    # Keep only the skill tool. The default sandbox (NoSandbox) gives no file system or shell tools.
    builtins=agents.Builtins(subagents=False, programmatic_tool_calling=False),
    # Read-only connectors configured in the Studio workspace.
    connectors=[agents.connector(name, run_as="auto", auto_auth=False) for name in CONNECTOR_ALLOWED_TOOLS],
    # Denies any tool call that tool_allowlist.py has not reviewed.
    permission_policy=READ_ONLY_POLICY,
    # Each folder in skills/ is a skill bundle shipped with the agent.
    skills=[agents.LocalSkillBundle.from_directory(path) for path in sorted((ROOT / "skills").iterdir())],
    # A deployed worker has no .env: it gets the connector names from these workspace secrets.
    secrets={
        "SENTRY_CONNECTOR_NAME": agents.MistralSecret.workspace("sentry_connector_name", worker_env=True),
        "GITHUB_CONNECTOR_NAME": agents.MistralSecret.workspace("github_connector_name", worker_env=True),
    },
)
