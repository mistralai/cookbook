"""Agent definition deployed as the investigation agent runtime that the workflow opens sessions on."""

from pathlib import Path

from mistralai.agents import agents

from tool_allowlist import ALLOWED_TOOLS, CONNECTOR_ALLOWED_TOOLS, ConnectorToolAllowlist

ROOT = Path(__file__).parent

agent = agents.Agent(
    # The workflow loads the agent by this name (workflow/activities/agent.py).
    name="investigation-agent-runtime",
    model="mistral-large-4",
    description="Answers Slack engineering investigation questions with read-only Sentry and GitHub tools.",
    max_iterations=32,
    instructions=(ROOT / "instructions.md").read_text(),
    # Matches no runtime builtin, so file system, bash and process tools are all disabled.
    enabled_tools=sorted(ALLOWED_TOOLS),
    harness=agents.Harness(
        # Read-only connectors configured in the Studio workspace.
        connectors=[agents.connector(name, run_as="auto", auto_auth=False) for name in CONNECTOR_ALLOWED_TOOLS],
        # Refuses any connector tool that tool_allowlist.py has not reviewed.
        hooks=[ConnectorToolAllowlist()],
        # Each folder in skills/ is a skill bundle shipped with the agent.
        local_skills=[agents.LocalSkillBundle.from_directory(path) for path in sorted((ROOT / "skills").iterdir())],
    ),
)
