"""Activity that generates a weekly summary from Linear issues.

Uses the built-in Mistral workflows plugin activity
``mistralai_chat_complete`` (async) instead of a custom MCP connector.
"""

import mistralai.workflows.plugins.mistralai as workflows_mistralai
from mistralai import workflows


def _format_issues(issues: list[dict]) -> str:
    """Render the issue list as a human-readable bulleted block for the prompt."""
    formatted = []
    for issue in issues:
        assignee_name = (
            issue.get("assignee", {}).get("name", "Unassigned")
            if issue.get("assignee")
            else "Unassigned"
        )
        formatted.append(
            f"- {issue.get('identifier', 'N/A')}: {issue.get('title', '')} "
            f"(State: {issue.get('state', {}).get('name', 'N/A')}, "
            f"Priority: {issue.get('priority', 0)}, "
            f"Assignee: {assignee_name})"
        )
    return "\n".join(formatted)


@workflows.activity()
async def generate_weekly_summary(
    issues: list[dict],
) -> str:
    """Generate a concise weekly summary from Linear issue data.

    Calls Mistral's async chat completion activity directly.

    Args:
        issues: List of issue dicts from Linear (id, identifier, title,
            state, priority, etc.).

    Returns:
        Summary string from the model.
    """
    if not issues:
        return "No issues were updated in the past week."

    issues_text = _format_issues(issues)

    prompt = f"""Summarize the following Linear issues that had status updates in the past week.
Highlight key updates, status changes, and priorities. Be concise and actionable.

Issues:
{issues_text}

Provide a brief weekly summary (2-4 paragraphs)."""

    request = workflows_mistralai.ChatCompletionRequest(
        model="mistral-medium-latest",
        messages=[workflows_mistralai.UserMessage(content=prompt)],
    )
    response = await workflows_mistralai.mistralai_chat_complete(request)

    try:
        return response.choices[0].message.content or ""
    except (KeyError, IndexError, TypeError) as exc:
        # Surface upstream weirdness so it shows up clearly in the
        # Studio activity event view rather than as a stack trace.
        raise RuntimeError(
            f"Unexpected response shape from mistralai_chat_complete: {response!r}"
        ) from exc
