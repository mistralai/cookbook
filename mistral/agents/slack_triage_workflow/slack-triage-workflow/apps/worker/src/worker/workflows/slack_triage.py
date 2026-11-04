from typing import Any

from mistralai import workflows
from mistralai.workflows import workflow
from pydantic import BaseModel

with workflow.unsafe.imports_passed_through():
    from env.triage import env
    from mistralai.client.models import (
        ChatCompletionRequest,
        SystemMessage,
        UserMessage,
    )
    from mistralai.workflows.plugins.mistralai import chat_parse_to_model

    from worker.activities import (
        Classification,
        SlackMessage,
        create_bug_ticket,
        create_feature_request,
        describe_message,
        reply_in_thread,
    )

PROMPT = """You triage messages posted in a customer support Slack channel.
Classify the message as a "bug" (something is broken or behaves unexpectedly) or a
"feature_request" (a request for something new or different). If unsure, call it a bug.
Also give a short title (under 80 characters) and an urgency: low, medium or high."""


class SlackEvent(BaseModel):
    """The HTTP request a Studio Slack trigger delivers; the Slack event is in `body.event`."""

    body: dict[str, Any]


class TriageResult(BaseModel):
    category: str
    title: str
    urgency: str
    reference: str


@workflows.workflow.define(
    name="slack-triage",
    workflow_display_name="Slack triage",
    workflow_description=(
        "Classify a Slack message as a bug or a feature request, file it in Intercom or the tracker, "
        "and reply in the thread with the result."
    ),
)
class SlackTriage:
    @workflows.workflow.entrypoint
    async def run(self, request: SlackEvent) -> TriageResult | None:
        event = request.body["event"]
        # The trigger filter skips these too; this check keeps the bot from answering its own replies
        # if the filter is left out.
        if event.get("bot_id") or event.get("subtype") or event.get("thread_ts"):
            return None
        message = await describe_message(
            SlackMessage(
                channel_id=event["channel"],
                user_id=event["user"],
                ts=event["ts"],
                text=event["text"],
            )
        )

        triage = await chat_parse_to_model(
            Classification,
            ChatCompletionRequest(
                model=env.triage_model,
                messages=[
                    SystemMessage(content=PROMPT),
                    UserMessage(content=message.text),
                ],
            ),
        )

        if triage.category == "bug":
            link = await create_bug_ticket(message, triage)
            reply = f"Thanks {message.author}, this looks like a bug. I opened a support ticket: <{link}|{triage.title}>"
        else:
            link = await create_feature_request(message, triage)
            reply = f"Thanks {message.author}, I filed this as {link}: {triage.title}"

        await reply_in_thread(message, reply)
        return TriageResult(
            category=triage.category,
            title=triage.title,
            urgency=triage.urgency,
            reference=link,
        )
