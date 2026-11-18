"""The Slack triage activities: read the message, file it, and reply in its thread.

Intercom and the tracker are Studio HTTP connectors. The connectors gateway adds each connector's stored
credential, so this code holds only the Mistral API key and the connector names.
"""

from datetime import timedelta
from typing import Any, Literal

import httpx
from env.mistral import env as mistral_env
from env.triage import env
from mistralai import workflows
from pydantic import BaseModel
from utils.mistral import mistral_api_url

TIMEOUT = timedelta(seconds=30)
INTERCOM_HEADERS = {"Intercom-Version": "2.11"}


class Classification(BaseModel):
    category: Literal["bug", "feature_request"]
    title: str
    urgency: Literal["low", "medium", "high"]


class SlackMessage(BaseModel):
    channel_id: str
    user_id: str
    ts: str
    text: str
    author: str = ""
    channel: str = ""
    permalink: str = ""


async def _slack(method: str, **params: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=20) as client:
        res = await client.post(
            f"https://slack.com/api/{method}",
            headers={"Authorization": f"Bearer {env.require('slack_bot_token')}"},
            data=params,
        )
    body: dict[str, Any] = res.json()
    if not body.get("ok"):
        raise RuntimeError(f"Slack {method} failed: {body.get('error')}")
    return body


async def call_connector(name: str, method: str, path: str, **kwargs: Any) -> Any:
    """Call an HTTP connector's API at ``path`` through the Studio connectors gateway."""
    if not mistral_env.mistral_api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set")
    async with httpx.AsyncClient(timeout=20) as client:
        res = await client.request(
            method,
            f"{mistral_api_url()}/v1/connectors-gateway/{name}/http{path}",
            headers={
                "Authorization": f"Bearer {mistral_env.mistral_api_key}",
                **kwargs.pop("headers", {}),
            },
            **kwargs,
        )
    res.raise_for_status()
    return res.json()


@workflows.activity(start_to_close_timeout=TIMEOUT)
async def describe_message(message: SlackMessage) -> SlackMessage:
    user = (await _slack("users.info", user=message.user_id))["user"]
    channel = (await _slack("conversations.info", channel=message.channel_id))[
        "channel"
    ]
    link = await _slack(
        "chat.getPermalink", channel=message.channel_id, message_ts=message.ts
    )
    return message.model_copy(
        update={
            "author": user["profile"].get("display_name")
            or user.get("real_name")
            or user["name"],
            "channel": channel["name"],
            "permalink": link["permalink"],
        }
    )


# Not retried: a timeout after the record was created would file it twice.
@workflows.activity(start_to_close_timeout=TIMEOUT, retry_policy_max_attempts=1)
async def create_feature_request(message: SlackMessage, triage: Classification) -> str:
    record = await call_connector(
        env.require("tracker_connector_name"),
        "POST",
        "/api/v1/feature_requests",
        json={
            "summary": triage.title,
            "description": message.text,
            "customer": message.author,
            "source_channel": f"#{message.channel}",
            "slack_thread_url": message.permalink,
            "urgency": triage.urgency,
        },
    )
    return f"Feature request #{record['id']}"


@workflows.activity(start_to_close_timeout=TIMEOUT, retry_policy_max_attempts=1)
async def create_bug_ticket(message: SlackMessage, triage: Classification) -> str:
    intercom = env.require("intercom_connector_name")

    # A ticket needs an existing contact; a 409 means one was created for this Slack user before.
    external_id = f"slack-{message.user_id}"
    try:
        await call_connector(
            intercom,
            "POST",
            "/contacts",
            headers=INTERCOM_HEADERS,
            json={"role": "user", "external_id": external_id, "name": message.author},
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 409:
            raise

    ticket = await call_connector(
        intercom,
        "POST",
        "/tickets",
        headers=INTERCOM_HEADERS,
        json={
            "ticket_type_id": env.require("intercom_ticket_type_id"),
            "contacts": [{"external_id": external_id}],
            "ticket_attributes": {
                "_default_title_": triage.title,
                "_default_description_": (
                    f"{message.text}\n\nUrgency: {triage.urgency}\nReported in #{message.channel}: {message.permalink}"
                ),
            },
        },
    )
    workspace = env.require("intercom_workspace_id")
    app_url = (env.intercom_app_url or "https://app.intercom.com").rstrip("/")
    return f"{app_url}/a/inbox/{workspace}/inbox/conversation/{ticket['id']}"


@workflows.activity(start_to_close_timeout=TIMEOUT)
async def reply_in_thread(message: SlackMessage, text: str) -> None:
    await _slack(
        "chat.postMessage", channel=message.channel_id, thread_ts=message.ts, text=text
    )
