"""Run the slack-triage workflow once on a message already posted in Slack."""

import asyncio
from typing import Annotated

import httpx
import typer
from env.triage import env
from env.workflows import env as workflows_env
from mistralai_capabilities.workflows.client import get_client

app = typer.Typer()


@app.command(name="triage-message", help=__doc__)
def main(
    channel: Annotated[
        str, typer.Argument(help="Slack channel ID, for example C0123456789")
    ],
    ts: Annotated[
        str, typer.Argument(help="Message timestamp, for example 1790327758.660299")
    ],
    deployment: Annotated[
        str | None,
        typer.Option(
            help="Deployment to run on. Defaults to DEPLOYMENT_NAME, your local worker"
        ),
    ] = None,
) -> None:
    res = httpx.get(
        "https://slack.com/api/conversations.history",
        headers={"Authorization": f"Bearer {env.require('slack_bot_token')}"},
        params={"channel": channel, "latest": ts, "inclusive": "true", "limit": "1"},
        timeout=30,
    ).json()
    if not res.get("ok") or not res["messages"]:
        raise SystemExit(f"Could not read the message: {res.get('error', 'not found')}")
    event = {**res["messages"][0], "channel": channel}

    async def run() -> None:
        client = await get_client()
        execution = await client.workflows.execute_workflow_async(
            workflow_identifier="slack-triage",
            input={"body": {"event": event}},
            deployment_name=deployment or workflows_env.deployment_name,
        )
        typer.echo(
            f"Started execution {execution.execution_id}. Follow it in Studio > Workflows > slack-triage."
        )

    asyncio.run(run())
