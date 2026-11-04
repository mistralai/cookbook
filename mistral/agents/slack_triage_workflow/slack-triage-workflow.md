# Triage Slack messages into Intercom and a feature tracker with Mistral Apps

Build a Mistral App that reads every new message in a Slack channel, files bugs as Intercom tickets and feature requests in its own tracker, and replies in the thread with a link.

Support channels mix bug reports with feature requests, and someone has to sort them, file each one in the right place, and tell the customer it was heard. This cookbook builds a Mistral App that does that for every new message in a Slack channel. A Studio trigger starts the `slack-triage` workflow, which classifies the message with `mistral-medium-latest`. Bugs become Intercom tickets, and feature requests go into a tracker API that the app hosts on its own Postgres database. The workflow then replies in the thread with a link.

You'll build the app from the `mistral apps init` scaffold, which gives you three modules:

| Path | What it is |
|---|---|
| `apps/api` | A FastAPI service at `/api`. You'll add the tracker: create, list, and get feature requests. |
| `apps/worker` | The managed **Workflows** worker. You'll add the `slack-triage` workflow and its Slack, Intercom, and tracker activities. |
| `packages/py/db` | The managed Postgres database. You'll add the `feature_request` table, its migration, and its accessors. |

The workflow reaches Intercom and the tracker through two Studio HTTP **Connectors**. Studio stores their credentials, so the worker never holds an Intercom token.

Here's how a message flows through the system:

```mermaid
graph LR
    Slack["Slack: a new message in the channel"] -->|message event| Trigger["Studio webhook trigger<br/>(triggers/slack.json filter)"]
    Trigger -->|starts| Workflow["slack-triage workflow"]
    Workflow -->|classify| Model["mistral-medium-latest"]
    Workflow -->|bug| Intercom["Intercom connector:<br/>ticket"]
    Workflow -->|feature request| Tracker["Tracker connector:<br/>the app's /api/v1/feature_requests"]
    Workflow -->|replies in the thread| Slack
```

Intercom's 14-day free trial and a free Slack workspace are enough to run the whole thing.

This cookbook walks you through creating every file. You can view [the complete project on GitHub](https://github.com/mistralai/cookbook/tree/main/mistral/agents/slack_triage_workflow/slack-triage-workflow).

## Prerequisites

To complete this cookbook, you will need:

- [uv](https://docs.astral.sh/uv/), which installs the Python version the project pins
- Docker, for the local Postgres database
- The `mistral` CLI with access to Mistral Apps
- git, and a GitHub account with the Mistral GitHub app installed on the repository you'll push to. Deploys build from that repository.
- A Mistral account with access to Studio, Workflows, and Connectors, and a Mistral API key from the workspace you'll deploy to
- A Slack workspace where you can create and install apps, with a public channel to triage
- An Intercom workspace. The free trial is enough.

## Environment setup

Scaffold the app, look at what you'll build, and add your first settings.

### Install

Sign in to the Mistral CLI, then scaffold the app with the FastAPI, Postgres, Workflows, and Apps capabilities:

```bash
mistral login
mistral apps init slack-triage-workflow --caps fastapi,postgres,workflows,apps
cd slack-triage-workflow
```

`mistral apps init` creates the project folder, a git repository with a first commit, and a `.env` that holds the settings each capability needs. Unless a step says otherwise, run every command from this folder.

### Create the project

By the end of the cookbook, you'll have added these files to the scaffold:

```text
slack-triage-workflow/
├── apps/
│   ├── api/src/api/routers/api/v1/
│   │   ├── __init__.py
│   │   └── feature_requests.py
│   └── worker/src/worker/
│       ├── activities.py
│       └── workflows/slack_triage.py
├── packages/py/
│   ├── cli/src/cli/commands/
│   │   ├── bind_secrets.py
│   │   ├── connector.py
│   │   ├── intercom_ids.py
│   │   └── triage_message.py
│   ├── db/src/db/
│   │   ├── accessors/feature_request.py
│   │   ├── migrations/versions/0001_feature_request.py
│   │   └── models/feature_request.py
│   └── env/src/env/triage.py
└── triggers/
    └── slack.json
```

You'll also edit a few scaffold files along the way.

### Required environment variables

To complete this cookbook, you'll need a Mistral API key. In [Studio](https://console.mistral.ai), navigate to the [API keys section](https://console.mistral.ai/home?profile_dialog=api-keys) and create a new API key. Create it in the workspace you'll deploy the app to. The app's gateway only accepts keys from that workspace.

`mistral apps init` already created a `.env` at the root of your project, and it's gitignored. Add your Mistral API key, and pick a deployment name for the worker you'll run on your machine:

```
MISTRAL_API_KEY=your-mistral-api-key
DEPLOYMENT_NAME=slack-triage-your-name
```

Both lines are already in `.env`, so edit them rather than adding new lines. `DEPLOYMENT_NAME` starts with a default built from your username, which you can keep or replace with a name of your own. You'll add the other settings to `.env` as you go.

`mistral apps init` also asks you to fill in `POSTGRES_HOST`, `POSTGRES_PASSWORD`, and `POSTGRES_SSL_MODE`. Leave them empty. On your machine, the app connects through `DATABASE_URL` to the local Postgres that `mistral apps dev` starts, and the deployed app gets its Postgres settings from Mistral Apps.

## Step 1: Trim the scaffold

The scaffold includes JavaScript tooling (bun and Nx) and documentation for coding agents. This app is Python only, so remove them.

### 1.1 Remove the JavaScript tooling

Delete the Nx and bun files, the packages bun installed, the shell scripts that call them, and the `.agents/` folder:

```bash
rm -rf .agents node_modules .nxignore package.json bun.lock bunfig.toml nx.json \
  tools/install.sh tools/db.sh tools/lib.sh \
  apps/api/project.json apps/worker/project.json packages/py/db/project.json
```

Running `mistral apps capability add` later may bring some of these files back.

### 1.2 Update the files that referred to them

Six scaffold files still refer to the tooling you removed. In `.dockerignore`, delete these two lines:

```text
**/.nx
.nx
```

In `.gitignore`, delete the `.nx` line under `# Nx`.

The `migrations` command prints a hint that names an Nx target. In `packages/py/cli/src/cli/commands/migrations.py`, change the `hint` in the `no_migrations_present` log call to:

```python
            hint="no selected capability owns a table; add a model and an Alembic revision for it",
```

The settings package finds the project root by looking for `package.json`, which no longer exists. In `packages/py/env/src/env/_base.py`, make `_find_repo_root` look for `apps.json` instead:

```python
def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "apps.json").is_file() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()
```

Without this change, the settings miss your `.env` when a command runs from a subfolder.

In `tools/uv.sh`, two comments name `tools/install.sh`. In the comment that starts with `# Executed (`, replace its first two lines with:

```bash
# Executed (`bash tools/uv.sh <args>`) -> strict mode, then exec uv. Sourced (by hand from bash or
# zsh) -> only export the environment and leave the caller's shell options alone:
```

Then replace the three comment lines that start with `# Executed directly (an NX` with one line:

```bash
# Executed directly -> run uv with this environment. Sourced -> only export the environment.
```

The live-Postgres migration test names an Nx target in its docstring. In `packages/py/db/tests/test_migrations_pg.py`, delete `` (``bunx nx run db:test-pg-contract``)`` so the line reads:

```text
``DATABASE_URL``; skipped otherwise.
```

## Step 2: Add the dependencies

The worker needs the Mistral plugin for Workflows, which parses model output into a Pydantic model, and both the worker and the CLI call HTTP APIs with httpx.

In `apps/worker/pyproject.toml`, make the dependencies list read as follows. It needs `mistralai-workflows` 3.16 or later with the `mistralai` extra, and httpx:

```toml
dependencies = [
    "utils",
    "env",
    "mistralai>=2.9.4",
    "mistralai-workflows[mistralai]>=3.16.0",
    "httpx>=0.28.1",
    "pydantic>=2.14.1",
    "structlog>=25.5.0",
    "watchfiles>=1.3",
]
```

In `packages/py/cli/pyproject.toml`, add httpx at the end of the dependencies list:

```toml
dependencies = [
    "typer>=0.27",
    "utils",
    "env",
    "mistralai>=2.9.4",
    "structlog>=25.5.0",
    "httpx>=0.28.1",
]
```

Switch the worker entry point to the same `workflows` import the workflow and activities use. In `apps/worker/src/worker/entrypoints/worker.py`, replace `import mistralai.workflows as workflows` so the imports read:

```python
import asyncio
import signal

import structlog
from env.logging import env
from mistralai import workflows
from mistralai_capabilities.workflows import client as workflows_client
from utils.logging import configure_logging
```

Install everything:

```bash
uv sync --all-packages
```

## Step 3: Add the settings

The workflow reads its settings from `.env` on your machine. Once deployed, the worker gets the same variables from workspace secrets, which you bind in Step 16.

Create the settings module. The connector names, the Intercom IDs, and the Slack token have no defaults, so nothing points at someone else's workspace by accident:

**`packages/py/env/src/env/triage.py`**

```python
"""Settings for the Slack triage workflow.

The connector names, the Intercom IDs and the Slack token have no defaults: set them in `.env` locally,
or as workspace secrets for the deployed worker.
"""

from typing import Annotated

from env._base import BaseEnv
from env._mistral_secrets import WorkspaceSecret


class Env(BaseEnv):
    triage_model: str = "mistral-medium-latest"
    slack_bot_token: Annotated[str | None, WorkspaceSecret()] = None
    intercom_connector_name: Annotated[str | None, WorkspaceSecret()] = None
    intercom_ticket_type_id: Annotated[str | None, WorkspaceSecret()] = None
    intercom_workspace_id: Annotated[str | None, WorkspaceSecret()] = None
    tracker_connector_name: Annotated[str | None, WorkspaceSecret()] = None
    # Where ticket links in Slack replies point when not https://app.intercom.com, such as
    # https://app.eu.intercom.com for an EU workspace.
    intercom_app_url: str | None = None

    def require(self, name: str) -> str:
        """The value of setting ``name``, or an error naming the variable to set."""
        value = getattr(self, name)
        if not value:
            raise RuntimeError(f"{name.upper()} is not set")
        return value


env = Env()
```

`TRIAGE_MODEL` defaults to `mistral-medium-latest`. To try another model, add it to `.env`:

```
TRIAGE_MODEL=mistral-medium-latest
```

`.env` is gitignored, so the scaffold's `.env.example` is where the project records the settings its own code reads. Add the new settings at the end of `.env.example`, names only, with no values:

```
TRIAGE_MODEL=
SLACK_BOT_TOKEN=
INTERCOM_CONNECTOR_NAME=
INTERCOM_TICKET_TYPE_ID=
INTERCOM_WORKSPACE_ID=
TRACKER_CONNECTOR_NAME=
INTERCOM_APP_URL=
```

## Step 4: Add the feature request table

The tracker stores feature requests in the app's managed Postgres database. The scaffold discovers every model in `db/models/` and applies every migration in `db/migrations/versions/`, so adding a table means adding files, not editing a list.

### 4.1 Add the model

Create the model for the `feature_request` table:

**`packages/py/db/src/db/models/feature_request.py`**

```python
"""A feature request filed in the tracker."""

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(UTC)


class FeatureRequest(SQLModel, table=True):
    __tablename__ = "feature_request"

    id: int | None = Field(default=None, primary_key=True)
    summary: str
    description: str
    customer: str
    source_channel: str
    slack_thread_url: str
    urgency: str
    status: str = "open"
    created_at: datetime = Field(default_factory=_now)
```

### 4.2 Add the migration

Create the migration that creates the table. The `branch_labels` entry keeps it on its own branch, as the scaffold expects from every table owner:

**`packages/py/db/src/db/migrations/versions/0001_feature_request.py`**

```python
"""Create the feature_request table.

Revision ID: 0001
Revises:
Create Date: 2026-10-27

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = ("feature_request",)
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feature_request",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("summary", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("customer", sa.String(), nullable=False),
        sa.Column("source_channel", sa.String(), nullable=False),
        sa.Column("slack_thread_url", sa.String(), nullable=False),
        sa.Column("urgency", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("feature_request")
```

`mistral apps dev` applies it locally, and the deployed API applies it before it starts.

### 4.3 Add the accessors

Create the functions the API uses to read and write the table:

**`packages/py/db/src/db/accessors/feature_request.py`**

```python
"""Reads and writes of the ``feature_request`` table."""

from sqlmodel import col, select

from db.engine import get_session_maker
from db.models.feature_request import FeatureRequest


async def create_feature_request(request: FeatureRequest) -> FeatureRequest:
    async with get_session_maker()() as session:
        session.add(request)
        await session.commit()
        await session.refresh(request)
    return request


async def get_feature_request(request_id: int) -> FeatureRequest | None:
    async with get_session_maker()() as session:
        return await session.get(FeatureRequest, request_id)


async def list_feature_requests(limit: int = 50) -> list[FeatureRequest]:
    async with get_session_maker()() as session:
        rows = await session.execute(
            select(FeatureRequest).order_by(col(FeatureRequest.id).desc()).limit(limit)
        )
        return list(rows.scalars())
```

## Step 5: Add the tracker API

The tracker is a small REST API in the app's FastAPI service. The scaffold mounts every router module under `routers/` at its folder path, so the files below serve `/api/v1/feature_requests`.

### 5.1 Require the Apps gateway

A deployed app sits behind the Apps gateway, which signs each caller in and forwards a token. The `dependencies` in a router package's `__init__.py` apply to every route in it, so this check rejects any `/api/v1` request that didn't come through the gateway:

**`apps/api/src/api/routers/api/v1/__init__.py`**

```python
"""The tracker API. Every route needs a caller signed in through the Apps gateway."""

from fastapi import HTTPException, Request, params, status
from mistralai_capabilities.apps.tokens import token_from


def require_caller(request: Request) -> None:
    # The gateway signs the caller in and forwards a token; a request without one bypassed it.
    if not token_from(request.headers):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Call this API through the app's gateway"
        )


dependencies: tuple[params.Depends, ...] = (params.Depends(require_caller),)
```

### 5.2 Add the routes

Create the routes that file, list, and get feature requests:

**`apps/api/src/api/routers/api/v1/feature_requests.py`**

```python
from datetime import datetime
from typing import Literal

from db.accessors.feature_request import (
    create_feature_request,
    get_feature_request,
    list_feature_requests,
)
from db.models.feature_request import FeatureRequest
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

router = APIRouter(tags=["feature_requests"])

Urgency = Literal["low", "medium", "high"]


class NewFeatureRequest(BaseModel):
    summary: str
    description: str
    customer: str
    source_channel: str
    slack_thread_url: str
    urgency: Urgency


class FeatureRequestOut(NewFeatureRequest):
    id: int
    status: str
    created_at: datetime


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: NewFeatureRequest) -> FeatureRequestOut:
    """File a feature request and return it with its ID."""
    record = await create_feature_request(FeatureRequest(**body.model_dump()))
    return FeatureRequestOut.model_validate(record, from_attributes=True)


@router.get("")
async def list_all(limit: int = 50) -> list[FeatureRequestOut]:
    """List the most recent feature requests, newest first."""
    return [
        FeatureRequestOut.model_validate(r, from_attributes=True)
        for r in await list_feature_requests(limit)
    ]


@router.get("/{request_id}")
async def get(request_id: int) -> FeatureRequestOut:
    """Get one feature request by ID."""
    record = await get_feature_request(request_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such feature request")
    return FeatureRequestOut.model_validate(record, from_attributes=True)
```

## Step 6: Add the workflow

The worker discovers every workflow class in `worker/workflows/`. The workflow calls activities for everything that touches the outside world, so a retry never repeats a model call or a Slack post that already succeeded.

### 6.1 Add the activities

The activities read the message's author, channel, and link from Slack, file the bug or the feature request, and reply in the thread. `call_connector` reaches Intercom and the tracker through the Studio connectors gateway at `/v1/connectors-gateway/<name>/http<path>`. The gateway adds the connector's stored credential to the request, so this code holds only the Mistral API key and the connector names.

**`apps/worker/src/worker/activities.py`**

```python
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
```

The two activities that file items run once. A retry after a timeout could file the same item twice. An Intercom ticket also needs a contact, so `create_bug_ticket` creates one per Slack user the first time that user reports a bug.

### 6.2 Add the workflow

The workflow takes the HTTP request the Studio trigger delivers, so the Slack event is at `body.event`. It classifies the message with `chat_parse_to_model`, which returns a `Classification`, then files it and replies:

**`apps/worker/src/worker/workflows/slack_triage.py`**

```python
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
```

## Step 7: Add the setup commands

The scaffold's CLI mounts every module in `cli/commands/` as a command of `python -m cli`. Add four commands for the setup steps Studio's UI doesn't cover.

### 7.1 Create HTTP connectors

Triggered runs have no signed-in user, so they can only use a connector credential shared with the workspace. A connector created in Studio's UI takes a user credential, so this command creates the connector and a workspace credential with the SDK. It sends the token as a bearer token by default, or in the header you name with `--header`:

**`packages/py/cli/src/cli/commands/connector.py`**

```python
"""Create a Studio HTTP connector whose credential is shared with the workspace."""

from typing import Annotated

import typer
from mistralai.client.models import (
    BearerAuthMethod,
    ConnectorAuthenticationHeader,
    CreateHTTPConnectorRequest,
    NoneAuthMethod,
)
from utils.mistral import mistral_client

app = typer.Typer()


@app.command(name="connector", help=__doc__)
def main(
    name: Annotated[
        str, typer.Argument(help="Connector name, for example intercom or tracker")
    ],
    url: Annotated[
        str, typer.Argument(help="API base URL, for example https://api.intercom.io")
    ],
    token: Annotated[
        str,
        typer.Option(prompt=True, hide_input=True, help="Token the connector sends"),
    ],
    header: Annotated[
        str | None,
        typer.Option(
            help="Send the token in this header, such as X-Api-Key, instead of as a bearer token"
        ),
    ] = None,
) -> None:
    if header:
        auth = NoneAuthMethod(headers=[ConnectorAuthenticationHeader(name=header)])
        credentials = {"headers": {header: token.strip()}}
    else:
        auth = BearerAuthMethod()
        credentials = {"bearer_token": token.strip()}
    client = mistral_client()
    client.beta.connectors.create(
        request=CreateHTTPConnectorRequest(
            name=name,
            title=name,
            description=f"{name} API",
            server=url,
            visibility="shared_workspace",
            auth_methods=[auth],
        )
    )
    # Workspace scope: a credential scoped to your user is invisible to triggered runs.
    client.beta.connectors.create_credentials(
        connector_id_or_name=name,
        consumer_scope="workspace",
        name="default",
        credentials=credentials,
    )
    typer.echo(f"Created connector {name} -> {url} with a workspace credential")
```

### 7.2 Look up the Intercom IDs

This command prints the Intercom workspace ID and ticket type IDs. It calls Intercom through the worker's `call_connector`, so it also checks that the connector and its credential work:

**`packages/py/cli/src/cli/commands/intercom_ids.py`**

```python
"""Print the Intercom workspace ID and the ticket type IDs to put in `.env`."""

import asyncio

import typer
from env.triage import env

app = typer.Typer()


@app.command(name="intercom-ids", help=__doc__)
def main() -> None:
    # The worker's connector client, so this also checks that the connector and its credential work.
    from worker.activities import INTERCOM_HEADERS, call_connector

    async def fetch() -> tuple[dict, dict]:
        name = env.require("intercom_connector_name")
        me = await call_connector(name, "GET", "/me", headers=INTERCOM_HEADERS)
        types = await call_connector(
            name, "GET", "/ticket_types", headers=INTERCOM_HEADERS
        )
        return me, types

    me, types = asyncio.run(fetch())
    typer.echo(f"INTERCOM_WORKSPACE_ID={me['app']['id_code']}")
    for ticket_type in types["data"]:
        if not ticket_type["archived"]:
            typer.echo(
                f"INTERCOM_TICKET_TYPE_ID={ticket_type['id']}  # {ticket_type['name']} ({ticket_type['category']})"
            )
```

### 7.3 Run the workflow on one message

This command reads a message already posted in Slack and starts the workflow on it, the same way a trigger would. It runs on your local worker unless you pass `--deployment`:

**`packages/py/cli/src/cli/commands/triage_message.py`**

```python
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
```

### 7.4 Bind the workspace secrets

`mistral apps deploy` binds only the database settings to the managed worker. This command binds the workflow's settings to the workspace secrets of the same name, and keeps the bindings deploy made:

**`packages/py/cli/src/cli/commands/bind_secrets.py`**

```python
"""Bind the workflow's workspace secrets to the managed worker deployment.

`mistral apps deploy` binds only the database settings, and a deployed worker has no `.env`. Each
variable below is bound to the workspace secret of the same name, so create those secrets first.
"""

from typing import Annotated, Any

import httpx
import typer
from env.mistral import env as mistral_env
from env.triage import env
from utils.mistral import mistral_api_url

app = typer.Typer()

SECRETS = (
    "MISTRAL_API_KEY",
    "SLACK_BOT_TOKEN",
    "INTERCOM_CONNECTOR_NAME",
    "INTERCOM_TICKET_TYPE_ID",
    "INTERCOM_WORKSPACE_ID",
    "TRACKER_CONNECTOR_NAME",
)


@app.command(name="bind-secrets", help=__doc__)
def main(
    deployment: Annotated[
        str, typer.Argument(help="Worker deployment name, for example my-app-worker")
    ],
) -> None:
    url = f"{mistral_api_url()}/v1/workflows/deployments/{deployment}"
    headers = {"Authorization": f"Bearer {mistral_env.mistral_api_key}"}
    current = httpx.get(url, headers=headers, timeout=30).raise_for_status().json()
    backend: dict[str, Any] = dict(
        (current.get("spec") or {}).get("backend_spec") or {}
    )
    # The PATCH replaces the whole list, so keep the bindings deploy made, such as POSTGRES_*.
    bindings = {b["env_var_name"]: b for b in backend.get("secrets") or []}
    # INTERCOM_APP_URL is optional, so bind it only when it's set here and its secret should exist.
    names = (*SECRETS, "INTERCOM_APP_URL") if env.intercom_app_url else SECRETS
    bindings |= {
        name: {"env_var_name": name, "reference": f"secret:workspace:{name}"}
        for name in names
    }
    backend["secrets"] = list(bindings.values())
    httpx.patch(
        url, headers=headers, json={"spec": {"backend_spec": backend}}, timeout=30
    ).raise_for_status()
    typer.echo(f"Bound {', '.join(names)} on {deployment}")
```

## Step 8: Add the trigger filter

Slack sends an event for every message in the channels the bot is in, including the bot's own replies, edits, and thread replies. The Studio trigger uses this filter to start the workflow only for new top-level messages from people. The workflow applies the same rule as a backstop.

**`triggers/slack.json`**

```json
{
  "message": "event.type == 'message' && event.text && event.bot_id == null && event.subtype == null && event.thread_ts == null"
}
```

You'll paste the filter expression into Studio in Step 16.

Format the project with Ruff, so the scaffold files you edited match the code style of the new ones:

```bash
uvx ruff format apps packages
```

Then check that the CLI finds the new commands:

```bash
uv run --all-packages python -m cli --help
```

The command list includes `bind-secrets`, `connector`, `intercom-ids`, and `triage-message`, next to the scaffold's `migrations` and `pre-start`.

## Step 9: Push the project to GitHub

Deploys build from the pushed commit, not from your local files. Create an empty GitHub repository (for example `slack-triage-workflow`), install the Mistral GitHub app on it, then push:

```bash
git remote add origin https://github.com/<your-username>/slack-triage-workflow.git
git add -A
git commit -m "Add the Slack triage workflow"
git push -u origin main
```

`.env` is gitignored, so your settings stay on your machine.

## Step 10: Create the Slack app

Create the Slack app whose bot reads the channel and replies:

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and click **Create New App > From scratch**. Name it (for example *Triage Bot*) and pick your workspace.
2. Under **OAuth & Permissions > Scopes > Bot Token Scopes**, add:

   | Scope | Used for |
   |---|---|
   | `channels:history` | Reading the message |
   | `channels:read` | Reading the channel name |
   | `chat:write` | Replying in the thread |
   | `users:read` | Reading the author's name |

3. Click **Install to Workspace** and approve. Copy the **Bot User OAuth Token** (`xoxb-...`) into `.env`:

   ```
   SLACK_BOT_TOKEN=xoxb-your-bot-token
   ```

4. Under **Basic Information > App Credentials**, copy the **Signing Secret**. You'll need it in Step 16.
5. In the channel you want to triage, invite the bot with `/invite @Triage Bot`. Slack only shares messages from channels the bot is in.

Leave **Event Subscriptions** for now. You'll set them up once the workflow is deployed.

## Step 11: Set up Intercom

The workflow files bugs as Intercom tickets, so Intercom needs an access token and a ticket type:

1. Start an Intercom trial, or use a workspace you own.
2. In the **Developer Hub**, open your app (or create one), go to **Authentication**, and copy the access token.
3. In **Settings > Inbox > Tickets**, check that a ticket type in the **Customer** category exists, and create one if it doesn't. Bug reports become tickets of this type.
4. Note your region from the address bar: `app.intercom.com` is the US, `app.eu.intercom.com` the EU, and `app.au.intercom.com` Australia.

## Step 12: Create the Intercom connector

Create the Intercom connector with your region's API address: `https://api.intercom.io` for the US, `https://api.eu.intercom.io` for the EU, or `https://api.au.intercom.io` for Australia. The command prompts for the Intercom access token and doesn't echo it:

```bash
uv run --all-packages python -m cli connector intercom https://api.intercom.io
```

```text
Token:
Created connector intercom -> https://api.intercom.io with a workspace credential
```

Connector names are unique in a workspace. If `intercom` is taken, pick another name. Add the name to `.env`. Outside the US, also set `INTERCOM_APP_URL` to your region's app address, so the ticket links in replies open:

```
INTERCOM_CONNECTOR_NAME=intercom
INTERCOM_APP_URL=https://app.eu.intercom.com
```

Then look up the Intercom IDs:

```bash
uv run --all-packages python -m cli intercom-ids
```

```text
INTERCOM_WORKSPACE_ID=abcd1234
INTERCOM_TICKET_TYPE_ID=1234567  # Bug report (Customer)
```

Copy the `INTERCOM_WORKSPACE_ID` line and the ticket type you want into `.env`:

```
INTERCOM_TICKET_TYPE_ID=1234567
INTERCOM_WORKSPACE_ID=abcd1234
```

Both IDs are also in Intercom's address bar. The workspace ID follows `/apps/`, and opening the ticket type in **Settings > Inbox > Tickets** shows its ID at the end.

## Step 13: Deploy the app

Deploy the app to Mistral Apps:

```bash
mistral apps deploy
```

The command asks which organization and workspace to deploy to. Pick the workspace your API key belongs to. It builds the API and the worker from the commit you pushed, sets up the managed Postgres database, and prints the app's URL. It also creates the worker deployment `<app>-worker`, where `<app>` is your app's name, and adds a `deployment` block to `apps.json` that records the target. The command offers to commit and push `apps.json` for you. If you declined, commit and push it yourself, so the next deploy goes to the same app:

```bash
git add apps.json
git commit -m "Save the Mistral Apps deployment target"
git push
```

## Step 14: Create the tracker connector

The tracker connector calls the app's API through the Apps gateway. The gateway lets a request through when it carries a Mistral API key from the app's workspace in an `X-Api-Key` header, so the connector stores your key and sends it in that header.

Use the app's URL as the connector's base, without `/api`, since the routes already start with it. The command prompts for your Mistral API key:

```bash
uv run --all-packages python -m cli connector tracker https://<app-url> --header X-Api-Key
```

```text
Token:
Created connector tracker -> https://<app-url> with a workspace credential
```

Add the connector's name to `.env`:

```
TRACKER_CONNECTOR_NAME=tracker
```

Check that the tracker answers through the gateway. Run `source .env` first if `MISTRAL_API_KEY` isn't set in your shell, and replace `<app-url>` with your app's URL:

```bash
curl -H "X-Api-Key: $MISTRAL_API_KEY" https://<app-url>/api/v1/feature_requests
```

```json
[]
```

The list is empty until the first feature request is filed.

## Step 15: Try it on one message

Run the workflow on your machine before connecting Slack to it. Start the API, the worker, and a local Postgres database:

```bash
mistral apps dev
```

Your local worker registers on the deployment you named in `DEPLOYMENT_NAME`. Post a message in the channel, such as *"The export button gives me a blank page since this morning"*. Copy the message link (**Copy link** in the message menu). It has the form `.../archives/<channel id>/p<timestamp without the dot>`. In a second terminal, pass the channel ID and the timestamp, with a dot before the last six digits:

```bash
uv run --all-packages python -m cli triage-message C0123456789 1790327758.660299
```

```text
Started execution 0a1b2c3d-.... Follow it in Studio > Workflows > slack-triage.
```

Within a few seconds, the bot replies in the thread with a link to the new Intercom ticket. Try a feature request too, such as *"Could you add a dark mode to the dashboard?"*. The reply names the new feature request, which the tracker check from Step 14 now lists.

Stop `mistral apps dev` when you're done. A local worker and the deployed one must not both triage the same channel, or both reply.

## Step 16: Connect Slack to the workflow

A deployed worker has no `.env`, so it reads its settings from workspace secrets. In Studio, open **Secrets** and create a workspace secret for each of these, with the same values as in `.env`:

- `MISTRAL_API_KEY`
- `SLACK_BOT_TOKEN`
- `INTERCOM_CONNECTOR_NAME`
- `INTERCOM_TICKET_TYPE_ID`
- `INTERCOM_WORKSPACE_ID`
- `TRACKER_CONNECTOR_NAME`
- `INTERCOM_APP_URL`, only if you set it

Bind them to the deployed worker:

```bash
uv run --all-packages python -m cli bind-secrets <app>-worker
```

```text
Bound MISTRAL_API_KEY, SLACK_BOT_TOKEN, INTERCOM_CONNECTOR_NAME, INTERCOM_TICKET_TYPE_ID, INTERCOM_WORKSPACE_ID, TRACKER_CONNECTOR_NAME on <app>-worker
```

Then connect Slack to the workflow with a Studio webhook and trigger:

1. **Create the webhook.** In Studio, go to **Webhooks > Create**, pick **Slack** as the provider, and paste the app's **Signing Secret** from Step 10. Copy the URL Studio shows. Treat this URL as a credential.
2. **Create the trigger.** Go to **Triggers > Create**, pick that webhook, and choose the action **Start workflow** with `slack-triage`. As the filter, paste the expression from `triggers/slack.json`:

   ```text
   event.type == 'message' && event.text && event.bot_id == null && event.subtype == null && event.thread_ts == null
   ```

3. **Point Slack at the webhook.** In your Slack app, open **Event Subscriptions** and turn on **Enable Events**. Paste the webhook URL as the **Request URL** and wait for it to show as verified. Under **Subscribe to bot events**, add `message.channels`, then click **Save Changes**. If Slack asks you to reinstall the app, do so.

## Step 17: Try it in Slack

Post a bug report or a feature request in the channel. The trigger starts the deployed workflow, and the reply lands in the thread within a few seconds:

> Thanks Ada, this looks like a bug. I opened a support ticket: Export button shows a blank page

Each run appears in **Studio > Workflows > slack-triage**, with its input, the classification, and each activity.

## Updating, operating and cleaning up

Use these steps to ship changes and remove everything you created.

**Shipping a change**

Commit and push, then run `mistral apps deploy` again. If the worker then reports a missing setting, run `bind-secrets` again. To add a setting, create its workspace secret and add it to `SECRETS` in `bind_secrets.py` before you run `bind-secrets`.

**Rules of thumb**

- Never log Slack message text, event payloads, tokens, or the webhook URL.
- Never hardcode connector names. They come only from `INTERCOM_CONNECTOR_NAME` and `TRACKER_CONNECTOR_NAME`.
- Never run a local worker while the trigger is active on the same channel.

**Cleaning up**

In Studio, delete the trigger, the webhook, the two connectors, and the workspace secrets, then delete the app and its worker deployment. Uninstall the Slack app from your workspace, and cancel the Intercom trial if you started one.

## Summary

This cookbook built a Mistral App that triages a Slack channel: a workflow classifies each new message, files bugs as Intercom tickets and feature requests in the app's own tracker, and replies in the thread with a link.

**What you built**

- A Mistral App with a FastAPI tracker API on a managed Postgres database, behind the Apps gateway
- A `slack-triage` workflow that classifies messages with structured output and files them through two HTTP connectors
- CLI commands that create HTTP connectors with workspace credentials, look up Intercom IDs, run the workflow on one message, and bind workspace secrets
- A webhook trigger with a filter that starts the workflow only for new messages from people

**Mistral features used**

- Mistral Apps and the `mistral` CLI (`mistral apps init`, `mistral apps dev`, `mistral apps deploy`)
- Workflows
- Connectors
- Chat Completions API with structured output
- Webhooks and triggers
- Workspace secrets

**Other services**

- Slack
- Intercom

View your Connectors in [Studio](https://console.mistral.ai/build/connectors).
