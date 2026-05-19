# Connectors in Workflows Cookbook

Use MCP connectors inside Mistral Workflows — declare connector dependencies on a worker, call connector tools from activities with specific credentials, and execute workflows with automatic OAuth handling.

> **API status:** The workflow connector integration uses `mistralai-workflows-plugins-mistralai`. These are **beta** features and may change.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Create a Worker with Connectors](#create-a-worker-with-connectors)
- [Execute a Workflow with Connectors](#execute-a-workflow-with-connectors)
- [Execute a Workflow via AI Studio](#execute-a-workflow-via-ai-studio)

---

## Prerequisites

This cookbook covers two separate roles:

| Role | What it does | Where it runs |
|---|---|---|
| **Worker** | Hosts the workflow and calls connector tools | A long-running server process, separate from client scripts |
| **Client** | Triggers workflow execution and handles OAuth redirects | Any script or AI Studio |

### Worker and example client install

The worker needs the Mistral workflows plugin, which bundles the Mistral AI SDK and the workflows runtime:

```bash
uv sync
```

### Required environment variables

**Worker:**

```bash
MISTRAL_CLIENT_SERVER_URL=https://api.mistral.ai
SERVER_URL=https://api.mistral.ai
MISTRAL_API_KEY=<your_api_key>
MISTRAL_CLIENT_API_KEY=<your_api_key>
DEPLOYMENT_NAME=<your_deployment_name>
TASK_QUEUE=<your_task_queue>
```

Deployment name and task queue can just be your name for testing.

**Client:**

```bash
MISTRAL_API_KEY=your-mistral-api-key
SERVER_URL=https://api.mistral.ai

```
You need to use the same deployment name and task queue as the worker to trigger the execution of the right workflow.

### What you need before starting
- Dedicate a workspace and a mistral api tied to this workspace where you will conduct your experiments. Note that people executing your workflow need to be in the same workspace as you.
- At least one registered MCP connector — see the [Connectors Management Cookbook](./01-connectors-management.md) to create one
- For OAuth-authenticated connectors (e.g. Notion, Gmail): no pre-existing credentials are needed — the auth flow is triggered automatically at workflow execution time; for bearer-authenticated connectors (e.g. GitHub PAT), credentials must be stored in the Mistral dashboard before running the workflow

---

## Create a Worker with Connectors

### Concepts

A workflow that uses connectors has three building blocks:

| Building block | What it does |
|---|---|
| `connector(name)` | Declares a named connector slot — a dependency on an MCP connector |
| `@uses_connectors(slot, ...)` | Attaches declared slots to a workflow class so the runtime knows which connectors to authenticate |
| `ToolCallClient` | Activity-level client for calling connector tools, injected via `Depends` |

The `ConnectorAuthInterceptor` is automatically registered by the plugin when `run_worker` starts. It runs an auth preflight before every workflow execution: if valid credentials exist (matching `credentials_name` if specified, else if default credentials exist) the workflow proceeds immediately; if not, it triggers an OAuth flow and waits for the user to authenticate. Note that bearer on the fly authentication is not supported yet.

---

### Step 1 — Declare connector slots

Connector slots are declared at module level. Each slot holds the connector name and auth configuration:

```python
from mistralai.workflows.plugins.mistralai.connectors import connector

github_connector = connector("github_app")
notion_connector = connector("Notion")
```

`connector()` accepts:

| Parameter | Default | Description |
|---|---|---|
| `name` | — | Connector name or ID as registered in the Mistral dashboard |
| `auto_auth` | `True` | Run OAuth preflight before the workflow starts |
| `credentials_name` | `None` | Pin to a specific shared credential name (ex: workspace scoped). Not supported yet, only runtime credentials are supported |

---

### Step 2 — Write an activity that calls a connector tool

Activities receive a `ToolCallClient` via dependency injection using `Depends`:

```python
from typing import Any

import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai.connectors import ToolCallClient, connector

github_connector = connector("github_app")


@workflows.activity(name="create-github-issue")
async def create_github_issue(
    owner: str,
    repo: str,
    title: str,
    body: str,
    github: ToolCallClient = Depends(github_connector),
) -> None:
    await github.call_tool(
        tool_name="issue_write",
        arguments={
            "method": "create",
            "owner": owner,
            "repo": repo,
            "title": title,
            "body": body,
        },
    )
```

`call_tool(tool_name, arguments)` dispatches the call to the MCP connector and returns the raw tool response.

---

### Step 3 — Define the workflow class

Declare connector slots on the workflow class with `@uses_connectors`:

```python
import pydantic
import mistralai.workflows as workflows
from mistralai.workflows.plugins.mistralai.connectors import connector, uses_connectors

github_connector = connector("github_app")


class GitHubIssuePrompt(pydantic.BaseModel):
    owner: str
    repo: str
    title: str
    body: str


@workflows.workflow.define(name="github-issue-creator", on_behalf_of=True)
@uses_connectors(github_connector)
class GitHubIssueCreatorWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, prompt: GitHubIssuePrompt) -> None:
        await create_github_issue(
            prompt.owner,
            prompt.repo,
            prompt.title,
            prompt.body,
        )
```

- `on_behalf_of=True` runs the workflow using the caller's identity — required for per-user connector credentials
- `@uses_connectors(...)` registers the slots so the auth interceptor knows which connectors to authenticate before the workflow body runs. 
    - Note: if you want to use several connectors in the same workflow, you can use `@uses_connectors(github_connector, notion_connector)`
- :warning: The order of `@uses_connectors` and `@workflow.define` matters! you need to apply the `@uses_connectors` after workflow definition

---

### Step 4 — Run the worker

Full self-contained worker file:

```python
from __future__ import annotations

import asyncio

import pydantic
import structlog

import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.core.config.config import config
from mistralai.workflows.core.logging import setup_logging
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

logger = structlog.get_logger(__name__)

github_connector = connector("github_app")


class GitHubIssuePrompt(pydantic.BaseModel):
    owner: str
    repo: str
    title: str
    body: str


@workflows.activity(name="create-github-issue")
async def create_github_issue(
    owner: str,
    repo: str,
    title: str,
    body: str,
    github: ToolCallClient = Depends(github_connector),
) -> None:
    await github.call_tool(
        tool_name="issue_write",
        arguments={
            "method": "create",
            "owner": owner,
            "repo": repo,
            "title": title,
            "body": body,
        },
    )


@workflows.workflow.define(name="github-issue-creator", on_behalf_of=True)
@uses_connectors(github_connector)
class GitHubIssueCreatorWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, prompt: GitHubIssuePrompt) -> None:
        await create_github_issue(
            prompt.owner,
            prompt.repo,
            prompt.title,
            prompt.body,
        )


if __name__ == "__main__":
    setup_logging(
        log_format=config.common.log_format,
        log_level=config.common.log_level,
        app_version=config.common.app_version,
    )
    asyncio.run(workflows.run_worker([GitHubIssueCreatorWorkflow]))
```

Start the worker:

```bash
MISTRAL_CLIENT_SERVER_URL=https://api.mistral.ai SERVER_URL=https://api.mistral.ai MISTRAL_API_KEY=<your_api_key> MISTRAL_CLIENT_API_KEY=<your_api_key> DEPLOYMENT_NAME=<your_deployement_name> TASK_QUEUE=<your_task_queue> 
uv run python -m worker
```

**How it works:**

- `run_worker` connects to the Temporal backend, registers the workflow class, and starts polling for tasks
- The `ConnectorAuthInterceptor` is automatically loaded by the plugin system — no manual setup is needed
- Before each workflow execution, the interceptor checks credentials for every `auto_auth=True` connector slot:
  - Valid credentials found → proceed immediately
  - OAuth2 connector with no credentials → emit an auth URL and wait for the user to authenticate
  - Bearer connector with no stored credential → raise `ConnectorError`

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `ConnectorError: Credential 'x' not found` | Named credential doesn't exist for this connector | Create the credential first, or omit `credentials_name` to use default credentials if available |
| `ConnectorAuthTimeout` | User didn't complete the OAuth flow within 10 minutes | Re-run the workflow and complete the browser auth step promptly |
| `ConnectorError: ... requires bearer authentication` | Bearer connector has no stored credential | Add a credential via the Mistral dashboard before running |
| `ConnectorError: Extension bindings reference unknown connectors` | A runtime binding names a connector not in `@uses_connectors` | Check that `connector_name` in the binding matches a declared slot |

---

## Execute a Workflow with Connectors

Use `execute_with_connector_auth_async` from the `mistralai` SDK to trigger a workflow. This helper polls for task events and — if the worker signals that connector authorization is required — prints the OAuth URL and waits for the user to complete the flow before the workflow resumes.

---
### Example client script

```python
import asyncio
import os

import pydantic
from mistralai.client import Mistral
from mistralai.extra.workflows.connector_auth import (
    ConnectorAuthTaskState,
    execute_with_connector_auth_async,
)
from mistralai.extra.workflows.connector_slot import ConnectorSlot


class GitHubIssuePrompt(pydantic.BaseModel):
    owner: str
    repo: str
    title: str
    body: str


async def on_auth_required(state: ConnectorAuthTaskState) -> None:
    """Default callback: opens the OAuth URL in the browser and waits."""
    if state.auth_url:
        logger.info(
            "Auth required — opening browser (connector=%s, auth_url=%s)",
            state.connector_name,
            state.auth_url,
        )
        webbrowser.open(state.auth_url)
    else:
        logger.info(
            "Auth required — authenticate the connector manually (connector=%s)",
            state.connector_name,
        )
    input("Press Enter after completing the OAuth flow...")



async def main(args) -> None:
    bindings = json.loads(args.bindings) if args.bindings else []
    connector_slots: Sequence[ConnectorSlot] = [
        ConnectorSlot(**binding) for binding in bindings
    ]

    logger.info("Running workflow with connector slots: %s", connector_slots)
    async with Mistral(api_key=args.api_key, server_url=args.server_url) as client:
        response = await execute_with_connector_auth_async(
            client=client,
            workflow_identifier="emma-tests-scenarios",
            input_data=GitHubIssuePrompt(
                owner="my-org",
                repo="my-repo",
                title="Bug: something is broken",
                body="Steps to reproduce...",
            ),
            deployment_name=args.deployment_name,
            connectors=connector_slots,
            task_queue=args.task_queue,
            on_auth_required=on_auth_required,
        )
        print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search meetings")
    parser.add_argument("--api-key", required=True, help="Mistral API key")
    parser.add_argument(
        "--server_url",
        required=False,
        default="https://api.mistral.ai",
        help="Mistral server URL",
    )
    parser.add_argument("--task-queue", required=True, help="Task queue")
    parser.add_argument("--deployment-name", required=True, help="Deployment name")
    parser.add_argument("--workflow_name", required=True, help="workflow to execute")
    parser.add_argument(
        "--bindings",
        default=None,
        help="dict containing connector bindings",
    )
    asyncio.run(main(parser.parse_args()))

```

### Simple execution (pre-registered connectors)

When the connector uses a bearer token that is already stored, execution is straightforward:
Run it:

```bash
uv run python -m 09_workflow_executor_with_connectors --api-key $MISTRAL_API_KEY --task-queue emma --workflow_name emma-tests-scenarios --deployment-name emma
```

---

### Oauth/Bearer execution with runtime connector binding
To specify which credentials to use for each connector when executing the workflow, use the bindings parameter of the script. It will register workflow execution extensions and communicate to the worker which credentials to use for this user.
```
uv run python -m 09_workflow_executor_with_connectors --api-key <your_api_key> --query meeting  --bindings '[{"connector_name": "github_app", "credentials_name": "galilou"}]' --task-queue emma --workflow_name emma-tests-scenarios --deployment-name emma
```

**Binding fields:**

| Field | Description |
|---|---|
| `connector_name` | Must match a connector slot declared with `@uses_connectors` on the workflow |
| `credentials_name` | Select specific stored credentials for this execution |

---

### The OAuth flow

When a connector requires OAuth and the user has no stored credentials, the workflow pauses. If the workflow is executed via cli, the `on_auth_required` callback prints the URL and waits:

```
Connector 'Notion' requires authorization.
Open this URL in your browser to authenticate:
  https://api.notion.com/v1/oauth/authorize?client_id=...

Waiting for authorization... (press Ctrl+C to cancel)
✓ Authorization complete.
```

Once the user authenticates in their browser, the worker detects the new credentials and the workflow resumes automatically.

---

**How it works:**

- `execute_with_connector_auth_async` executes the workflow and polls for task events from the Mistral Workflows API
- When the worker emits a `connector_auth_started` event, `on_auth_required` is called with the OAuth URL
- A long-running heartbeating activity on the worker polls the credentials API until credentials appear
- Once credentials are verified with a `list_tools` call, the workflow proceeds
- The client call returns when the workflow completes, or raises on failure

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `ConnectorAuthTimeout` | OAuth flow not completed within 10 minutes | Re-run and complete the browser auth step |
| `404 Not Found` on workflow execute | Workflow not registered or worker not running | Start the worker first and verify the workflow name matches exactly |
| `ConnectorError: Extension bindings reference unknown connectors` | `bindings` names a connector not declared with `@uses_connectors` | Match `connector_name` to a slot declared in the workflow |

---

## Execute a Workflow via AI Studio

When credentials are not specified, the worker will try to use your default credentials.

> To promote credentials to default, see the [Multiple Authentication Cookbook](./06-multiple-authentication.md).

### OAuth flow in the execution panel

If you have no credentials for a given connector and the connector is OAuth2, the worker will trigger an auth flow on the fly — you will receive an event in the execution panel prompting you to authenticate (the orange key icon indicates an action is required):

<img width="1881" height="586" alt="image" src="https://github.com/user-attachments/assets/385d5e56-1d44-40fb-bb07-64b70c5eafb3" />

Once you complete the flow, the newly created credentials are stored as your default and the workflow resumes automatically.
