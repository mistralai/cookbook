# Connectors in Workflows Cookbook

Use Connectors inside Workflows — declare connector dependencies on a worker, call connector tools from activities with specific credentials, and execute workflows with automatic OAuth handling.

> **API status:** The workflow connector integration uses `mistralai-workflows-plugins-mistralai`. These are **beta** features and may change.

---

## Prerequisites

This cookbook covers two separate roles:

| Role | What it does | Where it runs |
|---|---|---|
| **Worker** | Hosts the workflow and calls connector tools | A long-running server process, separate from client scripts |
| **Client** | Triggers workflow execution and handles OAuth redirects | Any script or AI Studio |


### What you need before starting

To complete this cookbook, you'll need:

- A Mistral workspace to run your workflow.
- At least one registered Connector. See [Build a Database Advisor Agent](./01-build-a-database-advisor-agent.ipynb) for a full connector lifecycle example, or create one directly in [Studio](https://console.mistral.ai).
- For OAuth-authenticated connectors like Notion and Gmail: no pre-existing credentials are needed — the auth flow is triggered automatically at workflow execution time; for bearer-authenticated connectors (e.g. GitHub PAT), credentials must be stored in the Mistral console before running the workflow.

---


## Create a Worker with Connectors

### Concepts

A workflow that uses connectors has three building blocks:

| Building block | What it does |
|---|---|
| `connector(name)` | Declares a named connector slot — a dependency on a Connector |
| `@uses_connectors(slot, ...)` | Attaches declared slots to a workflow class so the runtime knows which connectors to authenticate |
| `ToolCallClient` | Activity-level client for calling connector tools, injected via `Depends` |

The `ConnectorAuthInterceptor` is automatically registered by the plugin when `run_worker` starts. It runs an auth preflight before every workflow execution: if valid credentials exist (matching `credentials_name` if specified, else if default credentials exist) the workflow proceeds immediately; if not, it triggers an OAuth flow and waits for the user to authenticate. Bearer on-the-fly authentication is not currently supported.

### Worker install

The Mistral CLI creates a project with the worker already configured. Install it on macOS or Linux:

```bash
curl -fsSL https://mistral.ai/cli/install.sh | bash
```

Then create the project:

```bash
mistral workflows init connectors-demo
```

Then change into the project:

```bash
cd connectors-demo
```

The project puts the worker in `apps/worker`. Workflow modules go in `apps/worker/src/worker/workflows/`, which the worker scans recursively. `mistral apps dev` reads `MISTRAL_API_KEY` from the project's `.env` and registers the workflow in that key's workspace, not the workspace you logged in to. If `.env` has no key, set `MISTRAL_API_KEY` in it before you start the worker.

---

### Step 1 — Declare connector slots

Create `apps/worker/src/worker/workflows/connectors_example.py`.

Connector slots are declared at module level. Each slot holds the connector name and auth configuration:

```python
from mistralai.workflows.plugins.mistralai.connectors import connector

github_connector = connector("github_app")
notion_connector = connector("notion")
```

`connector()` accepts:

| Parameter | Default | Description |
|---|---|---|
| `name` | — | Connector name or ID as registered in Studio |
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

`call_tool(tool_name, arguments)` dispatches the call to the Connector and returns the raw tool response.

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
    - Note: if you want to use several connectors in the same workflow, you can use `@uses_connectors(github_app, notion)`
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

Start the worker from the project root:

```bash
mistral apps dev
```

**How it works:**

- The command runs the worker locally with hot reload, connects to the Mistral API, and registers your workflow so it can wait for tasks.
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

Use `execute_with_connector_auth_async` from the `mistralai` SDK to trigger a workflow. This helper polls for events. If the worker signals that connector authorization is required, it prints the OAuth URL and waits for the user to complete the flow before the workflow resumes.

---
### Example client script

Install the `mistralai` SDK in the project:

```bash
uv add mistralai
```

Save this script as `client.py`:

```python
import asyncio
import json
import logging
import os
import webbrowser
from argparse import ArgumentParser
from collections.abc import Sequence

import pydantic
from mistralai import Mistral
from mistralai.extra.workflows.connector_auth import (
    ConnectorAuthTaskState,
    execute_with_connector_auth_async,
)
from mistralai.extra.workflows.connector_slot import ConnectorSlot

logger = logging.getLogger(__name__)


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
    async with Mistral(api_key=os.environ["MISTRAL_API_KEY"], server_url=args.server_url) as client:
        response = await execute_with_connector_auth_async(
            client=client,
            workflow_identifier=args.workflow_name,
            input_data=GitHubIssuePrompt(
                owner="my-org",
                repo="my-repo",
                title="Bug: something is broken",
                body="Steps to reproduce...",
            ),
            deployment_name=args.deployment_name,
            connectors=connector_slots,
            on_auth_required=on_auth_required,
        )
        print(response)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(description="Create a GitHub issue with a workflow")
    parser.add_argument(
        "--server_url",
        required=False,
        default="https://api.mistral.ai",
        help="Mistral server URL",
    )
    parser.add_argument("--deployment-name", required=True, help="Deployment name")
    parser.add_argument("--workflow-name", required=True, help="workflow to execute")
    parser.add_argument(
        "--bindings",
        default=None,
        help="dict containing connector bindings",
    )
    asyncio.run(main(parser.parse_args()))
```

The script reads the API key from `MISTRAL_API_KEY`. `mistral workflows init` saves the key to the project's `.env` when you enter it at the prompt.

`mistral apps dev` registers the worker on the deployment named by `DEPLOYMENT_NAME` in the same `.env`, which is `deployment-<project-name>`. Pass that name with `--deployment-name`.

### Simple execution (pre-registered connectors)

When the connector already has stored credentials, run the client without `--bindings`:

```bash
uv run python client.py --workflow-name github-issue-creator --deployment-name deployment-connectors-demo
```

---

### Oauth/Bearer execution with runtime connector binding
To choose which credentials the worker uses for each connector, pass a `--bindings` JSON list to the client script:

```bash
uv run python client.py --workflow-name github-issue-creator --deployment-name deployment-connectors-demo --bindings '[{"connector_name": "github_app", "credentials_name": "your-credentials-name"}]'
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

## Execute a workflow via Studio

When credentials are not specified, the worker will use your default credentials.

> To promote credentials to default, see the [Multiple Authentication Cookbook](./06-multiple-authentication.md).

Open [Workflows](https://console.mistral.ai/build/workflows) in Studio, select `github-issue-creator`, and start it with this input:

```json
{"owner": "your-username", "repo": "your-repo", "title": "Hello World", "body": "Hello World"}
```

### OAuth flow in the execution panel

If you have no credentials for a given connector and the connector is OAuth2, the worker triggers an auth flow — you will receive an event in the execution panel prompting you to authenticate (an orange key icon indicates an action is required).

Once you complete the flow, the newly created credentials are stored as your default and the workflow resumes automatically.

---

## Summary

This cookbook covered how to use Connectors inside Mistral Workflows — declaring connector slots, calling Connector tools from activities, and executing workflows with automatic OAuth handling from a client script or from Studio.

**What this cookbook covers:**
- Declaring connector slots on a Workflows worker
- Writing activities that call Connector tools via `ToolCallClient`
- Defining a workflow class with `@uses_connectors`
- Running a worker and executing a workflow
- Client-side execution with automatic OAuth flow handling
- Triggering a workflow from Studio

**Mistral features used:**
- Workflows (beta)
- Connectors (beta)

**Other services:**
- GitHub MCP — bearer-authenticated Connector
- Notion — OAuth2-authenticated Connector

[View the documentation]() <!-- TODO: add link to relevant documentation -->
