# Summarize a week of Linear activity with a workflow

This workflow reads the issues updated in the past week in a Linear team or project and asks a Mistral model to write a weekly summary.

*Anton Kluge, [@AntonKluge](https://github.com/AntonKluge), Mistral AI*

The workflow takes a team and a project as input, either by name or by ID. It resolves both through the Linear Connector. If a team or project can't be resolved, the workflow pauses and asks you to pick one in Studio. It then fetches the issues updated in the last 7 days and passes them to `mistral-medium-latest` for summarization.

The connector call runs with `on_behalf_of=True`, so Linear resolves the credentials stored for the person who starts the execution. This needs a hardened deployment, and the steps are in [Hardened deployments](https://docs.mistral.ai/studio/workflows/managing-workflows-in-production/hardened_deployments).

## Prerequisites

To complete this cookbook, you will need:
- macOS or Linux
- git
- The Mistral CLI, installed in the next step. `mistral apps dev` also needs Bun, which the installer offers to install.
- A Mistral account
- A Linear account, with a Linear Connector created in [Studio](https://console.mistral.ai/build/connectors) and named `linear`
- An organization with access to hardened deployments, which the on-behalf-of mode requires

## Environment setup

Run these steps once from the directory where you want the project.

### Install

Install the Mistral CLI with the installer script. The installer offers to install Bun, which `mistral apps dev` needs to run the worker.

```bash
curl -fsSL https://mistral.ai/cli/install.sh | bash
```

Create a new Workflows project with `mistral workflows init`. On a terminal, it asks for your API key, checks it, and saves it in the project's `.env`.

```bash
mistral workflows init linear-summarization-cookbook
```

From the cookbook repository root, copy the `linear_summarization` folder into the worker's workflows directory. The worker discovers every workflow package under `apps/worker/src/worker/workflows/`, including subpackages.

```bash
cp -R mistral/workflows/linear-summarization/linear_summarization \
  linear-summarization-cookbook/apps/worker/src/worker/workflows/
```

### Connector setup

The workflow reads Linear through the Connector, so no Linear token goes in the environment. Create the connector once in Studio:

1. Open [Studio › Connectors](https://console.mistral.ai/build/connectors) and create a Linear connector named `linear`. The name must match `connector("linear")` in `linear_activities.py`.
2. Open the connector's Credentials tab and complete the Linear auth flow.

### Deploy

The on-behalf-of call needs a managed deployment, so deploy the worker before you run it in Studio. The deploy builds the commit you pushed, so commit the copied `linear_summarization` package first.

Run `mistral login`, commit the project, and push it to a GitHub `origin` with the Mistral GitHub app installed:

```bash
git add .
git commit -m "Add the linear_summarization workflow"
gh repo create <your-org>/linear-summarization-cookbook --private --source . --remote origin --push
```

Then deploy from the project root:

```bash
mistral apps deploy
```

The first deploy creates the workflows deployment hardened, so its workflows can act on behalf of the user who runs them. `mistral apps deploy` needs a per-org feature flag. If the command is missing, ask for it in `#project-mistral-apps-cli`. Off a terminal, the deploy fails with `No deploy target.` unless you pass `--organization`.

## Run the worker

Start the worker locally from the project root. It reloads when you edit a workflow file.

```bash
cd linear-summarization-cookbook
mistral apps dev
```

## Start the workflow in Studio

Open [Workflows in Studio](https://console.mistral.ai/build/workflows?workflowSearch=linear-weekly-summary), select `linear-weekly-summary`, and click Start workflow. Paste this input to summarize every project in the Engineering team:

```json
{
  "team": "Engineering",
  "project": "*"
}
```

`team` and `project` accept a name or an ID. Use `"*"` for `project` to cover all projects in the team. To summarize a single project, pass its name or ID instead:

```json
{
  "team": "Engineering",
  "project": "Roadmap"
}
```

Start the workflow with `{}` to pick the team and project from prompts in Studio.

## Inspect the execution

Open [the executions page](https://console.mistral.ai/build/workflows/executions) and select the run. `fetch_recent_issues` runs on the resolved team and project, then `generate_weekly_summary` writes the summary. When the team or project name doesn't resolve, the execution pauses at a `wait_for_input` step. Answer the prompt in Studio and the execution continues with the chosen values.

The final output is the summary text. Check that it covers the issues updated in the last 7 days.

## Summary

This cookbook ran a workflow that reads Linear issues through a connector and summarizes a week of activity with a Mistral model.

**What you built:**
- A workflow that resolves a Linear team and project by name or ID
- A human-in-the-loop fallback that asks you to pick a team or project when resolution fails
- An activity that summarizes the issues updated in the last 7 days

**Mistral features used:**
- Workflows, with `InteractiveWorkflow` and `wait_for_input` for the human-in-the-loop step
- Linear Connector with on-behalf-of authentication
- Chat completions through the `mistralai_chat_complete` workflow activity with `mistral-medium-latest`

**Other services:**
- Linear

To learn how connectors and on-behalf-of authentication work in workflows, see the [connectors documentation](https://docs.mistral.ai/studio/workflows/building-workflows/connectors).
