# Investigation agent

A Slack mention triggers `investigation-agent-router`, which checks the channel is public and routes the message to the thread's `investigation-agent-thread` workflow (starting it if needed). That workflow batches tags that arrive mid-turn into the next turn, runs every turn in one Agent Session (`mistral-large-4`), replies in the thread and exits after 1 idle day.

`agent/` is the Agents SDK runtime: read-only Sentry and GitHub tools, a fail-closed tool allowlist hook (`tool_allowlist.py`) and 2 skills. `workflow/` holds `investigation.py` (both workflows), every side effect in `activities/` (`slack.py`, `agent.py`, `routing.py`), helpers (`models.py`, `formatting.py`) and `worker.py`. `triggers/slack.json` holds the webhook trigger filter, `sample-app/` a small buggy checkout script that sends a real error to Sentry for the agent to investigate, and `docs/` the architecture graphs.

Both connectors work on free plans, so you can run the whole demo with your own accounts.

## Setup

1. Sentry and GitHub: sign up (free tiers are enough) and create a Sentry connector and a GitHub connector in your Studio workspace. Studio gives every connector a unique name (such as `sentry_d183`); put the exact names in `.env` as `SENTRY_CONNECTOR_NAME` and `GITHUB_CONNECTOR_NAME`.
2. Slack app: create one in your workspace with bot scopes `app_mentions:read`, `channels:history`, `channels:read`, `channels:join`, `chat:write` and `reactions:write`, install it and copy the bot token.
3. Workspace secrets: `INVESTIGATION_AGENT_MISTRAL_API_KEY` (a Mistral API key) and `SLACK_BOT_TOKEN_INVESTIGATION_AGENT` (the bot token) for the workflow, plus `sentry_connector_name` and `github_connector_name` holding the same connector names as `.env`. The deployed agent has no `.env`; it reads the names from these secrets.
4. Deploy the agent first. Deploys build from the pushed GitHub commit, not your local files, so fork this repo if you change anything and push before each deploy. `uv` installs the `agents` CLI from `agent/pyproject.toml` on first run, and `--env-file` loads `MISTRAL_API_KEY` and the connector names, which the CLI only reads from the environment:
   ```
   cd agent && uv run --env-file ../.env agents deploy . \
     --secret-binding SENTRY_CONNECTOR_NAME=secret:workspace:sentry_connector_name \
     --secret-binding GITHUB_CONNECTOR_NAME=secret:workspace:github_connector_name
   ```
   Pass both bindings on every deploy: a deploy without them unbinds the secrets and the agent fails to start.
5. Deploy the workflow from `workflow/Dockerfile` (build context is the repo root) with the `INVESTIGATION_AGENT_MISTRAL_API_KEY` and `SLACK_BOT_TOKEN_INVESTIGATION_AGENT` secrets.
6. Create a Studio webhook trigger that starts `investigation-agent-router` with the filter in `triggers/slack.json`, and set its URL as the Slack app's Event Subscriptions Request URL, subscribed to `app_mention`. Treat the URL as a credential.
7. Invite the bot to a public channel and tag it.

## Try it with a real error

`sample-app/checkout.py` totals a cart whose prices are parsed by `sample-app/pricing.py`, which assumes whole-dollar prices, so the `"12.50"` poster raises a `ValueError`. Running it once sends that error to Sentry, tagged with the current commit.

You can ask the agent locally from your terminal, which only needs the Sentry and GitHub connectors (Setup step 1) and a filled-in `.env` (`MISTRAL_API_KEY`, the connector names and `INVESTIGATION_AGENT_NAME`), or through Slack once the full setup is deployed.

1. In Sentry, create a project (**Projects → Create Project**), pick **Python** and select only **Error Monitoring**; the agent never reads logs, traces, profiles or metrics. Copy the DSN from the setup snippet (or later from **Settings → Projects → your project → Client Keys (DSN)**) and add it to `.env` as `SENTRY_DSN`. Keep it out of git.
2. Make sure your GitHub connector can read the repo that contains `sample-app/`, and that your commit is pushed.
3. Send the error: `make send-error`. It should appear as a new `ValueError` issue in your Sentry project.
4. Ask the agent about it, either way:
   - Locally: from the repo root, run `make serve` and leave it running. In a second terminal, run `make ask`. On its own it asks the question below; run `make ask Q="your question"` to ask something else. It prints each tool call as it investigates, then the answer.
   - In Slack: tag the bot with "What's causing the latest ValueError in Sentry, and which code is responsible?"

   It should find the issue, follow the stack trace to `parse_amount` in `sample-app/pricing.py` on GitHub, and explain the whole-dollar assumption.

## Development

`cp .env.example .env`, fill it in, then `make check` (ruff, ty and `agents info`). `make build` builds both images.

To run the agent on your machine, set `INVESTIGATION_AGENT_NAME` in `.env` (for example `investigation-agent-dev`) and run `make serve`. It serves the agent under that name, so it never collides with the deployed `investigation-agent-runtime`, and stays running. While it is up, ask it a question from a second terminal with `make ask Q="your question"` (without `Q` it asks about the sample-app error). A local workflow worker (`cd workflow && uv run --env-file ../.env python worker.py`) reads the same variable and opens sessions on the local agent; the deployed workflow leaves it unset and uses the deployed agent.

Ops rules: deploy the agent before the workflow; never run local and managed workers concurrently; drain executions before changing the activity sequence; never log Slack text, payloads or secrets.

