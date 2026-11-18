# Slack triage workflow

A Mistral App that triages every new message in a Slack channel. A Studio trigger starts the `slack-triage` workflow, which classifies the message with `mistral-medium-latest` as a bug or a feature request. Bugs become Intercom tickets, and feature requests go into the app's own tracker. The workflow then replies in the thread with a link.

The app is built from `mistral apps init --caps fastapi,postgres,workflows,apps` and has three modules (see `apps.json`):

| Path | What it is |
|---|---|
| `apps/api` | FastAPI at `/api`. `routers/api/v1/feature_requests.py` is the tracker: create, list and get feature requests. |
| `apps/worker` | The managed workflows worker. `workflows/slack_triage.py` holds the workflow and its prompt, `activities.py` the Slack, Intercom and tracker calls. |
| `packages/py/db` | The managed Postgres database: the `feature_request` model, its accessors and its migration. |
| `packages/py/env/src/env/triage.py` | The workflow's settings, read from `.env` locally and from workspace secrets when deployed. |
| `packages/py/cli` | `python -m cli`, for the setup steps Studio's UI can't do yet. |
| `triggers/slack.json` | The trigger filter. |

The scaffold is trimmed to what a Python-only app needs: the JS tooling (bun, nx), the `project.json` files and the coding-agent docs in `.agents/` are removed. Re-running `mistral apps capability add` may bring some of them back.

The workflow reaches Intercom and the tracker through two Studio HTTP connectors. Studio stores their credentials, so the worker never holds an Intercom token. Intercom has a 14-day free trial, which is enough to run the whole demo.

## Setup

1. Tools: install [uv](https://docs.astral.sh/uv/), Docker (for the local Postgres) and the `mistral` CLI, then run `mistral login`. Run `uv sync --all-packages` in this folder and `cp .env.example .env`, then add your `MISTRAL_API_KEY` and a `DEPLOYMENT_NAME` of your own for the local worker.
2. Slack app: create one with the bot scopes `channels:history`, `channels:read`, `chat:write` and `users:read`, install it, and put the bot token (`xoxb-...`) in `.env` as `SLACK_BOT_TOKEN`. Keep the signing secret from `Basic Information > App Credentials` for step 8.
3. Intercom: start a trial, then create an access token in `Developer Hub > your app > Authentication`. The workspace needs a ticket type in the Customer category (`Settings > Inbox > Tickets`). Note your region from the address bar: `app.intercom.com` is the US, `app.eu.intercom.com` the EU and `app.au.intercom.com` Australia.
4. Intercom connector: create it with a credential shared with the workspace. Triggered runs have no signed-in user, so they can't see a credential that belongs to yours, and a connector made in Studio only takes a user credential. Use your region's API address: `https://api.intercom.io` for the US, `https://api.eu.intercom.io` for the EU or `https://api.au.intercom.io` for Australia. The command prompts for the token:
   ```bash
   uv run --all-packages python -m cli connector intercom https://api.intercom.io
   ```
   Put the connector's name in `.env` as `INTERCOM_CONNECTOR_NAME`. Outside the US, also set `INTERCOM_APP_URL` to your region's app address, such as `https://app.eu.intercom.com`, so ticket links in replies open.

   Then fill in the two Intercom IDs. Run `uv run --all-packages python -m cli intercom-ids`, which also checks the connector works, and copy the `INTERCOM_WORKSPACE_ID` line and the ticket type you want as `INTERCOM_TICKET_TYPE_ID`. Both are also in Intercom's address bar: the workspace ID follows `/apps/`, and opening the ticket type in `Settings > Inbox > Tickets` shows its ID at the end.
5. Deploy the app. Deploys build from the pushed Git commit, so push this folder to a GitHub repository with the Mistral GitHub app installed, then run `mistral apps deploy`. It prints the app's URL and creates the worker deployment `<app>-worker`. Commit and push the `apps.json` deployment block it adds.
6. Tracker connector: the app's gateway lets a request through when it carries a Mistral API key from the app's workspace in an `X-Api-Key` header. Create the connector with the app's URL as its base, without `/api`, since the routes already start with it. The command prompts for the key:
   ```bash
   uv run --all-packages python -m cli connector tracker https://<app-url> --header X-Api-Key
   ```
   Put its name in `.env` as `TRACKER_CONNECTOR_NAME`. Check the tracker answers with `curl -H "X-Api-Key: $MISTRAL_API_KEY" https://<app-url>/api/v1/feature_requests`, which returns `[]` before the first request is filed.
7. Secrets: a deployed worker has no `.env`. In `Studio > Secrets`, create a workspace secret for each of `MISTRAL_API_KEY`, `SLACK_BOT_TOKEN`, `INTERCOM_CONNECTOR_NAME`, `INTERCOM_TICKET_TYPE_ID`, `INTERCOM_WORKSPACE_ID` and `TRACKER_CONNECTOR_NAME`, plus `INTERCOM_APP_URL` if you set it, with the same values as `.env`. Bind them to the worker. The command keeps the database bindings the deploy made:
   ```bash
   uv run --all-packages python -m cli bind-secrets <app>-worker
   ```
8. Trigger: in Studio, create a Slack webhook with the app's signing secret and copy its URL. Create a trigger on that webhook with the action `Start workflow` and the workflow `slack-triage`, and paste the filter from `triggers/slack.json`. The filter drops bot messages (including the workflow's own replies), edits and thread replies. In the Slack app, go to `Event Subscriptions > Enable Events`, paste the webhook URL, subscribe to the `message.channels` bot event and save. Treat the URL as a credential.
9. Invite the bot to the channel you want to triage. Slack only sends events from channels the bot is in.

Post a bug report or a feature request in the channel. The reply lands in the thread within a few seconds.

## Run it locally

`mistral apps dev` starts the API, the worker and Postgres, and applies the migration. To run the workflow on a message already in the channel, pass its channel ID and timestamp, both in the message link (`.../archives/<channel id>/p<ts without the dot>`):

```bash
uv run --all-packages python -m cli triage-message C0123456789 1790327758.660299
```

It runs on your local worker. Pass `--deployment <app>-worker` to use the deployed one. The execution appears in `Studio > Workflows > slack-triage`. Don't run a local worker and a Slack trigger against the same channel, or both will answer.

## Configuration

Every variable is in `.env.example`. When deployed, each comes from the workspace secret of the same name.

| Variable | Purpose |
|---|---|
| `MISTRAL_API_KEY` | Classification and connector calls |
| `DEPLOYMENT_NAME` | The deployment your local worker registers on. Deploy sets it on the managed worker |
| `SLACK_BOT_TOKEN` | Reading names and posting replies |
| `TRIAGE_MODEL` | Model used for classification. Defaults to `mistral-medium-latest` |
| `INTERCOM_CONNECTOR_NAME` | Name of the Intercom HTTP connector |
| `INTERCOM_TICKET_TYPE_ID` | Ticket type for bug reports |
| `INTERCOM_WORKSPACE_ID` | Used to build the ticket link in the reply |
| `TRACKER_CONNECTOR_NAME` | Name of the tracker HTTP connector |
| `INTERCOM_APP_URL` | Optional. Your region's Intercom app address for ticket links. Defaults to `https://app.intercom.com` |

## Notes

- **Trigger input.** A trigger starts the workflow with the incoming HTTP request, so the Slack event is at `body.event`. A manual run takes `{"body": {"event": {"channel": ..., "user": ..., "ts": ..., "text": ...}}}`.
- **No duplicate filings.** The activities that create tickets and feature requests aren't retried, because a retry after a timeout could file the same item twice.
- **Intercom contacts.** Intercom tickets need a contact. The workflow creates one per Slack user (`external_id: slack-<user id>`) the first time that user reports a bug.
- **Tracker auth.** Every `/api/v1` route rejects requests that didn't come through the Apps gateway. The gateway admits a browser session or a workspace API key in `X-Api-Key`, and the tracker connector stores the key.
- **Logging.** Never log Slack text, payloads, tokens or webhook URLs.

## Tests

Run the tests with `uv run --all-packages --with pytest --with pytest-asyncio pytest -o asyncio_mode=auto`. They need no `.env`, accounts or network, except `test_migrations_pg.py`, which needs a local Postgres and skips without one.
