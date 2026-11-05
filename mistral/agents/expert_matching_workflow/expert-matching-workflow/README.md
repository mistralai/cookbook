# Expert matching workflow

A Mistral App that matches experts to project roles. You upload an expert's CV as a PDF in a small web page, and the `match-expert` workflow reads it with Mistral OCR and a document annotation, which returns a structured profile. `mistral-medium-latest` then picks the open role that suits the expert best and explains why. A reviewer approves or declines each match in the page.

The step-by-step guide is in [expert-matching-workflow.md](expert-matching-workflow.md).

The app is built from `mistral apps init --caps fastapi,postgres,workflows,apps` and has three modules (see `apps.json`):

| Path | What it is |
|---|---|
| `apps/api` | FastAPI at `/api`. `routers/api/ui/` serves the web page at `/api`, `routers/api/v1/roles.py` lists and adds open roles, and `routers/api/v1/experts.py` uploads CVs, starts the workflow, stores its results and records reviews. |
| `apps/worker` | The managed workflows worker. `workflows/match_expert.py` holds the workflow, its prompt and its models. It uses the `mistralai_ocr` and `chat_parse_to_model` activities of the Workflows plugin for Mistral. |
| `packages/py/db` | The managed Postgres database: the `role` and `expert` models, their accessors, and the migration that creates them and adds three sample roles. |
| `packages/py/env/src/env/matching.py` | The model settings, `OCR_MODEL` and `MATCHING_MODEL`. |

The scaffold is trimmed to what a Python-only app needs: the JS tooling (bun, nx), the `project.json` files and the coding-agent docs in `.agents/` are removed. Re-running `mistral apps capability add` may bring some of them back.

The API deletes each CV from the Files API once its workflow run finishes, and keeps only the profile read from it.

## Setup

1. Tools: install [uv](https://docs.astral.sh/uv/), Docker (for the local Postgres) and the `mistral` CLI, then run `mistral login`. Run `uv sync --all-packages` in this folder, and put your `MISTRAL_API_KEY` and a `DEPLOYMENT_NAME` of your own for the local worker in `.env`.
2. Deploy the app. Deploys build from the pushed Git commit, so push this folder to a GitHub repository with the Mistral GitHub app installed, then run `mistral apps deploy`. It prints the app's URL. Commit and push the `apps.json` deployment block it adds.
3. Open `https://<app-url>/api`, sign in, add your open roles, and upload a CV.

## Run it locally

`mistral apps dev` starts the API, the worker and Postgres, applies the migration, and serves the app at `http://expert-matching-workflow.apps.localhost:8080`. Open `/api` there in your browser, or call the API with your key in an `X-Api-Key` header:

```bash
curl -H "X-Api-Key: $MISTRAL_API_KEY" -F "file=@cv.pdf" http://expert-matching-workflow.apps.localhost:8080/api/v1/experts
curl -H "X-Api-Key: $MISTRAL_API_KEY" http://expert-matching-workflow.apps.localhost:8080/api/v1/experts
```

Each run appears in `Studio > Workflows > match-expert`.

## Configuration

`OCR_MODEL` (default `mistral-ocr-latest`) and `MATCHING_MODEL` (default `mistral-medium-latest`) are declared in `.env.example`. Set them in `.env` to try other models.
