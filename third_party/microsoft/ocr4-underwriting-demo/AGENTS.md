# AGENTS.md: working guide

This file is the shared brief for anyone working in this repo, human or agentic
development tool. Read it before making changes. It explains what the project is, how it
is wired, the conventions that are not obvious from the code, and the few things you must
not do. Start with the README for the deploy-and-run story, then come here for how the
pieces fit and where to change them.

## What this project is

A document processing pipeline built on Mistral models on Azure AI Foundry, with two
runtime tracks that share one `src/` codebase:

1. **Event-driven pipeline.** A file lands in blob storage, Event Grid bridges the
   `BlobCreated` event into a Storage Queue, an Azure Function
   drains the queue, and the document flows through OCR then reasoning to a JSON result.
2. **Mortgage underwriting demo with a human in the loop.** A FastAPI backend (`api.py`) with
   a separate Gradio UI (`app.py`, views `/apply`, `/review`, `/chat`) turns an application
   package into a decision. Clear cases auto-decide; borderline cases route to a reviewer.

Two models, split by role. `mistral-ocr-4` is the specialist: it turns pixels into
markdown and, in the same call, returns a `document_annotation` (type, language, summary).
`mistral-medium-3-5` is the light generalist: it extracts fields and entities, and it
underwrites against rules that live in code, not in the prompt.

## Golden rules (do not skip)

- **Python is `uv` only.** No `pip`, no `venv`, no `poetry`. Install with `uv sync`, run
  with `uv run <script>`. Add dependencies with `uv add <pkg>`, which updates
  `pyproject.toml` and `uv.lock` together.
- **Prose conventions.** No em dashes anywhere, define acronyms on first use. This
  applies to comments, commit messages, and PR descriptions.
- **Diagrams are rendered images, never ASCII art.**
- **Commit under your own name.** Keep commit messages focused on the behavior change.
- **Secrets stay out of git.** `.env`, `cases.db*`, and `.venv/` are ignored; keep it that
  way. Only the `.example` files carry placeholders. The demo sample documents under
  `inputs/` are tracked on purpose so the team can run the app and demos; generated OCR
  output (`inputs/*.result.json`, `*.md.out`) stays ignored. Do not add real customer data
  to `inputs/`; the samples there are synthetic.
- **Deploy and tear down through the scripts.** `scripts/up.sh` provisions and runs the demo
  (the azd path uses the `rg-mistral-demo` resource group); `scripts/down.sh` deletes that stack
  and purges the soft-deleted Foundry account. Do not delete resources by hand, and do not run
  `down.sh` on a stack someone else is using without asking.

## Setup

```bash
uv sync                       # create the environment from uv.lock
cp .env.example .env          # fill AZURE_AI_KEY + the storage connection string
az login                      # only needed for infra deploy or blob/queue access
```

`src/config.py` loads settings from the environment (via `.env`). It **requires**
`AZURE_AI_KEY`, `AZURE_OCR_ENDPOINT`, `AZURE_OCR_DEPLOYMENT`, `AZURE_INFERENCE_ENDPOINT`,
and `AZURE_CHAT_DEPLOYMENT`, and raises a clear error naming any that is missing. Storage
variables are optional for the CLI paths and needed only for the blob and queue flows.

## How to run each entry point

The fastest path is `scripts/up.sh`: it provisions if needed, writes `.env`, and starts both
processes (see the README). The commands below run the pieces directly for local development.

The `src/` modules import each other by **bare module name** (`from config import ...`,
`import documents as D`). So run from inside `src/`, or make sure `src/` is on
`sys.path`. The Function does the latter with a `sys.path.insert`. `uv run src/x.py`
works because the script's own directory is added to the path.

```bash
# OCR only: markdown out
uv run src/ocr.py inputs/loan_application_1003.png

# Full pipeline: OCR -> extractor -> <name>.result.json
uv run src/pipeline.py inputs/loan_application_1003.png

# Traced upload into the inbox (drives the event path once infra is deployed)
uv run src/upload.py inputs/loan_application_1003.png

# Underwriting demo: two processes, backend API + UI (start the backend first).
cd src && uv run uvicorn api:api --port 8001                              # backend service
cd src && BACKEND_URL=http://127.0.0.1:8001 uv run uvicorn app:ui --port 8000  # UI (2nd terminal)
#   Home:            http://localhost:8000/
#   Customer:        http://localhost:8000/apply
#   Reviewer:        http://localhost:8000/review
#   Document chat:   http://localhost:8000/chat
#   Backend API docs: http://localhost:8001/docs

# Azure Function locally (needs Azure Functions Core Tools)
cd function_app && cp local.settings.json.example local.settings.json && func start
```

## Architecture, module by module

### The shared OCR + reasoning core (`src/`)

| Module | Responsibility |
|---|---|
| `config.py` | Frozen `Settings` dataclass loaded from env. Builds the chat `base_url` as `<inference>/openai/v1`. |
| `ocr_client.py` | Mistral OCR 4 HTTP client. `ocr_with_annotation(data, name)` returns markdown, the annotation, page count, usage, and layout blocks. |
| `agents.py` | The `DocumentAnalysis` Pydantic schema, `build_extractor` (the Medium 3.5 agent), and `parse_json_object` (tolerant JSON parse for model output). |
| `workflow.py` | The agent-framework graph: `OcrExecutor` -> `ExtractorExecutor` -> `DocumentAnalysis`. OCR is a first-class node, so the graph owns the pipeline. |
| `pipeline.py` | End-to-end CLI over the workflow, with correlation metadata for tracing. |
| `observability.py` | OpenTelemetry setup, span helpers, and metric recording, exporting to Azure Monitor. |
| `upload.py` | Traced uploader; stamps `doc_id` and a W3C `traceparent` into blob metadata. |
| `ocr.py` | OCR-only CLI. |
| `entities.py` | Entity-to-region reconciliation: map an extracted field to the OCR block it came from, for the highlight overlay. |

### The underwriting demo (`src/`)

| Module | Responsibility |
|---|---|
| `api.py` | Backend service: the JSON + streaming (SSE) API over the underwriting core. The only process that touches OCR, the agents, and the case store. |
| `app.py` | UI process: three Gradio views (`/apply`, `/review`, `/chat`), wired to `api.py` over HTTP. The largest file; UI wiring lives here, not business logic. |
| `service_client.py` | The HTTP client the UI calls, shaped like the old in-process service so the UI code reads the same. |
| `docchat.py` | The document-chat logic behind `/chat` and the `/doc/*` API: OCR a standalone file and answer questions over it. |
| `underwriting_service.py` | Orchestration used by both the API and the UIs. Ties OCR, extraction, intake, underwriting, and review together over the store, with stage guards and an append-only audit. |
| `underwriting_schema.py` | Pydantic models and the `Stage` / `Outcome` enums, plus `can_transition`, the state-machine guard. |
| `documents.py` | The standard mortgage document checklist (`REQUIRED_DOCS`) and `classify()`, which maps an OCR document type to a checklist slot. |
| `intake.py` | The missing-value conversation: extract facts, decide the next prompt, apply a customer message. |
| `underwriter.py` | `assess`, `route`, and `auto_outcome`: score a case, then decide auto-approve vs auto-decline vs send to a reviewer. |
| `mortgage_rules.py` | Policy in one place: LTV limits, the auto-approve ceiling, minimum confidence. Change lending policy here, not in prompts. |
| `case_store.py` | The `CaseStore` Protocol and a SQLite implementation with optimistic concurrency. The seam that lets the demo run on SQLite and production run on Cosmos DB. |
| `doc_render.py` | Rasterize an uploaded PDF or image page to a preview PNG for the reviewer and chat views. |
| `ui_render.py` | UI-side overlay of OCR blocks onto the rendered preview. No network; it works on already-fetched data. |

### The event path (`function_app/`)

`function_app/function_app.py` is a Python v2 Function with one `queue_trigger` on the
`ingest` queue. It decodes the Event Grid message (base64), resolves the blob name, pulls
the trace context stamped at upload, runs the pipeline, and writes `<name>.result.json`
to the `results` container. It reuses `src/` by adding it to `sys.path`; the deploy script
vendors `src/` into the function folder so the code ships with it.

### Infrastructure (`infra/`, Bicep)

- `main.bicep`: Foundry account, both model deployments, Storage (containers + queue),
  Event Grid subscription, Log Analytics, App Insights, diagnostic settings, dead-letter.
- `functionapp.bicep`: the Flex Consumption (`FC1`) Function App and its app settings.
- `alerts.bicep`: scheduled-query alerts for exceptions and 429 throttling.
- `main.named.bicepparam`: parameter values for the manual named stack (account name, region,
  model SKUs). Pass it explicitly: `-p infra/main.named.bicepparam`. It is deliberately NOT
  named `main.bicepparam` so `azd` does not auto-pick it; the azd path uses
  `main.parameters.json` (unique auto-generated names).

## The case lifecycle (underwriting)

A case moves through `Stage` values: `received` -> `reading` (OCR) -> `extracting` ->
`collecting` (missing-value chat) -> `ready` -> `underwriting` -> then either
`auto_decided`, or `pending_review` -> `in_review` -> `finalized`. `can_transition`
guards every move, so add new transitions there rather than setting `stage` directly.

`Outcome` is `approve`, `decline`, or `refer`. `underwriter.route` sends a case to
auto-decision only when the metrics clear the thresholds in `mortgage_rules.py` and
confidence is high enough; otherwise it goes to a reviewer.

The store uses etag-based optimistic concurrency: `put` rejects a write whose expected
etag no longer matches, so the customer chat and a reviewer cannot clobber each other.
`get_store()` is a process singleton; the SQLite path is `CASE_DB_PATH` (default
`cases.db`).

## Observability

One document is one connected trace. `upload.py` stamps `doc_id` and a `traceparent`
into blob metadata; the Function reads them back and continues the same trace through OCR,
extraction, and the written result. External uploads that skip `upload.py` still get a
`since_upload_ms` metric from the Event Grid event time. The alert rules are defined in
`infra/alerts.bicep`.

## Known constraints and gotchas

- **Tool calling is unsupported on this Foundry Mistral deployment (a serving-layer limit,
  not our code).** Root cause, confirmed 2026-08-25: the model server runs with
  `--grammar-backend none`, so the structured or guided decoding that function calling needs
  is off. Evidence: a raw `/openai/v1/chat/completions` call with a correct `tools` payload
  returns `tool_calls: null` under `tool_choice="auto"`, and `tool_choice="required"` returns
  `400 "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported
  when the server is launched with --grammar-backend none"`. Through the Agent Framework it
  shows up as a 500 on the Responses path or the tool being silently ignored on the chat path,
  and the Foundry portal shows an "Unsupported tools" dialog that strips tools (for example Web
  search) from Mistral agents. So the agents run tool-free, invoked server-side through Agent Service (Responses API with an `agent_reference` carrying name + id, which is what surfaces the runs in the portal traces) and orchestrated by the workflow
  engine, and human-in-the-loop uses `ctx.request_info()`, not an approval tool. Do not switch
  to `Agent.as_tool()`. A tool-calling agent needs an Azure OpenAI model, which runs on a
  different serving stack; OCR 4 can stay the Mistral perception model.
- **Use `OpenAIChatCompletionClient`** (the `/chat/completions` route), not
  `OpenAIChatClient` (the Responses API, which Mistral rejects).
- **OCR quota:** `GlobalStandard` OCR quota was exhausted on this subscription, so OCR
  runs on `DataZoneStandard`. OCR 4 is Preview and region-limited.
- **OCR annotation limits:** classification and summary via `document_annotation_format`
  cap at roughly 8 pages and can time out; content safety applies to annotations.
- **Functions deploy via pip**, so `function_app/requirements.txt` mirrors the root
  `pyproject.toml`. Update both when you change dependencies used by the Function.
- **Queue messages are base64-encoded Event Grid events**; the Function decodes them
  defensively and handles batched events.

## Making changes: where to look

- New extracted field or lending rule: `underwriting_schema.py` (the model) and
  `mortgage_rules.py` (the policy). Auto-decision logic lives in `underwriter.py`.
- New document type in the checklist: `documents.py` (`REQUIRED_DOCS` and `classify`).
- Change what the extractor pulls: the instructions in `agents.py`.
- UI behavior for either view: `app.py`. Orchestration behind the UI:
  `underwriting_service.py`.
- Pipeline shape (add or reorder a node): `workflow.py`.
- Swap the store for Cosmos DB: implement the `CaseStore` Protocol in `case_store.py`;
  nothing else should need to change.

## Verifying your work

There is no test suite yet. Sanity-check changes by exercising the real path:

- Pipeline changes: `uv run src/pipeline.py inputs/loan_application_1003.png` and inspect
  the `.result.json`.
- Underwriting or UI changes: run the app and drive `/apply` (upload the sample scans,
  including `loan_application_thin.png`, which is borderline and routes to a reviewer),
  then `/review` to approve, decline, or refer.
- If you add tests, use `pytest` under `uv run` and put them in a top-level `tests/`.

Keep commits focused, describe the behavior change (not the file list), and follow the
prose rules above in the message.
