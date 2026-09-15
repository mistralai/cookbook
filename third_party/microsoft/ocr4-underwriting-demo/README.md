# Document Processing Pipeline: Mistral OCR 4 and Medium 3.5 on Azure Foundry

An event-driven document pipeline on **Microsoft Agent Framework**. The pipeline is
code-orchestrated: a fixed workflow graph owns the control flow, and the model steps are
hosted, governed, portal-traced Foundry agents (they answer; the code decides).

- **Ingestion:** a blob upload raises a `BlobCreated` event; **Event Grid** bridges it into
  the **`ingest` Storage Queue**, which buffers work and lets the Function scale independently
  of upload spikes.
- **OCR + classification (the specialist):** `mistral-ocr-4` extracts markdown **and** returns a
  `document_annotation` (document type, language, summary) in a single call, with no
  separate classifier LLM needed.
- **Reasoning (the light generalist):** one `mistral-medium-3-5` extractor pulls deep structured
  fields + entities. By default it runs as a hosted Foundry Agent Service agent
  (`mortgage-extractor-agent`, server-side, with per-agent portal traces), falling back to a
  direct `/chat/completions` call. Agents are prompt-only: no tool calling on this Mistral
  deployment.
- **Auth:** account key for the OCR + inference model routes; Entra managed identity (Azure AI
  User) for the hosted-agent data-plane calls.

This repo holds the runnable code. Session decks, diagrams, screenshots, and demo
recordings are kept separately and are not part of this repository.

### Which Azure and Foundry services are in use

Fourteen resource types across five groups: the Foundry account with its `mistral-ocr-4`
and `mistral-medium-3-5` deployments, one Storage account (blob containers plus the
`ingest` queue), the Event Grid bridge, a Flex Consumption Function App, and the
observability stack (Log Analytics, Application Insights, diagnostic settings, alert
rules). Every resource is defined in the Bicep under [`infra/`](infra).

## Layout

`infra/`
- `main.bicep` Foundry account, OCR + Medium 3.5, Storage, Queue, Event Grid, Log Analytics, App Insights, diagnostic settings, dead-letter
- `alerts.bicep` scheduled-query alerts (exceptions, 429 throttling)
- `functionapp.bicep` Flex Consumption Function App hosting

`src/`
- `config.py` env/settings loader
- `observability.py` OpenTelemetry / Azure Monitor wiring + trace propagation
- `upload.py` traced uploader (stamps doc_id + traceparent in blob metadata)
- `ocr_client.py` Mistral OCR 4 client (plain + document_annotation)
- `ocr.py` OCR-only CLI (`uv run src/ocr.py <file>`)
- `agents.py` Extractor agent + DocumentAnalysis schema
- `workflow.py` OcrExecutor to ExtractorExecutor workflow (instrumented)
- `pipeline.py` end-to-end CLI (`uv run src/pipeline.py <file>`)

`function_app/` Azure Function (Python v2), queue-triggered pipeline

**Observability:** traces + metrics via OpenTelemetry → Application Insights, plus
diagnostic settings on all resources → Log Analytics. **End-to-end traceability**:
`upload.py` stamps a `doc_id` + W3C `traceparent` into blob metadata, so the whole
flow (upload → OCR → extractor → result) is **one connected trace**; external
uploads still get a `since_upload_ms` metric from the Event Grid event time. The alert
rules ship in [`infra/alerts.bicep`](infra/alerts.bicep).

## Working in this repo

The module-by-module map is in the [Layout](#layout) section above. Policy lives in
`mortgage_rules.py`, the case state machine in `underwriting_schema.py`, and the pipeline
shape in `workflow.py`. Python is managed with `uv` (`uv sync`, `uv run <script>`,
`uv add <pkg>`).

## Prerequisites

- Python + [uv](https://docs.astral.sh/uv/), Azure CLI (`az login`).
- For the one-command deploy: [Azure Developer CLI](https://aka.ms/azd) (`azd`). The
  deployer needs **Owner** or **User Access Administrator** on the resource group, because
  the deploy grants the Function's identity the Azure AI User role.
- For the manual path: infra deployed (below) and `.env` populated (`cp .env.example .env`).

## Commands

Everything runs through two scripts, no `make` needed. The two you need:

| Command | What it does |
|---|---|
| `scripts/up.sh` | Everything: sign in if needed, find or deploy a stack, write `.env`, start the app, open the browser |
| `scripts/down.sh` | Tear the stack down, purge the account, stop the app |

`up.sh` takes no arguments, it makes the choices for you. Overrides if you want them:
`--rg <name>` (use an existing resource group as the backend), `--deploy` (force a fresh
deploy), `--new-name` (deploy under a fresh Foundry account name when a soft-deleted,
unpurgeable name blocks a same-RG redeploy), `--no-open` (don't open the browser).
Less-common tools:

| Command | What it does |
|---|---|
| `uv run python scripts/probe_agent.py` | Check the hosted agent runs server-side (so portal Traces populate) |
| `uv run python inputs/generate_samples.py` | Regenerate the sample documents |
| `uv run src/pipeline.py <file>` | Run the OCR + extraction pipeline on one document |

Details are in the sections below.

## 1. Provision infra (Bicep)

```bash
az group create -n rg-mistral-ocr-example -l westus
az deployment group create \
  -g rg-mistral-ocr-example \
  -f infra/main.bicep -p infra/main.named.bicepparam
```

Deploys: Foundry account, `mistral-ocr-4` (DataZoneStandard) + `mistral-medium-3-5`
(GlobalStandard), a Storage account (`inbox`/`results`/`deadletter` containers +
`ingest` queue), an Event Grid subscription routing `inbox` BlobCreated events into
the queue, and observability (Log Analytics + App Insights + diagnostic settings).
Outputs feed straight into `.env`.

Optional: deploy the alert rules (uses the `appInsightsId` output):

```bash
az deployment group create -g rg-mistral-ocr-example -f infra/alerts.bicep \
  -p appInsightsId="$(az deployment group show -g rg-mistral-ocr-example -n observability \
     --query properties.outputs.appInsightsId.value -o tsv)" location=westus
```

## 2. Configure + install

```bash
uv sync
```

Then create `.env`. Rather than hand-copying endpoints and keys, generate it from the
deployed stack:

```bash
# azd deploy: reads the current azd environment's outputs
uv run python scripts/sync_env.py

# manual deploy: reads the stack by resource group
uv run python scripts/sync_env.py --rg rg-mistral-ocr-example
```

The script pulls endpoints and deployment names from the stack's Bicep outputs and fetches
the account key + storage connection string via `az`. Or copy the template by hand:
`cp .env.example .env`. Note the deployed Function does not need `.env` at all, its settings
are wired directly by `functionapp.bicep`; `.env` is only for running the app or CLI locally.

## 3. Run (CLI)

```bash
uv run src/ocr.py inputs/sample_invoice.pdf              # OCR only -> .md
uv run src/pipeline.py inputs/sample_invoice.pdf         # full pipeline -> .result.json
```

## 4. Run the event-driven path (Function)

### One command (recommended)

From a fresh clone, `azd up` provisions all infra, registers the hosted extractor agent, and
deploys the Function. It deploys **only** the event pipeline, not the Gradio UI.

```bash
azd up      # provision (main.bicep incl. the Function host) + register agent + deploy
```

Behind it: provision runs `infra/main.bicep`; the `postprovision` hook runs
`scripts/provision_agents.py` to register `mortgage-extractor-agent` on the project; the
`prepackage` hook vendors `src/` into `function_app/`; then the Function publishes. The
extractor runs as a hosted Foundry agent by default (`USE_FOUNDRY_AGENTS=true` is set on the
Function), falling back to `/chat/completions` if a server-side run errors.

### Where to watch progress

Deployment has two phases, and they surface in different places.

**Provisioning** (creating the account, models, storage, Function) is ARM-level work, so it
does **not** appear in the Foundry portal. Watch it in the `azd up` terminal output, or in the
Azure portal (`portal.azure.com`) under the resource group's **Deployments** blade.

**The agent running** shows in the Foundry portal (`ai.azure.com`). Open the project (its
endpoint is the `AZURE_AI_PROJECT_ENDPOINT` output):

| What you're checking | Where | When it appears |
|---|---|---|
| The agent was registered | **Agents** | after the `postprovision` hook runs |
| Per-agent runs | **Tracing** (filter by the agent) | after a document is processed *and* the server-side run succeeds |
| Tokens, latency, volume | **Monitoring** | as documents flow |

The portal's Tracing and Monitoring tabs read from the App Insights connection that
`main.bicep` wires onto the project, so no manual portal setup is needed.

One catch: a per-agent trace appears **only if the hosted path ran server-side**. If Agent
Service falls back to `/chat/completions`, the result is still correct but no per-agent trace
is produced. To tell which happened, check Application Insights (Azure portal) for the log
line `server-side run failed ... using client fallback` from `foundry_agents.py`: present
means it fell back; absent means the hosted path worked and the trace is in the portal.

### Local

```bash
cd function_app
cp local.settings.json.example local.settings.json      # fill in values
func start                                               # requires Azure Functions Core Tools
```

### Manual deploy (advanced / without azd)

```bash
# 1) hosting infra (Flex Consumption plan + app settings, wired to Foundry/Storage/App Insights)
az deployment group create -g rg-mistral-ocr-example -f infra/functionapp.bicep \
  -p foundryAccountName=mistral-ocr-foundry-mistralai \
     storageAccountName=<storageAccountName> \
     appInsightsName=mistral-ocr-foundry-mistralai-ai

# 2) publish the code (vendors src/ then publishes; needs func core tools)
./function_app/deploy.sh
```

Then upload a document and watch it flow (inbox -> Event Grid -> ingest queue -> Function -> results):
```bash
uv run src/upload.py inputs/loan_application_1003.png    # traced upload
```

## 5. Mortgage underwriting demo (human in the loop)

Extends the pipeline from "document to JSON" to "application to decision," with a
person in the loop. OCR 4 reads and classifies each document, Medium 3.5 extracts and
underwrites, and lending policy lives in code (`mortgage_rules.py`), not in prompts.

The app runs locally against the deployed cloud stack (models, hosted agents, and portal
traces are all in the cloud; the UI is not exposed publicly). One command does everything:
sign in if needed, find or deploy a stack, write `.env`, start the app, open the browser.

```bash
scripts/up.sh                 # everything, no arguments
scripts/up.sh --rg <name>     # use an existing resource group as the backend
```

Or run the two processes directly if `.env` is already populated. The backend API and the UI
are separate apps: the UI talks to the backend over HTTP, so start the backend first.

```bash
uv sync
# 1) Backend API (owns OCR, agents, the case store)
cd src && uv run python -m uvicorn api:api --port 8001   # use `python -m uvicorn`, not the bare entrypoint
# 2) UI, in a second terminal, pointed at the backend
cd src && BACKEND_URL=http://127.0.0.1:8001 uv run python -m uvicorn app:ui --port 8000
# Home (pick a view): http://localhost:8000/
# Customer view:      http://localhost:8000/apply
# Reviewer view:      http://localhost:8000/review
# Document chat:      http://localhost:8000/chat
# Backend API docs:   http://localhost:8001/docs
```

- **Customer view** (`/apply`): build the application package. Upload each standard
  document (loan application, pay stub, W-2, bank statement); OCR 4 classifies each
  into a checklist slot, the intake agent asks for anything still missing (documents
  or field values), and once the package is complete a clear case auto-decides while a
  borderline one goes to a reviewer. The reviewer's decision returns to the chat live.
- **Reviewer view** (`/review`): a live pending queue, the package-completeness panel,
  a per-document overlay selector (the original scan layered with the extracted values,
  flagged values marked for review and editable), and approve, decline, or refer.
- Sample inputs (shown as clickable thumbnails that add to the package):
  `loan_application_1003.png`, `paystub.png`, `w2.png`, `bank_statement.png`, and
  `loan_application_thin.png` (borderline, routes to a reviewer). Two PDF loan
  applications (`mortgage_clean.pdf`, `mortgage_borderline.pdf`) are in the tray as well,
  so you can show that PDF and image take the same path. All sample docs live in
  `inputs/` and are tracked in the repo.
- OCR 4 reads and classifies; Medium 3.5 extracts and underwrites. The case store is
  SQLite for the demo, behind an interface so it swaps for Cosmos DB in production.

## Notes / gotchas

- **Regions & quota:** OCR 4 (Preview) is region-limited; on this subscription
  `GlobalStandard` OCR quota was exhausted, so OCR uses `DataZoneStandard`. Check with
  `az cognitiveservices usage list -l <region>`.
- **Redeploy collision (soft-deleted workspace):** an AI Foundry account keeps a backing
  Azure ML workspace shadow of the same name. Teardown soft-deletes it, its purge is
  asynchronous, and it never shows in the Cognitive Services soft-deleted list, so a same-RG,
  same-name redeploy can fail with `Soft-deleted workspace exists` (`Kind: AmlRp`). `down.sh`
  fires a best-effort purge of the shadow, but some subscriptions do not serve that purge
  endpoint. When that happens, deploy under a fresh name: `scripts/up.sh --new-name` (or set
  `AZURE_FOUNDRY_NAME_SUFFIX` yourself, which salts the derived account name in `main.bicep`).
- **Blob-to-queue bridge:** blob events do not land on a queue on their own; **Event Grid**
  subscribes to the storage account's `BlobCreated` events and delivers them into the
  `ingest` Storage Queue. Those queue messages are Event Grid events, **base64-encoded**, so
  the Function decodes them before processing.
- **OCR4 annotation:** classification/summary via `document_annotation_format` is
  capped at ~8 pages and can time out; content-safety applies to annotations.
- **Tool-calling limitation:** the Foundry Mistral deployment currently
  mis-serializes function/tool calls (`[TOOL_CALLS]` leaks into content). So agents
  are called directly / orchestrated by the Workflow engine, not via `Agent.as_tool()`.
- **Chat client:** use `OpenAIChatCompletionClient` (the `/chat/completions` route),
  not `OpenAIChatClient` (Responses API, which Mistral rejects).
- **Functions & uv:** Azure Functions deploys via `requirements.txt` (pip), so
  `function_app/requirements.txt` mirrors the root `pyproject.toml`.

## Teardown

The whole system is one resource group. While idle it costs almost nothing, since the
Function App (`FC1` Flex Consumption) and the Foundry models bill per use, not per hour.
To pause instead of delete, scale the Function App to zero and leave the models in place;
they only bill per call.

### One command

```bash
scripts/down.sh                 # tear down, purge the account, stop the app
scripts/down.sh --rg <name>     # for a stack deployed manually with az
```

It runs `azd down --purge --force` (or, with `--rg`, deletes the group and purges the
account). The `--purge` matters: Cognitive Services accounts and Log Analytics workspaces are
soft-deleted, so a plain delete leaves them holding the name for ~48 hours and blocks a clean
redeploy. The Bicep template itself never deletes anything, it only creates or updates;
teardown is always `down.sh`, not a Bicep run.
