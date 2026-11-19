# Triage an insurance claim with vision

This workflow analyzes the photos of an insurance claim in parallel, checks the claimant's story for inconsistencies, routes the claim by a fixed severity rule, and scores fraud risk.

*Anton Kluge, [@AntonKluge](https://github.com/AntonKluge), Mistral AI*

The workflow runs one `analyze_photo` activity per photo, compares the photo findings with the claimant's description, and routes the claim to a queue. The severity and routing queue come from workflow code, not from a model call, so every decision is auditable. The run ends with a triage report that cites the evidence behind each finding.

## Prerequisites

To complete this cookbook, you will need:

- macOS or Linux
- git
- The Mistral CLI, installed in the next step. `mistral apps dev` also needs Bun, which the installer offers to install.
- A Mistral account

## Environment setup

Run these steps once from the directory where you want the project.

### Install

Install the Mistral CLI with the installer script. The installer offers to install Bun, which `mistral apps dev` needs to run the worker.

```bash
curl -fsSL https://mistral.ai/cli/install.sh | bash
```

Create a new Workflows project with `mistral workflows init`. On a terminal, it asks for your API key, checks it, and saves it in the project's `.env`.

```bash
mistral workflows init insurance-claims-cookbook
```

From the cookbook repository root, copy the `insurance_claims` folder into the worker's workflows directory. The worker discovers every workflow package under `apps/worker/src/worker/workflows/`, including subpackages.

```bash
cp -R mistral/workflows/insurance-claims/insurance_claims \
  insurance-claims-cookbook/apps/worker/src/worker/workflows/
```

## Run the worker

Start the worker locally from the project root. It reloads when you edit a workflow file.

```bash
cd insurance-claims-cookbook
mistral apps dev
```

## Start the workflow in Studio

Open [Workflows in Studio](https://console.mistral.ai/build/workflows?workflowSearch=insurance-claims-triage), select `insurance-claims-triage`, and click Start workflow. Paste the contents of `insurance_claims/sample_data/claim_high_severity.json`, which analyzes two photos of a totaled car and sends the claim to the high severity path.

The photo paths in the sample file point at the sample photos in your local project, so they work with `mistral apps dev`. In a deployed worker, pass HTTPS URLs for the photos.

The package ships three sample inputs in `insurance_claims/sample_data/`. Each one matches a routing outcome:

| File | Photos | Routing queue |
|---|---|---|
| `claim_high_severity.json` | Two totaled vehicles | `full-investigation` |
| `claim_medium_severity.json` | Rear dent and cracked tail light | `adjuster-review` |
| `claim_low_severity.json` | A keyed door panel | `fast-track` |

To try the other routing outcomes, paste the contents of `claim_medium_severity.json` or `claim_low_severity.json`. To use your own claim, set `photos` to HTTPS URLs or to local paths inside the worker directory.

## Inspect the execution

Open [the executions page](https://console.mistral.ai/build/workflows/executions) and select the run. The timeline shows the steps below in order.

The `analyze_photo` activities start together, one per photo, and each retries up to three times with a 2x backoff if the vision call fails. Then `check_consistency` compares the photo findings with the claimant's description, and `score_fraud_risk` returns a score with cited indicators.

`_aggregate_severity` takes the worst damage label across all photos, and `_route` maps it to a queue. Both run in workflow code, so the routing decision shows as a value in the execution, not as a separate step.

The run ends with `generate_triage_report`, which returns a `TriageReport` with the findings, the consistency check, the fraud score, and cited evidence.

## Summary

This cookbook ran the `insurance-claims-triage` workflow from a local worker and read its execution in Studio.

**What you built:**
- A workflow that analyzes claim photos in parallel with `execute_activities_in_parallel`
- A consistency check and a fraud score that cite their evidence
- Deterministic severity and routing rules that run in workflow code

**Mistral features used:**
- Workflows, with parallel activities and retry policies
- Vision and chat completions with `mistral-small-latest`
- Structured outputs through `chat_parse_to_model` with Pydantic models

[View your workflow executions in Studio](https://console.mistral.ai/build/workflows/executions).
