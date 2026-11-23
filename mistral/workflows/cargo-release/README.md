# Release cargo after dangerous goods and customs checks with a workflow

This workflow reads a shipping document with OCR, classifies the cargo, runs a dangerous goods sub-workflow and a customs check, and pauses for a compliance officer before it issues the release certificate.

*Anton Kluge, [@AntonKluge](https://github.com/AntonKluge), Mistral AI*

The workflow reads the shipping document, classifies the cargo, checks dangerous goods when they apply, and validates customs rules. A missed check means a detained shipment or a safety incident at port.

When it finds an anomaly, such as a missing emergency contact on a UN 1203 shipment or a failed customs rule, it sends a review message and pauses on `wait_for_input`. No container stays busy while the workflow waits. A reviewer approves or blocks the shipment in Studio, and the workflow resumes from the approval step.

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
mistral workflows init cargo-release-cookbook
```

From the cookbook repository root, copy the `cargo_release` folder into the worker's workflows directory. The worker discovers every workflow package under `apps/worker/src/worker/workflows/`, including subpackages, module by module. Keep the package's `__init__.py` free of imports from the workflow module: importing the workflow classes there registers each workflow twice.

```bash
cp -R mistral/workflows/cargo-release/cargo_release \
  cargo-release-cookbook/apps/worker/src/worker/workflows/
```

## Run the worker

Start the worker locally from the project root. It reloads when you edit a workflow file.

```bash
cd cargo-release-cookbook
mistral apps dev
```

## Start the workflow in Studio

Start `cargo-release-compliance` twice, once with each sample document. The two runs cover the two outcomes of the workflow: a clean pass and a flagged shipment that waits for a reviewer.

The first run uses the anomaly document, a UN 1203 flammable shipment with no emergency contact. The dangerous goods sub-workflow flags the missing contact, the workflow pauses for review, and you approve or block it in Studio:

```json
{
  "document_uri": "src/worker/workflows/cargo_release/sample_data/shipping_doc_anomaly.png",
  "shipment_id": "BL-2024-RTD-004812"
}
```

The second run uses the normal document, a cotton textile shipment with no dangerous goods. It passes the dangerous goods and customs checks, so there is no review step: the workflow goes straight to the release certificate and returns `status: "released"`:

```json
{
  "document_uri": "src/worker/workflows/cargo_release/sample_data/shipping_doc_normal.png",
  "shipment_id": "BL-2024-HAM-009371"
}
```

The customs rules cover HS code, country of origin, consignee name and address, declared value, and gross weight, plus an emergency contact for dangerous goods. The normal sample has all of them, so it does not fail customs. A customs failure is the other way to reach the review pause.

`document_uri` accepts an `https://` URL, a `data:` URI, a `file://` URI, or a path relative to `apps/worker`, which is the working directory of the local worker. `http://` URLs are not fetched, and local paths resolve against the worker directory: the activity refuses to read files outside it, and a path that does not exist fails with a raw `FileNotFoundError`.

## Inspect the execution

Open [the executions page](https://console.mistral.ai/build/workflows/executions) and select the run. The timeline shows these steps:

- `ocr_extract_shipping_doc` shows the extracted text and allows three attempts.
- `classify_cargo_type` returns a `CargoClassification` with the cargo type and the model's reasoning.
- `dangerous-goods-validation` runs as a child workflow with its own events. Open it to see the `validate_dangerous_goods` activity and the `DangerousGoodsResult` it returns.
- `validate_customs_compliance` returns a `ComplianceCheck` with `passed`, `failed_rules`, and `notes`.

On the anomaly path, the workflow sends a review message and pauses at `wait_for_input`. The timeline stops there until you approve or block the shipment in Studio. Approving runs `generate_release_certificate` and returns a `CargoReleaseResult` with `status: "released"`. Blocking returns the same model with `status: "blocked"` and a `block_reason`, and the certificate step doesn't run. On the normal path, no review step appears: `generate_release_certificate` runs after `validate_customs_compliance` and the run ends with `status: "released"`.

## Summary

This cookbook ran a cargo release workflow end to end, once as a clean pass and once paused for a human decision on a flagged shipment.

**What you built:**
- A workflow that extracts text from a shipping document with Mistral OCR
- A dangerous goods validation sub-workflow with its own retry surface
- A human-in-the-loop review that pauses on `wait_for_input` and resumes after approval or block

**Mistral features used:**
- Workflows, with `InteractiveWorkflow`, `wait_for_input`, and `execute_workflow` for the child workflow
- Mistral OCR through the `mistralai_ocr` workflow activity
- Structured outputs through `chat_parse_to_model` with Pydantic models
- Chat completions through the `mistralai_chat_complete` workflow activity with `mistral-small-latest`

To learn how human-in-the-loop steps and sub-workflows work, see the [workflows documentation](https://docs.mistral.ai/studio/workflows/building-workflows/workflows).
