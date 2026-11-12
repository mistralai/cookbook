# Modernize a legacy Python codebase with a workflow

This workflow scans a Python repository, modernizes one file per child workflow, validates the generated code, and pauses for a reviewer before it writes a PR proposal.

*Anton Kluge, [@AntonKluge](https://github.com/AntonKluge), Mistral AI*

Modernizing a legacy codebase with a single LLM call does not scale. One large prompt is slow, expensive, and gives the reviewer one diff to accept or reject as a whole. This workflow splits the work per file. Each file runs in its own child workflow, so a file that generates invalid Python retries on its own without touching the others. When every file finishes, the parent aggregates the results into a change set and pauses on `wait_for_input`. A reviewer approves or discards the change set in Studio, and only an approved change set produces a PR proposal.

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
mistral workflows init code-modernization-cookbook
```

From the cookbook repository root, copy the `code_modernization` folder into the worker's workflows directory. The worker discovers every workflow package under `apps/worker/src/worker/workflows/`, including subpackages.

```bash
cp -R mistral/workflows/code-modernization/code_modernization \
  code-modernization-cookbook/apps/worker/src/worker/workflows/
```

## Run the worker

Start the worker locally from the project root. It reloads when you edit a workflow file.

```bash
cd code-modernization-cookbook
mistral apps dev
```

## Start the workflow in Studio

Open [Workflows in Studio](https://console.mistral.ai/build/workflows?workflowSearch=code-modernization), select `code-modernization`, and click Start workflow. Paste this input, which points at the sample repository in the package:

```json
{
  "repo_path": "src/worker/workflows/code_modernization/sample_data/legacy_repo",
  "target": "Python 2.7 to 3.12"
}
```

`repo_path` is an absolute path to a repository inside the worker directory, or a path relative to `apps/worker`, which is the working directory of the local worker. Paths outside the worker directory are rejected. `target` is optional and defaults to `Python 2.7 to 3.12`.

To modernize a different repository, copy it into `apps/worker` and pass its absolute path. The scan skips `__init__.py` files, `__pycache__`, `.venv`, `.git`, and `node_modules`.

## Inspect the execution

Open [the executions page](https://console.mistral.ai/build/workflows/executions) and select the run. The `scan_files_to_modernize` activity returns the three sample files. Each file then appears as its own `code-modernization-file` child execution, and all three start at nearly the same timestamp because the parent dispatches them with `asyncio.gather`.

Open a child execution to see its steps. `read_file` returns the file, `modernize_file_llm` calls `mistral-small-latest` and parses the answer into a `ModernizedFile`, and `validate_python_syntax` compiles the generated code in a subprocess. When the generated code fails to compile, the child runs `modernize_file_llm` again, up to three attempts. If all three fail, the child returns the file with `syntax_valid: false` and the parent lists it as WARN in the review summary.

After the children finish, the parent sends a summary message and pauses at `wait_for_input` with the label `Change Set Review`. The workflow holds no compute while it waits. Click Open PR to run `write_pr_proposal`, which writes `PR_PROPOSAL.md` into the sample repository and returns a `WorkflowResult` with `status: "approved"`. Click Discard to return `status: "declined"` without writing the file.

## Summary

This cookbook ran a code modernization workflow that fans out one child workflow per source file, validates each generated file, and waits for a human decision before it writes a PR proposal.

**What you built:**
- A parent workflow that scans a repository and dispatches one child workflow per file in parallel
- A child workflow with an explicit modernize and validate retry loop
- A human-in-the-loop review that pauses on `wait_for_input` and only writes the PR proposal after approval

**Mistral features used:**
- Workflows, with `InteractiveWorkflow`, `wait_for_input`, and `execute_workflow` for the child workflows
- Structured outputs through `chat_parse_to_model` with Pydantic models
- Chat completions with `mistral-small-latest`
- Assistant messages through `send_assistant_message`

To learn how sub-workflows and human-in-the-loop steps work, see the [workflows documentation](https://docs.mistral.ai/studio/workflows/building-workflows/workflows).
