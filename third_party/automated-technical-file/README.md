# Codebase-to-Wiki Q&A — on Mistral

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mistralai/cookbook/blob/main/third_party/automated-technical-file/notebooks/automated_technical_file.ipynb)

---

## Problem Statement

Documentation rots the moment code changes. Operational knowledge — how to install the system, why the last run failed, what a specific component does — is scattered across source files, run logs, and whoever set it up first. A new operator cannot answer "how do I install this?" or "why did the last run fail?" without spelunking through the repository.

The running example is **Robot Ross**: an autonomous robotic arm that draws customer orders, narrates in real time, and records every action in a structured JSONL ledger. The ATF — Automated Technical File — is that ledger's grounded operational memory: a cited, regenerable wiki compiled automatically from the code and logs themselves, never written by hand.

The same pipeline generalises to any system with structured logs.

---

## What This Does

1. **Compiles** a wiki from source code and operational logs — not written by hand.
2. **Answers** questions with citations to the actual source file or log event, so every claim is verifiable.
3. **Stays current**: re-run the generator on fresh logs and the wiki reflects the current system state. No manual editing required.

Live deployment: [api.robotross.art/atf-mistral](https://api.robotross.art/atf-mistral/)

---

## How It Differs from IndustrialKnowledgeAgent

[IndustrialKnowledgeAgent](../../mistral/agents/non_framework/industrial_knowledge_agent/IndustrialKnowledgeAgent.ipynb) does RAG over manuals a human has already written. This cookbook compiles the knowledge base automatically from code and telemetry. The distinction matters: curated documentation is always behind the current system; a generated wiki is regenerated from the latest logs.

---

## Architecture

![Pipeline diagram](img/architecture.png)

*Source Code + Run Logs → Operational Ledger → Wiki Generator → Markdown Wiki → Mistral Document Library → Cited Answer.*

Full layer-by-layer breakdown: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Quickstart

```bash
pip install -r requirements.txt
export MISTRAL_API_KEY=<your-key>   # console.mistral.ai
jupyter notebook notebooks/automated_technical_file.ipynb
```

Run cells top to bottom. Step 1 (wiki generation) runs offline. Steps 2–4 require `MISTRAL_API_KEY`.

---

## What's in This Folder

| Path | Description |
| :--- | :--- |
| `notebooks/automated_technical_file.ipynb` | End-to-end notebook walkthrough |
| `tools/ledger_to_md.py` | JSONL → Markdown corpus generator |
| `tools/runtime_adapter.py` | Hosted Mistral / local Ministral switch |
| `artifacts/ledger/sample_events.jsonl` | Sample robot run log |
| `artifacts/wiki/sample_run_summary.md` | Generated wiki output (committed for reference) |

---

## Where This Goes Next

Everything here grounds a stock Mistral model at **inference time**: the wiki corpus is supplied as retrieval context per query, with citations proving the answer came from the corpus rather than the weights.

The deeper version encodes this corpus — and the institutional vocabulary it represents — at **training time**. That is precisely what [Mistral Forge](https://mistral.ai/news/forge/) is built for. The corpus this pipeline compiles — structured, regenerable, drawn straight from code and telemetry — is the kind of input a Forge training pipeline consumes.

---

*Built by [Agentegra](https://agentegra.com) / [Big Bear Engineering](https://bigbearengineering.com). Apache-2.0.*
