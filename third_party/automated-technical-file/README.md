# Automated Technical File — Mistral AI Cookbook Integration

This cookbook demonstrates how to use Mistral AI models to produce and maintain an **Automated Technical File** (ATF) — a queryable, citation-backed compliance artifact generated automatically from the operational logs of an AI system.

The reference system is **Robot Ross**, an autonomous robotic painting arm. Every action the arm takes is logged. Those logs are compiled into a structured wiki corpus, indexed by Mistral's Document Library, and made queryable with full source citations — satisfying the record-keeping, transparency, and oversight requirements of the EU AI Act for high-risk AI systems.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Robot Ross                               │
│           (autonomous painting arm / agentic system)            │
└─────────────────────────┬────────────────────────────────────────┘
                          │  structured action events
                          ▼
                  ┌───────────────┐
                  │  JSONL Ledger │  ◄── every tool call, decision,
                  │  (run logs)   │       sensor read, and outcome
                  └───────┬───────┘
                          │  ledger_to_md.py  (notebook cell 2)
                          ▼
                  ┌───────────────┐
                  │  Wiki Corpus  │  ◄── one Markdown page per
                  │  (Markdown)   │       session / component / topic
                  └───────┬───────┘
                          │  Mistral Files + Document Library  (cells 3-4)
                          ▼
              ┌─────────────────────────┐
              │  Mistral Document       │
              │  Library (chunked,      │
              │  indexed, hosted)       │
              └────────────┬────────────┘
                           │  cited Q&A  (atf_qa.py / cell 4)
                           ▼
              ┌─────────────────────────┐
              │  Answer + Citations     │  "The robot paused at 14:23
              │  (Technical File query) │   because sensor_L read 0 V.
              └─────────────────────────┘  Source: session_042.md §3
```

Queries resolve to answers that cite the exact wiki page and section, creating an auditable chain from natural-language question back to the raw operational log.

## EU AI Act Compliance Framing

The ATF pipeline maps directly to three Articles of the EU AI Act that apply to **high-risk AI systems** (Annex III):

| Article | Requirement | How this cookbook satisfies it |
| :--- | :--- | :--- |
| **Art. 12 — Logging** | Systems must automatically record events throughout their lifetime. | The JSONL ledger captures every action, decision, and sensor event in structured, append-only form. Nothing is discarded between the robot acting and the ledger entry being written. |
| **Art. 13 — Transparency** | Operators must be able to interpret system outputs and trace their origin. | `ledger_to_md.py` converts raw logs into human-readable wiki pages. The cited Q&A layer makes any fact traceable: every answer includes the source document and section. |
| **Art. 14 — Human oversight** | Humans must be able to effectively oversee the system during operation. | The Technical File is queryable in plain language. Operators can ask "why did the arm stop?" and receive a cited answer grounded in the actual log, enabling informed intervention. |

This architecture means the Technical File is not a document written after the fact — it is compiled continuously from the system's own operational record.

## Requirements

```bash
pip install -r requirements.txt
```

Only one API key is required:

```bash
export MISTRAL_API_KEY=<your-key>   # from console.mistral.ai
```

No other credentials are needed. The `runtime_adapter.py` also supports a local [Ollama](https://ollama.com/) backend (Ministral 3B) for offline use — see `tools/runtime_adapter.py --help`.

## Notebooks

| Notebook | Description |
| :--- | :--- |
| `automated_technical_file.ipynb` | **End-to-end walkthrough.** Six cells: (1) setup and imports, (2) ledger → wiki corpus via `ledger_to_md`, (3) upload corpus to Mistral Files, (4) create Document Library and cited Q&A, (5) structured-output log analysis, (6) voice shell via Voxtral (stub, opt-in). |

## Tools

| File | Description |
| :--- | :--- |
| `tools/runtime_adapter.py` | Model abstraction layer. Tries hosted Mistral first, falls back to local Ministral (Ollama), then `aichat`, then generic Ollama. Run `--check` to verify your backend. |

## Attribution

Developed by [Agentegra](https://agentegra.com) / [Big Bear Engineering](https://bigbearengineering.com).

## License

Dual-licensed under **Apache 2.0** (matching the parent [mistralai/cookbook](https://github.com/mistralai/cookbook) repository) and **MIT** (for standalone use). See [LICENSE](LICENSE) for details.
