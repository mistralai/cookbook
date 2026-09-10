# The Self-Tooling Agent: Dynamic Tool Generation with Codestral

*By (https://github.com/jellewas)*

An agent that **writes, tests, and registers its own tools at runtime** — then reuses them across turns. Unlike code interpreters that run throwaway scripts, this agent builds a **persistent, growing toolkit** where each tool is syntax-checked, tested, and iteratively fixed by Codestral before registration.

## What Makes This Different from Code Interpreters?

| | Code Interpreter | Self-Tooling Agent |
|---|---|---|
| **Execution** | Throwaway scripts | Persistent, reusable tools |
| **Memory** | Isolated per run | Tools accumulate across turns |
| **Composition** | None | Tool B can call Tool A |
| **Validation** | None | Syntax check → test → iterative fix |
| **Dependencies** | Pre-installed | Detected, user-approved, auto-installed |
| **Output** | Results only | Exportable Python module |

## How It Works

The agent uses a **dual-model architecture**:

- **Mistral Large** (`mistral-large-latest`) — orchestrates decisions: use an existing tool, create a new one, or answer directly.
- **Codestral** (`codestral-latest`) — generates, validates, tests, and iteratively fixes tool code.

### The Tool Forge Pipeline

```
User request
    → Mistral Large decides: NEED_TOOL / USE_TOOL / DIRECT
    → If NEED_TOOL:
        1. Design   — Codestral designs function spec (name, params, description)
        2. Generate  — Codestral writes the Python function
        3. Validate  — ast.parse() checks syntax; loops on failure
        4. Packages  — Detects third-party imports, asks user before installing
        5. Test      — Codestral generates test cases, runs them
        6. Fix       — If tests fail, Codestral rewrites; up to 3 attempts
        7. Register  — Tool is stored in the persistent registry
    → If USE_TOOL:
        Execute the existing tool with provided arguments
```

## Prerequisites

- Python 3.10+
- A [Mistral AI API key](https://console.mistral.ai/api-keys/)

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

Set your Mistral API key:

```bash
export MISTRAL_API_KEY="your_api_key_here"
```

## Usage

Open the notebook in Jupyter or Google Colab and run the cells sequentially:

```bash
jupyter notebook self_tooling_agent_cookbook.ipynb
```

The demo starts with **zero tools** and progressively builds a toolkit:

1. **Request 1** — "Analyze sentiment" → agent creates `sentiment_analysis` tool
2. **Request 2** — "Calculate statistics" → agent creates `calculate_statistics` tool
3. **Request 3** — "Analyze sentiment of another text" → agent **reuses** the existing tool (no creation!)
4. **Request 4** — "Find outliers" → agent creates `identify_outliers_iqr` tool

At the end, all tools are exported as a reusable `generated_toolkit.py` module.

## Key Features

- 🔧 **Iterative code generation** — Codestral fixes syntax and runtime errors automatically (up to 3 attempts)
- 📦 **Dependency management** — Detects third-party imports and asks the user before installing
- 🧪 **Auto-testing** — Each tool is tested with LLM-generated test cases before registration
- 🔄 **Tool composition** — New tools can call previously created tools
- 📤 **Exportable toolkit** — The entire registry exports as a standalone Python module
- 🔁 **Tool versioning** — Re-creating a tool increments its version

## Files

```
├── self_tooling_agent_cookbook.ipynb   # Main notebook
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```
