# Modernize legacy Python code with GLM

Use the `zai-glm-5-2` (GLM) model and the GitHub connector to read outdated Python code from a repository and produce modernized versions.

> **API status:** Agents use `client.beta.agents`. Conversations use `client.beta.conversations`. These are **beta** endpoints and may change.

---

## Prerequisites

To complete this cookbook, you will need:
- Python 3.10+
- A Mistral account and API key
- The `github_app` connector authenticated in Studio

## Environment setup

### Install

Install the Mistral Python SDK and `python-dotenv` for loading your API key from a `.env` file:

```bash
pip install mistralai python-dotenv
```

### Required environment variables

To complete this cookbook, you'll need a Mistral API key. In [Studio](https://console.mistral.ai), navigate to the [API keys section](https://console.mistral.ai/home?profile_dialog=api-keys) and create a new API key.

Create a `.env` at the root of your project and add your Mistral API key:

```
MISTRAL_API_KEY=your-mistral-api-key
```

### Authenticate the GitHub connector

The `github_app` connector is a built-in connector that provides access to GitHub repositories. You authenticate it through Studio, not through the API:

1. Open [Studio](https://console.mistral.ai) and go to **Connectors**
2. Find `github_app` and authenticate with your GitHub account
3. Grant access to the `mistralai/cookbook` repository (or your fork)

Once authenticated, the connector is available by name (`"github_app"`) in any agent or conversation.

---

## Step 1 — Initialize the client

Create `modernize_code.py` in your project directory:

```bash
touch modernize_code.py
```

Open the file and add the imports and client initialization. The remaining steps build the agent, conversation, and output logic.

```python
"""Modernize legacy Python code using GLM and the GitHub connector."""

import asyncio
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from mistralai.client import Mistral

load_dotenv()

# Step 1 — Initialize the client
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


async def main() -> None:
    # Step 2 — Define the modernization target
    # Step 3 — Create the agent with the GitHub connector
    # Step 4 — Start the conversation and extract results
    # Step 5 — Save the modernized code
    pass


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Step 2 — Define the modernization target

Specify the repository, branch, and file paths to modernize. This cookbook targets a pair of intentionally outdated Python files in the `legacy_app/` directory of this repository.

Add the following constants below the client initialization:

```python
# Step 2 — Define the modernization target
REPO = "mistralai/cookbook"
BRANCH = "main"
FILE_PATHS = [
    "mistral/agents/glm_code_modernizer/legacy_app/app.py",
    "mistral/agents/glm_code_modernizer/legacy_app/utils.py",
]
```

These files contain a CLI to-do list manager written with Python 2/3.5-era patterns: `%`-formatting, `os.path`, manual `open()`/`close()`, bare `except:`, `type()` checks, `sys.argv` parsing, and no type hints.

---

## Step 3 — Create the agent with the GitHub connector

Create an agent that pairs GLM with the GitHub connector. The agent's instructions tell it exactly which patterns to modernize and how to format the output.

Add the following inside `main`, wrapped in a `try`/`finally` block to ensure cleanup:

```python
    agent_id: str | None = None
    try:
        # Step 3 — Create the agent with the GitHub connector
        file_list = "\n".join(f"- `{path}`" for path in FILE_PATHS)
        agent = await client.beta.agents.create_async(
            name="code_modernizer",
            description="Reads legacy Python files from GitHub and produces modernized versions",
            model="zai-glm-5-2",
            instructions=(
                "You are an expert Python developer who modernizes legacy code.\n\n"
                "IMPORTANT: You MUST use the GitHub connector to read each file's "
                "actual contents from the repository. Do NOT guess or invent code. "
                "Read the real source code first, then modernize it.\n\n"
                "Your workflow:\n"
                "1. Use the GitHub connector to read each file from the repository\n"
                "2. Analyze the actual code you retrieved for legacy patterns\n"
                "3. Produce a modernized version of THAT SAME CODE — same logic, "
                "same structure, same functionality, but with modern Python idioms\n\n"
                "Apply these modernizations to the code you read:\n"
                "- Replace %-formatting with f-strings\n"
                "- Replace os.path with pathlib.Path\n"
                "- Replace manual open()/close() with 'with' statements\n"
                "- Replace json.loads(f.read()) with json.load(f)\n"
                "- Replace bare except: with specific exceptions\n"
                "- Replace type() checks with isinstance()\n"
                "- Replace mutable default arguments with None sentinels\n"
                "- Replace sys.argv parsing with argparse\n"
                "- Add type hints to all functions\n"
                "- Replace == True/False comparisons with truthiness checks\n"
                "- Replace range(len(...)) with enumerate or direct iteration\n\n"
                "Return each modernized file in a separate ```python code fence. "
                "Include a comment with the original filename at the top of each block. "
                "The modernized code must preserve the original functionality exactly."
            ),
            tools=[
                {
                    "type": "connector",
                    "connector_id": "github_app",
                },
            ],
        )
        agent_id = agent.id
        print(f"Created agent: {agent.name} ({agent.id})")
```

Two things to note:

- **Model**: `zai-glm-5-2` is a code-generation model designed for producing and transforming code.
- **Connector**: `"github_app"` references the built-in GitHub connector by name. The agent uses it to read repository files server-side.

---

## Step 4 — Start the conversation and extract results

Start a conversation with the agent. The prompt tells it which repository and files to read. The GitHub connector handles the file access — the model reads the files, analyzes the patterns, and returns modernized code in a single response.

Add the following helper functions above `main`:

```python
def extract_python_blocks(text: str) -> list[str]:
    """Extract Python code blocks from the model response."""
    return re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)


def get_response_text(response) -> str:
    """Extract text content from a conversation response."""
    parts = []
    for output in response.outputs:
        if output.type == "message.output":
            content = output.content
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.append(
                    "".join(
                        chunk.text if hasattr(chunk, "text") else str(chunk)
                        for chunk in content
                    )
                )
    return "\n".join(parts)
```

Then continue inside the `try` block after the agent creation:

```python
        # Step 4 — Start the conversation
        print(f"Reading files from {REPO} and modernizing...")
        print("This may take a few minutes.\n")

        response = await client.beta.conversations.start_async(
            agent_id=agent.id,
            inputs=[
                {
                    "role": "user",
                    "content": (
                        f"Read these files from the `{REPO}` repository "
                        f"(branch: `{BRANCH}`) using the GitHub connector, "
                        f"then produce modernized versions of each:\n\n{file_list}\n\n"
                        "Return each modernized file in a ```python code fence "
                        "with a comment at the top indicating the original filename."
                    ),
                }
            ],
            timeout_ms=600_000,
        )

        text = get_response_text(response)
```

The `timeout_ms=600_000` parameter sets a 10-minute timeout. GLM generates large code outputs, so this headroom prevents the request from timing out.

When you pass `agent_id` instead of `model`, the conversation uses the agent's model, instructions, and tools automatically. The Conversations API handles all tool calls server-side — the agent decides to call the GitHub connector to read files, processes the content, and returns the modernized code, all in one response cycle.

---

## Step 5 — Save the modernized code

Extract the Python code blocks from the response and save each one to a `modernized/` directory:

```python
        # Step 5 — Save the modernized code
        code_blocks = extract_python_blocks(text)

        if not code_blocks:
            print("No Python code blocks found in the response.")
            print("\nRaw response:\n")
            print(text)
            return

        output_dir = Path("modernized")
        output_dir.mkdir(exist_ok=True)

        filenames = [Path(p).name for p in FILE_PATHS]
        for i, block in enumerate(code_blocks):
            name = filenames[i] if i < len(filenames) else f"file_{i}.py"
            output_path = output_dir / name
            output_path.write_text(block.strip() + "\n", encoding="utf-8")
            print(f"Saved: {output_path}")

        print(f"\nModernized {len(code_blocks)} file(s) in {output_dir}/")
```

---

## Cleanup

Delete the agent when you're done. The `finally` block ensures cleanup even if an error occurs:

```python
    finally:
        # Cleanup — Delete the agent
        if agent_id:
            await client.beta.agents.delete_async(agent_id=agent_id)
            print(f"Deleted agent: {agent_id}")
```

---

## Run

Once all steps are in place, run the script:

```bash
python modernize_code.py
```

The script creates a GLM agent with the GitHub connector, reads the legacy files from the repository, produces modernized versions, and saves them to a `modernized/` directory.

Example output:

```
Created agent: code_modernizer (a1b2c3d4-5678-90ab-cdef-1234567890ab)
Reading files from mistralai/cookbook and modernizing...
This may take a few minutes.

Saved: modernized/app.py
Saved: modernized/utils.py

Modernized 2 file(s) in modernized/
Deleted agent: a1b2c3d4-5678-90ab-cdef-1234567890ab
```

---

## Try different targets

Change `REPO`, `BRANCH`, and `FILE_PATHS` to modernize files from a different repository. You can also adjust the agent's instructions to focus on different modernization patterns.

**Modernize a Django views file:**

```python
REPO = "your-org/your-django-app"
FILE_PATHS = ["myapp/views.py"]
```

**Target a specific branch or pull request:**

```python
BRANCH = "feature/legacy-cleanup"
```

**Modernize JavaScript instead of Python:**

Update the agent instructions to target JavaScript patterns (e.g., `var` to `const`/`let`, callbacks to `async`/`await`, CommonJS to ES modules) and change the code fence extraction to look for ` ```javascript ` blocks.

---

## Complete script

For reference, here is the full script with all steps combined:

```python
"""Modernize legacy Python code using GLM and the GitHub connector."""

import asyncio
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from mistralai.client import Mistral

load_dotenv()

# Step 1 — Initialize the client
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# Step 2 — Define the modernization target
REPO = "mistralai/cookbook"
BRANCH = "main"
FILE_PATHS = [
    "mistral/agents/glm_code_modernizer/legacy_app/app.py",
    "mistral/agents/glm_code_modernizer/legacy_app/utils.py",
]


def extract_python_blocks(text: str) -> list[str]:
    """Extract Python code blocks from the model response."""
    return re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)


def get_response_text(response) -> str:
    """Extract text content from a conversation response."""
    parts = []
    for output in response.outputs:
        if output.type == "message.output":
            content = output.content
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.append(
                    "".join(
                        chunk.text if hasattr(chunk, "text") else str(chunk)
                        for chunk in content
                    )
                )
    return "\n".join(parts)


async def main() -> None:
    agent_id: str | None = None
    try:
        # Step 3 — Create the agent with the GitHub connector
        file_list = "\n".join(f"- `{path}`" for path in FILE_PATHS)
        agent = await client.beta.agents.create_async(
            name="code_modernizer",
            description="Reads legacy Python files from GitHub and produces modernized versions",
            model="zai-glm-5-2",
            instructions=(
                "You are an expert Python developer who modernizes legacy code.\n\n"
                "IMPORTANT: You MUST use the GitHub connector to read each file's "
                "actual contents from the repository. Do NOT guess or invent code. "
                "Read the real source code first, then modernize it.\n\n"
                "Your workflow:\n"
                "1. Use the GitHub connector to read each file from the repository\n"
                "2. Analyze the actual code you retrieved for legacy patterns\n"
                "3. Produce a modernized version of THAT SAME CODE — same logic, "
                "same structure, same functionality, but with modern Python idioms\n\n"
                "Apply these modernizations to the code you read:\n"
                "- Replace %-formatting with f-strings\n"
                "- Replace os.path with pathlib.Path\n"
                "- Replace manual open()/close() with 'with' statements\n"
                "- Replace json.loads(f.read()) with json.load(f)\n"
                "- Replace bare except: with specific exceptions\n"
                "- Replace type() checks with isinstance()\n"
                "- Replace mutable default arguments with None sentinels\n"
                "- Replace sys.argv parsing with argparse\n"
                "- Add type hints to all functions\n"
                "- Replace == True/False comparisons with truthiness checks\n"
                "- Replace range(len(...)) with enumerate or direct iteration\n\n"
                "Return each modernized file in a separate ```python code fence. "
                "Include a comment with the original filename at the top of each block. "
                "The modernized code must preserve the original functionality exactly."
            ),
            tools=[
                {
                    "type": "connector",
                    "connector_id": "github_app",
                },
            ],
        )
        agent_id = agent.id
        print(f"Created agent: {agent.name} ({agent.id})")

        # Step 4 — Start the conversation
        print(f"Reading files from {REPO} and modernizing...")
        print("This may take a few minutes.\n")

        response = await client.beta.conversations.start_async(
            agent_id=agent.id,
            inputs=[
                {
                    "role": "user",
                    "content": (
                        f"Read these files from the `{REPO}` repository "
                        f"(branch: `{BRANCH}`) using the GitHub connector, "
                        f"then produce modernized versions of each:\n\n{file_list}\n\n"
                        "Return each modernized file in a ```python code fence "
                        "with a comment at the top indicating the original filename."
                    ),
                }
            ],
            timeout_ms=600_000,
        )

        text = get_response_text(response)

        # Step 5 — Save the modernized code
        code_blocks = extract_python_blocks(text)

        if not code_blocks:
            print("No Python code blocks found in the response.")
            print("\nRaw response:\n")
            print(text)
            return

        output_dir = Path("modernized")
        output_dir.mkdir(exist_ok=True)

        filenames = [Path(p).name for p in FILE_PATHS]
        for i, block in enumerate(code_blocks):
            name = filenames[i] if i < len(filenames) else f"file_{i}.py"
            output_path = output_dir / name
            output_path.write_text(block.strip() + "\n", encoding="utf-8")
            print(f"Saved: {output_path}")

        print(f"\nModernized {len(code_blocks)} file(s) in {output_dir}/")

    finally:
        # Cleanup — Delete the agent
        if agent_id:
            await client.beta.agents.delete_async(agent_id=agent_id)
            print(f"Deleted agent: {agent_id}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Summary

This cookbook demonstrated how to combine GLM's code generation with the GitHub connector to build a code modernization pipeline — the agent reads legacy files directly from a repository and returns idiomatic, modern Python.

**What you built:**
- A code modernizer that reads outdated Python files from GitHub and produces modernized versions
- An agent that pairs GLM (`zai-glm-5-2`) with the GitHub connector for server-side file access
- A pipeline that extracts code blocks from the model response and saves them locally

**Mistral features used:**
- Agents API (beta) with the `zai-glm-5-2` model
- Conversations API (beta) for server-side tool execution
- GitHub connector (`github_app`) for repository file access
- Extended timeout (`timeout_ms`) for long code generation

Try pointing the script at your own repositories to modernize real legacy code. For more on connectors, see the [connectors documentation](https://docs.mistral.ai/capabilities/connectors/).
