# Connectors Tool Calling Cookbook

Call individual tools on an MCP connector directly, without going through a conversation.

> **API status:** Tool calling uses `client.beta.connectors.call_tool`. This is a **beta** endpoint and may change.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [When to Use Direct Tool Calling](#when-to-use-direct-tool-calling)
- [Recipes](#recipes)
  - [1. Initialize the Client](#1-initialize-the-client)
  - [2. Create a Connector](#2-create-a-connector)
  - [3. Call a Tool on a Connector](#3-call-a-tool-on-a-connector)
  - [4. Full Example — Create, Call a Tool, and Clean Up](#4-full-example--create-call-a-tool-and-clean-up)
- [Python / TypeScript Naming Conventions](#python--typescript-naming-conventions)
- [Troubleshooting](#troubleshooting)
- [Error Codes Reference](#error-codes-reference)

---

## Prerequisites

### Install

```bash
# Python
pip install mistralai
# or with uv
uv add mistralai
```

```bash
# TypeScript
pnpm add @mistralai/mistralai
```

### Required environment variables

Create a `.env` file at your project root:

```bash
MISTRAL_API_KEY=your-mistral-api-key
```

Get your API key from the [Mistral AI dashboard](https://console.mistral.ai/).

### What you need before starting

- A working `client` (see [Recipe 1](#1-initialize-the-client))
- An existing connector — see the [Connectors Management Cookbook](./01-connectors-management.md) to create one

---

## When to Use Direct Tool Calling

Direct tool calling (`call_tool`) lets you invoke a single MCP tool on a connector without starting a conversation. This is useful when:

- **You already know which tool to call** — no need for the model to decide.
- **You want raw tool output** — e.g., fetching structured data for downstream processing.
- **Building pipelines** — chaining tool calls programmatically rather than relying on the model's orchestration.
- **Debugging** — verifying that a connector's tools work correctly before using them in conversations.

For scenarios where the model should autonomously pick which tools to call, use the [Conversations API](./02-connectors-in-conversations-and-agents.md) instead.

---

## Recipes

---

### 1. Initialize the Client

**Prereqs:** `MISTRAL_API_KEY` set in your environment or `.env` file.

**Python:**

```python
import os
from mistralai.client import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: process.env.MISTRAL_API_KEY });
```

**curl:**

```bash
# All subsequent curl examples assume these variables are set:
export MISTRAL_API_KEY="your-api-key"
export BASE_URL="https://api.mistral.ai"
```

**Output:** None — the client is ready to use.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | Invalid API key | Regenerate your key on the Mistral dashboard |

---

### 2. Create a Connector

**Goal:** Register a connector so its tools can be called directly.

**When to use:**
- First step before calling tools — you need a connector to target.
- If you already have a connector, skip to [Recipe 3](#3-call-a-tool-on-a-connector).

**Prereqs:** An initialized `client` ([Recipe 1](#1-initialize-the-client)).

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    connector = await client.beta.connectors.create_async(
        name="my_deepwiki",
        description="DeepWiki MCP connector for code repository exploration",
        server="https://mcp.deepwiki.com/mcp",
        visibility="private",
    )
    print(f"ID:   {connector.id}")
    print(f"Name: {connector.name}")


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const connector = await client.beta.connectors.create({
    name: "my_deepwiki",
    description: "DeepWiki MCP connector for code repository exploration",
    server: "https://mcp.deepwiki.com/mcp",
    visibility: "private",
  });
  console.log(`ID:   ${connector.id}`);
  console.log(`Name: ${connector.name}`);
}

main();
```

**curl:**

```bash
curl -X POST "${BASE_URL}/v1/connectors" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_deepwiki",
    "description": "DeepWiki MCP connector for code repository exploration",
    "server": "https://mcp.deepwiki.com/mcp",
    "visibility": "private"
  }'
```

**Output:**

```
ID:   a1b2c3d4-5678-90ab-cdef-1234567890ab
Name: my_deepwiki
```

**How it works:**
- See the [Connectors Management Cookbook](./01-connectors-management.md) for full details on creating, updating, and deleting connectors.
- The connector must be created before you can call its tools.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `409 Conflict` | A connector with this name already exists | Choose a different name or delete the existing one first |

---

### 3. Call a Tool on a Connector

**Goal:** Invoke a specific tool exposed by an MCP connector and get the raw result.

**When to use:**
- You know exactly which tool to call and what arguments to pass.
- You want the raw tool output without model interpretation.
- Building automated pipelines that chain tool calls.
- Debugging or verifying that a connector's tools respond correctly.

**Prereqs:**
- An existing connector — by name or ID.
- Knowledge of the tool name and its expected arguments. You can discover available tools by asking the model to list them in a [conversation](./02-connectors-in-conversations-and-agents.md#4-combining-multiple-tools-in-one-conversation), or by checking the connector's `tools` field via `client.beta.connectors.get`.

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    result = await client.beta.connectors.call_tool_async(
        connector_id_or_name="my_deepwiki",
        tool_name="read_wiki_structure",
        arguments={"repoName": "sqlite/sqlite"},
    )
    print(f"Tool output:\n{result.content}")


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const result = await client.beta.connectors.callTool({
    connectorIdOrName: "my_deepwiki",
    toolName: "read_wiki_structure",
    arguments: { repoName: "sqlite/sqlite" },
  });
  console.log(`Tool output:\n${result.content}`);
}

main();
```

**curl:**

```bash
curl -X POST "${BASE_URL}/v1/connectors/my_deepwiki/call_tool" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "read_wiki_structure",
    "arguments": {"repoName": "sqlite/sqlite"}
  }'
```

**Example of output:**

```
Tool output:
# sqlite/sqlite Wiki Structure

- Overview
  - Architecture
  - Build System
  - SQL Language
- Core Components
  - Parser
  - Code Generator
  - Virtual Machine
  ...
```

**How it works:**
- `call_tool` / `call_tool_async` sends a direct request to the MCP server through the connector, bypassing the conversation model entirely.
- `connector_id_or_name` accepts either the connector's **name** or **UUID**.
- `tool_name` must match exactly one of the tools the MCP server exposes.
- `arguments` is a dictionary/object of key-value pairs expected by the tool.
- The response contains a `content` field with the raw output from the MCP server.

### 4. Full Example — Create, Call a Tool, and Clean Up

**Goal:** End-to-end workflow: create a connector, call one of its tools directly, then clean up.

**When to use:**
- Integration tests for tool calling.
- Ephemeral connectors for one-off data fetching.
- Template for programmatic tool-calling pipelines.

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    connector_id: str | None = None

    try:
        # 1. Create a connector
        connector = await client.beta.connectors.create_async(
            name="toolcall_deepwiki",
            description="Temporary connector for direct tool calling",
            server="https://mcp.deepwiki.com/mcp",
            visibility="private",
        )
        connector_id = str(connector.id)
        print(f"Created connector: {connector.name} ({connector.id})")

        # 2. Call a tool directly
        result = await client.beta.connectors.call_tool_async(
            connector_id_or_name=str(connector.id),
            tool_name="read_wiki_structure",
            arguments={"repoName": "sqlite/sqlite"},
        )
        print(f"\nTool 'read_wiki_structure' output:\n{result.content[:500]}")

        # 3. Call another tool
        result = await client.beta.connectors.call_tool_async(
            connector_id_or_name=str(connector.id),
            tool_name="ask_question",
            arguments={
                "repoName": "sqlite/sqlite",
                "question": "What is the purpose of the VDBE?",
            },
        )
        print(f"\nTool 'ask_question' output:\n{result.content[:500]}")

    finally:
        # 4. Always clean up
        if connector_id:
            result = await client.beta.connectors.delete_async(
                connector_id=connector_id,
            )
            print(f"\nCleaned up connector: {result.message}")


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  let connectorId: string | undefined;

  try {
    // 1. Create a connector
    const connector = await client.beta.connectors.create({
      name: "toolcall_deepwiki",
      description: "Temporary connector for direct tool calling",
      server: "https://mcp.deepwiki.com/mcp",
      visibility: "private",
    });
    connectorId = connector.id;
    console.log(`Created connector: ${connector.name} (${connector.id})`);

    // 2. Call a tool directly
    let result = await client.beta.connectors.callTool({
      connectorIdOrName: connector.id,
      toolName: "read_wiki_structure",
      arguments: { repoName: "sqlite/sqlite" },
    });
    console.log(`\nTool 'read_wiki_structure' output:\n${result.content?.substring(0, 500)}`);

    // 3. Call another tool
    result = await client.beta.connectors.callTool({
      connectorIdOrName: connector.id,
      toolName: "ask_question",
      arguments: {
        repoName: "sqlite/sqlite",
        question: "What is the purpose of the VDBE?",
      },
    });
    console.log(`\nTool 'ask_question' output:\n${result.content?.substring(0, 500)}`);
  } finally {
    // 4. Always clean up
    if (connectorId) {
      const deleteResult = await client.beta.connectors.delete({ connectorId });
      console.log(`\nCleaned up connector: ${deleteResult.message}`);
    }
  }
}

main();
```

**curl:**

```bash
# 1. Create a connector
curl -X POST "${BASE_URL}/v1/connectors" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "toolcall_deepwiki",
    "description": "Temporary connector for direct tool calling",
    "server": "https://mcp.deepwiki.com/mcp",
    "visibility": "private"
  }'

# 2. Call a tool (use the connector name or ID)
curl -X POST "${BASE_URL}/v1/connectors/toolcall_deepwiki/call_tool" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "read_wiki_structure",
    "arguments": {"repoName": "sqlite/sqlite"}
  }'

# 3. Call another tool
curl -X POST "${BASE_URL}/v1/connectors/toolcall_deepwiki/call_tool" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "ask_question",
    "arguments": {"repoName": "sqlite/sqlite", "question": "What is the purpose of the VDBE?"}
  }'

# 4. Clean up (use the connector UUID from the create response)
curl -X DELETE "${BASE_URL}/v1/connectors/${CONNECTOR_ID}" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"
```
