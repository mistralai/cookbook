# Using Connectors in Conversations Cookbook

Use MCP connectors, built-in tools, and agents in Mistral AI conversations.

> **API status:** Conversations use `client.beta.conversations`. Agents use `client.beta.agents`. These are **beta** endpoints and may change.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Reading Conversation Responses](#reading-conversation-responses)
- [Recipes](#recipes)
  - [1. Hello World — First Conversation](#1-hello-world--first-conversation)
  - [2. Conversation with Web Search](#2-conversation-with-web-search)
  - [3. Conversation with a Custom Connector](#3-conversation-with-a-custom-connector)
  - [4. Combining Multiple Tools in One Conversation](#4-combining-multiple-tools-in-one-conversation)
  - [5. Filtering Connector Tools — Include / Exclude](#5-filtering-connector-tools--include--exclude)
  - [6. Creating an Agent with Connectors](#6-creating-an-agent-with-connectors)
  - [7. OAuth-Authenticated Connectors (Gmail)](#7-oauth-authenticated-connectors-gmail)
  - [8. Full Example — Create a Connector, Chat, and Clean Up](#8-full-example--create-a-connector-chat-and-clean-up)
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

```bash
MISTRAL_API_KEY=your-mistral-api-key
```

Get your API key from the [Mistral AI dashboard](https://console.mistral.ai/).

### What you need before starting

Most recipes assume you already have:
- A working `client` (see [Recipe 1](#1-hello-world--first-conversation))
- An existing connector for the custom-connector recipes — see the [Connectors Management Cookbook](./connectors-management.md) to create one

---

## Reading Conversation Responses

Every recipe in this cookbook uses a small `display_response` helper to print the model's text output. The Conversations API returns a list of `outputs`; each output with `type == "message.output"` holds the model's reply. Content can be a plain string or a list of content chunks.

**Python:**

```python
def display_response(response) -> None:
    for output in response.outputs:
        if output.type == "message.output":
            content = output.content
            if isinstance(content, str):
                print(content)
            else:
                text = "".join(
                    chunk.text if hasattr(chunk, "text") else str(chunk)
                    for chunk in content
                )
                print(text)
```

**TypeScript:**

```typescript
function displayResponse(response: any): void {
  for (const output of response.outputs ?? []) {
    if (output.type === "message.output") {
      const content = output.content;
      if (typeof content === "string") {
        console.log(content);
      } else if (Array.isArray(content)) {
        const text = content
          .map((chunk: any) => chunk.text ?? String(chunk))
          .join("");
        console.log(text);
      }
    }
  }
}
```

All recipes below reference this helper. Copy it into your project, or inline the logic.

---

## Recipes

---

### 1. Hello World — First Conversation

**Goal:** Send your first message and read the response — no tools, no connectors.

**When to use:**
- Verifying your setup works end-to-end.
- Getting familiar with the response structure before adding connectors.

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    response = await client.beta.conversations.start_async(
        model="mistral-small-latest",
        inputs=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )
    display_response(response)


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const response = await client.beta.conversations.start({
    model: "mistral-small-latest",
    inputs: [
      { role: "user", content: "What is the capital of France?" },
    ],
  });
  displayResponse(response);
}

main();
```

**curl:**

```bash
curl -X POST "https://api.mistral.ai/v1/conversations" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-latest",
    "inputs": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

**Example of output:**

```
The capital of France is Paris.
```

**How it works:**
- `conversations.start` / `start_async` sends a stateless conversation turn to the model.
- The response contains a list of `outputs`. Each with `type == "message.output"` holds the model's reply.
- No tools or connectors are required for a basic conversation.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | Bad API key | Check `MISTRAL_API_KEY` |
| `422 Unprocessable Entity` | Invalid model name | Use a valid model like `mistral-small-latest` |

---

### 2. Conversation with Web Search

**Goal:** Give the model access to real-time web information — no custom connector needed.

**When to use:**
- The user's question requires up-to-date information (weather, news, current events).
- Quick integration without creating or managing connectors.

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    response = await client.beta.conversations.start_async(
        model="mistral-small-latest",
        inputs=[
            {
                "role": "user",
                "content": "What is the current weather in Paris? Use web search.",
            }
        ],
        tools=[
            {"type": "web_search"},
        ],
    )
    display_response(response)


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const response = await client.beta.conversations.start({
    model: "mistral-small-latest",
    inputs: [
      {
        role: "user",
        content: "What is the current weather in Paris? Use web search.",
      },
    ],
    tools: [
      { type: "web_search" },
    ],
  });
  displayResponse(response);
}

main();
```

**curl:**

```bash
curl -X POST "https://api.mistral.ai/v1/conversations" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-latest",
    "inputs": [{"role": "user", "content": "What is the current weather in Paris?"}],
    "tools": [{"type": "web_search"}]
  }'
```

**Example of output:**

```
Based on current web search results, the weather in Paris today is 8°C with partly cloudy skies...
```

**How it works:**
- `web_search` is a built-in tool type — no connector creation required.
- The model autonomously decides whether to invoke the search based on the query.
- Search results are incorporated into the model's response automatically.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `422 Unprocessable Entity` | Invalid tool type | Ensure the type is exactly `"web_search"` |

---

### 3. Conversation with a Custom Connector

**Goal:** Use a custom MCP connector in a conversation so the model can call external tools.

**When to use:**
- You've registered a connector (e.g., DeepWiki) and want the model to use its tools.
- Connecting domain-specific capabilities to the model (code search, documentation lookup, actions, etc.).

**Prereqs:**
- An existing connector — see the [Connectors Management Cookbook](./connectors-management.md) to create one.

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    response = await client.beta.conversations.start_async(
        model="mistral-small-latest",
        inputs=[
            {
                "role": "user",
                "content": "Using deepwiki, tell me about the structure of the sqlite/sqlite repository.",
            }
        ],
        tools=[
            {
                "type": "connector",
                "connector_id": "my_deepwiki",  # name or UUID
            },
        ],
    )
    display_response(response)


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const response = await client.beta.conversations.start({
    model: "mistral-small-latest",
    inputs: [
      {
        role: "user",
        content:
          "Using deepwiki, tell me about the structure of the sqlite/sqlite repository.",
      },
    ],
    tools: [
      {
        type: "connector",
        connectorId: "my_deepwiki", // name or UUID
      },
    ],
  });
  displayResponse(response);
}

main();
```

**curl:**

```bash
curl -X POST "https://api.mistral.ai/v1/conversations" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-latest",
    "inputs": [{"role": "user", "content": "Using deepwiki, tell me about the structure of the sqlite/sqlite repository."}],
    "tools": [{"type": "connector", "connector_id": "my_deepwiki"}]
  }'
```

**Example of output:**

```
The sqlite/sqlite repository is organized into several key directories:
- src/ — core SQLite source code
- ext/ — extensions
- test/ — test suite
...
```

**How it works:**
- The `connector_id` field accepts the connector's **name** or **UUID**.
- The model discovers the tools exposed by the MCP server and decides which to call.
- Tool calls and results are handled server-side — you only see the final response.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `404 Not Found` (or graceful fallback) | Connector name/ID doesn't exist — the API may return a 404 or handle gracefully | Verify with `client.beta.connectors.get` |
| `422 Unprocessable Entity` | Connector is inactive or MCP server unreachable | Check the MCP server URL |

---

### 4. Combining Multiple Tools in One Conversation

**Goal:** Give the model access to web search **and** custom connectors simultaneously.

**When to use:**
- You want the model to choose the best tool for the job from several options.
- Building a multi-capability assistant (e.g., search the web + query internal docs).

**Prereqs:**
- An existing connector.

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    response = await client.beta.conversations.start_async(
        model="mistral-small-latest",
        inputs=[
            {
                "role": "user",
                "content": "What tools do you have access to? List them briefly.",
            }
        ],
        tools=[
            {"type": "web_search"},
            {
                "type": "connector",
                "connector_id": "my_deepwiki",
            },
        ],
    )
    display_response(response)


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const response = await client.beta.conversations.start({
    model: "mistral-small-latest",
    inputs: [
      {
        role: "user",
        content: "What tools do you have access to? List them briefly.",
      },
    ],
    tools: [
      { type: "web_search" },
      {
        type: "connector",
        connectorId: "my_deepwiki",
      },
    ],
  });
  displayResponse(response);
}

main();
```

**curl:**

```bash
curl -X POST "https://api.mistral.ai/v1/conversations" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-latest",
    "inputs": [{"role": "user", "content": "What tools do you have access to? List them briefly."}],
    "tools": [
      {"type": "web_search"},
      {"type": "connector", "connector_id": "my_deepwiki"}
    ]
  }'
```

**Example of output:**

```
I have access to the following tools:
1. Web Search — search the internet for real-time information
2. read_wiki_structure — explore repository wiki structure
3. read_wiki_contents — read specific wiki pages
4. ask_question — ask questions about a repository
```

**How it works:**
- The `tools` array accepts any mix of built-in tools (`web_search`) and custom connectors.
- Each connector exposes its own set of MCP tools; the model sees all of them.
- The model decides which tool(s) to invoke based on the user's question.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `422 Unprocessable Entity` | Duplicate connector IDs in the tools array | Each connector should appear only once |

---

### 5. Filtering Connector Tools — Include / Exclude

**Goal:** Control which MCP tools from a connector are visible to the model.

**When to use:**
- A connector exposes many tools but you only need a subset.
- You want to prevent the model from calling a specific tool (e.g., a write/delete operation).
- Reducing tool noise to improve model accuracy.

**Prereqs:**
- An existing connector and knowledge of the tool names it exposes.

> **Note:** Use **either** `include` or `exclude`, not both simultaneously.

#### Excluding specific tools

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    response = await client.beta.conversations.start_async(
        model="mistral-small-latest",
        inputs=[
            {
                "role": "user",
                "content": "What tools do you have access to? List their names.",
            }
        ],
        tools=[
            {
                "type": "connector",
                "connector_id": "my_deepwiki",
                "tool_configuration": {
                    "exclude": ["read_wiki_structure"],
                },
            },
        ],
    )
    display_response(response)


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const response = await client.beta.conversations.start({
    model: "mistral-small-latest",
    inputs: [
      {
        role: "user",
        content: "What tools do you have access to? List their names.",
      },
    ],
    tools: [
      {
        type: "connector",
        connectorId: "my_deepwiki",
        toolConfiguration: {
          exclude: ["read_wiki_structure"],
        },
      },
    ],
  });
  displayResponse(response);
}

main();
```

**curl:**

```bash
curl -X POST "https://api.mistral.ai/v1/conversations" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-latest",
    "inputs": [{"role": "user", "content": "What tools do you have access to?"}],
    "tools": [{
      "type": "connector",
      "connector_id": "my_deepwiki",
      "tool_configuration": { "exclude": ["read_wiki_structure"] }
    }]
  }'
```

**Output:**

```
I have access to: read_wiki_contents, ask_question
```

#### Including only specific tools

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    response = await client.beta.conversations.start_async(
        model="mistral-small-latest",
        inputs=[
            {
                "role": "user",
                "content": "What tools do you have access to? List their names.",
            }
        ],
        tools=[
            {
                "type": "connector",
                "connector_id": "my_deepwiki",
                "tool_configuration": {
                    "include": ["ask_question"],
                },
            },
        ],
    )
    display_response(response)


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const response = await client.beta.conversations.start({
    model: "mistral-small-latest",
    inputs: [
      {
        role: "user",
        content: "What tools do you have access to? List their names.",
      },
    ],
    tools: [
      {
        type: "connector",
        connectorId: "my_deepwiki",
        toolConfiguration: {
          include: ["ask_question"],
        },
      },
    ],
  });
  displayResponse(response);
}

main();
```

**curl:**

```bash
curl -X POST "https://api.mistral.ai/v1/conversations" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-latest",
    "inputs": [{"role": "user", "content": "What tools do you have access to?"}],
    "tools": [{
      "type": "connector",
      "connector_id": "my_deepwiki",
      "tool_configuration": { "include": ["ask_question"] }
    }]
  }'
```

**Example of output:**

```
I have access to: ask_question
```

**How it works:**
- `tool_configuration.exclude` removes listed tools from the connector. All others remain available.
- `tool_configuration.include` allowlists listed tools. All others are hidden.
- Tool names must match exactly what the MCP server exposes. Ask the model to list its tools first (without any filter) to discover the exact names.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `422 Unprocessable Entity` | Tool name doesn't match any tool on the MCP server | List all tools first to get exact names |
| `400 Bad Request` | Both `include` and `exclude` provided | Use only one at a time |

---

### 6. Creating an Agent with Connectors

**Goal:** Create a persistent agent pre-configured with connectors and custom instructions, then chat with it.

**When to use:**
- You want a reusable agent that always has access to specific tools — no need to pass the `tools` array on every call.
- Building a product feature where users interact with a specialized assistant.

**Prereqs:**
- An existing connector.

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    agent_id: str | None = None
    try:
        # 1. Create the agent
        agent = await client.beta.agents.create_async(
            name="deepwiki_agent",
            description="Agent with DeepWiki access for code repository exploration",
            model="mistral-small-latest",
            instructions="You are a helpful assistant that can explore code repositories using DeepWiki. Be concise.",
            tools=[
                {
                    "type": "connector",
                    "connector_id": "my_deepwiki",
                },
            ],
        )
        agent_id = agent.id
        print(f"Created agent: {agent.name} ({agent.id})")

        # 2. Start a conversation using the agent
        response = await client.beta.conversations.start_async(
            agent_id=agent.id,
            inputs=[
                {
                    "role": "user",
                    "content": "What is the main purpose of the sqlite repository?",
                }
            ],
        )
        display_response(response)

    finally:
        # 3. Cleanup
        if agent_id:
            await client.beta.agents.delete_async(agent_id=agent_id)
            print(f"Deleted agent: {agent_id}")


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  let agentId: string | undefined;
  try {
    // 1. Create the agent
    const agent = await client.beta.agents.create({
      name: "deepwiki_agent",
      description: "Agent with DeepWiki access for code repository exploration",
      model: "mistral-small-latest",
      instructions:
        "You are a helpful assistant that can explore code repositories using DeepWiki. Be concise.",
      tools: [
        {
          type: "connector",
          connectorId: "my_deepwiki",
        },
      ],
    });
    agentId = agent.id;
    console.log(`Created agent: ${agent.name} (${agent.id})`);

    // 2. Start a conversation using the agent
    const response = await client.beta.conversations.start({
      agentId: agent.id,
      inputs: [
        {
          role: "user",
          content: "What is the main purpose of the sqlite repository?",
        },
      ],
    });
    displayResponse(response);
  } finally {
    // 3. Cleanup
    if (agentId) {
      await client.beta.agents.delete({ agentId });
      console.log(`Deleted agent: ${agentId}`);
    }
  }
}

main();
```

**curl:**

```bash
# Create agent
curl -X POST "https://api.mistral.ai/v1/agents" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "deepwiki_agent",
    "description": "Agent with DeepWiki access",
    "model": "mistral-small-latest",
    "instructions": "You are a helpful assistant that can explore code repositories using DeepWiki. Be concise.",
    "tools": [{"type": "connector", "connector_id": "my_deepwiki"}]
  }'

# Start conversation with agent (use the agent ID from the response above)
curl -X POST "https://api.mistral.ai/v1/conversations" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "<agent-id>",
    "inputs": [{"role": "user", "content": "What is the main purpose of the sqlite repository?"}]
  }'

# Delete agent when done
curl -X DELETE "https://api.mistral.ai/v1/agents/<agent-id>" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"
```

**Example of output:**

```
Created agent: deepwiki_agent (b2c3d4e5-6789-01ab-cdef-234567890abc)
SQLite is a self-contained, serverless, zero-configuration SQL database engine...
Deleted agent: b2c3d4e5-6789-01ab-cdef-234567890abc
```

**How it works:**
- Agents are persistent configurations: **model + instructions + tools**.
- When you start a conversation with `agent_id`, the model, instructions, and tools are pre-loaded — no need to pass them again.
- The agent's `tools` array uses the same format as the `tools` parameter in conversations.
- Use `agent_id` **instead of** `model` when starting a conversation — not both.
- Delete agents with `client.beta.agents.delete` / `delete_async` when they are no longer needed.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `404 Not Found` | The connector referenced in the agent's tools doesn't exist | Create the connector first |
| `422 Unprocessable Entity` | Both `model` and `agent_id` provided, or invalid tool format | Use only `agent_id` when chatting with an agent |

---

### 7. OAuth-Authenticated Connectors (Gmail)

**Goal:** Use a connector that requires OAuth2 authentication, such as Gmail.

**When to use:**
- Integrating with services that require user-level OAuth tokens (Gmail, Google Drive, Slack, etc.).
- Building features where the model accesses user-specific data.

**Prereqs:**
- A valid OAuth2 access token for the target service.

**Python:**

```python
import asyncio
from mistralai.client import Mistral

client = Mistral(api_key="your-api-key")


async def main() -> None:
    google_oauth_token = "your-google-oauth-token"

    response = await client.beta.conversations.start_async(
        model="mistral-small-latest",
        inputs=[
            {
                "role": "user",
                "content": "What's the latest email I received?",
            }
        ],
        tools=[
            {
                "type": "connector",
                "connector_id": "gmail",
                "authorization": {
                    "type": "oauth2-token",
                    "value": google_oauth_token,
                },
            },
        ],
    )
    display_response(response)


asyncio.run(main())
```

**TypeScript:**

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const googleOauthToken = "your-google-oauth-token";

  const response = await client.beta.conversations.start({
    model: "mistral-small-latest",
    inputs: [
      {
        role: "user",
        content: "What's the latest email I received?",
      },
    ],
    tools: [
      {
        type: "connector",
        connectorId: "gmail",
        authorization: {
          type: "oauth2-token",
          value: googleOauthToken,
        },
      },
    ],
  });
  displayResponse(response);
}

main();
```

**curl:**

```bash
curl -X POST "https://api.mistral.ai/v1/conversations" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-latest",
    "inputs": [{"role": "user", "content": "What is the latest email I received?"}],
    "tools": [{
      "type": "connector",
      "connector_id": "gmail",
      "authorization": {
        "type": "oauth2-token",
        "value": "<your-google-oauth-token>"
      }
    }]
  }'
```

**Example of output:**

```
Your latest email is from John Doe with the subject "Q1 Report Review" received at 2:30 PM today...
```

**How it works:**
- The `authorization` field is passed **per-tool**, not globally — different connectors can use different tokens.
- `type: "oauth2-token"` tells the backend to forward the token to the MCP server.
- The token is **not stored** by Mistral — it is used for the duration of the request only.
- Built-in connectors like `gmail` are pre-registered; you don't need to create them.

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | OAuth token is expired | Refresh the token and retry |
| `403 Forbidden` | Token doesn't have required scopes | Request the correct scopes (e.g., `gmail.readonly`) |

---

### 8. Full Example — Create a Connector, Chat, and Clean Up

**Goal:** End-to-end workflow: create a connector, use it in a conversation, then clean up.

**When to use:**
- Integration tests.
- Ephemeral connectors for one-off tasks.
- Template for production workflows.

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
            name="ephemeral_deepwiki",
            description="Temporary connector for a one-off task",
            server="https://mcp.deepwiki.com/mcp",
            visibility="private",
        )
        connector_id = str(connector.id)
        print(f"Created connector: {connector.name} ({connector.id})")

        # 2. Use it in a simple conversation
        response = await client.beta.conversations.start_async(
            model="mistral-small-latest",
            inputs=[
                {
                    "role": "user",
                    "content": "Using deepwiki, summarize the sqlite/sqlite repo in one sentence.",
                }
            ],
            tools=[
                {"type": "connector", "connector_id": "ephemeral_deepwiki"},
            ],
        )
        print("\nConversation response:")
        display_response(response)

        # 3. Use it alongside web search
        response = await client.beta.conversations.start_async(
            model="mistral-small-latest",
            inputs=[
                {
                    "role": "user",
                    "content": "Search the web for the latest SQLite release version, then use deepwiki to find where the version number is defined in the sqlite/sqlite repo.",
                }
            ],
            tools=[
                {"type": "web_search"},
                {"type": "connector", "connector_id": "ephemeral_deepwiki"},
            ],
        )
        print("\nMulti-tool response:")
        display_response(response)

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
      name: "ephemeral_deepwiki",
      description: "Temporary connector for a one-off task",
      server: "https://mcp.deepwiki.com/mcp",
      visibility: "private",
    });
    connectorId = connector.id;
    console.log(`Created connector: ${connector.name} (${connector.id})`);

    // 2. Use it in a simple conversation
    const response = await client.beta.conversations.start({
      model: "mistral-small-latest",
      inputs: [
        {
          role: "user",
          content:
            "Using deepwiki, summarize the sqlite/sqlite repo in one sentence.",
        },
      ],
      tools: [
        { type: "connector", connectorId: "ephemeral_deepwiki" },
      ],
    });
    console.log("\nConversation response:");
    displayResponse(response);

    // 3. Use it alongside web search
    const multiResponse = await client.beta.conversations.start({
      model: "mistral-small-latest",
      inputs: [
        {
          role: "user",
          content:
            "Search the web for the latest SQLite release version, then use deepwiki to find where the version number is defined in the sqlite/sqlite repo.",
        },
      ],
      tools: [
        { type: "web_search" },
        { type: "connector", connectorId: "ephemeral_deepwiki" },
      ],
    });
    console.log("\nMulti-tool response:");
    displayResponse(multiResponse);
  } finally {
    // 4. Always clean up
    if (connectorId) {
      const result = await client.beta.connectors.delete({ connectorId });
      console.log(`\nCleaned up connector: ${result.message}`);
    }
  }
}

main();
```

**Output:**

```
Created connector: ephemeral_deepwiki (c3d4e5f6-...)

Conversation response:
SQLite is a self-contained, serverless SQL database engine used worldwide.

Multi-tool response:
The latest SQLite release is 3.45.1. The version number is defined in src/sqlite.h...

Cleaned up connector: Connector deleted successfully
```

**How it works:**
- The `try/finally` pattern ensures the connector is always deleted, even if the conversation fails.
- This recipe combines connector CRUD (from the [Connectors Management Cookbook](./connectors-management.md)) with conversation usage.
- The second conversation demonstrates the model intelligently routing between web search and the custom connector within a single request.
