# Using Connectors in Chat Completions Cookbook

Use Connectors, built-in tools, and agents with the Chat Completions API (`/v1/chat/completions`) and Agent Completions API (`/v1/agents/completions`).

> **SDK support:** The `mistralai` SDK supports connector-style tools in `chat.complete()`. Note that responses use `messages` (array) instead of `message` (object) when tools are invoked.

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
# TypeScript / Node.js
npm install @mistralai/mistralai
```

### Required environment variables

To complete this cookbook, you'll need a Mistral API key. In [Studio](https://console.mistral.ai), navigate to the [API keys section](https://console.mistral.ai/home?profile_dialog=api-keys) and create a new API key.

Create a `.env` at the root of your project and add your Mistral API key:

```
MISTRAL_API_KEY=your-mistral-api-key
```

### What you need before starting

Most recipes assume:
- A valid `MISTRAL_API_KEY`
- An existing connector for custom-connector recipes — see [Build a Database Advisor Agent](./01-build-a-database-advisor-agent.ipynb) for a full connector lifecycle example, or create one in [Studio](https://console.mistral.ai)

---

## Conversations vs Completions — When to Use Which

Mistral offers two APIs for interacting with models:

| Feature | Conversations API | Chat Completions API |
|---------|-------------------|---------------------|
| Endpoint | `/v1/conversations` | `/v1/chat/completions` |
| SDK support | `client.beta.conversations` | `client.chat.complete()` |
| Response format | `outputs[]` with `type: "message.output"` | `choices[]` with `messages` (array) when tools are used |
| Multi-turn state | Stateless (send full history each call) | Stateless (send full history each call) |
| Tool execution | Server-side (automatic) | Server-side with `multi_completion` format |
| Best for | New integrations, agent workflows | OpenAI-compatible apps, existing chat implementations |

**Use Chat Completions when:**
- You're migrating from OpenAI and want a familiar response format
- Your existing code uses the `choices[].message` structure
- You need compatibility with OpenAI-style SDKs

**Use Conversations when:**
- You're building new integrations from scratch
- You want the recommended, newest API surface
- You're using the SDK's beta features

---

## Reading Completion Responses

Chat completions with tools return responses in a `multi_completion` format where each choice contains a `messages` array instead of a single `message`. The helper below handles both formats.

**Python (SDK):**

```python
def display_response(response) -> None:
    """Display text content from SDK chat completion response.

    When using connector tools, responses use `messages` array instead of `message`.
    """
    for choice in response.choices:
        # Handle multi_completion format (messages array) - used with connector tools
        if hasattr(choice, 'messages') and choice.messages:
            for message in choice.messages:
                content = message.content
                if content:
                    if isinstance(content, str):
                        print(content[:500] if len(content) > 500 else content)
                    elif isinstance(content, list):
                        for chunk in content:
                            if hasattr(chunk, 'type'):
                                if chunk.type == "text":
                                    print(getattr(chunk, 'text', ''))
                                elif chunk.type == "image_url":
                                    print(f"[Image: {getattr(chunk, 'image_url', '')[:80]}...]")
                tool_calls = getattr(message, 'tool_calls', None)
                if tool_calls:
                    print(f"Tool calls: {len(tool_calls)}")
                    for tc in tool_calls:
                        func = tc.function
                        print(f"  - {func.name}: {func.arguments}")
        # Handle standard completion format (single message)
        elif hasattr(choice, 'message') and choice.message:
            message = choice.message
            content = message.content
            if content:
                print(content[:500] if len(content) > 500 else content)
            tool_calls = getattr(message, 'tool_calls', None)
            if tool_calls:
                print(f"Tool calls: {len(tool_calls)}")
                for tc in tool_calls:
                    func = tc.function
                    print(f"  - {func.name}: {func.arguments}")
```

**TypeScript:**

```typescript
function displayResponse(data: any): void {
  for (const choice of data.choices ?? []) {
    // Handle multi_completion format (messages array)
    const messages = choice.messages ?? [];
    if (messages.length > 0) {
      for (const message of messages) {
        const content = message.content;
        if (content) {
          if (typeof content === "string") {
            console.log(content.length > 500 ? content.slice(0, 500) : content);
          } else if (Array.isArray(content)) {
            for (const chunk of content) {
              if (chunk.type === "text") {
                console.log(chunk.text ?? "");
              } else if (chunk.type === "image_url") {
                console.log(`[Image: ${(chunk.image_url ?? "").slice(0, 80)}...]`);
              }
            }
          }
        }
        const toolCalls = message.tool_calls ?? [];
        if (toolCalls.length > 0) {
          console.log(`Tool calls: ${toolCalls.length}`);
          for (const tc of toolCalls) {
            const func = tc.function ?? {};
            console.log(`  - ${func.name}: ${func.arguments}`);
          }
        }
      }
    // Handle standard completion format (single message)
    } else {
      const message = choice.message ?? {};
      const content = message.content;
      if (content) {
        console.log(typeof content === "string" && content.length > 500 ? content.slice(0, 500) : content);
      }
      const toolCalls = message.tool_calls ?? [];
      if (toolCalls.length > 0) {
        console.log(`Tool calls: ${toolCalls.length}`);
        for (const tc of toolCalls) {
          const func = tc.function ?? {};
          console.log(`  - ${func.name}: ${func.arguments}`);
        }
      }
    }
  }
}
```

All recipes below reference this helper. Copy it into your project, or inline the logic.

---

## Recipes

---

### 1. Hello World — Basic Chat Completion

**Goal:** Send your first chat completion request — no tools, no connectors.

**When to use:**
- Verifying your SDK setup works end-to-end
- Getting familiar with the response structure before adding connectors

**Python:**

```python
import asyncio
from mistralai.client import Mistral

API_KEY = "your-api-key"


async def main() -> None:
    client = Mistral(api_key=API_KEY)

    response = await client.chat.complete_async(
        model="mistral-small-latest",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )

    # Standard completion uses choice.message
    print(response.choices[0].message.content)


asyncio.run(main())
```

**TypeScript:**

```typescript
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const response = await client.chat.complete({
    model: "mistral-small-latest",
    messages: [
      { role: "user", content: "What is the capital of France?" },
    ],
  });

  console.log(response.choices[0].message.content);
}

main();
```

**curl:**

```bash
curl -X POST "https://api.mistral.ai/v1/chat/completions" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-latest",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

**Example of output:**

```
The capital of France is Paris.
```

**How it works:**
- `/v1/chat/completions` is the standard OpenAI-compatible chat endpoint
- Response contains `choices[]` with each choice having a `message` object
- No tools or connectors are required for a basic completion

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | Bad API key | Check `MISTRAL_API_KEY` |
| `422 Unprocessable Entity` | Invalid model name | Use a valid model like `mistral-small-latest` |

---

### 2. Completion with Image Generation

**Goal:** Use the built-in `image_generation` tool in a chat completion.

**When to use:**
- Generating images based on user prompts
- Quick integration without custom connectors

**Python:**

```python
import asyncio
from mistralai.client import Mistral

API_KEY = "your-api-key"


async def main() -> None:
    client = Mistral(api_key=API_KEY)

    response = await client.chat.complete_async(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": "Generate an image of a sunset over the ocean.",
            }
        ],
        tools=[
            {"type": "image_generation"},
        ],
    )

    # With tools, use choice.messages (array) instead of choice.message
    display_response(response)


asyncio.run(main())
```

**TypeScript:**

```typescript
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const response = await client.chat.complete({
    model: "mistral-small-latest",
    messages: [
      {
        role: "user",
        content: "Generate an image of a sunset over the ocean.",
      },
    ],
    tools: [
      { type: "image_generation" },
    ],
  });

  displayResponse(response);
}

main();
```

**curl:**

```bash
curl -X POST "https://api.mistral.ai/v1/chat/completions" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-latest",
    "messages": [{"role": "user", "content": "Generate an image of a sunset over the ocean."}],
    "tools": [{"type": "image_generation"}]
  }'
```

**Example of output:**

```
[Image: https://files.mistral.ai/generated/abc123...]
Here's a beautiful sunset over the ocean as requested.
```

**How it works:**
- `image_generation` is a built-in tool type — no connector creation required
- The model decides to invoke the tool based on the user's request
- Generated images are returned as URLs in the response content
- When tools are invoked, responses use `choice.messages` (array) instead of `choice.message`

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `422 Unprocessable Entity` | Invalid tool type | Ensure the type is exactly `"image_generation"` |

---

### 3. Completion with a Custom Connector

**Goal:** Use a Connector in a chat completion so the model can call external tools.

**When to use:**
- You've registered a connector (e.g., DeepWiki) and want the model to use its tools
- Connecting domain-specific capabilities to the model in a chat completion context

**Prereqs:**
- An existing connector — see [Build a Database Advisor Agent](./01-build-a-database-advisor-agent.ipynb) for a full connector lifecycle example, or create one in [Studio](https://console.mistral.ai)

**Python:**

```python
import asyncio
from mistralai.client import Mistral

API_KEY = "your-api-key"


async def main() -> None:
    client = Mistral(api_key=API_KEY)

    response = await client.chat.complete_async(
        model="mistral-small-latest",
        messages=[
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

    # With connector tools, use choice.messages (array)
    display_response(response)


asyncio.run(main())
```

**TypeScript:**

```typescript
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const response = await client.chat.complete({
    model: "mistral-small-latest",
    messages: [
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
curl -X POST "https://api.mistral.ai/v1/chat/completions" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-latest",
    "messages": [{"role": "user", "content": "Using deepwiki, tell me about the structure of the sqlite/sqlite repository."}],
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
- The `connector_id` field accepts the connector's **name** or **UUID**
- The model discovers the tools exposed by the MCP server and decides which to call
- Tool calls and results are handled server-side — you only see the final response
- Responses use `choice.messages` (array) when tools are invoked

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `404 Not Found` | Connector name/ID doesn't exist | Verify with connector list/get API |
| `422 Unprocessable Entity` | Connector is inactive or MCP server unreachable | Check the MCP server URL |

---

### 4. Combining Multiple Tools

**Goal:** Give the model access to built-in tools **and** custom connectors simultaneously in a chat completion.

**When to use:**
- You want the model to choose the best tool for the job from several options
- Building a multi-capability assistant

**Prereqs:**
- An existing connector

**Python:**

```python
import asyncio
from mistralai.client import Mistral

API_KEY = "your-api-key"


async def main() -> None:
    client = Mistral(api_key=API_KEY)

    response = await client.chat.complete_async(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": "What tools do you have access to? List them briefly.",
            }
        ],
        tools=[
            {"type": "image_generation"},
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
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const response = await client.chat.complete({
    model: "mistral-small-latest",
    messages: [
      {
        role: "user",
        content: "What tools do you have access to? List them briefly.",
      },
    ],
    tools: [
      { type: "image_generation" },
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
curl -X POST "https://api.mistral.ai/v1/chat/completions" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-latest",
    "messages": [{"role": "user", "content": "What tools do you have access to? List them briefly."}],
    "tools": [
      {"type": "image_generation"},
      {"type": "connector", "connector_id": "my_deepwiki"}
    ]
  }'
```

**Example of output:**

```
I have access to the following tools:
1. Image Generation — create images from text descriptions
2. read_wiki_structure — explore repository wiki structure
3. read_wiki_contents — read specific wiki pages
4. ask_question — ask questions about a repository
```

**How it works:**
- The `tools` array accepts any mix of built-in tools (`image_generation`) and custom connectors
- Each connector exposes its own set of MCP tools; the model sees all of them
- The model decides which tool(s) to invoke based on the user's question

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `422 Unprocessable Entity` | Duplicate connector IDs in the tools array | Each connector should appear only once |

---

### 5. Creating an Agent with Connectors

**Goal:** Create a persistent agent pre-configured with connectors and custom instructions for use with the Agent Completions API.

**When to use:**
- You want a reusable agent that always has access to specific tools
- Building a product feature where users interact with a specialized assistant
- Simplifying API calls by pre-configuring model, instructions, and tools

**Prereqs:**
- An existing connector

**Python:**

```python
import asyncio
from mistralai.client import Mistral

API_KEY = "your-api-key"


async def main() -> None:
    client = Mistral(api_key=API_KEY)
    agent_id: str | None = None

    try:
        # Create the agent
        agent = await client.beta.agents.create_async(
            name="deepwiki_completion_agent",
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
        agent_id = str(agent.id)
        print(f"Created agent: {agent.name} ({agent_id})")

    finally:
        # Clean up
        if agent_id:
            await client.beta.agents.delete_async(agent_id=agent_id)
            print(f"Deleted agent: {agent_id}")


asyncio.run(main())
```

**TypeScript:**

```typescript
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  let agentId: string | undefined;

  try {
    // Create the agent
    const agent = await client.beta.agents.create({
      name: "deepwiki_completion_agent",
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
    console.log(`Created agent: ${agent.name} (${agentId})`);
  } finally {
    // Clean up
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
    "name": "deepwiki_completion_agent",
    "description": "Agent with DeepWiki access",
    "model": "mistral-small-latest",
    "instructions": "You are a helpful assistant that can explore code repositories using DeepWiki. Be concise.",
    "tools": [{"type": "connector", "connector_id": "my_deepwiki"}]
  }'

# Delete agent when done (use the agent ID from the response above)
curl -X DELETE "https://api.mistral.ai/v1/agents/<agent-id>" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}"
```

**Example of output:**

```
Created agent: deepwiki_completion_agent (b2c3d4e5-6789-01ab-cdef-234567890abc)
Deleted agent: b2c3d4e5-6789-01ab-cdef-234567890abc
```

**How it works:**
- Agents are persistent configurations: **model + instructions + tools**
- Once created, use the agent's ID with `/v1/agents/completions` to chat
- The agent's `tools` array uses the same format as the `tools` parameter in completions
- Delete agents when they are no longer needed

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `404 Not Found` | The connector referenced in the agent's tools doesn't exist | Create the connector first |
| `409 Conflict` | An agent with this name already exists | Choose a different name or delete the existing agent |

---

### 6. Agent Completions

**Goal:** Chat with a pre-configured agent using the Agent Completions API.

**When to use:**
- You have an existing agent with tools configured
- You want to avoid passing model, instructions, and tools on every request
- Building conversational flows with a specialized assistant

**Prereqs:**
- An existing agent ID (see [Recipe 5](#5-creating-an-agent-with-connectors))

**Python:**

```python
import asyncio
from mistralai.client import Mistral

API_KEY = "your-api-key"


async def main() -> None:
    client = Mistral(api_key=API_KEY)
    agent_id = "your-agent-id"  # From agent creation

    response = await client.agents.complete_async(
        agent_id=agent_id,
        messages=[
            {
                "role": "user",
                "content": "What is the main purpose of the sqlite repository?",
            }
        ],
    )

    display_response(response)


asyncio.run(main())
```

**TypeScript:**

```typescript
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const agentId = "your-agent-id"; // From agent creation

  const response = await client.agents.complete({
    agentId,
    messages: [
      {
        role: "user",
        content: "What is the main purpose of the sqlite repository?",
      },
    ],
  });

  displayResponse(response);
}

main();
```

**curl:**

```bash
curl -X POST "https://api.mistral.ai/v1/agents/completions" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "<agent-id>",
    "messages": [{"role": "user", "content": "What is the main purpose of the sqlite repository?"}]
  }'
```

**Example of output:**

```
SQLite is a self-contained, serverless, zero-configuration SQL database engine. It is the most widely deployed database in the world, embedded in countless applications including web browsers, mobile phones, and operating systems.
```

**How it works:**
- `/v1/agents/completions` uses the agent's pre-configured model, instructions, and tools
- You only need to provide `agent_id` and `messages` — no need to repeat configuration
- The response format is the same as `/v1/chat/completions`
- For multi-turn conversations, include the full message history in the `messages` array

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `404 Not Found` | Invalid agent ID | The agent may have been deleted |
| `422 Unprocessable Entity` | Missing `agent_id` or invalid message format | Ensure `agent_id` is provided |

---

### 7. OAuth-Authenticated Connectors (Gmail)

**Goal:** Use a connector that requires OAuth2 authentication in a chat completion.

**When to use:**
- Integrating with services that require user-level OAuth tokens (Gmail, Google Drive, Slack, etc.)
- Building features where the model accesses user-specific data

**Prereqs:**
- A valid OAuth2 access token for the target service

**Python:**

```python
import asyncio
from mistralai.client import Mistral

API_KEY = "your-api-key"


async def main() -> None:
    client = Mistral(api_key=API_KEY)
    google_oauth_token = "your-google-oauth-token"

    response = await client.chat.complete_async(
        model="mistral-small-latest",
        messages=[
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
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  const googleOauthToken = "your-google-oauth-token";

  const response = await client.chat.complete({
    model: "mistral-small-latest",
    messages: [
      {
        role: "user",
        content: "What's the latest email I received?",
      },
    ],
    tools: [
      {
        type: "connector",
        connector_id: "gmail",
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
curl -X POST "https://api.mistral.ai/v1/chat/completions" \
  -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-latest",
    "messages": [{"role": "user", "content": "What is the latest email I received?"}],
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
- The `authorization` field is passed **per-tool**, not globally — different connectors can use different tokens
- `type: "oauth2-token"` tells the backend to forward the token to the MCP server
- The token is **not stored** by Mistral — it is used for the duration of the request only
- Built-in connectors like `gmail` are pre-registered; you don't need to create them

**Common errors & fixes:**

| Error | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | OAuth token is expired | Refresh the token and retry |
| `403 Forbidden` | Token doesn't have required scopes | Request the correct scopes (e.g., `gmail.readonly`) |

---

### 8. Full Example — Create, Complete, and Clean Up

**Goal:** End-to-end workflow: create a connector, use it in chat completions and with an agent, then clean up.

**When to use:**
- Integration tests
- Ephemeral connectors for one-off tasks
- Template for production workflows

**Python:**

```python
import asyncio
from mistralai.client import Mistral

API_KEY = "your-api-key"


async def main() -> None:
    client = Mistral(api_key=API_KEY)
    connector_id: str | None = None
    agent_id: str | None = None

    try:
        # 1. Create a connector
        connector = await client.beta.connectors.create_async(
            name="completions_deepwiki",
            description="DeepWiki connector for completion testing",
            server="https://mcp.deepwiki.com/mcp",
            visibility="private",
        )
        connector_id = str(connector.id)
        print(f"Created connector: {connector.name} ({connector_id})")

        # 2. Use it in a chat completion
        response = await client.chat.complete_async(
            model="mistral-small-latest",
            messages=[
                {
                    "role": "user",
                    "content": "Using deepwiki, summarize the sqlite/sqlite repo in one sentence.",
                }
            ],
            tools=[
                {"type": "connector", "connector_id": "completions_deepwiki"},
            ],
        )
        print("\nChat completion response:")
        display_response(response)

        # 3. Create an agent with the connector
        agent = await client.beta.agents.create_async(
            name="completions_test_agent",
            description="Test agent for completion cookbook",
            model="mistral-small-latest",
            instructions="You are a helpful assistant. Be concise.",
            tools=[
                {"type": "connector", "connector_id": connector_id},
            ],
        )
        agent_id = str(agent.id)
        print(f"\nCreated agent: {agent.name} ({agent_id})")

        # 4. Use agent completions
        response = await client.agents.complete_async(
            agent_id=agent_id,
            messages=[
                {
                    "role": "user",
                    "content": "What programming language is SQLite written in?",
                }
            ],
        )
        print("\nAgent completion response:")
        display_response(response)

        print("\n" + "=" * 60)
        print("  SUCCESS")
        print("=" * 60)

    finally:
        # Clean up
        print("\nCleaning up...")
        if agent_id:
            try:
                await client.beta.agents.delete_async(agent_id=agent_id)
                print(f"Deleted agent: {agent_id}")
            except Exception:
                pass

        if connector_id:
            try:
                await client.beta.connectors.delete_async(
                    connector_id=connector_id,
                )
                print(f"Deleted connector: {connector_id}")
            except Exception:
                pass


asyncio.run(main())
```

**TypeScript:**

```typescript
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: "your-api-key" });

async function main(): Promise<void> {
  let connectorId: string | undefined;
  let agentId: string | undefined;

  try {
    // 1. Create a connector
    const connector = await client.beta.connectors.create({
      name: "completions_deepwiki",
      description: "DeepWiki connector for completion testing",
      server: "https://mcp.deepwiki.com/mcp",
      visibility: "private",
    });
    connectorId = connector.id;
    console.log(`Created connector: ${connector.name} (${connectorId})`);

    // 2. Use it in a chat completion
    let response = await client.chat.complete({
      model: "mistral-small-latest",
      messages: [
        {
          role: "user",
          content: "Using deepwiki, summarize the sqlite/sqlite repo in one sentence.",
        },
      ],
      tools: [
        { type: "connector", connectorId: "completions_deepwiki" },
      ],
    });
    console.log("\nChat completion response:");
    displayResponse(response);

    // 3. Create an agent with the connector
    const agent = await client.beta.agents.create({
      name: "completions_test_agent",
      description: "Test agent for completion cookbook",
      model: "mistral-small-latest",
      instructions: "You are a helpful assistant. Be concise.",
      tools: [
        { type: "connector", connectorId },
      ],
    });
    agentId = agent.id;
    console.log(`\nCreated agent: ${agent.name} (${agentId})`);

    // 4. Use agent completions
    response = await client.agents.complete({
      agentId,
      messages: [
        {
          role: "user",
          content: "What programming language is SQLite written in?",
        },
      ],
    });
    console.log("\nAgent completion response:");
    displayResponse(response);

    console.log("\n" + "=".repeat(60));
    console.log("  SUCCESS");
    console.log("=".repeat(60));
  } finally {
    // Clean up
    console.log("\nCleaning up...");
    if (agentId) {
      try {
        await client.beta.agents.delete({ agentId });
        console.log(`Deleted agent: ${agentId}`);
      } catch {
        // ignore
      }
    }
    if (connectorId) {
      try {
        await client.beta.connectors.delete({ connectorId });
        console.log(`Deleted connector: ${connectorId}`);
      } catch {
        // ignore
      }
    }
  }
}

main();
```

**Output:**

```
Created connector: completions_deepwiki (c3d4e5f6-...)

Chat completion response:
SQLite is a self-contained, serverless SQL database engine used worldwide.

Created agent: completions_test_agent (d4e5f6a7-...)

Agent completion response:
SQLite is primarily written in C.

============================================================
  SUCCESS
============================================================

Cleaning up...
Deleted agent: d4e5f6a7-...
Deleted connector: c3d4e5f6-...
```

**How it works:**
- The `try/finally` pattern ensures resources are always cleaned up, even if requests fail
- This recipe demonstrates the full workflow: connector creation → chat completion → agent creation → agent completion
- Both `/v1/chat/completions` and `/v1/agents/completions` support the same tool formats

---

## Python / TypeScript Naming Conventions

| Concept | Python SDK | TypeScript SDK |
|---------|-----------|----------------|
| Connector ID | `connector_id` | `connectorId` |
| Agent ID | `agent_id` | `agentId` |
| Tool type | `type` | `type` |
| OAuth authorization | `authorization.type`, `authorization.value` | `authorization.type`, `authorization.value` |

---

## Troubleshooting

### "Connector not found" error
- Verify the connector exists by listing connectors or getting by name/ID
- Check that the connector name/ID is spelled correctly
- Ensure the connector was created in the same workspace

### "Tool calls not appearing in response"
- The model decides whether to call tools based on the user's message
- Try being more explicit in your prompt (e.g., "Use deepwiki to...")
- Ensure the tool type is valid (`connector`, `image_generation`, etc.)

### "Timeout errors"
- MCP servers may take time to respond — increase the timeout
- Check if the MCP server URL is reachable
- Try with a simpler query first

### "OAuth token expired"
- OAuth tokens have limited lifetimes
- Implement token refresh logic in your application
- Check the token's expiration before making requests

### "AttributeError: 'Unset' object has no attribute 'content'"
- When using connector tools, responses use `choice.messages` (array) instead of `choice.message`
- Use the `display_response` helper function or check for both formats
- Access content via `response.choices[0].messages[0].content`

---

## Error Codes Reference

| HTTP Status | Error | Common Causes |
|-------------|-------|---------------|
| `400` | Bad Request | Invalid JSON, missing required fields |
| `401` | Unauthorized | Invalid or missing API key, expired OAuth token |
| `403` | Forbidden | Insufficient permissions, invalid OAuth scopes |
| `404` | Not Found | Connector/agent doesn't exist, wrong ID |
| `409` | Conflict | Resource with same name already exists |
| `422` | Unprocessable Entity | Invalid model name, invalid tool format, unreachable MCP server |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server-side issue, retry with backoff |
| `502` | Bad Gateway | MCP server unreachable or returned an error |
| `504` | Gateway Timeout | MCP server took too long to respond |

---

## Summary

This cookbook covered eight recipes for using Connectors, built-in tools, and agents with the Chat Completions and Agent Completions APIs — from a basic completion to OAuth-authenticated Connectors and a full create-complete-cleanup lifecycle example.

**What this cookbook covers:**
- Basic chat completion
- Completion with image generation
- Completion with a custom Connector
- Combining multiple tools in one completion
- Creating an agent with Connectors for use with completions
- Agent completions
- OAuth-authenticated Connectors (Gmail)
- Full lifecycle: create a Connector, complete, and clean up

**Mistral features used:**
- Chat Completions API
- Agent Completions API
- Agents API (beta)
- Connectors (beta)
- Image generation built-in tool

**Other services:**
- [DeepWiki](https://deepwiki.com) — MCP server for GitHub repository exploration
- Gmail — OAuth2-authenticated Connector

View your Connectors in [Studio](https://console.mistral.ai/build/connectors).
