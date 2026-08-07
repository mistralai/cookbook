# Build a Quickstart Generator with Connectors and Web Search (TypeScript)

You want to learn Polars, the fast DataFrame library for Python. You could read the docs, skim blog posts, and piece together a quickstart yourself — or you could let Mistral do it for you, with access to the right tools.

This script sends the **same prompt** five times, each time with a different tool configuration, so you can see how output quality improves as you give the model better sources:

| Step | Tools | What the model can access |
|---|---|---|
| 1 | None | Training data only |
| 2 | Web search | Blog posts, Stack Overflow, release notes |
| 3 | Context7 connector | Official Polars documentation |
| 4 | Both | Docs + web — the model picks the best source per sub-topic |
| 5 | Filtered connector | A single doc-retrieval tool (skip the resolver) |

> **API status:** This script uses `client.beta.connectors`, `client.beta.agents`, and `client.beta.conversations`. These are **beta** endpoints and may change. See the [Connectors documentation](https://docs.mistral.ai/studio-api/connectors) for the latest API reference.

A Python version of the same tutorial is also available [here](../02-build-a-quickstart-generator.ipynb).

---

## Prerequisites

To complete this cookbook, you will need:
- Node.js and a package manager (npm, pnpm, or yarn)
- A Mistral account and API key

## Environment setup

### Install

Use one of the following methods to install the Mistral TypeScript SDK and `dotenv` for loading your API key from a `.env` file:

**npm**:
```bash
npm install @mistralai/mistralai dotenv
```

**pnpm**:
```bash
pnpm add @mistralai/mistralai dotenv
```

**yarn**:
```bash
yarn add @mistralai/mistralai dotenv
```

### Required environment variables

To complete this cookbook, you'll need a Mistral API key. In [Studio](https://console.mistral.ai), navigate to the [API keys section](https://console.mistral.ai/home?profile_dialog=api-keys), choose **Private and shared connectors** for **Connector access scope** and create a new API key.

Create a `.env` at the root of your project and add your Mistral API key:

```
MISTRAL_API_KEY=your-mistral-api-key
```

---

## Step 1 — Setup

Create `build-a-quickstart-generator.ts` in your project directory:

```bash
touch build-a-quickstart-generator.ts
```

Open the file and add the client, the Context7 server URL, both prompts, and a helper function for extracting output text. The remaining steps fill in the `main` function's `try` and `finally` blocks.

We define `PROMPT` once and reuse it for Steps 1 through 4 so the comparison is fair. Step 5 uses a different, more focused prompt that pre-specifies the Polars library so the connector's resolver tool is unnecessary.

```typescript
import "dotenv/config";
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: process.env.MISTRAL_API_KEY });

// Workaround: the Connectors beta API now returns `owner_type` as a string,
// but the SDK's Zod schema still expects a number. This wrapper catches the
// validation error and returns the raw (correctly shaped) response instead.
// Remove this once the SDK ships a fix.
async function createConnector(
  params: Parameters<typeof client.beta.connectors.create>[0],
) {
  try {
    return await client.beta.connectors.create(params);
  } catch (e: any) {
    if (e.rawValue && e.statusCode === 201) return e.rawValue;
    throw e;
  }
}

const CONTEXT7_URL = "https://mcp.context7.com/mcp";

const PROMPT =
  "Write a Polars quickstart for a developer who knows pandas. " +
  "Cover: installation, reading a CSV, filtering rows, groupby aggregation, " +
  "and lazy evaluation. End with 3 gotchas when migrating from pandas. " +
  "Include runnable code examples.";

const FILTERED_PROMPT =
  "Using the Polars documentation, explain lazy evaluation in Polars. " +
  "Cover: what LazyFrame is, how to build a lazy query with .lazy(), " +
  "how .collect() triggers execution, and when to prefer lazy over eager. " +
  "Include a runnable before/after code example.";

// Extract printable text from a conversation output entry.
function outputText(output: any): string {
  const content = output.content;
  if (typeof content === "string") return content;
  return (content as any[]).map((c) => c.text ?? "").join("");
}

async function main(): Promise<void> {
  let agentId: string | undefined;
  let connectorId: string | undefined;

  try {
    // Step 2 — Baseline: no tools
    // Step 3 — Web search
    // Step 4 — Context7 connector: create connector, create agent, stream
    // Step 5 — Both tools combined: update agent
    // Step 6 — Filtered connector: update agent
  } finally {
    // Cleanup — delete agent and connector
  }
}

main().catch(console.error);
```

---

## Step 2 — Baseline: no tools

We start by asking the model to generate a Polars quickstart with no tools at all. The model can only draw on its training data, so anything that changed after the knowledge cutoff will be missing or wrong.

Replace `// Step 2 — Baseline: no tools` with:

```typescript
    // Step 2 — Baseline: no tools
    console.log("--- Step 1: Baseline (no tools) ---\n");
    const baseline = await client.beta.conversations.start({
      model: "mistral-medium-latest",
      inputs: [{ role: "user", content: PROMPT }],
    });
    for (const output of baseline.outputs ?? []) {
      if ((output as any).type === "message.output") {
        console.log(outputText(output));
      }
    }
```

---

## Step 3 — Web search

Adding `web_search` as a built-in tool lets the model pull in current information — recent blog posts, Stack Overflow answers, and release notes. No connector setup required.

Compare the output with Step 1: you should see more up-to-date syntax and community tips.

Replace `// Step 3 — Web search` with:

```typescript
    // Step 3 — Web search
    console.log("\n--- Step 2: Web search ---\n");
    const webSearch = await client.beta.conversations.start({
      model: "mistral-medium-latest",
      inputs: [{ role: "user", content: PROMPT }],
      tools: [{ type: "web_search" }],
    });
    for (const output of webSearch.outputs ?? []) {
      if ((output as any).type === "message.output") {
        console.log(outputText(output));
      }
    }
```

---

## Step 4 — Context7 connector: create connector, create agent, stream

[Context7](https://context7.com) is an MCP server that serves up-to-date documentation for popular open-source libraries. It requires no authentication, making it a good first connector.

This step has four parts:
1. **Create** the connector pointing at Context7's MCP endpoint
2. **Register credentials** (empty, since Context7 is public — but the record must exist)
3. **List tools** to see what the connector exposes (we will use these names in Step 6)
4. **Create an agent** with the connector attached and stream a conversation — we use an agent instead of the conversations API directly so we can watch connector tool calls (`tool.execution.started`, `tool.execution.delta`, `tool.execution.done`) happen in real time

Replace `// Step 4 — Context7 connector: create connector, create agent, stream` with:

```typescript
    // Step 4 — Context7 connector: create connector, create agent, stream
    console.log("\n--- Step 3: Context7 connector (official docs) ---\n");

    const connector = await createConnector({
      name: "quickstart_context7",
      description: "Context7 connector — library documentation lookup",
      server: CONTEXT7_URL,
      visibility: "private",
    });
    connectorId = connector.id;
    console.log(`Created: ${connector.name}  (id=${connector.id})`);

    await client.beta.connectors.createOrUpdateUserCredentials({
      connectorIdOrName: connector.name!,
      credentialsCreateOrUpdate: {
        name: `${connector.name}-default`,
        credentials: { headers: {} },
        isDefault: true,
      },
    });
    console.log(`Credentials registered for ${connector.name}`);

    const toolsList = await client.beta.connectors.listTools({
      connectorIdOrName: connector.name!,
    });
    let docToolName: string | undefined;
    console.log(`\nTools exposed by ${connector.name}:`);
    for (const tool of toolsList) {
      console.log(`  - ${tool.name}: ${tool.description}`);
      if (
        (tool.description ?? "").toLowerCase().includes("documentation") ||
        tool.name.toLowerCase().includes("doc")
      ) {
        docToolName = tool.name;
      }
    }
    if (docToolName) {
      console.log(`\nDoc-retrieval tool for Step 5: ${docToolName}`);
    }

    const agent = await client.beta.agents.create({
      name: "quickstart_context7_agent",
      model: "mistral-medium-latest",
      instructions:
        "You are a helpful programming assistant. " +
        "When asked about a library, always use the Context7 connector to look up " +
        "the official documentation before answering. Do not rely on training data alone.",
      tools: [{ type: "connector" as const, connectorId: connector.id }],
    });
    agentId = agent.id;
    console.log(`Agent ready: ${agent.name}  (id=${agent.id})\n`);

    let conversationId: string | undefined;
    const stream3 = await client.beta.conversations.startStream(
      {
        agentId: agent.id,
        inputs: [{ role: "user", content: PROMPT }],
      },
      { timeoutMs: 300_000 },
    );
    for await (const item of stream3) {
      const data = item.data;
      const eventType = (data as any).type;
      const name = (data as any).name ?? "";
      if (eventType === "conversation.response.started") {
        conversationId = (data as any).conversationId;
      } else if (eventType === "message.output.delta") {
        process.stdout.write(".");
      } else {
        console.log(`\n[${eventType}]${name ? ` ${name}` : ""}`);
      }
    }

    const messages3 = await client.beta.conversations.getMessages({
      conversationId: conversationId!,
    });
    const lastOutput3 = [...(messages3.messages ?? [])]
      .reverse()
      .find((m) => (m as any).type === "message.output") as any;
    if (lastOutput3) {
      console.log("\n");
      console.log(outputText(lastOutput3));
    }
```

View your registered Connectors in [Studio](https://console.mistral.ai/build/connectors).

---

## Step 5 — Both tools combined: update agent

This is the payoff. We update the agent, giving it both web search **and** the Context7 connector. The model can pull official documentation for accurate API examples and web results for community wisdom, migration gotchas, and recent release notes. It decides which source to use for each sub-topic.

Compare this output with Steps 1-3 — the combined version is noticeably richer.

Replace `// Step 5 — Both tools combined: update agent` with:

```typescript
    // Step 5 — Both tools combined: update agent
    console.log("\n--- Step 4: Web search + Context7 connector ---\n");

    await client.beta.agents.update({
      agentId: agent.id,
      updateAgentRequest: {
        tools: [
          { type: "web_search" },
          { type: "connector" as const, connectorId: connector.id },
        ],
      },
    });

    conversationId = undefined;
    const stream4 = await client.beta.conversations.startStream(
      {
        agentId: agent.id,
        inputs: [{ role: "user", content: PROMPT }],
      },
      { timeoutMs: 300_000 },
    );
    for await (const item of stream4) {
      const data = item.data;
      const eventType = (data as any).type;
      const name = (data as any).name ?? "";
      if (eventType === "conversation.response.started") {
        conversationId = (data as any).conversationId;
      } else if (eventType === "message.output.delta") {
        process.stdout.write(".");
      } else {
        console.log(`\n[${eventType}]${name ? ` ${name}` : ""}`);
      }
    }

    const messages4 = await client.beta.conversations.getMessages({
      conversationId: conversationId!,
    });
    const lastOutput4 = [...(messages4.messages ?? [])]
      .reverse()
      .find((m) => (m as any).type === "message.output") as any;
    if (lastOutput4) {
      console.log("\n");
      console.log(outputText(lastOutput4));
    }
```

---

## Step 6 — Filtered connector: update agent

Context7 exposes multiple tools: a resolver (to find the library ID from a name) and a doc-retrieval tool (to fetch pages by library ID). If you already know the library ID, you can skip the resolver by using `toolConfiguration.include` to restrict the connector to just the doc-retrieval tool.

We update the agent one last time, replacing its tools with a single filtered connector. This step also uses a different, more focused prompt that pre-specifies the Polars library so the resolver is unnecessary.

Replace `// Step 6 — Filtered connector: update agent` with:

```typescript
    // Step 6 — Filtered connector: update agent
    if (docToolName) {
      console.log(
        `\n--- Step 5: Filtered connector (only ${docToolName}) ---\n`,
      );

      await client.beta.agents.update({
        agentId: agent.id,
        updateAgentRequest: {
          tools: [
            {
              type: "connector" as const,
              connectorId: connector.id,
              toolConfiguration: {
                include: [docToolName],
              },
            },
          ],
        },
      });

      conversationId = undefined;
      const stream5 = await client.beta.conversations.startStream(
        {
          agentId: agent.id,
          inputs: [{ role: "user", content: FILTERED_PROMPT }],
        },
        { timeoutMs: 300_000 },
      );
      for await (const item of stream5) {
        const data = item.data;
        const eventType = (data as any).type;
        const name = (data as any).name ?? "";
        if (eventType === "conversation.response.started") {
          conversationId = (data as any).conversationId;
        } else if (eventType === "message.output.delta") {
          process.stdout.write(".");
        } else {
          console.log(`\n[${eventType}]${name ? ` ${name}` : ""}`);
        }
      }

      const messages5 = await client.beta.conversations.getMessages({
        conversationId: conversationId!,
      });
      const lastOutput5 = [...(messages5.messages ?? [])]
        .reverse()
        .find((m) => (m as any).type === "message.output") as any;
      if (lastOutput5) {
        console.log("\n");
        console.log(outputText(lastOutput5));
      }
    } else {
      console.log(
        "\nSkipped Step 5 — DOC_TOOL_NAME was not detected. Set it manually from the tool list in Step 3.",
      );
    }
```

---

## Cleanup

Delete the agent and connector when done. Since we reused a single agent across Steps 3-5 (updating its tools each time), there is only one agent to clean up.

Replace `// Cleanup — delete agent and connector` in the `finally` block with:

```typescript
    // Cleanup — delete agent and connector
    if (agentId) {
      await client.beta.agents.delete({ agentId });
      console.log(`\nAgent deleted: ${agentId}`);
    }
    if (connectorId) {
      const result = await client.beta.connectors.delete({
        connectorId,
      });
      console.log(`Connector deleted: ${(result as any).message}`);
    }
```

---

## Run

Once all steps are in place, run the script:

```bash
npx tsx build-a-quickstart-generator.ts
```

Or with `npm start` if you have `tsx` installed as a dev dependency.

---

## Complete script

For reference, here is the full script with all steps combined. You can also view the complete project [on GitHub](https://github.com/mistralai/cookbook/mistral/connectors/quickstart-generator-typescript).

```typescript
import "dotenv/config";
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: process.env.MISTRAL_API_KEY });

const CONTEXT7_URL = "https://mcp.context7.com/mcp";

const PROMPT =
  "Write a Polars quickstart for a developer who knows pandas. " +
  "Cover: installation, reading a CSV, filtering rows, groupby aggregation, " +
  "and lazy evaluation. End with 3 gotchas when migrating from pandas. " +
  "Include runnable code examples.";

const FILTERED_PROMPT =
  "Using the Polars documentation, explain lazy evaluation in Polars. " +
  "Cover: what LazyFrame is, how to build a lazy query with .lazy(), " +
  "how .collect() triggers execution, and when to prefer lazy over eager. " +
  "Include a runnable before/after code example.";

// Extract printable text from a conversation output entry.
function outputText(output: any): string {
  const content = output.content;
  if (typeof content === "string") return content;
  return (content as any[]).map((c) => c.text ?? "").join("");
}

async function main(): Promise<void> {
  let agentId: string | undefined;
  let connectorId: string | undefined;

  try {
    // Step 2 — Baseline: no tools
    console.log("--- Step 1: Baseline (no tools) ---\n");
    const baseline = await client.beta.conversations.start({
      model: "mistral-medium-latest",
      inputs: [{ role: "user", content: PROMPT }],
    });
    for (const output of baseline.outputs ?? []) {
      if ((output as any).type === "message.output") {
        console.log(outputText(output));
      }
    }

    // Step 3 — Web search
    console.log("\n--- Step 2: Web search ---\n");
    const webSearch = await client.beta.conversations.start({
      model: "mistral-medium-latest",
      inputs: [{ role: "user", content: PROMPT }],
      tools: [{ type: "web_search" }],
    });
    for (const output of webSearch.outputs ?? []) {
      if ((output as any).type === "message.output") {
        console.log(outputText(output));
      }
    }

    // Step 4 — Context7 connector: create connector, create agent, stream
    console.log("\n--- Step 3: Context7 connector (official docs) ---\n");

    const connector = await createConnector({
      name: "quickstart_context7",
      description: "Context7 connector — library documentation lookup",
      server: CONTEXT7_URL,
      visibility: "private",
    });
    connectorId = connector.id;
    console.log(`Created: ${connector.name}  (id=${connector.id})`);

    await client.beta.connectors.createOrUpdateUserCredentials({
      connectorIdOrName: connector.name!,
      credentialsCreateOrUpdate: {
        name: `${connector.name}-default`,
        credentials: { headers: {} },
        isDefault: true,
      },
    });
    console.log(`Credentials registered for ${connector.name}`);

    const toolsList = await client.beta.connectors.listTools({
      connectorIdOrName: connector.name!,
    });
    let docToolName: string | undefined;
    console.log(`\nTools exposed by ${connector.name}:`);
    for (const tool of toolsList) {
      console.log(`  - ${tool.name}: ${tool.description}`);
      if (
        (tool.description ?? "").toLowerCase().includes("documentation") ||
        tool.name.toLowerCase().includes("doc")
      ) {
        docToolName = tool.name;
      }
    }
    if (docToolName) {
      console.log(`\nDoc-retrieval tool for Step 5: ${docToolName}`);
    }

    const agent = await client.beta.agents.create({
      name: "quickstart_context7_agent",
      model: "mistral-medium-latest",
      instructions:
        "You are a helpful programming assistant. " +
        "When asked about a library, always use the Context7 connector to look up " +
        "the official documentation before answering. Do not rely on training data alone.",
      tools: [{ type: "connector" as const, connectorId: connector.id }],
    });
    agentId = agent.id;
    console.log(`Agent ready: ${agent.name}  (id=${agent.id})\n`);

    let conversationId: string | undefined;
    const stream3 = await client.beta.conversations.startStream(
      {
        agentId: agent.id,
        inputs: [{ role: "user", content: PROMPT }],
      },
      { timeoutMs: 300_000 },
    );
    for await (const item of stream3) {
      const data = item.data;
      const eventType = (data as any).type;
      const name = (data as any).name ?? "";
      if (eventType === "conversation.response.started") {
        conversationId = (data as any).conversationId;
      } else if (eventType === "message.output.delta") {
        process.stdout.write(".");
      } else {
        console.log(`\n[${eventType}]${name ? ` ${name}` : ""}`);
      }
    }

    const messages3 = await client.beta.conversations.getMessages({
      conversationId: conversationId!,
    });
    const lastOutput3 = [...(messages3.messages ?? [])]
      .reverse()
      .find((m) => (m as any).type === "message.output") as any;
    if (lastOutput3) {
      console.log("\n");
      console.log(outputText(lastOutput3));
    }

    // Step 5 — Both tools combined: update agent
    console.log("\n--- Step 4: Web search + Context7 connector ---\n");

    await client.beta.agents.update({
      agentId: agent.id,
      updateAgentRequest: {
        tools: [
          { type: "web_search" },
          { type: "connector" as const, connectorId: connector.id },
        ],
      },
    });

    conversationId = undefined;
    const stream4 = await client.beta.conversations.startStream(
      {
        agentId: agent.id,
        inputs: [{ role: "user", content: PROMPT }],
      },
      { timeoutMs: 300_000 },
    );
    for await (const item of stream4) {
      const data = item.data;
      const eventType = (data as any).type;
      const name = (data as any).name ?? "";
      if (eventType === "conversation.response.started") {
        conversationId = (data as any).conversationId;
      } else if (eventType === "message.output.delta") {
        process.stdout.write(".");
      } else {
        console.log(`\n[${eventType}]${name ? ` ${name}` : ""}`);
      }
    }

    const messages4 = await client.beta.conversations.getMessages({
      conversationId: conversationId!,
    });
    const lastOutput4 = [...(messages4.messages ?? [])]
      .reverse()
      .find((m) => (m as any).type === "message.output") as any;
    if (lastOutput4) {
      console.log("\n");
      console.log(outputText(lastOutput4));
    }

    // Step 6 — Filtered connector: update agent
    if (docToolName) {
      console.log(
        `\n--- Step 5: Filtered connector (only ${docToolName}) ---\n`,
      );

      await client.beta.agents.update({
        agentId: agent.id,
        updateAgentRequest: {
          tools: [
            {
              type: "connector" as const,
              connectorId: connector.id,
              toolConfiguration: {
                include: [docToolName],
              },
            },
          ],
        },
      });

      conversationId = undefined;
      const stream5 = await client.beta.conversations.startStream(
        {
          agentId: agent.id,
          inputs: [{ role: "user", content: FILTERED_PROMPT }],
        },
        { timeoutMs: 300_000 },
      );
      for await (const item of stream5) {
        const data = item.data;
        const eventType = (data as any).type;
        const name = (data as any).name ?? "";
        if (eventType === "conversation.response.started") {
          conversationId = (data as any).conversationId;
        } else if (eventType === "message.output.delta") {
          process.stdout.write(".");
        } else {
          console.log(`\n[${eventType}]${name ? ` ${name}` : ""}`);
        }
      }

      const messages5 = await client.beta.conversations.getMessages({
        conversationId: conversationId!,
      });
      const lastOutput5 = [...(messages5.messages ?? [])]
        .reverse()
        .find((m) => (m as any).type === "message.output") as any;
      if (lastOutput5) {
        console.log("\n");
        console.log(outputText(lastOutput5));
      }
    } else {
      console.log(
        "\nSkipped Step 5 — DOC_TOOL_NAME was not detected. Set it manually from the tool list in Step 3.",
      );
    }
  } finally {
    // Cleanup — delete agent and connector
    if (agentId) {
      await client.beta.agents.delete({ agentId });
      console.log(`\nAgent deleted: ${agentId}`);
    }
    if (connectorId) {
      const result = await client.beta.connectors.delete({
        connectorId,
      });
      console.log(`Connector deleted: ${(result as any).message}`);
    }
  }
}

main().catch(console.error);
```

---

## Summary

This script demonstrated how adding tools to a Mistral conversation progressively improves output quality — from a baseline response using only training data, through web search and a documentation connector, to combining both for the richest result.

**What you built:**
- A Polars quickstart generator that improves with each tool added
- A Context7 connector for fetching official library documentation
- A filtered tool configuration that skips unnecessary connector tools

**Mistral features used:**
- [Connectors](https://docs.mistral.ai/studio-api/connectors) (beta)
- Conversations API (beta)
- Agents API (beta) — used to stream tool execution events
- [Web search](https://docs.mistral.ai/studio-api/agents/agent-tools/websearch) built-in tool
- [Tool filtering](https://docs.mistral.ai/studio-api/connectors/conversations#filtering-tools) (`toolConfiguration.include`)

**Other services:**
- [Context7](https://context7.com) — MCP server for open-source library documentation

View your Connectors in [Studio](https://console.mistral.ai/build/connectors).
