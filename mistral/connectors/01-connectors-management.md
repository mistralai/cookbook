# Build a Database Advisor Agent with the DeepWiki Connector (TypeScript)

You need a database for write-heavy local analytics. SQLite, DuckDB, and LevelDB are all strong contenders, but which one actually fits? Rather than reading documentation by hand, this script lets a Mistral Agent read their actual source code via the [DeepWiki](https://deepwiki.com) Connector and decide.

This script demonstrates the full [Mistral Connector](https://docs.mistral.ai/studio-api/connectors) lifecycle:

| Step | Operation | What happens |
|---|---|---|
| 1 | **Create** | Register a connector for each database candidate |
| 2 | **List** | Verify all three are registered |
| 3 | **Use** | Build an agent that compares them via their GitHub repos |
| 4 | **Update** | Mark the winner's connector as selected |
| 5 | **Delete** | Clean up the losing connectors |

> **API status:** This script uses `client.beta.connectors` and `client.beta.agents`. These are **beta** endpoints and may change.

A Python version of the same agent is also available [here](./01-build-a-database-advisor-agent.ipynb).

---

## Prerequisites

To complete this cookbook, you will need:
- Node.js and a package manager (npm, pnpm, or yarn)
- A Mistral account and API key

## Environment setup

### Install

Use one of the following methods to install the Mistral TypeScript SDK:

```bash
# npm
npm install @mistralai/mistralai

# pnpm
pnpm add @mistralai/mistralai

# yarn
yarn add @mistralai/mistralai
```

### Required environment variables

To complete this cookbook, you'll need a Mistral API key. In [Studio](https://console.mistral.ai), navigate to the [API keys section](https://console.mistral.ai/home?profile_dialog=api-keys) and create a new API key.

Create a `.env` at the root of your project and add your Mistral API key:

```
MISTRAL_API_KEY=your-mistral-api-key
```

### Run

Once your `.env` file is in place, run the script:

```bash
npx tsx build-a-database-advisor-agent.ts
```

---

## Step 1 — Setup

Create `build-a-database-advisor-agent.ts` and add the client, the DeepWiki server URL, and the list of candidates. The remaining steps fill in the `main` function's `try` and `finally` blocks.

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: process.env.MISTRAL_API_KEY });

const DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp";

const candidates = [
  { name: "showdown_sqlite",  description: "DeepWiki connector — sqlite/sqlite" },
  { name: "showdown_duckdb",  description: "DeepWiki connector — duckdb/duckdb" },
  { name: "showdown_leveldb", description: "DeepWiki connector — google/leveldb" },
];

async function main(): Promise<void> {
  let agentId: string | undefined;
  const connectorIds: Record<string, string> = {};

  try {
    // Step 2 — Create one connector per candidate
    // Step 3 — List to verify
    // Step 4 — Build the comparison agent
    // Step 5 — Run the comparison
    // Step 6 — Promote the winner, retire the rest
  } finally {
    // Cleanup — delete the agent
  }
}

main().catch(console.error);
```

---

## Step 2 — Create one connector per candidate

Each connector points at the DeepWiki MCP server, which lets Mistral read and reason about any public GitHub repository. Three named connectors — one per database — give the agent independent slots to query.

Replace `// Step 2 — Create one connector per candidate` with:

```typescript
    // Step 2 — Create one connector per candidate
    for (const c of candidates) {
      const connector = await client.beta.connectors.create({
        name: c.name,
        description: c.description,
        server: DEEPWIKI_URL,
        visibility: "private",
      });
      connectorIds[c.name] = connector.id;
      console.log(`Created: ${connector.name}  (id=${connector.id})`);
    }
```

View your registered Connectors in [Studio](https://console.mistral.ai/build/connectors).

---

## Step 3 — List to verify

Confirm all three connectors are registered before proceeding.

Replace `// Step 3 — List to verify` with:

```typescript
    // Step 3 — List to verify
    const page = await client.beta.connectors.list({ pageSize: 50 });
    const showdown = (page.items ?? []).filter((c) =>
      c.name?.startsWith("showdown_")
    );
    console.log(`${showdown.length} showdown connectors registered:`);
    for (const c of showdown) {
      console.log(`  ${(c.name ?? "").padEnd(22)}  ${c.description}`);
    }
```

---

## Step 4 — Build the comparison agent

Create a Mistral agent with all three connectors attached. Its instructions require a structured output — the agent must end every response with a `RECOMMENDATION:` line so we can parse the winner programmatically.

Replace `// Step 4 — Build the comparison agent` with:

```typescript
    // Step 4 — Build the comparison agent
    const agent = await client.beta.agents.create({
      name: "Database Showdown Judge",
      description: "Compares database candidates using their source code via DeepWiki.",
      model: "mistral-large-latest",
      instructions:
        "You are a database selection expert. " +
        "Use the DeepWiki connectors to read each repository's source code and documentation. " +
        "Evaluate: storage model, ACID guarantees, query capabilities, write throughput, and Python API simplicity. " +
        "Be direct and data-driven. " +
        "Always end your response with a line in exactly this format:\n" +
        "RECOMMENDATION: <connector_name>\n" +
        "where <connector_name> is one of: showdown_sqlite, showdown_duckdb, showdown_leveldb.",
      tools: candidates.map((c) => ({
        type: "connector" as const,
        connectorId: connectorIds[c.name],
      })),
    });
    agentId = agent.id;
    console.log(`Agent ready: ${agent.name}  (id=${agent.id})`);
```

View your agents in [Studio](https://console.mistral.ai/build/agents).

---

## Step 5 — Run the comparison

Ask the agent to evaluate all three databases for a write-heavy local analytics workload. The agent calls DeepWiki tools on each connector to read actual source code before answering — this may take a minute.

Replace `// Step 5 — Run the comparison` with:

```typescript
    // Step 5 — Run the comparison
    const response = await client.beta.conversations.start({
      agentId: agent.id,
      inputs: [
        {
          role: "user",
          content:
            "Compare sqlite/sqlite, duckdb/duckdb, and google/leveldb for a write-heavy " +
            "local analytics workload. Evaluate storage model, ACID guarantees, query " +
            "capabilities, write throughput, and Python API simplicity. Recommend one.",
        },
      ],
    });

    let fullText = "";
    for (const output of response.outputs ?? []) {
      if (output.type === "message.output") {
        const content = output.content;
        if (typeof content === "string") {
          fullText += content;
        } else if (Array.isArray(content)) {
          fullText += content
            .map((chunk: any) => chunk.text ?? String(chunk))
            .join("");
        }
      }
    }
    console.log(fullText);

    const match = fullText.match(/RECOMMENDATION:\s*(\S+)/);
    if (!match) {
      throw new Error("Agent did not return a RECOMMENDATION line.");
    }
    const winnerName = match[1].trim();
    const loserNames = candidates
      .map((c) => c.name)
      .filter((n) => n !== winnerName);

    console.log(`\nWinner: ${winnerName}`);
    console.log(`Losers: ${loserNames.join(", ")}`);
```

---

## Step 6 — Promote the winner, retire the rest

Update the winning connector's description to mark it as selected, then delete the losing connectors. This completes the full lifecycle: create → list → use → update → delete.

Replace `// Step 6 — Promote the winner, retire the rest` with:

```typescript
    // Step 6 — Promote the winner, retire the rest
    const winnerDescription =
      candidates.find((c) => c.name === winnerName)?.description ?? "";

    const updated = await client.beta.connectors.update({
      connectorId: connectorIds[winnerName],
      connectorMCPUpdate: {
        description: `[SELECTED] ${winnerDescription}`,
      },
    });
    console.log(`Updated:  ${updated.name}  —  ${updated.description}`);

    for (const name of loserNames) {
      const result = await client.beta.connectors.delete({
        connectorId: connectorIds[name],
      });
      console.log(`Deleted:  ${name}  —  ${result.message}`);
    }

    const winner = await client.beta.connectors.get({
      connectorIdOrName: winnerName,
    });
    console.log(`\nWinner confirmed:`);
    console.log(`  Name:        ${winner.name}`);
    console.log(`  Description: ${winner.description}`);
    console.log(`  ID:          ${winner.id}`);
```

---

## Cleanup

Delete the agent when done. Replace `// Cleanup — delete the agent` in the `finally` block with:

```typescript
    // Cleanup — delete the agent
    if (agentId) {
      await client.beta.agents.delete({ agentId });
      console.log(`\nAgent deleted: ${agentId}`);
    }
    // To also remove the winning connector, capture winnerName before the
    // finally block and uncomment:
    // await client.beta.connectors.delete({ connectorId: connectorIds[winnerName] });
```

---

## Complete script

For reference, here is the full script with all steps combined.

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: process.env.MISTRAL_API_KEY });

const DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp";

const candidates = [
  { name: "showdown_sqlite",  description: "DeepWiki connector — sqlite/sqlite" },
  { name: "showdown_duckdb",  description: "DeepWiki connector — duckdb/duckdb" },
  { name: "showdown_leveldb", description: "DeepWiki connector — google/leveldb" },
];

async function main(): Promise<void> {
  let agentId: string | undefined;
  const connectorIds: Record<string, string> = {};

  try {
    // Step 2 — Create one connector per candidate
    for (const c of candidates) {
      const connector = await client.beta.connectors.create({
        name: c.name,
        description: c.description,
        server: DEEPWIKI_URL,
        visibility: "private",
      });
      connectorIds[c.name] = connector.id;
      console.log(`Created: ${connector.name}  (id=${connector.id})`);
    }

    // Step 3 — List to verify
    const page = await client.beta.connectors.list({ pageSize: 50 });
    const showdown = (page.items ?? []).filter((c) =>
      c.name?.startsWith("showdown_")
    );
    console.log(`${showdown.length} showdown connectors registered:`);
    for (const c of showdown) {
      console.log(`  ${(c.name ?? "").padEnd(22)}  ${c.description}`);
    }

    // Step 4 — Build the comparison agent
    const agent = await client.beta.agents.create({
      name: "Database Showdown Judge",
      description: "Compares database candidates using their source code via DeepWiki.",
      model: "mistral-large-latest",
      instructions:
        "You are a database selection expert. " +
        "Use the DeepWiki connectors to read each repository's source code and documentation. " +
        "Evaluate: storage model, ACID guarantees, query capabilities, write throughput, and Python API simplicity. " +
        "Be direct and data-driven. " +
        "Always end your response with a line in exactly this format:\n" +
        "RECOMMENDATION: <connector_name>\n" +
        "where <connector_name> is one of: showdown_sqlite, showdown_duckdb, showdown_leveldb.",
      tools: candidates.map((c) => ({
        type: "connector" as const,
        connectorId: connectorIds[c.name],
      })),
    });
    agentId = agent.id;
    console.log(`Agent ready: ${agent.name}  (id=${agent.id})`);

    // Step 5 — Run the comparison
    const response = await client.beta.conversations.start({
      agentId: agent.id,
      inputs: [
        {
          role: "user",
          content:
            "Compare sqlite/sqlite, duckdb/duckdb, and google/leveldb for a write-heavy " +
            "local analytics workload. Evaluate storage model, ACID guarantees, query " +
            "capabilities, write throughput, and Python API simplicity. Recommend one.",
        },
      ],
    });

    let fullText = "";
    for (const output of response.outputs ?? []) {
      if (output.type === "message.output") {
        const content = output.content;
        if (typeof content === "string") {
          fullText += content;
        } else if (Array.isArray(content)) {
          fullText += content
            .map((chunk: any) => chunk.text ?? String(chunk))
            .join("");
        }
      }
    }
    console.log(fullText);

    const match = fullText.match(/RECOMMENDATION:\s*(\S+)/);
    if (!match) {
      throw new Error("Agent did not return a RECOMMENDATION line.");
    }
    const winnerName = match[1].trim();
    const loserNames = candidates
      .map((c) => c.name)
      .filter((n) => n !== winnerName);

    console.log(`\nWinner: ${winnerName}`);
    console.log(`Losers: ${loserNames.join(", ")}`);

    // Step 6 — Promote the winner, retire the rest
    const winnerDescription =
      candidates.find((c) => c.name === winnerName)?.description ?? "";

    const updated = await client.beta.connectors.update({
      connectorId: connectorIds[winnerName],
      connectorMCPUpdate: {
        description: `[SELECTED] ${winnerDescription}`,
      },
    });
    console.log(`Updated:  ${updated.name}  —  ${updated.description}`);

    for (const name of loserNames) {
      const result = await client.beta.connectors.delete({
        connectorId: connectorIds[name],
      });
      console.log(`Deleted:  ${name}  —  ${result.message}`);
    }

    const winner = await client.beta.connectors.get({
      connectorIdOrName: winnerName,
    });
    console.log(`\nWinner confirmed:`);
    console.log(`  Name:        ${winner.name}`);
    console.log(`  Description: ${winner.description}`);
    console.log(`  ID:          ${winner.id}`);
  } finally {
    // Cleanup — delete the agent
    if (agentId) {
      await client.beta.agents.delete({ agentId });
      console.log(`\nAgent deleted: ${agentId}`);
    }
    // To also remove the winning connector, uncomment:
    // await client.beta.connectors.delete({ connectorId: connectorIds[winnerName] });
  }
}

main().catch(console.error);
```

---

## Summary

This script demonstrated the full Mistral Connector lifecycle — create, list, use, update, and delete — using the DeepWiki Connector to let the model read actual GitHub repository source code and produce a data-driven database recommendation.

**What you built:**
- Three named Connectors pointing at the DeepWiki MCP server
- An agent (Database Showdown Judge) with all three Connectors attached
- A conversation that produced a structured recommendation, updated the winner's Connector, and cleaned up the rest

**Mistral features used:**
- Connectors (beta)
- Agents API (beta)
- Conversations API (beta)

**Other services:**
- [DeepWiki](https://deepwiki.com) — MCP server for reading public GitHub repositories

View your Connectors in [Studio](https://console.mistral.ai/build/connectors).
