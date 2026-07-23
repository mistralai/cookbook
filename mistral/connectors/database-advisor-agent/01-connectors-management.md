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

Create `build-a-database-advisor-agent.ts` in your project directory:

```bash
touch build-a-database-advisor-agent.ts
```

Open the file and add the client, the DeepWiki server URL, the list of candidates, and the response schema. The remaining steps fill in the `main` function's `try` and `finally` blocks.

The response schema defines the exact JSON structure the agent must return. Defining it as a typed constant means it serves as both the TypeScript type source and the schema passed to the API — no duplication.

```typescript
import "dotenv/config";
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: process.env.MISTRAL_API_KEY });

const DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp";

const candidates = [
  { name: "showdown_sqlite",  description: "DeepWiki connector — sqlite/sqlite" },
  { name: "showdown_duckdb",  description: "DeepWiki connector — duckdb/duckdb" },
  { name: "showdown_leveldb", description: "DeepWiki connector — google/leveldb" },
];

// The JSON schema the agent must conform to. Passed to the API via completionArgs
// so the structure is enforced at the API level, not just by prompt instructions.
const RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    queries: {
      type: "array",
      items: {
        type: "object",
        properties: {
          connector: { type: "string" },
          question:  { type: "string" },
          summary:   { type: "string" },
        },
        required: ["connector", "question", "summary"],
      },
    },
    comparison: {
      type: "object",
      properties: {
        storage_model:      { type: "string" },
        acid_guarantees:    { type: "string" },
        query_capabilities: { type: "string" },
        write_throughput:   { type: "string" },
        python_api:         { type: "string" },
      },
      required: ["storage_model", "acid_guarantees", "query_capabilities", "write_throughput", "python_api"],
    },
    reasoning:      { type: "string" },
    recommendation: { type: "string", enum: ["showdown_sqlite", "showdown_duckdb", "showdown_leveldb"] },
  },
  required: ["queries", "comparison", "reasoning", "recommendation"],
} as const;

interface ComparisonResult {
  queries: { connector: string; question: string; summary: string }[];
  comparison: Record<string, string>;
  reasoning: string;
  recommendation: string;
}

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

Create a Mistral agent with all three connectors attached. The instructions tell the agent to ask each connector a natural-language question rather than reading raw source files — this keeps responses concise enough to fit in the context window. Passing `RESPONSE_SCHEMA` via `completionArgs.responseFormat` enforces the JSON structure at the API level, so the recommendation field is always a valid connector name and the output is always parseable without regex.

Replace `// Step 4 — Build the comparison agent` with:

```typescript
    // Step 4 — Build the comparison agent
    const agent = await client.beta.agents.create({
      name: "Database Showdown Judge",
      description: "Compares database candidates using their source code via DeepWiki.",
      model: "mistral-medium-latest",
      instructions:
        "You are a database selection expert. " +
        "When given a comparison task, call each DeepWiki connector once with a focused " +
        "natural-language question about the repository — do NOT read raw source files.",
      completionArgs: {
        responseFormat: {
          type: "json_schema" as const,
          jsonSchema: {
            name: "comparison_result",
            schemaDefinition: RESPONSE_SCHEMA,
            strict: true,
          },
        },
      },
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

Ask the agent to evaluate all three databases. Before replying, the agent will call each DeepWiki connector with a natural-language question — this may take a minute.

The response loop does two things: it logs any non-message outputs (connector calls, internal steps) so you can confirm the connectors are being invoked, and it collects the final message text for JSON parsing.

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

    // Collect the agent's full reply and surface tool call activity
    let rawText = "";
    for (const output of response.outputs ?? []) {
      if (output.type === "message.output") {
        const content = output.content;
        if (typeof content === "string") {
          rawText += content;
        } else if (Array.isArray(content)) {
          rawText += content
            .map((chunk: any) => chunk.text ?? String(chunk))
            .join("");
        }
      } else {
        const name = (output as any).name ?? (output as any).toolName ?? "";
        const args = (output as any).arguments;
        let detail = name ? ` — ${name}` : "";
        if (args) {
          try {
            const parsed = typeof args === "string" ? JSON.parse(args) : args;
            detail += `\n    ${JSON.stringify(parsed, null, 2)}`;
          } catch {
            detail += `\n    ${args}`;
          }
        }
        console.log(`[${output.type}]${detail}`);
      }
    }

    // The agent enforces json_schema via completionArgs, so the response is
    // guaranteed to be valid JSON matching RESPONSE_SCHEMA.
    const result = JSON.parse(rawText) as ComparisonResult;

    console.log("\n--- Connector queries ---");
    for (const q of result.queries) {
      console.log(`\n  [${q.connector}]`);
      console.log(`  Q: ${q.question}`);
      console.log(`  A: ${q.summary}`);
    }

    console.log("\n--- Comparison ---");
    for (const [key, val] of Object.entries(result.comparison)) {
      console.log(`  ${key}: ${val}`);
    }

    console.log(`\n--- Reasoning ---\n  ${result.reasoning}`);
    console.log(`\n--- Recommendation ---\n  ${result.recommendation}`);

    // Extract winner and losers directly from the parsed JSON
    const winnerName = result.recommendation;
    const loserNames = candidates.map((c) => c.name).filter((n) => n !== winnerName);

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
      updateConnectorRequest: {
        description: `[SELECTED] ${winnerDescription}`,
      },
    });
    console.log(`Updated:  ${updated.name}  —  ${updated.description}`);

    for (const name of loserNames) {
      const deleteResult = await client.beta.connectors.delete({
        connectorId: connectorIds[name],
      });
      console.log(`Deleted:  ${name}  —  ${deleteResult.message}`);
    }

    // Confirm the winner is still registered with its updated description
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

## Run

Once all steps are in place, run the script:

```bash
npx tsx build-a-database-advisor-agent.ts
```

Or with `npm start` if you have `tsx` installed as a dev dependency.

---

## Complete script

For reference, here is the full script with all steps combined. You can also view the complete project [on GitHub](https://github.com/mistralai/cookbook/mistral/connectors/database-advisor-agent)

```typescript
import "dotenv/config";
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: process.env.MISTRAL_API_KEY });

const DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp";

const candidates = [
  { name: "showdown_sqlite",  description: "DeepWiki connector — sqlite/sqlite" },
  { name: "showdown_duckdb",  description: "DeepWiki connector — duckdb/duckdb" },
  { name: "showdown_leveldb", description: "DeepWiki connector — google/leveldb" },
];

const RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    queries: {
      type: "array",
      items: {
        type: "object",
        properties: {
          connector: { type: "string" },
          question:  { type: "string" },
          summary:   { type: "string" },
        },
        required: ["connector", "question", "summary"],
      },
    },
    comparison: {
      type: "object",
      properties: {
        storage_model:      { type: "string" },
        acid_guarantees:    { type: "string" },
        query_capabilities: { type: "string" },
        write_throughput:   { type: "string" },
        python_api:         { type: "string" },
      },
      required: ["storage_model", "acid_guarantees", "query_capabilities", "write_throughput", "python_api"],
    },
    reasoning:      { type: "string" },
    recommendation: { type: "string", enum: ["showdown_sqlite", "showdown_duckdb", "showdown_leveldb"] },
  },
  required: ["queries", "comparison", "reasoning", "recommendation"],
} as const;

interface ComparisonResult {
  queries: { connector: string; question: string; summary: string }[];
  comparison: Record<string, string>;
  reasoning: string;
  recommendation: string;
}

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
      model: "mistral-medium-latest",
      instructions:
        "You are a database selection expert. " +
        "When given a comparison task, call each DeepWiki connector once with a focused " +
        "natural-language question about the repository — do NOT read raw source files.",
      completionArgs: {
        responseFormat: {
          type: "json_schema" as const,
          jsonSchema: {
            name: "comparison_result",
            schemaDefinition: RESPONSE_SCHEMA,
            strict: true,
          },
        },
      },
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

    let rawText = "";
    for (const output of response.outputs ?? []) {
      if (output.type === "message.output") {
        const content = output.content;
        if (typeof content === "string") {
          rawText += content;
        } else if (Array.isArray(content)) {
          rawText += content
            .map((chunk: any) => chunk.text ?? String(chunk))
            .join("");
        }
      } else {
        const name = (output as any).name ?? (output as any).toolName ?? "";
        const args = (output as any).arguments;
        let detail = name ? ` — ${name}` : "";
        if (args) {
          try {
            const parsed = typeof args === "string" ? JSON.parse(args) : args;
            detail += `\n    ${JSON.stringify(parsed, null, 2)}`;
          } catch {
            detail += `\n    ${args}`;
          }
        }
        console.log(`[${output.type}]${detail}`);
      }
    }

    const result = JSON.parse(rawText) as ComparisonResult;

    console.log("\n--- Connector queries ---");
    for (const q of result.queries) {
      console.log(`\n  [${q.connector}]`);
      console.log(`  Q: ${q.question}`);
      console.log(`  A: ${q.summary}`);
    }

    console.log("\n--- Comparison ---");
    for (const [key, val] of Object.entries(result.comparison)) {
      console.log(`  ${key}: ${val}`);
    }

    console.log(`\n--- Reasoning ---\n  ${result.reasoning}`);
    console.log(`\n--- Recommendation ---\n  ${result.recommendation}`);

    const winnerName = result.recommendation;
    const loserNames = candidates.map((c) => c.name).filter((n) => n !== winnerName);

    console.log(`\nWinner: ${winnerName}`);
    console.log(`Losers: ${loserNames.join(", ")}`);

    // Step 6 — Promote the winner, retire the rest
    const winnerDescription =
      candidates.find((c) => c.name === winnerName)?.description ?? "";

    const updated = await client.beta.connectors.update({
      connectorId: connectorIds[winnerName],
      updateConnectorRequest: {
        description: `[SELECTED] ${winnerDescription}`,
      },
    });
    console.log(`Updated:  ${updated.name}  —  ${updated.description}`);

    for (const name of loserNames) {
      const deleteResult = await client.beta.connectors.delete({
        connectorId: connectorIds[name],
      });
      console.log(`Deleted:  ${name}  —  ${deleteResult.message}`);
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

This script demonstrated the full Mistral Connector lifecycle — create, list, use, update, and delete — using the DeepWiki Connector to let the model ask natural-language questions about GitHub repository source code and produce a data-driven database recommendation.

**What you built:**
- Three named Connectors pointing at the DeepWiki MCP server
- An agent (Database Showdown Judge) with all three Connectors attached and a `json_schema` response format enforced via `completionArgs`
- A conversation that logged connector tool calls, parsed a structured JSON recommendation, updated the winner's Connector, and cleaned up the rest

**Mistral features used:**
- Connectors (beta)
- Agents API (beta)
- Conversations API (beta)

**Other services:**
- [DeepWiki](https://deepwiki.com) — MCP server for reading public GitHub repositories

View your Connectors in [Studio](https://console.mistral.ai/build/connectors).
