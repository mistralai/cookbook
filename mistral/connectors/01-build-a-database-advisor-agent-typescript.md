# Build a Database Advisor Agent with the DeepWiki Connector (TypeScript)

You need a database for write-heavy local analytics. SQLite, DuckDB, and LevelDB are all strong contenders — but which one actually fits? Rather than reading documentation by hand, this script lets Mistral read their actual source code via the [DeepWiki](https://deepwiki.com) MCP connector and decide.

This script demonstrates the full Mistral Connector lifecycle:

| Step | Operation | What happens |
|---|---|---|
| 1 | **Create** | Register a connector for each database candidate |
| 2 | **List** | Verify all three are registered |
| 3 | **Use** | Build an agent that compares them via their GitHub repos |
| 4 | **Update** | Mark the winner's connector as selected |
| 5 | **Delete** | Clean up the losing connectors |

> **API status:** This script uses `client.beta.connectors` and `client.beta.agents`. These are **beta** endpoints and may change.

A Python version of the same workflow is in [`01-build-a-database-advisor-agent.ipynb`](./01-build-a-database-advisor-agent.ipynb).

---

## Prerequisites

### Install

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

```bash
npx tsx build-a-database-advisor-agent.ts
```

---

## The Script

Save the full script below as `build-a-database-advisor-agent.ts` and run it with `npx tsx`.

```typescript
import Mistral from "@mistralai/mistralai";

const client = new Mistral({ apiKey: process.env.MISTRAL_API_KEY });

const DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp";

const candidates = [
  { name: "showdown_sqlite",  description: "DeepWiki connector — sqlite/sqlite" },
  { name: "showdown_duckdb",  description: "DeepWiki connector — duckdb/duckdb" },
  { name: "showdown_leveldb", description: "DeepWiki connector — google/leveldb" },
];

// ─── Step 1: Create one connector per candidate ───────────────────────────────
//
// Each connector points at the DeepWiki MCP server, which lets Mistral read and
// reason about any public GitHub repository. We create three named connectors —
// one per candidate — so each one acts as a named slot the agent can query
// independently.

async function main(): Promise<void> {
  let agentId: string | undefined;
  const connectorIds: Record<string, string> = {};

  try {
    console.log("=== Step 1: Create connectors ===");
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

    // ─── Step 2: List to verify ─────────────────────────────────────────────
    //
    // Confirm all three connectors are registered before proceeding.

    console.log("\n=== Step 2: List connectors ===");
    const page = await client.beta.connectors.list({ pageSize: 50 });
    const showdown = (page.items ?? []).filter((c) =>
      c.name?.startsWith("showdown_")
    );
    console.log(`${showdown.length} showdown connectors registered:`);
    for (const c of showdown) {
      console.log(`  ${(c.name ?? "").padEnd(22)}  ${c.description}`);
    }

    // ─── Step 3: Build the comparison agent ─────────────────────────────────
    //
    // We create a Mistral agent with all three connectors attached. Its
    // instructions require a structured output — the agent must end every
    // response with a RECOMMENDATION: line so we can parse the winner
    // programmatically.

    console.log("\n=== Step 3: Create agent ===");
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

    // ─── Step 4: Run the comparison ─────────────────────────────────────────
    //
    // Ask the agent to evaluate all three databases for a write-heavy local
    // analytics workload. The agent will call DeepWiki tools on each connector
    // to read actual source code before answering — this may take a minute.

    console.log("\n=== Step 4: Run comparison (this may take a minute) ===");
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

    // Extract the agent's full reply
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

    // Parse the RECOMMENDATION: line
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

    // ─── Step 5: Promote the winner, retire the rest ────────────────────────
    //
    // Update the winning connector's description to mark it as selected, then
    // delete the losing connectors. This completes the full lifecycle:
    // create → list → use → update → delete.

    console.log("\n=== Step 5: Promote winner, delete losers ===");
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

    // Confirm the winner is still there with its updated description
    const winner = await client.beta.connectors.get({
      connectorIdOrName: winnerName,
    });
    console.log(`\nWinner confirmed:`);
    console.log(`  Name:        ${winner.name}`);
    console.log(`  Description: ${winner.description}`);
    console.log(`  ID:          ${winner.id}`);
  } finally {
    // ─── Cleanup ────────────────────────────────────────────────────────────
    //
    // Delete the agent. The winning connector is kept — uncomment the lines
    // below to remove it too.

    if (agentId) {
      await client.beta.agents.delete({ agentId });
      console.log(`\nAgent deleted: ${agentId}`);
    }
    // To also remove the winning connector, uncomment:
    // const winnerName = ...; // capture before finally block if needed
    // await client.beta.connectors.delete({ connectorId: connectorIds[winnerName] });
  }
}

main().catch(console.error);
```

---

## Expected output

```
=== Step 1: Create connectors ===
Created: showdown_sqlite   (id=a1b2c3d4-...)
Created: showdown_duckdb   (id=b2c3d4e5-...)
Created: showdown_leveldb  (id=c3d4e5f6-...)

=== Step 2: List connectors ===
3 showdown connectors registered:
  showdown_sqlite         DeepWiki connector — sqlite/sqlite
  showdown_duckdb         DeepWiki connector — duckdb/duckdb
  showdown_leveldb        DeepWiki connector — google/leveldb

=== Step 3: Create agent ===
Agent ready: Database Showdown Judge  (id=d4e5f6a7-...)

=== Step 4: Run comparison (this may take a minute) ===
## Storage Model
- **SQLite**: B-tree based, row-oriented...
- **DuckDB**: Columnar storage optimized for analytics...
- **LevelDB**: LSM-tree, optimized for sequential writes...

[... full comparison ...]

RECOMMENDATION: showdown_duckdb

Winner: showdown_duckdb
Losers: showdown_sqlite, showdown_leveldb

=== Step 5: Promote winner, delete losers ===
Updated:  showdown_duckdb  —  [SELECTED] DeepWiki connector — duckdb/duckdb
Deleted:  showdown_sqlite  —  Connector deleted successfully
Deleted:  showdown_leveldb  —  Connector deleted successfully

Winner confirmed:
  Name:        showdown_duckdb
  Description: [SELECTED] DeepWiki connector — duckdb/duckdb
  ID:          b2c3d4e5-...

Agent deleted: d4e5f6a7-...
```
