import "dotenv/config";
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: process.env.MISTRAL_API_KEY });

// The agent can produce multiple message outputs across turns (e.g. an
// intermediate summary before the final answer). The API stores them
// concatenated in the content field regardless of retrieval method. This
// helper scans for all JSON objects and returns the last one, which is always
// the json_schema-constrained final answer.
function lastJsonIn(s: string): unknown {
  let remaining = s;
  let last: unknown;
  while (remaining) {
    const i = remaining.indexOf("{");
    if (i === -1) break;
    remaining = remaining.slice(i);
    try {
      last = JSON.parse(remaining);
      break;
    } catch (e: any) {
      const pos = e.message?.match(/position (\d+)/)?.[1];
      if (pos !== undefined) {
        last = JSON.parse(remaining.slice(0, Number(pos)));
        remaining = remaining.slice(Number(pos));
      } else {
        remaining = remaining.slice(1);
      }
    }
  }
  if (last === undefined) throw new Error("No JSON object found in response");
  return last;
}

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
    for (const c of candidates) {
      const connector = await client.beta.connectors.create({
        name: c.name,
        description: c.description,
        server: DEEPWIKI_URL,
        visibility: "private",
      });
      connectorIds[c.name] = connector.id;
      console.log(`Created: ${connector.name}  (id=${connector.id})`);
      await client.beta.connectors.createOrUpdateUserCredentials({
        connectorIdOrName: connector.name,
        credentialsCreateOrUpdate: {
          name: `${connector.name}-default`,
          credentials: { headers: {} },
          isDefault: true,
        },
      });
      console.log(`  Credentials registered for ${connector.name}`);
    }

    // Step 3 — List to verify
    const page = await client.beta.connectors.list({ pageSize: 50 });
    const showdown = (page.items ?? []).filter((c) =>
      c.name?.startsWith("showdown_")
    );
    console.log(`${showdown.length} showdown connectors registered:`);
    for (const c of showdown) {
      console.log(`  ${(c.name ?? "").padEnd(22)}  ${c.description}`);
      const tools = await client.beta.connectors.listTools({
        connectorIdOrName: c.name ?? "",
      });
      for (const tool of tools) {
        console.log(`    - ${tool.name}: ${tool.description}`);
      }
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
    const stream = await client.beta.conversations.startStream(
      {
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
      },
      { timeoutMs: 300_000 },
    );

    // startStream keeps the connection alive and lets you observe connector
    // calls in real time. Capture the conversation_id from the first event so
    // we can retrieve the final output after the stream ends.
    let conversationId: string | undefined;
    for await (const item of stream) {
      const data = item.data;
      const eventType = data.type;
      const name = (data as any).name ?? "";
      if (eventType === "conversation.response.started") {
        conversationId = (data as any).conversationId;
      } else if (eventType !== "message.output.delta") {
        console.log(`[${eventType}]${name ? ` ${name}` : ""}`);
      }
    }

    // getMessages returns the completed conversation entries. The last
    // message.output entry is the agent's final answer, which is guaranteed
    // to match RESPONSE_SCHEMA because json_schema was set on the agent.
    const messages = await client.beta.conversations.getMessages({
      conversationId: conversationId!,
    });
    const lastOutput = [...(messages.messages ?? [])].reverse()
      .find((m) => (m as any).type === "message.output") as any;
    const rawContent = lastOutput.content;
    const rawText = typeof rawContent === "string"
      ? rawContent
      : (rawContent as any[]).map((c) => c.text ?? "").join("");
    const result = lastJsonIn(rawText) as ComparisonResult;

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
  } finally {
    // Cleanup — delete the agent
    if (agentId) {
      await client.beta.agents.delete({ agentId });
      console.log(`\nAgent deleted: ${agentId}`);
    }
    // To also remove the winning connector, capture winnerName before the
    // finally block and uncomment:
    // await client.beta.connectors.delete({ connectorId: connectorIds[winnerName] });
  }
}

main().catch(console.error);
