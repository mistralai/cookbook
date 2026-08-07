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
    // Step 1 — Baseline: no tools
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

    // Step 2 — Web search
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

    // Step 3 — Context7 connector: create connector, create agent, stream
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

    // Step 4 — Both tools combined: update agent
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

    // Step 5 — Filtered connector: update agent
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
