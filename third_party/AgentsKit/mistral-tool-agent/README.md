# Mistral tool-calling agent with AgentsKit

This example builds a small Mistral-powered weather agent in TypeScript. Mistral decides when to call a typed tool, AgentsKit runs the tool loop, and the final result reports both the answer and the operations performed.

The weather data is intentionally local and deterministic so the example focuses on Mistral tool calling rather than an unrelated API.

## What you will learn

- connect `mistral-small-latest` to an agent runtime;
- describe a tool with JSON Schema and validate its arguments at the execution boundary;
- inspect the number of model steps and completed tool calls;
- test the complete model → tool → model loop without spending API credits.

## Requirements

- Node.js 20 or newer;
- a Mistral API key for the live example.

## Run the example

```bash
npm install
cp .env.example .env
```

Set `MISTRAL_API_KEY` in `.env`, then run:

```bash
npm start -- "What is the temperature in Lisbon?"
```

The program prints Mistral's answer followed by a compact execution receipt:

```json
{
  "steps": 2,
  "tools": [{ "name": "get_weather", "status": "complete" }],
  "durationMs": 342
}
```

## Test without an API key

```bash
npm test
npm run typecheck
```

The test replaces the network boundary with deterministic Mistral-compatible streaming responses. It verifies the Mistral endpoint, tool schema, tool result, second model turn, and final answer. No credential or network call is required.

## Swap the model without rewriting the agent

Set `MISTRAL_MODEL` to another chat-completions model supported by your Mistral account. The tool and runtime code stay unchanged:

```bash
MISTRAL_MODEL=mistral-large-latest npm start -- "What is the temperature in Tokyo?"
```

AgentsKit is open source under the MIT license. The adapter, runtime, and tool contracts used here are independently installable packages; the complete source is available at [AgentsKit-io/agentskit](https://github.com/AgentsKit-io/agentskit).
