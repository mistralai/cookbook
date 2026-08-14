# Build a quickstart generator with Connectors and web search (TypeScript)

Generate a Polars quickstart guide using Mistral Connectors and web search. The script sends the same prompt five times with different tool configurations to show how output quality improves with better sources.

See the [full cookbook](./02-build-a-quickstart-generator-typescript.md) for a step-by-step walkthrough.

## Setup

### 1. Install dependencies

```bash
npm install
```

### 2. Set up your API key

Copy the example environment file and add your Mistral API key:

```bash
cp .env.example .env
```

To get an API key, open [Studio](https://console.mistral.ai) and navigate to the [API keys section](https://console.mistral.ai/home?profile_dialog=api-keys). Choose **Private and shared connectors** for **Connector access scope** and create a new key. Paste it into `.env`:

```
MISTRAL_API_KEY=your-mistral-api-key
```

### 3. Run

```bash
npm start
```

This runs `build-a-quickstart-generator.ts` via `tsx`. The script creates temporary connectors, runs five comparison rounds, and cleans up after itself.
