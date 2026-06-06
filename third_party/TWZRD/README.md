# TWZRD Agent Intel + Mistral

This example shows how to use [TWZRD Agent Intel](https://intel.twzrd.xyz) with
Mistral models to verify the trustworthiness of autonomous AI agents before
routing value through an [x402](https://x402.org) payment workflow.

TWZRD scores agents (0–100) based on their on-chain Solana transaction history.
Agents with repeated, consistent payment behavior score higher. The score is
returned via a free MCP server at `https://intel.twzrd.xyz/mcp`.

## What it demonstrates

- Connecting to a remote MCP server (streamable-http) from a Python script
- Using `score_agent` to evaluate an agent wallet before transacting
- Using `preflight_check` to surface context about an unknown agent

## Requirements

```shell
pip install mistralai mcp
```

No API key is required for the TWZRD tools. You will need a Mistral API key:

```shell
export MISTRAL_API_KEY=your_key_here
```

## Run

```shell
python agent_trust_check.py
```

## MCP server config

```json
{
  "mcpServers": {
    "twzrd-agent-intel": {
      "url": "https://intel.twzrd.xyz/mcp"
    }
  }
}
```

## Available free tools

| Tool | Description |
|------|-------------|
| `score_agent` | On-chain trust score (0–100) for a Solana wallet |
| `resolve_agent` | Resolve a handle / domain to a wallet address |
| `preflight_check` | Human-readable context before sending a payment |
| `verify_trust_receipt` | Verify a signed receipt from a paid `get_trust_receipt` call |

## Resources

- MCP endpoint: `https://intel.twzrd.xyz/mcp`
- PyPI: `pip install twzrd-agent-intel`
- Docs: [https://intel.twzrd.xyz](https://intel.twzrd.xyz)
