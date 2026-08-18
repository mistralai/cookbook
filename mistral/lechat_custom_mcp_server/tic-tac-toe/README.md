# Tic-Tac-Toe MCP Server

An MCP server that lets you play tic-tac-toe against Mistral directly from [Vibe](https://chat.mistral.ai). The server exposes game tools over SSE that Vibe discovers and calls automatically.

For the full step-by-step tutorial on how to build this project, see the [Custom MCP Server cookbook](../custom-mcp-server.md).

## Prerequisites

- Python 3.11+
- A [Mistral API key](https://console.mistral.ai)

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your API key:

```bash
export MISTRAL_API_KEY=your_api_key_here
```

Run the MCP server:

```bash
python mcp_server.py
```

The server starts on port 7860 with SSE transport.

## Project structure

| File | Description |
|------|-------------|
| `app.py` | Game logic (Room class, AI functions) and Flask REST endpoints |
| `mcp_server.py` | MCP server wrapping game logic as six callable tools |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container config for Hugging Face Spaces deployment |

## Deployment

See the [cookbook tutorial](../custom-mcp-server.md#step-4--deploy-to-hugging-face-spaces) for deploying to Hugging Face Spaces and connecting to Vibe.
