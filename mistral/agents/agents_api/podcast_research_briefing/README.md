# Podcast Research Briefing Agent

A research agent that searches Spotify's podcast catalog for episodes on a topic, enriches them with web research, and synthesizes a structured briefing with ranked recommendations.

## Architecture

```
├── app.py                              # Chainlit UI + agent orchestration
├── chainlit.md                         # Chainlit welcome message
├── pyproject.toml                      # uv project config
├── .env.example                        # Required env vars template
└── mcp_servers/
    ├── stdio_spotify_server.py         # Spotify podcast search (spotipy)
    └── stdio_briefing_server.py        # LLM-powered briefing generation
```

### Main Application
- **app.py**: Chainlit interface with a Mistral agent that orchestrates podcast research across two MCP servers and built-in web search.

### MCP Servers
- **stdio_spotify_server.py**: Wraps the Spotify Web API via `spotipy` with Client Credentials auth. Provides tools for searching podcasts, searching episodes, and fetching details.
- **stdio_briefing_server.py**: Uses `mistral-medium-latest` to generate a structured research briefing from collected podcast data and web research.

### Built-in Tools
- **web_search**: The agent uses Mistral's built-in web search to find transcripts, guest bios, and episode summaries to enrich the briefing.

## Prerequisites

1. **Mistral API key**: Get one at [console.mistral.ai](https://console.mistral.ai)
2. **Spotify Developer credentials**: Create an app at [developer.spotify.com](https://developer.spotify.com/dashboard) to get a Client ID and Client Secret. No user login is required — this uses Client Credentials (read-only public catalog access).

## Installation

```bash
cd mistral/agents/agents_api/podcast_research_briefing
uv sync
```

## Environment Setup

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Or export them directly:

```bash
export MISTRAL_API_KEY="your_mistral_api_key"
export SPOTIFY_CLIENT_ID="your_spotify_client_id"
export SPOTIFY_CLIENT_SECRET="your_spotify_client_secret"
```

## Usage

```bash
chainlit run app.py -w
```

Open your browser and try queries like:
- "Research podcasts about AI safety and alignment"
- "Find podcast episodes covering climate technology from the past year"
- "Research podcast interviews with startup founders about scaling companies"

The agent will search Spotify, gather web context, and generate a briefing with:
- Executive Summary
- Ranked Episode Recommendations (with Spotify links)
- Key Themes Across Episodes
- Notable Experts & Guests
- Suggested Deep Dives
- Gaps & Limitations
