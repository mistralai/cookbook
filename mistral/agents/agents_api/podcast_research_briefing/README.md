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
2. **Spotify Developer credentials**: Follow the steps below to get a Client ID and Client Secret.

### Setting up Spotify credentials

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and log in with your Spotify account. **A Spotify Premium subscription is required** to use the Web API (as of February 2026).
2. Click **Create app**.
3. Fill in the form:
   - **App name**: Any name (e.g. "Podcast Research Agent")
   - **App description**: Any description
   - **Redirect URI**: Enter `https://localhost:8080/callback` (this won't be used, but the field is required)
   - **Which API/SDKs are you planning to use?**: Select **Web API**
4. Check the terms of service box and click **Save**.
5. On your app's dashboard, click **Settings**.
6. Copy the **Client ID** and **Client Secret** (click "View client secret" to reveal it).

This cookbook uses the **Client Credentials** auth flow, which provides read-only access to Spotify's public catalog (podcast search, show details, episode details). No user login or OAuth redirect is needed at runtime.

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
