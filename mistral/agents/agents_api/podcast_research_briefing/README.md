# Podcast Research Briefing Agent

A research agent that searches Spotify's podcast catalog for episodes on a topic, enriches them with web research, and synthesizes a structured research briefing with ranked recommendations.

## Architecture

The notebook defines six function tools (Spotify search/details + briefing generation), registers them on a Mistral agent alongside built-in web search, and streams the agent's output while executing tool calls locally.

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

## Usage

Open the notebook in Jupyter or Colab and run the cells top-to-bottom. The notebook installs its own dependencies and prompts for API keys via `getpass`.
