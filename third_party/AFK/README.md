# Use Mistral AI with AFK coding-agent sessions

Author: AFK team, [@lcavadas](https://github.com/lcavadas)  
Affiliation: [AFK](https://afk.mooglest.com)

[AFK](https://afk.mooglest.com) is a browser-based command center for persistent coding-agent sessions. AFK supports Mistral AI as a built-in LLM connection, so you can bring your Mistral API key, choose a Mistral model per session, and supervise agent work from the web UI.

Use this integration when you want Mistral-hosted models in an AFK workflow for code changes, reviews, debugging, documentation updates, or longer-running agent tasks.

## Prerequisites

- A Mistral AI API key from [La Plateforme](https://console.mistral.ai/)
- An AFK account at [afk.mooglest.com](https://afk.mooglest.com)
- An AFK daemon connected to the machine that has access to your project files

## 1. Create or sign in to AFK

Open [afk.mooglest.com](https://afk.mooglest.com) and create an account or sign in.

AFK runs from the browser UI while a daemon gives sessions access to your local or remote project directories.

## 2. Install and connect an AFK daemon

In AFK:

1. Open **Account → API Keys**.
2. Create a daemon token.
3. Follow the install command shown in the app.
4. Confirm the daemon appears as connected in the browser.

## 3. Add Mistral AI as an LLM connection

In AFK:

1. Open **Account → LLM**.
2. Click **Add connection**.
3. Choose **Mistral AI**.
4. Paste your Mistral API key.
5. Leave **Base URL** blank unless you are routing through a custom proxy or gateway.
6. Save or test the connection.

AFK uses Mistral's default OpenAI-compatible endpoint automatically for the built-in Mistral provider.

## 4. Start a session with a Mistral model

Click **New session** in AFK, then:

1. Select the connected daemon and project directory.
2. Choose the Mistral AI connection.
3. Select or type a Mistral model name, for example:

   ```text
   mistral-large-latest
   mistral-medium-latest
   mistral-small-latest
   codestral-latest
   ```

4. Choose a permission mode.
5. Enter the coding task and start the session.

AFK will route the session's model requests through Mistral while the browser UI shows progress, tool usage, diffs, and session history.

## Optional: use a proxy or gateway

If your team routes Mistral traffic through an internal gateway, set **Base URL** to the gateway's OpenAI-compatible endpoint.

For example:

```text
https://your-gateway.example.com/v1
```

Keep Base URL blank for normal Mistral AI usage.

## Troubleshooting

| Issue | Check |
|-------|-------|
| Connection test fails | Verify the Mistral API key and confirm your network can reach Mistral AI. |
| Model is missing | Manually type the Mistral model name in AFK. Provider model discovery can lag behind newly released models. |
| Custom gateway errors | Confirm the Base URL includes the OpenAI-compatible `/v1` path expected by your gateway. |
| Session cannot access files | Confirm the selected AFK daemon is connected and has the project directory under an allowed root. |

## Resources

- [AFK](https://afk.mooglest.com)
- [AFK provider setup docs](https://docs.mooglest.com/providers)
- [Mistral AI documentation](https://docs.mistral.ai/)
- [Mistral AI models](https://docs.mistral.ai/getting-started/models/models_overview/)
