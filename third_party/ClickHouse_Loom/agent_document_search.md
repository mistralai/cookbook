# Search a team document from a Mistral agent

**Contributor** Mohammad Lareb Zafar ([@zlareb1](https://github.com/zlareb1)), ClickHouse Loom

Use the Mistral Agents SDK to retrieve a team document from a ClickHouse Loom namespace over MCP. This example gives a support agent read access to a runbook so it can answer a question and cite the retrieved source.

## Set up the connection

Install the SDK with Python 3.12–3.14.

```sh
python -m pip install mistralai-agents==1.1.0
```

Set `MISTRAL_API_KEY` in the agent environment. In Loom, index a test runbook with a source URL and create a viewer token scoped to that namespace. See the [Loom MCP setup guide](https://github.com/ClickHouse/loom/blob/v1.3.0/docs/features/integrations/mcp-clients.md).

Set `LOOM_BASE_URL` to the HTTPS origin of the Loom deployment. Set `LOOM_MCP_AUTHORIZATION` to the complete header value, including the `Bearer ` prefix. The SDK reads this value from the environment when the agent runs. For a deployed agent, configure it as a secret with the same variable name.

The assistant toolset exposes read-only operations, including `loom_knowledge_search`. The viewer token limits which namespaces the agent can read. For different access scopes, run separate agents with their own credentials.

## Retrieve the document

Save this example as `main.py`. Change the final question to one covered by the indexed document.

```python
import asyncio
import os

from mistralai import agents


agent = agents.Agent(
    model="mistral-large-4",
    instructions=(
        "Use the Loom knowledge search tool for runbook questions. "
        "Cite the source URL when a passage supports the answer. Otherwise say the runbook does not answer the question."
    ),
    mcps={
        "loom": agents.RemoteMCP(
            url=os.environ["LOOM_BASE_URL"].rstrip("/") + "/mcp?toolset=assistant",
            auth_header_env_vars={"Authorization": "LOOM_MCP_AUTHORIZATION"},
            model_access="direct",
        )
    },
)


async def main():
    result = await agent.run("What does the team runbook say about rotating a service token?")
    if not isinstance(result, agents.Completed):
        raise RuntimeError(f"Agent stopped with {type(result).__name__}")
    print(result.text)
    print([call.name for call in result.tool_calls])


if __name__ == "__main__":
    asyncio.run(main())
```

Run `python main.py`. Look for `loom.loom_knowledge_search` in the tool calls and compare the answer with the runbook. If the tool is unavailable, check that the worker can reach Loom and that the environment variable contains a valid viewer token in the expected namespace.

See [Connectors and MCP](https://docs.mistral.ai/studio/agents/learn/agent/connectors) for remote server notices and credential configuration.
