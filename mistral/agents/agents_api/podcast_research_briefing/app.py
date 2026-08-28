import os
import chainlit as cl
from mistralai import Mistral, MessageOutputEvent, FunctionCallEvent, ResponseErrorEvent
from mistralai.extra.run.context import RunContext
from mcp import StdioServerParameters
from mistralai.extra.mcp.stdio import MCPClientSTDIO
from pathlib import Path
from loguru import logger

cwd = Path(__file__).parent
MODEL = "mistral-medium-latest"

# Maps tool names to their corresponding MCP server for UI display
server_to_tool_map = {
    "search_podcasts": "Spotify Podcast Server",
    "search_episodes": "Spotify Podcast Server",
    "get_podcast_details": "Spotify Podcast Server",
    "get_podcast_episodes": "Spotify Podcast Server",
    "get_episode_details": "Spotify Podcast Server",
    "generate_research_briefing": "Briefing Generation Server",
}

AGENT_INSTRUCTIONS = """You are a podcast research assistant. Your job is to help users
find and analyze podcast content on any topic by searching Spotify's catalog and
synthesizing a structured research briefing.

Follow this workflow for each research request:

1. **Search broadly**: Use search_episodes and search_podcasts with varied query
   phrasings to cast a wide net. Try 2-3 different search queries to find diverse results.

2. **Get details on top results**: For the most promising episodes and shows,
   use get_episode_details and get_podcast_details to get full descriptions.

3. **Enrich with web research**: Use web_search to find additional context about
   the top episodes — look for transcripts, guest bios, episode summaries, or reviews.

4. **Generate the briefing**: Pass all collected data to generate_research_briefing
   to produce the final structured output.

5. **Present results**: Share the briefing with the user. If results are sparse,
   note the gaps and suggest alternative search angles.

Always aim for at least 5-10 relevant episodes before generating the briefing.
If a niche topic yields few results, acknowledge this in your response."""


class PodcastResearchAgent:
    """Podcast research agent that searches Spotify and generates briefings using MCP servers"""

    def __init__(self):
        self.client = None
        self.agent = None

    async def initialize(self):
        """Initialize the Mistral client and create the podcast research agent"""
        api_key = os.environ["MISTRAL_API_KEY"]
        self.client = Mistral(api_key=api_key)

        self.agent = self.client.beta.agents.create(
            model=MODEL,
            name="podcast-research-agent",
            instructions=AGENT_INSTRUCTIONS,
            description="Podcast research briefing agent",
            tools=[{"type": "web_search"}],
        )

    async def process_query(self, query: str):
        """Process user query using the agent with MCP servers for podcast research"""
        async def run_in_context():
            if not self.client or not self.agent:
                await self.initialize()

            # Configure MCP servers
            server_params = [
                # Spotify server for podcast and episode search
                StdioServerParameters(
                    command="python",
                    args=[str((cwd / "mcp_servers/stdio_spotify_server.py").resolve())],
                    env={
                        **os.environ,
                        "SPOTIFY_CLIENT_ID": os.environ["SPOTIFY_CLIENT_ID"],
                        "SPOTIFY_CLIENT_SECRET": os.environ["SPOTIFY_CLIENT_SECRET"],
                    },
                ),
                # Briefing generation server for structured output
                StdioServerParameters(
                    command="python",
                    args=[str((cwd / "mcp_servers/stdio_briefing_server.py").resolve())],
                    env={
                        **os.environ,
                        "MISTRAL_API_KEY": os.environ["MISTRAL_API_KEY"],
                    },
                ),
            ]

            async with RunContext(
                agent_id=self.agent.id,
                continue_on_fn_error=False,
            ) as run_ctx:
                # Register all MCP clients with the run context
                mcp_clients = [MCPClientSTDIO(stdio_params=params) for params in server_params]
                await run_ctx.register_mcp_clients(mcp_clients=mcp_clients)

                # Stream agent responses with tool execution events
                result_events = await self.client.beta.conversations.run_stream_async(
                    run_ctx=run_ctx,
                    inputs=query,
                )

                async for event in result_events:
                    yield event
        return run_in_context()


# Initialize the global agent instance
podcast_agent = PodcastResearchAgent()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="AI Safety Podcasts",
            message="Research podcasts about AI safety and alignment. Find episodes featuring leading researchers and recent developments.",
        ),
        cl.Starter(
            label="Climate Tech Deep Dive",
            message="Find podcast episodes covering climate technology and clean energy innovations from the past year.",
        ),
        cl.Starter(
            label="Startup Founders",
            message="Research podcast interviews with startup founders about lessons learned from building companies from zero to IPO.",
        ),
    ]


@cl.on_chat_start
async def start():
    """Initialize the chat session and set up the podcast research agent"""
    try:
        await podcast_agent.initialize()
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming user messages and stream agent responses"""
    user_query = message.content.strip()
    server_tools = []

    if not user_query:
        return

    try:
        result_events = await podcast_agent.process_query(user_query)
        msg = cl.Message(content="")

        async for event in result_events:
            if hasattr(event, 'data') and event.data:
                match event.data:
                    # Stream agent text responses to the UI
                    case MessageOutputEvent():
                        match event.data.content:
                            case str():
                                await msg.stream_token(event.data.content)

                    # Display which MCP server and tool is being executed
                    case FunctionCallEvent():
                        tool_name = event.data.name
                        server_name = server_to_tool_map.get(tool_name, "Web Search")
                        server_tool = f"**Selected Server: {server_name} & Running Tool:** {tool_name}\n\n"
                        if server_tool not in server_tools:
                            server_tools.append(server_tool)
                            await msg.stream_token(server_tool)

                    # Handle and display any errors from the agent
                    case ResponseErrorEvent():
                        logger.debug(event.data)
                        await msg.stream_token(f"\n\nError: {event.data.message}\n\n")

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()


if __name__ == "__main__":
    cl.run()
