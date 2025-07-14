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
    "get_current_stock_price": "Yahoo Finance Server",
    "get_historical_stock_prices": "Yahoo Finance Server",
    "get_analyst_recommendations": "Yahoo Finance Server",
    "generate_report_content": "Report Generation Server",
    "save_report": "Save Report Server",
}

class FinanceAgent:
    """Main financial analysis agent that handles user queries using MistralAI with MCP servers"""
    
    def __init__(self):
        self.client = None
        self.agent = None
        
    async def initialize(self):
        """Initialize the MistralAI client and create the financial agent"""
        api_key = os.environ["MISTRAL_API_KEY"]
        self.client = Mistral(api_key=api_key)
        
        self.agent = self.client.beta.agents.create(
            model=MODEL,
            name="finance-agent",
            instructions="Agent to answer user queries regarding finance, generate reports and save them.",
            description="Financial analysis agent",
        )
    
    async def process_query(self, query: str):
        """Process user query using the agent with MCP servers for financial data and reporting"""
        async def run_in_context():
            if not self.client or not self.agent:
                await self.initialize()

            # Configure MCP servers for different financial capabilities
            server_params = [
                # Yahoo Finance server for stock data and financial metrics
                StdioServerParameters(
                    command="python",
                    args=[str((cwd / "mcp_servers/stdio_yfinance_server.py").resolve())],
                    env=None,
                ),
                # Report generation server for creating financial reports
                StdioServerParameters(
                    command="python",
                    args=[str((cwd / "mcp_servers/stdio_report_gen_server.py").resolve())],
                    env=None,
                ),
                # Report saving server for persisting generated reports
                StdioServerParameters(
                    command="python",
                    args=[str((cwd / "mcp_servers/stdio_save_report_server.py").resolve())],
                    env=None,
                )
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

# Initialize the global finance agent instance
finance_agent = FinanceAgent()

@cl.on_chat_start
async def start():
    """Initialize the chat session and set up the finance agent"""
    try:
        await finance_agent.initialize()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming user messages and stream agent responses"""
    user_query = message.content.strip()
    server_tools = []
    
    if not user_query:
        return
    
    try:
        result_events = await finance_agent.process_query(user_query)
        msg = cl.Message(content="")

        async for event in result_events:
            print(event)
            if hasattr(event, 'data') and event.data:
                match event.data:
                    # Stream agent text responses to the UI
                    case MessageOutputEvent():
                        match event.data.content:
                            case str():
                                print(event.data.content)
                                await msg.stream_token(event.data.content)

                    # Display which MCP server and tool is being executed
                    case FunctionCallEvent():
                        server_tool = f"**Selected Server: {server_to_tool_map[event.data.name]} & Running Tool:** {event.data.name}\n\n"
                        if server_tool not in server_tools:
                            server_tools.append(server_tool)
                            await msg.stream_token(server_tool)

                    # Handle and display any errors from the agent
                    case ResponseErrorEvent():
                        logger.debug(event.data)
                        await msg.stream_token(f"\n\n 🔴 Error: {event.data.message}\n\n")
        
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

if __name__ == "__main__":
    cl.run()