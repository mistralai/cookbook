import os
import chainlit as cl
from dotenv import load_dotenv
import httpx
from mistralai import (
    Mistral,
   
)

from mistralai.extra.run.context import RunContext

from mistralai.extra.mcp.sse import (
    MCPClientSSE,
    SSEServerParams,
)
from mcp import StdioServerParameters
from mistralai.extra.mcp.stdio import (
    MCPClientSTDIO,
)
from mcp import ClientSession
load_dotenv()


MODEL = "mistral-medium-latest"
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

MCP_TOOLS=[]
messages=[]
github_agent = client.beta.agents.create(
        model=MODEL,
        name="github agent",
        instructions="You are able to handle issues and request on a github repository.You can use your tools to help with the task",
        description="You are able to handle issues and request on a github repository.You can use your tools to help with the task ",
    )

def format_messages(all_cl_messages: list[cl.Message]):

    api_input_list = []

    if not all_cl_messages:
        
        return []

    current_cl_message_obj = all_cl_messages[-1]
    historical_cl_messages = all_cl_messages[:-1]

    processed_history_for_api = []
    for msg in historical_cl_messages:
        api_role = "user" if msg.author == "User" else "assistant"
        # Merge if last message in processed_history_for_api has the same API role
        if processed_history_for_api and processed_history_for_api[-1]["role"] == api_role:
            processed_history_for_api[-1]["content"] += f"\\n{msg.content}"
        else:
            processed_history_for_api.append({"role": api_role, "content": msg.content})
    api_input_list.extend(processed_history_for_api)
    api_input_list.append({
        "role": "user", 
        "content": current_cl_message_obj.content
    })
    return api_input_list

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """Called when an MCP connection is established"""
    MCP_TOOLS.append(connection)
    
@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Called when an MCP connection is terminated"""
    MCP_TOOLS.remove(session)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [])

@cl.on_message
async def on_message(message: cl.Message):
        async with RunContext(
        agent_id=github_agent.id,
    ) as run_ctx:
            messages = cl.user_session.get("messages", [])  # Add default [] to avoid None error
            messages.append(message)
            cl.user_session.set("messages", messages)
            
            if MCP_TOOLS:
                #REgistering any tools that are inside the 
                for tool in MCP_TOOLS:
                    if tool.clientType == "sse":
                        print("sse")
                        temp_mcp_client = MCPClientSSE(sse_params=SSEServerParams(url=tool.url, timeout=100))
                    if tool.clientType=="stdio":
                        print("stdio",tool)
                        server_params = StdioServerParameters(
                            command=tool.command,
                            args=tool.args,
                           env=None
                        )
                        temp_mcp_client = MCPClientSTDIO(stdio_params=server_params)


                    await run_ctx.register_mcp_client(mcp_client=temp_mcp_client)    
            #then add github ones :
            server_params = StdioServerParameters(
                command="docker",
                args=[
                    "run",
                    "-i",
                    "--rm",
                    "-e",
                    "GITHUB_PERSONAL_ACCESS_TOKEN",
                    "ghcr.io/github/github-mcp-server"
                    ],
                env={
        "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"]
      }
            )
            temp_mcp_client = MCPClientSTDIO(stdio_params=server_params) 
            await run_ctx.register_mcp_client(mcp_client=temp_mcp_client)   
        

                
            inputs_messages=format_messages(messages)
            response = await client.beta.conversations.run_async(
                run_ctx=run_ctx,
                inputs=inputs_messages,
            )

     

            for entry in response.output_entries:
                if hasattr(entry, 'type') and entry.type == 'function.call':
                    if hasattr(entry, 'name') and hasattr(entry, 'arguments'):
                        mes= cl.Message(
                            content=f"⚙️ Used tool: **{entry.name}**\nArguments: `{entry.arguments}`",
                            author="Agent Steps"
                        )
                        await mes.send()
                        messages.append(mes)
            
            if response.output_as_text:
                 msg = cl.Message(content=response.output_as_text)
                 await msg.send()
                 messages.append(msg)
            cl.user_session.set("messages", messages)
                 


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
