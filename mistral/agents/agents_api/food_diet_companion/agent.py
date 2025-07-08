import chainlit as cl
from dotenv import load_dotenv
from mistralai import (
    ToolReferenceChunk,
    FunctionResultEntry,
    MessageOutputEvent,
    AgentHandoffDoneEvent,
    FunctionCallEvent,
    ToolExecutionStartedEvent,
    ResponseErrorEvent,
)
import json
from tools.daily_progress_tool import DAILY_PROGRESS_TOOL, daily_progress
from tools.goal_setting_tool import GOAL_SETTING_TOOL, goal_setting
from tools.food_logging_tool import FOOD_LOGGING_TOOL, food_logging
from tools.food_recommendations_tool import FOOD_RECOMMENDATIONS_TOOL, food_recommendations
from collections import defaultdict
from loguru import logger
from tools.configs import mistral_model, client
load_dotenv()

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Welcome to NutriSense—your personal diet assistant! How about we start by setting your daily diet goals?").send()

function_tool_mapping = {
    "goal_setting": goal_setting,
    "food_log": food_logging,
    "food_recommendations": food_recommendations,
    "daily_progress": daily_progress,
}

router_agent = client.beta.agents.create(
    model=mistral_model,
    description="Agent used to route user queries to the correct agent.",
    instructions="User query can cater to multiple agents. But your job is to select one agent at a time.",
    name="router-agent",
)

food_logging_agent = client.beta.agents.create(
    model=mistral_model,
    name="food-logging-agent",
    description="""Agent to log the food taken by the user. 
                   Few sample queries:
                   1. Log the food for lunch - 1 bowl chicken, 2 eggs.
                   2. I had 2 eggs for breakfast.
                   3. I had a bowl of fish and coffee.
                   """,
    instructions="Use food logging tool when users share information about the meals they had during the day. Answer should be short and concise.",
    tools=[FOOD_LOGGING_TOOL],
)

goal_setting_agent = client.beta.agents.create(
    model=mistral_model,
    name="goal-setting-agent",
    description="Agent to set/ update the goals when user shares diet goals.",
    instructions="Use goal setting tool when users share information about the diet goals. Answer should be short and concise.",
    tools=[GOAL_SETTING_TOOL],
)

food_recommendations_agent = client.beta.agents.create(
    model=mistral_model,
    name="food-recommendations-agent",
    description="Agent that provides food recommendations based on user diet goals.",
    instructions="Use food recommendations tool when users ask about food recommendations. Answer should be short and concise.",
    tools=[FOOD_RECOMMENDATIONS_TOOL],
)

daily_progress_agent = client.beta.agents.create(
    model=mistral_model,
    name="daily-progress-agent",
    description="Agent that provides daily progress of the user based on the food intake.",
    instructions="Use daily progress tool when user asks about his daily progress. Answer should be short and concise.",
    tools=[DAILY_PROGRESS_TOOL],
)

web_search_agent = client.beta.agents.create(
    model=mistral_model,
    description="Agent that is useful in searching the web for real time information.",
    name="web-search-agent",
    tools=[{"type": "web_search"}],
)

client.beta.agents.update(
  agent_id=router_agent.id,
  handoffs=[food_logging_agent.id, goal_setting_agent.id, food_recommendations_agent.id, daily_progress_agent.id, web_search_agent.id]
)

client.beta.agents.update(
  agent_id=goal_setting_agent.id,
  handoffs=[food_logging_agent.id, food_recommendations_agent.id, daily_progress_agent.id, web_search_agent.id]
)

client.beta.agents.update(
  agent_id=food_logging_agent.id,
  handoffs=[food_recommendations_agent.id, daily_progress_agent.id, web_search_agent.id]
)

client.beta.agents.update(
  agent_id=food_recommendations_agent.id,
  handoffs=[daily_progress_agent.id, web_search_agent.id]
)

client.beta.agents.update(
  agent_id=web_search_agent.id,
  handoffs=[food_logging_agent.id, daily_progress_agent.id]
)

client.beta.agents.update(
  agent_id=daily_progress_agent.id,
  handoffs=[food_recommendations_agent.id, food_logging_agent.id]
)

async def handle_tool_execution(
    msg: cl.Message, tool_call_id: str, function_output: dict
):
    result = function_tool_mapping[function_output["name"]](
        **json.loads(function_output["arguments"])
    )

    user_entry = FunctionResultEntry(
        tool_call_id=tool_call_id,
        result=result,
    )

    # Append Function Result to Conversation
    response = client.beta.conversations.append_stream(
        conversation_id=cl.user_session.get("conversation_id"),
        inputs=[user_entry],
    )
    await msg.stream_token(f"\n\n")

    with response as event_stream:
        for event in event_stream:
            if isinstance(event.data, MessageOutputEvent):
                await msg.stream_token(event.data.content)


@cl.on_message
async def on_message(message: cl.Message):
    if cl.user_session.get("conversation_id"):
        conversation_id = cl.user_session.get("conversation_id")
        logger.debug("Conversation ID", conversation_id)
        response = client.beta.conversations.append_stream(
            conversation_id=conversation_id,
            inputs=f"{message.content}",
        )

    else:
        response = client.beta.conversations.start_stream(
            agent_id=router_agent.id, inputs=message.content
        )

    msg = cl.Message(content="")
    tool_call_id = None
    function_output = {"name": "", "arguments": ""}
    ref_index = defaultdict(int)
    counter = 1

    with response as event_stream:
        # Fetching conversation id
        conversation_id = next(iter(event_stream)).data.conversation_id
        logger.debug(f"Conversation ID: {conversation_id}")
        cl.user_session.set("conversation_id", conversation_id)
        for event in event_stream:
            print(event)

            # handle message output event
            if isinstance(event.data, MessageOutputEvent):
                if isinstance(event.data.content, str):
                    await msg.stream_token(event.data.content)
                elif isinstance(event.data.content, list):
                    await msg.stream_token(event.data.content[0].text)
                elif isinstance(event.data.content, ToolReferenceChunk):
                    # handle citations
                    tool_reference = event.data.content
                    if tool_reference.url not in ref_index:
                        ref_index[tool_reference.url] = counter
                        counter += 1

                        link_text = (
                            f" [{ref_index[tool_reference.url]}]({tool_reference.url}) "
                        )
                    await msg.stream_token(link_text)

            # handle Tool Execution Started Event
            elif isinstance(event.data, ToolExecutionStartedEvent):
                await msg.stream_token(f"**Running Tool:** `{event.data.name}`\n\n")

            # handle Agent Handoff Done Event
            elif isinstance(event.data, AgentHandoffDoneEvent):
                await msg.stream_token(
                    f"🔄 **Forwarding to** `{event.data.next_agent_name}`\n\n"
                )

            # handle Function Call Event
            elif isinstance(event.data, FunctionCallEvent):
                tool_call_id = event.data.tool_call_id
                tool_name = event.data.name
                function_output["name"] = tool_name
                function_output["arguments"] += event.data.arguments

            elif isinstance(event.data, ResponseErrorEvent):
                logger.debug(event.data)
                await msg.stream_token(f"\n\n 🔴 **Error:** `{event.data.message}`\n\n")

    # handle Function Call Event
    if tool_call_id:
        await handle_tool_execution(msg, tool_call_id, function_output)

    await msg.update()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
