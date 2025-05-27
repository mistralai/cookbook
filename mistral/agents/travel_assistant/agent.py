import os
import chainlit as cl
from dotenv import load_dotenv
from mistralai import (
    Mistral,
    ToolReferenceChunk,
    FunctionResultEntry,
    MessageOutputEvent,
    AgentHandoffDoneEvent,
    FunctionCallEvent,
    ToolExecutionStartedEvent,
    ResponseErrorEvent,
)

import json
from tools.search_hotel_tool import SEARCH_HOTEL_TOOL, search_hotel_serpapi
from tools.customer_care_tool import MODIFY_BOOKING_TOOL, modify_booking
from collections import defaultdict
from loguru import logger

load_dotenv()

# Mistral API Client
client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

# Functions Mapping
functions_mapping = {
    "search_hotel_serpapi": search_hotel_serpapi,
    "modify_booking": modify_booking,
}

# Customer Care Agent
customer_care_agent = client.beta.agents.create(
    model="mistral-medium-2505",
    name="customer-care-agent",
    description="Respond to customer queries, help them modify their booking, cancel their booking, or ask about their booking",
    instructions="Customer care agent, your goal is to help customer modify their booking, cancel their booking, or ask about their booking. Answer must be short and concise, use emojis to make it more engaging.",
    tools=[MODIFY_BOOKING_TOOL],
)

# Hotel Search Agent
hotel_search = client.beta.agents.create(
    model="mistral-medium-2505",
    name="hotel-search-agent",
    description="Use search hotels based on user request, feel free to use hotel search tool to find hotels, answer must be short and concise, use emojis to make it more engaging",
    instructions="Hotel search agent, your goal is to search for hotels based on user request, always ask user for their preferences before searching for hotels. Answer must be short and concise, use emojis to make it more engaging.",
    tools=[SEARCH_HOTEL_TOOL],
)

# Trip Itinerary Agent
trip_itinerary = client.beta.agents.create(
    model="mistral-medium-2505",
    name="trip-itinerary-agent",
    description="Use trip itinerary agent with web search tool to find events and activities based on user preferences",
    instructions="Trip itinerary agent, your goal is to plan trip itinerary based on user preferences, always ask user for their preferences before planning the itinerary. Feel free to use emojis to make it more engaging.",
    tools=[{"type": "web_search"}],
)

# Triage Agent
triage_agent = client.beta.agents.create(
    model="mistral-medium-2505",
    name="triage-agent",
    description="Agent which provide assistant to users for their travel",
    instructions="""
    You are a travel assistant agent and your goal is to handoff user request to the right agent, you have 3 agents to forward to, trip itinerary agent, hotel search agent, and customer care agent. Be nice and friendly with engaging answers.
    """,
)

# Handoffs
client.beta.agents.update(
    agent_id=triage_agent.id,
    handoffs=[trip_itinerary.id, hotel_search.id, customer_care_agent.id],
)
client.beta.agents.update(
    agent_id=hotel_search.id,
    handoffs=[customer_care_agent.id, trip_itinerary.id],
)

client.beta.agents.update(
    agent_id=trip_itinerary.id,
    handoffs=[hotel_search.id, customer_care_agent.id],
)

client.beta.agents.update(
    agent_id=customer_care_agent.id,
    handoffs=[trip_itinerary.id, hotel_search.id],
)


async def handle_tool_execution(
    msg: cl.Message, tool_call_id: str, function_output: dict
):
    result = functions_mapping[function_output["name"]](
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
    await msg.stream_token("\n\n")

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
            agent_id=triage_agent.id, inputs=message.content
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
            match event.data:
                # handle message output event
                case MessageOutputEvent():
                    match event.data.content:
                        case str():
                            await msg.stream_token(event.data.content)
                        case ToolReferenceChunk():
                            # handle citations
                            tool_reference = event.data.content
                            if tool_reference.url not in ref_index:
                                ref_index[tool_reference.url] = counter
                                counter += 1

                                link_text = f" [{ref_index[tool_reference.url]}]({tool_reference.url}) "
                            await msg.stream_token(link_text)

                # handle Tool Execution Started Event
                case ToolExecutionStartedEvent():
                    await msg.stream_token(f"**Running Tool:** `{event.data.name}`\n\n")

                # handle Agent Handoff Done Event
                case AgentHandoffDoneEvent():
                    await msg.stream_token(
                        f"🔄 **Routing to Agent:** `{event.data.next_agent_name}`\n\n"
                    )

                # handle Function Call Event+
                case FunctionCallEvent():
                    tool_call_id = event.data.tool_call_id
                    tool_name = event.data.name
                    function_output["name"] = tool_name
                    function_output["arguments"] += event.data.arguments

                case ResponseErrorEvent():
                    logger.debug(event.data)
                    await msg.stream_token(
                        f"\n\n 🔴 **Error:** `{event.data.message}`\n\n"
                    )

    # handle Function Call Event
    if tool_call_id:
        await msg.stream_token(f"**Running Tool:** `{tool_name}`\n\n")
        await msg.stream_token(f"```json\n {function_output['arguments']}\n```\n")
        await handle_tool_execution(msg, tool_call_id, function_output)

    await msg.update()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
