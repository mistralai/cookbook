# Human-in-the-Loop Tool Confirmation Cookbook

Add approval flows to tool calls so users can review and confirm or reject actions before they execute.

> **API status:** This cookbook uses `client.beta.conversations`. This is a **beta** endpoint and may change.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Concepts](#concepts)
- [Recipes](#recipes)
  - [1. Local Functions with Confirmation](#1-local-functions-with-confirmation)
  - [2. Connector (Gmail) with Confirmation](#2-connector-gmail-with-confirmation)
  - [3. Stateless / API-Friendly — Serialize and Resume](#3-stateless--api-friendly--serialize-and-resume)

---

## Prerequisites

### Install

```bash
# Python
pip install mistralai
# or with uv
uv add mistralai
```

### Required environment variables

```bash
MISTRAL_API_KEY=your-mistral-api-key
```

Get your API key from the [Mistral AI dashboard](https://console.mistral.ai/).

---

## Concepts

There are two kinds of deferral flows with the `RunContext.run_async` loop:
- Client-side: [local MCP clients & functions](https://docs.mistral.ai/agents/tools/mcp) registered through `register_mcp_client` or `register_func`
- Server-side: remote Mistral connectors (gmail, ...)

The `run_async` loop is responsible for interrupting itself when encountering a deferred tool call, deferred tool calls can be manifested both by local functions or by server-side events (through a `FunctionCallEntry` with `confirmation_status: "pending"`).

### Configuration

To configure confirmation requirement behavior we use the following tool declaration structure:

```
tools=[
      {
          "type": "connector",
          "connector_id": "gmail",
          "tool_configuration": {
              "include": ["gmail_search"],
              "exclude": ["gmail_send"],           # mutually exclusive with include
              "requires_confirmation": ["gmail_search"],
          },
      },
      {
          "type": "web_search_premium",
          "tool_configuration": {
              "requires_confirmation": ["web_search", "news_search"],
          },
      },
  ]
```


---

### Loop pattern

The recipes below use a `while True` loop to catch `DeferredToolCallsException`, prompt for approval, and resume the conversation in one script. This is convenient for demos and CLI tools.

In production, the deferral and resumption typically happen in separate contexts — for example, your backend catches the deferral, sends the pending tool calls to a frontend for user approval, then resumes the conversation when the frontend responds. [Recipe 3 (Serialize and Resume)](#3-stateless--api-friendly--serialize-and-resume) shows this pattern.

---

## Recipes

---

### 1. Local Functions with Confirmation

**When to use:**
- You have local Python functions you want the model to be able to execute directly without any wiring
  - Some are safe (e.g., read-only lookups) and should auto-execute and keep the agentic loop going
  - Some require human approval (e.g., write-operations, booking, ...) and require approval before execution

```python
import asyncio
import os
import random

from mistralai.client import Mistral
from mistralai.extra.run.context import RunContext
from mistralai.extra.exceptions import DeferredToolCallsException

MODEL = "mistral-large-latest"


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    temp = random.randint(10, 30)
    conditions = random.choice(["sunny", "cloudy", "partly cloudy"])
    return f"The weather in {city} is {conditions}, {temp}C"


def book_flight(destination: str, date: str) -> str:
    """Book a flight to a destination."""
    return f"Flight booked to {destination} on {date}. Confirmation: FL-{random.randint(10000, 99999)}"


def request_approval(dc) -> bool:
    print(f"\n[APPROVAL REQUIRED] {dc.tool_name}")
    print(f"  Arguments: {dc.arguments}")
    return input("  Approve? (y/n): ").strip().lower() == "y"


async def main():
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    conversation_id = None
    pending_inputs = [
        {"role": "user", "content": "I need a vacation somewhere warm next Friday. Can you help?"}
    ]

    while True:
        async with RunContext(model=MODEL) as run_ctx:
            run_ctx.conversation_id = conversation_id
            run_ctx.register_func(get_weather, requires_confirmation=False)
            run_ctx.register_func(book_flight, requires_confirmation=True)

            try:
                result = await client.beta.conversations.run_async(
                    run_ctx=run_ctx,
                    inputs=pending_inputs,
                    instructions="You are a travel assistant. Available destinations are: Guingamp, Aurillac, Brive-la-Gaillarde, Rodez, and Millau. Check the weather and book a flight to the warmest one. Do not ask for confirmation, just book it.",
                )

                print(f"\nFinal response: {result.output_entries}")
                break

            except DeferredToolCallsException as deferred:
                conversation_id = deferred.conversation_id

                pending_inputs = [
                    dc.confirm() if request_approval(dc) else dc.reject("Denied by user")
                    for dc in deferred.deferred_calls
                ]


asyncio.run(main())
```

**How it works:**
- Functions registered with `requires_confirmation=False` auto-execute when the model calls them.
- Functions registered with `requires_confirmation=True` pause execution and raise a `DeferredToolCallsException` instead.
- The exception contains the pending tool calls. Call `dc.confirm()` or `dc.reject()` on each one, then pass them back as `inputs` to resume the conversation.

---

### 2. Connector (Gmail) with Confirmation

**When to use:**
- You want to give the model access to a remote Mistral connector (e.g: Gmail)
- You want human approval for some operations.

**Prereqs:** A valid Google OAuth2 token (`GMAIL_OAUTH_TOKEN` env var).

```python
import asyncio
import os

from mistralai.client import Mistral
from mistralai.extra.exceptions import DeferredToolCallsException
from mistralai.extra.run.context import RunContext

MODEL = "mistral-large-latest"


def request_approval(dc) -> bool:
    print(f"\n[APPROVAL REQUIRED] {dc.tool_name}")
    print(f"  Arguments: {dc.arguments}")
    return input("  Approve? (y/n): ").strip().lower() == "y"


async def main():
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    conversation_id = None
    pending_inputs = [
        {"role": "user", "content": "Summarize my latest emails from Gmail."}
    ]

    while True:
        async with RunContext(model=MODEL) as run_ctx:
            run_ctx.conversation_id = conversation_id

            try:
                result = await client.beta.conversations.run_async(
                    run_ctx=run_ctx,
                    inputs=pending_inputs,
                    instructions="You are a helpful assistant. Use the Gmail connector to access the user's emails.",
                    tools=[
                        {
                            "type": "connector",
                            "connector_id": "gmail",
                            "authorization": {
                                "type": "oauth2-token",
                                "value": os.environ["GMAIL_OAUTH_TOKEN"],
                            },
                            "tool_configuration": {
                                "requires_confirmation": ["gmail_search"],
                            },
                        },
                    ],
                )

                for entry in result.output_entries:
                    if hasattr(entry, "content"):
                        print(f"\n[{entry.type}] {entry.content}")
                    else:
                        print(f"\n[{entry.type}] {entry.name}({entry.arguments})")
                break

            except DeferredToolCallsException as deferred:
                conversation_id = deferred.conversation_id

                pending_inputs = [
                    dc.confirm() if request_approval(dc) else dc.reject("Denied by user")
                    for dc in deferred.deferred_calls
                ]


asyncio.run(main())
```

**How it works:**
- Tool names listed in `requires_confirmation` are paused server-side instead of executing immediately.
- `run_async` detects the paused tools and raises a `DeferredToolCallsException`.
- Call `dc.confirm()` to allow execution or `dc.reject()` to deny it. The decision is sent back to the server when the conversation resumes.

---

### 3. Stateless / API-Friendly — Serialize and Resume

**When to use:**
- The approval step happens in a different process or service than the one that started the conversation.
- You need serialization of the tool calls, executions, and resume requests.

Split into two scripts to simulate a real API boundary (e.g., backend returns deferred state to frontend, frontend sends back approvals).

#### Script 1: Start the conversation, catch the deferral, serialize it

```python
import asyncio
import json
import os

from mistralai.client import Mistral
from mistralai.extra.run.context import RunContext
from mistralai.extra.exceptions import DeferredToolCallsException


def book_flight(destination: str, date: str) -> str:
    """Book a flight to a destination."""
    return f"Flight booked to {destination} on {date}"


async def main():
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    async with RunContext(model="mistral-large-latest") as run_ctx:
        run_ctx.register_func(book_flight, requires_confirmation=True)

        try:
            result = await client.beta.conversations.run_async(
                run_ctx=run_ctx,
                inputs=[{"role": "user", "content": "Book me a flight to Paris next Friday."}],
                instructions="You are a travel assistant. Book the flight directly.",
            )
            print("No confirmation needed:", result.output_as_text)

        except DeferredToolCallsException as deferred:
            state = deferred.to_dict()
            serialized = json.dumps(state)

            print("Deferred state (send this to your frontend / store it):")
            print(serialized)


asyncio.run(main())
```

#### Script 2: Receive approvals, deserialize, and resume

```python
import asyncio
import json
import os

from mistralai.client import Mistral
from mistralai.extra.run.context import RunContext
from mistralai.extra.exceptions import DeferredToolCallsException, DeferredToolCallEntry


def book_flight(destination: str, date: str) -> str:
    """Book a flight to a destination."""
    return f"Flight booked to {destination} on {date}"


async def main():
    # In a real app: receive this from the frontend / load from DB
    serialized = os.environ["DEFERRED_STATE"]  # the JSON string from Script 1
    state = json.loads(serialized)

    # Reconstruct the exception from the serialized state
    deferred = DeferredToolCallsException.from_dict(state)

    # Build confirmations (in a real app, the frontend tells you which to approve/reject)
    pending_inputs = []
    for dc in deferred.deferred_calls:
        print(f"Tool: {dc.tool_name}, Args: {dc.arguments}")
        pending_inputs.append(dc.confirm())

    # Include any already-executed results
    pending_inputs = list(deferred.executed_results) + pending_inputs

    # Resume the conversation
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    async with RunContext(model="mistral-large-latest") as run_ctx:
        run_ctx.conversation_id = deferred.conversation_id
        run_ctx.register_func(book_flight, requires_confirmation=True)

        result = await client.beta.conversations.run_async(
            run_ctx=run_ctx,
            inputs=pending_inputs,
            instructions="You are a travel assistant. Book the flight directly.",
        )
        print("Final response:", result.output_as_text)


asyncio.run(main())
```

**How it works:**
- `deferred.to_dict()` serializes the full deferral state (conversation ID, pending tool calls, already-executed results) to a plain dict you can store or send over the wire.
- In a separate process, `DeferredToolCallsException.from_dict(state)` reconstructs that state. From there, confirm or reject calls and resume the conversation as usual.
- The resuming process must re-register the same local functions with `register_func` so they can be executed after approval.
