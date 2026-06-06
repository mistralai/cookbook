"""
TWZRD Agent Intel + Mistral — agent trust verification before x402 transactions.

Uses the TWZRD Agent Intel MCP server (https://intel.twzrd.xyz/mcp) to score
an AI agent's on-chain reputation, then feeds the result to Mistral to
generate a plain-language recommendation.

Install: pip install mistralai mcp
"""

import asyncio
import json
import os

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mistralai import Mistral


MCP_URL = "https://intel.twzrd.xyz/mcp"
MISTRAL_MODEL = "mistral-small-latest"


async def score_agent(wallet: str) -> dict:
    """Call score_agent on the TWZRD MCP server and return the raw result."""
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("score_agent", {"wallet": wallet})
            return json.loads(result.content[0].text)


def interpret_score(wallet: str, score_data: dict) -> str:
    """Ask Mistral to explain the trust score in plain language."""
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    prompt = (
        f"An autonomous AI agent with wallet {wallet} has the following trust score data:\n"
        f"{json.dumps(score_data, indent=2)}\n\n"
        "Explain what this score means in 2-3 sentences. "
        "Should a system operator route a payment through this agent? Why or why not?"
    )

    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def main():
    # Example: a known active agent on Solana mainnet
    wallet = "D1QkbFJKiPsymJ65RKHhF6DFB8sPMfpBaFBzuHKfJGWi"

    print(f"Checking trust score for {wallet}...")
    score_data = await score_agent(wallet)
    print("Raw score data:", json.dumps(score_data, indent=2))

    interpretation = interpret_score(wallet, score_data)
    print("\nMistral interpretation:")
    print(interpretation)


if __name__ == "__main__":
    asyncio.run(main())
