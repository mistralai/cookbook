# pip install agent-framework
# Use `az login` to authenticate with Azure CLI
# https://github.com/microsoft/agent-framework
import os
import asyncio
from dotenv import load_dotenv
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from azure.identity import AzureCliCredential

load_dotenv()

async def main():
    # Initialize a chat agent with Microsoft Foundry using the chat completions API
    # (Mistral models don't support the OpenAI Responses API used by FoundryChatClient)

    AZURE_AI_PROJECT_ENDPOINT = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    AZURE_AI_DEPLOYMENT_NAME = os.getenv("AZURE_AI_DEPLOYMENT_NAME")
    
    agent = Agent(
      client=OpenAIChatCompletionClient(
          model=AZURE_AI_DEPLOYMENT_NAME,
          azure_endpoint=AZURE_AI_PROJECT_ENDPOINT,
          credential=AzureCliCredential(),
      ),
      name="HaikuBot",
      instructions="You are an upbeat assistant that writes beautifully.",
    )

    print(await agent.run("Write a haiku about Microsoft Agent Framework."))

if __name__ == "__main__":
    asyncio.run(main())