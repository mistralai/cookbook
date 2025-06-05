# Building an AI Email Assistant with Mistral AI and Composio

This notebook demonstrates how to build an AI-powered email assistant using [Mistral AI](https://mistral.ai/) and [Composio](https://app.composio.dev/). By using Mistral’s large language models (LLMs) and Composio’s Gmail tool, we can create a system capable of performing common Gmail actions such as sending, fetching, drafting emails etc., all through natural language commands.

In this guide, we'll create an AI email assistant that can automatically reply to received emails!

To work with Python Jupyter Notebooks in VSCode, activate an Anaconda environment or another Python environment in which you've installed the Jupyter package. You can run an individual cell using the Run icon and the output will be displayed below the code cell.

The script largely performs the following steps- 

- The API key(s) are loaded in.
- User connects their account and creates an integration.
- Initializes Composio's toolset and actions/triggers associated with Gmail.
- Initializes Mistral's LLMs with custom prompt.
- Initializes triggers and registers listeners.
- When trigger gets new data, callback function is triggered. Specified actions are executed with help from the LLM. 

The provided examples can be modified by adding more tools and customizing the prompts based on your needs.

## Connect with Composio and learn more

If you encounter any problems, please let us know at out [Discord](https://discord.com/invite/cNruWaAhQk).

Check [Composio docs](https://docs.composio.dev/introduction/intro/overview) to learn more about how to use and integrate various tools for different usecases.