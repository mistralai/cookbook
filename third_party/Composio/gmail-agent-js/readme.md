# Building an AI Email Assistant with Mistral AI and Composio

This guide demonstrates how to build an AI-powered email assistant using Mistral AI and Composio. By using Mistral’s large language models (LLMs) and Composio’s Gmail tool, we can create a system capable of performing common Gmail actions such as sending, replying to and drafting emails etc., all through natural language commands.

This guide provides a couple of examples of how Composio's tools can be used to perform these tasks. 

## Setup and Dependencies

### Install dependencies - 

To install all dependencies, run: 

```
npm install
```

### Set up environment variables - 

Create a .env file in the root directory and add your API keys:

Get a Mistral API key.
- Add the Mistral API key to the .env file

Get a Composio API key.
- Add the Composio API key to the .env file

### Run an example - 

To run `email-send-example.js`, which allows the agent to send, fetch and draft emails based on user input, run -

```
npm run start
```

To run `trigger-example.js`, which allows the agent to automatically respond to new emails, run - 

```
npm run trigger
```

The scripts largely perform the following steps- 

- The API keys are loaded from the environment variables.
- User connection is set up.
- Initializes Composio's toolset and actions/triggers associated with Gmail.
- Initializes Mistral's LLMs with custom prompt.
- Takes user input as task, the LLM populates the provided JSON schema and returns the output.

The provided examples can be modified by adding more tools and customizing the prompts based on your needs. 

## Connect with Composio and learn more

If you encounter any problems, please let us know at out [Discord](https://discord.com/invite/cNruWaAhQk).

Check [Composio docs](https://docs.composio.dev/introduction/intro/overview) to learn more about how to use and integrate various tools for different usecases.