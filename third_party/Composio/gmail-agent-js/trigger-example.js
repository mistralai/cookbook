import dotenv from "dotenv";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { LangchainToolSet } from "composio-core";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatMistralAI } from "@langchain/mistralai";

dotenv.config();

// Initialize the toolset
const toolset = new LangchainToolSet({ apiKey: process.env.COMPOSIO_API_KEY });

// Get the user
const userEntity = await toolset.client.getEntity("default");

// Set up the trigger for new emails
await userEntity.setupTrigger("gmail", "gmail_new_gmail_message", {
  interval: 10,
});

// Subscribe to the trigger and listen for new emails
toolset.client.triggers.subscribe(async (data) => {
  try {
    console.log("data received", data);
    const from = data.originalPayload.payload.headers[16].value;
    const message = data.originalPayload.snippet;
    const id = data.originalPayload.threadId;
    executeAgent("default", { from, message, id });
  } catch (error) {
    console.log("Error: ", error);
  }
});
// Execute when new emails are received
async function executeAgent(entityName, { from, message, id }) {
  try {
    // Step 1: Get the entity from the toolset
    const entity = await toolset.client.getEntity(entityName);

    // Step 2: Get the action for replying to a new email
    const tools = await toolset.getActions(
      { actions: ["gmail_reply_to_thread"] },
      entity.id
    );

    // Step 3: Create prompt to pass to the agent
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are a helpful and thorough AI email assistant who can reply to mails. Your goal is to understand the guidelines provided by the user and perform the specific actions requested by the user.",
      ],
      ["human", "{input}"],
      ["placeholder", "{agent_scratchpad}"],
    ]);

    // Step 4: Create an instance of ChatMistralAI
    const llm = new ChatMistralAI({
      model: "mistral-large-latest",
      apiKey: process.env.MISTRAL_API_KEY,
    });

    // Step 5: Prepare the input data
    const body = `This is the mail you have to respond to: ${message}. It's from ${from} and the threadId is ${id}`;
    const agent = await createToolCallingAgent({ llm, tools: tools, prompt });

    // Step 6: Create an instance of the AgentExecutor
    const agentExecutor = new AgentExecutor({ agent, tools, verbose: true });

    // Step 7: Invoke the agent
    const result = await agentExecutor.invoke({ input: body });
    console.log(result.output);
  } catch (error) {
    console.log("Error: ", error);
  }
}
