import dotenv from "dotenv";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { LangchainToolSet, Composio } from "composio-core";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatMistralAI } from "@langchain/mistralai";

dotenv.config();

// Initialize the toolset
const toolset = new LangchainToolSet({ apiKey: process.env.COMPOSIO_API_KEY });
const trigger_toolset = new Composio();

// Get the user
const userEntity = await toolset.client.getEntity("default");

// // Set up the trigger for new emails
await userEntity.setupTrigger("gmail", "GMAIL_NEW_GMAIL_MESSAGE", {
  interval: 10,
  userId: "me",
  labelIds: ["INBOX"]
});


// Subscribe to the trigger and listen for new emails
await trigger_toolset.triggers.subscribe( 
  async(data) => {
  try {
    await console.log("data received", data);
    const from = await data.payload.sender;
    const message = await data.payload.messageText;
    const id = await data.payload.threadId;
    const res = await executeAgent("default", { from, message, id });
  } catch (error) {
    console.log("Error: ", error);
  }
}, {triggerName: "GMAIL_NEW_GMAIL_MESSAGE"});
// Execute when new emails are received
async function executeAgent(entityName, { from, message, id }) {
    // Step 1: Get the entity from the toolset
    const entity = await toolset.client.getEntity(entityName);

    // Step 2: Get the action for replying to a new email
    const tools = await toolset.getTools(
      { actions: ["GMAIL_REPLY_TO_THREAD"] },
      entity.id
    );

    // Step 3: Create prompt to pass to the agent
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are an AI email assistant called Mistral that can write and reply to emails. You also have access to Gmail Tools which let you execute actions on Gmail. Use the tool when the user asks you to.",
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
    const body = `This is the mail you have to respond to: ${message}. It's from ${from} and the threadId is ${id}
    Generate the reply based on the ${message}. Write the subject based on the reply content generated. The sender email is ${from}.
    Call the gmail tool and actions available to you. Add a subject line if there isnt any email.`;
    const agent = await createToolCallingAgent({ llm, tools: tools, prompt });

    // Step 6: Create an instance of the AgentExecutor
    const agentExecutor = new AgentExecutor({ agent, tools, verbose: true });

    // Step 7: Invoke the agent
    const result = await agentExecutor.invoke({ input: body });
    console.log(result);

}
