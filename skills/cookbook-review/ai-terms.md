# AI terminology guide

## General principle

Focus on what the reader wants to accomplish, not on the underlying technology. When it's necessary to discuss the technology itself—such as in developer documentation—use the terms below consistently.

Before you invent a term, make sure a suitable term doesn't already exist.

---

## Preferred terms

| Term | Usage |
|---|---|
| **AI** | Use *AI* for all audiences and in most content. Don't spell out *artificial intelligence* unless you're defining the term for the first time for a general audience. Use *intelligent* or *intelligence* to describe the benefits of AI. |
| **model** | Use *model* to refer to a Mistral language model (e.g., Mistral Large, Codestral). Don't use *AI* as a noun in place of *model* (not "the AI responded"; instead, "the model responded"). |
| **agent** | Use *agent* to refer to a pre-configured model instance with instructions and tools. Distinct from a general-purpose assistant. |
| **bot, chatbot** | Use *bot* to refer to an app that performs automated tasks or engages with humans through a conversational interface. Use *chatbot* only when you need to clarify that it uses conversation. After first use, just use *bot*. |
| **tool** | Use *tool* to refer to a function or capability the model can invoke (built-in tools, MCP Connector tools). |
| **Connector / Connectors** | Capitalize when referring to the Mistral Connectors product or feature. Lowercase *connector* only when referring to a specific registered instance ("create a connector named...", "the connector returned..."). |
| **conversation** | Use *conversation* for a session in which a model responds to a series of messages. |
| **prompt** | Use *prompt* for the input text sent to a model. In UI text, format prompts in quotation marks. |
| **context** | Use *context* or *context window* to refer to the amount of text a model can process in a single request. Don't use *memory* as a synonym unless referring to an explicit memory feature. |
| **inference** | Use *inference* for the process of generating a response from a model. Acceptable in developer content. Define on first use for general audiences. |
| **embedding** | Use *embedding* for a numerical vector representation of text. Acceptable in developer content. |
| **token** | Use *token* for the units of text the model processes. Acceptable in developer content without definition. |
| **fine-tuning** | Use *fine-tuning* (hyphenated) for the process of further training a model on a specific dataset. |
| **RAG** | Use *RAG* after spelling out *retrieval-augmented generation* on first use. Acceptable to use without spelling out in developer content. |

---

## Terms to avoid

| Avoid | Use instead | Reason |
|---|---|---|
| artificial intelligence (spelled out) | AI | Use the abbreviation; spell out only when defining for a new audience |
| the AI | the model | "AI" isn't a noun for a model instance |
| smart technology | intelligent technology / describe the capability | Too vague |
| magic, magical | describe the actual mechanism | Obscures how things work |
| knows, thinks, feels, understands (attributed to a model) | returns, generates, identifies, detects | Models don't have human cognition |
| hallucinates | generates incorrect output, produces inaccurate results | Anthropomorphizes the model |
| brain, mind (for AI) | model, system | Anthropomorphizes |
| kill switch | emergency stop, shutdown mechanism | Militaristic language |

---

## Capitalization for AI terms

- Lowercase *model*, *agent*, *tool*, *prompt*, *token*, *embedding*, *inference*, *fine-tuning*, and *context* in general use. Capitalize *Connector* and *Connectors* when referring to the Mistral product; use lowercase *connector* only for a specific registered instance.
- Capitalize proper model names: *Mistral Large*, *Mistral Small*, *Codestral*, *Pixtral*, *Mistral Embed*.
- Capitalize *Le Chat* as a proper product name.
- Capitalize *Mistral AI* when referring to the company.
- Use *AI* (uppercase) always—never *Ai* or *ai* in prose.
- Use *RAG*, *MCP*, *API*, *SDK* (uppercase abbreviations).

---

## Describing model behavior

Don't anthropomorphize models. They don't think, know, or feel—they generate, return, detect, and produce.

| Avoid | Use instead |
|---|---|
| The model knows the answer. | The model returns an answer. |
| The model thinks you want to... | The model infers from context that... |
| The model understood your request. | The model parsed your request. |
| The model made a mistake. | The model generated an incorrect response. |
| The model remembers your preferences. | The model uses context from previous messages. |
| Ask the AI. | Send a message to the model. / Call the API. |

---

## Mistral product names

Use these names exactly as written. Don't prefix them with "Mistral" or "Mistral AI" unless indicated.

| Correct | Do not use |
|---|---|
| Connectors | MCP connectors, Mistral Connectors |
| Workflows | Mistral Workflows, Mistral AI Workflows |
| Studio | Mistral Studio, AI Studio, Mistral AI Studio |
| Vibe | Le Chat |
| Vibe Code | Mistral Vibe Code |
| Vibe Work | Mistral Vibe Work |

**Connectors** — capitalize when referring to the Mistral Connectors product or feature. Lowercase *connector* only when referring to a specific registered instance.
- Correct: "Register a Connector in Studio."
- Correct: "Use Connectors to access external tools."
- Correct: "Create a connector named `github_app`." (specific instance)
- Incorrect: "Use MCP connectors to call external APIs."

**Workflows** — always standalone, never prefixed.
- Correct: "Build a pipeline in Workflows."
- Incorrect: "Use Mistral Workflows to orchestrate tasks."

**Studio** — always standalone, never prefixed.
- Correct: "Open the file in Studio."
- Incorrect: "Log in to Mistral AI Studio."

**Vibe** — the correct name for the product. Do not use *Le Chat*.
- Correct: "Send a message in Vibe."
- Incorrect: "Use Le Chat to chat with the model."

**Vibe Code** and **Vibe Work** — full product names, used as written.
- Correct: "Vibe Code integrates with your development environment."
- Correct: "Manage your team's tasks in Vibe Work."

---

## Voice and tone for AI content

- Focus on what the developer can build, not on how impressive the technology is.
- Describe capabilities in terms of outcomes: "Retrieve information from external sources" not "Leverage our powerful AI capabilities."
- Avoid superlatives: *most advanced*, *state of the art*, *groundbreaking*, *revolutionary*. Just describe what it does.
- Don't hype. Let the capability speak for itself through accurate, concrete description.
