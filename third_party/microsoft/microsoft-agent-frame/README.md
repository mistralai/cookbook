# Microsoft Agent Framework — CookBook

Hands-on notebooks and scripts for building agents and agentic workflows with the
**Microsoft Agent Framework** on **Azure AI Foundry**, using **Mistral** models.

---

## Contents

| File | Description |
|------|-------------|
| [`BasicAgent.py`](BasicAgent.py) | Minimal runnable script — one agent, one question |
| [`Agent_Framework_Demo.ipynb`](Agent_Framework_Demo.ipynb) | Six progressive patterns: basic → tools → MCP → sessions → memory → workflows |
| [`Foundry_Agent_Tool_Calling.ipynb`](Foundry_Agent_Tool_Calling.ipynb) | Foundry agent with a remote MCP tool server (GitHub API) |
| [`OCR-RAG-Agentic-Workflow.ipynb`](OCR-RAG-Agentic-Workflow.ipynb) | Three-stage pipeline: PDF OCR → vector index → RAG answer |

---

## Setup

1. Copy `env.example` to `.env` and fill in your values.
2. Log in with the Azure CLI: `az login`
3. Install dependencies (each notebook has a `%pip install` cell at the top).

### Required environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `AZURE_AI_PROJECT_ENDPOINT` | all | Azure AI Foundry project URL |
| `AZURE_AI_DEPLOYMENT_NAME` | Demo, Foundry | Deployed Mistral model name |
| `AZURE_AI_QNA_MODEL` | OCR-RAG | Model for the answering stage |
| `AZURE_AI_OCR_NAME` | OCR-RAG | Mistral OCR deployment name |
| `MISTRAL_OCR_ENDPOINT` | OCR-RAG | Mistral OCR REST endpoint |
| `MISTRAL_API_KEY` | OCR-RAG | API key for OCR + embeddings |
| `AZURE_MISTRAL_EMBEDDING_MODEL` | OCR-RAG | Embedding model (`mistral-embed`) |
| `AZURE_SEARCH_ENDPOINT` | OCR-RAG | Azure AI Search service URL |
| `AZURE_SEARCH_INDEX_NAME` | OCR-RAG | Index to create or reuse |
| `AZURE_SEARCH_API_KEY` | OCR-RAG | Optional — falls back to CLI auth |
| `AZURE_SEARCH_SEMANTIC_CONFIG` | OCR-RAG | Optional semantic ranker config |
| `MCP_SERVER_URL` | Foundry | Hosted MCP server HTTPS URL |
| `MCP_CONNECTION_NAME` | Foundry | Foundry connection ID for MCP credentials |
| `PROJECT_API_NAME` | Foundry | API path suffix for `AIProjectClient` |

---

## Architecture Diagrams

### 1 — BasicAgent.py

The simplest possible pattern: one agent, one synchronous call.

```mermaid
flowchart LR
    User["User prompt"] --> Agent
    subgraph Agent Framework
        Agent["Agent\n(HaikuBot)"] --> Client["OpenAIChatCompletionClient"]
    end
    Client -->|"Chat Completions API"| Foundry["Azure AI Foundry\n(Mistral Large 3)"]
    Foundry --> Agent
    Agent --> Response["Printed response"]
```

---

### 2 — Agent_Framework_Demo.ipynb

Six patterns in one notebook, building up from a basic agent to a full workflow.

#### 2a — Basic agent (non-streaming & streaming)

```mermaid
flowchart LR
    Prompt["User prompt"] --> Agent["Agent"]
    Agent -->|"Chat Completions"| Foundry["Azure AI Foundry\n(Mistral Large 3)"]
    Foundry --> Agent
    Agent -->|"result.text"| Out["Response"]

    Prompt2["User prompt"] --> Agent2["Agent\nstream=True"]
    Agent2 -->|"Chat Completions\n(streaming)"| Foundry
    Foundry -->|"token chunks"| Agent2
    Agent2 -->|"AgentResponseUpdate chunks"| Stream["Streamed output"]
```

#### 2b — Tool calling

```mermaid
flowchart TD
    Q["What's the weather in Seattle?"] --> Agent["Agent\n(WeatherAgent)"]
    Agent -->|"1 decide to call tool"| Schema["@tool get_weather\n(JSON schema)"]
    Schema -->|"2 execute locally"| Fn["get_weather(location)"]
    Fn -->|"3 tool result"| Agent
    Agent -->|"4 compose reply\nwith Chat Completions"| Foundry["Azure AI Foundry\n(Mistral Large 3)"]
    Foundry --> Agent
    Agent --> Answer["Natural-language answer"]
```

#### 2c — Multi-turn sessions

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant S as Session (history)
    participant F as Foundry LLM

    U->>A: "My name is Alice and I love hiking."
    A->>S: append user message
    A->>F: messages = [system + history + user]
    F-->>A: reply
    A->>S: append assistant message

    U->>A: "What did I just tell you?"
    A->>S: append user message
    A->>F: messages = [system + full history]
    F-->>A: "You said your name is Alice..."
    A->>S: append assistant message
    A-->>U: reply
```

#### 2d — Memory via ContextProvider

```mermaid
flowchart TD
    subgraph "Each agent.run() call"
        direction TB
        BR["before_run()\nread state → extend_instructions()"]
        LLM["LLM call\n(Mistral Large 3)"]
        AR["after_run()\nparse 'my name is' → write state"]
    end

    State[("Session state\n{user_name: ...}")] -->|"inject name"| BR
    BR --> LLM
    LLM --> AR
    AR -->|"persist name"| State
```

#### 2e — Workflow (UpperCase → reverse_text)

```mermaid
flowchart LR
    Input["'hello world'"] --> UC

    subgraph Workflow
        UC["UpperCase\n(class-based Executor)\ntext.upper()"]
        -->|"ctx.send_message"| RT["reverse_text\n(@executor function)\ntext reversed"]
    end

    RT -->|"ctx.yield_output"| Output["'DLROW OLLEH'"]
```

---

### 3 — Foundry_Agent_Tool_Calling.ipynb

A Foundry-managed agent uses a remote **Model Context Protocol (MCP)** server to call GitHub APIs across a multi-turn conversation.

```mermaid
flowchart TD
    Dev["Developer\n(notebook)"]

    subgraph Azure AI Foundry
        AC["AIProjectClient"]
        FAgent["PromptAgentDefinition\n(Mistral Medium 3.5)\n+ MCPTool"]
        Conv["Conversation\n(server-side history)"]
    end

    subgraph MCP Server ["MCP Server (hosted in Foundry)"]
        Tool1["get_me\n(GitHub profile)"]
        Tool2["repo summary\n(GitHub API)"]
    end

    Dev -->|"create_version()"| AC
    AC --> FAgent
    Dev -->|"conversations.create()"| Conv
    Dev -->|"responses.create(input='...')"| Conv
    Conv --> FAgent
    FAgent -->|"invoke MCP tool"| Tool1
    Tool1 -->|"tool result"| FAgent
    FAgent -->|"compose reply"| Conv
    Conv -->|"response.output_text"| Dev

    Dev -->|"second turn"| Conv
    FAgent -->|"invoke MCP tool"| Tool2
    Tool2 --> FAgent

    Dev -->|"delete_version()"| AC
```

---

### 4 — OCR-RAG-Agentic-Workflow.ipynb

A three-stage linear workflow. Each executor has one responsibility; the Agent Framework routes state between them automatically.

```mermaid
flowchart TD
    PDF["table.png.pdf\n(local file)"]

    subgraph Stage1 ["Stage 1 — OcrExecutor"]
        B64["base64-encode PDF"]
        OCR["POST to Mistral OCR\n(Azure Foundry endpoint)"]
        MD["Markdown text\n(per-page joined)"]
        B64 --> OCR --> MD
    end

    subgraph Stage2 ["Stage 2 — KnowledgeIndexExecutor"]
        Embed["MistralEmbeddingClient\nmistral-embed → 1024-d vector"]
        IdxCheck{"Index exists?"}
        Create["Create HNSW index\n(Azure AI Search)"]
        Upload["Upload document\n{id, source, content, contentVector}"]
        Embed --> IdxCheck
        IdxCheck -->|"no"| Create --> Upload
        IdxCheck -->|"yes"| Upload
    end

    subgraph Stage3 ["Stage 3 — AnswerExecutor"]
        VecSearch["AzureAISearchContextProvider\nvector similarity → top-3 passages"]
        LLM["OpenAIChatCompletionClient\n(Mistral via Foundry)\nAnswer from retrieved context"]
        VecSearch --> LLM
    end

    PDF --> Stage1
    Stage1 -->|"state.ocr_text\nctx.send_message()"| Stage2
    Stage2 -->|"state\nctx.send_message()"| Stage3
    Stage3 -->|"ctx.yield_output()\nstate.answer"| Answer["Printed answer\n'The sum of unpaid\nbalances is ...'"]

    Query["Query:\n'what's the sum of unpaid\nbalance in the invoice'"] --> Stage3
```

#### Data flow summary

```
table.png.pdf
      │
      ▼  base64 + HTTP POST
 Mistral OCR  ──────────────────────────► Markdown text
                                               │
                                               ▼  mistral-embed (1024-d)
                                        Azure AI Search index
                                               │
                                               ▼  vector similarity (top-3)
 Query ──────────────────────────────► Grounding context
                                               │
                                               ▼  Chat Completions
                                        Mistral (Foundry)
                                               │
                                               ▼
                                         Final answer
```

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| `Agent` | Wraps an LLM client with instructions, tools, and context providers |
| `OpenAIChatCompletionClient` | Connects to any OpenAI-compatible endpoint (e.g. Azure AI Foundry) |
| `@tool` | Decorator that exposes a Python function as a callable tool for the LLM |
| `MCPTool` | Declares a remote Model Context Protocol server as a tool source |
| `Session` | Carries conversation history across multiple `agent.run()` calls |
| `ContextProvider` | Injects dynamic system-prompt content and reads responses each turn |
| `Executor` / `@executor` | A single stage in a workflow — receives state, transforms it, forwards it |
| `WorkflowBuilder` | Wires executors into a directed graph and builds a runnable workflow |
| `ctx.send_message()` | Forwards state to the next executor in the pipeline |
| `ctx.yield_output()` | Marks the terminal result of a workflow |

---

## References

- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- [Azure AI Foundry documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/)
- [Mistral on Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-foundry/models/mistral)
- [Azure AI Foundry MCP tools](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/tools/model-context-protocol?pivots=python)
- [Azure AI Search vector search](https://learn.microsoft.com/en-us/azure/search/vector-search-overview)
