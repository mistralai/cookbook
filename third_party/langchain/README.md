# LangChain <> MistralAI Cookbooks

LLM agents use [planning, memory, and tools](https://lilianweng.github.io/posts/2023-06-23-agent/) to accomplish tasks. [LangGraph](https://python.langchain.com/docs/langgraph) is a library from LangChain that can be used to build reliable agents and workflows.

### Code generation

We'll combine the code generation capabilities of Codestral the self-correction approach presented in the [AlphaCodium](https://github.com/Codium-ai/AlphaCodium) paper, [constructing an answer to a coding question iteratively](https://x.com/karpathy/status/1748043513156272416?s=20).  

We will implement some of these ideas from scratch using [LangGraph](https://python.langchain.com/docs/langgraph) to 1) produce structured code generation output from Codestral-instruct, 2) perform inline unit tests to confirm imports and code execution work, 3) feed back any errors for Codestral for self-correction.

<img width="1029" alt="Screenshot 2024-05-29 at 7 03 29 AM" src="https://github.com/rlancemartin/mistral-cookbook/assets/122662504/4ad2c6f3-2fe5-4a0a-b33c-02f489170c0a">

Video overview:

* https://youtu.be/zXFxmI9f06M

--- 

### RAG

We'll apply LangGraph to build RAG agents that use ideas from 3 papers:

* Corrective-RAG (CRAG) [paper](https://arxiv.org/pdf/2401.15884.pdf) uses self-grading on retrieved documents and web-search fallback if documents are not relevant.
* Self-RAG [paper](https://arxiv.org/abs/2310.11511) adds self-grading on generations for hallucinations and for ability to answer the question.
* Adaptive RAG [paper](https://arxiv.org/abs/2403.14403) routes queries between different RAG approaches based on their complexity.

We implement each approach as a control flow in LangGraph:
- **Planning:** The sequence of RAG steps (e.g., retrieval, grading, generation) that we want the agent to take
- **Memory:** All the RAG-related information (input question, retrieved documents, etc) that we want to pass between steps
- **Tool use:** All the tools needed for RAG (e.g., decide web search or vectorstore retrieval based on the question)

In the 3 notebooks, we will build from CRAG (blue, below) to Self-RAG (green) and finally to Adaptive RAG (red):

![RAG](./img/langgraph_adaptive_rag.png "RAG control flow")

Each notebook builds on the prior one, so CRAG is a good entry point.

Video overview:

* https://www.youtube.com/watch?v=sgnrL7yo1TE
