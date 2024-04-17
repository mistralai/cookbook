# LangChain <> MistralAI Cookbooks

LLM agents use [planning, memory, and tools](https://lilianweng.github.io/posts/2023-06-23-agent/) to accomplish tasks.

[LangGraph](https://python.langchain.com/docs/langgraph) is a library from LangChain that can be used to build reliable agents.

LangGraph can be used to build agents with a few pieces:
- **Planning:** Define a control flow of steps that you want the agent to take (a graph)
- **Memory:** Persist information (graph state) across these steps
- **Tool use:** Tools can be used at any step to modify state

To make this concrete, we will apply LangGraph to build RAG agents that use ideas from 3 papers:

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