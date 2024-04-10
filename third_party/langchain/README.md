# LangChain <> MistralAI Cookbooks

Many LLM applications can be broken down into a "control flow": 

* Break down the problem into small steps. 
* Use steps to check intermediate solution(s), and re-try / course correct if needed.

This idea of [flow engineering](https://twitter.com/karpathy/status/1748043513156272416?s=20) to build answers iteratively has shown strong results for code generation.

Several recent papers apply this same idea to RAG (retrieval augmented generation), making it more robust to errors:

* Corrective-RAG (CRAG) [paper](https://arxiv.org/pdf/2401.15884.pdf) uses self-grading on retrieved documents and web-search fallback if documents are not relevant.
* Self-RAG [paper](https://arxiv.org/abs/2310.11511) adds self-grading on generations for hallucinations and for ability to answer the question.
* Adaptive RAG [paper](https://arxiv.org/abs/2403.14403) routes queries between different RAG approaches based on their complexity.

In 3 notebooks, we implement each approach as a control flow using [LangGraph](https://python.langchain.com/docs/langgraph):

* We use a graph to represent the control flow
* The graph state includes information (question, documents, etc) that we want to pass between nodes 
* Each graph node modifies the state in a specific way (e.g., adds documents from retrieval, performs grading, etc)
* Each graph edge decides which node to visit next (e.g., decide web search or vectorstore retrieval based on the question)

In the 3 notebooks, we will build from CRAG (blue, below) to Self-RAG (green) and finally to Adaptive RAG (red):

![RAG](./img/langgraph_adaptive_rag.png "RAG control flow")

Each notebook builds on the prior one, so CRAG is a good entry point.

A significant benefit of this approach is that the flows can be run with smaller (e.g., locally running) LLMs:

* The LLM has a specific job within each node and this constrained scope improves reliability.
* See notebooks [here](https://github.com/langchain-ai/langgraph/tree/main/examples/rag) for running each of these locally with `Mistral-7b`.