# LangChain <> MistralAI Cookbooks

Many LLM applications can be broken down into a "control flow": 

* Break down the problem into small, logical steps. 
* Think of these steps as functions or blocks of code that perform a specific task.
* Use conditional logic to route between steps, moving towards the solution.

Several recent papers have applies this approach to RAG (retrieval augmented generation), making it more robust to errors:

* Corrective-RAG (CRAG) [paper](https://arxiv.org/pdf/2401.15884.pdf) uses self-grading on retrieved documents and add web-search if any documents are not relevant to the question.
* Self-RAG [paper](https://arxiv.org/abs/2310.11511) adds self-grading on generations for hallucinations and whether it answers the question.
* Adaptive RAG [paper](https://arxiv.org/abs/2403.14403) routes queries between different RAG approaches based on their complexitity.

In these 3 notebooks, we implement each approach as a control flow using [LangGraph](https://python.langchain.com/docs/langgraph):

* We use a graph to represent the control flow
* The graph state includes information (question, documents, etc) that we want to pass between nodes 
* Each graph node modifies the state in a specific way (e.g., adds documents from retrieval)
* Each graph edge decides which node to visit next (e.g., decide web search or vectorstore retrieval based on the question)

In the 3 notebooks, we will build from CRAG (blue, below) to Self-RAG (green) and finally to Adaptive RAG (red), gradually adding more components to our flow. 

Using LangGraph to model the flow has trade-offs relative to an typical "agent" implementation:

* Pro: We have found higher reliability because the structure of the flow is set up-front and the LLM's role are constrained at each node
* Con: The graph sequence is set, whereas typical "agent" implementation can choose an arbirary sequence of steps to solve a problem