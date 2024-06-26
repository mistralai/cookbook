# Mistral Cookbook

The Mistral Cookbook features examples contributed by our community and partners. If you have cool examples showcasing Mistral models, feel free to share them by submitting a PR to this repo.

## Submission Guidelines:

- File Format: Please submit your example in the .md or .ipynb format.
- Runnable on Colab: If you're sharing a notebook example, try to make sure it's runnable on Google Colab.
- Authorship: Kindly include your name and affiliation at the beginning of the file.
- Descriptions: Please include your notebook along with its category and descriptions in the table below.
- Tone: Kindly maintain a neural tone and minimize any excessive marketing materials.
- Reproducibility: To ensure others can reproduce your work, kindly tag package versions in your code.
- Image size: If you have images, please make sure each image's size is below 500KB.
- Copyright: Always respect copyright and intellectual property laws.

Disclaimer: Examples contributed by the community and partners do not represent Mistral's views and opinions.

## Content Guidelines:

- Originality: Is your content original and offering a fresh perspective?
- Clear: Is your content well-structured and clearly written?
- Value: Is your content valuable to the community? Does the community need it?

## Main Notebooks

| Notebook                                                                                                     | Category         | Description                                                                      |
| :----------------------------------------------------------------------------------------------------------- | :--------------- | :------------------------------------------------------------------------------- |
| [quickstart.ipynb](https://github.com/mistralai/cookbook/blob/main/quickstart.ipynb)                         | chat, embeddings | Basic quickstart with chat and embeddings with Mistral AI API                    |
| [prompting_capabilities.ipynb](https://github.com/mistralai/cookbook/blob/main/prompting_capabilities.ipynb) | prompting        | Write prompts for classification, summarization, personalization, and evaluation |
| [basic_RAG.ipynb](https://github.com/mistralai/cookbook/blob/main/basic_RAG.ipynb)                           | RAG              | RAG from scratch with Mistral AI API                                             |
| [embeddings.ipynb](https://github.com/mistralai/cookbook/blob/main/embeddings.ipynb)                         | embeddings       | Use Mistral embeddings API for classification and clustering                     |
| [function_calling.ipynb](https://github.com/mistralai/cookbook/blob/main/function_calling.ipynb)             | function calling | Use Mistral API for function calling                                             |
| [evaluation.ipynb](https://github.com/mistralai/cookbook/blob/main/evaluation.ipynb)                         | evaluation       | Evaluate models with Mistral API                                                 |
| [mistral_finetune_api.ipynb](https://github.com/mistralai/cookbook/blob/main/mistral_finetune_api.ipynb)     | fine-tuning      | Finetune a model with Mistral fine-tuning API                                    |
| [mistral-search-engine.ipynb](https://github.com/mistralai/cookbook/blob/main/mistral-search-engine.ipynb)   | RAG, function calling      | Search engine  built with Mistral API, function calling and RAG        |
| [prefix_use_cases.ipynb](https://github.com/mistralai/cookbook/blob/main/prefix_use_cases.ipynb)             | prefix, prompting| Cool examples with Mistral's prefix feature                                      |

## Third Party Tools

| Tools                                                                                                                                            | Category        | Party      |
| :----------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- | :--------- |
| [azure_ai_search_rag.ipynb](third_party/Azure_AI_Search/azure_ai_search_rag.ipynb) | RAG, embeddings | Azure      |
| [E2B Code Interpreter SDK with Codestral](third_party/E2B_Code_Interpreter)              | tool, agent | E2B   |
| [haystack_chat_with_docs.ipynb](third_party/Haystack/haystack_chat_with_docs.ipynb)              | RAG, embeddings | Haystack   |
| [adaptive_rag_mistral.ipynb](third_party/langchain/adaptive_rag_mistral.ipynb)                   | RAG             | Langchain  |
| [corrective_rag_mistral.ipynb](third_party/langchain/corrective_rag_mistral.ipynb)                 | RAG             | Langchain  |
| [langgraph_code_assistant_mistral.ipynb](third_party/langchain/langgraph_code_assistant_mistral.ipynb)           | code            | Langchain  |
| [langgraph_crag_mistral.ipynb](third_party/langchain/langgraph_crag_mistral.ipynb)               | RAG             | Langchain  |
| [self_rag_mistral.ipynb](third_party/langchain/self_rag_mistral.ipynb)                           | RAG             | Langchain  |
| [Adaptive_RAG.ipynb](third_party/LlamaIndex/Adaptive_RAG.ipynb)                                  | RAG             | LLamaIndex |
| [Agents_Tools.ipynb](third_party/LlamaIndex/Agents_Tools.ipynb)                                  | agent           | LLamaIndex |
| [llamaindex_agentic_rag.ipynb](third_party/LlamaIndex/llamaindex_agentic_rag.ipynb)              | RAG, agent      | LLamaIndex |
| [RAG.ipynb](third_party/LlamaIndex/RAG.ipynb)                                                    | RAG             | LLamaIndex |
| [RouterQueryEngine.ipynb](third_party/LlamaIndex/RouterQueryEngine.ipynb)                        | agent           | LLamaIndex |
| [SubQuestionQueryEngine.ipynb](third_party/LlamaIndex/RouterQueryEngine.ipynb)                   | agent           | LLamaIndex |
| [neon_text_to_sql.ipynb](third_party/Neon/neon_text_to_sql.ipynb)                                | code            | Neon       |
| [Ollama Meetup Demo](https://github.com/mistralai/cookbook/blob/main/third_party/Ollama/20240321_ollama_meetup)                                  | demo            | Ollama     |
| [function_calling_local.ipynb](third_party/Ollama/function_calling_local.ipynb)                  | tool call       | Ollama     | 
| [arize_phoenix_tracing.ipynb](third_party/Phoenix/arize_phoenix_tracing.ipynb)                   | tracing data    | Phoenix    |
| [pinecone_rag.ipynb](third_party/Pinecone/pinecone_rag.ipynb)                                    | RAG             | Pinecone   |
| [Gradio Integration - Chat with PDF](third_party/gradio/README.md)                               | demo, RAG       | Gradio     |
| [Panel Integration - Chat with PDF](third_party/panel/README.md)                                 | demo, RAG       | Panel      |
| [Streamlit Integration - Chat with PDF](third_party/streamlit/README.md)                         | demo, RAG       | Streamlit  |
| [distilabel_synthetic_dpo_dataset.ipynb](distilabel_synthetic_dpo_dataset.ipynb)                 | synthetic data  | Argilla    |