# Mistral Cookbook

The Mistral Cookbook features examples contributed by Mistralers and our community, as well as our partners. If you have cool examples showcasing Mistral models, feel free to share them by submitting a PR to this repo.

## Submission Guidelines:

- File Format: Please submit your example in the .md or .ipynb format.
- Runnable on Colab: If you're sharing a notebook example, try to make sure it's runnable on Google Colab.
- Authorship: Kindly include your name, your Github handle, and affiliation at the beginning of the file.
- Descriptions: Please include your notebook along with its category and descriptions in the table below.
- Tone: Kindly maintain a neutral tone and minimize any excessive marketing materials.
- Reproducibility: To ensure others can reproduce your work, kindly tag package versions in your code.
- Image size: If you have images, please make sure each image's size is below 500KB.
- Copyright: Always respect copyright and intellectual property laws.

Disclaimer: Examples contributed by the community and partners do not represent Mistral's views and opinions.

## Content Guidelines:

- Originality: Is your content original and offering a fresh perspective?
- Clear: Is your content well-structured and clearly written?
- Value: Is your content valuable to the community? Does the community need it?

## Main Notebooks

| Notebook                                                                       | Category                     | Description                                                                      |
|--------------------------------------------------------------------------------|-----------------------------|----------------------------------------------------------------------------------|
| [quickstart.ipynb](quickstart.ipynb)                                           | chat, embeddings             | Basic quickstart with chat and embeddings with Mistral AI API                    |
| [prompting_capabilities.ipynb](mistral/prompting/prompting_capabilities.ipynb) | prompting                    | Write prompts for classification, summarization, personalization, and evaluation |
| [basic_RAG.ipynb](mistral/rag/basic_RAG.ipynb)                                 | RAG                          | RAG from scratch with Mistral AI API                                             |
| [embeddings.ipynb](mistral/embeddings/embeddings.ipynb)                        | embeddings                   | Use Mistral embeddings API for classification and clustering                     |                                           |
| [function_calling.ipynb](mistral/function_calling/function_calling.ipynb)      | function calling             | Use Mistral API for function calling                                             |
| [text_to_SQL.ipynb](mistral/function_calling/text_to_SQL.ipynb)      | function calling             | Use Mistral API for function calling on a multi tables text to SQL usecase                                             |
| [evaluation.ipynb](mistral/evaluation/evaluation.ipynb)                        | evaluation                   | Evaluate models with Mistral API                                                 |
| [mistral_finetune_api.ipynb](mistral/fine_tune/mistral_finetune_api.ipynb)     | fine-tuning                  | Finetune a model with Mistral fine-tuning API                                    |
| [mistral-search-engine.ipynb](mistral/rag/mistral-search-engine.ipynb)         | RAG, function calling        | Search engine built with Mistral API, function calling and RAG                   |
| [rag_via_function_calling.ipynb](mistral/rag/rag_via_function_calling.ipynb)         | RAG, function calling        | Use function calling as a router for a RAG based on multiple data sources                   |
| [prefix_use_cases.ipynb](mistral/prompting/prefix_use_cases.ipynb)             | prefix, prompting            | Cool examples with Mistral's prefix feature                                      |
| [synthetic_data_gen_and_finetune.ipynb](mistral/data_generation/synthetic_data_gen_and_finetune.ipynb) | data generation, fine-tuning | Simple data generation and fine-tuning guide        |
| [data_generation_refining_news.ipynb](mistral/data_generation/data_generation_refining_news.ipynb) | data generation | Simple data generation to refine news articles                                |
| [image_description_extraction_pixtral.ipynb](mistral/image_understanding/image_description_extraction_pixtral.ipynb) | image processing, prompting  | Extract structured image descriptions using Mistral's Pixtral model and JSON response formatting |
| [multimodality meets function calling.ipynb](mistral/image_understanding/multimodality_meets_function_calling.ipynb) | image processing, function calling  | Extract table from image using Mistral's Pixtral model and use for function calling |
| [mistral-reference-rag.ipynb](mistral/rag/mistral-reference-rag.ipynb) | RAG, function calling, references | Reference RAG with Mistral API |
| [moderation-explored.ipynb](mistral/moderation/moderation-explored.ipynb) | moderation | Quick exploration on safeguarding and Mistral's moderation API |
| [system-level-guardrails.ipynb](mistral/moderation/system-level-guardrails.ipynb) | moderation | How to implement System Level Guardrails with Mistral API |
| [document_understanding.ipynb](mistral/ocr/document_understanding.ipynb) | OCR, function calling | Document Understanding and Tool Usage with OCR |
| [batch_ocr.ipynb](mistral/ocr/batch_ocr.ipynb) | OCR, batch | Using OCR to extract text data from datasets. |
| [structured_ocr.ipynb](mistral/ocr/structured_ocr.ipynb) | OCR, structured outputs | Extracting structured outputs from documents. |
| [RAG_evaluation.ipynb](mistral/evaluation/RAG_evaluation.ipynb) | evaluation, structured outputs, LLM As a Judge | Evaluate RAG with LLM as a Judge and structured outputs |
| [product_classification.ipynb](mistral/classifier_factory/product_classification.ipynb) | fine-tuning, classifier | Fine-tuning a classifier for food classification. |
| [intent_classification.ipynb](mistral/classifier_factory/intent_classification.ipynb) | fine-tuning, classifier | Fine-tuning a classifier for intent classification. |
| [moderation_classifier.ipynb](mistral/classifier_factory/moderation_classifier.ipynb) | fine-tuning, classifier | Fine-tuning a classifier for moderation. |
| [pixtral_finetune_on_satellite_data.ipynb](mistral/fine_tune/pixtral_finetune_on_satellite_data.ipynb) | fine-tuning, image processing, batch | Fine-tuning a Pixtral-12B for satellite images classification. |



## Third Party Tools

| Tools                                                                                                                 | Category                        | Party      |
|:----------------------------------------------------------------------------------------------------------------------|:--------------------------------| :--------- |
| [adaptive_rag_mistral.ipynb](third_party/langchain/adaptive_rag_mistral.ipynb)                                        | RAG                             | Langchain  |
| [Adaptive_RAG.ipynb](third_party/LlamaIndex/Adaptive_RAG.ipynb)                                                       | RAG                             | LLamaIndex |
| [Agents_Tools.ipynb](third_party/LlamaIndex/Agents_Tools.ipynb)                                                       | agent                           | LLamaIndex |
| [arize_phoenix_tracing.ipynb](third_party/Phoenix/arize_phoenix_tracing.ipynb)                                        | tracing data                    | Arize Phoenix  |
| [arize_phoenix_evaluate_rag.ipynb](third_party/Phoenix/arize_phoenix_evaluate_rag.ipynb)                              | evaluation                      | Arize Phoenix  |
| [azure_ai_search_rag.ipynb](third_party/Azure_AI_Search/azure_ai_search_rag.ipynb)                                    | RAG, embeddings                 | Azure      |
| [CAMEL Graph RAG with Mistral Models](third_party/CAMEL_AI/camel_graph_rag.ipynb)                                     | multi-agent, tool, data gen     | CAMEL-AI.org|
| [CAMEL Role-Playing Scraper](third_party/CAMEL_AI/camel_roleplaying_scraper.ipynb)                                    | multi-agent, tool, data gen     | CAMEL-AI.org|
| [Chainlit - Mistral reasoning.ipynb](third_party/Chainlit/Chainlit_Mistral_reasoning.ipynb)                           | UI chat, tool calling           | Chainlit   |
| [corrective_rag_mistral.ipynb](third_party/langchain/corrective_rag_mistral.ipynb)                                    | RAG                             | Langchain  |
| [distilabel_synthetic_dpo_dataset.ipynb](third_party/argilla/distilabel_synthetic_dpo_dataset.ipynb)                  | synthetic data                  | Argilla    |
| [E2B Code Interpreter SDK with Codestral](third_party/E2B_Code_Interpreting)                                          | tool, agent                     | E2B        |
| [function_calling_local.ipynb](third_party/Ollama/function_calling_local.ipynb)                                       | tool call                       | Ollama     |
| [Gradio Integration - Chat with PDF](third_party/gradio/README.md)                                                    | UI chat, demo, RAG              | Gradio     |
| [haystack_chat_with_docs.ipynb](third_party/Haystack/haystack_chat_with_docs.ipynb)                                   | RAG, embeddings                 | Haystack   |
| [Indexify Integration - PDF Entity Extraction](third_party/Indexify/pdf-entity-extraction)                            | entity extraction, PDF          | Indexify   |
| [Indexify Integration - PDF Summarization](third_party/Indexify/pdf-summarization)                                    | summarization, PDF              | Indexify   |
| [langgraph_code_assistant_mistral.ipynb](third_party/langchain/langgraph_code_assistant_mistral.ipynb)                | code                            | Langchain  |
| [langgraph_crag_mistral.ipynb](third_party/langchain/langgraph_crag_mistral.ipynb)                                    | RAG                             | Langchain  |
| [langtrace_mistral.ipynb](third_party/langtrace/langtrace_mistral.ipynb)                                              | OTEL Observability              | Langtrace  |
| [llamaindex_agentic_rag.ipynb](third_party/LlamaIndex/llamaindex_agentic_rag.ipynb)                                   | RAG, agent                      | LLamaIndex |
| [llamaindex_arxiv_agentic_rag.ipynb](third_party/LlamaIndex/llamaindex_arxiv_agentic_rag.ipynb)                       | RAG, agent, Arxiv summarization | LLamaIndex |
| [llamaindex_mistralai_finetuning.ipynb](third_party/LlamaIndex/llamaindex_mistralai_finetuning.ipynb)                 | fine-tuning                     | LLamaIndex |
| [llamaindex_mistral_multi_modal.ipynb](third_party/LlamaIndex/llamaindex_mistral_multi_modal.ipynb)                   | MultiModalLLM-Pixtral           | LLamaIndex |
| [Maxim AI - Observability ](third_party/Maxim/cookbook_maxim_mistral_integration.ipynb) | Observability, Evaluation           | Maxim AI |
| [Microsoft Autogen - Function calling a pgsql db ](third_party/MS_Autogen_pgsql/mistral_pgsql_function_calling.ipynb) | Tool call, agent, RAG           | Ms Autogen |
| [Mesop Integration - Chat with PDF](third_party/mesop/README.md)                                                      | UI chat, demo, RAG              | Mesop      |
| [Monitoring Mistral AI using OpenTelemetry](third_party/openlit/cookbook_mistral_opentelemetry.ipynb)                 | AI Observability                | OpenLIT    |
| [neon_text_to_sql.ipynb](third_party/Neon/neon_text_to_sql.ipynb)                                                     | code                            | Neon       |
| [ollama_mistral_llamaindex.ipynb](third_party/LlamaIndex/ollama_mistral_llamaindex.ipynb)                             | RAG                             | LLamaIndex |
| [Ollama Meetup Demo](third_party/Ollama/20240321_ollama_meetup)                                                       | demo                            | Ollama     |
| [Open-source LLM engineering](third_party/Langfuse)                                                                   | LLM Observability               | Langfuse   |
| [Panel Integration - Chat with PDF](third_party/panel/README.md)                                                      | UI chat, demo, RAG              | Panel      |
| [phospho integration](third_party/phospho/cookbook_phospho_mistral_integration.ipynb)                                 | Evaluation, Analytics           | phospho    |
| [pinecone_rag.ipynb](third_party/Pinecone/pinecone_rag.ipynb)                                                         | RAG                             | Pinecone   |
| [RAG.ipynb](third_party/LlamaIndex/RAG.ipynb)                                                                         | RAG                             | LLamaIndex |
| [RouterQueryEngine.ipynb](third_party/LlamaIndex/RouterQueryEngine.ipynb)                                             | agent                           | LLamaIndex |
| [self_rag_mistral.ipynb](third_party/langchain/self_rag_mistral.ipynb)                                                | RAG                             | Langchain  |
| [Solara Integration - Chat with PDFs](third_party/solara/README.md)                                                   | UI chat, demo, RAG              | Solara     |
| [Streamlit Integration - Chat with PDF](third_party/streamlit/README.md)                                              | UI chat, demo, RAG              | Streamlit  |
| [Neo4j rag](third_party/Neo4j/neo4j_rag.ipynb)                                                                        | RAG                             | Neo4j      |
| [SubQuestionQueryEngine.ipynb](third_party/LlamaIndex/RouterQueryEngine.ipynb)                                        | agent                           | LLamaIndex |
| [LLM Judge: Detecting hallucinations in language models](third_party/wandb/README.md)                                 | fine-tuning, evaluation         | Weights & Biases |
| [`x mistral`: CLI & TUI APP Module in X-CMD](third_party/x-cmd/README.md)                                             | CLI, TUI APP, Chat              | x-cmd      |
| [Incremental Prompt Engineering and Model Comparison](third_party/Pixeltable/README.md)                               | Prompt Engineering, Evaluation  | Pixeltable |
| [Build a bank support agent with Pydantic AI and Mistral AI](third_party/PydanticAI/pydantic_bank_support_agent.ipynb)| Agent                           | Pydantic   |
| [Mistral and MLflow Tracing](third_party/MLflow/mistral-mlflow-tracing.ipynb)                                         | Tracing, Observability          | MLflow     |
| [Mistral OCR with Gradio](third_party/gradio/MistralOCR.md)                                                           | OCR                             | Gradio     |
| [european_company_enrichment_serper.ipynb](mistral/data_enrichment/european_company_enrichment_serper.ipynb)          | data enrichment, web search     | Serper     |
