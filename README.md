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
| [rag_via_function_calling.ipynb](mistral/rag/RAG_via_function_calling.ipynb)         | RAG, function calling        | Use function calling as a router for a RAG based on multiple data sources                   |
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

---

*This is a test README for PR purposes - please ignore*
