# PDF Entity Extraction with Indexify and Mistral

This cookbook demonstrates how to build a robust entity extraction pipeline for PDF documents using Indexify and Mistral's large language models. You will learn how to efficiently extract named entities from PDF files for various applications such as information retrieval, content analysis, and data mining.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
   - [Install Indexify](#install-indexify)
   - [Install Required Extractors](#install-required-extractors)
4. [Creating the Extraction Graph](#creating-the-extraction-graph)
5. [Implementing the Entity Extraction Pipeline](#implementing-the-entity-extraction-pipeline)
6. [Running the Entity Extraction](#running-the-entity-extraction)
7. [Customization and Advanced Usage](#customization-and-advanced-usage)
8. [Conclusion](#conclusion)

## Introduction

Entity extraction, also known as named entity recognition (NER) involves identifying and classifying named entities in text into predefined categories such as persons, organizations, locations, dates, and more. By applying this technique to PDF documents, we can automatically extract structured information from unstructured text, making it easier to analyze and utilize the content of these documents.

## Prerequisites

Before we begin, ensure you have the following:

- Create a virtual env with Python 3.9 or later
  ```shell
  python3.9 -m venv ve
  source ve/bin/activate
  ```
- `pip` (Python package manager)
- A Mistral API key
- Basic familiarity with Python and command-line interfaces

## Setup

### Install Indexify

First, let's install Indexify using the official installation script:

```bash
curl https://getindexify.ai | sh
```

Start the Indexify server:
```bash
./indexify server -d
```
This starts a long running server that exposes ingestion and retrieval APIs to applications.

### Install Required Extractors

Next, we'll install the necessary extractors in a new terminal:

```bash
pip install indexify-extractor-sdk
indexify-extractor download tensorlake/pdfextractor
indexify-extractor download tensorlake/mistral
```

Once the extractors are downloaded, you can start them:
```bash
indexify-extractor join-server
```

## Creating the Extraction Graph

The extraction graph defines the flow of data through our entity extraction pipeline. We'll create a graph that first extracts text from PDFs, then sends that text to Mistral for entity extraction.

Create a new Python file called `pdf_entity_extraction_pipeline.py` and add the following code:

```python
from indexify import IndexifyClient, ExtractionGraph

client = IndexifyClient()

extraction_graph_spec = """
name: 'pdf_entity_extractor'
extraction_policies:
  - extractor: 'tensorlake/pdfextractor'
    name: 'pdf_to_text'
  - extractor: 'tensorlake/mistral'
    name: 'text_to_entities'
    input_params:
      model_name: 'mistral-large-latest'
      key: 'YOUR_MISTRAL_API_KEY'
      system_prompt: 'Extract and categorize all named entities from the following text. Provide the results in a JSON format with categories: persons, organizations, locations, dates, and miscellaneous.'
    content_source: 'pdf_to_text'
"""

extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
client.create_extraction_graph(extraction_graph)
```

Replace `'YOUR_MISTRAL_API_KEY'` with your actual Mistral API key.

You can run this script to set up the pipeline:
```bash
python pdf_entity_extraction_pipeline.py
```

## Implementing the Entity Extraction Pipeline

Now that we have our extraction graph set up, we can upload files and retrieve the entities:

Create a file `upload_and_retreive.py`

```python
import json
import os
import requests
from indexify import IndexifyClient

def download_pdf(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"PDF downloaded and saved to {save_path}")


def extract_entities_from_pdf(pdf_path):
    client = IndexifyClient()
    
    # Upload the PDF file
    content_id = client.upload_file("pdf_entity_extractor", pdf_path)
    
    # Wait for the extraction to complete
    client.wait_for_extraction(content_id)
    
    # Retrieve the extracted entities
    entities_content = client.get_extracted_content(
        content_id=content_id,
        graph_name="pdf_entity_extractor",
        policy_name="text_to_entities"
    )
    
    # Parse the JSON response
    entities = json.loads(entities_content[0]['content'].decode('utf-8'))
    return entities

# Example usage
if __name__ == "__main__":
    pdf_url = "https://arxiv.org/pdf/2310.06825.pdf"
    pdf_path = "reference_document.pdf"

    # Download the PDF
    download_pdf(pdf_url, pdf_path)
    extracted_entities = extract_entities_from_pdf(pdf_path)
    
    print("Extracted Entities:")
    for category, entities in extracted_entities.items():
        print(f"\n{category.capitalize()}:")
        for entity in entities:
            print(f"- {entity}")
```

You can run the Python script as many times, or use this in an application to continue generating summaries:
```bash
python upload_and_retreive.py
```

## Customization and Advanced Usage

You can customize the entity extraction process by modifying the `system_prompt` in the extraction graph. For example:

- To focus on specific entity types:
  ```yaml
  system_prompt: 'Extract only person names and organizations from the following text. Provide the results in a JSON format with categories: persons and organizations.'
  ```

- To include entity relationships:
  ```yaml
  system_prompt: 'Extract named entities and their relationships from the following text. Provide the results in a JSON format with categories: entities (including type and name) and relationships (including type and involved entities).'
  ```

You can also experiment with different Mistral models by changing the `model_name` parameter to find the best balance between speed and accuracy for your specific use case.

## Conclusion

While the example might look simple, there are some unique advantages of using Indexify for this -

1. **Scalable and Highly Availability**: Indexify server can be deployed on a cloud and it can process 1000s of PDFs uploaded into it, and if any step in the pipeline fails it automatically retries on another machine.
2. **Flexibility**: You can use any other [PDF extraction model](https://docs.getindexify.ai/usecases/pdf_extraction/) we used here doesn't work for the document you are using. 

## Next Steps

- Learn more about Indexify on our docs - https://docs.getindexify.ai
- Go over an example, which uses Mistral for [building summarization at scale](../pdf-summarization/)
