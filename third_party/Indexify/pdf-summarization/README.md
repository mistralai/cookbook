# PDF Summarization with Indexify and Mistral

In this cookbook, we'll explore how to create a PDF summarization pipeline using Indexify and Mistral's large language models. By the end of the document, you should have a pipeline capable of ingesting 1000s of PDF documents, and using Mistral for summarization. 

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
   - [Install Indexify](#install-indexify)
   - [Install Required Extractors](#install-required-extractors)
4. [Creating the Extraction Graph](#creating-the-extraction-graph)
5. [Implementing the Summarization Pipeline](#implementing-the-summarization-pipeline)
6. [Running the Summarization](#running-the-summarization)
7. [Customization and Advanced Usage](#customization-and-advanced-usage)
8. [Conclusion](#conclusion)

## Introduction

The summarization pipeline is going to be composed of two steps -
- PDF to Text extraction. We are going to use a pre-built extractor for this - `tensorlake/pdfextractor`.
- We use Mistral for summarization.


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

The extraction graph defines the flow of data through our summarization pipeline. We'll create a graph that first extracts text from PDFs, then sends that text to Mistral for summarization.

Create a new Python file called `pdf_summarization_graph.py` and add the following code:

```python
from indexify import IndexifyClient, ExtractionGraph

client = IndexifyClient()

extraction_graph_spec = """
name: 'pdf_summarizer'
extraction_policies:
  - extractor: 'tensorlake/pdfextractor'
    name: 'pdf_to_text'
  - extractor: 'tensorlake/mistral'
    name: 'text_to_summary'
    input_params:
      model_name: 'mistral-large-latest'
      key: 'YOUR_MISTRAL_API_KEY'
      system_prompt: 'Summarize the following text in a concise manner, highlighting the key points:'
    content_source: 'pdf_to_text'
"""

extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
client.create_extraction_graph(extraction_graph)
```

Replace `'YOUR_MISTRAL_API_KEY'` with your actual Mistral API key.

You can run this script to set up the pipeline:
```bash
python pdf_summarization_graph.py
```

## Implementing the Summarization Pipeline

Now that we have our extraction graph set up, we can upload files and make the pipeline generate summaries:

Create a file `upload_and_retreive.py`

```python
import os
import requests
from indexify import IndexifyClient

def download_pdf(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"PDF downloaded and saved to {save_path}")

def summarize_pdf(pdf_path):
    client = IndexifyClient()
    
    # Upload the PDF file
    content_id = client.upload_file("pdf_summarizer", pdf_path)
    
    # Wait for the extraction to complete
    client.wait_for_extraction(content_id)
    
    # Retrieve the summarized content
    summary = client.get_extracted_content(
        content_id=content_id,
        graph_name="pdf_summarizer",
        policy_name="text_to_summary"
    )
    
    return summary[0]['content'].decode('utf-8')

# Example usage
if __name__ == "__main__":
    pdf_url = "https://arxiv.org/pdf/2310.06825.pdf"
    pdf_path = "reference_document.pdf"
    
    # Download the PDF
    download_pdf(pdf_url, pdf_path)
    
    # Summarize the PDF
    summary = summarize_pdf(pdf_path)
    print("Summary of the PDF:")
    print(summary)
```

You can run the Python script as many times, or use this in an application to continue generating summaries:
```bash
python upload_and_retreive.py
```

## Customization and Advanced Usage

You can customize the summarization process by modifying the `system_prompt` in the extraction graph. For example:

- To generate bullet-point summaries:
  ```yaml
  system_prompt: 'Summarize the following text as a list of bullet points:'
  ```

- To focus on specific aspects of the document:
  ```yaml
  system_prompt: 'Summarize the main arguments and supporting evidence from the following text:'
  ```

You can also experiment with different Mistral models by changing the `model_name` parameter to find the best balance between speed and accuracy for your specific use case.

## Conclusion

While the example might look simple, there are some unique advantages of using Indexify for this -

1. **Scalable and Highly Availability**: Indexify server can be deployed on a cloud and it can process 1000s of PDFs uploaded into it, and if any step in the pipeline fails it automatically retries on another machine.
2. **Flexibility**: You can use any other [PDF extraction model](https://docs.getindexify.ai/usecases/pdf_extraction/) we used here doesn't work for the document you are using.

## Next Steps

- Learn more about Indexify on our docs - https://docs.getindexify.ai
- Learn how to use Indexify and Mistral for [entity extraction from PDF documents](../pdf-entity-extraction/) 
