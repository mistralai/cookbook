# Mistral OCR with Gradio

This cookbook provides a step-by-step guide to setting up and using Mistral OCR with Gradio.  
The application allows you to extract text and images from PDFs and images using Mistral's OCR capabilities.

## Prerequisites

Before you begin, ensure you have the following:

* Python installed on your system.
* An API key from Mistral AI.
* Necessary Python packages installed.

## Step 1: Install Required Packages

First, install the required Python packages. You can do this using pip:

```bash
pip install gradio requests mistralai
```

## Step 2: Set Up Environment Variables

You need to set up your Mistral API key as an environment variable.  
You can do this in your terminal or add it to your script.

You can create an API key on our [Platforme](https://console.mistral.ai/api-keys/).

```python
import os

os.environ["MISTRAL_API_KEY"] = "your_mistral_api_key_here"
```

## Step 3: Import Libraries

Import the necessary libraries in your Python script.

```python
import gradio as gr
import os
import base64
import requests
from mistralai import Mistral
```

## Step 4: Initialize Mistral Client

Initialize the Mistral client using your API key.

```python
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)
```

## Step 5: Define Helper Functions

### Encode Image to Base64

This function encodes an image to a base64 string.  
It will be required to provide local images to the service.

```python
def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return "Error: The file was not found."
    except Exception as e:
        return f"Error: {e}"
```

### Replace Images in Markdown

This function replaces image placeholders in markdown with base64-encoded images.  
Mistral OCR is capable of outputting interleaved text and images; this function will replace placeholders to render with Gradio.

```python
def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})")
    return markdown_str
```

### Get Combined Markdown

This function combines the markdown from all pages of the OCR response.  
It will output a ready-to-be-rendered version and a raw markdown output without the images.

```python
def get_combined_markdown(ocr_response) -> tuple:
    markdowns = []
    raw_markdowns = []
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))
        raw_markdowns.append(page.markdown)
    return "\n\n".join(markdowns), "\n\n".join(raw_markdowns)
```

### Fetch Content Type

This function fetches the content type of a URL.  
The objective is to detect if we are handling PDF files or image files.

```python
def get_content_type(url):
    """Fetch the content type of the URL."""
    try:
        response = requests.head(url)
        return response.headers.get('Content-Type')
    except Exception as e:
        return f"Error fetching content type: {e}"
```

## Step 6: Define OCR Functions

### Perform OCR on File

To perform OCR on local PDF files, it is required to upload them first to the Platforme and get a signed URL that will be used for OCR tasks.

```python
def perform_ocr_file(file, ocr_method="Mistral OCR"):
    if ocr_method == "Mistral OCR":
        if file.name.endswith('.pdf'):
            uploaded_pdf = client.files.upload(
                file={
                    "file_name": file.name,
                    "content": open(file.name, "rb"),
                },
                purpose="ocr"
            )
            signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                },
                include_image_base64=True
            )
            client.files.delete(file_id=uploaded_pdf.id)

        elif file.name.endswith(('.png', '.jpg', '.jpeg')):
            base64_image = encode_image(file.name)
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                },
                include_image_base64=True
            )

        combined_markdown, raw_markdown = get_combined_markdown(ocr_response)
        return combined_markdown, raw_markdown

    return "## Method not supported.", "Method not supported."
```

### Perform OCR on URL

Next, we can define a function to perform OCR on URLs.  
We need different ones for images and for PDF documents.

```python
def perform_ocr_url(url, ocr_method="Mistral OCR"):
    if ocr_method == "Mistral OCR":
        content_type = get_content_type(url)
        if 'application/pdf' in content_type:
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": url,
                },
                include_image_base64=True
            )

        elif any(image_type in content_type for image_type in ['image/png', 'image/jpeg', 'image/jpg']):
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": url,
                },
                include_image_base64=True
            )
        else:
            return "Unsupported file type. Please provide a URL to a PDF or an image.", ""

        combined_markdown, raw_markdown = get_combined_markdown(ocr_response)
        return combined_markdown, raw_markdown

    return "## Method not supported.", "Method not supported."
```

## Step 7: Create Gradio Interface

Lastly, we can create the Gradio interface to interact with the OCR functions!

```python
with gr.Blocks() as demo:
    gr.Markdown("# Mistral OCR")
    gr.Markdown("Upload a PDF or an image, or provide a URL to extract text and images using Mistral OCR capabilities.\n\nLearn more in the blog post [here](https://mistral.ai/news/mistral-ocr).")

    with gr.Tab("Upload File"):
        file_input = gr.File(label="Upload a PDF or Image")
        ocr_method_file = gr.Dropdown(choices=["Mistral OCR"], label="Select OCR Method", value="Mistral OCR")
        file_output = gr.Markdown(label="Rendered Markdown")
        file_raw_output = gr.Textbox(label="Raw Markdown")
        file_button = gr.Button("Process")

        example_files = gr.Examples(
            examples=[
                "pixtral-12b.pdf",
                "receipt.png"
            ],
            inputs=[file_input]
        )

        file_button.click(
            fn=perform_ocr_file,
            inputs=[file_input, ocr_method_file],
            outputs=[file_output, file_raw_output]
        )

    with gr.Tab("Enter URL"):
        url_input = gr.Textbox(label="Enter a URL to a PDF or Image")
        ocr_method_url = gr.Dropdown(choices=["Mistral OCR"], label="Select OCR Method", value="Mistral OCR")
        url_output = gr.Markdown(label="Rendered Markdown")
        url_raw_output = gr.Textbox(label="Raw Markdown")
        url_button = gr.Button("Process")

        example_urls = gr.Examples(
            examples=[
                "https://arxiv.org/pdf/2410.07073",
                "https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/receipt.png"
            ],
            inputs=[url_input]
        )

        url_button.click(
            fn=perform_ocr_url,
            inputs=[url_input, ocr_method_url],
            outputs=[url_output, url_raw_output]
        )

demo.launch(max_threads=1)
```

## Step 8: Run the Application

Run the script to launch the Gradio interface. You can interact with the interface to perform OCR on files or URLs.

```bash
python your_script_name.py
```

A live demo can be found [here](https://pandora-s-mistral-ocr.hf.space), and more information can be found on the [blog post](https://mistral.ai/news/mistral-ocr).
