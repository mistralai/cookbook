# Document Chunking

Some documents can be very large and exceed the size (MB) and page limits (count) of what models and services like Mistral OCR and Document AI can provide in a single inference call. The following is a utility script that can help with splitting a large document into smaller chunks, and then asynchronously calling the model endpoint via REST. This can be easily modified to work with existing document processing pipelines, and integrated into many of the cookbooks in this repository.

This script can work with Mistral's OCR and Document AI endpoints on La Platforme, or with MaaS (model-as-a-service) endpoints on Azure, GCP, or AWS Sagemaker.

## Setup

1. Install the required packages in requirements.txt
2. Gather inputs for the script:
    - ```--pdf_file``` path for the document to be processed
    - ```--max_size_mb``` maximum size allowed by the API in MB
    - ```--max_pages``` maximum number of pages allowed by the API
    - ```--endpoint``` endpoint URL for the model
    - ```--api_key``` API key for the model
    - ```--model_name``` the name of the model that is specified in the request body e.g. "mistral-ocr-2505"
3. Run the script with the required inputs:

    ```python documentAIchunking.py --pdf_file /path/to/your/document.pdf --max_size_mb 10 --max_pages 10 --endpoint https://modelprovider/endpoint --api_key your_api_key --model_name mistral-ocr-2505```

4. Check the console or logs for final results. 

---

**⚠️ Note:**
This utility is made to be flexible for many document processing approaches and endpoints. By default it will only output the results of the API calls to the console or logs. You will need to modify the script to save the results, or process as your needs require.