# OCR Search with Mistral and Qdrant

This notebook demonstrates how to use **Mistral OCR** to extract text from documents, embed with **FastEmbed**, and store it in **Qdrant** for semantic search.

---

## What It Does
- Extracts text from scanned documents (PDFs, images) using Mistral OCR  
- Creates embeddings of the recognized text  
- Stores embeddings in Qdrant for scalable semantic search  
- Lets you query with natural language and retrieve the most relevant passages  

---

## Credentials

You’ll need credentials for both **Mistral** and **Qdrant**.

### Qdrant Cloud
1. Create a cluster at [Qdrant Cloud](https://cloud.qdrant.io)  
2. Generate an API key in the cluster’s **API Keys** section  
3. Export both the URL and the key:
   ```bash
   export QDRANT_URL="https://YOUR-CLUSTER-UUID.region.cloud.qdrant.io:6333"
   export QDRANT_API_KEY="your_qdrant_key"
   ```

### Mistral
1. Sign up at [Mistral](https://mistral.ai)  
2. Create an API key in the console  
3. Set it as an environment variable:
   ```bash
   export MISTRAL_API_KEY=your_mistral_key
   ```
---

## Setup
Install dependencies:
```bash
pip install mistralai qdrant-client python-dotenv
```

Optionally, create a `.env` file:
```bash
MISTRAL_API_KEY=your_mistral_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key   # optional for local
```

Run the notebook in this directory to see the full workflow.

---
## Coming Soon
**Multimodal Search with Qdrant and Mistral**
