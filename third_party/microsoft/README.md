# Microsoft Cookbook Workspace

This folder contains a small set of sample projects and experiments built around Microsoft Azure AI, Mistral models, and document intelligence workflows. The material is intended for quick exploration rather than production deployment.

## Subprojects

### document-ai
A collection of Azure AI document and evaluation samples, including notebook experiments and a Python script for model evaluation workflows. It is a good starting point for testing Foundry-style document AI and prompt/evaluator patterns.

### microsoft-agent-frame
A lightweight Agent Framework example that uses Azure CLI authentication and an OpenAI-compatible chat client to run a simple AI agent. This folder is useful for understanding how a Microsoft-style agent can be wired up with model endpoints and deployment names.

### mistral-azure-ocr-invoice
This is the most complete sample in the workspace. It provides an end-to-end invoice OCR pipeline built on Azure-hosted Mistral Document AI:
- Upload a PDF and extract raw text, tables, and structure.
- Apply OCR correction rules to fix common invoice formatting and text errors.
- Check for missing rows, broken table continuity, and extraction gaps.
- Analyze extracted values against reference ranges and flag anomalies.
- Run the experience through a Gradio web UI for interactive review.

To try it locally, install the Python requirements in its folder and start the app with `python app.py` (default URL: http://localhost:7860).

### mistral-medium-3-5
A compact sample area for Mistral Medium 3.5 experiments and notebook-based exploration. Use this folder for quick prototyping and testing of model behavior without the extra UI and pipeline complexity of the invoice project.

### ocr4-underwriting-demo
An end-to-end, event-driven document pipeline plus a human-in-the-loop mortgage underwriting demo on Azure AI Foundry. `mistral-ocr-4` turns a document into markdown and a `document_annotation` (type, language, summary) in a single call; `mistral-medium-3-5` extracts fields and underwrites against rules that live in code, not in the prompt. One `src/` codebase runs two ways: an Azure Function drains a Storage Queue fed by Event Grid (blob upload, then OCR, then reasoning, then a JSON result), and a FastAPI backend with a separate Gradio UI (`/apply`, `/review`, `/chat`) turns an application package into a decision, routing borderline cases to a reviewer. Every document is one connected OpenTelemetry trace through OCR, extraction, and the result, exported to Application Insights.

This sample uses `uv`, not pip. Deploy and run the whole stack with `scripts/up.sh` and tear it down with `scripts/down.sh`; the infrastructure is Bicep under `infra/`, deployed with `azd`. See the folder's `README.md` and `AGENTS.md` for details.

## Suggested order
1. Start with `mistral-azure-ocr-invoice` if you want the most complete end-to-end demo.
2. Review `document-ai` for evaluation and document intelligence concepts.
3. Explore `microsoft-agent-frame` and `mistral-medium-3-5` for agent and model experimentation.
4. Try `ocr4-underwriting-demo` for a full event-driven pipeline with agents, human-in-the-loop review, and end-to-end tracing.
