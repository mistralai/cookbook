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

## Suggested order
1. Start with `mistral-azure-ocr-invoice` if you want the most complete end-to-end demo.
2. Review `document-ai` for evaluation and document intelligence concepts.
3. Explore `microsoft-agent-frame` and `mistral-medium-3-5` for agent and model experimentation.
