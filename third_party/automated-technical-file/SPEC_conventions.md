# Mistral AI Cookbook: Conventions for third_party integrations

## 1. Folder Naming
Based on an audit of existing `third_party/` directories, naming follows branding rather than a strict casing rule.
- **Recommendation:** Use `automated-technical-file` (kebab-case) to match descriptive tool naming (like `x-cmd`) or lowercase group (`langchain`, `streamlit`).
- **Alternative:** `AutomatedTechnicalFile` (PascalCase) if treating as a standalone brand (like `LlamaIndex`).

## 2. README.md Structure
- **Title:** Partner/Tool Name (`# Automated Technical File`)
- **Description:** Brief overview of the integration.
- **Installation:** `pip install mistralai` + other deps.
- **Quick Links:** Installation, Documentation, SDK.
- **Example:** Link to the notebook in the same directory.

## 3. Notebook (.ipynb) Layout
- **Author Cell (First):** Name, GitHub Handle, Affiliation.
- **Setup Cell:** `pip install` with `%%capture`.
- **Auth Cell:** API Key initialization (`os.environ["MISTRAL_API_KEY"]`).
- **Headings:** Clear Step-by-Step Markdown headers (`## Step 1: ...`).
- **Colab Ready:** Ensure paths and assets are reachable from a hosted environment.
