# Check Stale Cookbooks

Scan Mistral AI cookbook files for outdated content — deprecated models, old SDK imports, and changed API methods — and surface a prioritized list of issues with reference links.

## When to use this skill

Trigger when the user asks to:
- Check which cookbooks are out of date or stale
- Find cookbooks using deprecated models, old SDK imports, or old API methods
- Audit the `mistral/` directory before a release
- Run the stale checker locally before opening a fix PR

## Prerequisites

```bash
pip install -r skills/check-stale-cookbooks/requirements.txt
```

Set your Mistral API key for LLM-assisted analysis (optional but recommended):

```bash
export MISTRAL_API_KEY=your-api-key
```

## Commands

**Scan all cookbooks with LLM analysis (recommended):**
```bash
python skills/check-stale-cookbooks/check_stale.py --dir mistral --use-llm
```

**Scan a single file:**
```bash
python skills/check-stale-cookbooks/check_stale.py --file mistral/rag/basic_RAG.ipynb --use-llm
```

**Pattern-only mode (no network calls, no API key needed):**
```bash
python skills/check-stale-cookbooks/check_stale.py --dir mistral --no-fetch
```

**Write a JSON report for further processing:**
```bash
python skills/check-stale-cookbooks/check_stale.py --dir mistral --use-llm --output report.json --format json
```

## All flags

| Flag | Description |
|---|---|
| `--dir DIR` | Directory to scan (default: `mistral`) |
| `--file FILE` | Scan a single file |
| `--output FILE` | Write JSON report to this path |
| `--format markdown\|json` | stdout format (default: markdown) |
| `--no-fetch` | Skip network calls; run static pattern checks only |
| `--use-llm` | Enable Mistral API analysis (requires `MISTRAL_API_KEY`) |
| `--no-llm` | Disable LLM analysis even if the key is set |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | No stale content found |
| `1` | One or more files have stale content |
| `2` | Bad arguments (file/dir not found) |

## What is checked

### Static pattern checks (always run)

| Pattern detected | Issue type | Reference |
|---|---|---|
| `from mistralai.client import MistralClient` | `deprecated_import` | [client-python](https://github.com/mistralai/client-python/blob/main/README.md) |
| `from mistralai import MistralClient` | `deprecated_import` | [client-python](https://github.com/mistralai/client-python/blob/main/README.md) |
| `from mistralai.models import ...` | `deprecated_import` | [client-python](https://github.com/mistralai/client-python/blob/main/README.md) |
| `MistralClient(` | `deprecated_class` | [client-python](https://github.com/mistralai/client-python/blob/main/README.md) |
| `ChatMessage(` | `deprecated_class` | [client-python](https://github.com/mistralai/client-python/blob/main/README.md) |
| `client.chat(` (without `.complete`) | `deprecated_method` | [client-python](https://github.com/mistralai/client-python/blob/main/README.md) |
| `client.embeddings(` (without `.create`) | `deprecated_method` | [client-python README](https://github.com/mistralai/client-python/blob/main/README.md) |
| `client.fine_tuning` | `deprecated_api` | [Fine-tuning docs](https://docs.mistral.ai/capabilities/fine-tuning/) |
| `/v1/fine_tuning/` in curl/REST calls | `deprecated_api` | [Fine-tuning docs](https://docs.mistral.ai/capabilities/fine-tuning/) |
| `pip install mistralai==0.` | `outdated_version` | [client-python README](https://github.com/mistralai/client-python/blob/main/README.md) |
| `@mistralai/mistralai@0.` | `outdated_version` | [client-ts README](https://github.com/mistralai/client-ts/blob/main/README.md) |
| `"mistral-tiny"` | `deprecated_model` | [Model list](https://docs.mistral.ai/getting-started/models/models_overview/) |
| `"mistral-medium"` | `deprecated_model` | [Model list](https://docs.mistral.ai/getting-started/models/models_overview/) |
| `"open-mistral-7b"` | `verify_model` | [Model list](https://docs.mistral.ai/getting-started/models/models_overview/) |
| `"open-mixtral-8x7b"` | `verify_model` | [Model list](https://docs.mistral.ai/getting-started/models/models_overview/) |
| `"open-mixtral-8x22b"` | `verify_model` | [Model list](https://docs.mistral.ai/getting-started/models/models_overview/) |
| Model IDs with dated versions (`-2309`, `-2402`) | `pinned_model_version` | [Model list](https://docs.mistral.ai/getting-started/models/models_overview/) |

### Dynamic model check (requires network, skipped with `--no-fetch`)

Fetches the live model list from `/v1/models` and flags any Mistral model ID used in an API call (`model="..."`) that isn't in that list as `unknown_model`.

### LLM-assisted check (requires `--use-llm` and `MISTRAL_API_KEY`)

Sends each file's code content and excerpts from the fetched reference docs to `mistral-medium-latest` for deeper semantic review. Catches issues pattern matching misses: renamed parameters, changed response structures, removed features, outdated best practices.

The LLM is explicitly instructed:
- `_async` suffix methods (`create_async`, `list_async`, `start_async`, etc.) are **valid** — never flag them
- Fine-tuning jobs API (`client.fine_tuning`) is deprecated
- Every flagged issue must include a `quote` field with text from the reference that confirms the deprecation

## Suppressing false positives

Add `# stale-check: ignore` as a comment on any source line to skip it:

```python
# Showing the legacy pattern as a migration example — stale-check: ignore
old_client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
```

This works in Python files, notebook code cells, and code blocks in Markdown.

## File types scanned

| Extension | How code is extracted |
|---|---|
| `.ipynb` | Code cells only (markdown and output cells are skipped) |
| `.md` | Fenced code blocks with language tags `python`, `typescript`, `javascript`, `bash`, `sh`, `shell`, or `unknown` |
| `.py` | Full file |

## Output format

### Markdown (default)

```
# Stale Cookbook Report

**Scanned at:** 2026-07-20T09:00:00Z
**Files with issues:** 2
**Valid models found:** 24
**References checked:** platform-docs-public/openapi.yaml, client-python/README.md, client-ts/README.md

---

## `mistral/rag/basic_RAG.ipynb`

### Deprecated Class
- Deprecated `MistralClient` class (SDK v0). Use `Mistral(api_key=...)` instead.
  - **Location:** `Cell 2, line 4`
  - **Found:** `client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])`
  - **Reference:** [https://github.com/mistralai/client-python/blob/main/README.md](...)
```

### JSON (`--format json`)

```json
{
  "scanned_at": "2026-07-20T09:00:00Z",
  "references_checked": [
    "platform-docs-public/openapi.yaml",
    "client-python/README.md",
    "client-ts/README.md"
  ],
  "valid_models_count": 24,
  "files": [
    {
      "path": "mistral/rag/basic_RAG.ipynb",
      "issues": [
        {
          "type": "deprecated_class",
          "detail": "Deprecated `MistralClient` class (SDK v0). Use `Mistral(api_key=...)` instead.",
          "location": "Cell 2, line 4",
          "matched_line": "client = MistralClient(api_key=os.environ[\"MISTRAL_API_KEY\"])",
          "reference_url": "https://github.com/mistralai/client-python/blob/main/README.md"
        }
      ]
    }
  ]
}
```

## GitHub Action

`.github/workflows/check-stale-cookbooks.yml` runs this script automatically:
- **Weekly** on Monday at 9am UTC
- **Manually** via `workflow_dispatch`

For each file with issues, it creates a GitHub issue (if one doesn't already exist) with a `stale-cookbook` label. When a previously stale file is clean on the next run, the action comments on and closes the existing issue.

Required secret in the repository: `MISTRAL_API_KEY`.

## Reference repositories

- [mistralai/platform-docs-public](https://github.com/mistralai/platform-docs-public) — deprecation notices, fine-tuning docs, changelogs
- [mistralai/client-python](https://github.com/mistralai/client-python) — Python SDK README (fetched and passed to LLM as actual content)
- [mistralai/client-ts](https://github.com/mistralai/client-ts) — TypeScript SDK README (fetched and passed to LLM as actual content)
- [docs.mistral.ai](https://docs.mistral.ai) — model list, fine-tuning deprecation notices
