Integrate a new extraction system into ExtractBench: $ARGUMENTS

You are integrating a new pipeline into the ExtractBench benchmark. The user will provide a system name and any relevant context (API docs, SDK links, product website, etc.). Your job is to create all the files needed so that `uv run extract-bench run <pipeline_name>` works end-to-end.

ExtractBench scores **schema-guided extraction**: the benchmark hands the system a document *and* a JSON Schema, and the system returns JSON that validates against that schema, plus optional per-field evidence (page and bounding box). Everything below assumes an `EXTRACT` provider. The repository also carries a parse and layout-detection roster inherited from ParseBench — only touch those if the user explicitly asks for a parse pipeline.

---

## Step 1: Understand the provider

Before writing any code, research the provider:

1. If the user gave a URL, fetch and read it to understand the API/SDK.
2. Determine:
   - **Schema input**: Does it accept a JSON Schema directly? A proprietary field list? A prompt-only "describe what you want" interface? If it is not JSON Schema, you will need a converter (see `nuextract3_extract.py`, which converts JSON Schema to a NuExtract template).
   - **Integration style**: Cloud API (needs an API key), self-hosted model (needs an endpoint URL you run yourself), coding-agent CLI, or two-stage (parse first, then extract from text).
   - **SDK/API pattern**: Python SDK? REST? Sync or job-poll? What is the auth method?
   - **Input format**: Does it accept the PDF directly, or does it need page images (rasterize with a configurable DPI)?
   - **Evidence**: Does it return per-field citations with a page and/or bounding box? If yes, in what coordinate system? If no, `field_citations` stays empty and the system simply scores 0 on the grounding metrics — that is expected and fine.
   - **Cost**: Does the response carry token usage or a billed amount? The bench records `cost_usd` per document; a flat per-page price is also acceptable.

---

## Step 2: Find the closest existing provider to use as a template

Look at `src/extract_bench/inference/providers/extract/` and pick the best template:

- **Cloud LLM/VLM with structured output**: `openai_responses.py`, `gemini_direct.py`, `anthropic_direct.py`
- **Cloud extraction API (REST, job-poll)**: `reducto.py`, `extend.py`, `datalab.py`, `llamaextract_v2_api.py`
- **Self-hosted vLLM endpoint**: `vllm_extract.py` (JSON Schema guided decoding), `nuextract3_extract.py` (template conversion), `lift.py` (SDK server)
- **Coding-agent CLI**: `claude_code_extract.py`, `codex_code_extract.py`
- **Two-stage (parse → text extract)**: the `input_mode="parsed_text"` path plus `parsed_text_source.py`

Read the template file to understand the exact pattern before writing anything.

---

## Step 3: Create the provider file

Create `src/extract_bench/inference/providers/extract/<provider_name>.py`.

The provider must:

1. Use the `@register_provider("<provider_name>")` decorator from `extract_bench.inference.providers.registry`
2. Subclass `Provider` from `extract_bench.inference.providers.base`
3. Implement `__init__`, `run_inference`, and `normalize`:

```python
from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType


@register_provider("<provider_name>")
class MyExtractProvider(Provider):
    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        # Read config from self.base_config; read API keys from os.environ.
        # Import the SDK lazily (inside __init__, not at module level).
        # Raise ProviderConfigError for missing keys/deps/endpoints.

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        # Reject anything that is not ProductType.EXTRACT.
        # Require request.schema_override — that is the JSON Schema for this document.
        # Call the external API/SDK with the document at request.source_file_path.
        # Return RawInferenceResult with raw_output as a dict (the full response).
        # ProviderTransientError for retryable errors (network, rate limits, 5xx);
        # ProviderPermanentError for non-retryable ones (bad file, 4xx, unparseable output).

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        # Build ExtractOutput(task_type="extract", example_id=..., pipeline_name=...,
        #                     extracted_data=<the JSON the system returned>,
        #                     field_citations=[FieldCitation(...), ...])
        # Return InferenceResult wrapping both raw and normalized output.
```

Key conventions:

- **`extracted_data` is scored as-is.** Do not reshape, reorder, or "fix" it to look more like the ground truth — the metric handles ordering (arrays match as unordered sets) and an omitted key already scores as an explicit `null`. Repairing output here silently inflates the system's score.
- **Empty output is a failure, not a zero-value success.** If the system returns nothing parseable, raise `ProviderPermanentError`. The evaluator scores failed documents as zero; swallowing the error hides it behind a misleadingly high success rate.
- `FieldCitation` needs `field_path` (dotted path into the extracted JSON), `page` (1-indexed), and `bbox` as normalized COCO `[x, y, width, height]`. Convert the provider's coordinate system; if it only cites a page, leave `bbox=None` and the field still counts for page-level grounding.
- Import SDKs lazily (inside `__init__` or methods) so a core install does not need every optional dependency.
- API keys come from `os.environ`, not from the config dict. Config options (model, timeout, dpi, mode, tier) come from `self.base_config`.
- Self-hosted providers must have **no default endpoint**. Read `server_url` from config, falling back to the env var named by the pipeline's `endpoint_env_var` — see `vllm_extract.py`.
- `raw_output` should preserve the full API response for debugging and cost accounting.

---

## Step 4: Register the provider module

Add the module name to `_PROVIDER_MODULES` in `src/extract_bench/inference/providers/extract/__init__.py`. The list is alphabetically sorted; the module name is the filename without `.py`.

---

## Step 5: Register pipeline configurations

Add pipeline definitions inside `register_extract_pipelines()` in `src/extract_bench/inference/pipelines/extract.py`, using the `_pipeline_spec` helper:

```python
register_fn(
    _pipeline_spec(
        pipeline_name="acme_extract_accurate",   # {provider}_extract_{variant}
        provider_name="acme_extract",             # must match @register_provider
        config={                                   # passed to Provider.__init__ as base_config
            "model": "acme-v2",
            "timeout": 900,
        },
    )
)
```

Naming conventions:

- Pipeline names are `{provider}_extract_{variant}` (`reducto_deep_extract`, `extend_extract_max`, `datalab_parse_accurate_extract_balanced`).
- **Registry names must not be prefixes of one another in a way that hides a distinct system.** They are matched as whole path components, but a name like `acme_extract` alongside `acme_extract_max` is still confusing in reports — prefer a distinguishing suffix on both.
- Add a comment section header for the new provider, and register each mode/tier the provider offers as its own pipeline rather than making the mode a runtime flag.
- Put the pipeline in the section that matches how it runs (hosted API, coding agent, self-hosted), so `extract-bench pipelines` groups sensibly.

---

## Step 6: Update documentation

Add the new pipeline(s) to the **Extract Pipelines** table in `docs/pipelines.md`:

```markdown
| `pipeline_name` | Provider label | Short note, and the env var it needs |
```

If the pipeline needs a new environment variable, add it to `.env.example` with a one-line comment naming the pipelines that read it. Self-hosted entries must show a placeholder host, never a real deployment URL.

---

## Step 7: Verify

```bash
# The pipeline appears in the extract roster
uv run extract-bench pipelines

# End-to-end on the 6-document test split (needs credentials for the system)
uv run extract-bench run <pipeline_name> --test

# Inspect what it actually returned, including per-field evidence
uv run extract-bench serve <pipeline_name>
```

Check the run summary before declaring success: a pipeline that "completes" with every document failing scores zero, and the CLI warns about it. Confirm that `extracted_data` is populated and, if the system returns evidence, that `field_citations` is non-empty and the boxes land on the right page in the report.

If there are import errors or missing dependencies, fix them. The lazy import pattern in `providers/extract/__init__.py` means a missing optional SDK skips only that provider.

---

## Summary checklist

- [ ] Provider file created in `providers/extract/`
- [ ] Provider registered with `@register_provider()` and rejects non-`EXTRACT` product types
- [ ] `schema_override` required; unparseable output raises instead of returning `{}`
- [ ] `field_citations` populated (or deliberately empty, if the system returns no evidence)
- [ ] Module added to `_PROVIDER_MODULES` in `providers/extract/__init__.py`
- [ ] Pipeline(s) registered in `pipelines/extract.py`
- [ ] `docs/pipelines.md` and, if needed, `.env.example` updated
- [ ] `uv run extract-bench pipelines` shows the new pipeline(s)
- [ ] `uv run extract-bench run <pipeline_name> --test` produces non-zero scores
