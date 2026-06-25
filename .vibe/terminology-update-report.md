# Terminology update report

## Scope

This report tracks the cookbook terminology update after adding the local Quarto preview.

Canonical replacements requested:

- `La Plateforme`, `Plateforme`, and `LaPlateforme` -> `Studio`
- `Le Chat` -> `Vibe Work`

## Clear updates applied

Updated authored prose and clear code comments in:

- `quickstart.ipynb`
  - Replaced the API setup reference to `La Plateforme` with `Studio`.
- `third_party/MongoDB/mongodb_mistral.ipynb`
  - Replaced `Mistral AI “La plateforme” embedding endpoints` with `Studio embedding endpoints`.
- `mistral/ocr/tool_usage.ipynb`
  - Replaced `Plateforme` API key setup wording with `Studio`.
- `mistral/ocr/structured_ocr.ipynb`
  - Replaced `Plateforme` API key setup wording with `Studio`.
- `mistral/fine_tune/pixtral_finetune_on_satellite_data.ipynb`
  - Replaced `LaPlateforme` references in comments with `Studio`.
- `mistral/agents/non_framework/transcript_linearticket_agent/TranscriptToLinearTicketAgent.ipynb`
  - Replaced generated PRD references from `Le Chat` to `Vibe Work`.

## Left for manual review

I did not change these cases because they are ambiguous or historical:

1. `third_party/Haystack/haystack_chat_with_docs.ipynb`
   - Contains the source URL `https://mistral.ai/news/la-plateforme/`.
   - Reason: this appears to be a historical blog URL used as data for the example. Changing it could break the source fetch or change the example data.

2. `third_party/CAMEL_AI/camel_graph_rag.ipynb`
   - Contains generated graph output: `Node la Plateforme (label: Platform)`.
   - Reason: this is notebook output, not authored prose. Changing it manually could make the output inconsistent with the executed notebook state.

## Verification run

Ran:

```bash
python3 scripts/generate_quarto_index.py
python3 -m py_compile scripts/generate_quarto_index.py
git diff --check
```

All checks passed.

## Note on remaining grep hits

A broad grep for `Le Chat` variants still returns false positives for generic words such as `chat` and `ChatGPT` in unrelated content. Those are not product-name references and were not changed.
