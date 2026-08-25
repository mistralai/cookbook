# Available Pipelines

To see the full list:

```bash
uv run extract-bench pipelines
```

## Extract Pipelines (ExtractBench)

Extraction pipelines run per split over the ExtractBench dataset:

```bash
uv run extract-bench download
uv run extract-bench run <pipeline_name>
```

| Pipeline | Provider | Notes |
|---|---|---|
| `llamaextract_cost_effective` / `llamaextract_agentic` | LlamaExtract V2 API | word-grounded citations, via a parse at the same tier (`LLAMA_CLOUD_API_KEY`) |
| `llamaextract_agentic_plus` | LlamaExtract V2 API | highest tier; returns word-level citation boxes natively, no parse pass needed |
| `llamaextract_cost_effective_standard_bbox` / `llamaextract_agentic_standard_bbox` | LlamaExtract V2 API | same tiers with block-level citation boxes and no granular parse pass |
| `openai_gpt_5_4_extract_oneshot_structured_output_file` (+ `_nano`) | OpenAI Responses | one-shot structured output over the uploaded file |
| `gemini_3_5_flash_extract_oneshot_structured_output_file` | Gemini | one-shot structured output (thinking: low) |
| `gemini_3_6_flash_extract_oneshot_structured_output_file` | Gemini | one-shot structured output (thinking: medium) |
| `gemini_3_7_flash_extract_oneshot_structured_output_file` | Gemini | one-shot structured output (thinking: medium) |
| `anthropic_haiku_4_5_extract_oneshot_structured_output_file` | Anthropic | one-shot structured output |
| `*_extract_twostage_parse_agentic_structured_output_text` | OpenAI / Gemini / Anthropic | LlamaParse agentic markdown → text extract; cost totals parse + extract |
| `deepseek_v4_pro_extract_twostage_parse_agentic_structured_output_text` | DeepSeek (Fireworks) | two-stage text extract |
| `claude_code_extract_opus_4_8` | Claude Code CLI | agentic extraction; cost from CLI `total_cost_usd` |
| `codex_code_extract_gpt_5_4_low` / `codex_code_extract_gpt_5_5_low` / `codex_code_extract_gpt_5_5_high` | Codex CLI | agentic extraction; cost estimated from token usage |
| `reducto_extract` / `reducto_deep_extract` | Reducto | deep variant adds citations |
| `extend_extract` / `extend_extract_max` | Extend | citations enabled; max-context array strategy variant |
| `landingai_extract` | LandingAI ADE | |
| `datalab_parse_accurate_extract_fast` / `_balanced` | Datalab | accurate parse + fast or balanced extraction, JSON tree citations |
| `lift_extract` | Self-hosted lift SDK | requires `LIFT_ENDPOINT_URL` |
| `qwen3_6_35b_a3b_fp8_vllm_extract_oneshot_structured_output_file` | Self-hosted vLLM | JSON Schema guided decoding; requires `QWEN35_SERVER_URL` |
| `gemma4_26b_vllm_extract_oneshot_structured_output_file` | Self-hosted vLLM | JSON Schema guided decoding; requires `GEMMA4_SERVER_URL` |
| `nuextract3_extract` | Self-hosted vLLM | schema converted to a NuExtract template; requires `NUEXTRACT3_SERVER_URL` |

## Parse Pipelines

Parse pipelines (inherited from ParseBench, used here by the two-stage extract
baselines and grounding cross-eval) can be run with:

```bash
uv run extract-bench run <pipeline_name>
```

## Setup

Copy `.env.example` to `.env` and fill in the API keys / endpoints for the providers you want to use:

```bash
cp .env.example .env
```

---

## Cloud API Pipelines

These pipelines use hosted APIs. You only need an API key in your `.env` file.

**Bold** pipelines are baselines evaluated in the [ParseBench paper](https://arxiv.org/abs/2604.08538). The name used in the paper is shown in parentheses.

### LlamaParse

| Pipeline | Description | Env Var |
|---|---|---|
| **`llamaparse_agentic`** | Agentic tier (In paper: *LlamaParse Agentic*) | `LLAMA_CLOUD_API_KEY` |
| **`llamaparse_cost_effective`** | Cost-effective tier (In paper: *LlamaParse Cost Effective*) | `LLAMA_CLOUD_API_KEY` |
| `llamaparse_agentic_plus` | Agentic plus tier | `LLAMA_CLOUD_API_KEY` |

### OpenAI

| Pipeline | Description | Env Var |
|---|---|---|
| `openai_gpt5_mini_reasoning_medium_parse` | GPT-5 Mini, medium reasoning, image mode | `OPENAI_API_KEY` |
| `openai_gpt5_mini_reasoning_medium_parse_file` | GPT-5 Mini, medium reasoning, PDF file mode | `OPENAI_API_KEY` |
| `openai_gpt5_mini_reasoning_minimal_parse` | GPT-5 Mini, minimal reasoning | `OPENAI_API_KEY` |
| `openai_gpt5_mini_reasoning_minimal_parse_file` | GPT-5 Mini, minimal reasoning, file mode | `OPENAI_API_KEY` |
| `openai_gpt5_mini_reasoning_medium_parse_with_layout` | GPT-5 Mini, medium reasoning + layout | `OPENAI_API_KEY` |
| **`openai_gpt5_mini_reasoning_medium_parse_with_layout_file`** | GPT-5 Mini, medium reasoning + layout, file (In paper: *OpenAI GPT-5 Mini (Reasoning Medium)*) | `OPENAI_API_KEY` |
| `openai_gpt5_mini_reasoning_minimal_parse_with_layout` | GPT-5 Mini, minimal reasoning + layout | `OPENAI_API_KEY` |
| **`openai_gpt5_mini_reasoning_minimal_parse_with_layout_file`** | GPT-5 Mini, minimal reasoning + layout, file (In paper: *OpenAI GPT-5 Mini (Reasoning Minimal)*) | `OPENAI_API_KEY` |
| `openai_gpt_5_4_parse` | GPT-5.4, image mode | `OPENAI_API_KEY` |
| `openai_gpt_5_4_parse_file` | GPT-5.4, PDF file mode | `OPENAI_API_KEY` |
| **`openai_gpt_5_4_parse_with_layout_file`** | GPT-5.4, parse + layout, file mode (In paper: *OpenAI GPT-5.4*) | `OPENAI_API_KEY` |

### Anthropic Claude

| Pipeline | Description | Env Var |
|---|---|---|
| `anthropic_haiku_parse` | Claude Haiku 4.5, image mode | `ANTHROPIC_API_KEY` |
| `anthropic_haiku_parse_file` | Claude Haiku 4.5, PDF file mode | `ANTHROPIC_API_KEY` |
| `anthropic_haiku_parse_with_layout` | Claude Haiku 4.5, parse + layout | `ANTHROPIC_API_KEY` |
| **`anthropic_haiku_parse_with_layout_file`** | Claude Haiku 4.5, parse + layout, file mode (In paper: *Anthropic Haiku 4.5 (Disable Thinking)*) | `ANTHROPIC_API_KEY` |
| **`anthropic_haiku_thinking_parse_with_layout_file`** | Claude Haiku 4.5, extended thinking + layout (In paper: *Anthropic Haiku 4.5 (Thinking)*) | `ANTHROPIC_API_KEY` |
| `anthropic_opus_4_6_parse` | Claude Opus 4.6, image mode | `ANTHROPIC_API_KEY` |
| `anthropic_opus_4_6_parse_file` | Claude Opus 4.6, PDF file mode | `ANTHROPIC_API_KEY` |
| **`anthropic_opus_4_6_parse_with_layout_file`** | Claude Opus 4.6, parse + layout, file mode (In paper: *Anthropic Opus 4.6*) | `ANTHROPIC_API_KEY` |
| **`anthropic_opus_4_8_parse_with_layout_file`** | Claude Opus 4.8, parse + layout, file mode (In paper: *Anthropic Opus 4.8*) | `ANTHROPIC_API_KEY` |
| `anthropic_sonnet_5_parse_with_layout_file` | Claude Sonnet 5, adaptive thinking + layout, file mode | `ANTHROPIC_API_KEY` |
| `anthropic_fable_5_parse_with_layout_file` | Claude Fable 5, parse + layout, file mode | `ANTHROPIC_API_KEY` |

### Google Gemini

| Pipeline | Description | Env Var |
|---|---|---|
| `google_gemini_3_flash_lite_parse` | Gemini 3 Flash Lite, image mode | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_flash_lite_parse_file` | Gemini 3 Flash Lite, file mode | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_flash_thinking_minimal_parse` | Gemini 3 Flash, minimal thinking | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_flash_thinking_minimal_parse_file` | Gemini 3 Flash, minimal thinking, file | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_flash_thinking_high_parse` | Gemini 3 Flash, high thinking | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_flash_thinking_high_parse_file` | Gemini 3 Flash, high thinking, file | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_flash_thinking_minimal_parse_with_layout` | Gemini 3 Flash, minimal thinking + layout | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_flash_thinking_high_parse_with_layout` | Gemini 3 Flash, high thinking + layout | `GOOGLE_GEMINI_API_KEY` |
| **`google_gemini_3_flash_thinking_minimal_parse_with_layout_file`** | Gemini 3 Flash, minimal thinking + layout file (In paper: *Google Gemini 3 Flash (Thinking Minimal)*) | `GOOGLE_GEMINI_API_KEY` |
| **`google_gemini_3_flash_thinking_high_parse_with_layout_file`** | Gemini 3 Flash, high thinking + layout file (In paper: *Google Gemini 3 Flash (Thinking High)*) | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_flash_thinking_minimal_parse_with_layout_agentic_vision` | Agentic vision, minimal thinking | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_flash_thinking_medium_parse_with_layout_agentic_vision` | Agentic vision, medium thinking | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_flash_thinking_high_parse_with_layout_agentic_vision` | Agentic vision, high thinking | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_1_flash_lite_parse` | Gemini 3.1 Flash Lite | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_1_flash_lite_thinking_high_parse` | Gemini 3.1 Flash Lite, high thinking | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_1_pro_parse` | Gemini 3.1 Pro, default thinking | `GOOGLE_GEMINI_API_KEY` |
| **`google_gemini_3_1_pro_parse_with_layout_file`** | Gemini 3.1 Pro, parse + layout, file mode (In paper: *Google Gemini 3.1 Pro*) | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_5_flash_parse_with_layout` | Gemini 3.5 Flash, default thinking + layout | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_5_flash_no_thinking_parse_with_layout` | Gemini 3.5 Flash, minimal thinking + layout | `GOOGLE_GEMINI_API_KEY` |
| **`google_gemini_3_5_flash_parse_with_layout_file`** | Gemini 3.5 Flash, default thinking + layout, file mode (In paper: *Google Gemini 3.5 Flash (Thinking Medium)*) | `GOOGLE_GEMINI_API_KEY` |
| **`google_gemini_3_5_flash_no_thinking_parse_with_layout_file`** | Gemini 3.5 Flash, minimal thinking + layout, file mode (In paper: *Google Gemini 3.5 Flash (Thinking Minimal)*) | `GOOGLE_GEMINI_API_KEY` |
| **`google_gemini_3_5_flash_lite_parse_with_layout_file`** | Gemini 3.5 Flash Lite, layout + file mode (In paper: *Google Gemini 3.5 Flash Lite*) | `GOOGLE_GEMINI_API_KEY` |
| **`google_gemini_3_6_flash_parse_with_layout_file`** | Gemini 3.6 Flash, default thinking + layout, file mode (In paper: *Google Gemini 3.6 Flash (Thinking Medium)*) | `GOOGLE_GEMINI_API_KEY` |
| **`google_gemini_3_6_flash_no_thinking_parse_with_layout_file`** | Gemini 3.6 Flash, minimal thinking + layout, file mode (In paper: *Google Gemini 3.6 Flash (Thinking Minimal)*) | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_7_flash_parse_with_layout` | Gemini 3.7 Flash, default thinking + layout | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_7_flash_no_thinking_parse_with_layout` | Gemini 3.7 Flash, minimal thinking + layout | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_7_flash_parse_with_layout_file` | Gemini 3.7 Flash, default thinking + layout, file mode | `GOOGLE_GEMINI_API_KEY` |
| `google_gemini_3_7_flash_no_thinking_parse_with_layout_file` | Gemini 3.7 Flash, minimal thinking + layout, file mode | `GOOGLE_GEMINI_API_KEY` |

### Azure Document Intelligence

| Pipeline | Description | Env Vars |
|---|---|---|
| **`azure_di_layout`** | Layout model (In paper: *Azure Document Intelligence*) | `AZURE_DOCUMENT_INTELLIGENCE_KEY`, `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` |
| `azure_di_read` | Read model | `AZURE_DOCUMENT_INTELLIGENCE_KEY`, `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` |

### AWS Textract

| Pipeline | Description | Env Vars |
|---|---|---|
| **`aws_textract`** | Standard Textract (In paper: *AWS Textract*) | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| `aws_textract_with_forms` | Textract with forms | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| `aws_textract_text_only` | Textract text only | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |

### Google Document AI

| Pipeline | Description | Env Vars |
|---|---|---|
| `google_docai` | Document AI OCR | `GOOGLE_DOCAI_PROJECT_ID`, `GOOGLE_DOCAI_PROCESSOR_ID` |
| **`google_docai_layout`** | Document AI Layout (In paper: *Google Cloud Document AI*) | `GOOGLE_DOCAI_PROJECT_ID`, `GOOGLE_DOCAI_LAYOUT_PROCESSOR_ID` |

### Reducto

| Pipeline | Description | Env Var |
|---|---|---|
| **`reducto`** | Default Reducto (In paper: *Reducto*) | `REDUCTO_API_KEY` |
| **`reducto_agentic`** | Agentic mode (In paper: *Reducto (Agentic)*) | `REDUCTO_API_KEY` |

### Pulse

| Pipeline | Description | Env Var |
|---|---|---|
| `pulse` | Default model with native markdown output, `/tables` reconstruction on every document, no refinement | `PULSE_API_KEY` |
| `pulse_ultra_2` | `pulse-ultra-2` hosted tier with native markdown output and refinement enabled | `PULSE_API_KEY` |

### Chunkr

| Pipeline | Description | Env Var |
|---|---|---|
| `chunkr` | Default quality | `CHUNKR_API_KEY` |
| `chunkr_high_res` | High resolution | `CHUNKR_API_KEY` |

### Datalab (Marker)

| Pipeline | Description | Env Var |
|---|---|---|
| `datalab_fast` | Fast mode | `DATALAB_API_KEY` |
| `datalab_balanced` | Balanced mode | `DATALAB_API_KEY` |
| `datalab_accurate` | Accurate mode | `DATALAB_API_KEY` |

### Extend AI

| Pipeline | Description | Env Var |
|---|---|---|
| **`extend_parse`** | Default (In paper: *Extend*) | `EXTEND_API_KEY` |
| `extend_parse_2` | 2.0 engine (v2.0.0, GA) | `EXTEND_API_KEY` |
| `extend_parse_light` | Light engine (v1.0.0) | `EXTEND_API_KEY` |
| `extend_parse_document` | Document scope | `EXTEND_API_KEY` |
| `extend_parse_section` | Section scope | `EXTEND_API_KEY` |

### Landing AI

| Pipeline | Description | Env Var |
|---|---|---|
| **`landingai_parse`** | Default (In paper: *LandingAI*) | `LANDING_AI_API_KEY` |

### Unstructured

| Pipeline | Description | Env Var |
|---|---|---|
| `unstructured_auto` | Auto strategy | `UNSTRUCTURED_API_KEY` |
| `unstructured_fast` | Fast strategy | `UNSTRUCTURED_API_KEY` |
| `unstructured_hi_res` | Hi-res strategy | `UNSTRUCTURED_API_KEY` |

### OpenInnovation Parser (oi-parser)

Hosted document-parsing API. Sign up at [oi-parser.ai](https://oi-parser.ai/) to get an API key.

| Pipeline | Description | Env Vars |
|---|---|---|
| **`oi_parser`** | oi-parser hosted `/v1/extract` API | `OI_PARSER_API_KEY`, `OI_PARSER_BASE_URL` (optional) |

---

## Self-hosted Model Pipelines

These pipelines require you to deploy the model on your own infrastructure (e.g., via vLLM, Modal, etc.) and set the endpoint URL in `.env`.

### Gemma 4

| Pipeline | Description | Env Var |
|---|---|---|
| `gemma4_26b_vllm` | Gemma 4 26B-A4B, parse mode | `GEMMA4_SERVER_URL` |
| `gemma4_26b_vllm_with_layout` | Gemma 4 26B-A4B, layout mode | `GEMMA4_SERVER_URL` |
| `gemma4_e4b_vllm` | Gemma 4 E4B (dense 8B), parse mode | `GEMMA4_SERVER_URL` |
| `gemma4_e4b_vllm_with_layout` | Gemma 4 E4B, layout mode | `GEMMA4_SERVER_URL` |

### Qwen3.5-4B

| Pipeline | Description | Env Var |
|---|---|---|
| **`qwen3_5_4b_vllm_parse`** | Parse mode, markdown (In paper: *Qwen 3 VL*) | `QWEN35_SERVER_URL` |
| **`qwen3_5_4b_vllm_layout`** | Layout mode, JSON with bboxes (In paper: *Qwen 3 VL*) | `QWEN35_SERVER_URL` |

### Chandra OCR 2

| Pipeline | Description | Env Var |
|---|---|---|
| `chandra2_vllm` | OpenAI-compatible vLLM API | `CHANDRA2_SERVER_URL` |
| `chandra2_sdk` | Official SDK endpoint | `CHANDRA2_SERVER_URL` |

### DeepSeek-OCR-2

| Pipeline | Description | Env Var |
|---|---|---|
| `deepseekocr2_vllm` | With grounding layout detection | `DEEPSEEKOCR2_SERVER_URL` |
| `deepseekocr2_freeocr` | Free OCR, no grounding | `DEEPSEEKOCR2_SERVER_URL` |

### Granite Vision

| Pipeline | Description | Env Var |
|---|---|---|
| `granite_vision_pipeline` | PP-DocLayout + per-region Granite Vision | `GRANITE_VISION_SERVER_URL` |
| `granite_vision_4_1_4b` | Granite Vision 4.1 4B (vLLM, multi-task) | `VLLM_API_KEY` |

### PaddleOCR-VL

| Pipeline | Description | Env Var |
|---|---|---|
| `paddleocr_vl_vllm` | OpenAI-compatible vLLM API | `PADDLEOCR_SERVER_URL` |
| `paddleocr_vl_pipeline` | Full pipeline (layout + chart routing) | `PADDLEOCR_SERVER_URL` |
| `paddleocr_vl_1_6_vllm` | PaddleOCR-VL-1.6, OCR prompt | `PADDLEOCR_SERVER_URL` |
| `paddleocr_vl_1_6_vllm_table` | PaddleOCR-VL-1.6, table recognition prompt | `PADDLEOCR_SERVER_URL` |
| `paddleocr_vl_1_6_pipeline` | PaddleOCR-VL-1.6, full pipeline (layout + routing) | `PADDLEOCR_SERVER_URL` |

### dots.ocr

| Pipeline | Description | Env Var |
|---|---|---|
| `dots_ocr_1_0_parse` | dots.ocr 1.0 | `DOTS_OCR_ENDPOINT_URL` |
| **`dots_ocr_1_5_parse`** | dots.ocr 1.5, layout+text prompt (In paper: *Dots OCR 1.5*) | `DOTS_OCR_ENDPOINT_URL` |

### Docling

| Pipeline | Description | Env Vars |
|---|---|---|
| **`docling_parse`** | Docling HTTP endpoint (In paper: *Docling*) | `DOCLING_PARSE_ENDPOINT_URL`, `DOCLING_PARSE_API_KEY` (optional) |
| `docling_serve` | Docling Serve HTTP endpoint | `DOCLING_SERVE_ENDPOINT_URL`, `DOCLING_SERVE_API_KEY` (optional) |

### MinerU 2.5

| Pipeline | Description | Env Var |
|---|---|---|
| `mineru25_vllm` | MinerU2.5-2509-1.2B vLLM server (two-step layout + recognition) | `MINERU25_SERVER_URL` |
| `mineru2605pro_vllm` | MinerU2.5-Pro-2605-1.2B vLLM server (adds chart/image analysis) | `MINERU2605PRO_SERVER_URL` |

### MinerU-Diffusion

| Pipeline | Description | Env Var |
|---|---|---|
| `mineru_diffusion` | MinerU-Diffusion-V1-0320-2.5B server (diffusion-decoding OCR, two-stage layout + recognition) | `MINERU_DIFFUSION_SERVER_URL` |

### Nemotron-Omni

| Pipeline | Description | Env Var |
|---|---|---|
| `nemotron_omni_30b_vllm_thinking` | Nemotron-3-Nano-Omni 30B-A3B Reasoning, thinking enabled | `NEMOTRON_OMNI_SERVER_URL` |

### Surya OCR 2

| Pipeline | Description | Env Var |
|---|---|---|
| `surya2_sdk` | Surya OCR 2 SDK server (full-page OCR + layout) | `SURYA2_SERVER_URL` |

---

## Local Pipelines (No API key needed)

These run entirely locally with no external dependencies.

| Pipeline | Description | Requirements |
|---|---|---|
| `pypdf_baseline` | PyPDF text extraction | None |
| `pymupdf_text` | PyMuPDF text extraction | None |
| `pymupdf_html` | PyMuPDF HTML extraction | None |
| `warp_ingest` | Warp-Ingest local parser | `warp-ingest[ocr]>=2.0.1` installed |
| `tesseract_eng` | Tesseract OCR (English) | `tesseract` installed |
| `tesseract_fast` | Tesseract OCR (fast) | `tesseract` installed |
| `tesseract_high_quality` | Tesseract OCR (high quality) | `tesseract` installed |
| `infinity_parser2_flash` | Infinity-Parser2-Flash (vLLM server, JSON layout) | `infinity_parser2`, running vLLM server |
| `infinity_parser2_pro` | Infinity-Parser2-Pro (vLLM server, JSON layout) | `infinity_parser2`, running vLLM server |

---

## Layout Detection Pipelines

| Pipeline | Description | Requirements |
|---|---|---|
| `docling_layout_heron` | Docling Heron layout | Self-hosted endpoint |
| `docling_layout_heron_101` | Docling Heron 1.0.1 | Self-hosted endpoint |
| `docling_layout_old` | Docling legacy layout | Self-hosted endpoint |
| `ppdoclayout_plus_l` | PaddleDetection layout | Self-hosted endpoint |
| `qwen3vl_layout` | Qwen3-VL layout | Self-hosted endpoint |
| `surya_layout` | Surya layout detection | `surya` installed |
| `yolo_doclaynet` | YOLO DocLayNet | Self-hosted endpoint |
