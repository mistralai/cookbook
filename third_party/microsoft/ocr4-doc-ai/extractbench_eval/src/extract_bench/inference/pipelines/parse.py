"""Parse pipelines - PDF/document parsing to markdown/HTML.

LlamaParse pipeline names use the ``llamaparse_`` prefix followed by a tier name:
``llamaparse_cost_effective``, ``llamaparse_agentic``, ``llamaparse_agentic_plus``.

V2 SDK pipeline configs (provider_name="llamaparse") must conform to
llama_cloud.types.parsing_create_params.ParsingCreateParams.

Self-hosted model pipelines (e.g. Gemma4, Qwen3.5, Chandra2, DeepSeek-OCR-2,
dots.ocr, PaddleOCR-VL, Granite Vision) have ``server_url`` or ``endpoint_url``
set to empty strings. Users must provide their own deployment endpoint to use
these pipelines.
"""

from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.product import ProductType


def register_parse_pipelines(register_fn) -> None:  # type: ignore[no-untyped-def]
    """Register all parse-related pipelines."""

    # =========================================================================
    # LlamaParse Production Pipelines (V2 SDK)
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="llamaparse_cost_effective",
            provider_name="llamaparse",
            product_type=ProductType.PARSE,
            config={
                "tier": "cost_effective",
                "version": "latest",
                "disable_cache": True,
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="llamaparse_agentic",
            provider_name="llamaparse",
            product_type=ProductType.PARSE,
            config={
                "tier": "agentic",
                "version": "latest",
                "disable_cache": True,
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="llamaparse_agentic_plus",
            provider_name="llamaparse",
            product_type=ProductType.PARSE,
            config={
                "tier": "agentic_plus",
                "version": "latest",
                "disable_cache": True,
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="llamaparse_agentic_granular_bboxes",
            provider_name="llamaparse",
            product_type=ProductType.PARSE,
            config={
                "tier": "agentic",
                "version": "latest",
                "disable_cache": True,
                "output_options": {
                    "granular_bboxes": ["word"],
                },
            },
        )
    )

    # =========================================================================
    # Extend AI Parse Pipelines
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="extend_parse",
            provider_name="extend_parse",
            product_type=ProductType.PARSE,
            config={
                "target": "markdown",
                "chunking_strategy": "page",
                "block_options": {
                    "tables": {
                        "target_format": "html",
                    }
                },
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="extend_parse_2",
            provider_name="extend_parse",
            product_type=ProductType.PARSE,
            config={
                "target": "markdown",
                "chunking_strategy": "page",
                "engine": "parse_performance",
                "engineVersion": "2.0.0",
                "block_options": {
                    "tables": {"target_format": "html"},
                    "figures": {
                        "enabled": True,
                        "figureImageClippingEnabled": True,
                        "advancedChartExtractionEnabled": True,
                    },
                    "formulas": {"enabled": True},
                },
                "advanced_options": {
                    "enrichmentFormat": "xml",
                    "formattingDetection": [{"type": "change_tracking"}],
                },
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="extend_parse_light",
            provider_name="extend_parse",
            product_type=ProductType.PARSE,
            config={
                "target": "markdown",
                "chunking_strategy": "page",
                "engine": "parse_light",
                "engineVersion": "1.0.0",
                "block_options": {
                    "tables": {"target_format": "html"},
                    "figures": {
                        "enabled": True,
                        "figureImageClippingEnabled": True,
                        "advancedChartExtractionEnabled": True,
                    },
                    "formulas": {"enabled": True},
                },
                "advanced_options": {
                    "enrichmentFormat": "xml",
                },
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="extend_parse_document",
            provider_name="extend_parse",
            product_type=ProductType.PARSE,
            config={
                "target": "markdown",
                "chunking_strategy": "document",
                "block_options": {
                    "tables": {
                        "target_format": "html",
                    }
                },
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="extend_parse_section",
            provider_name="extend_parse",
            product_type=ProductType.PARSE,
            config={
                "target": "markdown",
                "chunking_strategy": "section",
                "block_options": {
                    "tables": {
                        "target_format": "html",
                    }
                },
            },
        )
    )

    # =========================================================================
    # Datalab Pipelines
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="datalab_fast",
            provider_name="datalab",
            product_type=ProductType.PARSE,
            config={
                "output_format": "html,json",
                "max_pages": 25,
                "skip_cache": True,
                "mode": "fast",
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="datalab_balanced",
            provider_name="datalab",
            product_type=ProductType.PARSE,
            config={
                "output_format": "html,json",
                "max_pages": 25,
                "skip_cache": True,
                "mode": "balanced",
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="datalab_accurate",
            provider_name="datalab",
            product_type=ProductType.PARSE,
            config={
                "output_format": "html,json",
                "max_pages": 25,
                "skip_cache": True,
                "mode": "accurate",
            },
        )
    )

    # =========================================================================
    # Pulse Pipelines
    # =========================================================================

    pulse_common_endpoint_config = {
        "async_extract": True,
        "return_html": True,
        "storage": {"enabled": True},
        "markdown_source": "markdown",
        "poll_interval": 1.0,
    }
    pulse_tables_config = {
        "merge": True,
        "table_format": "html",
        "charts_to_tables": True,
    }
    pulse_tables_endpoint_config = {
        "async_tables": True,
        "use_tables_endpoint": True,
        "tables_config": pulse_tables_config,
        "merge_tables_into_markdown": True,
        "replace_existing_tables": True,
        **pulse_common_endpoint_config,
    }
    pulse_ultra_2_image_prompt = (
        "For ParseBench charts, output a markdown table where each row is one data point. "
        "Include every visible chart label needed to identify the value as explicit row or column text, "
        "including series names, legend labels, axis labels, sign/category labels such as "
        "Favorable attitude or Unfavorable attitude, units, and years. Do not leave grouping labels only in prose."
    )
    pulse_ultra_2_prompt = "Preserve chart captions, title hierarchy, table structure, and semantic formatting."
    pulse_ultra_2_config = {
        "model": "pulse-ultra-2",
        "credits_per_page": 10,
        "refine": True,
        "extract_figure": True,
        "figure_description": True,
        "additional_prompt": pulse_ultra_2_prompt,
        "custom_image_prompt": pulse_ultra_2_image_prompt,
        **pulse_common_endpoint_config,
    }

    register_fn(
        PipelineSpec(
            pipeline_name="pulse",
            provider_name="pulse",
            product_type=ProductType.PARSE,
            config={
                "model": "default",
                "refine": False,
                "credits_per_page": 1,
                "figure_processing": {"description": True},
                **pulse_tables_endpoint_config,
            },
        )
    )

    # pulse-ultra-2: hosted tier with native figure extraction.
    # Refinement is enabled for the submitted leaderboard configuration.
    register_fn(
        PipelineSpec(
            pipeline_name="pulse_ultra_2",
            provider_name="pulse",
            product_type=ProductType.PARSE,
            config=pulse_ultra_2_config,
        )
    )

    # =========================================================================
    # Chunkr Pipelines
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="chunkr",
            provider_name="chunkr",
            product_type=ProductType.PARSE,
            config={
                "segmentation_strategy": "LayoutAnalysis",
                "ocr_strategy": "Auto",
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="chunkr_high_res",
            provider_name="chunkr",
            product_type=ProductType.PARSE,
            config={
                "segmentation_strategy": "LayoutAnalysis",
                "ocr_strategy": "All",
                "high_resolution": True,
            },
        )
    )

    # =========================================================================
    # Docling Pipelines
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="docling_parse",
            provider_name="docling_parse",
            product_type=ProductType.PARSE,
            config={
                "endpoint_url": "",  # Set via environment or override
                "timeout": 120,
            },
        )
    )

    # =========================================================================
    # Docling Serve
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="docling_serve",
            provider_name="docling_serve",
            product_type=ProductType.PARSE,
            config={
                "endpoint_url": "",  # Set via environment or override
                "timeout": 120,
            },
        )
    )

    # =========================================================================
    # Landing AI Pipelines
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="landingai_parse",
            provider_name="landingai",
            product_type=ProductType.PARSE,
            config={
                "model": "dpt-2-latest",
                "split": "page",
            },
        )
    )

    # ===========================
    # Azure Document Intelligence
    # ===========================

    # Azure Document Intelligence with prebuilt-layout model (default)
    register_fn(
        PipelineSpec(
            pipeline_name="azure_di_layout",
            provider_name="azure_document_intelligence",
            product_type=ProductType.PARSE,
            config={
                "model_id": "prebuilt-layout",
                "output_content_format": "markdown",
            },
        )
    )

    # Azure Document Intelligence with prebuilt-read model (OCR-focused)
    # Note: prebuilt-read does NOT support markdown output (only prebuilt-layout does).
    # Using "text" format which is the correct option for this model.
    register_fn(
        PipelineSpec(
            pipeline_name="azure_di_read",
            provider_name="azure_document_intelligence",
            product_type=ProductType.PARSE,
            config={
                "model_id": "prebuilt-read",
                "output_content_format": "text",
            },
        )
    )

    # =========================================================================
    # AWS Textract Pipelines
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="aws_textract",
            provider_name="textract",
            product_type=ProductType.PARSE,
            config={
                "output_tables_as_html": True,
                "detect_tables": True,
                "detect_forms": False,
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="aws_textract_with_forms",
            provider_name="textract",
            product_type=ProductType.PARSE,
            config={
                "output_tables_as_html": True,
                "detect_tables": True,
                "detect_forms": True,
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="aws_textract_text_only",
            provider_name="textract",
            product_type=ProductType.PARSE,
            config={
                "output_tables_as_html": False,
                "detect_tables": False,
                "detect_forms": False,
            },
        )
    )

    # =========================================================================
    # Google Document AI Pipelines
    # =========================================================================

    # Google Document AI OCR processor
    register_fn(
        PipelineSpec(
            pipeline_name="google_docai",
            provider_name="google_docai",
            product_type=ProductType.PARSE,
            config={
                "enable_native_pdf_parsing": True,
                "enable_symbol_detection": False,
            },
        )
    )

    # Google Document AI Layout Parser processor
    register_fn(
        PipelineSpec(
            pipeline_name="google_docai_layout",
            provider_name="google_docai",
            product_type=ProductType.PARSE,
            config={
                "use_layout_parser": True,
            },
        )
    )

    # =========================================================================
    # Baseline/OSS Pipelines
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="liteparse_markdown",
            provider_name="liteparse",
            product_type=ProductType.PARSE,
            config={
                "output_format": "markdown",
                "ocr_enabled": False,
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="liteparse_text",
            provider_name="liteparse",
            product_type=ProductType.PARSE,
            config={
                "output_format": "text",
                "ocr_enabled": False,
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="pypdf_baseline",
            provider_name="pypdf",
            product_type=ProductType.PARSE,
            config={},
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="pymupdf_text",
            provider_name="pymupdf",
            product_type=ProductType.PARSE,
            config={
                "text_format": "text",
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="pymupdf_html",
            provider_name="pymupdf",
            product_type=ProductType.PARSE,
            config={
                "text_format": "html",
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="pymupdf4llm_markdown",
            provider_name="pymupdf4llm",
            product_type=ProductType.PARSE,
            config={},
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="markitdown",
            provider_name="markitdown",
            product_type=ProductType.PARSE,
            config={},
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="warp_ingest",
            provider_name="warp_ingest",
            product_type=ProductType.PARSE,
            config={},
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="opendataloader_markdown",
            provider_name="opendataloader",
            product_type=ProductType.PARSE,
            config={"format": "markdown", "table_method": "cluster"},
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="pdf_inspector",
            provider_name="pdf_inspector",
            product_type=ProductType.PARSE,
            config={},
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="tesseract_eng",
            provider_name="tesseract",
            product_type=ProductType.PARSE,
            config={
                "lang": "eng",
                "dpi": 300,
                "output_type": "text",
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="tesseract_high_quality",
            provider_name="tesseract",
            product_type=ProductType.PARSE,
            config={
                "lang": "eng",
                "dpi": 600,
                "output_type": "text",
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="tesseract_fast",
            provider_name="tesseract",
            product_type=ProductType.PARSE,
            config={
                "lang": "eng",
                "dpi": 150,
                "output_type": "text",
            },
        )
    )

    # =========================================================================
    # PaddleOCR Pipelines
    # =========================================================================

    # PaddleOCR-VL vLLM (OpenAI-compatible API)
    register_fn(
        PipelineSpec(
            pipeline_name="paddleocr_vl_vllm",
            provider_name="paddleocr",
            product_type=ProductType.PARSE,
            config={
                "api_format": "openai",
                "task": "table",
            },
        )
    )

    # PaddleOCR-VL Full Pipeline (layout + chart routing)
    register_fn(
        PipelineSpec(
            pipeline_name="paddleocr_vl_pipeline",
            provider_name="paddleocr",
            product_type=ProductType.PARSE,
            config={
                "api_format": "simple",
            },
        )
    )

    # PaddleOCR-VL 1.5 (0.9B) vLLM — OCR prompt (general text/structure)
    register_fn(
        PipelineSpec(
            pipeline_name="paddleocr_vl_1_5_vllm",
            provider_name="paddleocr",
            product_type=ProductType.PARSE,
            config={
                "api_format": "openai",
                "task": "ocr",
            },
        )
    )

    # PaddleOCR-VL 1.5 (0.9B) vLLM — Table Recognition prompt
    register_fn(
        PipelineSpec(
            pipeline_name="paddleocr_vl_1_5_vllm_table",
            provider_name="paddleocr",
            product_type=ProductType.PARSE,
            config={
                "api_format": "openai",
                "task": "table",
            },
        )
    )

    # PaddleOCR-VL 1.5 (0.9B) full pipeline (layout detection + per-region routing)
    register_fn(
        PipelineSpec(
            pipeline_name="paddleocr_vl_1_5_pipeline",
            provider_name="paddleocr",
            product_type=ProductType.PARSE,
            config={
                "api_format": "simple",
            },
        )
    )

    # PaddleOCR-VL 1.6 (0.9B) vLLM — OCR prompt (general text/structure)
    register_fn(
        PipelineSpec(
            pipeline_name="paddleocr_vl_1_6_vllm",
            provider_name="paddleocr",
            product_type=ProductType.PARSE,
            config={
                "api_format": "openai",
                "task": "ocr",
                "served_model_name": "PaddleOCR-VL-1.6-0.9B",
            },
        )
    )

    # PaddleOCR-VL 1.6 (0.9B) vLLM — Table Recognition prompt
    register_fn(
        PipelineSpec(
            pipeline_name="paddleocr_vl_1_6_vllm_table",
            provider_name="paddleocr",
            product_type=ProductType.PARSE,
            config={
                "api_format": "openai",
                "task": "table",
                "served_model_name": "PaddleOCR-VL-1.6-0.9B",
            },
        )
    )

    # PaddleOCR-VL 1.6 (0.9B) full pipeline (layout detection + per-region routing)
    register_fn(
        PipelineSpec(
            pipeline_name="paddleocr_vl_1_6_pipeline",
            provider_name="paddleocr",
            product_type=ProductType.PARSE,
            config={
                "api_format": "simple",
            },
        )
    )

    # =========================================================================
    # Falcon-OCR (TII, 300M early-fusion VLM with built-in layout-aware OCR)
    # =========================================================================

    # Layout-aware OCR via model.generate_with_layout (PP-DocLayoutV3 inside).
    register_fn(
        PipelineSpec(
            pipeline_name="falconocr_pipeline",
            provider_name="falconocr",
            product_type=ProductType.PARSE,
            config={
                "task": "ocr",
            },
        )
    )

    # Plain single-shot OCR (no layout routing) for ablation.
    register_fn(
        PipelineSpec(
            pipeline_name="falconocr_plain",
            provider_name="falconocr",
            product_type=ProductType.PARSE,
            config={
                "task": "plain",
            },
        )
    )

    # =========================================================================
    # Anthropic Claude Vision Parse
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="anthropic_haiku_parse",
            provider_name="anthropic",
            product_type=ProductType.PARSE,
            config={
                "model": "claude-haiku-4-5-20251001",
                "dpi": 150,
                "max_tokens": 4096,
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="anthropic_opus_4_6_parse",
            provider_name="anthropic",
            product_type=ProductType.PARSE,
            config={
                "model": "claude-opus-4-6",
                "dpi": 150,
                "max_tokens": 8192,
            },
        )
    )

    # =========================================================================
    # OpenAI Vision Parse
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt5_mini_reasoning_medium_parse",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5-mini",
                "dpi": 150,
                "max_tokens": 65536,
            },
        )
    )

    # GPT-5 Mini with reasoning=none (no thinking tokens, lower budget sufficient)
    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt5_mini_reasoning_minimal_parse",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5-mini",
                "dpi": 150,
                "max_tokens": 8192,
                "reasoning_effort": "minimal",
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt_5_4_parse",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5.4-2026-03-05",
                "dpi": 150,
                "max_tokens": 65536,
            },
        )
    )

    # =========================================================================
    # Gemini 3 Flash Vision Parse
    # =========================================================================

    # Gemini 3.1 Pro - Parse (default thinking)
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_1_pro_parse",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.1-pro-preview",
                "dpi": 150,
                "max_tokens": 32768,
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_flash_lite_parse",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.1-flash-lite-preview",
                "dpi": 150,
                "max_tokens": 8192,
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_1_flash_lite_parse",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.1-flash-lite-preview",
                "dpi": 150,
                "max_tokens": 32768,
            },
        )
    )

    # Gemini 3.1 Flash Lite with high thinking budget
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_1_flash_lite_thinking_high_parse",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.1-flash-lite-preview",
                "dpi": 150,
                "max_tokens": 65536,
                "thinking_level": "high",
            },
        )
    )

    # Gemini 3 Flash with high thinking budget (10x output tokens for thinking room)
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_flash_thinking_high_parse",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3-flash-preview",
                "dpi": 150,
                "max_tokens": 65536,
                "mode": "image",
                "thinking_level": "high",
            },
        )
    )

    # Gemini 3 Flash with minimal thinking (same token budget, thinking disabled)
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_flash_thinking_minimal_parse",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3-flash-preview",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "image",
                "thinking_level": "minimal",
            },
        )
    )

    # =========================================================================
    # LLM Parse File Mode Pipelines
    # These pipelines send the raw PDF file to the LLM API instead of
    # converting to images first. This allows the LLM to use its native
    # PDF processing capabilities.
    # =========================================================================

    # Anthropic Haiku - File Mode
    register_fn(
        PipelineSpec(
            pipeline_name="anthropic_haiku_parse_file",
            provider_name="anthropic",
            product_type=ProductType.PARSE,
            config={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 32768,
                "mode": "file",
            },
        )
    )

    # Anthropic Opus 4.6 - File Mode
    register_fn(
        PipelineSpec(
            pipeline_name="anthropic_opus_4_6_parse_file",
            provider_name="anthropic",
            product_type=ProductType.PARSE,
            config={
                "model": "claude-opus-4-6",
                "max_tokens": 8192,
                "mode": "file",
            },
        )
    )

    # OpenAI GPT-5 Mini - File Mode (default reasoning=medium, needs large budget)
    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt5_mini_reasoning_medium_parse_file",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5-mini",
                "max_tokens": 65536,
                "mode": "file",
            },
        )
    )

    # OpenAI GPT-5 Mini - File Mode - Reasoning None
    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt5_mini_reasoning_minimal_parse_file",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5-mini",
                "max_tokens": 32768,
                "mode": "file",
                "reasoning_effort": "minimal",
            },
        )
    )

    # OpenAI GPT-5.4 - File Mode (default reasoning=medium, needs large budget)
    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt_5_4_parse_file",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5.4-2026-03-05",
                "max_tokens": 65536,
                "mode": "file",
            },
        )
    )

    # Gemini 3 Flash Lite - File Mode
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_flash_lite_parse_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.1-flash-lite-preview",
                "max_tokens": 8192,
                "mode": "file",
            },
        )
    )

    # =========================================================================
    # dots.ocr Pipelines
    # =========================================================================

    # dots.ocr 1.0 (original)
    register_fn(
        PipelineSpec(
            pipeline_name="dots_ocr_1_0_parse",
            provider_name="dots_ocr_parse",
            product_type=ProductType.PARSE,
            config={
                "model": "dots-ocr",
                "timeout": 300,
                "dpi": 300,
            },
        )
    )

    # dots.ocr 1.5 (layout+text prompt -> parse + cross-eval for layout detection)
    register_fn(
        PipelineSpec(
            pipeline_name="dots_ocr_1_5_parse",
            provider_name="dots_ocr_parse",
            product_type=ProductType.PARSE,
            config={
                "model": "dots-ocr-1.5",
                "prompt_mode": "prompt_layout_all_en_v1_5",
                "timeout": 300,
                "dpi": 300,
            },
        )
    )

    # =========================================================================
    # Unstructured Pipelines
    # =========================================================================

    # Unstructured hi_res strategy (default/recommended)
    register_fn(
        PipelineSpec(
            pipeline_name="unstructured_hi_res",
            provider_name="unstructured",
            product_type=ProductType.PARSE,
            config={
                "strategy": "hi_res",
                "languages": ["eng"],
                "coordinates": True,
                "include_page_breaks": True,
                "split_pdf_concurrency_level": 5,
            },
        )
    )

    # Unstructured fast strategy
    register_fn(
        PipelineSpec(
            pipeline_name="unstructured_fast",
            provider_name="unstructured",
            product_type=ProductType.PARSE,
            config={
                "strategy": "fast",
                "languages": ["eng"],
                "include_page_breaks": True,
            },
        )
    )

    # Unstructured auto strategy
    register_fn(
        PipelineSpec(
            pipeline_name="unstructured_auto",
            provider_name="unstructured",
            product_type=ProductType.PARSE,
            config={
                "strategy": "auto",
                "languages": ["eng"],
                "include_page_breaks": True,
                "split_pdf_concurrency_level": 5,
            },
        )
    )

    # =========================================================================
    # Reducto Pipelines
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="reducto_agentic",
            provider_name="reducto",
            product_type=ProductType.PARSE,
            config={
                "ocr_system": "standard",
                "agentic": True,
                "agentic_scopes": ["text", "table", "figure"],
                "table_output_format": "html",
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="reducto_agentic_formatting",
            provider_name="reducto",
            product_type=ProductType.PARSE,
            config={
                "ocr_system": "standard",
                "agentic": True,
                "agentic_scopes": ["text", "table", "figure"],
                "table_output_format": "html",
                "formatting_include": ["change_tracking", "highlight", "comments"],
            },
        )
    )

    # =========================================================================
    # DeepSeek-OCR-2
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="deepseekocr2_vllm",
            provider_name="deepseekocr2",
            product_type=ProductType.PARSE,
            config={},
        )
    )

    # DeepSeek-OCR-2 Free OCR (no grounding, more token budget for tables)
    register_fn(
        PipelineSpec(
            pipeline_name="deepseekocr2_freeocr",
            provider_name="deepseekocr2",
            product_type=ProductType.PARSE,
            config={},
        )
    )

    # =========================================================================
    # Unlimited-OCR (baidu/Unlimited-OCR, DeepSeek-OCR successor with grounding)
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="unlimitedocr",
            provider_name="unlimitedocr",
            product_type=ProductType.PARSE,
            config={},
        )
    )

    # =========================================================================
    # Qwen3.5-4B vLLM
    # =========================================================================

    # Qwen3.5-4B vLLM — parse mode (pure markdown, no layout)
    register_fn(
        PipelineSpec(
            pipeline_name="qwen3_5_4b_vllm_parse",
            provider_name="qwen3_5",
            product_type=ProductType.PARSE,
            config={
                "model": "qwen3.5-4b",
                "prompt_mode": "parse",
            },
        )
    )

    # Qwen3.5-4B vLLM — layout mode (structured JSON with bboxes + content)
    register_fn(
        PipelineSpec(
            pipeline_name="qwen3_5_4b_vllm_layout",
            provider_name="qwen3_5",
            product_type=ProductType.PARSE,
            config={
                "model": "qwen3.5-4b",
                "prompt_mode": "layout",
            },
        )
    )

    # Qwen3.5-9B vLLM — layout mode (structured JSON with bboxes + content)
    register_fn(
        PipelineSpec(
            pipeline_name="qwen3_5_9b_vllm_layout",
            provider_name="qwen3_5",
            product_type=ProductType.PARSE,
            config={
                "model": "qwen3.5-9b",
                "prompt_mode": "layout",
            },
        )
    )

    # Qwen3.5-2B vLLM — layout mode
    register_fn(
        PipelineSpec(
            pipeline_name="qwen3_5_2b_vllm_layout",
            provider_name="qwen3_5",
            product_type=ProductType.PARSE,
            config={
                "model": "qwen3.5-2b",
                "prompt_mode": "layout",
            },
        )
    )

    # Qwen3.5-0.8B vLLM — layout mode
    register_fn(
        PipelineSpec(
            pipeline_name="qwen3_5_0_8b_vllm_layout",
            provider_name="qwen3_5",
            product_type=ProductType.PARSE,
            config={
                "model": "qwen3.5-0.8b",
                "prompt_mode": "layout",
            },
        )
    )

    # =========================================================================
    # Qwen3.5-35B-A3B FP8 (unified multimodal, GDN + attention hybrid, MoE 35B/3B)
    # =========================================================================

    # Qwen3.5-35B-A3B FP8 vLLM — parse mode (pure markdown, no layout)
    register_fn(
        PipelineSpec(
            pipeline_name="qwen3_5_35b_a3b_fp8_vllm_parse",
            provider_name="qwen3_5",
            product_type=ProductType.PARSE,
            config={
                "model": "qwen3.5-35b-a3b-fp8",
                "prompt_mode": "parse",
            },
        )
    )

    # =========================================================================
    # Qwen3.6-35B-A3B FP8 (unified multimodal, GDN + attention hybrid, MoE 35B/3B)
    # =========================================================================

    # Qwen3.6-35B-A3B FP8 vLLM — parse mode (pure markdown, no layout)
    register_fn(
        PipelineSpec(
            pipeline_name="qwen3_6_35b_a3b_fp8_vllm_parse",
            provider_name="qwen3_5",
            product_type=ProductType.PARSE,
            config={
                "model": "qwen3.6-35b-a3b-fp8",
                "prompt_mode": "parse",
            },
        )
    )

    # Qwen3.6-35B-A3B FP8 vLLM — parse_layout (unified: one layout-prompt call,
    # cross-evaluated on both parse and layout detection, same pattern as dots_ocr_1_5_parse)
    register_fn(
        PipelineSpec(
            pipeline_name="qwen3_6_35b_a3b_fp8_vllm_parse_layout",
            provider_name="qwen3_5",
            product_type=ProductType.PARSE,
            config={
                "model": "qwen3.6-35b-a3b-fp8",
                "prompt_mode": "layout",
            },
        )
    )

    # =========================================================================
    # Gemma 4
    # =========================================================================

    # Gemma 4 26B-A4B vLLM — parse mode (pure markdown, no layout)
    register_fn(
        PipelineSpec(
            pipeline_name="gemma4_26b_vllm",
            provider_name="gemma4",
            product_type=ProductType.PARSE,
            config={
                "model": "gemma-4-26b-a4b",
                "prompt_mode": "parse",
            },
        )
    )

    # Gemma 4 26B-A4B vLLM — layout mode (div+bbox wrappers, Gemini-style)
    register_fn(
        PipelineSpec(
            pipeline_name="gemma4_26b_vllm_with_layout",
            provider_name="gemma4",
            product_type=ProductType.PARSE,
            config={
                "model": "gemma-4-26b-a4b",
                "prompt_mode": "layout",
            },
        )
    )

    # Gemma 4 31B vLLM — layout mode (div+bbox wrappers, Gemini-style)
    register_fn(
        PipelineSpec(
            pipeline_name="gemma4_31b_vllm_with_layout",
            provider_name="gemma4",
            product_type=ProductType.PARSE,
            config={
                "model": "gemma-4-31b",
                "prompt_mode": "layout",
            },
        )
    )

    # Gemma 4 E4B vLLM — dense 8B variant (4.5B effective)
    register_fn(
        PipelineSpec(
            pipeline_name="gemma4_e4b_vllm",
            provider_name="gemma4",
            product_type=ProductType.PARSE,
            config={
                "model": "gemma-4-e4b",
            },
        )
    )

    # Gemma 4 E4B vLLM — parse with layout (structured output + layout_pages)
    register_fn(
        PipelineSpec(
            pipeline_name="gemma4_e4b_vllm_with_layout",
            provider_name="gemma4",
            product_type=ProductType.PARSE,
            config={
                "model": "gemma-4-e4b",
                "prompt_mode": "layout",
                "swap_bbox": True,
            },
        )
    )

    # =========================================================================
    # Nemotron-3-Nano-Omni 30B-A3B Reasoning (BF16)
    # =========================================================================

    # Thinking enabled. Uses the shared parse prompt (byte-identical to
    # openai/anthropic/gemma4) so this model stays apples-to-apples comparable.
    # Budget mirrors Gemini's `_thinking_high` (max_tokens=65536,
    # reasoning_budget=16384 ≈ 10x output headroom); the server needs
    # max_model_len=131072 to cover prompt + image + output.
    register_fn(
        PipelineSpec(
            pipeline_name="nemotron_omni_30b_vllm_thinking",
            provider_name="nemotron_omni",
            product_type=ProductType.PARSE,
            config={
                "server_url": "",  # Set via NEMOTRON_OMNI_SERVER_URL or override
                "model": "nemotron-omni-30b",
                "enable_thinking": True,
                "temperature": 0.6,
                "top_k": None,
                "top_p": 0.95,
                "max_tokens": 65536,
                "reasoning_budget": 16384,
                # grace_period=1024 per card → thinking_token_budget=16384+1024
                "thinking_token_budget": 17408,
                "timeout": 1800,
            },
        )
    )

    # =========================================================================
    # Chandra OCR 2
    # =========================================================================

    # Chandra OCR 2 vLLM (OpenAI-compatible API, H100)
    register_fn(
        PipelineSpec(
            pipeline_name="chandra2_vllm",
            provider_name="chandra2",
            product_type=ProductType.PARSE,
            config={
                "api_format": "openai",
                "task": "ocr_layout",
            },
        )
    )

    # Chandra OCR 2 SDK (official SDK with built-in layout + output parsing)
    register_fn(
        PipelineSpec(
            pipeline_name="chandra2_sdk",
            provider_name="chandra2",
            product_type=ProductType.PARSE,
            config={
                "api_format": "simple",
                "task": "ocr_layout",
            },
        )
    )

    # =========================================================================
    # Granite Vision
    # =========================================================================

    # Granite Vision pipeline (PP-DocLayout-V3 layout + per-region Granite Vision)
    register_fn(
        PipelineSpec(
            pipeline_name="granite_vision_pipeline",
            provider_name="granite_vision",
            product_type=ProductType.PARSE,
            config={
                "api_format": "simple",
                "task": "ocr",
            },
        )
    )

    # Granite Vision 4.1 4B (ibm-granite/granite-vision-4.1-4b)
    # Native vLLM support. Provider runs three task tags concurrently per
    # image and concatenates outputs -- one pipeline covers tables, charts,
    # and text without per-dataset task selection.
    register_fn(
        PipelineSpec(
            pipeline_name="granite_vision_4_1_4b",
            provider_name="granite_vision",
            product_type=ProductType.PARSE,
            config={
                "api_format": "openai",
                "task": ["ocr", "tables_html", "chart2csv"],
                "served_model_name": "granite-vision-4.1-4b",
            },
        )
    )

    # =========================================================================
    # Gemini - Parse with Layout
    # =========================================================================

    # Gemini 3 Flash - Parse with Layout - Thinking Minimal
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_flash_thinking_minimal_parse_with_layout",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3-flash-preview",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "parse_with_layout",
                "thinking_level": "minimal",
            },
        )
    )

    # Gemini 3 Flash - Parse with Layout - Thinking High
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_flash_thinking_high_parse_with_layout",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3-flash-preview",
                "dpi": 150,
                "max_tokens": 65536,
                "mode": "parse_with_layout",
                "thinking_level": "high",
            },
        )
    )

    # Gemini 3 Flash - Parse with Layout File - Thinking Minimal
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_flash_thinking_minimal_parse_with_layout_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3-flash-preview",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
                "thinking_level": "minimal",
            },
        )
    )

    # Gemini 3 Flash - Parse with Layout File - Thinking High
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_flash_thinking_high_parse_with_layout_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3-flash-preview",
                "max_tokens": 65536,
                "mode": "parse_with_layout_file",
                "thinking_level": "high",
            },
        )
    )

    # Gemini 3.1 Flash Lite - Parse with Layout
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_1_flash_lite_parse_with_layout",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.1-flash-lite-preview",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "parse_with_layout",
            },
        )
    )

    # Gemini 3.1 Flash Lite - Parse with Layout File
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_1_flash_lite_parse_with_layout_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.1-flash-lite-preview",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # Gemini 3.5 Flash Lite - Parse with Layout File
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_5_flash_lite_parse_with_layout_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.5-flash-lite",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # Gemini 3.1 Pro - Parse with Layout File (default thinking)
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_1_pro_parse_with_layout_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.1-pro-preview",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # =========================================================================
    # Gemini 3.5 Flash (GA) - Parse with Layout
    # =========================================================================

    # Gemini 3.5 Flash - Parse with Layout (default thinking)
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_5_flash_parse_with_layout",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.5-flash",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "parse_with_layout",
            },
        )
    )

    # Gemini 3.5 Flash - Parse with Layout - Thinking Minimal
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_5_flash_no_thinking_parse_with_layout",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.5-flash",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "parse_with_layout",
                "thinking_level": "minimal",
            },
        )
    )

    # Gemini 3.5 Flash - Parse with Layout File (default thinking)
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_5_flash_parse_with_layout_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.5-flash",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # Gemini 3.5 Flash - Parse with Layout File - Thinking Minimal
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_5_flash_no_thinking_parse_with_layout_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.5-flash",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
                "thinking_level": "minimal",
            },
        )
    )

    # =========================================================================
    # Gemini 3.6 Flash (GA) - Parse with Layout & Image
    # =========================================================================

    # Gemini 3.6 Flash - Parse with Layout (default thinking)
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_6_flash_parse_with_layout",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.6-flash",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "parse_with_layout",
            },
        )
    )

    # Gemini 3.6 Flash - Parse with Layout - Thinking Minimal
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_6_flash_no_thinking_parse_with_layout",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.6-flash",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "parse_with_layout",
                "thinking_level": "minimal",
            },
        )
    )

    # Gemini 3.6 Flash - Parse with Layout File (default thinking)
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_6_flash_parse_with_layout_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.6-flash",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # Gemini 3.6 Flash - Parse with Layout File - Thinking Minimal
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_6_flash_no_thinking_parse_with_layout_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.6-flash",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
                "thinking_level": "minimal",
            },
        )
    )

    # =========================================================================
    # Gemini 3.7 Flash - Parse with Layout & File
    # =========================================================================

    # Gemini 3.7 Flash - Parse with Layout (default thinking)
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_7_flash_parse_with_layout",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.7-flash",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "parse_with_layout",
            },
        )
    )

    # Gemini 3.7 Flash - Parse with Layout - Thinking Minimal
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_7_flash_no_thinking_parse_with_layout",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.7-flash",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "parse_with_layout",
                "thinking_level": "minimal",
            },
        )
    )

    # Gemini 3.7 Flash - Parse with Layout File (default thinking)
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_7_flash_parse_with_layout_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.7-flash",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # Gemini 3.7 Flash - Parse with Layout File - Thinking Minimal
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_7_flash_no_thinking_parse_with_layout_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3.7-flash",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
                "thinking_level": "minimal",
            },
        )
    )

    # =========================================================================
    # Gemini - Agentic Vision
    # =========================================================================

    # Gemini 3 Flash - Parse with Layout Agentic Vision - Thinking Minimal
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_flash_thinking_minimal_parse_with_layout_agentic_vision",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3-flash-preview",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "parse_with_layout_agentic_vision",
                "thinking_level": "minimal",
                "enable_explicit_context_cache": True,
                "context_cache_ttl_seconds": 900,
                "min_cacheable_tokens": 1024,
            },
        )
    )

    # Gemini 3 Flash - Parse with Layout Agentic Vision - Thinking Medium
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_flash_thinking_medium_parse_with_layout_agentic_vision",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3-flash-preview",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "parse_with_layout_agentic_vision",
                "thinking_level": "medium",
                "enable_explicit_context_cache": True,
                "context_cache_ttl_seconds": 900,
                "min_cacheable_tokens": 1024,
            },
        )
    )

    # Gemini 3 Flash - Parse with Layout Agentic Vision - Thinking High
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_flash_thinking_high_parse_with_layout_agentic_vision",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3-flash-preview",
                "dpi": 150,
                "max_tokens": 65536,
                "mode": "parse_with_layout_agentic_vision",
                "thinking_level": "high",
                "enable_explicit_context_cache": True,
                "context_cache_ttl_seconds": 900,
                "min_cacheable_tokens": 1024,
            },
        )
    )

    # =========================================================================
    # Gemini - File Mode (additional thinking variants)
    # =========================================================================

    # Gemini 3 Flash - File Mode - Thinking Minimal
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_flash_thinking_minimal_parse_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3-flash-preview",
                "max_tokens": 32768,
                "mode": "file",
                "thinking_level": "minimal",
            },
        )
    )

    # Gemini 3 Flash - File Mode - Thinking High
    register_fn(
        PipelineSpec(
            pipeline_name="google_gemini_3_flash_thinking_high_parse_file",
            provider_name="google",
            product_type=ProductType.PARSE,
            config={
                "model": "gemini-3-flash-preview",
                "max_tokens": 65536,
                "mode": "file",
                "thinking_level": "high",
            },
        )
    )

    # =========================================================================
    # OpenAI - Parse with Layout
    # =========================================================================

    # OpenAI GPT-5 Mini - Parse with Layout (default reasoning)
    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt5_mini_reasoning_medium_parse_with_layout",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5-mini",
                "dpi": 150,
                "max_tokens": 65536,
                "mode": "parse_with_layout",
            },
        )
    )

    # OpenAI GPT-5 Mini - Parse with Layout - Reasoning Minimal
    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt5_mini_reasoning_minimal_parse_with_layout",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5-mini",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "parse_with_layout",
                "reasoning_effort": "minimal",
            },
        )
    )

    # OpenAI GPT-5 Mini - Parse with Layout File
    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt5_mini_reasoning_medium_parse_with_layout_file",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5-mini",
                "max_tokens": 65536,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # OpenAI GPT-5 Mini - Parse with Layout File - Reasoning Minimal
    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt5_mini_reasoning_minimal_parse_with_layout_file",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5-mini",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
                "reasoning_effort": "minimal",
            },
        )
    )

    # OpenAI GPT-5.4 - Parse with Layout File (default reasoning)
    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt_5_4_parse_with_layout_file",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5.4-2026-03-05",
                "max_tokens": 65536,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # OpenAI GPT-5.5 - Parse with Layout File - Reasoning Medium
    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt_5_5_reasoning_medium_parse_with_layout_file",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5.5",
                "max_tokens": 65536,
                "mode": "parse_with_layout_file",
                "reasoning_effort": "medium",
            },
        )
    )

    # OpenAI GPT-5.5 - Parse with Layout File - Reasoning None
    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt_5_5_reasoning_none_parse_with_layout_file",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5.5",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
                "reasoning_effort": "none",
            },
        )
    )

    # OpenAI GPT-5.6 (sol / terra / luna) - Parse with Layout File - Reasoning None
    for _gpt56_suffix, _gpt56_model in (
        ("sol", "gpt-5.6-sol"),
        ("terra", "gpt-5.6-terra"),
        ("luna", "gpt-5.6-luna"),
    ):
        register_fn(
            PipelineSpec(
                pipeline_name=(f"openai_gpt_5_6_{_gpt56_suffix}_reasoning_none_parse_with_layout_file"),
                provider_name="openai",
                product_type=ProductType.PARSE,
                config={
                    "model": _gpt56_model,
                    "max_tokens": 32768,
                    "mode": "parse_with_layout_file",
                    "reasoning_effort": "none",
                },
            )
        )

    # OpenAI GPT-5.4 Nano - Parse with Layout
    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt_5_4_nano_parse_with_layout",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5.4-nano",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "parse_with_layout",
            },
        )
    )

    # OpenAI GPT-5.4 Nano - Parse with Layout File
    register_fn(
        PipelineSpec(
            pipeline_name="openai_gpt_5_4_nano_parse_with_layout_file",
            provider_name="openai",
            product_type=ProductType.PARSE,
            config={
                "model": "gpt-5.4-nano",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # =========================================================================
    # Anthropic - Parse with Layout
    # =========================================================================

    # Anthropic Haiku - Parse with Layout
    register_fn(
        PipelineSpec(
            pipeline_name="anthropic_haiku_parse_with_layout",
            provider_name="anthropic",
            product_type=ProductType.PARSE,
            config={
                "model": "claude-haiku-4-5-20251001",
                "dpi": 150,
                "max_tokens": 32768,
                "mode": "parse_with_layout",
            },
        )
    )

    # Anthropic Haiku - Parse with Layout File
    register_fn(
        PipelineSpec(
            pipeline_name="anthropic_haiku_parse_with_layout_file",
            provider_name="anthropic",
            product_type=ProductType.PARSE,
            config={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # Anthropic Opus 4.6 - Parse with Layout File
    register_fn(
        PipelineSpec(
            pipeline_name="anthropic_opus_4_6_parse_with_layout_file",
            provider_name="anthropic",
            product_type=ProductType.PARSE,
            config={
                "model": "claude-opus-4-6",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # Anthropic Opus 4.7 - Parse with Layout File
    register_fn(
        PipelineSpec(
            pipeline_name="anthropic_opus_4_7_parse_with_layout_file",
            provider_name="anthropic",
            product_type=ProductType.PARSE,
            config={
                "model": "claude-opus-4-7",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # Anthropic Opus 4.8 - Parse with Layout File
    register_fn(
        PipelineSpec(
            pipeline_name="anthropic_opus_4_8_parse_with_layout_file",
            provider_name="anthropic",
            product_type=ProductType.PARSE,
            config={
                "model": "claude-opus-4-8",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # Anthropic Sonnet 5 - Parse with Layout File - Adaptive Thinking
    register_fn(
        PipelineSpec(
            pipeline_name="anthropic_sonnet_5_parse_with_layout_file",
            provider_name="anthropic",
            product_type=ProductType.PARSE,
            config={
                "model": "claude-sonnet-5",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
                "thinking": {"type": "adaptive"},
            },
        )
    )

    # Anthropic Fable 5 - Parse with Layout File
    register_fn(
        PipelineSpec(
            pipeline_name="anthropic_fable_5_parse_with_layout_file",
            provider_name="anthropic",
            product_type=ProductType.PARSE,
            config={
                "model": "claude-fable-5",
                "max_tokens": 32768,
                "mode": "parse_with_layout_file",
            },
        )
    )

    # Anthropic Haiku - Parse with Layout File - Thinking
    register_fn(
        PipelineSpec(
            pipeline_name="anthropic_haiku_thinking_parse_with_layout_file",
            provider_name="anthropic",
            product_type=ProductType.PARSE,
            config={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 64000,
                "mode": "parse_with_layout_file",
                "thinking": {"type": "enabled", "budget_tokens": 32768},
            },
        )
    )

    # =========================================================================
    # Reducto - Non-agentic
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="reducto",
            provider_name="reducto",
            product_type=ProductType.PARSE,
            config={
                "ocr_system": "standard",
                "agentic": False,
                "table_output_format": "html",
            },
        )
    )

    # =========================================================================
    # MinerU 2.5 (opendatalab/MinerU2.5-2509-1.2B, 1.2B Qwen2-VL derivative)
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="kdl_frontier_nano",
            provider_name="kdl_frontier_nano",
            product_type=ProductType.PARSE,
            config={
                "endpoint_url": "",  # via KDL_NANO_ENDPOINT_URL
                "model": "",  # via KDL_NANO_MODEL (default kdl-frontier-parser-nano)
                "dpi": 144,
                "timeout": 900,
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="mineru25_vllm",
            provider_name="mineru25",
            product_type=ProductType.PARSE,
            config={
                "server_url": "",  # Set via MINERU25_SERVER_URL or override
            },
        )
    )

    # =========================================================================
    # MinerU 2.5 Pro 2605 (opendatalab/MinerU2.5-Pro-2605-1.2B)
    # Same 1.2B Qwen2-VL arch as mineru25, newer checkpoint with improved
    # layout detection + chart/image analysis (server runs the official
    # image_analysis client flag the old deployment was missing).
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="mineru2605pro_vllm",
            provider_name="mineru2605pro",
            product_type=ProductType.PARSE,
            config={
                "server_url": "",  # Set via MINERU2605PRO_SERVER_URL or override
            },
        )
    )

    # =========================================================================
    # MinerU-Diffusion-V1 (opendatalab/MinerU-Diffusion-V1-0320-2.5B)
    # Diffusion-decoding OCR VLM. Runs the official two-stage parse: layout
    # detection, then per-region recognition (table -> OTSL, formula -> LaTeX,
    # else text), merged into markdown with OTSL converted to HTML server-side.
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="mineru_diffusion",
            provider_name="mineru_diffusion",
            product_type=ProductType.PARSE,
            config={
                "server_url": "",  # Set via MINERU_DIFFUSION_SERVER_URL or override
            },
        )
    )

    # =========================================================================
    # Surya OCR 2 (datalab-to/surya-ocr-2, 650M VLM)
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="surya2_sdk",
            provider_name="surya2",
            product_type=ProductType.PARSE,
            config={
                "server_url": "",  # Set via SURYA2_SERVER_URL or override
            },
        )
    )

    # =========================================================================
    # Databricks ai_parse_document
    # =========================================================================

    # Infinity-Parser2-Flash (infly/Infinity-Parser2-Flash, vLLM server)
    register_fn(
        PipelineSpec(
            pipeline_name="infinity_parser2_flash",
            provider_name="infinity_parser2",
            product_type=ProductType.PARSE,
            config={
                "model_name": "infly/Infinity-Parser2-Flash",
                "backend": "vllm-server",
                "task_type": "doc2json",
                "output_format": "json",
            },
        )
    )

    # Infinity-Parser2-Pro (infly/Infinity-Parser2-Pro, vLLM server)
    register_fn(
        PipelineSpec(
            pipeline_name="infinity_parser2_pro",
            provider_name="infinity_parser2",
            product_type=ProductType.PARSE,
            config={
                "model_name": "infly/Infinity-Parser2-Pro",
                "backend": "vllm-server",
                "task_type": "doc2json",
                "output_format": "json",
            },
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="databricks_ai_parse",
            provider_name="databricks_ai_parse",
            product_type=ProductType.PARSE,
            config={
                "version": "2.0",
            },
        )
    )

    # Batched variant: Databricks' recommended operating mode — submit the
    # whole dataset as ONE SQL statement so warehouse spin-up/idle is paid
    # once instead of per micro-batch. Model DBUs are unchanged (per-page
    # billing). Run with max_concurrent >= dataset size so every request is
    # queued before the debounce window flushes.
    register_fn(
        PipelineSpec(
            pipeline_name="databricks_ai_parse_batch",
            provider_name="databricks_ai_parse",
            product_type=ProductType.PARSE,
            config={
                "version": "2.0",
                "batch_size": 1000,
                "batch_wait_seconds": 120,
                "timeout": 7200,
                "per_request_timeout": 9000,
            },
        )
    )

    # =========================================================================
    # Mistral OCR Pipelines
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="mistral_ocr_4",
            provider_name="mistral_ocr",
            product_type=ProductType.PARSE,
            config={
                "model": "mistral-ocr-4-0",
                "include_blocks": True,
                "max_pages": 50,
            },
        )
    )

    # Annotation ("Document AI") mode: same /v1/ocr endpoint with
    # ``bbox_annotation_format`` so each figure's data is transcribed into the
    # markdown (charts -> data tables). Billed at $5/1000 annotated pages.
    # Annotation runs a vision pass per figure, so it is heavier and hits 429s
    # more readily than plain OCR — give it a larger Retry-After backoff budget
    # (the provider already honors the server's Retry-After header).
    register_fn(
        PipelineSpec(
            pipeline_name="mistral_ocr_4_annotation",
            provider_name="mistral_ocr",
            product_type=ProductType.PARSE,
            config={
                "model": "mistral-ocr-4-0",
                "include_blocks": True,
                "bbox_annotation": True,
                "max_pages": 50,
                "rate_limit_retries": 8,
                "rate_limit_base_wait": 2.0,
                "rate_limit_max_wait": 30.0,
            },
        )
    )

    # =========================================================================
    # OpenInnovation Parser (oi-parser)
    # =========================================================================
    register_fn(
        PipelineSpec(
            pipeline_name="oi_parser",
            provider_name="oi_parser",
            product_type=ProductType.PARSE,
            # Endpoint resolves in the provider: OI_PARSER_BASE_URL env → the provider's
            # default public endpoint. Left unset here so the env override is honored
            # (matches the LlamaParse configurable-base-URL convention).
            config={},
        )
    )
