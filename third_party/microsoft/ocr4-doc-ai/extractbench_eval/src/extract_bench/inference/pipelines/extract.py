"""Extract pipelines - structured data extraction from documents.

The registered set covers LlamaExtract tiers, one-shot VLM/LLM APIs, two-stage
(parse -> text extract) baselines, coding agents, commercial extraction APIs,
and open-weight pipelines.
"""

from typing import Any

from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.product import ProductType

GRANULAR_BBOX_OUTPUT_OPTIONS = {
    "granular_bboxes": ["word"],
}


def _granular_parse_config(tier: str) -> dict[str, Any]:
    """Parse config emitting word-level boxes at the given tier."""
    return {
        "tier": tier,
        "version": "latest",
        "disable_cache": True,
        "output_options": GRANULAR_BBOX_OUTPUT_OPTIONS,
    }


def _pipeline_spec(
    *,
    pipeline_name: str,
    provider_name: str,
    config: dict[str, Any],
    per_file_timeout: float | None = None,
) -> PipelineSpec:
    return PipelineSpec(
        pipeline_name=pipeline_name,
        provider_name=provider_name,
        product_type=ProductType.EXTRACT,
        config=config,
        per_file_timeout=per_file_timeout,
    )


def register_extract_pipelines(register_fn) -> None:  # type: ignore[no-untyped-def]
    """Register the public extract pipeline roster."""

    # =========================================================================
    # LlamaExtract (hosted V2 extract API, /api/v2/extract)
    # =========================================================================

    for _tier in ("cost_effective", "agentic", "agentic_plus"):
        register_fn(
            _pipeline_spec(
                pipeline_name=f"llamaextract_{_tier}",
                provider_name="llamaextract_v2",
                config={
                    "tier": _tier,
                    "parse_tier": _tier,
                    "timeout": 3000,
                    "cite_sources": True,
                    "parse_config": _granular_parse_config(_tier),
                },
                # The provider polls the hosted job for up to `timeout` seconds; give
                # the runner-level watchdog headroom past that so long documents are
                # not killed (and re-submitted) at the 600s run default.
                per_file_timeout=3600.0,
            )
        )

    for _tier in ("cost_effective", "agentic"):
        register_fn(
            _pipeline_spec(
                pipeline_name=f"llamaextract_{_tier}_standard_bbox",
                provider_name="llamaextract_v2",
                config={
                    "tier": _tier,
                    "cite_sources": True,
                },
            )
        )

    # =========================================================================
    # One-shot VLM/LLM APIs (schema-guided structured output over the file)
    # =========================================================================

    register_fn(
        _pipeline_spec(
            pipeline_name="openai_gpt_5_4_extract_oneshot_structured_output_file",
            provider_name="openai_extract",
            config={
                "model": "gpt-5.4",
                "additional_properties_false": True,
            },
        )
    )

    register_fn(
        _pipeline_spec(
            pipeline_name="openai_gpt_5_4_nano_extract_oneshot_structured_output_file",
            provider_name="openai_extract",
            config={
                "model": "gpt-5.4-nano",
                "additional_properties_false": True,
            },
        )
    )

    register_fn(
        _pipeline_spec(
            pipeline_name="gemini_3_5_flash_extract_oneshot_structured_output_file",
            provider_name="gemini_extract",
            config={
                "model": "gemini-3.5-flash",
                "additional_properties_false": True,
                "thinking_level": "low",
            },
        )
    )

    register_fn(
        _pipeline_spec(
            pipeline_name="gemini_3_6_flash_extract_oneshot_structured_output_file",
            provider_name="gemini_extract",
            config={
                "model": "gemini-3.6-flash",
                "additional_properties_false": True,
                "thinking_level": "medium",
                "max_tokens": 65536,
            },
        )
    )

    register_fn(
        _pipeline_spec(
            pipeline_name="gemini_3_7_flash_extract_oneshot_structured_output_file",
            provider_name="gemini_extract",
            config={
                "model": "gemini-3.7-flash",
                "additional_properties_false": True,
                "thinking_level": "medium",
                "max_tokens": 65536,
            },
        )
    )

    register_fn(
        _pipeline_spec(
            pipeline_name="anthropic_haiku_4_5_extract_oneshot_structured_output_file",
            provider_name="anthropic_extract",
            config={
                "model": "claude-haiku-4-5-20251001",
                "additional_properties_false": True,
                "all_properties_required": True,
                "max_tokens": 32768,
            },
        )
    )

    # =========================================================================
    # Two-stage baselines (LlamaParse agentic markdown -> text extract)
    # =========================================================================
    # Same models and structured-output settings as the *_oneshot_*_file
    # pipelines above, but the model receives per-page parsed markdown as text
    # instead of the uploaded/inlined file. cost_usd totals parse + extract,
    # with parse_cost_usd / extract_cost_usd breakdowns.

    for _prefix, _provider_name, _model_config in (
        ("openai_gpt_5_4", "openai_extract", {"model": "gpt-5.4"}),
        ("openai_gpt_5_4_nano", "openai_extract", {"model": "gpt-5.4-nano"}),
        ("gemini_3_5_flash", "gemini_extract", {"model": "gemini-3.5-flash", "thinking_level": "low"}),
        (
            "gemini_3_6_flash",
            "gemini_extract",
            {"model": "gemini-3.6-flash", "thinking_level": "medium", "max_tokens": 65536},
        ),
        (
            "gemini_3_7_flash",
            "gemini_extract",
            {"model": "gemini-3.7-flash", "thinking_level": "medium", "max_tokens": 65536},
        ),
        (
            "anthropic_haiku_4_5",
            "anthropic_extract",
            {"model": "claude-haiku-4-5-20251001", "all_properties_required": True},
        ),
    ):
        register_fn(
            _pipeline_spec(
                pipeline_name=f"{_prefix}_extract_twostage_parse_agentic_structured_output_text",
                provider_name=_provider_name,
                config={
                    **_model_config,
                    "additional_properties_false": True,
                    "input_mode": "parsed_text",
                    "parse_source": {
                        "type": "llamaparse",
                        "tier": "agentic",
                        "version": "latest",
                    },
                },
            )
        )

    register_fn(
        _pipeline_spec(
            pipeline_name="deepseek_v4_pro_extract_twostage_parse_agentic_structured_output_text",
            provider_name="deepseek_extract",
            config={
                "model": "accounts/fireworks/models/deepseek-v4-pro",
                "base_url": "https://api.fireworks.ai/inference/v1",
                "input_mode": "parsed_text",
                "thinking_type": "disabled",
                "max_tokens": 131072,
                "max_cost_usd": 5.00,
            },
        )
    )

    # =========================================================================
    # Coding agents driving extraction (agentic CLI loops)
    # =========================================================================

    # These run a model-driven CLI agent on your machine against benchmark
    # documents. Run them in a container or VM.
    register_fn(
        _pipeline_spec(
            pipeline_name="claude_code_extract_opus_4_8",
            provider_name="claude_code_extract",
            config={"model": "claude-opus-4-8", "bare": True},
        )
    )

    for _model_slug, _model, _reasoning_effort in (
        ("gpt_5_4", "gpt-5.4", "low"),
        ("gpt_5_5", "gpt-5.5", "low"),
        ("gpt_5_5", "gpt-5.5", "high"),
    ):
        register_fn(
            _pipeline_spec(
                pipeline_name=f"codex_code_extract_{_model_slug}_{_reasoning_effort}",
                provider_name="codex_code_extract",
                config={
                    "model": _model,
                    "reasoning_effort": _reasoning_effort,
                    "sandbox": "workspace-write",
                    "max_cost_usd": 5.00,
                },
            )
        )

    # =========================================================================
    # Commercial extraction APIs
    # =========================================================================

    register_fn(
        _pipeline_spec(
            pipeline_name="reducto_extract",
            provider_name="reducto_extract",
            config={
                "citations": False,
                "array_extract": False,
            },
        )
    )

    register_fn(
        _pipeline_spec(
            pipeline_name="reducto_deep_extract",
            provider_name="reducto_extract",
            config={
                "citations": True,
                "array_extract": False,
                "deep_extract": True,
            },
        )
    )

    register_fn(
        _pipeline_spec(
            pipeline_name="extend_extract",
            provider_name="extend",
            config={
                "baseProcessor": "extraction_performance",
                "baseVersion": "4.1.1",
                "advancedOptions": {
                    "citationsEnabled": True,
                    "advancedFigureParsingEnabled": True,
                },
            },
        )
    )

    register_fn(
        _pipeline_spec(
            pipeline_name="extend_extract_max",
            provider_name="extend",
            config={
                "baseProcessor": "extraction_performance",
                "advancedOptions": {
                    "citationsEnabled": True,
                    "advancedFigureParsingEnabled": True,
                    "arrayStrategy": {"type": "large_array_max_context"},
                },
            },
        )
    )

    register_fn(
        _pipeline_spec(
            pipeline_name="landingai_extract",
            provider_name="landingai_extract",
            config={},
        )
    )

    for _extraction_mode in ("fast", "balanced"):
        register_fn(
            _pipeline_spec(
                pipeline_name=f"datalab_parse_accurate_extract_{_extraction_mode}",
                provider_name="datalab_extract",
                config={
                    "parse_mode": "accurate",
                    "extraction_mode": _extraction_mode,
                    "output_format": "json",
                },
            )
        )

    # =========================================================================
    # Open-weight pipelines (self-hosted)
    # =========================================================================
    # These have no vendor API. Each reads the URL of a deployment you run
    # yourself from the env var named below (or a `server_url` config override);
    # there are no default endpoints. See .env.example.

    # lift SDK /extract deployment (LIFT_ENDPOINT_URL).
    register_fn(
        _pipeline_spec(
            pipeline_name="lift_extract",
            provider_name="lift_extract",
            config={},
        )
    )

    # One-shot structured output through a self-hosted vLLM vision model, the
    # same contract as the *_oneshot_structured_output_file cloud pipelines.
    register_fn(
        _pipeline_spec(
            pipeline_name="qwen3_6_35b_a3b_fp8_vllm_extract_oneshot_structured_output_file",
            provider_name="vllm_extract",
            config={
                "model": "qwen3.6-35b-a3b-fp8",
                "endpoint_env_var": "QWEN35_SERVER_URL",
                "additional_properties_false": True,
                # Large multi-page docs produce large JSON; server max_model_len
                # is 1.01M (262k native, extended via YaRN).
                "max_tokens": 65536,
            },
        )
    )

    register_fn(
        _pipeline_spec(
            pipeline_name="gemma4_26b_vllm_extract_oneshot_structured_output_file",
            provider_name="vllm_extract",
            config={
                "model": "gemma-4-26b-a4b",
                "endpoint_env_var": "GEMMA4_SERVER_URL",
                "additional_properties_false": True,
                # 100 instead of the provider's 150 default: a letter page is
                # 850x1100 rather than 1275x1650, so 2.25x fewer pixels to
                # rasterize, base64, and ship. Long scanned documents were
                # exhausting a memory-limited runner and taking long enough that
                # the endpoint expired the request.
                "dpi": 100,
                # Large multi-page docs produce large JSON; server max_model_len is 256k.
                "max_tokens": 65536,
                # Long documents need more than the provider's 15-minute
                # default, and SDK retries are disabled to avoid duplicate work.
                "timeout_s": 3600,
                # json_object (not full-schema guided decoding): the extract schemas
                # are large enough that xgrammar's per-step mask compute starves the
                # GPU (~1.5 tok/s at concurrency). json_object keeps syntactic-JSON
                # enforcement + schema-in-prompt without the heavy grammar.
                "structured_output": False,
            },
        )
    )

    # NuExtract3 extracts natively from a template rather than a JSON Schema;
    # the provider converts the schema before the call (NUEXTRACT3_SERVER_URL).
    register_fn(
        _pipeline_spec(
            pipeline_name="nuextract3_extract",
            provider_name="nuextract3_extract",
            config={"endpoint_env_var": "NUEXTRACT3_SERVER_URL"},
        )
    )

    # =========================================================================
    # Mistral OCR-4 two-stage pipelines (OCR-4 parse → Mistral LLM extract)
    # Requires: AZURE_MISTRAL_DOCUMENT_AI_KEY, AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT,
    #           MISTRAL_API_KEY env vars.
    # =========================================================================

    register_fn(
        _pipeline_spec(
            pipeline_name="mistral_ocr4_large_extract",
            provider_name="mistral_ocr4_extract",
            config={
                "extract_model": "mistral-large-latest",
                "max_tokens": 32768,
                "additional_properties_false": True,
            },
        )
    )

    register_fn(
        _pipeline_spec(
            pipeline_name="mistral_ocr4_medium_extract",
            provider_name="mistral_ocr4_extract",
            config={
                "extract_model": "mistral-medium-3-5",
                "max_tokens": 32768,
                "additional_properties_false": True,
            },
        )
    )
