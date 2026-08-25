"""Layout detection pipelines - document layout analysis and bounding boxes."""

from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.product import ProductType


def register_layout_pipelines(register_fn) -> None:  # type: ignore[no-untyped-def]
    """Register all layout detection pipelines."""

    # =========================================================================
    # Docling RT-DETR Pipelines
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="docling_layout_old",
            provider_name="docling_layout",
            product_type=ProductType.LAYOUT_DETECTION,
            config={},  # Endpoint URL read from config.yaml
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="docling_layout_heron_101",
            provider_name="docling_layout_heron_101",
            product_type=ProductType.LAYOUT_DETECTION,
            config={},
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="docling_layout_heron",
            provider_name="docling_layout_heron",
            product_type=ProductType.LAYOUT_DETECTION,
            config={},
        )
    )

    # =========================================================================
    # YOLO Layout Pipelines
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="yolo_doclaynet",
            provider_name="yolo_layout",
            product_type=ProductType.LAYOUT_DETECTION,
            config={},
        )
    )

    # =========================================================================
    # Paddle Layout Pipelines
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="ppdoclayout_plus_l",
            provider_name="paddle_layout",
            product_type=ProductType.LAYOUT_DETECTION,
            config={},  # Endpoint URL read from config.yaml
        )
    )

    # =========================================================================
    # VLM-based Layout Pipelines
    # =========================================================================

    # Layout+content mode (structured JSON with bboxes + text)
    register_fn(
        PipelineSpec(
            pipeline_name="qwen3vl_layout",
            provider_name="qwen3vl_layout",
            product_type=ProductType.PARSE,
            config={},
        )
    )

    # Pure parse mode (markdown output, no layout — fair comparison with GPT/Claude/Gemini)
    register_fn(
        PipelineSpec(
            pipeline_name="qwen3vl_parse",
            provider_name="qwen3vl_layout",
            product_type=ProductType.PARSE,
            config={"prompt_mode": "parse"},
        )
    )

    # =========================================================================
    # Surya/Chandra OCR Layout Pipelines
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="surya_layout",
            provider_name="surya_layout",
            product_type=ProductType.LAYOUT_DETECTION,
            config={},  # endpoint from SURYA_ENDPOINT_URL (self-hosted server required)
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="chandra_layout",
            provider_name="chandra_layout",
            product_type=ProductType.LAYOUT_DETECTION,
            config={},  # endpoint from CHANDRA_ENDPOINT_URL (self-hosted server required)
        )
    )

    # =========================================================================
    # Layout V3 Pipelines (RT-DETRv2 + Figure Classification)
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="layout_v3",
            provider_name="layout_v3",
            product_type=ProductType.LAYOUT_DETECTION,
            config={},  # Uses default endpoint URL from provider
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="layout_v3_byoc_cpu",
            provider_name="layout_v3_byoc_cpu",
            product_type=ProductType.LAYOUT_DETECTION,
            config={},  # Uses LAYOUT_V3_BYOC_CPU_URL env var or localhost:8001
        )
    )

    register_fn(
        PipelineSpec(
            pipeline_name="layout_v3_byoc_gpu",
            provider_name="layout_v3_byoc_gpu",
            product_type=ProductType.LAYOUT_DETECTION,
            config={},  # Uses LAYOUT_V3_BYOC_GPU_URL env var or localhost:8002
        )
    )

    # =========================================================================
    # LlamaParse Layout Detection (uses V3 labels)
    # =========================================================================

    register_fn(
        PipelineSpec(
            pipeline_name="llamaparse_agentic_layout",
            provider_name="llamaparse",
            product_type=ProductType.LAYOUT_DETECTION,
            config={
                "max_pages": 25,
                "invalidate_cache": True,
                "tier": "agentic",
                "version": "latest",
                "high_res_ocr": True,
                "adaptive_long_table": True,
                "outlined_table_extraction": True,
                "output_tables_as_HTML": True,
                "precise_bounding_box": True,
            },
        )
    )
