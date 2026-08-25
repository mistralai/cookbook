"""Pipeline registry for named pipeline configurations.

Pipelines are organized by product type:
- parse.py: PDF/document parsing (LlamaParse, Reducto, Datalab, etc.)
- layout.py: Layout detection (Docling, YOLO, Surya, etc.)
"""

from extract_bench.schemas.pipeline import PipelineSpec

# Registry of named pipelines (pipeline_name -> PipelineSpec)
_PIPELINE_REGISTRY: dict[str, PipelineSpec] = {}


def register_pipeline(pipeline: PipelineSpec) -> None:
    """
    Register a pipeline configuration.

    :param pipeline: PipelineSpec to register (uses pipeline.pipeline_name as the registry key)
    :raises ValueError: If pipeline is already registered
    """
    pipeline_name = pipeline.pipeline_name
    if pipeline_name in _PIPELINE_REGISTRY:
        raise ValueError(f"Pipeline already registered: '{pipeline_name}'")

    _PIPELINE_REGISTRY[pipeline_name] = pipeline


def get_pipeline(pipeline_name: str) -> PipelineSpec:
    """
    Get a pipeline specification by name.

    :param pipeline_name: Name of the pipeline (e.g., "llamaextract_agentic_plus")
    :return: PipelineSpec
    :raises ValueError: If pipeline is not registered
    """
    if pipeline_name not in _PIPELINE_REGISTRY:
        available = ", ".join(sorted(_PIPELINE_REGISTRY.keys()))
        raise ValueError(f"Pipeline '{pipeline_name}' not found. Available pipelines: {available}")
    return _PIPELINE_REGISTRY[pipeline_name]


def list_pipelines() -> list[str]:
    """List all registered pipeline names."""
    return sorted(_PIPELINE_REGISTRY.keys())


def _register_builtin_pipelines() -> None:
    """Register all built-in pipeline configurations from submodules."""
    from extract_bench.inference.pipelines.extract import register_extract_pipelines
    from extract_bench.inference.pipelines.layout import register_layout_pipelines
    from extract_bench.inference.pipelines.parse import register_parse_pipelines

    register_parse_pipelines(register_pipeline)
    register_layout_pipelines(register_pipeline)
    register_extract_pipelines(register_pipeline)


# Auto-register built-in pipelines on import
_register_builtin_pipelines()
