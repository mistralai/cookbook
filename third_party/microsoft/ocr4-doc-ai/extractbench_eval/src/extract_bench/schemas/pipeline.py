from typing import Any

from pydantic import BaseModel, Field

from extract_bench.schemas.product import ProductType


class PipelineSpec(BaseModel):
    """Specification for a pipeline configuration."""

    pipeline_name: str = Field(description="Name of this pipeline")
    provider_name: str = Field(description="Name of the provider (e.g., 'llama', 'openai')")
    product_type: ProductType = Field(description="Type of product task (parse or extract)")
    config: dict[str, Any] = Field(default_factory=dict, description="Configuration dictionary to pass to the provider")
    per_file_timeout: float | None = Field(
        default=None,
        description=(
            "Optional per-pipeline override (seconds) for the runner's per-file inference timeout. "
            "When set, it takes precedence over the run-level default / --per_file_timeout. "
            "None → use the run-level value. Kept off `config` so providers never receive it."
        ),
    )
