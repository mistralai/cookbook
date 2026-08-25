from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, Discriminator, Field

from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.layout_detection_output import LayoutOutput
from extract_bench.schemas.parse_output import ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.product import ProductType


class InputFile(BaseModel):
    """One role-tagged input file for a multi-file extract request.

    Used when a pipeline merges a GROUP of files (a base document plus N
    amendments) into a single master output. ``role`` is ``"base"`` or
    ``"amendment"``; ``order`` is the apply order (base=0, amendments 1..N).
    """

    path: str = Field(description="Local filesystem path to the input file")
    role: str | None = Field(default=None, description="Role tag: 'base' or 'amendment'")
    order: int | None = Field(default=None, description="Apply order (base=0, amendments 1..N)")
    filename: str | None = Field(
        default=None, description="Display filename (defaults to the path basename when omitted)"
    )


class InferenceRequest(BaseModel):
    """Request for running inference on a document."""

    example_id: str = Field(description="Unique identifier for the example")
    source_file_path: str = Field(description="Path to the source file (PDF, etc.)")
    product_type: ProductType = Field(description="Type of product task to run")
    input_files: list[InputFile] | None = Field(
        default=None,
        description=(
            "Optional ordered, role-tagged input files for multi-file extract "
            "pipelines (base + amendments -> merged master). When None "
            "(the single-file case), the provider uses ``source_file_path`` "
            "alone and treats it as the base (role='base', order 0)."
        ),
    )
    schema_override: dict[str, Any] | None = Field(
        default=None,
        description="Optional schema override (for EXTRACT product type)",
    )
    config_override: dict[str, Any] | None = Field(
        default=None,
        description=("Optional configuration override to merge with pipeline config"),
    )
    trace_metadata: dict[str, str] | None = Field(
        default=None,
        description="Metadata to forward to trace spans (unused when tracing is disabled)",
    )
    trace_tags: list[str] | None = Field(
        default=None,
        description="Tags to forward to trace spans (unused when tracing is disabled)",
    )


PipelineOutputType = Annotated[
    ParseOutput | LayoutOutput | ExtractOutput,
    Discriminator("task_type"),
]


class RawInferenceResult(BaseModel):
    """Raw result from provider before normalization."""

    request: InferenceRequest = Field(description="Original inference request")
    pipeline: PipelineSpec = Field(description="Pipeline used")
    pipeline_name: str = Field(description="Name of the pipeline used")
    product_type: ProductType = Field(description="Type of product task that was run")
    raw_output: dict = Field(description="Raw output from the provider API")
    started_at: datetime = Field(description="Timestamp when inference started")
    completed_at: datetime = Field(description="Timestamp when inference completed")
    latency_in_ms: int = Field(ge=0, description="Latency in milliseconds")


class InferenceResult(BaseModel):
    """Result of running inference on a document with both raw and normalized outputs."""

    request: InferenceRequest = Field(description="Original inference request")
    pipeline_name: str = Field(description="Name of the pipeline used")
    product_type: ProductType = Field(description="Type of product task that was run")

    # Both outputs stored here
    raw_output: dict = Field(description="Raw output from the provider (for debugging/re-normalization)")
    output: PipelineOutputType = Field(description="Normalized output from the pipeline")

    # metadata
    started_at: datetime = Field(description="Timestamp when inference started")
    completed_at: datetime = Field(description="Timestamp when inference completed")
    latency_in_ms: int = Field(ge=0, description="Latency in milliseconds")
