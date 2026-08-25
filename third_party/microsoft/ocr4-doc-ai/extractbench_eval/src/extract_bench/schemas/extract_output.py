"""
Normalized output schema for extract tasks.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class FieldCitation(BaseModel):
    """Normalized bbox evidence for an extracted field."""

    field_path: str = Field(description="Dotted path of the extracted field this citation supports")
    page: int = Field(ge=1, description="1-indexed page number")
    bbox: list[float] | None = Field(
        default=None,
        description="Normalized COCO [x, y, width, height] bbox. None for page-only citations.",
    )
    polygon: list[list[float]] | None = Field(default=None, description="Normalized polygon points, when available")
    reference_text: str | None = Field(default=None, description="Provider reference text for the citation")
    confidence: float | None = Field(default=None, description="Provider confidence score, when available")
    source: str | None = Field(default=None, description="Provider/source label for debugging")
    metadata: dict[str, Any] | None = Field(default=None, description="Provider-specific citation metadata")


class ExtractOutput(BaseModel):
    """Normalized output for extract tasks."""

    task_type: Literal["extract"] = Field(default="extract", frozen=True, description="Task type discriminator")
    example_id: str = Field(description="Unique identifier for the example")
    pipeline_name: str = Field(description="Name of the pipeline that produced this output")
    extracted_data: dict[str, Any] | list[dict[str, Any]] = Field(
        default_factory=lambda: {},
        description="Extracted structured data (dict for single extraction, list for per-page/per-row)",
    )
    field_citations: list[FieldCitation] = Field(
        default_factory=list,
        description="Normalized field-level citation bboxes from the provider",
    )
