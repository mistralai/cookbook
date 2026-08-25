"""
Normalized output schema for parse tasks.
"""

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class PageIR(BaseModel):
    """Intermediate representation of a single page."""

    page_index: int = Field(ge=0, description="0-indexed page number")
    markdown: str = Field(description="Markdown content of the page")


class LayoutSegmentIR(BaseModel):
    """Normalized literal layout segment coordinates and attribution span metadata."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    x: float = Field(
        description="Literal unrotated bbox x coordinate. Normalize before storage when page coords are [0,1]."
    )
    y: float = Field(
        description="Literal unrotated bbox y coordinate. Normalize before storage when page coords are [0,1]."
    )
    w: float = Field(description="Literal unrotated bbox width. Normalize by page width when page coords are [0,1].")
    h: float = Field(description="Literal unrotated bbox height. Normalize by page height when page coords are [0,1].")
    confidence: float | None = None
    label: str | None = None
    r: float | None = Field(
        default=None,
        description=(
            "Optional page/SVG rotation in degrees applied clockwise around the "
            "center of the literal bbox. Scale normalized x/y/w/h to page units "
            "before applying this rotation."
        ),
    )
    start_index: int | None = Field(
        default=None,
        validation_alias=AliasChoices("start_index", "startIndex"),
    )
    end_index: int | None = Field(
        default=None,
        validation_alias=AliasChoices("end_index", "endIndex"),
    )


class LayoutItemIR(BaseModel):
    """Normalized layout item used for attribution/layout reconstruction."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    type: str = "text"
    md: str = Field(
        default="",
        validation_alias=AliasChoices("md", "markdown"),
    )
    html: str = ""
    value: str = ""
    bbox: LayoutSegmentIR | None = Field(
        default=None,
        validation_alias=AliasChoices("bbox", "bBox"),
    )
    layout_segments: list[LayoutSegmentIR] = Field(
        default_factory=list,
        validation_alias=AliasChoices("layout_segments", "layoutAwareBbox"),
    )
    # Populated by layout-detection providers; left default for parse providers.
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Detector confidence in [0,1]. None for parse-pipeline items.",
    )
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Semantic attributes (e.g. scope=mark, picture_type=chart).",
    )

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, value: object) -> str:
        if value is None:
            return "text"
        return str(value)

    @field_validator("md", "html", "value", mode="before")
    @classmethod
    def _normalize_value(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)


GranularUnitType = Literal["line", "word", "cell", "checkbox"]
GranularLayerAvailability = Literal["available", "empty", "unavailable"]


class GranularUnitIR(BaseModel):
    """Provider-neutral granular grounding unit for page overlays."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    unit_id: str
    granularity: GranularUnitType
    order_index: int
    text: str = ""
    bbox: LayoutSegmentIR
    bboxes: list[LayoutSegmentIR] = Field(default_factory=list)
    label: str | None = None
    row_index: int | None = None
    column_index: int | None = None
    row_span: int | None = None
    column_span: int | None = None
    source_path: str | None = None
    provider: str | None = None

    @field_validator("text", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)


class GranularLayerIR(BaseModel):
    """Provider-neutral layer of granular grounding units for one page."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    granularity: GranularUnitType
    availability: GranularLayerAvailability
    units: list[GranularUnitIR] = Field(default_factory=list)
    reason: str | None = None
    source: str | None = None


class ParseLayoutPageIR(BaseModel):
    """Normalized per-page layout payload embedded in ParseOutput."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    page_number: int = Field(
        ge=1,
        validation_alias=AliasChoices("page_number", "page"),
    )
    width: float | None = Field(
        default=None,
        validation_alias=AliasChoices("width", "page_width"),
    )
    height: float | None = Field(
        default=None,
        validation_alias=AliasChoices("height", "page_height"),
    )
    md: str = ""
    text: str = ""
    page_header_markdown: str = Field(
        default="",
        validation_alias=AliasChoices("page_header_markdown", "pageHeaderMarkdown"),
    )
    page_footer_markdown: str = Field(
        default="",
        validation_alias=AliasChoices("page_footer_markdown", "pageFooterMarkdown"),
    )
    printed_page_number: str = Field(
        default="",
        validation_alias=AliasChoices("printed_page_number", "printedPageNumber"),
    )
    original_orientation_angle: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "original_orientation_angle",
            "originalOrientationAngle",
        ),
    )
    items: list[LayoutItemIR] = Field(default_factory=list)
    granular_layers: list[GranularLayerIR] = Field(
        default_factory=list,
        description=(
            "Provider-neutral line/word/cell/checkbox grounding layers with "
            "normalized [0,1] literal bboxes and optional center rotation."
        ),
    )

    @field_validator(
        "md",
        "text",
        "page_header_markdown",
        "page_footer_markdown",
        "printed_page_number",
        mode="before",
    )
    @classmethod
    def _normalize_optional_text(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)


class ParseOutput(BaseModel):
    """Normalized output for parse tasks."""

    task_type: Literal["parse"] = Field(default="parse", frozen=True, description="Task type discriminator")
    example_id: str = Field(description="Unique identifier for the example")
    pipeline_name: str = Field(description="Name of the pipeline that produced this output")
    pages: list[PageIR] = Field(default_factory=list, description="List of parsed pages")
    layout_pages: list[ParseLayoutPageIR] = Field(
        default_factory=list,
        description=("Normalized page/item/segment layout payload used by layout attribution and overlays"),
    )
    grounded_pages: list[dict[str, Any]] = Field(
        default_factory=list,
        description=("Optional grounded line/word sidecar payload exposed by providers that support granular bboxes"),
    )
    markdown: str = Field(description="Markdown content of the entire document")
    job_id: str | None = Field(default=None, description="Optional job ID from the provider (e.g., LlamaParse job ID)")
