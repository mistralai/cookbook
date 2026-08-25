"""Normalized schemas for layout detection outputs."""

from enum import IntEnum, StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, Tag, field_validator

from extract_bench.schemas.layout_ontology import CanonicalLabel


class YoloLabel(IntEnum):
    """YOLO-DocLayNet layout detection labels (11 classes, 0-indexed)."""

    CAPTION = 0
    FOOTNOTE = 1
    FORMULA = 2
    LIST_ITEM = 3
    PAGE_FOOTER = 4
    PAGE_HEADER = 5
    PICTURE = 6
    SECTION_HEADER = 7
    TABLE = 8
    TEXT = 9
    TITLE = 10


class DoclingLabel(IntEnum):
    """Docling RT-DETR layout detection labels (17 classes, 0-indexed)."""

    CAPTION = 0
    FOOTNOTE = 1
    FORMULA = 2
    LIST_ITEM = 3
    PAGE_FOOTER = 4
    PAGE_HEADER = 5
    PICTURE = 6
    SECTION_HEADER = 7
    TABLE = 8
    TEXT = 9
    TITLE = 10
    DOCUMENT_INDEX = 11
    CODE = 12
    CHECKBOX_SELECTED = 13
    CHECKBOX_UNSELECTED = 14
    FORM = 15
    KEY_VALUE_REGION = 16


class LayoutV3Label(IntEnum):
    """Layout-V3 layout detection labels (17 classes, 0-indexed)."""

    CAPTION = 0
    FOOTNOTE = 1
    FORMULA = 2
    LIST_ITEM = 3
    PAGE_FOOTER = 4
    PAGE_HEADER = 5
    PICTURE = 6
    SECTION_HEADER = 7
    TABLE = 8
    TEXT = 9
    TITLE = 10
    DOCUMENT_INDEX = 11
    CODE = 12
    CHECKBOX_SELECTED = 13
    CHECKBOX_UNSELECTED = 14
    FORM = 15
    KEY_VALUE_REGION = 16


class PPDocLayoutLabel(IntEnum):
    """Paddle PP-DocLayout labels (20 classes, 0-indexed)."""

    PARAGRAPH_TITLE = 0
    IMAGE = 1
    TEXT = 2
    NUMBER = 3
    ABSTRACT = 4
    CONTENT = 5
    FIGURE_TITLE = 6
    FORMULA = 7
    TABLE = 8
    REFERENCE = 9
    DOC_TITLE = 10
    FOOTNOTE = 11
    HEADER = 12
    ALGORITHM = 13
    FOOTER = 14
    SEAL = 15
    CHART = 16
    FORMULA_NUMBER = 17
    ASIDE_TEXT = 18
    REFERENCE_CONTENT = 19


PPDOCLAYOUT_STR_TO_LABEL: dict[str, PPDocLayoutLabel] = {
    "paragraph_title": PPDocLayoutLabel.PARAGRAPH_TITLE,
    "image": PPDocLayoutLabel.IMAGE,
    "text": PPDocLayoutLabel.TEXT,
    "number": PPDocLayoutLabel.NUMBER,
    "abstract": PPDocLayoutLabel.ABSTRACT,
    "content": PPDocLayoutLabel.CONTENT,
    "figure_title": PPDocLayoutLabel.FIGURE_TITLE,
    "formula": PPDocLayoutLabel.FORMULA,
    "table": PPDocLayoutLabel.TABLE,
    "reference": PPDocLayoutLabel.REFERENCE,
    "doc_title": PPDocLayoutLabel.DOC_TITLE,
    "footnote": PPDocLayoutLabel.FOOTNOTE,
    "header": PPDocLayoutLabel.HEADER,
    "algorithm": PPDocLayoutLabel.ALGORITHM,
    "footer": PPDocLayoutLabel.FOOTER,
    "seal": PPDocLayoutLabel.SEAL,
    "chart": PPDocLayoutLabel.CHART,
    "formula_number": PPDocLayoutLabel.FORMULA_NUMBER,
    "aside_text": PPDocLayoutLabel.ASIDE_TEXT,
    "reference_content": PPDocLayoutLabel.REFERENCE_CONTENT,
}


class Qwen3VLLabel(IntEnum):
    """Qwen3-VL layout detection labels (11 Core11 classes, 0-indexed)."""

    CAPTION = 0
    FOOTNOTE = 1
    FORMULA = 2
    LIST_ITEM = 3
    PAGE_FOOTER = 4
    PAGE_HEADER = 5
    PICTURE = 6
    SECTION_HEADER = 7
    TABLE = 8
    TEXT = 9
    TITLE = 10


QWEN3VL_STR_TO_LABEL: dict[str, Qwen3VLLabel] = {
    "caption": Qwen3VLLabel.CAPTION,
    "footnote": Qwen3VLLabel.FOOTNOTE,
    "formula": Qwen3VLLabel.FORMULA,
    "list_item": Qwen3VLLabel.LIST_ITEM,
    "page_footer": Qwen3VLLabel.PAGE_FOOTER,
    "page_header": Qwen3VLLabel.PAGE_HEADER,
    "picture": Qwen3VLLabel.PICTURE,
    "section_header": Qwen3VLLabel.SECTION_HEADER,
    "table": Qwen3VLLabel.TABLE,
    "text": Qwen3VLLabel.TEXT,
    "title": Qwen3VLLabel.TITLE,
}


class SuryaLabel(IntEnum):
    """Surya OCR layout detection labels (16 classes, 0-indexed)."""

    CAPTION = 0
    FOOTNOTE = 1
    FORMULA = 2
    LIST_ITEM = 3
    PAGE_FOOTER = 4
    PAGE_HEADER = 5
    PICTURE = 6
    FIGURE = 7
    SECTION_HEADER = 8
    TABLE = 9
    FORM = 10
    TABLE_OF_CONTENTS = 11
    HANDWRITING = 12
    TEXT = 13
    TEXT_INLINE_MATH = 14
    CODE = 15


SURYA_STR_TO_LABEL: dict[str, SuryaLabel] = {
    "Caption": SuryaLabel.CAPTION,
    "Footnote": SuryaLabel.FOOTNOTE,
    "Formula": SuryaLabel.FORMULA,
    "Equation": SuryaLabel.FORMULA,
    "ListItem": SuryaLabel.LIST_ITEM,
    "PageFooter": SuryaLabel.PAGE_FOOTER,
    "PageHeader": SuryaLabel.PAGE_HEADER,
    "Picture": SuryaLabel.PICTURE,
    "Figure": SuryaLabel.FIGURE,
    "SectionHeader": SuryaLabel.SECTION_HEADER,
    "Table": SuryaLabel.TABLE,
    "Form": SuryaLabel.FORM,
    "TableOfContents": SuryaLabel.TABLE_OF_CONTENTS,
    "Handwriting": SuryaLabel.HANDWRITING,
    "Text": SuryaLabel.TEXT,
    "TextInlineMath": SuryaLabel.TEXT_INLINE_MATH,
    "Code": SuryaLabel.CODE,
    "List-item": SuryaLabel.LIST_ITEM,
    "Page-footer": SuryaLabel.PAGE_FOOTER,
    "Page-header": SuryaLabel.PAGE_HEADER,
    "Section-header": SuryaLabel.SECTION_HEADER,
    "Table-of-contents": SuryaLabel.TABLE_OF_CONTENTS,
    "Text-inline-math": SuryaLabel.TEXT_INLINE_MATH,
}


class ChandraLabel(IntEnum):
    """Chandra OCR layout detection labels (15 classes, 0-indexed)."""

    CAPTION = 0
    FOOTNOTE = 1
    EQUATION_BLOCK = 2
    LIST_GROUP = 3
    PAGE_HEADER = 4
    PAGE_FOOTER = 5
    IMAGE = 6
    SECTION_HEADER = 7
    TABLE = 8
    TEXT = 9
    COMPLEX_BLOCK = 10
    CODE_BLOCK = 11
    FORM = 12
    TABLE_OF_CONTENTS = 13
    FIGURE = 14


CHANDRA_STR_TO_LABEL: dict[str, ChandraLabel] = {
    "Caption": ChandraLabel.CAPTION,
    "Footnote": ChandraLabel.FOOTNOTE,
    "Equation-Block": ChandraLabel.EQUATION_BLOCK,
    "List-Group": ChandraLabel.LIST_GROUP,
    "Page-Header": ChandraLabel.PAGE_HEADER,
    "Page-Footer": ChandraLabel.PAGE_FOOTER,
    "Image": ChandraLabel.IMAGE,
    "Section-Header": ChandraLabel.SECTION_HEADER,
    "Table": ChandraLabel.TABLE,
    "Text": ChandraLabel.TEXT,
    "Complex-Block": ChandraLabel.COMPLEX_BLOCK,
    "Code-Block": ChandraLabel.CODE_BLOCK,
    "Form": ChandraLabel.FORM,
    "Table-Of-Contents": ChandraLabel.TABLE_OF_CONTENTS,
    "Figure": ChandraLabel.FIGURE,
}


class ChunkrLabel(IntEnum):
    """Chunkr layout detection labels (17 classes, 0-indexed)."""

    CAPTION = 0
    FOOTNOTE = 1
    FORMULA = 2
    FORM_REGION = 3
    GRAPHICAL_ITEM = 4
    LEGEND = 5
    LINE_NUMBER = 6
    LIST_ITEM = 7
    PAGE_FOOTER = 8
    PAGE_HEADER = 9
    PAGE_NUMBER = 10
    PICTURE = 11
    TABLE = 12
    TEXT = 13
    TITLE = 14
    UNKNOWN = 15
    PAGE = 16


CHUNKR_STR_TO_LABEL: dict[str, ChunkrLabel] = {
    "Caption": ChunkrLabel.CAPTION,
    "Footnote": ChunkrLabel.FOOTNOTE,
    "Formula": ChunkrLabel.FORMULA,
    "FormRegion": ChunkrLabel.FORM_REGION,
    "GraphicalItem": ChunkrLabel.GRAPHICAL_ITEM,
    "Legend": ChunkrLabel.LEGEND,
    "LineNumber": ChunkrLabel.LINE_NUMBER,
    "ListItem": ChunkrLabel.LIST_ITEM,
    "PageFooter": ChunkrLabel.PAGE_FOOTER,
    "PageHeader": ChunkrLabel.PAGE_HEADER,
    "PageNumber": ChunkrLabel.PAGE_NUMBER,
    "Picture": ChunkrLabel.PICTURE,
    "Table": ChunkrLabel.TABLE,
    "Text": ChunkrLabel.TEXT,
    "Title": ChunkrLabel.TITLE,
    "Unknown": ChunkrLabel.UNKNOWN,
    "Page": ChunkrLabel.PAGE,
}


class LayoutDetectionModel(StrEnum):
    """Supported layout detection models."""

    YOLO_DOCLAYNET = "yolo_doclaynet"
    PPDOCLAYOUT_PLUS_L = "ppdoclayout_plus_l"
    DOCLING_LAYOUT_OLD = "docling_layout_old"
    DOCLING_LAYOUT_HERON_101 = "docling_layout_heron_101"
    DOCLING_LAYOUT_HERON = "docling_layout_heron"
    DOCLING_PARSE_LAYOUT = "docling_parse_layout"
    QWEN3_VL_8B = "qwen3_vl_8b"
    LLAMAPARSE = "llamaparse"
    SURYA_LAYOUT = "surya_layout"
    CHANDRA = "chandra"
    LAYOUT_V3 = "layout_v3"
    CHUNKR = "chunkr"
    DOTS_OCR = "dots_ocr"
    PULSE_LAYOUT = "pulse_layout"
    REDUCTO_LAYOUT = "reducto_layout"
    TEXTRACT_LAYOUT = "textract_layout"
    LANDINGAI_LAYOUT = "landingai_layout"
    EXTEND_LAYOUT = "extend_layout"
    AZURE_DI_LAYOUT = "azure_di_layout"
    GOOGLE_DOCAI_LAYOUT = "google_docai_layout"
    UNSTRUCTURED_LAYOUT = "unstructured_layout"
    DEEPSEEK_OCR2_LAYOUT = "deepseek_ocr2_layout"
    MINERU25_LAYOUT = "mineru25_layout"
    KDL_FRONTIER_NANO_LAYOUT = "kdl_frontier_nano_layout"
    CHANDRA2_LAYOUT = "chandra2_layout"
    QFOCR_LAYOUT = "qfocr_layout"
    DATALAB_LAYOUT = "datalab_layout"
    QWEN3_5_LAYOUT = "qwen3_5_layout"
    GEMINI_LAYOUT = "gemini_layout"
    OPENAI_LAYOUT = "openai_layout"
    ANTHROPIC_LAYOUT = "anthropic_layout"
    GEMMA4_LAYOUT = "gemma4_layout"
    DATABRICKS_LAYOUT = "databricks_layout"
    INFINITY_PARSER2_LAYOUT = "infinity_parser2_layout"
    OI_PARSER_LAYOUT = "oi_parser_layout"


LAYOUT_MODEL_INFO: dict[LayoutDetectionModel, dict[str, str]] = {
    LayoutDetectionModel.PPDOCLAYOUT_PLUS_L: {
        "name": "PP-DocLayout-plus-L",
        "hf_url": "https://huggingface.co/llamaindex/paddleOCRDocLayoutPlusL",
    },
    LayoutDetectionModel.DOCLING_LAYOUT_OLD: {
        "name": "Docling RT-DETR DocLayNet",
        "hf_url": "https://huggingface.co/llamaindex/layout_rtdetrdoclaynet",
    },
    LayoutDetectionModel.DOCLING_LAYOUT_HERON_101: {
        "name": "Docling RT-DETR Heron 101",
        "hf_url": "https://huggingface.co/llamaindex/layout_rtdetrdoclaynet",
    },
    LayoutDetectionModel.DOCLING_LAYOUT_HERON: {
        "name": "Docling RT-DETR Heron",
        "hf_url": "https://huggingface.co/llamaindex/layout_rtdetrdoclaynet",
    },
    LayoutDetectionModel.DOCLING_PARSE_LAYOUT: {
        "name": "Docling Parse Layout",
        "hf_url": "https://huggingface.co/llamaindex/docling-parse",
    },
    LayoutDetectionModel.QWEN3_VL_8B: {
        "name": "Qwen3-VL-8B-Instruct",
        "hf_url": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct",
    },
    LayoutDetectionModel.LLAMAPARSE: {
        "name": "LlamaParse Layout Detection",
        "hf_url": "https://cloud.llamaindex.ai",
    },
    LayoutDetectionModel.SURYA_LAYOUT: {
        "name": "Surya OCR Layout Detection",
        "hf_url": "https://github.com/datalab-to/surya",
    },
    LayoutDetectionModel.CHANDRA: {
        "name": "Chandra OCR Layout Detection",
        "hf_url": "https://huggingface.co/datalab-to/chandra",
    },
    LayoutDetectionModel.LAYOUT_V3: {
        "name": "Layout V3 (RT-DETRv2 + Figure Classification)",
        "hf_url": "https://huggingface.co/llamaindex/layout-v3",
    },
    LayoutDetectionModel.CHUNKR: {
        "name": "Chunkr Layout Detection",
        "hf_url": "https://www.chunkr.ai/",
    },
    LayoutDetectionModel.DOTS_OCR: {
        "name": "dots.ocr",
        "hf_url": "https://huggingface.co/rednote-hilab/dots.ocr",
    },
    LayoutDetectionModel.OI_PARSER_LAYOUT: {
        "name": "oi-parser",
        "hf_url": "https://oi-parser.ai/",
    },
    LayoutDetectionModel.PULSE_LAYOUT: {
        "name": "Pulse Layout",
        "hf_url": "https://www.runpulse.com/",
    },
    LayoutDetectionModel.REDUCTO_LAYOUT: {
        "name": "Reducto Layout",
        "hf_url": "https://www.reducto.ai/",
    },
    LayoutDetectionModel.TEXTRACT_LAYOUT: {
        "name": "AWS Textract Layout",
        "hf_url": "https://aws.amazon.com/textract/",
    },
    LayoutDetectionModel.LANDINGAI_LAYOUT: {
        "name": "LandingAI ADE Layout",
        "hf_url": "https://landing.ai/",
    },
    LayoutDetectionModel.EXTEND_LAYOUT: {
        "name": "Extend AI Layout",
        "hf_url": "https://extend.ai/",
    },
    LayoutDetectionModel.AZURE_DI_LAYOUT: {
        "name": "Azure Document Intelligence Layout",
        "hf_url": "https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence",
    },
    LayoutDetectionModel.GOOGLE_DOCAI_LAYOUT: {
        "name": "Google Document AI Layout",
        "hf_url": "https://cloud.google.com/document-ai",
    },
    LayoutDetectionModel.UNSTRUCTURED_LAYOUT: {
        "name": "Unstructured Layout",
        "hf_url": "https://unstructured.io/",
    },
    LayoutDetectionModel.DEEPSEEK_OCR2_LAYOUT: {
        "name": "DeepSeek-OCR-2 Layout",
        "hf_url": "https://huggingface.co/deepseek-ai/DeepSeek-OCR-2",
    },
    LayoutDetectionModel.CHANDRA2_LAYOUT: {
        "name": "Chandra OCR 2 Layout",
        "hf_url": "https://huggingface.co/datalab-to/chandra-ocr-2",
    },
    LayoutDetectionModel.DATALAB_LAYOUT: {
        "name": "Datalab Layout (Marker/Surya)",
        "hf_url": "https://datalab.to",
    },
    LayoutDetectionModel.GEMINI_LAYOUT: {
        "name": "Gemini Layout (parse_with_layout)",
        "hf_url": "https://ai.google.dev/",
    },
    LayoutDetectionModel.OPENAI_LAYOUT: {
        "name": "OpenAI Layout (parse_with_layout)",
        "hf_url": "https://platform.openai.com/",
    },
    LayoutDetectionModel.ANTHROPIC_LAYOUT: {
        "name": "Anthropic Layout (parse_with_layout)",
        "hf_url": "https://docs.anthropic.com/",
    },
    LayoutDetectionModel.GEMMA4_LAYOUT: {
        "name": "Gemma 4 Layout (parse_with_layout)",
        "hf_url": "https://huggingface.co/google/gemma-4-E4B-it",
    },
    LayoutDetectionModel.DATABRICKS_LAYOUT: {
        "name": "Databricks ai_parse_document Layout",
        "hf_url": "https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document",
    },
    LayoutDetectionModel.INFINITY_PARSER2_LAYOUT: {
        "name": "Infinity-Parser2 Layout",
        "hf_url": "https://huggingface.co/collections/infly/infinity-parser2",
    },
}


class LayoutTextContent(BaseModel):
    """Text content for layout elements (paragraphs, headers, captions, etc.)."""

    type: Literal["text"] = "text"
    text: str = Field(description="Aggregated text content from PDF cells")


class LayoutTableContent(BaseModel):
    """Table content with HTML representation."""

    type: Literal["table"] = "table"
    html: str = Field(description="HTML table representation")


LayoutContent = Annotated[
    Annotated[LayoutTextContent, Tag("text")] | Annotated[LayoutTableContent, Tag("table")],
    Discriminator("type"),
]


class LayoutPrediction(BaseModel):
    """Provider-agnostic layout prediction."""

    bbox: list[float] = Field(description="[x1, y1, x2, y2] in pixel coordinates")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score")
    label: str = Field(description="Raw provider label")
    page: int | None = Field(default=None, description="1-indexed page number")
    content: LayoutContent | None = Field(
        default=None,
        description="Optional content associated with this element",
    )
    attributes: dict[str, str] = Field(default_factory=dict)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("label", mode="before")
    @classmethod
    def _normalize_label(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)


class BaseCanonicalizablePrediction(BaseModel):
    """Base class used for runtime label projection results."""

    bbox: list[float]
    score: float = Field(ge=0.0, le=1.0)
    attributes: dict[str, str] = Field(default_factory=dict)
    original_label: int | str
    page: int | None = None


class CoreLayoutPrediction(BaseCanonicalizablePrediction):
    """Runtime-projected Core11 label prediction."""

    core_class: CanonicalLabel


class CanonicalLayoutPrediction(BaseCanonicalizablePrediction):
    """Runtime-projected Canonical17 label prediction."""

    canonical_class: CanonicalLabel


class LayoutOutput(BaseModel):
    """Normalized output for layout detection tasks."""

    task_type: Literal["layout_detection"] = Field(
        default="layout_detection",
        frozen=True,
        description="Task type discriminator",
    )
    example_id: str = Field(description="Unique identifier for the example")
    pipeline_name: str = Field(description="Name of the pipeline that produced this output")
    model: LayoutDetectionModel = Field(description="Layout detection model used")
    image_width: int = Field(ge=1, description="Width of the input image in pixels")
    image_height: int = Field(ge=1, description="Height of the input image in pixels")
    predictions: list[LayoutPrediction] = Field(default_factory=list)
    markdown: str = Field(
        default="",
        description=("Optional document markdown for providers that can supply it (e.g., LlamaParse layout runs)."),
    )
