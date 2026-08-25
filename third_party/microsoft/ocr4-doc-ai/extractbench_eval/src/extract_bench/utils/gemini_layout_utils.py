"""Gemini utilities for layout content extraction.

- Unified multi-turn conversation for layout analysis
- Reading order detection
- Picture chart classification
- Table HTML generation with text context
"""

import json
import logging
import os
from enum import StrEnum
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

# Load environment variables
ENV_PATH = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic models for structured outputs
# =============================================================================


class ReadingOrderResponse(BaseModel):
    """Response schema for reading order detection."""

    reading_order: list[int] = Field(description="List of element IDs in the correct reading order")


class PictureType(StrEnum):
    """Valid picture types for classification."""

    bar_chart = "bar_chart"
    bar_code = "bar_code"
    chemistry_markush_structure = "chemistry_markush_structure"
    chemistry_molecular_structure = "chemistry_molecular_structure"
    flow_chart = "flow_chart"
    icon = "icon"
    line_chart = "line_chart"
    logo = "logo"
    map = "map"
    other = "other"
    pie_chart = "pie_chart"
    qr_code = "qr_code"
    remote_sensing = "remote_sensing"
    screenshot = "screenshot"
    signature = "signature"
    stamp = "stamp"


class PictureClassification(BaseModel):
    """Classification result for a single picture."""

    element_id: int = Field(description="The ID of the element being classified")
    picture_type: PictureType = Field(description="The classified picture type")


class PictureClassificationsResponse(BaseModel):
    """Response schema for picture classification."""

    classifications: list[PictureClassification] = Field(description="List of classifications for each picture element")


class TableTranscription(BaseModel):
    """Transcription result for a single table."""

    element_id: int = Field(description="The ID of the table element")
    html: str = Field(description="HTML table representation")


class TableTranscriptionsResponse(BaseModel):
    """Response schema for table transcription."""

    tables: list[TableTranscription] = Field(description="List of HTML transcriptions for each table element")


# =============================================================================
# Analysis result type
# =============================================================================


class LayoutAnalysisResult(BaseModel):
    """Result of unified layout analysis."""

    reading_order: list[int] | None = Field(default=None, description="Element IDs in reading order")
    picture_types: dict[int, str] = Field(default_factory=dict, description="Mapping of element ID to picture type")
    table_html: dict[int, str] = Field(default_factory=dict, description="Mapping of element ID to HTML table")


# System instructions
TABLE_SYSTEM_INSTRUCTION = """You convert tables from images into clean HTML.
Output only HTML with no commentary.
Preserve exact text content without modification."""

PICTURE_CLASSIFIER_SYSTEM_INSTRUCTION = (
    "You classify pictures/figures in documents into specific categories. "
    "You will be shown a document page with a thick red bounding box "
    "highlighting the picture to classify. Output JSON only with the picture_type field."
)

# Valid picture types for classification (kept for backwards compatibility)
PICTURE_TYPES = [pt.value for pt in PictureType]


class GeminiLayoutClient:
    """Gemini client for layout-related tasks."""

    MODEL = "gemini-3-flash-preview"

    def __init__(self, api_key: str | None = None):
        """Initialize the Gemini client.

        :param api_key: Google Gemini API key. If not provided, reads from
                       GOOGLE_GEMINI_API_KEY environment variable.
        :raises ValueError: If no API key is found.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY not found")
        self.client = genai.Client(api_key=self.api_key)

        # Config for table HTML generation
        self.table_config = types.GenerateContentConfig(
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=8192,
            stop_sequences=["</table>"],
            system_instruction=TABLE_SYSTEM_INSTRUCTION,
        )

        # Config for picture classification
        self.classifier_config = types.GenerateContentConfig(
            temperature=0.3,
            top_p=0.95,
            max_output_tokens=256,
            system_instruction=PICTURE_CLASSIFIER_SYSTEM_INSTRUCTION,
        )

    def _image_to_part(self, image: Image.Image) -> types.Part:
        """Convert PIL Image to Gemini Part."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")

    def generate_table_html(self, table_image: Image.Image, text_context: str | None = None) -> str:
        """Generate HTML for a table image.

        :param table_image: Cropped table image
        :param text_context: Aggregated text from the table region (helps reduce hallucinations)
        :return: HTML table string, or empty string on failure
        """
        image_part = self._image_to_part(table_image)

        prompt = """Convert the table in the image into an HTML table.

Rules:
- Output must begin with <table> and end with </table>.
- Only use table-related tags: <table>, <thead>, <tbody>, <tr>, <th>, <td>.
- Specify rowspan and colspan attributes only when they are greater than 1.
- Do not include any other attributes, CSS, captions, or surrounding text.
- Preserve the table's structure and the exact cell text (including punctuation and symbols).
- If a cell is blank in the image, output an empty cell (<td></td> or <th></th>).
- Use <th> for header cells (typically first row or column).
- For multi-row headers, wrap them in <thead>."""

        if text_context:
            prompt += f"""

IMPORTANT: Below is the text extracted from this table region using PDF text extraction.
Use this as a reference to ensure accuracy - DO NOT hallucinate text that isn't present.
Preserve line breaks as they indicate row boundaries:

---
{text_context}
---"""

        try:
            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=[  # type: ignore[arg-type]
                    types.Content(
                        role="user",
                        parts=[image_part, types.Part.from_text(text=prompt)],
                    )
                ],
                config=self.table_config,
            )

            if not response.candidates:
                return ""

            content = response.candidates[0].content
            if content is None or content.parts is None:
                return ""
            text = content.parts[0].text
            if text is None:
                return ""
            return self._extract_html(text)

        except Exception as e:
            logger.warning(f"Error generating table HTML: {e}")
            return ""

    def _extract_html(self, text: str) -> str:
        """Extract HTML table from response text."""
        start = text.find("<table")
        if start == -1:
            return ""

        end = text.rfind("</table>")
        if end == -1:
            return text[start:].strip() + "</table>"
        return text[start : end + 8]

    def _draw_bbox_overlay(
        self,
        image: Image.Image,
        bbox: list[float],
        color: str = "red",
        line_width: int = 4,
    ) -> Image.Image:
        """Draw a thick bounding box overlay on an image.

        :param image: Source image
        :param bbox: Bounding box in COCO format [x, y, width, height]
        :param color: Box color
        :param line_width: Line width in pixels
        :return: Copy of image with bounding box drawn
        """
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        x, y, w, h = bbox
        # Draw rectangle (PIL uses [x1, y1, x2, y2] format)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)
        return img_copy

    def _bbox_to_normalized_1000(self, bbox: list[float], img_width: int, img_height: int) -> list[int]:
        """Convert COCO bbox to Gemini's normalized 0-1000 format.

        :param bbox: Bounding box in COCO format [x, y, width, height]
        :param img_width: Image width in pixels
        :param img_height: Image height in pixels
        :return: Normalized bbox [y_min, x_min, y_max, x_max] in 0-1000 range
        """
        x, y, w, h = bbox
        return [
            int(y * 1000 / img_height),  # y_min
            int(x * 1000 / img_width),  # x_min
            int((y + h) * 1000 / img_height),  # y_max
            int((x + w) * 1000 / img_width),  # x_max
        ]

    def classify_picture(self, page_image: Image.Image, bbox: list[float]) -> str:
        """Classify a picture into a specific category.

        Shows the full page with a bounding box overlay around the picture
        to give Gemini more context for classification.

        :param page_image: Full page image
        :param bbox: Bounding box of the picture in COCO format [x, y, width, height]
        :return: One of PICTURE_TYPES (defaults to "other" on error)
        """
        # Draw bounding box overlay on the page
        annotated_image = self._draw_bbox_overlay(page_image, bbox)
        image_part = self._image_to_part(annotated_image)

        # Convert bbox to normalized 1000x1000 coordinates
        img_width, img_height = page_image.size
        bbox_norm = self._bbox_to_normalized_1000(bbox, img_width, img_height)

        # Build the prompt with categories
        categories_str = ", ".join(PICTURE_TYPES)
        prompt = f"""Classify the picture highlighted by the RED bounding box in this document page.

Bounding box location (normalized 0-1000, [y_min, x_min, y_max, x_max]):
{bbox_norm}

Image dimensions: {img_width}x{img_height} pixels

Classify the picture into ONE of these categories:
{categories_str}

Category descriptions:
- bar_chart: Bar graphs, column charts, histograms showing categorical data
- bar_code: 1D barcodes (UPC, EAN, Code 128, etc.)
- chemistry_markush_structure: Chemical structure with variable groups (R1, R2, etc.)
- chemistry_molecular_structure: Chemical molecule diagrams, structural formulas
- flow_chart: Process flows, decision trees, workflow diagrams
- icon: Small symbolic graphics, UI icons, emoji-like elements
- line_chart: Line graphs, time series, trend charts
- logo: Company logos, brand marks, organizational emblems
- map: Geographic maps, floor plans, site layouts
- other: Photographs, illustrations, diagrams not fitting other categories
- pie_chart: Pie charts, donut charts, circular percentage visualizations
- qr_code: 2D QR codes, data matrix codes
- remote_sensing: Satellite imagery, aerial photos, radar/sonar images
- screenshot: Screen captures of software, websites, or digital interfaces
- signature: Handwritten signatures, autographs
- stamp: Official stamps, seals, certification marks

Output JSON only:
{{"picture_type": "<category>"}}"""

        try:
            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=[  # type: ignore[arg-type]
                    types.Content(
                        role="user",
                        parts=[image_part, types.Part.from_text(text=prompt)],
                    )
                ],
                config=self.classifier_config,
            )

            if not response.candidates:
                return "other"

            content = response.candidates[0].content
            if content is None or content.parts is None:
                return "other"
            text = content.parts[0].text
            if text is None:
                return "other"

            # Parse JSON response
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(text[json_start:json_end])
                picture_type = str(result.get("picture_type", "other"))
                # Validate the returned type
                if picture_type in PICTURE_TYPES:
                    return picture_type
                logger.warning(f"Invalid picture_type returned: {picture_type}")

            return "other"

        except Exception as e:
            logger.warning(f"Error classifying picture: {e}")
            return "other"  # Default to other on error

    # =========================================================================
    # Unified multi-turn conversation API
    # =========================================================================

    def _crop_bbox(self, image: Image.Image, bbox: list[float], padding: int = 5) -> Image.Image:
        """Crop a bounding box region from an image with padding.

        :param image: Source image
        :param bbox: Bounding box in COCO format [x, y, width, height]
        :param padding: Padding in pixels
        :return: Cropped image
        """
        x, y, w, h = bbox
        x1 = max(0, int(x - padding))
        y1 = max(0, int(y - padding))
        x2 = min(image.width, int(x + w + padding))
        y2 = min(image.height, int(y + h + padding))
        return image.crop((x1, y1, x2, y2))

    def analyze_page_layout(
        self,
        page_image: Image.Image,
        elements: list[dict],
        picture_indices: list[int],
        table_indices: list[int],
        detect_reading_order: bool = True,
        classify_pictures: bool = True,
        transcribe_tables: bool = True,
        use_code_execution: bool = False,
    ) -> LayoutAnalysisResult:
        """Unified multi-turn conversation for layout analysis.

        Uses a single conversation context across all analysis tasks, allowing
        Gemini to leverage prior context (e.g., seeing the full page before
        classifying cropped pictures).

        :param page_image: Full page image
        :param elements: List of element dicts with keys: id, bbox, class, text
        :param picture_indices: Indices of Picture elements in the elements list
        :param table_indices: Indices of Table elements in the elements list
        :param detect_reading_order: Whether to detect reading order
        :param classify_pictures: Whether to classify pictures
        :param transcribe_tables: Whether to transcribe tables to HTML
        :param use_code_execution: Whether to enable code execution for all turns
        :return: LayoutAnalysisResult with reading_order, picture_types, table_html
        """
        conversation: list[types.Content] = []
        result = LayoutAnalysisResult()

        # Config with optional code execution
        tools = None
        if use_code_execution:
            tools = [types.Tool(code_execution=types.ToolCodeExecution())]

        # Turn 1: Reading Order Detection
        if detect_reading_order and elements:
            result.reading_order = self._detect_reading_order(page_image, elements, conversation, tools)

        # Turn 2: Picture Classification
        if classify_pictures and picture_indices:
            result.picture_types = self._classify_pictures(page_image, elements, picture_indices, conversation, tools)

        # Turn 3: Table Transcription
        if transcribe_tables and table_indices:
            result.table_html = self._transcribe_tables(page_image, elements, table_indices, conversation, tools)

        return result

    def _detect_reading_order(
        self,
        page_image: Image.Image,
        elements: list[dict],
        conversation: list[types.Content],
        tools: list[types.Tool] | None,
    ) -> list[int] | None:
        """First turn: analyze page and determine reading order.

        :param page_image: Full page image
        :param elements: List of element dicts with keys: id, bbox, class, text
        :param conversation: Conversation history (modified in place)
        :param tools: Optional tools for code execution
        :return: List of element IDs in reading order, or None on error
        """
        image_part = self._image_to_part(page_image)
        img_width, img_height = page_image.size

        # Format elements with normalized bboxes
        elements_json = []
        for elem in elements:
            bbox_norm = self._bbox_to_normalized_1000(elem["bbox"], img_width, img_height)
            text_snippet = elem.get("text", "") or ""
            elements_json.append(
                {
                    "id": elem["id"],
                    "bbox": bbox_norm,
                    "class": elem["class"],
                    "text": text_snippet[:100] if text_snippet else "",
                }
            )

        prompt = f"""Analyze this document page and determine the reading order.

Elements on this page:
{json.dumps(elements_json, indent=2)}

Image dimensions: {img_width}x{img_height} pixels
Bounding boxes are in [y_min, x_min, y_max, x_max] format, normalized to 0-1000.

Determine the correct reading order considering:
- Document structure (titles first, then body text)
- Multi-column layouts (left column before right, or top-to-bottom within columns)
- Headers/footers (page headers/footers are typically read last)
- Captions should be near their figures/tables in the reading order"""

        # Add to conversation
        conversation.append(
            types.Content(
                role="user",
                parts=[image_part, types.Part.from_text(text=prompt)],
            )
        )

        config = types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192,
            tools=tools,  # type: ignore[arg-type]
            response_mime_type="application/json",
            response_schema=ReadingOrderResponse,
        )

        try:
            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=list(conversation),  # type: ignore[arg-type]
                config=config,
            )

            if not response.candidates:
                return None

            # Add response to conversation history
            candidate_content = response.candidates[0].content
            if candidate_content is not None:
                conversation.append(candidate_content)

            # Parse structured response
            if response.text is None:
                return None
            parsed = ReadingOrderResponse.model_validate_json(response.text)
            return parsed.reading_order

        except Exception as e:
            logger.warning(f"Error detecting reading order: {e}")
            return None

    def _classify_pictures(
        self,
        page_image: Image.Image,
        elements: list[dict],
        picture_indices: list[int],
        conversation: list[types.Content],
        tools: list[types.Tool] | None,
    ) -> dict[int, str]:
        """Second turn: classify picture elements.

        :param page_image: Full page image
        :param elements: List of element dicts with keys: id, bbox, class, text
        :param picture_indices: Indices of Picture elements in the elements list
        :param conversation: Conversation history (modified in place)
        :param tools: Optional tools for code execution
        :return: Mapping of element ID to picture type
        """
        parts: list[types.Part] = []

        categories_str = ", ".join(PICTURE_TYPES)
        prompt_text = f"""Now classify each of the following pictures from the page.

For each image, identify its type from these categories:
{categories_str}

Category descriptions:
- bar_chart: Bar graphs, column charts, histograms showing categorical data
- bar_code: 1D barcodes (UPC, EAN, Code 128, etc.)
- chemistry_markush_structure: Chemical structure with variable groups (R1, R2, etc.)
- chemistry_molecular_structure: Chemical molecule diagrams, structural formulas
- flow_chart: Process flows, decision trees, workflow diagrams
- icon: Small symbolic graphics, UI icons, emoji-like elements
- line_chart: Line graphs, time series, trend charts
- logo: Company logos, brand marks, organizational emblems
- map: Geographic maps, floor plans, site layouts
- other: Photographs, illustrations, diagrams not fitting other categories
- pie_chart: Pie charts, donut charts, circular percentage visualizations
- qr_code: 2D QR codes, data matrix codes
- remote_sensing: Satellite imagery, aerial photos, radar/sonar images
- screenshot: Screen captures of software, websites, or digital interfaces
- signature: Handwritten signatures, autographs
- stamp: Official stamps, seals, certification marks

Pictures to classify:
"""

        for idx in picture_indices:
            elem = elements[idx]
            cropped = self._crop_bbox(page_image, elem["bbox"])
            parts.append(self._image_to_part(cropped))
            prompt_text += f"\nImage for element_id={elem['id']} (class={elem['class']}):\n"

        parts.append(types.Part.from_text(text=prompt_text))

        conversation.append(types.Content(role="user", parts=parts))

        config = types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=4096,
            tools=tools,  # type: ignore[arg-type]
            response_mime_type="application/json",
            response_schema=PictureClassificationsResponse,
        )

        try:
            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=list(conversation),  # type: ignore[arg-type]
                config=config,
            )

            if not response.candidates:
                return {}

            candidate_content = response.candidates[0].content
            if candidate_content is not None:
                conversation.append(candidate_content)

            # Parse structured response
            if response.text is None:
                return {}
            parsed = PictureClassificationsResponse.model_validate_json(response.text)
            return {c.element_id: c.picture_type.value for c in parsed.classifications}

        except Exception as e:
            logger.warning(f"Error classifying pictures: {e}")
            return {}

    def _transcribe_tables(
        self,
        page_image: Image.Image,
        elements: list[dict],
        table_indices: list[int],
        conversation: list[types.Content],
        tools: list[types.Tool] | None,
    ) -> dict[int, str]:
        """Third turn: transcribe tables to HTML.

        :param page_image: Full page image
        :param elements: List of element dicts with keys: id, bbox, class, text
        :param table_indices: Indices of Table elements in the elements list
        :param conversation: Conversation history (modified in place)
        :param tools: Optional tools for code execution
        :return: Mapping of element ID to HTML table string
        """
        parts: list[types.Part] = []

        prompt_text = """Now transcribe each table to HTML.

Rules:
- Output clean HTML tables with <table>, <thead>, <tbody>, <tr>, <th>, <td>
- Use rowspan/colspan only when greater than 1
- Preserve exact text content
- Use <th> for header cells (typically first row or column)
- For multi-row headers, wrap them in <thead>

Tables to transcribe:
"""

        for idx in table_indices:
            elem = elements[idx]
            cropped = self._crop_bbox(page_image, elem["bbox"])
            parts.append(self._image_to_part(cropped))
            text_context = elem.get("text", "") or ""
            prompt_text += f"\nTable element_id={elem['id']}:\n"
            if text_context:
                prompt_text += f"Text extracted from PDF (use as reference): {text_context[:500]}\n"

        parts.append(types.Part.from_text(text=prompt_text))

        conversation.append(types.Content(role="user", parts=parts))

        config = types.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=16384,
            tools=tools,  # type: ignore[arg-type]
            response_mime_type="application/json",
            response_schema=TableTranscriptionsResponse,
        )

        try:
            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=list(conversation),  # type: ignore[arg-type]
                config=config,
            )

            if not response.candidates:
                return {}

            candidate_content = response.candidates[0].content
            if candidate_content is not None:
                conversation.append(candidate_content)

            # Parse structured response
            if response.text is None:
                return {}
            parsed = TableTranscriptionsResponse.model_validate_json(response.text)
            return {t.element_id: t.html for t in parsed.tables}

        except Exception as e:
            logger.warning(f"Error transcribing tables: {e}")
            return {}
