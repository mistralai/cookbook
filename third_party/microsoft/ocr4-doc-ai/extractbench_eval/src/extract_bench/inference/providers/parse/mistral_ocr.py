"""Provider for Mistral OCR (the standard ``/v1/ocr`` API).

This provider calls Mistral's hosted OCR endpoint
(``POST https://api.mistral.ai/v1/ocr``) with the OCR 4.0 model
(``mistral-ocr-4-0``).  The PDF is sent inline as a base64 ``data:`` URI, so no
temporary file host is required.

Output
------
The API returns one object per page with:

- ``markdown``: page text as markdown (tables rendered as markdown pipe tables)
- ``dimensions``: ``{dpi, height, width}`` in pixels
- ``blocks``: paragraph-level layout blocks (only when ``include_blocks=True``,
  available on OCR 4+).  Each block has pixel coordinates
  ``top_left_x/top_left_y/bottom_right_x/bottom_right_y``, the block ``content``,
  and a ``type`` in {header, footer, title, text, list, table, caption, image}.

``normalize()`` concatenates the per-page markdown (converting markdown pipe
tables to HTML ``<table>`` so GriTS/TEDS can score them) and, when blocks are
present, builds ``layout_pages`` with each block's bbox normalized to ``[0,1]``
xywh and its ``type`` mapped to a Canonical17 label for layout evaluation.

Annotation ("Document AI") mode
-------------------------------
When ``bbox_annotation=True`` (the ``mistral_ocr_4_annotation`` pipeline), the
request also carries a ``bbox_annotation_format`` JSON schema. Mistral then runs
a vision model over every extracted figure bbox and returns an ``image_annotation``
per image. ``normalize()`` splices each figure's transcribed ``data_markdown``
(e.g. a chart's underlying data table) back into the page markdown in place of the
opaque ``![img-N.jpeg](img-N.jpeg)`` placeholder, so chart/plot data reaches the
scorer. This mode is billed at the higher "annotated pages" rate ($5/1000 vs
$4/1000 for plain OCR). See ``_FIGURE_ANNOTATION_SCHEMA`` / ``_inject_image_annotations``.
"""

import base64
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import markdown2
import requests
from pypdf import PdfReader

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

_MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"


# ---------------------------------------------------------------------------
# Pipe-table -> HTML conversion
# ---------------------------------------------------------------------------
#
# Mistral OCR renders tables as markdown pipe tables, but ParseBench's table
# metrics (GriTS/TEDS) only score HTML ``<table>`` elements. ``normalize()``
# converts pipe tables to HTML before storing the markdown; non-table content
# is left untouched so rule-based text evaluation still works.

# Regex for a separator row: only pipes, dashes, colons, and whitespace.
_SEPARATOR_RE = re.compile(r"^\|?[\s:|-]+\|?$")


def _is_pipe_table_line(line: str) -> bool:
    """Return True if *line* looks like a markdown pipe-table row."""
    stripped = line.strip()
    return "|" in stripped and not stripped.startswith("<!--")


def _convert_pipe_tables_to_html(md_content: str) -> str:
    """Replace markdown pipe tables with HTML ``<table>`` elements.

    Non-table content is left untouched so that rule-based evaluation
    (which operates on the raw markdown text) still works.
    """
    if not md_content or "|" not in md_content:
        return md_content

    lines = md_content.split("\n")
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if _is_pipe_table_line(line):
            # Collect consecutive pipe-table lines.
            table_lines: list[str] = [line]
            j = i + 1
            while j < len(lines) and _is_pipe_table_line(lines[j]):
                table_lines.append(lines[j])
                j += 1

            # A valid table needs >= 3 lines (header + separator + data)
            # and must contain a separator row.
            has_separator = any(_SEPARATOR_RE.match(tl.strip()) for tl in table_lines)
            if len(table_lines) >= 3 and has_separator:
                table_md = "\n".join(table_lines)
                rendered = markdown2.markdown(table_md, extras=["tables"])
                if "<table" in rendered.lower():
                    result.append(rendered.strip())
                    i = j
                    continue

            # Not a valid table; keep the first line and advance by one.
            result.append(line)
            i += 1
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


# Mistral OCR block ``type`` -> Canonical17 label (the dataset GT label set).
#
# The eight types on the left are the full vocabulary Mistral OCR 4 emits in
# ``blocks`` (observed across the tables/charts/layout benchmark documents);
# the rest are defensive entries for types Mistral may emit on other content.
# The mapping was verified against the layout_attribution ground truth, whose
# ontology is Basic7 {Text, Table, Page-header, Section, Picture, ...}:
#   - ``header``  -> running page identifiers (patent no. / page no. at the top)
#                    -> GT ``Page-header``.
#   - ``title``   -> section headings (e.g. "### Example 23") -> GT ``Section``
#                    (Title and Section-header both fold into Section in Basic7).
#   - ``caption`` -> table/figure captions ("TABLE 78") -> Text in Basic7.
# Mistral uses a separate ``title`` type for headings, so bare ``header`` /
# ``footer`` are page furniture (Page-header / Page-footer), not section headers.
# Unknown types fall back to ``Text`` (a valid Canonical17 label, so the
# evaluator never silently drops the region).
MISTRAL_LABEL_MAP: dict[str, str] = {
    # Full observed OCR-4 block vocabulary
    "header": "Page-header",
    "footer": "Page-footer",
    "title": "Title",
    "text": "Text",
    "list": "List-item",
    "table": "Table",
    "caption": "Caption",
    "image": "Picture",
    # Defensive entries for types seen on other document kinds
    "section_header": "Section-header",
    "footnote": "Footnote",
    "formula": "Formula",
    "code": "Code",
    "picture": "Picture",
    "figure": "Picture",
}

# ---------------------------------------------------------------------------
# Figure annotation ("Document AI" mode)
# ---------------------------------------------------------------------------
#
# In annotation mode the provider sends this schema as ``bbox_annotation_format``
# with the OCR request. Mistral runs a vision model over every extracted figure
# bbox and returns, per image, an ``image_annotation`` JSON string with these
# fields. ``data_markdown`` is the lever for the charts benchmark: standard OCR
# emits charts as opaque ``![img-N.jpeg](img-N.jpeg)`` placeholders (no data),
# whereas annotation transcribes a chart/plot's underlying numbers into a markdown
# table. The schema instructs the model to return an *empty* ``data_markdown`` for
# figures without extractable data (photos, logos, decorative icons) so those keep
# their original placeholder and we never inject prose the source doesn't contain.
_FIGURE_ANNOTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "FigureData",
    "properties": {
        "figure_type": {
            "type": "string",
            "title": "figure_type",
            "description": (
                "Kind of figure: one of bar_chart, line_chart, pie_chart, "
                "scatter_plot, area_chart, table, diagram, map, photo, logo, icon, other."
            ),
        },
        "data_markdown": {
            "type": "string",
            "title": "data_markdown",
            "description": (
                "If the figure is a chart, plot, graph, or data table, transcribe ALL "
                "of its underlying data as one or more GitHub-flavored markdown tables: "
                "include axis titles, every category/x label, every series name, and "
                "every numeric value (with units or % signs) exactly as shown. Be "
                "exhaustive and precise. If the figure carries no extractable tabular "
                "data (a photo, logo, decorative icon, or pure illustration), return an "
                "empty string."
            ),
        },
        "caption": {
            "type": "string",
            "title": "caption",
            "description": "A concise one-sentence caption describing what the figure shows.",
        },
    },
    "required": ["figure_type", "data_markdown", "caption"],
    "additionalProperties": False,
}


def _inject_image_annotations(markdown: str, images: list[dict[str, Any]]) -> str:
    """Splice transcribed figure data into the page markdown.

    For each extracted image carrying a non-empty ``data_markdown`` annotation (a
    markdown transcription of a chart/plot/table's underlying data), replace the bare
    ``![alt](image_id)`` placeholder in ``markdown`` with that transcription. Figures
    whose annotation has an empty ``data_markdown`` (photos, logos, decorative icons)
    keep their original placeholder, so this never injects descriptive prose the
    source document doesn't contain. A no-op when no image carries an annotation
    (standard OCR mode), so the markdown is byte-identical to the non-annotated path.
    """
    for img in images:
        ann_raw = img.get("image_annotation")
        img_id = img.get("id")
        if not ann_raw or not img_id:
            continue
        try:
            ann = json.loads(ann_raw) if isinstance(ann_raw, str) else ann_raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(ann, dict):
            continue
        data_md = str(ann.get("data_markdown") or "").strip()
        if not data_md:
            continue
        placeholder = re.compile(r"!\[[^\]]*\]\(" + re.escape(str(img_id)) + r"\)")
        # Double backslashes so re.sub treats data_md as a literal replacement
        # (no ``\1`` / ``\g<...>`` backreference interpretation).
        markdown = placeholder.sub(data_md.replace("\\", "\\\\"), markdown)
    return markdown


# ---------------------------------------------------------------------------
# Layout pages from blocks
# ---------------------------------------------------------------------------


def _build_layout_pages(pages: list[dict[str, Any]]) -> list[ParseLayoutPageIR]:
    """Build ``layout_pages`` from Mistral OCR per-page ``blocks``.

    Each block carries pixel coordinates relative to the page ``dimensions``.
    Coordinates are normalized to ``[0,1]`` xywh for cross-evaluation.
    """
    layout_pages: list[ParseLayoutPageIR] = []

    for page_idx, page in enumerate(pages):
        blocks = page.get("blocks") or []
        if not isinstance(blocks, list):
            continue

        dims = page.get("dimensions") or {}
        page_w = float(dims.get("width") or 0.0) or 1.0
        page_h = float(dims.get("height") or 0.0) or 1.0

        items: list[LayoutItemIR] = []
        for block in blocks:
            block_type = str(block.get("type", "")).strip().lower()
            canonical_label = MISTRAL_LABEL_MAP.get(block_type, "Text")

            try:
                x1 = float(block["top_left_x"])
                y1 = float(block["top_left_y"])
                x2 = float(block["bottom_right_x"])
                y2 = float(block["bottom_right_y"])
            except (KeyError, TypeError, ValueError):
                continue

            seg = LayoutSegmentIR(
                x=x1 / page_w,
                y=y1 / page_h,
                w=(x2 - x1) / page_w,
                h=(y2 - y1) / page_h,
                confidence=1.0,
                label=canonical_label,
            )

            norm_label = canonical_label.lower()
            if norm_label == "table":
                item_type = "table"
            elif norm_label == "picture":
                item_type = "image"
            else:
                item_type = "text"

            items.append(
                LayoutItemIR(
                    type=item_type,
                    value=block.get("content", "") or "",
                    bbox=seg,
                    layout_segments=[seg],
                )
            )

        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_idx + 1,
                width=page_w,
                height=page_h,
                items=items,
            )
        )

    return layout_pages


@register_provider("mistral_ocr")
class MistralOCRProvider(Provider):
    """Provider for the Mistral OCR ``/v1/ocr`` endpoint.

    Config keys
    -----------
    api_key : str
        Mistral API key. Falls back to ``MISTRAL_API_KEY`` env var.
    model : str
        OCR model id (default ``"mistral-ocr-4-0"``).
    include_blocks : bool
        Request paragraph-level layout blocks (default ``True``; OCR 4+ only).
    bbox_annotation : bool
        Enable "Document AI" annotation mode (default ``False``): transcribe each
        figure's data into the markdown via ``bbox_annotation_format``. Billed at
        the higher annotated-pages rate.
    max_pages : int
        Cap on pages sent to the API for very long documents (default 50).
    timeout : int
        HTTP request timeout in seconds (default 300).
    """

    # OCR 4.0 list prices. Plain OCR is $4 / 1000 pages; annotation ("Document AI")
    # mode is billed at $5 / 1000 annotated pages.
    COST_PER_PAGE_USD = 0.004
    COST_PER_PAGE_ANNOTATED_USD = 0.005

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        api_key = self.base_config.get("api_key") or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "Mistral API key is required. Set MISTRAL_API_KEY environment variable or pass api_key in base_config."
            )
        self._api_key: str = str(api_key)

        self._model: str = self.base_config.get("model", "mistral-ocr-4-0")
        self._include_blocks: bool = self.base_config.get("include_blocks", True)
        self._max_pages: int = self.base_config.get("max_pages", 50)
        self._timeout: int = self.base_config.get("timeout", 300)
        # Annotation ("Document AI") mode: request per-figure data transcription so
        # charts/plots reach the scorer as data tables instead of opaque image
        # placeholders. Billed at the higher annotated-pages rate ($5 vs $4 /1000).
        self._bbox_annotation: bool = self.base_config.get("bbox_annotation", False)
        self._cost_per_page: float = (
            self.COST_PER_PAGE_ANNOTATED_USD if self._bbox_annotation else self.COST_PER_PAGE_USD
        )
        # Internal backoff for 429 rate limits and 5xx, honoring the server's
        # Retry-After header. Mistral OCR rate-limits aggressively; retrying in
        # the provider (instead of letting docs exhaust the runner's budget)
        # keeps the per-doc success rate at ~100% even under sustained 429s.
        self._rate_limit_retries: int = self.base_config.get("rate_limit_retries", 6)
        self._rate_limit_base_wait: float = self.base_config.get("rate_limit_base_wait", 2.0)
        self._rate_limit_max_wait: float = self.base_config.get("rate_limit_max_wait", 20.0)

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"MistralOCRProvider only supports PARSE, got {request.product_type}")

        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"File not found: {pdf_path}")

        started_at = datetime.now()

        pdf_bytes = pdf_path.read_bytes()
        try:
            num_pages = len(PdfReader(pdf_path).pages)
        except Exception:
            num_pages = 0

        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        document = {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{b64}",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "document": document,
            "include_blocks": self._include_blocks,
            "include_image_base64": False,
        }
        # Annotation mode: ask Mistral to transcribe each extracted figure into a
        # structured ``image_annotation`` (``_inject_image_annotations`` splices the
        # chart data back into the markdown in ``normalize``).
        if self._bbox_annotation:
            payload["bbox_annotation_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": _FIGURE_ANNOTATION_SCHEMA,
                    "name": "figure_data",
                    "strict": True,
                },
            }
        # Cap pages for very long docs (the smoke/benchmark docs are short, so
        # this is a no-op there). Only send an explicit page list when the doc
        # exceeds the cap to avoid an out-of-range page error.
        if num_pages and num_pages > self._max_pages:
            payload["pages"] = list(range(self._max_pages))

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Retry 429 (rate limit) and 5xx in-provider, honoring Retry-After, so a
        # rate-limited document waits and succeeds rather than failing once the
        # runner's outer retry budget is exhausted.
        resp = None
        for attempt in range(self._rate_limit_retries + 1):
            try:
                resp = requests.post(
                    _MISTRAL_OCR_URL,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout,
                )
            except requests.exceptions.Timeout as e:
                raise ProviderTransientError(f"Mistral OCR request timed out: {e}") from e
            except requests.exceptions.ConnectionError as e:
                raise ProviderTransientError(f"Mistral OCR connection error: {e}") from e

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self._rate_limit_retries:
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        wait = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        wait = 0.0
                    if wait <= 0:
                        wait = self._rate_limit_base_wait * (2.0**attempt)
                    time.sleep(min(wait, self._rate_limit_max_wait))
                    continue
                # Retries exhausted — fall through to the outer runner retry.
                if resp.status_code == 429:
                    raise ProviderRateLimitError(
                        f"Mistral OCR rate limit (429) after {self._rate_limit_retries} retries: {resp.text[:300]}"
                    )
                raise ProviderTransientError(
                    f"Mistral OCR server error ({resp.status_code}) after "
                    f"{self._rate_limit_retries} retries: {resp.text[:300]}"
                )
            break

        assert resp is not None
        if resp.status_code == 401:
            raise ProviderConfigError(f"Mistral OCR unauthorized (401): {resp.text[:500]}")
        if resp.status_code >= 400:
            raise ProviderPermanentError(f"Mistral OCR client error ({resp.status_code}): {resp.text[:500]}")

        raw_output: dict[str, Any] = resp.json()

        # Cost tracking.
        usage = raw_output.get("usage_info") or {}
        pages_processed = usage.get("pages_processed") or len(raw_output.get("pages", [])) or num_pages
        cost_usd = pages_processed * self._cost_per_page
        raw_output["cost_usd"] = cost_usd
        raw_output["cost_per_page_usd"] = self._cost_per_page
        raw_output["_config"] = {
            "model": self._model,
            "include_blocks": self._include_blocks,
            "bbox_annotation": self._bbox_annotation,
            "max_pages": self._max_pages,
            "total_pages": num_pages,
        }

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=latency_ms,
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"MistralOCRProvider only supports PARSE, got {raw_result.product_type}")

        pages = raw_result.raw_output.get("pages") or []
        page_markdowns: list[str] = []
        for p in pages:
            page_md = str(p.get("markdown", "") or "")
            # Annotation mode: splice each figure's transcribed data table into the
            # markdown in place of its image placeholder. No-op for standard OCR
            # (images carry no ``image_annotation``), so output stays identical.
            images = p.get("images") or []
            if images:
                page_md = _inject_image_annotations(page_md, images)
            page_markdowns.append(page_md)
        markdown = "\n\n".join(page_markdowns).strip()
        markdown = _convert_pipe_tables_to_html(markdown)

        layout_pages = _build_layout_pages(pages) if pages else []

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            layout_pages=layout_pages,
            markdown=markdown,
        )

        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
