"""Provider for Landing AI PARSE."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from landingai_ade import LandingAIADE

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    PageIR,
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

# LandingAI chunk type -> Canonical17 label string
LANDINGAI_LABEL_MAP: dict[str, str] = {
    "text": "Text",
    "table": "Table",
    "figure": "Picture",
    "marginalia": "Page-header",  # headers/footers/page numbers consolidated
    "logo": "Picture",
    "card": "Key-Value Region",
    # "attestation" and "scan_code" have no canonical equivalent — skipped
}

# Virtual page dimensions for normalized coordinate conversion.
# LandingAI bbox is already [0,1], so these cancel out during evaluation.
_VIRTUAL_PAGE_DIM = 1000.0


@register_provider("landingai")
class LandingAIParseProvider(Provider):
    """
    Provider for Landing AI PARSE.

    This provider uses the Landing AI ADE API for parsing tasks.
    """

    CREDIT_RATE_USD = 0.01  # $0.01 per credit (Explore plan)

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `api_key`: Landing AI API key (defaults to LANDING_AI_API_KEY env var)
            - `model`: Model to use (default: "dpt-2-latest")
            - Any other parse parameters from Landing AI API
        """
        super().__init__(provider_name, base_config)

        # Get API key
        self._api_key = self.base_config.get("api_key") or os.getenv("LANDING_AI_API_KEY")
        if not self._api_key:
            raise ProviderConfigError(
                "Landing AI API key is required. "
                "Set LANDING_AI_API_KEY environment variable or pass api_key in base_config."
            )

        # Set VISION_AGENT_API_KEY for the SDK (it expects this env var)
        # Only set if not already set to avoid overriding existing values
        if not os.getenv("VISION_AGENT_API_KEY"):
            os.environ["VISION_AGENT_API_KEY"] = self._api_key

        # Get configuration with defaults
        self._model = self.base_config.get("model", "dpt-2-latest")

        # Initialize client
        self._client = LandingAIADE()

    def _parse_document(self, document_path: Path) -> dict[str, Any]:
        """
        Parse a document using Landing AI API.

        :param document_path: Path to the document file
        :return: Raw API response as dictionary
        :raises ProviderError: For any API errors
        """
        try:
            # Parse the document
            response = self._client.parse(
                document=document_path,
                model=self._model,
                **{k: v for k, v in self.base_config.items() if k not in ["api_key", "model"]},
            )

            # Convert response to dictionary format
            # The response has markdown, chunks, and grounding attributes
            result: dict[str, Any] = {
                "markdown": response.markdown if hasattr(response, "markdown") else "",
                "chunks": [],
                "splits": [],
                "grounding": {},
            }

            # Extract chunks if available
            if hasattr(response, "chunks"):
                chunks = response.chunks
                if chunks is not None:
                    # Convert chunks to serializable format
                    for chunk in chunks:
                        chunk_data: dict[str, Any] = {}
                        if hasattr(chunk, "id"):
                            chunk_data["id"] = chunk.id
                        if hasattr(chunk, "type"):
                            chunk_data["type"] = chunk.type
                        if hasattr(chunk, "markdown"):
                            chunk_data["markdown"] = chunk.markdown
                        if hasattr(chunk, "grounding") and chunk.grounding is not None:
                            # ChunkGrounding is a Pydantic model - convert to dict
                            chunk_data["grounding"] = chunk.grounding.model_dump()
                        result["chunks"].append(chunk_data)

            # Extract splits if available (populated when split="page" is used)
            if hasattr(response, "splits") and response.splits is not None:
                for split in response.splits:
                    split_data: dict[str, Any] = {}
                    if hasattr(split, "markdown"):
                        split_data["markdown"] = split.markdown
                    if hasattr(split, "pages"):
                        split_data["pages"] = split.pages
                    if hasattr(split, "chunks"):
                        split_data["chunks"] = split.chunks
                    if hasattr(split, "class_"):
                        split_data["class"] = split.class_
                    if hasattr(split, "identifier"):
                        split_data["identifier"] = split.identifier
                    result["splits"].append(split_data)

            # Extract grounding if available
            # response.grounding is Dict[str, Grounding] where Grounding is a Pydantic model
            if hasattr(response, "grounding") and response.grounding is not None:
                result["grounding"] = {k: v.model_dump() for k, v in response.grounding.items()}

            # Extract cost from metadata
            if hasattr(response, "metadata") and response.metadata is not None:
                meta = response.metadata
                credits = getattr(meta, "credit_usage", None)
                num_pages = getattr(meta, "page_count", None)
                if credits is not None and credits > 0:
                    cost_usd = credits * self.CREDIT_RATE_USD
                    result["credits_used"] = credits
                    result["cost_usd"] = cost_usd
                    if num_pages and num_pages > 0:
                        result["num_pages"] = num_pages
                        result["cost_per_page_usd"] = cost_usd / num_pages

            return result

        except Exception as e:
            # Check if it's a transient error (network, timeout, etc.)
            error_str = str(e).lower()
            transient_keywords = ["timeout", "network", "connection", "503", "502", "504"]
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during parsing: {e}") from e
            else:
                raise ProviderPermanentError(f"Error during parsing: {e}") from e

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        :param pipeline: Pipeline specification
        :param request: Inference request
        :return: Raw inference result
        :raises ProviderError: For any provider-related failures
        """
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"LandingAIParseProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        # Check if file exists
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        try:
            # Run parsing
            raw_output = self._parse_document(file_path)

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

        except ProviderPermanentError:
            # Re-raise provider errors as-is
            raise
        except ProviderTransientError:
            # Re-raise provider errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce ParseOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"LandingAIParseProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        # Extract markdown from raw output and promote table headers.
        # Landing AI emits all cells as <td>; downstream eval relies on <th>.
        markdown = _promote_first_row_to_header(raw_result.raw_output.get("markdown", ""))

        pages: list[PageIR] = []

        # Strategy 1: Use splits data if available (from split="page")
        splits = raw_result.raw_output.get("splits", [])
        if splits:
            for split in splits:
                if isinstance(split, dict) and "markdown" in split and "pages" in split:
                    split_pages = split["pages"]
                    split_md = _promote_first_row_to_header(split["markdown"])
                    # Each split may cover one or more pages; use the first page number
                    page_num = split_pages[0] if split_pages else 0
                    pages.append(PageIR(page_index=page_num, markdown=split_md))

        # Strategy 2: Fall back to chunk grounding for page splitting
        if not pages:
            chunks = raw_result.raw_output.get("chunks", [])
            grounding = raw_result.raw_output.get("grounding", {})

            page_content: dict[int, list[str]] = {}

            if isinstance(grounding, dict):
                for gid, gdata in grounding.items():
                    if isinstance(gdata, dict) and "page" in gdata:
                        page_num = gdata["page"]
                        if page_num not in page_content:
                            page_content[page_num] = []
                        for chunk in chunks:
                            if isinstance(chunk, dict) and chunk.get("id") == gid:
                                if "markdown" in chunk:
                                    page_content[page_num].append(_promote_first_row_to_header(chunk["markdown"]))

            if page_content:
                for page_num in sorted(page_content.keys()):
                    page_text = "\n".join(page_content[page_num])
                    pages.append(PageIR(page_index=page_num, markdown=page_text))

        # Strategy 3: Fallback — single page with all markdown
        if not pages:
            pages.append(PageIR(page_index=0, markdown=markdown))

        # Build layout_pages from chunk grounding for layout cross-evaluation
        chunks = raw_result.raw_output.get("chunks", [])
        layout_pages = _build_layout_pages(chunks)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=markdown,
            job_id=None,  # Landing AI parse doesn't return job_id
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


def _build_layout_pages(chunks: list[dict[str, Any]]) -> list[ParseLayoutPageIR]:
    """Build layout_pages from LandingAI chunk grounding for layout cross-evaluation.

    Groups chunks by page number and converts each chunk's normalized [0,1]
    bounding box into a LayoutSegmentIR with canonical label mapping.
    LandingAI grounding pages are 0-indexed; we convert to 1-indexed.
    """
    from collections import defaultdict

    pages_chunks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grounding = chunk.get("grounding")
        if not isinstance(grounding, dict):
            continue
        # LandingAI pages are 0-indexed
        page_num = grounding.get("page", 0)
        pages_chunks[page_num].append(chunk)

    layout_pages: list[ParseLayoutPageIR] = []
    for page_num in sorted(pages_chunks.keys()):
        page_chunks = pages_chunks[page_num]
        items: list[LayoutItemIR] = []

        for chunk in page_chunks:
            chunk_type = chunk.get("type", "")
            canonical_label = LANDINGAI_LABEL_MAP.get(chunk_type)
            if canonical_label is None:
                continue  # Skip unmapped types (e.g., attestation, scan_code)

            grounding = chunk.get("grounding", {})
            box = grounding.get("box", {})
            left = float(box.get("left", 0.0))
            top = float(box.get("top", 0.0))
            right = float(box.get("right", 0.0))
            bottom = float(box.get("bottom", 0.0))
            width = right - left
            height = bottom - top

            # Parse confidence (DPT-2 provides it)
            conf_raw = grounding.get("confidence")
            try:
                confidence = float(conf_raw) if conf_raw is not None else 1.0
            except (TypeError, ValueError):
                confidence = 1.0

            seg = LayoutSegmentIR(
                x=left,
                y=top,
                w=width,
                h=height,
                confidence=confidence,
                label=canonical_label,
            )

            content = chunk.get("markdown", "")
            norm_label = canonical_label.strip().lower()
            if norm_label == "table":
                item_type = "table"
            elif norm_label == "picture":
                item_type = "image"
            else:
                item_type = "text"

            items.append(
                LayoutItemIR(
                    type=item_type,
                    value=content,
                    bbox=seg,
                    layout_segments=[seg],
                )
            )

        # Convert 0-indexed page to 1-indexed for ParseLayoutPageIR
        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_num + 1,
                width=_VIRTUAL_PAGE_DIM,
                height=_VIRTUAL_PAGE_DIM,
                items=items,
            )
        )

    return layout_pages


def _promote_first_row_to_header(html: str) -> str:
    """Rewrite HTML tables so the first row uses ``<th>`` inside ``<thead>``.

    Landing AI emits all table cells as ``<td>`` with no
    ``<th>``/``<thead>``/``<tbody>``.  This promotes the first ``<tr>`` of each
    ``<table>`` to be a header row so that downstream evaluation code (which
    keys on ``<th>``) can identify column headers.

    Only tables that contain zero ``<th>`` elements are modified — tables that
    already have headers are left untouched.
    """
    from bs4 import BeautifulSoup

    if "<table" not in html:
        return html

    soup = BeautifulSoup(html, "lxml")
    modified = False

    for table in soup.find_all("table"):
        # Skip tables that already have <th> elements
        if table.find("th"):
            continue

        first_tr = table.find("tr")
        if first_tr is None:
            continue

        # Promote <td> -> <th> in the first row
        for td in first_tr.find_all("td"):
            td.name = "th"

        # Wrap first row in <thead>, remaining rows in <tbody>
        thead = soup.new_tag("thead")
        first_tr.extract()
        thead.append(first_tr)

        tbody = soup.new_tag("tbody")
        for tr in table.find_all("tr"):
            tr.extract()
            tbody.append(tr)

        table.clear()
        table.append(thead)
        if tbody.find("tr"):
            table.append(tbody)

        modified = True

    if not modified:
        return html

    # Return just the body content to avoid <html><body> wrapper
    body = soup.find("body")
    return body.decode_contents() if body else str(soup)
