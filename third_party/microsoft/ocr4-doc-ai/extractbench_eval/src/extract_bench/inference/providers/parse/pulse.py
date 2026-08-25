"""Provider for Pulse PARSE.

Calls the Pulse REST API directly via multipart/form-data. The provider exposes
the public /extract controls and can optionally run /tables after extraction so
registered pipelines can reproduce leaderboard runs without hidden harness code.
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

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

_DEFAULT_API_BASE_URL = "https://api.runpulse.com"
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
_DEFAULT_JOB_TIMEOUT_SECONDS = 4 * 60 * 60.0

# Evaluation consumes these coordinates consistently with the virtual page frame.
_VIRTUAL_PAGE_DIM = 1000.0

# Map Pulse bounding_boxes label keys to canonical layout labels. "Header" is
# disambiguated into Page-header vs Section-header by Y-position.
_PULSE_LABEL_MAP: dict[str, str] = {
    "Title": "Title",
    "Section-header": "Section-header",
    "Text": "Text",
    "List-item": "List-item",
    "List Items": "List-item",
    "Header": "Page-header",
    "Page-header": "Page-header",
    "Footer": "Page-footer",
    "Page-footer": "Page-footer",
    "Page Number": "Page-footer",
    "Page-number": "Page-footer",
    "Image": "Picture",
    "Images": "Picture",
    "Picture": "Picture",
    "Figure": "Picture",
    "Table": "Table",
    "Tables": "Table",
    "Caption": "Caption",
    "caption": "Caption",
    "Footnote": "Footnote",
    "Formula": "Formula",
    "Formulas": "Formula",
}

_PAGE_HEADER_TOP_BAND = 0.10
_PAGE_HEADER_BOTTOM_BAND = 0.90
_PULSE_BBOX_METADATA_KEYS = {"markdown_with_ids"}
_PULSE_BBOX_ORDERED_KEYS = {"ordered_elements"}


@register_provider("pulse")
class PulseProvider(Provider):
    """Provider for Pulse document extraction via REST."""

    CREDIT_RATE_USD = 0.015

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        raw_base_url = (
            self.base_config.get("base_url")
            or self.base_config.get("api_base_url")
            or os.getenv("PULSE_API_BASE_URL")
            or _DEFAULT_API_BASE_URL
        )
        if not isinstance(raw_base_url, str) or not raw_base_url.strip():
            raise ProviderConfigError("Pulse base URL must be a non-empty string")
        self._base_url = raw_base_url.strip().rstrip("/")
        if self._base_url.endswith("/extract"):
            self._base_url = self._base_url[: -len("/extract")]

        api_key = self.base_config.get("api_key") or os.getenv("PULSE_API_KEY")
        if not api_key or not isinstance(api_key, str):
            raise ProviderConfigError(
                "Pulse API key is required. Set PULSE_API_KEY environment variable or pass api_key in base_config."
            )
        self._api_key: str = api_key

        # Core controls
        self._model: str | None = self.base_config.get("model")
        self._pages: str | None = self.base_config.get("pages")

        # Pulse Ultra 2 controls
        self._refine: bool = bool(self.base_config.get("refine", False))
        refine_options = self.base_config.get("refine_options")
        if refine_options is not None and not isinstance(refine_options, dict):
            raise ProviderConfigError("refine_options must be a dict")
        self._refine_options: dict[str, bool] | None = refine_options
        self._extract_figure: bool = bool(self.base_config.get("extract_figure", False))
        self._figure_description: bool = bool(self.base_config.get("figure_description", False))
        self._additional_prompt: str | None = self.base_config.get("additional_prompt")
        self._custom_image_prompt: str | None = self.base_config.get("custom_image_prompt")
        self._custom_refine_prompt: str | None = self.base_config.get("custom_refine_prompt")

        # Public default-model figure processing and alternate outputs.
        figure_processing = self.base_config.get("figure_processing")
        if figure_processing is not None and not isinstance(figure_processing, dict):
            raise ProviderConfigError("figure_processing must be a dict")
        self._figure_processing: dict[str, Any] | None = figure_processing

        extensions = self.base_config.get("extensions")
        if extensions is not None and not isinstance(extensions, dict):
            raise ProviderConfigError("extensions must be a dict")
        self._extensions: dict[str, Any] | None = dict(extensions) if extensions else None
        if self.base_config.get("return_html"):
            self._extensions = dict(self._extensions or {})
            alt_outputs = dict(self._extensions.get("altOutputs") or self._extensions.get("alt_outputs") or {})
            alt_outputs["returnHtml"] = True
            self._extensions["altOutputs"] = alt_outputs

        storage = self.base_config.get("storage")
        if storage is not None and not isinstance(storage, dict):
            raise ProviderConfigError("storage must be a dict")
        self._storage: dict[str, Any] | None = storage

        # /tables post-processing
        tables_config = self.base_config.get("tables_config")
        if tables_config is not None and not isinstance(tables_config, dict):
            raise ProviderConfigError("tables_config must be a dict")
        self._use_tables_endpoint: bool = bool(self.base_config.get("use_tables_endpoint", False))
        self._tables_config: dict[str, Any] = dict(tables_config or {})
        raw_categories = self.base_config.get("tables_endpoint_categories")
        if raw_categories is None:
            self._tables_endpoint_categories: set[str] = set()
        elif isinstance(raw_categories, list):
            self._tables_endpoint_categories = {str(item).lower() for item in raw_categories}
        else:
            raise ProviderConfigError("tables_endpoint_categories must be a list when provided")
        self._merge_tables_into_markdown: bool = bool(self.base_config.get("merge_tables_into_markdown", False))
        self._replace_existing_tables: bool = bool(self.base_config.get("replace_existing_tables", True))

        # Misc / async controls
        self._poll_interval: float = float(self.base_config.get("poll_interval", os.getenv("PULSE_POLL_INTERVAL", 1.0)))
        self._async_extract: bool = bool(self.base_config.get("async_extract", self.base_config.get("async", False)))
        self._async_tables: bool = bool(self.base_config.get("async_tables", False))
        self._force_url: bool = bool(self.base_config.get("force_url", False))
        self._request_timeout: float = float(
            self.base_config.get(
                "request_timeout",
                os.getenv("PULSE_REQUEST_TIMEOUT", _DEFAULT_REQUEST_TIMEOUT_SECONDS),
            )
        )
        self._job_timeout: float = float(
            self.base_config.get(
                "job_timeout",
                os.getenv("PULSE_JOB_TIMEOUT", _DEFAULT_JOB_TIMEOUT_SECONDS),
            )
        )

        self._markdown_source = str(self.base_config.get("markdown_source", "html")).lower()
        if self._markdown_source not in {"html", "markdown"}:
            raise ProviderConfigError("markdown_source must be either 'html' or 'markdown'")

        self._credits_per_page: float | None = None
        raw_credits_per_page = self.base_config.get("credits_per_page")
        if raw_credits_per_page is not None:
            self._credits_per_page = float(raw_credits_per_page)
            if self._credits_per_page < 0:
                raise ProviderConfigError("credits_per_page must be non-negative")

    # --------------------------------------------------------------------- #
    # HTTP call
    # --------------------------------------------------------------------- #

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key}

    def _build_form_fields(self) -> list[tuple[str, tuple[None, str]]]:
        """Build the non-file multipart fields for the /extract POST."""
        fields: list[tuple[str, tuple[None, str]]] = []

        def add(name: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, bool):
                fields.append((name, (None, "true" if value else "false")))
            elif isinstance(value, (dict, list)):
                fields.append((name, (None, json.dumps(value))))
            else:
                fields.append((name, (None, str(value))))

        add("model", self._model)
        add("pages", self._pages)
        add("async", self._async_extract or None)
        add("force_url", self._force_url or None)
        add("figure_processing", self._figure_processing)
        add("extensions", self._extensions)
        add("storage", self._storage)
        add("refine", self._refine or None)
        add("refine_options", self._refine_options)
        add("extract_figure", self._extract_figure or None)
        add("figure_description", self._figure_description or None)
        add("additional_prompt", self._additional_prompt)
        add("custom_image_prompt", self._custom_image_prompt)
        add("custom_refine_prompt", self._custom_refine_prompt)

        return fields

    def _classify_bad_response(self, response: requests.Response, context: str) -> None:
        if response.status_code in (401, 403):
            raise ProviderConfigError(
                f"Pulse auth failed during {context} ({response.status_code}): {response.text[:300]}"
            )
        if response.status_code == 429:
            raise ProviderRateLimitError(f"Pulse rate limit during {context}: {response.text[:300]}")
        if response.status_code in (500, 502, 503, 504):
            raise ProviderTransientError(
                f"Pulse transient during {context} ({response.status_code}): {response.text[:300]}"
            )
        if response.status_code >= 400:
            raise ProviderPermanentError(
                f"Pulse error during {context} ({response.status_code}): {response.text[:300]}"
            )

    def _resolve_large_result(self, raw: dict[str, Any], context: str) -> dict[str, Any]:
        if not raw.get("is_url") or not raw.get("url"):
            return raw
        url_resp = requests.get(raw["url"], timeout=self._request_timeout)
        self._classify_bad_response(url_resp, f"{context} large-result fetch")
        try:
            url_result = url_resp.json()
        except ValueError as e:
            raise ProviderPermanentError(f"Pulse large-result fetch returned non-JSON response: {e}") from e
        if "plan_info" in raw or "plan-info" in raw:
            url_result["plan_info"] = raw.get("plan_info", raw.get("plan-info"))
        return url_result

    def _poll_job(self, job_id: str, context: str) -> dict[str, Any]:
        deadline = time.monotonic() + max(self._job_timeout, self._poll_interval)
        last_state: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            time.sleep(max(self._poll_interval, 0.1))
            response = requests.get(
                f"{self._base_url}/job/{job_id}",
                headers=self._headers(),
                timeout=self._request_timeout,
            )
            self._classify_bad_response(response, f"{context} poll")
            try:
                state = response.json()
            except ValueError as e:
                raise ProviderTransientError(f"Pulse {context} poll returned non-JSON response: {e}") from e
            if not isinstance(state, dict):
                raise ProviderTransientError(f"Pulse {context} poll returned invalid state: {state}")
            last_state = state

            status = state.get("status")
            if status == "completed":
                result = state.get("result")
                if isinstance(result, dict):
                    return self._resolve_large_result(result, f"{context} poll result")
                return state
            if status in {"failed", "canceled", "expired"}:
                raise ProviderPermanentError(
                    f"Pulse {context} job {job_id} ended with status={status}: {state.get('error', state)}"
                )
        raise ProviderTransientError(
            f"Pulse {context} job {job_id} did not complete within {self._job_timeout:.0f}s. Last state: {last_state}"
        )

    def _extract(self, file_path: str) -> dict[str, Any]:
        form_fields = self._build_form_fields()

        with open(file_path, "rb") as f:
            files: list[tuple[str, Any]] = [("file", (Path(file_path).name, f, "application/pdf"))]
            files.extend(form_fields)
            response = requests.post(
                f"{self._base_url}/extract",
                headers=self._headers(),
                files=files,
                timeout=self._request_timeout,
            )

        self._classify_bad_response(response, "extract submit")
        try:
            raw: dict[str, Any] = response.json()
        except ValueError as e:
            raise ProviderPermanentError(f"Pulse returned non-JSON response: {e}") from e

        job_id = raw.get("job_id")
        if isinstance(job_id, str) and (self._async_extract or raw.get("status") in {"pending", "processing"}):
            return self._poll_job(job_id, "extract")

        return self._resolve_large_result(raw, "extract")

    def _should_run_tables_endpoint(self, file_path: Path) -> bool:
        if not self._use_tables_endpoint:
            return False
        if not self._tables_endpoint_categories:
            return True
        parts = [part.lower() for part in file_path.parts]
        return any(cat in part for part in parts for cat in self._tables_endpoint_categories)

    def _extract_tables(self, extraction_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"extraction_id": extraction_id}
        if self._tables_config:
            payload["tables_config"] = self._tables_config
        if self._async_tables:
            payload["async"] = True

        response = requests.post(
            f"{self._base_url}/tables",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=self._request_timeout,
        )
        self._classify_bad_response(response, "tables submit")
        try:
            raw: dict[str, Any] = response.json()
        except ValueError as e:
            raise ProviderPermanentError(f"Pulse tables returned non-JSON response: {e}") from e

        job_id = raw.get("job_id") or raw.get("tables_id")
        if isinstance(job_id, str) and (self._async_tables or raw.get("status") in {"pending", "processing"}):
            return self._poll_job(job_id, "tables")

        return self._resolve_large_result(raw, "tables")

    # --------------------------------------------------------------------- #
    # Provider interface
    # --------------------------------------------------------------------- #

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"PulseProvider only supports PARSE product type, got {request.product_type}")

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        started_at = datetime.now()

        try:
            raw_output = self._extract(str(file_path))
            extraction_id = raw_output.get("extraction_id")
            if isinstance(extraction_id, str) and self._should_run_tables_endpoint(file_path):
                raw_output["_tables_result"] = self._extract_tables(extraction_id)
        except (
            ProviderPermanentError,
            ProviderTransientError,
            ProviderConfigError,
            ProviderRateLimitError,
        ):
            raise
        except requests.ConnectionError as e:
            raise ProviderTransientError(f"Pulse connection error: {e}") from e
        except requests.Timeout as e:
            raise ProviderTransientError(f"Pulse request timed out: {e}") from e
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

        raw_output["_config"] = {
            "base_url": self._base_url,
            "model": self._model,
            "async_extract": self._async_extract,
            "refine": self._refine,
            "refine_options": self._refine_options,
            "extract_figure": self._extract_figure,
            "figure_description": self._figure_description,
            "figure_processing": self._figure_processing,
            "extensions": self._extensions,
            "storage": self._storage,
            "custom_image_prompt": self._custom_image_prompt,
            "custom_refine_prompt": self._custom_refine_prompt,
            "additional_prompt": self._additional_prompt,
            "pages": self._pages,
            "use_tables_endpoint": self._use_tables_endpoint,
            "tables_config": self._tables_config,
            "tables_endpoint_categories": sorted(self._tables_endpoint_categories),
            "merge_tables_into_markdown": self._merge_tables_into_markdown,
            "replace_existing_tables": self._replace_existing_tables,
            "async_tables": self._async_tables,
            "markdown_source": self._markdown_source,
            "request_timeout": self._request_timeout,
            "job_timeout": self._job_timeout,
            "credits_per_page": self._credits_per_page,
        }

        if self._use_tables_endpoint:
            raw_output["_tables_endpoint_applied"] = "_tables_result" in raw_output

        plan_info = raw_output.get("plan-info", raw_output.get("plan_info", {}))
        pages_used = None
        if isinstance(plan_info, dict):
            pages_used = plan_info.get("pages_used")
        if pages_used is None:
            pages_used = raw_output.get("page_count", raw_output.get("num_pages"))
        try:
            pages_used_float = float(pages_used)
        except (TypeError, ValueError):
            pages_used_float = 0.0
        if pages_used_float > 0:
            raw_output["num_pages"] = int(pages_used_float) if pages_used_float.is_integer() else pages_used_float
            if self._credits_per_page is not None:
                cost_per_page_usd = self._credits_per_page * self.CREDIT_RATE_USD
                raw_output["cost_usd"] = pages_used_float * cost_per_page_usd
                raw_output["cost_per_page_usd"] = cost_per_page_usd

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
            raise ProviderPermanentError(
                f"PulseProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        raw = raw_result.raw_output
        html_content = _get_pulse_html(raw)
        native_markdown = raw.get("markdown")
        if self._markdown_source == "markdown" and isinstance(native_markdown, str) and native_markdown:
            markdown = native_markdown
        elif self._markdown_source == "html" and html_content:
            markdown = html_content
        elif html_content:
            markdown = html_content
        elif isinstance(native_markdown, str):
            markdown = native_markdown
        else:
            markdown = ""
        if self._merge_tables_into_markdown:
            markdown = _merge_tables_endpoint_into_markdown(
                markdown,
                raw.get("_tables_result"),
                replace_existing_tables=self._replace_existing_tables,
            )
        layout_pages = _build_layout_pages(raw.get("bounding_boxes", {}))

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            layout_pages=layout_pages,
            markdown=markdown,
            job_id=raw.get("extraction_id"),
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


# ------------------------------------------------------------------------- #
# Output normalization helpers
# ------------------------------------------------------------------------- #


def _polygon_to_xywh(coords: list[float]) -> tuple[float, float, float, float]:
    """Convert an 8-float polygon [x1,y1, x2,y2, x3,y3, x4,y4] to (x, y, w, h)."""
    xs = [coords[i] for i in range(0, 8, 2)]
    ys = [coords[i] for i in range(1, 8, 2)]
    x = min(xs)
    y = min(ys)
    return x, y, max(xs) - x, max(ys) - y


def _normalize_coords(raw_coords: Any) -> list[float] | None:
    """Normalize Pulse bbox variants to an 8-point polygon."""
    if not isinstance(raw_coords, list):
        return None
    if len(raw_coords) == 4:
        coords = [
            raw_coords[0],
            raw_coords[1],
            raw_coords[2],
            raw_coords[1],
            raw_coords[2],
            raw_coords[3],
            raw_coords[0],
            raw_coords[3],
        ]
    elif len(raw_coords) >= 8:
        coords = raw_coords[:8]
    else:
        return None

    try:
        floats = [float(v) for v in coords]
    except (TypeError, ValueError):
        return None

    max_coord = max(abs(v) for v in floats) if floats else 0.0
    if max_coord > 1.5:
        floats = [v / _VIRTUAL_PAGE_DIM for v in floats]

    clipped = [min(1.0, max(0.0, v)) for v in floats]
    _, _, w, h = _polygon_to_xywh(clipped)
    if w <= 0 or h <= 0:
        return None
    return clipped


def _get_pulse_html(raw: dict[str, Any]) -> str:
    extensions = raw.get("extensions")
    if isinstance(extensions, dict):
        for key in ("alt_outputs", "altOutputs"):
            alt_outputs = extensions.get(key)
            if isinstance(alt_outputs, dict):
                for html_key in ("html", "returnHtml", "return_html"):
                    html = alt_outputs.get(html_key)
                    if isinstance(html, str) and html:
                        return html

    html = raw.get("html")
    if isinstance(html, str) and html:
        return html

    return ""


def _table_endpoint_tables(tables_result: Any) -> list[dict[str, Any]]:
    if not isinstance(tables_result, dict):
        return []
    result = tables_result.get("result")
    if isinstance(result, dict):
        tables_result = result
    tables_output = tables_result.get("tables_output")
    if not isinstance(tables_output, dict):
        return []
    tables = tables_output.get("tables")
    if not isinstance(tables, list):
        return []
    return [table for table in tables if isinstance(table, dict)]


def _format_endpoint_table(table: dict[str, Any]) -> str:
    content = table.get("table_content")
    if not isinstance(content, str) or "<table" not in content.lower():
        return ""
    citations = table.get("citations")
    citations_text = ""
    if isinstance(citations, list) and citations:
        citations_text = " ".join(str(citation) for citation in citations)
    from_chart = "true" if table.get("from_chart") else "false"
    comment_parts = [f"from_chart={from_chart}"]
    if citations_text:
        comment_parts.append(f"citations={citations_text}")
    return f"<!-- pulse-tables-endpoint {'; '.join(comment_parts)} -->\n{content.strip()}"


def _strip_html_table_blocks(content: str) -> str:
    if "<table" not in content.lower():
        return content

    lower = content.lower()
    pieces: list[str] = []
    search_start = 0
    while True:
        start = lower.find("<table", search_start)
        if start == -1:
            pieces.append(content[search_start:])
            break
        tag_name_end = start + len("<table")
        if tag_name_end < len(lower) and lower[tag_name_end] not in (">", " ", "\t", "\n", "\r"):
            pieces.append(content[search_start : start + 1])
            search_start = start + 1
            continue
        pieces.append(content[search_start:start])

        depth = 0
        pos = start
        end = -1
        while pos < len(lower):
            next_open = lower.find("<table", pos + 1)
            next_close = lower.find("</table>", pos + 1)
            if next_close == -1:
                break
            if next_open != -1 and next_open < next_close:
                nested_name_end = next_open + len("<table")
                if nested_name_end < len(lower) and lower[nested_name_end] not in (
                    ">",
                    " ",
                    "\t",
                    "\n",
                    "\r",
                ):
                    pos = next_open
                    continue
                depth += 1
                pos = next_open
                continue
            if depth == 0:
                end = next_close + len("</table>")
                break
            depth -= 1
            pos = next_close

        if end == -1:
            search_start = len(content)
            break
        search_start = end

    return "\n".join(part for part in "".join(pieces).splitlines() if part.strip())


def _merge_tables_endpoint_into_markdown(
    markdown: str,
    tables_result: Any,
    *,
    replace_existing_tables: bool,
) -> str:
    table_blocks = [_format_endpoint_table(table) for table in _table_endpoint_tables(tables_result)]
    table_blocks = [block for block in table_blocks if block]
    if not table_blocks:
        return markdown

    base = _strip_html_table_blocks(markdown) if replace_existing_tables else markdown
    return f"{base.rstrip()}\n\n" + "\n\n".join(table_blocks)


def _canonical_label(raw_label: Any, y: float | None = None) -> str:
    label = str(raw_label or "Text")
    mapped = _PULSE_LABEL_MAP.get(label)
    if mapped is None:
        mapped = _PULSE_LABEL_MAP.get(label.strip())
    if mapped is None:
        mapped = _PULSE_LABEL_MAP.get(label.replace("_", "-"))
    if mapped is None:
        mapped = label

    if mapped == "Page-header" and y is not None and _PAGE_HEADER_TOP_BAND <= y <= _PAGE_HEADER_BOTTOM_BAND:
        return "Section-header"
    return mapped


def _extract_grouped_table(elem: dict[str, Any]) -> tuple[Any, Any, Any, str]:
    table_info = elem.get("table_info", {})
    location = table_info.get("location", {}) if isinstance(table_info, dict) else {}
    coords = location.get("coordinates") or elem.get("bounding_box") or elem.get("bbox_normalized") or elem.get("bbox")
    page_num = (
        location.get("page", elem.get("page_number", 1)) if isinstance(location, dict) else elem.get("page_number", 1)
    )
    conf_raw = table_info.get("confidence") if isinstance(table_info, dict) else elem.get("confidence")
    cell_texts = []
    for cell in elem.get("cell_data", []):
        text = str(cell.get("text", ""))
        if text.startswith("0t-"):
            text = text[3:]
        cell_texts.append(text)
    content = elem.get("content") or elem.get("original_content") or " ".join(cell_texts)
    return coords, page_num, conf_raw, str(content or "")


def _iter_bbox_elements(bounding_boxes: Any):
    """Yield normalized raw bbox entries from grouped or flat Pulse outputs."""
    if isinstance(bounding_boxes, list):
        for elem in bounding_boxes:
            if isinstance(elem, dict):
                yield elem.get("category", elem.get("source_category", elem.get("label", "Text"))), elem
        return

    if not isinstance(bounding_boxes, dict):
        return

    has_grouped_boxes = any(
        key not in _PULSE_BBOX_METADATA_KEYS | _PULSE_BBOX_ORDERED_KEYS and isinstance(elements, list)
        for key, elements in bounding_boxes.items()
    )
    for label_key, elements in bounding_boxes.items():
        if label_key in _PULSE_BBOX_METADATA_KEYS:
            continue
        if label_key in _PULSE_BBOX_ORDERED_KEYS and has_grouped_boxes:
            continue
        if not isinstance(elements, list):
            continue
        for elem in elements:
            if isinstance(elem, dict):
                if label_key in _PULSE_BBOX_ORDERED_KEYS:
                    yield elem.get("source_category", elem.get("category", elem.get("type", "Text"))), elem
                else:
                    yield label_key, elem


def _build_layout_pages(bounding_boxes: Any) -> list[ParseLayoutPageIR]:
    pages_items: dict[int, list[LayoutItemIR]] = defaultdict(list)

    for label_key, elem in _iter_bbox_elements(bounding_boxes):
        if not isinstance(elem, dict):
            continue

        if str(label_key) in {"Tables", "Table"} and "cell_data" in elem:
            coords, page_num, conf_raw, content = _extract_grouped_table(elem)
        else:
            coords = (
                elem.get("bounding_box") or elem.get("bbox_normalized") or elem.get("bbox") or elem.get("coordinates")
            )
            page_num = elem.get("page_number", elem.get("page", 1))
            conf_raw = elem.get("average_word_confidence", elem.get("confidence"))
            content = elem.get("original_content", elem.get("content", elem.get("text", "")))

        normalized = _normalize_coords(coords)
        if normalized is None:
            continue

        try:
            confidence = float(conf_raw) if conf_raw is not None and conf_raw != "N/A" else 1.0
        except (TypeError, ValueError):
            confidence = 1.0

        try:
            page_number = int(page_num)
        except (TypeError, ValueError):
            page_number = 1

        x, y, w, h = _polygon_to_xywh(normalized)
        elem_label = _canonical_label(label_key, y)

        seg = LayoutSegmentIR(x=x, y=y, w=w, h=h, confidence=confidence, label=elem_label)

        norm_label = elem_label.strip().lower()
        if norm_label == "table":
            item_type = "table"
        elif norm_label == "picture":
            item_type = "image"
        else:
            item_type = "text"

        pages_items[page_number].append(
            LayoutItemIR(type=item_type, value=str(content or ""), bbox=seg, layout_segments=[seg])
        )

    layout_pages: list[ParseLayoutPageIR] = []
    for page_num in sorted(pages_items.keys()):
        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_num,
                width=_VIRTUAL_PAGE_DIM,
                height=_VIRTUAL_PAGE_DIM,
                items=pages_items[page_num],
            )
        )
    return layout_pages
