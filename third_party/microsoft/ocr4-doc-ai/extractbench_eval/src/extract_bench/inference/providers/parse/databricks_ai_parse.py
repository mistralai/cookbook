"""Provider for Databricks ``ai_parse_document`` SQL function.

``ai_parse_document`` is a Databricks built-in SQL function. It has no
dedicated REST endpoint, so we invoke it via the Statement Execution API
on a SQL Warehouse. The input byte argument must reference a Unity Catalog
Volume (the ``BINARY`` parameter type is not supported by the SQL
parameters wire format).

Operating modes
---------------
``batch_size = 1`` (default): one SQL statement per request::

    PUT /api/2.0/fs/files/<volume>/<uuid>.pdf
    POST /api/2.0/sql/statements/  →  SELECT ai_parse_document(content)
                                       FROM READ_FILES('<volume>/<uuid>.pdf', format => 'binaryFile')
    poll until terminal
    DELETE /api/2.0/fs/files/<volume>/<uuid>.pdf

``batch_size > 1``: coalesce up to K concurrent requests into a single
statement::

    PUT /api/2.0/fs/directories/<volume>/batch-<uuid>
    PUT /api/2.0/fs/files/<volume>/batch-<uuid>/<i>.pdf  (xK)
    POST /api/2.0/sql/statements/  →  SELECT path, ai_parse_document(content)
                                       FROM READ_FILES('<volume>/batch-<uuid>', format => 'binaryFile')
    poll, follow next_chunk_internal_link if needed, demux by path
    DELETE files + DELETE directory

Batching amortizes SQL/warehouse warm-up overhead. ``ai_parse_document``
itself is billed per-page summed across the batch, so model DBUs do not
change — only orchestration cost drops.

The returned VARIANT is a JSON object shaped like::

    {
      "document": {
        "pages": [{"id": int, "image_uri": str}],
        "elements": [
          {"id": int, "type": str, "content": str,
           "confidence": float, "bbox": [{"coord": [...], "page_id": int}],
           "description": str}
        ]
      },
      "error_status": [...],
      "metadata": {...}
    }

Element ``type`` is one of: text, table, figure, title, caption,
section_header, page_header, page_footer, page_number, footnote.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

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

# ai_parse_document element type -> Canonical17 label
DATABRICKS_LABEL_MAP: dict[str, str] = {
    "title": "Title",
    "section_header": "Section-header",
    "text": "Text",
    "table": "Table",
    "figure": "Picture",
    "caption": "Caption",
    "page_header": "Page-header",
    "page_footer": "Page-footer",
    "page_number": "Page-footer",
    "footnote": "Footnote",
}

# ai_parse_document returns element bboxes in absolute pixel coordinates of
# the page it rendered internally. For PDFs that render happens at 200 DPI
# (v2 default), so true page dims in that space are points * 200/72; image
# inputs keep their native pixel dims. Normalizing by anything else (e.g. the
# max element extent on the page) shifts and stretches every box.
AI_PARSE_RENDER_DPI = 200.0

# Fallback page dimension when the source file can't be read at normalize
# time (bboxes then stay normalized by element extent — degraded but usable).
_VIRTUAL_PAGE_DIM = 1000.0

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELED", "CLOSED"}
_TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}

_QueueItem = tuple[InferenceRequest, PipelineSpec, "concurrent.futures.Future[RawInferenceResult]"]


@register_provider("databricks_ai_parse")
class DatabricksAiParseProvider(Provider):
    """Provider for Databricks ``ai_parse_document``.

    Config:
        - host (str, required): Workspace host, e.g.
          ``adb-xxx.azuredatabricks.net``. Reads ``DATABRICKS_HOST`` if unset.
        - token (str, required): PAT / OAuth bearer token. Reads
          ``DATABRICKS_TOKEN`` if unset.
        - warehouse_id (str, required): SQL Warehouse to run the statement
          on. Reads ``DATABRICKS_SQL_WAREHOUSE_ID`` if unset.
        - volume_path (str, required): UC Volume prefix used as a staging
          area, e.g. ``/Volumes/main/default/llamabench``. Reads
          ``DATABRICKS_AI_PARSE_VOLUME`` if unset.
        - version (str, default "2.0"): ai_parse_document schema version.
        - description_element_types (str, default ""): pass-through for the
          ``descriptionElementTypes`` option (``""``, ``"figure"``, ``"*"``).
        - poll_interval (float, default 2.0): seconds between polls.
        - timeout (int, default 900): total wait budget in seconds for the
          SQL statement.
        - batch_size (int, default 1): number of requests to coalesce into
          a single SQL statement. ``1`` = per-file mode.
        - batch_wait_seconds (float, default 10): when batch_size > 1, the
          debounce window — once the first request arrives, wait at most
          this long for the batch to fill before flushing.
        - per_request_timeout (int, default 1800): max seconds a single
          ``run_inference`` call will wait for its batch to complete.
          Only used when batch_size > 1.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        host = self.base_config.get("host") or os.getenv("DATABRICKS_HOST")
        token = self.base_config.get("token") or os.getenv("DATABRICKS_TOKEN")
        warehouse_id = self.base_config.get("warehouse_id") or os.getenv("DATABRICKS_SQL_WAREHOUSE_ID")
        volume_path = self.base_config.get("volume_path") or os.getenv("DATABRICKS_AI_PARSE_VOLUME")

        if not host:
            raise ProviderConfigError(
                "Databricks host is required. Set DATABRICKS_HOST env var or pass 'host' in base_config."
            )
        if not token:
            raise ProviderConfigError(
                "Databricks token is required. Set DATABRICKS_TOKEN env var or pass 'token' in base_config."
            )
        if not warehouse_id:
            raise ProviderConfigError(
                "Databricks warehouse_id is required. "
                "Set DATABRICKS_SQL_WAREHOUSE_ID env var or pass 'warehouse_id' in base_config."
            )
        if not volume_path:
            raise ProviderConfigError(
                "Databricks volume_path is required. "
                "Set DATABRICKS_AI_PARSE_VOLUME env var (e.g. '/Volumes/main/default/llamabench') "
                "or pass 'volume_path' in base_config."
            )
        if not volume_path.startswith("/Volumes/"):
            raise ProviderConfigError(f"volume_path must start with '/Volumes/' (got {volume_path!r}).")

        self._base_url = f"https://{host.rstrip('/').removeprefix('https://').removeprefix('http://')}"
        self._auth_headers = {"Authorization": f"Bearer {token}"}
        self._warehouse_id = warehouse_id
        self._volume_base = volume_path.rstrip("/")
        self._version = str(self.base_config.get("version", "2.0"))
        self._description_element_types = self.base_config.get("description_element_types", "")
        self._poll_interval = float(self.base_config.get("poll_interval", 2.0))
        self._timeout = int(self.base_config.get("timeout", 900))

        batch_size = int(self.base_config.get("batch_size", 1))
        self._batch_size = max(1, batch_size)
        self._batch_wait_s = float(self.base_config.get("batch_wait_seconds", 10.0))
        self._per_request_timeout = int(self.base_config.get("per_request_timeout", 1800))

        # Batch worker is lazy — only spawned when batch_size > 1 and the
        # first request arrives.
        self._queue: queue.Queue[_QueueItem] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()

    # ------------------------------------------------------------------ HTTP

    def _upload_file(self, local_path: Path, remote_path: str) -> None:
        url = f"{self._base_url}/api/2.0/fs/files{remote_path}"
        with open(local_path, "rb") as fh:
            resp = requests.put(
                url,
                params={"overwrite": "true"},
                headers={**self._auth_headers, "Content-Type": "application/octet-stream"},
                data=fh,
                timeout=self._timeout,
            )
        self._raise_for_http(resp, f"upload {remote_path}")

    def _delete_file(self, remote_path: str) -> None:
        url = f"{self._base_url}/api/2.0/fs/files{remote_path}"
        try:
            requests.delete(url, headers=self._auth_headers, timeout=60)
        except Exception:
            # Cleanup is best-effort; never mask a parse failure with a delete failure.
            pass

    def _create_directory(self, remote_dir: str) -> None:
        url = f"{self._base_url}/api/2.0/fs/directories{remote_dir}"
        resp = requests.put(url, headers=self._auth_headers, timeout=60)
        self._raise_for_http(resp, f"create directory {remote_dir}")

    def _delete_directory(self, remote_dir: str) -> None:
        url = f"{self._base_url}/api/2.0/fs/directories{remote_dir}"
        try:
            requests.delete(url, headers=self._auth_headers, timeout=60)
        except Exception:
            pass

    @staticmethod
    def _raise_for_http(resp: requests.Response, context: str) -> None:
        if resp.ok:
            return
        text = resp.text[:500]
        if resp.status_code in _TRANSIENT_HTTP:
            raise ProviderTransientError(f"HTTP {resp.status_code} during {context}: {text}")
        raise ProviderPermanentError(f"HTTP {resp.status_code} during {context}: {text}")

    # ------------------------------------------------------------------ SQL

    def _build_statement(self, source_ref: str, *, include_path: bool) -> str:
        options = [f"'version', '{self._version}'"]
        if self._description_element_types:
            safe = self._description_element_types.replace("'", "''")
            options.append(f"'descriptionElementTypes', '{safe}'")
        option_map = ", ".join(options)
        select_cols = "path, " if include_path else ""
        return (
            f"SELECT {select_cols}ai_parse_document(content, map({option_map})) AS result "
            f"FROM READ_FILES('{source_ref}', format => 'binaryFile')"
        )

    def _execute_statement(self, statement: str) -> dict[str, Any]:
        payload = {
            "warehouse_id": self._warehouse_id,
            "statement": statement,
            "wait_timeout": "50s",
            "on_wait_timeout": "CONTINUE",
            "disposition": "INLINE",
            "format": "JSON_ARRAY",
        }
        url = f"{self._base_url}/api/2.0/sql/statements/"
        resp = requests.post(
            url,
            headers={**self._auth_headers, "Content-Type": "application/json"},
            json=payload,
            timeout=self._timeout,
        )
        self._raise_for_http(resp, "submit statement")
        body = resp.json()

        deadline = time.time() + self._timeout
        while body["status"]["state"] not in _TERMINAL_STATES:
            if time.time() > deadline:
                raise ProviderTransientError(
                    f"Databricks statement {body.get('statement_id')!r} did not finish within {self._timeout}s."
                )
            time.sleep(self._poll_interval)
            poll = requests.get(
                f"{self._base_url}/api/2.0/sql/statements/{body['statement_id']}",
                headers=self._auth_headers,
                timeout=60,
            )
            self._raise_for_http(poll, "poll statement")
            body = poll.json()

        state = body["status"]["state"]
        if state != "SUCCEEDED":
            err = body["status"].get("error") or {}
            msg = err.get("message") or state
            raise ProviderPermanentError(f"Databricks statement ended in {state}: {msg}")

        return self._collect_all_result_chunks(body)

    def _collect_all_result_chunks(self, body: dict[str, Any]) -> dict[str, Any]:
        """Follow ``next_chunk_internal_link`` so callers see one unified
        ``result.data_array``. INLINE responses are capped at 25 MiB per
        chunk."""
        result = body.get("result") or {}
        all_rows: list[list[Any]] = list(result.get("data_array") or [])
        next_link = result.get("next_chunk_internal_link")
        while next_link:
            r = requests.get(
                f"{self._base_url}{next_link}",
                headers=self._auth_headers,
                timeout=self._timeout,
            )
            self._raise_for_http(r, "fetch result chunk")
            chunk = r.json()
            all_rows.extend(chunk.get("data_array") or [])
            next_link = chunk.get("next_chunk_internal_link")
        body.setdefault("result", {})["data_array"] = all_rows
        return body

    @staticmethod
    def _coerce_variant(cell: Any) -> dict[str, Any]:
        if cell is None:
            raise ProviderPermanentError("Databricks ai_parse_document returned NULL.")
        if isinstance(cell, str):
            try:
                parsed = json.loads(cell)
            except json.JSONDecodeError as e:
                raise ProviderPermanentError(f"Failed to decode VARIANT JSON: {e}") from e
            if not isinstance(parsed, dict):
                raise ProviderPermanentError(f"VARIANT JSON is not an object: {type(parsed).__name__}")
            return parsed
        if isinstance(cell, dict):
            return cell
        raise ProviderPermanentError(f"Unexpected VARIANT cell type: {type(cell).__name__}")

    @staticmethod
    def _normalize_row_path(row_path: str) -> str:
        """``READ_FILES`` returns full volume URIs. Strip any ``dbfs:``
        prefix that older runtimes add, just in case."""
        if row_path.startswith("dbfs:"):
            return row_path[len("dbfs:") :]
        return row_path

    # ------------------------------------------------------------------ Inference

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"DatabricksAiParseProvider only supports PARSE, got {request.product_type}")
        if self._batch_size <= 1:
            return self._run_single(pipeline, request)
        return self._run_batched(pipeline, request)

    # Per-file mode -------------------------------------------------------

    def _run_single(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        source = Path(request.source_file_path)
        if not source.exists():
            raise ProviderPermanentError(f"Source file not found: {source}")

        remote_name = f"{uuid.uuid4().hex}{source.suffix.lower()}"
        remote_path = f"{self._volume_base}/{remote_name}"

        started_at = datetime.now()
        try:
            self._upload_file(source, remote_path)
            statement = self._build_statement(remote_path, include_path=False)
            response = self._execute_statement(statement)
            rows = (response.get("result") or {}).get("data_array") or []
            if not rows or not rows[0]:
                raise ProviderPermanentError("Databricks statement returned no rows.")
            variant = self._coerce_variant(rows[0][0])
        finally:
            self._delete_file(remote_path)

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={
                "ai_parse_document": variant,
                "statement_id": response.get("statement_id"),
                "_config": self._config_snapshot(),
            },
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=latency_ms,
        )

    # Batch mode ----------------------------------------------------------

    def _run_batched(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        self._ensure_worker_started()
        fut: concurrent.futures.Future[RawInferenceResult] = concurrent.futures.Future()
        self._queue.put((request, pipeline, fut))
        return fut.result(timeout=self._per_request_timeout)

    def _ensure_worker_started(self) -> None:
        if self._worker is not None:
            return
        with self._worker_lock:
            if self._worker is None:
                t = threading.Thread(
                    target=self._worker_loop,
                    name="databricks-ai-parse-batch",
                    daemon=True,
                )
                t.start()
                self._worker = t

    def _worker_loop(self) -> None:
        while True:
            batch: list[_QueueItem] = [self._queue.get()]
            deadline = time.time() + self._batch_wait_s
            while len(batch) < self._batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                self._process_batch(batch)
            except Exception as exc:  # noqa: BLE001 — propagate to awaiting futures
                for _, _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)

    def _process_batch(self, batch: list[_QueueItem]) -> None:
        started_at = datetime.now()
        batch_id = uuid.uuid4().hex
        batch_dir = f"{self._volume_base}/batch-{batch_id}"

        self._create_directory(batch_dir)

        # Key the demux mapping by the full volume path READ_FILES echoes back.
        file_mapping: dict[str, _QueueItem] = {}
        uploaded: list[str] = []
        try:
            for idx, item in enumerate(batch):
                req, _pipe, fut = item
                src = Path(req.source_file_path)
                if not src.exists():
                    if not fut.done():
                        fut.set_exception(ProviderPermanentError(f"Source file not found: {src}"))
                    continue
                remote_name = f"{idx:04d}-{uuid.uuid4().hex}{src.suffix.lower()}"
                remote_path = f"{batch_dir}/{remote_name}"
                try:
                    self._upload_file(src, remote_path)
                except Exception as exc:  # noqa: BLE001
                    if not fut.done():
                        fut.set_exception(exc)
                    continue
                uploaded.append(remote_path)
                file_mapping[remote_path] = item

            if not file_mapping:
                return

            statement = self._build_statement(batch_dir, include_path=True)
            response = self._execute_statement(statement)
            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

            rows = (response.get("result") or {}).get("data_array") or []
            fulfilled: set[str] = set()
            for row in rows:
                if not row or len(row) < 2:
                    continue
                row_path = self._normalize_row_path(row[0])
                entry = file_mapping.get(row_path)
                if entry is None or entry[2].done():
                    fulfilled.add(row_path)
                    continue
                req_i, pipe_i, fut = entry
                try:
                    variant = self._coerce_variant(row[1])
                except Exception as exc:  # noqa: BLE001
                    fut.set_exception(exc)
                    fulfilled.add(row_path)
                    continue
                fut.set_result(
                    RawInferenceResult(
                        request=req_i,
                        pipeline=pipe_i,
                        pipeline_name=pipe_i.pipeline_name,
                        product_type=req_i.product_type,
                        raw_output={
                            "ai_parse_document": variant,
                            "statement_id": response.get("statement_id"),
                            "batch_id": batch_id,
                            "batch_size_actual": len(file_mapping),
                            "_config": self._config_snapshot(),
                        },
                        started_at=started_at,
                        completed_at=completed_at,
                        latency_in_ms=latency_ms,
                    )
                )
                fulfilled.add(row_path)

            for path, (_req, _pipe, fut) in file_mapping.items():
                if path not in fulfilled and not fut.done():
                    fut.set_exception(ProviderPermanentError(f"Databricks batch statement returned no row for {path}"))
        finally:
            for path in uploaded:
                self._delete_file(path)
            self._delete_directory(batch_dir)

    def _config_snapshot(self) -> dict[str, Any]:
        return {
            "version": self._version,
            "description_element_types": self._description_element_types,
            "warehouse_id": self._warehouse_id,
            "batch_size": self._batch_size,
            "batch_wait_seconds": self._batch_wait_s,
        }

    # ------------------------------------------------------------------ Normalize

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DatabricksAiParseProvider only supports PARSE, got {raw_result.product_type}"
            )

        variant = raw_result.raw_output.get("ai_parse_document") or {}
        document = variant.get("document") or {}
        elements: list[dict[str, Any]] = document.get("elements") or []

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            layout_pages=_build_layout_pages(document, raw_result.request.source_file_path),
            markdown=_render_markdown(elements),
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


def _primary_page_id(element: dict[str, Any]) -> int:
    bboxes = element.get("bbox") or []
    for box in bboxes:
        pid = box.get("page_id")
        if pid is not None:
            try:
                return int(pid)
            except (TypeError, ValueError):
                continue
    return 0


def _render_markdown(elements: list[dict[str, Any]]) -> str:
    """Concatenate element content in reading order, grouped by page."""
    from collections import defaultdict

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for el in elements:
        by_page[_primary_page_id(el)].append(el)

    parts: list[str] = []
    for page_id in sorted(by_page.keys()):
        for el in sorted(by_page[page_id], key=lambda e: e.get("id", 0)):
            content = (el.get("content") or "").strip()
            if not content:
                continue
            el_type = (el.get("type") or "").lower()
            if el_type == "title":
                parts.append(f"# {content}")
            elif el_type == "section_header":
                parts.append(f"## {content}")
            else:
                parts.append(content)
    return "\n\n".join(parts)


def _rendered_page_dims(source_file_path: str) -> list[tuple[float, float]] | None:
    """Per-page (width, height) of ai_parse's internally rendered pages, in
    the same pixel space as the returned bbox coordinates.

    Image inputs are processed at native pixel dims; PDFs are rendered at
    ``AI_PARSE_RENDER_DPI``, so true dims are page points * DPI/72. Returns
    None when the source file is missing or unreadable (e.g. renormalizing
    on a machine without the dataset) so callers can fall back gracefully.
    """
    path = Path(source_file_path)
    if not path.is_file():
        return None
    try:
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            from PIL import Image

            with Image.open(path) as im:
                return [(float(im.width), float(im.height))]

        import fitz  # PyMuPDF

        scale = AI_PARSE_RENDER_DPI / 72.0
        with fitz.open(path) as doc:
            return [(page.rect.width * scale, page.rect.height * scale) for page in doc]
    except Exception:  # noqa: BLE001 — never fail normalization over page dims
        return None


def _build_layout_pages(document: dict[str, Any], source_file_path: str) -> list[ParseLayoutPageIR]:
    """Group elements by page and convert bboxes to normalized LayoutSegmentIR.

    Coordinates are normalized into [0,1] by the true rendered page
    dimensions (see ``_rendered_page_dims``). When dims can't be derived the
    page falls back to normalizing by the max element extent, which keeps
    boxes on-page but loses absolute position and aspect ratio.
    """
    from collections import defaultdict

    elements: list[dict[str, Any]] = document.get("elements") or []

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for el in elements:
        for box in el.get("bbox") or []:
            page_id = box.get("page_id")
            if page_id is None:
                continue
            try:
                by_page[int(page_id)].append({"element": el, "coord": box.get("coord")})
            except (TypeError, ValueError):
                continue

    # ai_parse page ids are 0-indexed; derive the base from document.pages
    # (rather than assuming) so 1-indexed responses would still map cleanly.
    declared_ids = [p.get("id") for p in document.get("pages") or [] if isinstance(p.get("id"), int)]
    id_base = min(declared_ids) if declared_ids else 0

    rendered_dims = _rendered_page_dims(source_file_path)

    layout_pages: list[ParseLayoutPageIR] = []
    for page_id in sorted(by_page.keys()):
        entries = by_page[page_id]
        max_x = 1.0
        max_y = 1.0
        for entry in entries:
            coord = entry["coord"] or []
            if len(coord) >= 4:
                max_x = max(max_x, float(coord[2]))
                max_y = max(max_y, float(coord[3]))

        page_index = page_id - id_base
        dims = rendered_dims[page_index] if rendered_dims is not None and 0 <= page_index < len(rendered_dims) else None
        if dims is not None:
            # If an element extends past the computed page edge the render-DPI
            # assumption is off for this file; widen the denominator so boxes
            # stay in [0,1] instead of drifting off-page.
            denom_x = max(dims[0], max_x)
            denom_y = max(dims[1], max_y)
            page_w, page_h = dims
        else:
            denom_x, denom_y = max_x, max_y
            page_w = page_h = _VIRTUAL_PAGE_DIM

        items: list[LayoutItemIR] = []
        for entry in entries:
            el = entry["element"]
            coord = entry["coord"] or []
            if len(coord) < 4:
                continue
            x1, y1, x2, y2 = (float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3]))
            w = max(x2 - x1, 0.0)
            h = max(y2 - y1, 0.0)

            canonical = DATABRICKS_LABEL_MAP.get((el.get("type") or "").lower())
            if canonical is None:
                continue

            seg = LayoutSegmentIR(
                x=x1 / denom_x,
                y=y1 / denom_y,
                w=w / denom_x,
                h=h / denom_y,
                confidence=float(el.get("confidence")) if el.get("confidence") is not None else None,
                label=canonical,
            )

            norm_label = canonical.strip().lower()
            if norm_label == "table":
                item_type = "table"
            elif norm_label == "picture":
                item_type = "image"
            else:
                item_type = "text"

            items.append(
                LayoutItemIR(
                    type=item_type,
                    value=el.get("content") or "",
                    bbox=seg,
                    layout_segments=[seg],
                )
            )

        layout_pages.append(
            ParseLayoutPageIR(
                page_number=max(page_index + 1, 1),
                width=page_w,
                height=page_h,
                items=items,
            )
        )

    return layout_pages
