"""Provider for the OpenInnovation Parser public API (``oi-parser``).

The provider calls a single **synchronous** ``POST {base_url}/v1/extract`` with
``X-API-Key`` auth and the document as a multipart ``file`` (one doc per call, N
parallel — the runner's ``ThreadPoolExecutor``). The API is synchronous: the full
result is in the 200 body, which is an already-normalized ``InferenceResult`` JSON
(with ``output.markdown`` / ``output.layout_pages`` / ``output.pages``, plus
top-level ``status`` / ``metrics`` that the evaluator ignores). ``normalize()``
therefore just re-hydrates the body's ``output`` into a :class:`ParseOutput`,
forcing ``example_id`` to the request's id (the eval matches on it) and
``pipeline_name`` to the registered pipeline name.

Config keys
-----------
api_key : str
    oi-parser API key. Falls back to the ``OI_PARSER_API_KEY`` env var. Required.
base_url : str
    Base URL of the public API. Falls back to ``OI_PARSER_BASE_URL``, else the
    default below.
timeout : int
    HTTP request timeout in seconds (default 900, matching the server's extract cap).

Recommended ``--max_concurrent``: **20** — the server enforces a per-key concurrency
cap (20 on the validated key); exceeding it wastes retries and can drop docs.
"""

import html as _html
import os
import re
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
from extract_bench.schemas.parse_output import ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

_DEFAULT_BASE_URL = "https://parse.prod.openinnovation.ai"

# ParseBench's table metric (GriTS/TEDS) only scores HTML <table>; oi-parser markdown
# may render tables as GitHub-flavored pipe tables (correct data, wrong format for
# GriTS). normalize() converts pipe tables → HTML for scoring — same as the reference
# mistral_ocr provider. Only valid pipe-table blocks (header + separator + >=1 row) are
# converted; all other content (incl. HTML-native tables) is left byte-identical.
_SEP_RE = re.compile(r"^\|?[\s:|-]+\|?$")


def _is_pipe_row(line: str) -> bool:
    s = line.strip()
    return "|" in s and not s.startswith("<!--")


def _cells(row: str) -> list[str]:
    r = row.strip()
    if r.startswith("|"):
        r = r[1:]
    if r.endswith("|"):
        r = r[:-1]
    return [c.strip() for c in r.split("|")]


def _block_to_html(block: list[str]) -> str | None:
    sep_idx = next((i for i, ln in enumerate(block) if _SEP_RE.match(ln.strip())), None)
    if sep_idx is None or sep_idx == 0 or len(block) < 3:
        return None
    header = _cells(block[sep_idx - 1])
    data = [_cells(ln) for ln in block[sep_idx + 1 :] if ln.strip()]
    if not data:
        return None
    th = "".join(f"<th>{_html.escape(c)}</th>" for c in header)
    body = "".join("<tr>" + "".join(f"<td>{_html.escape(c)}</td>" for c in row) + "</tr>" for row in data)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _pipe_tables_to_html(md: str) -> str:
    """Replace markdown pipe tables with HTML <table>; leave everything else byte-identical."""
    if not md or "|" not in md:
        return md
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if _is_pipe_row(lines[i]):
            j = i
            block: list[str] = []
            while j < n and _is_pipe_row(lines[j]):
                block.append(lines[j])
                j += 1
            html_table = _block_to_html(block)
            if html_table is not None:
                out.append(html_table)
            else:
                out.extend(block)
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


@register_provider("oi_parser")
class OIParserProvider(Provider):
    """Provider for the oi-parser public ``/v1/extract`` API."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        api_key = self.base_config.get("api_key") or os.getenv("OI_PARSER_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "oi-parser API key is required. Set OI_PARSER_API_KEY environment "
                "variable or pass api_key in base_config."
            )
        self._api_key: str = str(api_key)
        self._base_url: str = (
            self.base_config.get("base_url") or os.getenv("OI_PARSER_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._timeout: int = int(self.base_config.get("timeout", 900))

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"OIParserProvider only supports PARSE, got {request.product_type}")

        path = Path(request.source_file_path)
        if not path.exists():
            raise ProviderPermanentError(f"File not found: {path}")

        started_at = datetime.now()
        try:
            with open(path, "rb") as fh:
                resp = requests.post(
                    f"{self._base_url}/v1/extract",
                    headers={"X-API-Key": self._api_key},
                    files={"file": (path.name, fh, "application/octet-stream")},
                    data={"example_id": request.example_id},
                    timeout=self._timeout,
                )
        except requests.exceptions.RequestException as e:
            # Any transport-level failure — timeout, connection reset ("peer closed
            # connection" under over-cap admission), or a ChunkedEncodingError
            # ("Response ended prematurely") when the body stream is cut — is
            # transient, so the runner retries it. A 4xx/5xx does NOT raise here
            # (requests returns a Response); those are classified by status below.
            raise ProviderTransientError(f"oi-parser request error: {type(e).__name__}: {e}") from e

        sc = resp.status_code
        if sc == 401:
            raise ProviderConfigError(f"oi-parser unauthorized (401): {resp.text[:300]}")
        if sc == 429:
            raise ProviderRateLimitError(f"oi-parser rate limit (429): {resp.text[:300]}")
        if sc >= 500:
            raise ProviderTransientError(f"oi-parser server error ({sc}): {resp.text[:300]}")
        if sc >= 400:
            raise ProviderPermanentError(f"oi-parser client error ({sc}): {resp.text[:300]}")

        try:
            raw_output: dict[str, Any] = resp.json()
        except (ValueError, requests.exceptions.RequestException) as e:
            # Truncated/invalid body (e.g. ChunkedEncodingError surfacing at read
            # time, or non-JSON) — transient; let the runner retry.
            raise ProviderTransientError(f"oi-parser bad/truncated response body: {type(e).__name__}: {e}") from e
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
            raise ProviderPermanentError(f"OIParserProvider only supports PARSE, got {raw_result.product_type}")

        body_output = raw_result.raw_output.get("output")
        if not isinstance(body_output, dict):
            raise ProviderPermanentError("oi-parser response missing an 'output' object")

        # The API returns an already-normalized, ParseOutput-shaped body. Re-hydrate it,
        # forcing two fields:
        #   example_id    -> the ParseBench request id (REQUIRED; the eval matches on it)
        #   pipeline_name -> the registered pipeline name ("oi_parser"); the body carries
        #                    the server-side "oi-parser" (hyphen), which is NOT registered.
        # The body's markdown / layout_pages / pages are preserved verbatim.
        out_dict = dict(body_output)
        out_dict["example_id"] = raw_result.request.example_id
        out_dict["pipeline_name"] = raw_result.pipeline_name
        if out_dict.get("markdown"):
            out_dict["markdown"] = _pipe_tables_to_html(out_dict["markdown"])
        output = ParseOutput.model_validate(out_dict)

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
