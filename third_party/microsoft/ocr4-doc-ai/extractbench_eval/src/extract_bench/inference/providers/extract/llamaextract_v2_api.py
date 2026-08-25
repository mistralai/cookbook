"""Provider for LlamaExtract V2 API (/api/v2/extract).

Uses the new job-based V2 extract endpoint with tier-based configuration
(cost_effective / agentic) and optional parse_tier control.

This is distinct from the existing llamaextract provider which uses the
V1 stateless extraction API (/api/v1/extraction/run).
"""

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.cancellation import CancellableClientRegistry
from extract_bench.inference.providers.extract.citations import extract_llamaextract_field_citations
from extract_bench.inference.providers.extract.llamaextract_pricing import (
    apply_pricing_fields,
    local_page_count,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

logger = logging.getLogger(__name__)

_PRODUCTION_BASE_URL = "https://api.cloud.llamaindex.ai"

_DEFAULT_TIMEOUT = 600
_POLL_INTERVAL = 3
_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}

# Pipeline config keys handled by this provider (not forwarded to extract config)
_PROVIDER_ONLY_PARAMS = {
    "api_key",
    "timeout",
    "invalidate_cache",
    "environment",
    "parse_config",
}


def _is_extract_product_type(value: Any) -> bool:
    extract_type = getattr(ProductType, "EXTRACT", None)
    if extract_type is not None and value == extract_type:
        return True
    return bool(getattr(value, "value", value) == "extract")


def _extract_output_cls() -> type[Any]:
    from extract_bench.schemas.extract_output import ExtractOutput

    return ExtractOutput


def _parse_config_needs_saved_config_flow(parse_config: dict[str, Any]) -> bool:
    """Whether ``parse_config`` requires the FILE_ID + parse_config_id flow.

    The extract service only propagates ``granular_bboxes`` onto engine
    params on the FILE_ID branch. The PARSE_JOB_ID branch - what
    ``_run_parse_first`` produces - does NOT propagate ``granular_bboxes``,
    so any pipeline asking for granular bboxes must instead mint a saved
    parse config and pass its id to extract via
    ``configuration.parse_config_id``.

    Detected by looking for ``output_options.granular_bboxes``. Other parse
    configs continue to use the default pre-parse flow, which captures
    parse latency and ``parse_job_id`` separately for evaluation.
    """
    output_options = parse_config.get("output_options")
    if not isinstance(output_options, dict):
        return False
    return bool(output_options.get("granular_bboxes"))


@register_provider("llamaextract_v2")
class LlamaExtractV2Provider(Provider):
    """Provider for the V2 Extract API (/api/v2/extract).

    Pipeline config keys:
        tier:           "cost_effective" | "agentic"  (default: cost_effective)
        parse_tier:     "fast" | "cost_effective" | "agentic"  (optional)
        parse_config:   LlamaParse config dict (V2 nested shape: tier, version,
                        output_options, ...). Routing into the V2 extract API
                        depends on the contents:
                          - With ``output_options.granular_bboxes``: minted as
                            a parse_v2 ProductConfiguration, extract receives
                            ``parse_config_id`` and ``file_input=<file_id>``
                            (FILE_ID flow; matcher gate opens).
                          - Otherwise: parse runs first via LlamaParseProvider
                            and extract receives ``file_input=<parse_job_id>``
                            (PARSE_JOB_ID flow; preserves separate parse
                            latency capture).
        api_key:        str   (optional, defaults to env var)

    Environment:
        LLAMA_CLOUD_API_KEY   required unless ``api_key`` is passed in config
        LLAMA_CLOUD_BASE_URL  optional API base URL override
        LLAMA_CLOUD_PROJECT_ID  optional project scoping
    """

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        super().__init__(provider_name, base_config)

        api_key = self.base_config.get("api_key") or os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "LLAMA_CLOUD_API_KEY is required. Set the environment variable or pass api_key in config."
            )
        self._api_key: str = api_key
        self._base_url: str = os.getenv("LLAMA_CLOUD_BASE_URL", _PRODUCTION_BASE_URL).rstrip("/")

        self._project_id: str = os.getenv("LLAMA_CLOUD_PROJECT_ID", "")
        self._timeout: float = float(self.base_config.get("timeout", _DEFAULT_TIMEOUT))

        # Track the per-request httpx.Client so cancel(example_id) can close
        # it from the runner's timeout path. Closing the client aborts any
        # in-flight upload / poll, letting the worker thread unwind cleanly
        # before the retry attempt is submitted (otherwise the previous
        # request would keep running server-side while a duplicate was
        # already in flight).
        self._inflight = CancellableClientRegistry(provider_name=provider_name)

        # When parse_config is set we delegate the parse pass to a fresh
        # ``LlamaParseProvider``; track it per example_id so cancel can
        # forward to it during that pass (and close its SDK client).
        self._inflight_parse_providers: dict[str, Any] = {}
        self._parse_provider_lock = threading.Lock()

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if not _is_extract_product_type(request.product_type):
            raise ProviderPermanentError(f"LlamaExtractV2Provider only supports EXTRACT, got {request.product_type}")
        if not request.schema_override:
            raise ProviderPermanentError("schema_override is required for EXTRACT. Provide a JSON schema.")

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        started_at = datetime.now()

        try:
            raw_output = self._run_v2_extract(
                pipeline=pipeline,
                data_schema=request.schema_override,
                file_path=file_path,
                example_id=request.example_id,
            )
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
        except (ProviderPermanentError, ProviderRateLimitError, ProviderTransientError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if not _is_extract_product_type(raw_result.product_type):
            raise ProviderPermanentError(f"LlamaExtractV2Provider only supports EXTRACT, got {raw_result.product_type}")

        raw_data = raw_result.raw_output.get("data")
        job_id = raw_result.raw_output.get("job_id")

        if raw_data is None:
            logger.warning(
                "V2 extract returned null data for %s (job_id=%s)",
                raw_result.request.example_id,
                job_id,
            )

        # Price the job before the result is handed on. Doing it here rather
        # than at request time means re-normalizing a stored result re-prices it
        # from the same inputs, with no second inference run.
        apply_pricing_fields(
            raw_result.raw_output,
            raw_result.pipeline.config or {},
            data_schema=raw_result.request.schema_override,
            fallback_page_count=local_page_count(raw_result.request.source_file_path),
        )

        extracted_data = _extract_data_from_result(raw_data)
        output = _extract_output_cls()(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data if extracted_data is not None else {},
            field_citations=extract_llamaextract_field_citations(
                raw_result.raw_output.get("extract_metadata"),
                source="llamaextract_v2",
            ),
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

    def cancel(self, example_id: str) -> bool:
        """Abort the in-flight V2 extract request for ``example_id``.

        We try the parse-first inner provider first (it may be the active
        step), then close the V2 extract httpx.Client. Either is sufficient
        on its own; we attempt both because the timeout could fire during
        either phase. Returns True if at least one cancel target existed.
        """
        cancelled_any = False
        with self._parse_provider_lock:
            parse_provider = self._inflight_parse_providers.pop(example_id, None)
        if parse_provider is not None:
            try:
                cancel = getattr(parse_provider, "cancel", None)
                if callable(cancel) and cancel(example_id):
                    cancelled_any = True
            except Exception as exc:  # noqa: BLE001 - cancel must not raise
                logger.debug("inner llamaparse cancel raised: %s", exc)
        if self._inflight.cancel(example_id):
            cancelled_any = True
        return cancelled_any

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_v2_extract(
        self,
        pipeline: PipelineSpec,
        data_schema: dict[str, Any],
        file_path: Path,
        example_id: str,
    ) -> dict[str, Any]:
        """Upload file, create V2 extract job, poll to completion."""
        config = pipeline.config
        extract_configuration = self._build_extract_configuration(config, data_schema)
        parse_config = config.get("parse_config")
        if parse_config is not None and not isinstance(parse_config, dict):
            raise ProviderPermanentError("parse_config must be a JSON object when provided")

        # Build the httpx.Client outside the ``with`` block so we can register
        # it for cancellation and then close it deterministically in finally.
        # Using the manual try/finally keeps the close semantics identical to
        # ``with httpx.Client(...)`` while letting cancel() reach the handle.
        client = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )
        self._inflight.register(example_id, client)
        try:
            params: dict[str, str] = {}
            if self._project_id:
                params["project_id"] = self._project_id

            parse_job_id: str | None = None
            parse_config_id: str | None = None
            if parse_config is not None and _parse_config_needs_saved_config_flow(parse_config):
                # FILE_ID + parse_config_id flow. Mint a parse config server-side
                # so the workflow can propagate granular_bboxes onto engine params
                # and the citation matcher gate opens.
                parse_config_id = self._create_saved_parse_config(client, parse_config, params, example_id=example_id)
                extract_configuration["parse_config_id"] = parse_config_id
                file_input = self._upload_file(client, file_path)
            elif parse_config is not None:
                # Legacy PARSE_JOB_ID flow: run parse first, hand the resulting
                # parse_job_id to extract. Preserves separate parse latency
                # capture and parse_job_id for downstream evaluation.
                parse_job_id = self._run_parse_first(
                    pipeline,
                    file_path,
                    parse_config,
                    example_id=example_id,
                )
                file_input = parse_job_id
            else:
                file_input = self._upload_file(client, file_path)

            body: dict[str, Any] = {
                "file_input": file_input,
                "configuration": extract_configuration,
            }

            # 3. Create job
            logger.info(
                "Creating V2 extract job: tier=%s, parse_tier=%s, parse_route=%s",
                extract_configuration.get("tier"),
                extract_configuration.get("parse_tier"),
                "saved_config" if parse_config_id else ("pre_parse" if parse_job_id else "none"),
            )

            resp = client.post("/api/v2/extract", params=params, json=body)
            resp.raise_for_status()
            job = resp.json()
            job_id = job["id"]
            logger.info("V2 extract job created: %s", job_id)

            # 4. Poll
            result = self._poll_job(client, job_id, params)
            if parse_job_id is not None:
                result["parse_job_id"] = parse_job_id
            if parse_config_id is not None:
                result["parse_config_id"] = parse_config_id
            return result
        finally:
            self._inflight.unregister(example_id, client)
            try:
                client.close()
            except Exception:  # noqa: BLE001 - close errors are best-effort
                # If cancel() already closed the client mid-request, the
                # second close raises httpx errors; these are not actionable.
                pass

    def _build_extract_configuration(
        self,
        config: dict[str, Any],
        data_schema: dict[str, Any],
    ) -> dict[str, Any]:
        configuration = {key: value for key, value in config.items() if key not in _PROVIDER_ONLY_PARAMS}
        configuration.setdefault("tier", "cost_effective")
        configuration["data_schema"] = data_schema
        return configuration

    def _run_parse_first(
        self,
        pipeline: PipelineSpec,
        file_path: Path,
        parse_config: dict[str, Any],
        *,
        example_id: str,
    ) -> str:
        parse_provider_config = dict(parse_config)
        for key in ("api_key",):
            if key in self.base_config and key not in parse_provider_config:
                parse_provider_config[key] = self.base_config[key]

        parse_pipeline = PipelineSpec(
            pipeline_name=f"{pipeline.pipeline_name}__parse",
            provider_name="llamaparse",
            product_type=ProductType.PARSE,
            config=parse_provider_config,
        )
        parse_request = InferenceRequest(
            example_id=example_id,
            source_file_path=str(file_path),
            product_type=ProductType.PARSE,
        )
        from extract_bench.inference.providers.parse.llamaparse import LlamaParseProvider

        # Hold the inner parse provider for the duration of the parse step so
        # cancel(example_id) can forward to it. Without this reference the
        # provider would be GC'd as a temporary and an external cancel would
        # have nothing to forward to.
        parse_provider = LlamaParseProvider(
            provider_name="llamaparse",
            base_config=parse_provider_config,
        )
        with self._parse_provider_lock:
            self._inflight_parse_providers[example_id] = parse_provider
        try:
            raw_parse_result = parse_provider.run_inference(parse_pipeline, parse_request)
        finally:
            with self._parse_provider_lock:
                # Only clear if it's still ours; cancel() may have popped it.
                if self._inflight_parse_providers.get(example_id) is parse_provider:
                    self._inflight_parse_providers.pop(example_id, None)
        parse_job_id = raw_parse_result.raw_output.get("job_id")
        if not isinstance(parse_job_id, str) or not parse_job_id:
            raise ProviderPermanentError("LlamaParse did not return a parse job id")
        return parse_job_id

    def _create_saved_parse_config(
        self,
        client: httpx.Client,
        parse_config: dict[str, Any],
        params: dict[str, str],
        *,
        example_id: str,
    ) -> str:
        """Mint a parse_v2 product configuration and return its id.

        Posts the pipeline-level ``parse_config`` dict to
        ``/api/v1/beta/configurations`` as a parse_v2 ProductConfiguration.
        The resulting ``parse_config_id`` is then passed to extract via
        ``configuration.parse_config_id``, which routes the workflow through
        the FILE_ID branch and triggers ``granular_bboxes`` propagation
        (and the citation matcher gate, when applicable).

        Strips provider-only keys (``invalidate_cache``,
        ``api_key``, etc.) and the V1-flat ``disable_cache`` key that the
        V2 nested schema rejects. Caller is responsible for providing
        ``output_options`` (and any other V2 nested fields) directly in
        ``parse_config``.
        """
        v2_parameters: dict[str, Any] = {
            k: v for k, v in parse_config.items() if k not in _PROVIDER_ONLY_PARAMS and k != "disable_cache"
        }
        v2_parameters["product_type"] = "parse_v2"
        v2_parameters.setdefault("version", "latest")

        body = {
            "name": f"bench-{self.provider_name}-{example_id}-{int(time.time())}",
            "parameters": v2_parameters,
        }
        resp = client.post("/api/v1/beta/configurations", params=params, json=body)
        resp.raise_for_status()
        config_id: str = resp.json()["id"]
        logger.info("Minted parse_v2 config %s for example %s", config_id, example_id)
        return config_id

    def _upload_file(self, client: httpx.Client, file_path: Path) -> str:
        """Upload a file and return its ID."""
        mime = _guess_mime(file_path)
        params: dict[str, str] = {}
        if self._project_id:
            params["project_id"] = self._project_id

        # Matches llama_cloud SDK's LlamaCloud.files.create: POST /api/v1/beta/files
        # with required multipart form field `purpose`. FileCreateParams marks
        # `purpose: Required[str]`; for extract flows the valid value is "extract".
        resp = client.post(
            "/api/v1/beta/files",
            params=params,
            files={"file": (file_path.name, file_path.read_bytes(), mime)},
            data={"purpose": "extract"},
        )
        resp.raise_for_status()
        file_id: str = resp.json()["id"]
        logger.info("File uploaded: %s -> %s", file_path.name, file_id)
        return file_id

    def _poll_job(self, client: httpx.Client, job_id: str, params: dict[str, str]) -> dict[str, Any]:
        """Poll V2 extract job until terminal state.

        Persist a compact status-transition history into the raw result so
        long or stuck jobs can be diagnosed from benchmark artifacts.
        """
        start = time.monotonic()
        poll_started_at = datetime.now().isoformat()

        # Request both expandable blocks on every poll. The V2 extract API
        # strips them from the GET response unless the caller opts in.
        #
        # ``extract_metadata``: without it citations have no ``bounding_boxes``
        # and bbox-recall metrics evaluate to 0 even when the engine populated
        # citations server-side.
        # ``metadata``: carries ``usage.num_pages_extracted`` and
        # ``usage.num_pages_billed``. It comes back null unless expanded, and
        # without it there is no billable page count, so cost falls back to a
        # schema estimate (or to nothing at all on non-Agentic-Plus tiers).
        poll_params: dict[str, Any] = {**params, "expand": ["metadata", "extract_metadata"]}

        poll_history: list[dict[str, Any]] = []
        last_recorded_status: str | None = None

        while True:
            elapsed = time.monotonic() - start
            if elapsed > self._timeout:
                raise ProviderTransientError(f"V2 extract job {job_id} did not complete within {self._timeout}s")

            resp = client.get(f"/api/v2/extract/{job_id}", params=poll_params)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "UNKNOWN")

            if status != last_recorded_status:
                poll_history.append(
                    {
                        "wall_clock": datetime.now().isoformat(),
                        "elapsed_s": round(elapsed, 2),
                        "status": status,
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                    }
                )
                last_recorded_status = status

            if status in _TERMINAL_STATUSES:
                if poll_history[-1].get("status") != status or len(poll_history) == 1:
                    poll_history.append(
                        {
                            "wall_clock": datetime.now().isoformat(),
                            "elapsed_s": round(elapsed, 2),
                            "status": status,
                            "created_at": data.get("created_at"),
                            "updated_at": data.get("updated_at"),
                        }
                    )

                if status == "FAILED":
                    error_msg = data.get("error_message", "Unknown error")
                    raise ProviderPermanentError(f"V2 extract job {job_id} failed: {error_msg}")

                if status == "CANCELLED":
                    raise ProviderPermanentError(f"V2 extract job {job_id} was cancelled")

                extract_metadata = data.get("extract_metadata") or {}
                spawned_parse_job_id = (
                    extract_metadata.get("parse_job_id") if isinstance(extract_metadata, dict) else None
                )

                return {
                    "data": data.get("extract_result"),
                    "job_id": job_id,
                    "extract_metadata": extract_metadata,
                    # Carries ``usage.num_pages_billed`` -- the effective pages
                    # the extract leg is billed on, which a large schema pushes
                    # above the page count. Pricing prefers it over any local
                    # estimate, so it has to survive into the raw result.
                    "metadata": data.get("metadata"),
                    "status": status,
                    "poll_history": poll_history,
                    "poll_started_at": poll_started_at,
                    "poll_completed_at": datetime.now().isoformat(),
                    "total_elapsed_s": round(elapsed, 2),
                    "spawned_parse_job_id": spawned_parse_job_id,
                }

            time.sleep(_POLL_INTERVAL)


def _guess_mime(path: Path) -> str:
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".html": "text/html",
        ".txt": "text/plain",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }.get(path.suffix.lower(), "application/octet-stream")


def _extract_data_from_result(result_payload: Any) -> Any:
    """Normalize known V2 result envelopes while preserving raw semantic shape."""
    if isinstance(result_payload, dict):
        document_result = result_payload.get("document_result")
        if isinstance(document_result, dict):
            return document_result

        page_results = result_payload.get("page_results")
        if isinstance(page_results, list):
            return page_results

        table_results = result_payload.get("table_results")
        if isinstance(table_results, list):
            return table_results

        return result_payload

    if isinstance(result_payload, list):
        return result_payload

    return None
