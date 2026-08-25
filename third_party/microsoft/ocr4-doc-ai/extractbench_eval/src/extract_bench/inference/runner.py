"""Inference runner for batch processing PDFs with concurrency control."""

import asyncio
import concurrent.futures
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from extract_bench.inference.providers.base import (
    Provider,
    ProviderError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.schema import TestCase

# Retry configuration for transient / rate-limit errors
MAX_RETRIES = 5
INITIAL_BACKOFF_S = 2.0  # seconds
BACKOFF_MULTIPLIER = 2.0  # exponential backoff factor

# Per-file timeout and retry configuration
DEFAULT_PER_FILE_TIMEOUT_S = 1800.0  # 30 minutes per file
DEFAULT_TIMEOUT_RETRIES = 2  # retry up to 2 times on timeout

LOCAL_ARTIFACT_PROVIDER_NAMES: set[str] = set()


@dataclass
class RunSummary:
    """Summary statistics for an inference run."""

    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_latency_ms: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.successful == 0:
            return 0.0
        return self.total_latency_ms / self.successful

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total == 0:
            return 0.0
        return (self.successful / self.total) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary for JSON serialization."""
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "success_rate": round(self.success_rate, 2),
            "errors": self.errors,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class JobStatus:
    """Status of a single job."""

    example_id: str
    pdf_path: Path
    status: str = "pending"  # pending, running, completed, failed, skipped
    latency_ms: int | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class InferenceRunner:
    """
    Runs inference on PDFs with concurrency control and saves structured results.

    Features:
    - Semaphore-based concurrency control
    - Saves both raw and normalized results as JSON
    - Skip logic for already-processed files
    - Rich terminal UI with live updates
    - Summary statistics
    - Error handling and tracking
    """

    def __init__(
        self,
        provider: Provider,
        pipeline: PipelineSpec,
        output_dir: Path,
        max_concurrent: int = 20,
        save_raw: bool = True,
        save_normalized: bool = True,
        force: bool = False,
        use_rich: bool = True,
        tags: list[str] | None = None,
        per_file_timeout: float | None = None,
        timeout_retries: int = DEFAULT_TIMEOUT_RETRIES,
    ):
        """
        Initialize the inference runner.

        :param provider: Provider instance for running inference
        :param pipeline: Pipeline specification
        :param output_dir: Directory to save results
        :param max_concurrent: Maximum concurrent inference requests
        :param save_raw: Whether to save RawInferenceResult JSON files
        :param save_normalized: Whether to save InferenceResult JSON files
        :param force: Force regeneration even if results already exist
        :param use_rich: Whether to use Rich for terminal UI (default: True)
        :param tags: Optional list of tags for this run (e.g., ['nightly', 'production'])
        :param per_file_timeout: Explicit run-level max seconds per file. None (default)
            falls back to the pipeline's per_file_timeout, then DEFAULT_PER_FILE_TIMEOUT_S.
        :param timeout_retries: Number of retries on per-file timeout (default: 2)
        """
        self.provider = provider
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.max_concurrent = max_concurrent
        self.save_raw = save_raw
        self.save_normalized = save_normalized
        self.force = force
        self.use_rich = use_rich
        self.tags = tags or []
        # Per-file timeout precedence (highest first):
        #   1. explicit run-level value (CLI --per_file_timeout, passed here non-None)
        #   2. per-pipeline override (PipelineSpec.per_file_timeout)
        #   3. the global default (DEFAULT_PER_FILE_TIMEOUT_S)
        if per_file_timeout is not None:
            self.per_file_timeout = per_file_timeout
        elif pipeline.per_file_timeout is not None:
            self.per_file_timeout = pipeline.per_file_timeout
        else:
            self.per_file_timeout = DEFAULT_PER_FILE_TIMEOUT_S
        self.timeout_retries = timeout_retries
        self.console = Console() if use_rich else None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Job tracking for Rich UI
        self.job_statuses: dict[str, JobStatus] = {}

        # Create a thread pool sized to match max_concurrent
        # The default asyncio thread pool is limited to min(32, os.cpu_count() + 4)
        # which can be as low as 6 on CI runners with 2 CPUs.
        # We create our own pool to ensure we can run max_concurrent tasks in parallel.
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="inference_worker"
        )

        # Track current summary for interrupt handling
        self._current_summary: RunSummary | None = None

    def shutdown(self) -> None:
        """Shutdown the thread pool. Call this when done with the runner.

        Uses cancel_futures=True to cancel any pending work items and
        wait=False to avoid blocking on threads stuck in network I/O
        (e.g., timed-out provider API calls that can't be interrupted).
        """
        self._thread_pool.shutdown(wait=False, cancel_futures=True)

    def get_current_summary(self) -> RunSummary | None:
        """Get the current run summary (useful for interrupt handling)."""
        return self._current_summary

    def save_partial_results(self) -> None:
        """Save partial results on interrupt. Call this when handling KeyboardInterrupt."""
        if self._current_summary is None:
            return

        self._current_summary.completed_at = datetime.now()

        # Save summary
        summary_path = self.output_dir / "_summary.json"
        summary_path.write_text(json.dumps(self._current_summary.to_dict(), indent=2))

        # Save errors if any
        if self._current_summary.errors:
            errors_path = self.output_dir / "_errors.json"
            errors_path.write_text(json.dumps(self._current_summary.errors, indent=2))

    def _get_result_paths(self, example_id: str) -> tuple[Path, Path]:
        """Get file paths for raw and normalized results."""
        raw_path = self.output_dir / f"{example_id}.raw.json"
        normalized_path = self.output_dir / f"{example_id}.result.json"
        return raw_path, normalized_path

    def _signal_cancel_and_cancel_future(
        self,
        example_id: str,
        future: concurrent.futures.Future[Any],
    ) -> None:
        """Signal provider cancellation and mark the Python future cancelled."""
        cancel_fn = getattr(self.provider, "cancel", None)
        if callable(cancel_fn):
            try:
                cancel_fn(example_id)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"  Warning: provider.cancel({example_id}) raised: {exc}")
        future.cancel()

    def _cancel_inflight_and_drain(
        self,
        example_id: str,
        future: concurrent.futures.Future[Any],
        *,
        drain_timeout_seconds: float = 5.0,
    ) -> None:
        """Best-effort timeout cancel for synchronous retry loops."""
        self._signal_cancel_and_cancel_future(example_id, future)
        try:
            future.result(timeout=drain_timeout_seconds)
        except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError, Exception):
            pass

    async def _cancel_inflight_and_drain_async(
        self,
        example_id: str,
        future: concurrent.futures.Future[Any],
        *,
        drain_timeout_seconds: float = 5.0,
    ) -> None:
        """Best-effort timeout cancel for async retry loops without blocking the event loop."""
        self._signal_cancel_and_cancel_future(example_id, future)
        try:
            await asyncio.wait_for(asyncio.wrap_future(future), timeout=drain_timeout_seconds)
        except (TimeoutError, concurrent.futures.CancelledError, asyncio.CancelledError, Exception):
            pass

    def _is_already_processed(self, example_id: str) -> bool:
        """Check if a file has already been processed."""
        if self.force:
            return False

        raw_path, normalized_path = self._get_result_paths(example_id)

        # Check if normalized result exists (primary check)
        if self.save_normalized and normalized_path.exists():
            try:
                # Verify it's valid JSON
                data = json.loads(normalized_path.read_text())
                # Check if it has required fields
                if "request" in data and "output" in data:
                    return True
            except (json.JSONDecodeError, KeyError):
                # Invalid file, should be regenerated
                return False

        # Check if raw result exists (if we only save raw)
        if self.save_raw and not self.save_normalized and raw_path.exists():
            try:
                data = json.loads(raw_path.read_text())
                if "request" in data and "raw_output" in data:
                    return True
            except (json.JSONDecodeError, KeyError):
                return False

        return False

    def _save_result(self, raw_result: RawInferenceResult | None, normalized_result: InferenceResult | None) -> None:
        """Save raw and/or normalized results to disk."""
        if raw_result is None and normalized_result is None:
            return

        example_id = (
            normalized_result.request.example_id if normalized_result else raw_result.request.example_id  # type: ignore[union-attr]
        )

        if self.save_raw and raw_result:
            raw_path, _ = self._get_result_paths(example_id)
            # Create parent directory if it doesn't exist (e.g., for group/test_id structure)
            raw_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if logs.jsonl lines are present in raw_output and save them separately
            if (
                hasattr(raw_result, "raw_output")
                and isinstance(raw_result.raw_output, dict)
                and "logs_jsonl_lines" in raw_result.raw_output
            ):
                # Extract base filename from raw_path to avoid path duplication
                base_name = raw_path.stem.removesuffix(".raw")
                logs_path = raw_path.parent / f"{base_name}.logs.jsonl"
                logs_lines = raw_result.raw_output["logs_jsonl_lines"]
                if isinstance(logs_lines, list):
                    with open(logs_path, "w") as f:
                        f.writelines(logs_lines)

                    # Remove logs from raw_output to avoid duplication in JSON
                    del raw_result.raw_output["logs_jsonl_lines"]

            # Save raw result (now without logs_jsonl_lines if they were extracted).
            # Note: parse job logs sidecars + token extraction happen earlier in
            # _fetch_parse_job_logs(), before normalize(), so that the resulting
            # token fields flow into the normalized InferenceResult.
            raw_path.write_text(raw_result.model_dump_json(indent=2))

        if self.save_normalized and normalized_result:
            _, normalized_path = self._get_result_paths(example_id)
            # Create parent directory if it doesn't exist (e.g., for group/test_id structure)
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            normalized_path.write_text(normalized_result.model_dump_json(indent=2))

    def _save_error_debug_payload(self, example_id: str, payload: dict[str, Any]) -> str | None:
        """Save provider-supplied debug payload for a failed example."""
        try:
            raw_path, _ = self._get_result_paths(example_id)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            base_name = raw_path.stem.removesuffix(".raw")
            debug_path = raw_path.parent / f"{base_name}.error.raw.json"
            debug_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            return str(debug_path.relative_to(self.output_dir).as_posix())
        except (TypeError, ValueError, OSError):
            return None

    def _fetch_parse_job_logs(self, raw_result: RawInferenceResult, example_id: str) -> None:
        """Download parse jobLogs sidecar and extract token usage before normalization.

        Best-effort: failures must never break the inference pipeline. Gated on
        save_raw because the sidecar lives next to the raw result file.
        """
        if not self.save_raw:
            return
        if not isinstance(raw_result.raw_output, dict):
            return
        try:
            raw_path, _ = self._get_result_paths(example_id)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_parse_job_log_artifacts(raw_result=raw_result, raw_path=raw_path)
        except Exception:
            # Never fail inference because optional parse logs retrieval failed.
            pass

    def _extract_job_logs_url(self, raw_output: dict[str, Any]) -> str | None:
        """Extract a job logs URL from raw provider payload."""
        direct_url = raw_output.get("job_logs_url")
        if isinstance(direct_url, str) and direct_url:
            return direct_url

        job_logs = raw_output.get("job_logs")
        if isinstance(job_logs, dict):
            nested_url = job_logs.get("url")
            if isinstance(nested_url, str) and nested_url:
                return nested_url

        return None

    @staticmethod
    def _extract_token_usage_from_log_entries(log_entries: list) -> dict[str, Any]:
        """Extract token usage from LLM_USAGE_TRACKER events in job log entries.

        Returns a structured dict with aggregate and per-model token counts,
        or empty dict if no usage events are found.
        """
        total_input = 0
        total_output = 0
        total_thinking = 0
        by_model: dict[str, dict[str, int]] = {}
        num_calls = 0

        for entry in log_entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "LLM_USAGE_TRACKER":
                continue

            content = entry.get("content", {})
            if not isinstance(content, dict):
                continue

            input_tok = content.get("inputTokens", 0) or 0
            output_tok = content.get("outputTokens", 0) or 0
            thinking_tok = content.get("thinkingTokens", 0) or 0
            model = content.get("model", "unknown")

            total_input += input_tok
            total_output += output_tok
            total_thinking += thinking_tok
            num_calls += 1

            if model not in by_model:
                by_model[model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thinking_tokens": 0,
                    "total_tokens": 0,
                    "calls": 0,
                }
            m = by_model[model]
            m["input_tokens"] += input_tok
            m["output_tokens"] += output_tok
            m["thinking_tokens"] += thinking_tok
            m["total_tokens"] += input_tok + output_tok + thinking_tok
            m["calls"] += 1

        if num_calls == 0:
            return {}

        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "thinking_tokens": total_thinking,
            "total_tokens": total_input + total_output + total_thinking,
            "num_llm_calls": num_calls,
            "by_model": by_model,
        }

    def _write_parse_job_log_artifacts(self, raw_result: RawInferenceResult, raw_path: Path) -> None:
        """Download and render parse job logs sidecars when available.

        Sidecar outputs:
          - `<example>.jobLogs.json`
        """
        if not isinstance(raw_result.raw_output, dict):
            return

        raw_output = raw_result.raw_output
        job_logs_url = self._extract_job_logs_url(raw_output)
        if not job_logs_url:
            return

        base_name = raw_path.stem.removesuffix(".raw")
        job_logs_path = raw_path.parent / f"{base_name}.jobLogs.json"

        # Download job logs JSON from presigned URL.
        try:
            with urllib_request.urlopen(job_logs_url, timeout=60) as response:
                content = response.read().decode("utf-8")
            # Ensure it is valid JSON and write pretty output.
            parsed = json.loads(content)
            job_logs_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False))
            raw_output["job_logs_local_path"] = str(job_logs_path.relative_to(self.output_dir).as_posix())

            # Extract token usage from the downloaded log entries
            if isinstance(parsed, list):
                token_usage = self._extract_token_usage_from_log_entries(parsed)
                if token_usage:
                    raw_output.setdefault("token_usage", token_usage)
                    # Surface top-level fields for consistency with _attach_usage_metadata()
                    for key in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens"):
                        if key in token_usage:
                            raw_output.setdefault(key, token_usage[key])
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            raw_output["job_logs_download_error"] = str(exc)

    def _prepare_source_file_for_provider(self, example_id: str, source_file_path: Path) -> Path:
        """
        Prepare source file path before provider invocation.

        Local worker provider writes sidecars next to source file, so we stage a symlink/copy
        under output_dir to co-locate all artifacts with .raw/.result files.
        """
        if self.pipeline.provider_name not in LOCAL_ARTIFACT_PROVIDER_NAMES:
            return source_file_path

        staged_suffix = source_file_path.suffix if source_file_path.suffix else ".pdf"
        staged_path = self.output_dir / f"{example_id}{staged_suffix}"
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        source_resolved = source_file_path.resolve()

        # Reuse existing staged file if it already points to the same source.
        if staged_path.is_symlink():
            try:
                if staged_path.resolve() == source_resolved:
                    return staged_path
            except OSError:
                pass
            staged_path.unlink(missing_ok=True)
        elif staged_path.exists():
            if self.force:
                staged_path.unlink(missing_ok=True)
            else:
                return staged_path

        try:
            staged_path.symlink_to(source_resolved)
        except OSError:
            shutil.copy2(source_resolved, staged_path)

        return staged_path

    def _process_document(
        self, pdf_path: Path, example_id: str, product_type: ProductType
    ) -> tuple[RawInferenceResult | None, InferenceResult | None, str | None]:
        """
        Process a single document (synchronous).

        :return: Tuple of (raw_result, normalized_result, error_message)
        """
        raw_result: RawInferenceResult | None = None

        # Update job status
        if self.use_rich and example_id in self.job_statuses:
            self.job_statuses[example_id].status = "running"
            self.job_statuses[example_id].started_at = datetime.now()

        try:
            prepared_source = self._prepare_source_file_for_provider(example_id, pdf_path)

            # Create inference request
            request = InferenceRequest(
                example_id=example_id,
                source_file_path=str(prepared_source),
                product_type=product_type,
            )

            # Run inference with retry for transient / rate-limit errors
            last_error: Exception | None = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    raw_result = self.provider.run_inference(self.pipeline, request)
                    break
                except (ProviderTransientError, ProviderRateLimitError) as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        backoff = INITIAL_BACKOFF_S * (BACKOFF_MULTIPLIER**attempt)
                        time.sleep(backoff)
                    else:
                        raise
            else:
                raise last_error  # type: ignore[misc]

            # Fetch parse jobLogs + extract token usage BEFORE normalize, so that
            # token fields land in the InferenceResult that evaluation reads.
            self._fetch_parse_job_logs(raw_result, example_id)

            # Normalize (phase 2: convert to structured output)
            normalized_result = self.provider.normalize(raw_result)

            # Save results
            self._save_result(raw_result, normalized_result)

            # Update job status
            if self.use_rich and example_id in self.job_statuses:
                self.job_statuses[example_id].status = "completed"
                self.job_statuses[example_id].completed_at = datetime.now()
                if normalized_result:
                    self.job_statuses[example_id].latency_ms = normalized_result.latency_in_ms
                elif raw_result:
                    self.job_statuses[example_id].latency_ms = raw_result.latency_in_ms

            return raw_result, normalized_result, None

        except ProviderError as e:
            import traceback

            error_msg = f"Provider error: {str(e)}"
            if raw_result is not None:
                self._save_result(raw_result, None)
            error_traceback = traceback.format_exc()
            if self.use_rich and example_id in self.job_statuses:
                self.job_statuses[example_id].status = "failed"
                self.job_statuses[example_id].error = error_msg
                self.job_statuses[example_id].completed_at = datetime.now()
            return None, None, (error_msg, error_traceback, type(e).__name__)  # type: ignore[return-value]
        except Exception as e:
            import traceback

            error_msg = f"Unexpected error: {str(e)}"
            if raw_result is not None:
                self._save_result(raw_result, None)
            error_traceback = traceback.format_exc()
            if self.use_rich and example_id in self.job_statuses:
                self.job_statuses[example_id].status = "failed"
                self.job_statuses[example_id].error = error_msg
                self.job_statuses[example_id].completed_at = datetime.now()
            return None, None, (error_msg, error_traceback, type(e).__name__)  # type: ignore[return-value]

    def _process_test_case(
        self, test_case: TestCase, product_type: ProductType
    ) -> tuple[RawInferenceResult | None, InferenceResult | None, str | None]:
        """
        Process a single test case (synchronous).

        :param test_case: Test case with file, schema, and config
        :param product_type: Product type (PARSE or EXTRACT)
        :return: Tuple of (raw_result, normalized_result, error_message)
        """
        # Update job status
        if self.use_rich and test_case.test_id in self.job_statuses:
            self.job_statuses[test_case.test_id].status = "running"
            self.job_statuses[test_case.test_id].started_at = datetime.now()

        raw_result: RawInferenceResult | None = None

        try:
            # Create inference request
            prepared_source = self._prepare_source_file_for_provider(
                test_case.test_id,
                test_case.file_path,
            )

            request = InferenceRequest(
                example_id=test_case.test_id,
                source_file_path=str(prepared_source),
                product_type=product_type,
                schema_override=getattr(test_case, "data_schema", None),
                config_override=getattr(test_case, "config", None),
            )

            # Run inference with retry for transient / rate-limit errors
            last_error: Exception | None = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    raw_result = self.provider.run_inference(self.pipeline, request)
                    break
                except (ProviderTransientError, ProviderRateLimitError) as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        backoff = INITIAL_BACKOFF_S * (BACKOFF_MULTIPLIER**attempt)
                        time.sleep(backoff)
                    else:
                        raise
            else:
                raise last_error  # type: ignore[misc]

            # Fetch parse jobLogs + extract token usage BEFORE normalize, so that
            # token fields land in the InferenceResult that evaluation reads.
            self._fetch_parse_job_logs(raw_result, test_case.test_id)

            # Normalize (phase 2: convert to structured output)
            normalized_result = self.provider.normalize(raw_result)

            # Save results
            self._save_result(raw_result, normalized_result)

            # Update job status
            if self.use_rich and test_case.test_id in self.job_statuses:
                self.job_statuses[test_case.test_id].status = "completed"
                self.job_statuses[test_case.test_id].completed_at = datetime.now()
                if normalized_result:
                    self.job_statuses[test_case.test_id].latency_ms = normalized_result.latency_in_ms
                elif raw_result:
                    self.job_statuses[test_case.test_id].latency_ms = raw_result.latency_in_ms

            return raw_result, normalized_result, None

        except ProviderError as e:
            import traceback

            error_msg = f"Provider error: {str(e)}"
            if raw_result is not None:
                self._save_result(raw_result, None)
            debug_payload_path = None
            debug_payload = getattr(e, "debug_payload", None)
            if isinstance(debug_payload, dict):
                debug_payload_path = self._save_error_debug_payload(test_case.test_id, debug_payload)
                if debug_payload_path:
                    error_msg += f" [debug payload: {debug_payload_path}]"
            error_traceback = traceback.format_exc()
            if self.use_rich and test_case.test_id in self.job_statuses:
                self.job_statuses[test_case.test_id].status = "failed"
                self.job_statuses[test_case.test_id].error = error_msg
                self.job_statuses[test_case.test_id].completed_at = datetime.now()
            return None, None, (error_msg, error_traceback, type(e).__name__)  # type: ignore[return-value]
        except Exception as e:
            import traceback

            error_msg = f"Unexpected error: {str(e)}"
            if raw_result is not None:
                self._save_result(raw_result, None)
            error_traceback = traceback.format_exc()
            if self.use_rich and test_case.test_id in self.job_statuses:
                self.job_statuses[test_case.test_id].status = "failed"
                self.job_statuses[test_case.test_id].error = error_msg
                self.job_statuses[test_case.test_id].completed_at = datetime.now()
            return None, None, (error_msg, error_traceback, type(e).__name__)  # type: ignore[return-value]

    def _run_files_sync(
        self,
        pdf_files: list[Path],
        product_type: ProductType,
        example_id_fn: Callable[[Path], str],
    ) -> RunSummary:
        """
        Process PDF files synchronously when max_concurrent=1.

        :param pdf_files: List of PDF file paths
        :param product_type: Product type (PARSE or EXTRACT)
        :param example_id_fn: Function to generate example_id from PDF path
        :return: Summary of the run
        """
        self._current_summary = summary = RunSummary()

        # Initialize job statuses for Rich UI
        if self.use_rich:
            for pdf_path in pdf_files:
                example_id = example_id_fn(pdf_path)
                self.job_statuses[example_id] = JobStatus(example_id=example_id, pdf_path=pdf_path, status="pending")

        # Create progress bar if using Rich UI
        if self.use_rich and self.console:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(
                    bar_width=None,
                    style="bright_blue",
                    complete_style="green",
                    finished_style="green",
                ),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("[cyan]{task.completed}/{task.total}"),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=self.console,
                expand=True,
            )
            task_id = progress.add_task(f"Processing {self.pipeline.pipeline_name}", total=len(pdf_files))
        else:
            progress = None
            task_id = None

        # Process each PDF file synchronously
        for pdf_path in pdf_files:
            example_id = example_id_fn(pdf_path)

            # Check if already processed
            if self._is_already_processed(example_id):
                summary.skipped += 1
                if self.use_rich and example_id in self.job_statuses:
                    self.job_statuses[example_id].status = "skipped"
                if progress and task_id is not None:
                    progress.update(task_id, advance=1, refresh=True)
                continue

            # Process document directly (synchronous)
            raw_result, normalized_result, error_info = self._process_document(pdf_path, example_id, product_type)

            summary.total += 1

            if error_info:
                summary.failed += 1
                # Handle both old format (string) and new format (tuple)
                if isinstance(error_info, tuple):
                    error_msg, error_traceback, error_type = error_info
                    summary.errors.append(
                        {
                            "example_id": example_id,
                            "file_path": str(pdf_path),
                            "error": error_msg,
                            "error_type": error_type,
                            "traceback": error_traceback,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                else:
                    # Legacy format (string only)
                    summary.errors.append(
                        {
                            "example_id": example_id,
                            "file_path": str(pdf_path),
                            "error": error_info,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
            else:
                summary.successful += 1
                if normalized_result:
                    summary.total_latency_ms += normalized_result.latency_in_ms
                elif raw_result:
                    summary.total_latency_ms += raw_result.latency_in_ms

            # Update progress
            if progress and task_id is not None:
                progress.update(task_id, advance=1, refresh=True)

        # Finalize summary
        summary.completed_at = datetime.now()

        # Save summary
        summary_path = self.output_dir / "_summary.json"
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2))

        # Save errors if any
        if summary.errors:
            errors_path = self.output_dir / "_errors.json"
            errors_path.write_text(json.dumps(summary.errors, indent=2))

        # Save run metadata
        metadata = {
            "pipeline": {
                "pipeline_name": self.pipeline.pipeline_name,
                "provider_name": self.pipeline.provider_name,
                "product_type": self.pipeline.product_type.value,
                "config": self.pipeline.config,
            },
            "run_config": {
                "max_concurrent": self.max_concurrent,
                "save_raw": self.save_raw,
                "save_normalized": self.save_normalized,
                "force": self.force,
            },
            "summary": summary.to_dict(),
        }
        # Store tags if provided
        if self.tags:
            metadata["tags"] = self.tags
        metadata_path = self.output_dir / "_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        return summary

    @staticmethod
    def _deduplicate_qa_test_cases(test_cases: list[TestCase]) -> list[TestCase]:
        """Ensure qa_configs test cases don't cause duplicate inference jobs.

        ``ParseTestCase`` with ``qa_configs`` (plural) contains multiple QA
        questions for one document.  The loader already keeps this as a single
        test case; this method is a safety net that strips ``qa_config`` /
        ``qa_configs`` before inference so the provider never sees QA fields
        (which are evaluation-only concerns).
        """
        from extract_bench.test_cases.schema import ParseTestCase as _PTC

        out: list[TestCase] = []
        for tc in test_cases:
            if isinstance(tc, _PTC) and tc.qa_configs:
                # Strip QA fields — inference only needs the file
                out.append(tc.model_copy(update={"qa_config": None, "qa_configs": None}))
            else:
                out.append(tc)
        return out

    def _run_test_cases_sync(
        self,
        test_cases: list[TestCase],
        product_type: ProductType,
        test_cases_dir: Path | None = None,
    ) -> RunSummary:
        """
        Process test cases synchronously when max_concurrent=1.

        :param test_cases: List of test cases to process
        :param product_type: Product type (PARSE or EXTRACT)
        :return: Summary of the run
        """
        self._current_summary = summary = RunSummary()

        # Initialize job statuses for Rich UI
        if self.use_rich:
            for test_case in test_cases:
                self.job_statuses[test_case.test_id] = JobStatus(
                    example_id=test_case.test_id,
                    pdf_path=test_case.file_path,
                    status="pending",
                )

        # Create progress bar if using Rich UI
        if self.use_rich and self.console:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(
                    bar_width=None,
                    style="bright_blue",
                    complete_style="green",
                    finished_style="green",
                ),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("[cyan]{task.completed}/{task.total}"),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=self.console,
                expand=True,
            )
            task_id = progress.add_task(f"Processing {self.pipeline.pipeline_name}", total=len(test_cases))
        else:
            progress = None
            task_id = None

        # Process each test case synchronously
        for test_case in test_cases:
            # Check if already processed
            if self._is_already_processed(test_case.test_id):
                summary.skipped += 1
                if self.use_rich and test_case.test_id in self.job_statuses:
                    self.job_statuses[test_case.test_id].status = "skipped"
                if progress and task_id is not None:
                    progress.update(task_id, advance=1, refresh=True)
                continue

            # Process test case with per-file timeout
            raw_result = None
            normalized_result = None
            error_info: str | tuple[str, str, str] | None = None

            for timeout_attempt in range(self.timeout_retries + 1):
                timeout_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = timeout_executor.submit(self._process_test_case, test_case, product_type)
                try:
                    raw_result, normalized_result, error_info = future.result(timeout=self.per_file_timeout)
                    break  # Success (or handled provider error) - exit retry loop
                except concurrent.futures.TimeoutError:
                    self._cancel_inflight_and_drain(test_case.test_id, future)
                    remaining = self.timeout_retries - timeout_attempt
                    if remaining > 0:
                        print(
                            f"  Timeout after {self.per_file_timeout}s for "
                            f"{test_case.test_id}, retrying ({remaining} left)"
                        )
                    else:
                        print(
                            f"  Timeout after {self.per_file_timeout}s for "
                            f"{test_case.test_id}, giving up after "
                            f"{self.timeout_retries + 1} attempts"
                        )
                        error_info = (
                            f"Per-file timeout ({self.per_file_timeout}s) exceeded "
                            f"after {self.timeout_retries + 1} attempts",
                            "",
                            "TimeoutError",
                        )
                        raw_result, normalized_result = None, None
                finally:
                    timeout_executor.shutdown(wait=False)

            summary.total += 1

            if error_info:
                summary.failed += 1
                # Handle both old format (string) and new format (tuple)
                if isinstance(error_info, tuple):
                    error_msg, error_traceback, error_type = error_info
                    summary.errors.append(
                        {
                            "example_id": test_case.test_id,
                            "file_path": str(test_case.file_path),
                            "error": error_msg,
                            "error_type": error_type,
                            "traceback": error_traceback,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                else:
                    # Legacy format (string only)
                    summary.errors.append(
                        {
                            "example_id": test_case.test_id,
                            "file_path": str(test_case.file_path),
                            "error": error_info,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
            else:
                summary.successful += 1
                if normalized_result:
                    summary.total_latency_ms += normalized_result.latency_in_ms
                elif raw_result:
                    summary.total_latency_ms += raw_result.latency_in_ms

            # Update progress
            if progress and task_id is not None:
                progress.update(task_id, advance=1, refresh=True)

        # Finalize summary
        summary.completed_at = datetime.now()

        # Save summary
        summary_path = self.output_dir / "_summary.json"
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2))

        # Save errors if any
        if summary.errors:
            errors_path = self.output_dir / "_errors.json"
            errors_path.write_text(json.dumps(summary.errors, indent=2))

        # Save run metadata
        metadata = {
            "pipeline": {
                "pipeline_name": self.pipeline.pipeline_name,
                "provider_name": self.pipeline.provider_name,
                "product_type": self.pipeline.product_type.value,
                "config": self.pipeline.config,
            },
            "run_config": {
                "max_concurrent": self.max_concurrent,
                "save_raw": self.save_raw,
                "save_normalized": self.save_normalized,
                "force": self.force,
            },
            "summary": summary.to_dict(),
        }
        # Store test_cases_dir if provided
        if test_cases_dir:
            metadata["test_cases_dir"] = str(test_cases_dir.resolve())
        # Store tags if provided
        if self.tags:
            metadata["tags"] = self.tags
        metadata_path = self.output_dir / "_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        return summary

    async def _process_test_case_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        test_case: TestCase,
        product_type: ProductType,
        summary: RunSummary,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> None:
        """Process a test case with semaphore-based concurrency control."""
        # Check if already processed before acquiring semaphore to avoid wasting slots
        if self._is_already_processed(test_case.test_id):
            summary.skipped += 1
            if self.use_rich and test_case.test_id in self.job_statuses:
                self.job_statuses[test_case.test_id].status = "skipped"
            if progress and task_id is not None:
                progress.update(task_id, advance=1, refresh=True)
            return

        async with semaphore:
            # Set status to "running" when semaphore is acquired (before processing starts)
            # This allows UI to show "in progress" status immediately
            if self.use_rich and test_case.test_id in self.job_statuses:
                self.job_statuses[test_case.test_id].status = "running"
                self.job_statuses[test_case.test_id].started_at = datetime.now()

            # Process test case using our custom thread pool with per-file timeout.
            raw_result = None
            normalized_result = None
            error_info: str | tuple[str, str, str] | None = None

            for timeout_attempt in range(self.timeout_retries + 1):
                future = self._thread_pool.submit(self._process_test_case, test_case, product_type)
                try:
                    raw_result, normalized_result, error_info = await asyncio.wait_for(
                        asyncio.wrap_future(future),
                        timeout=self.per_file_timeout,
                    )
                    break  # Success (or handled provider error) - exit retry loop
                except TimeoutError:
                    await self._cancel_inflight_and_drain_async(test_case.test_id, future)
                    remaining = self.timeout_retries - timeout_attempt
                    if remaining > 0:
                        print(
                            f"  Timeout after {self.per_file_timeout}s for "
                            f"{test_case.test_id}, retrying ({remaining} left)"
                        )
                    else:
                        print(
                            f"  Timeout after {self.per_file_timeout}s for "
                            f"{test_case.test_id}, giving up after "
                            f"{self.timeout_retries + 1} attempts"
                        )
                        error_info = (
                            f"Per-file timeout ({self.per_file_timeout}s) exceeded "
                            f"after {self.timeout_retries + 1} attempts",
                            "",
                            "TimeoutError",
                        )
                        raw_result, normalized_result = None, None

            summary.total += 1

            if error_info:
                summary.failed += 1
                # Handle both old format (string) and new format (tuple)
                if isinstance(error_info, tuple):
                    error_msg, error_traceback, error_type = error_info
                    summary.errors.append(
                        {
                            "example_id": test_case.test_id,
                            "file_path": str(test_case.file_path),
                            "error": error_msg,
                            "error_type": error_type,
                            "traceback": error_traceback,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                else:
                    # Legacy format (string only)
                    summary.errors.append(
                        {
                            "example_id": test_case.test_id,
                            "file_path": str(test_case.file_path),
                            "error": error_info,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
            else:
                summary.successful += 1
                if normalized_result:
                    summary.total_latency_ms += normalized_result.latency_in_ms
                elif raw_result:
                    summary.total_latency_ms += raw_result.latency_in_ms

            # Update progress after processing
            if progress and task_id is not None:
                progress.update(task_id, advance=1, refresh=True)

    async def _process_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        pdf_path: Path,
        example_id: str,
        product_type: ProductType,
        summary: RunSummary,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> None:
        """Process a document with semaphore-based concurrency control."""
        # Check if already processed before acquiring semaphore to avoid wasting slots
        if self._is_already_processed(example_id):
            summary.skipped += 1
            if self.use_rich and example_id in self.job_statuses:
                self.job_statuses[example_id].status = "skipped"
            if progress and task_id is not None:
                progress.update(task_id, advance=1, refresh=True)
            return

        async with semaphore:
            # Set status to "running" when semaphore is acquired (before processing starts)
            # This allows UI to show "in progress" status immediately
            if self.use_rich and example_id in self.job_statuses:
                self.job_statuses[example_id].status = "running"
                self.job_statuses[example_id].started_at = datetime.now()

            # Process document using our custom thread pool with per-file timeout.
            raw_result = None
            normalized_result = None
            error_info: str | tuple[str, str, str] | None = None

            for timeout_attempt in range(self.timeout_retries + 1):
                future = self._thread_pool.submit(self._process_document, pdf_path, example_id, product_type)
                try:
                    raw_result, normalized_result, error_info = await asyncio.wait_for(
                        asyncio.wrap_future(future),
                        timeout=self.per_file_timeout,
                    )
                    break  # Success (or handled provider error) - exit retry loop
                except TimeoutError:
                    await self._cancel_inflight_and_drain_async(example_id, future)
                    remaining = self.timeout_retries - timeout_attempt
                    if remaining > 0:
                        print(f"  Timeout after {self.per_file_timeout}s for {example_id}, retrying ({remaining} left)")
                    else:
                        print(
                            f"  Timeout after {self.per_file_timeout}s for "
                            f"{example_id}, giving up after "
                            f"{self.timeout_retries + 1} attempts"
                        )
                        error_info = (
                            f"Per-file timeout ({self.per_file_timeout}s) exceeded "
                            f"after {self.timeout_retries + 1} attempts",
                            "",
                            "TimeoutError",
                        )
                        raw_result, normalized_result = None, None

            summary.total += 1

            if error_info:
                summary.failed += 1
                # Handle both old format (string) and new format (tuple)
                if isinstance(error_info, tuple):
                    error_msg, error_traceback, error_type = error_info
                    summary.errors.append(
                        {
                            "example_id": example_id,
                            "file_path": str(pdf_path),
                            "error": error_msg,
                            "error_type": error_type,
                            "traceback": error_traceback,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                else:
                    # Legacy format (string only)
                    summary.errors.append(
                        {
                            "example_id": example_id,
                            "file_path": str(pdf_path),
                            "error": error_info,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
            else:
                summary.successful += 1
                if normalized_result:
                    summary.total_latency_ms += normalized_result.latency_in_ms
                elif raw_result:
                    summary.total_latency_ms += raw_result.latency_in_ms

            # Update progress after processing - use refresh=True to force update
            if progress and task_id is not None:
                progress.update(task_id, advance=1, refresh=True)

    def _create_status_table(self, summary: RunSummary) -> Table:
        """Create a table showing current job statuses."""
        table = Table(title="Active Jobs", show_header=True, header_style="bold magenta")
        table.add_column("Example ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="bold")
        table.add_column("Latency", justify="right")
        table.add_column("File", style="dim")

        # Show running and recently completed jobs (limit to 10 most recent)
        active_jobs = [job for job in self.job_statuses.values() if job.status in ("running", "completed", "failed")]
        active_jobs.sort(key=lambda j: j.completed_at or j.started_at or datetime.min, reverse=True)

        for job in active_jobs[:10]:
            status_style = {
                "running": "[yellow]● Running[/yellow]",
                "completed": "[green]✓ Done[/green]",
                "failed": "[red]✗ Failed[/red]",
                "skipped": "[dim]⊘ Skipped[/dim]",
                "pending": "[dim]○ Pending[/dim]",
            }.get(job.status, job.status)

            latency_str = f"{job.latency_ms}ms" if job.latency_ms is not None else "-"

            file_name = job.pdf_path.name[:40] + "..." if len(job.pdf_path.name) > 40 else job.pdf_path.name

            table.add_row(job.example_id, status_style, latency_str, file_name)

        if not active_jobs:
            table.add_row("[dim]No active jobs[/dim]", "", "", "")

        return table

    def _create_stats_panel(self, summary: RunSummary, total_files: int) -> Panel:
        """Create a panel with summary statistics."""
        elapsed = (
            (summary.completed_at - summary.started_at).total_seconds()
            if summary.completed_at
            else (datetime.now() - summary.started_at).total_seconds()
        )

        # Count in-progress jobs
        in_progress = sum(1 for job in self.job_statuses.values() if job.status == "running")

        stats_text = f"""
[bold]Pipeline:[/bold] {self.pipeline.pipeline_name}
[bold]Total Files:[/bold] {total_files}
[bold]Processed:[/bold] {summary.total}
[bold]In Progress:[/bold] [yellow]{in_progress}[/yellow]
[bold]Successful:[/bold] [green]{summary.successful}[/green]
[bold]Failed:[/bold] [red]{summary.failed}[/red]
[bold]Skipped:[/bold] [dim]{summary.skipped}[/dim]
[bold]Success Rate:[/bold] {summary.success_rate:.1f}%
[bold]Avg Latency:[/bold] {summary.avg_latency_ms:.1f}ms
[bold]Elapsed:[/bold] {elapsed:.1f}s
"""
        return Panel(stats_text, title="Statistics", border_style="blue")

    def _create_rich_ui(
        self,
        summary: RunSummary,
        total_files: int,
        progress: Progress,
        stats_panel: Panel | None = None,
        status_table: Table | None = None,
    ) -> Panel:
        """Create the main Rich UI layout."""
        # Recreate panels/tables if not provided (for updates)
        if stats_panel is None:
            stats_panel = self._create_stats_panel(summary, total_files)
        if status_table is None:
            status_table = self._create_status_table(summary)

        # Use Group to combine all elements
        # IMPORTANT: Progress object must be the same instance throughout
        # Don't recreate it, just pass the same object
        group = Group(
            stats_panel,
            status_table,
            progress,  # Same Progress instance - updates automatically
        )

        title = f"[bold]{self.pipeline.pipeline_name}[/bold]"
        return Panel(group, title=title, border_style="green")

    async def run_directory(
        self,
        pdf_directory: Path,
        product_type: ProductType,
        pattern: str = "*.pdf",
        recursive: bool = True,
    ) -> RunSummary:
        """
        Process all PDFs in a directory.

        :param pdf_directory: Directory containing PDFs
        :param product_type: Product type (PARSE or EXTRACT)
        :param pattern: Glob pattern for PDF files (default: "*.pdf")
        :param recursive: Whether to search recursively in subdirectories
        :return: Summary of the run
        """
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise ValueError(f"PDF directory does not exist: {pdf_directory}")

        # Find all PDFs
        if recursive:
            all_pdfs = list(pdf_dir.rglob(pattern))
        else:
            all_pdfs = list(pdf_dir.glob(pattern))

        all_pdfs.sort()

        if not all_pdfs:
            raise ValueError(f"No PDFs found matching pattern '{pattern}' in {pdf_directory}")

        return await self.run_files(all_pdfs, product_type)

    async def run_files(
        self,
        pdf_files: list[Path],
        product_type: ProductType,
        example_id_fn: Callable[[Path], str] | None = None,
    ) -> RunSummary:
        """
        Process a list of PDF files.

        :param pdf_files: List of PDF file paths
        :param product_type: Product type (PARSE or EXTRACT)
        :param example_id_fn: Optional function to generate example_id from PDF path.
                              Default: uses PDF filename without extension
        :return: Summary of the run
        """
        if example_id_fn is None:

            def default_example_id_fn(pdf_path: Path) -> str:
                """Generate example_id from PDF filename."""
                return pdf_path.stem

            example_id_fn = default_example_id_fn

        # When max_concurrent is 1, run synchronously without asyncio/threads
        if self.max_concurrent == 1:
            return self._run_files_sync(pdf_files, product_type, example_id_fn)

        self._current_summary = summary = RunSummary()

        # Initialize job statuses for Rich UI
        if self.use_rich:
            for pdf_path in pdf_files:
                example_id = example_id_fn(pdf_path)
                self.job_statuses[example_id] = JobStatus(example_id=example_id, pdf_path=pdf_path, status="pending")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Create progress bar with enhanced styling
        if self.use_rich:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(
                    bar_width=None,
                    style="bright_blue",
                    complete_style="green",
                    finished_style="green",
                ),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("[cyan]{task.completed}/{task.total}"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=self.console,
                expand=True,
            )
            task_id = progress.add_task(f"Processing {self.pipeline.pipeline_name}", total=len(pdf_files))
        else:
            progress = None
            task_id = None

        # Create tasks
        tasks = [
            self._process_with_semaphore(
                semaphore,
                pdf_path,
                example_id_fn(pdf_path),
                product_type,
                summary,
                progress,
                task_id,
            )
            for pdf_path in pdf_files
        ]

        # Process with Rich UI or simple progress
        if self.use_rich and self.console:
            # Create initial UI components
            stats_panel = self._create_stats_panel(summary, len(pdf_files))
            status_table = self._create_status_table(summary)
            last_update_time = datetime.now()
            update_interval = 0.2  # Update UI every 200ms to reduce flickering

            # Use Progress with Live
            # Progress updates automatically when we call progress.update()
            # Create the initial UI once - Progress object is stable
            initial_ui = self._create_rich_ui(
                summary,
                len(pdf_files),
                progress,  # type: ignore[arg-type]
                stats_panel,
                status_table,
            )

            with Live(
                initial_ui,
                console=self.console,
                refresh_per_second=10,  # Higher refresh for progress updates
            ) as live:
                # Use a list to allow modification in nested function
                last_update_time = [last_update_time]  # type: ignore[assignment]

                # Background task to periodically update UI to show status changes
                # (e.g., "running")
                async def update_ui_periodically():  # type: ignore[no-untyped-def]
                    """Background task to update UI periodically to show status changes."""
                    while True:
                        # Update every 1s to catch status changes
                        await asyncio.sleep(1.0)
                        now = datetime.now()
                        should_refresh_stats = (
                            now - last_update_time[0]  # type: ignore[index]
                        ).total_seconds() >= update_interval

                        if should_refresh_stats:
                            nonlocal stats_panel, status_table
                            stats_panel = self._create_stats_panel(summary, len(pdf_files))
                            status_table = self._create_status_table(summary)
                            last_update_time[0] = now  # type: ignore[index]

                        # Always update Live UI to show current status
                        # (including "running" status)
                        live.update(
                            self._create_rich_ui(
                                summary,
                                len(pdf_files),
                                progress,  # type: ignore[arg-type]
                                stats_panel,
                                status_table,
                            )
                        )

                # Start background UI update task
                ui_update_task = asyncio.create_task(update_ui_periodically())

                try:
                    for coro in asyncio.as_completed(tasks):
                        try:
                            await coro
                        except Exception as e:
                            summary.failed += 1
                            summary.errors.append(
                                {
                                    "error": f"Task execution error: {str(e)}",
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )
                        finally:
                            # Also update immediately when task completes to show progress
                            now = datetime.now()
                            should_refresh_stats = (
                                now - last_update_time[0]  # type: ignore[index]
                            ).total_seconds() >= update_interval

                            if should_refresh_stats:
                                stats_panel = self._create_stats_panel(summary, len(pdf_files))
                                status_table = self._create_status_table(summary)
                                last_update_time[0] = now  # type: ignore[index]

                            # Update Live UI immediately on task completion
                            live.update(
                                self._create_rich_ui(
                                    summary,
                                    len(pdf_files),
                                    progress,  # type: ignore[arg-type]
                                    stats_panel,
                                    status_table,
                                )
                            )
                finally:
                    # Cancel background UI update task
                    ui_update_task.cancel()
                    try:
                        await ui_update_task
                    except asyncio.CancelledError:
                        pass

                # Final update to ensure everything is current
                stats_panel = self._create_stats_panel(summary, len(pdf_files))
                status_table = self._create_status_table(summary)
                live.update(
                    self._create_rich_ui(
                        summary,
                        len(pdf_files),
                        progress,  # type: ignore[arg-type]
                        stats_panel,
                        status_table,
                    )
                )
        else:
            # Fallback to simple processing
            for coro in asyncio.as_completed(tasks):
                try:
                    await coro
                except Exception as e:
                    summary.failed += 1
                    summary.errors.append(
                        {
                            "error": f"Task execution error: {str(e)}",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

        # Finalize summary
        summary.completed_at = datetime.now()

        # Save summary
        summary_path = self.output_dir / "_summary.json"
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2))

        # Save errors if any
        if summary.errors:
            errors_path = self.output_dir / "_errors.json"
            errors_path.write_text(json.dumps(summary.errors, indent=2))

        # Save run metadata
        metadata = {
            "pipeline": {
                "pipeline_name": self.pipeline.pipeline_name,
                "provider_name": self.pipeline.provider_name,
                "product_type": self.pipeline.product_type.value,
                "config": self.pipeline.config,
            },
            "run_config": {
                "max_concurrent": self.max_concurrent,
                "save_raw": self.save_raw,
                "save_normalized": self.save_normalized,
                "force": self.force,
            },
            "summary": summary.to_dict(),
        }
        # Store tags if provided
        if self.tags:
            metadata["tags"] = self.tags
        metadata_path = self.output_dir / "_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        return summary

    async def run_test_cases(
        self,
        test_cases: list[TestCase],
        product_type: ProductType,
        test_cases_dir: Path | None = None,
    ) -> RunSummary:
        """
        Process a list of test cases.

        :param test_cases: List of test cases to process
        :param product_type: Product type (PARSE or EXTRACT)
        :return: Summary of the run
        """
        if not test_cases:
            raise ValueError("No test cases provided")

        # Deduplicate qa_configs test cases so each PDF is parsed only once
        test_cases = self._deduplicate_qa_test_cases(test_cases)

        # Deduplicate by test_id: categories like text_content and text_formatting
        # share the same PDF files, so they map to the same test_id. Only run
        # inference once per unique file.
        seen_ids: set[str] = set()
        unique: list[TestCase] = []
        for tc in test_cases:
            if tc.test_id not in seen_ids:
                seen_ids.add(tc.test_id)
                unique.append(tc)
        test_cases = unique

        # When max_concurrent is 1, run synchronously without asyncio/threads
        if self.max_concurrent == 1:
            return self._run_test_cases_sync(test_cases, product_type, test_cases_dir)

        self._current_summary = summary = RunSummary()

        # Log concurrency setting for debugging
        # Thread pool is sized to match max_concurrent to avoid default pool bottleneck
        print(
            "Starting async run_test_cases with "
            f"max_concurrent={self.max_concurrent} "
            f"(thread pool size: {self._thread_pool._max_workers})"
        )

        # Initialize job statuses for Rich UI
        if self.use_rich:
            for test_case in test_cases:
                self.job_statuses[test_case.test_id] = JobStatus(
                    example_id=test_case.test_id,
                    pdf_path=test_case.file_path,
                    status="pending",
                )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Create progress bar with enhanced styling
        if self.use_rich:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(
                    bar_width=None,
                    style="bright_blue",
                    complete_style="green",
                    finished_style="green",
                ),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("[cyan]{task.completed}/{task.total}"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=self.console,
                expand=True,
            )
            task_id = progress.add_task(f"Processing {self.pipeline.pipeline_name}", total=len(test_cases))
        else:
            progress = None
            task_id = None

        # Create tasks
        tasks = [
            self._process_test_case_with_semaphore(
                semaphore,
                test_case,
                product_type,
                summary,
                progress,
                task_id,
            )
            for test_case in test_cases
        ]

        # Process with Rich UI or simple progress
        if self.use_rich and self.console:
            # Create initial UI components
            stats_panel = self._create_stats_panel(summary, len(test_cases))
            status_table = self._create_status_table(summary)
            last_update_time = datetime.now()
            update_interval = 0.2  # Update UI every 200ms to reduce flickering

            # Create the initial UI once
            initial_ui = self._create_rich_ui(
                summary,
                len(test_cases),
                progress,  # type: ignore[arg-type]
                stats_panel,
                status_table,
            )

            with Live(
                initial_ui,
                console=self.console,
                refresh_per_second=10,
            ) as live:
                # Use a list to allow modification in nested function
                last_update_time = [last_update_time]  # type: ignore[assignment]

                # Background task to periodically update UI to show status changes
                # (e.g., "running")
                async def update_ui_periodically():  # type: ignore[no-untyped-def]
                    """Background task to update UI periodically to show status changes."""
                    while True:
                        # Update every 100ms to catch status changes
                        await asyncio.sleep(0.1)
                        now = datetime.now()
                        should_refresh_stats = (
                            now - last_update_time[0]  # type: ignore[index]
                        ).total_seconds() >= update_interval

                        if should_refresh_stats:
                            nonlocal stats_panel, status_table
                            stats_panel = self._create_stats_panel(summary, len(test_cases))
                            status_table = self._create_status_table(summary)
                            last_update_time[0] = now  # type: ignore[index]

                        # Always update Live UI to show current status
                        # (including "running" status)
                        live.update(
                            self._create_rich_ui(
                                summary,
                                len(test_cases),
                                progress,  # type: ignore[arg-type]
                                stats_panel,
                                status_table,
                            )
                        )

                # Start background UI update task
                ui_update_task = asyncio.create_task(update_ui_periodically())

                try:
                    for coro in asyncio.as_completed(tasks):
                        try:
                            await coro
                        except Exception as e:
                            summary.failed += 1
                            summary.errors.append(
                                {
                                    "error": f"Task execution error: {str(e)}",
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )
                        finally:
                            # Also update immediately when task completes to show progress
                            now = datetime.now()
                            should_refresh_stats = (
                                now - last_update_time[0]  # type: ignore[index]
                            ).total_seconds() >= update_interval

                            if should_refresh_stats:
                                stats_panel = self._create_stats_panel(summary, len(test_cases))
                                status_table = self._create_status_table(summary)
                                last_update_time[0] = now  # type: ignore[index]

                            # Update Live UI immediately on task completion
                            live.update(
                                self._create_rich_ui(
                                    summary,
                                    len(test_cases),
                                    progress,  # type: ignore[arg-type]
                                    stats_panel,
                                    status_table,
                                )
                            )
                finally:
                    # Cancel background UI update task
                    ui_update_task.cancel()
                    try:
                        await ui_update_task
                    except asyncio.CancelledError:
                        pass

                # Final update
                stats_panel = self._create_stats_panel(summary, len(test_cases))
                status_table = self._create_status_table(summary)
                live.update(
                    self._create_rich_ui(
                        summary,
                        len(test_cases),
                        progress,  # type: ignore[arg-type]
                        stats_panel,
                        status_table,
                    )
                )
        else:
            # Fallback to simple processing with progress indicators
            total = len(test_cases)
            print(f"Processing {total} test cases with pipeline '{self.pipeline.pipeline_name}'...")

            completed_count = 0
            last_progress_print = 0
            # Print every 10% or every 10 items, whichever is more frequent
            progress_interval = max(10, total // 10)

            for coro in asyncio.as_completed(tasks):
                try:
                    await coro
                    completed_count += 1

                    # Print progress periodically
                    if completed_count - last_progress_print >= progress_interval or completed_count == total:
                        percentage = (completed_count / total) * 100
                        print(
                            f"Progress: {completed_count}/{total} ({percentage:.1f}%) - "
                            f"Successful: {summary.successful}, Failed: {summary.failed}"
                        )
                        last_progress_print = completed_count
                except Exception as e:
                    summary.failed += 1
                    completed_count += 1
                    summary.errors.append(
                        {
                            "error": f"Task execution error: {str(e)}",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                    # Print progress on error too
                    if completed_count - last_progress_print >= progress_interval or completed_count == total:
                        percentage = (completed_count / total) * 100
                        print(
                            f"Progress: {completed_count}/{total} ({percentage:.1f}%) - "
                            f"Successful: {summary.successful}, Failed: {summary.failed}"
                        )
                        last_progress_print = completed_count

            # Print final summary
            print(f"\nCompleted processing {total} test cases:")
            print(f"  Successful: {summary.successful}")
            print(f"  Failed: {summary.failed}")
            print(f"  Skipped: {summary.skipped}")

        # Finalize summary
        summary.completed_at = datetime.now()

        # Save summary
        summary_path = self.output_dir / "_summary.json"
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2))

        # Save errors if any
        if summary.errors:
            errors_path = self.output_dir / "_errors.json"
            errors_path.write_text(json.dumps(summary.errors, indent=2))

        # Save run metadata
        metadata = {
            "pipeline": {
                "pipeline_name": self.pipeline.pipeline_name,
                "provider_name": self.pipeline.provider_name,
                "product_type": self.pipeline.product_type.value,
                "config": self.pipeline.config,
            },
            "run_config": {
                "max_concurrent": self.max_concurrent,
                "save_raw": self.save_raw,
                "save_normalized": self.save_normalized,
                "force": self.force,
            },
            "summary": summary.to_dict(),
        }
        # Store test_cases_dir if provided
        if test_cases_dir:
            metadata["test_cases_dir"] = str(test_cases_dir.resolve())
        # Store tags if provided
        if self.tags:
            metadata["tags"] = self.tags
        metadata_path = self.output_dir / "_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        return summary
