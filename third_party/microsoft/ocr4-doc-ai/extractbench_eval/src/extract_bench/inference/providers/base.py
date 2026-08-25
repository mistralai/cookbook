import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from typing import Any

from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)


class ProviderError(Exception):
    """Base exception for provider-related failures."""

    def __init__(
        self,
        message: str,
        *,
        debug_payload: dict[str, Any] | None = None,
        job_id: str | None = None,
    ):
        super().__init__(message)
        self.debug_payload = debug_payload
        self.job_id = job_id


class ProviderConfigError(ProviderError):
    """Raised when a provider is misconfigured (missing API keys, bad endpoint, etc.)."""


class ProviderRateLimitError(ProviderError):
    """Raised when a provider hits rate limits or quota issues."""


class ProviderTransientError(ProviderError):
    """
    Raised for transient errors that may succeed on retry
    (e.g. network issues, 5xx responses).
    """


class ProviderPermanentError(ProviderError):
    """
    Raised for permanent errors that are not expected to succeed on retry
    (e.g. unsupported file type, invalid request, 4xx errors).
    """


class Provider(ABC):
    """Abstract base class for document parsing providers."""

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """
        Initialize a provider.

        :param provider_name: Name of the provider
        :param base_config: Optional shared configuration dictionary.
            May include provider-specific environment options (e.g. `use_europe`).
        """
        self._provider_name = provider_name
        self._base_config = base_config or {}

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self._provider_name

    @property
    def base_config(self) -> dict[str, Any]:
        """Return the base configuration."""
        return self._base_config

    @property
    def credit_rate_usd(self) -> float | None:
        """USD cost per credit. Override in subclasses that charge credits."""
        return None

    @staticmethod
    def run_async_from_sync(coro: Coroutine[Any, Any, Any]) -> Any:
        """
        Run an async coroutine from a synchronous context.

        This helper handles both cases:
        - If there's no running event loop, uses asyncio.run()
        - If there's a running event loop, runs the coroutine in a new thread with a new event loop

        :param coro: The coroutine to run
        :return: The result of the coroutine
        """
        try:
            # Try to get the current event loop
            # If we get here, there's a running loop, so we need to run in a thread
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No running event loop, we can use asyncio.run() directly
            return asyncio.run(coro)

    @abstractmethod
    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference for a single request and return raw results.

        This method should only fetch raw data from the provider API.
        Normalization is handled separately by the normalize() method.

        :param pipeline: Pipeline specification
        :param request: Inference request
        :return: Raw inference result (before normalization)
        :raises ProviderError: For any provider-related failures
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce structured output.

        This method converts the raw API response into a structured
        format (ParseOutput or ExtractOutput) while preserving the
        raw output for potential re-normalization.

        Note: Each provider implementation is product-type specific
        and will return either ParseOutput or ExtractOutput, not both.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        raise NotImplementedError("Subclasses must implement this method")

    def cancel(self, example_id: str) -> bool:
        """
        Cancel any in-flight inference work for the given ``example_id``.

        Default implementation is a no-op for providers that do not spawn
        external resources (subprocesses, remote jobs) that need explicit
        teardown. Providers that fork subprocesses or hold network handles
        should override this to actually terminate the work — otherwise
        per-file timeouts in the runner will only release the calling
        thread while the underlying work continues to run, racing the
        retry attempt and producing duplicate / zombie processes.

        :param example_id: The ``InferenceRequest.example_id`` for the
            in-flight request that should be cancelled.
        :return: True if a matching in-flight request was found and a
            cancellation signal was issued; False otherwise.
        """
        return False

    def run_inference_normalized(self, pipeline: PipelineSpec, request: InferenceRequest) -> InferenceResult:
        """
        Run inference and normalize in one step (convenience method).

        This is a convenience method that combines run_inference() and
        normalize() for backward compatibility and simple use cases.

        :param pipeline: Pipeline specification
        :param request: Inference request
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any provider-related failures
        """
        raw_result = self.run_inference(pipeline, request)
        return self.normalize(raw_result)
