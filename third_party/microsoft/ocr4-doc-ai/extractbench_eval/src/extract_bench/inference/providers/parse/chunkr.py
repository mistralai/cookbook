"""Provider for Chunkr PARSE."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
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


@register_provider("chunkr")
class ChunkrProvider(Provider):
    """
    Provider for Chunkr PARSE.

    Uses the Chunkr API for parsing documents with HTML table output.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `api_key`: Chunkr API key (defaults to CHUNKR_API_KEY env var)
            - `segmentation_strategy`: "LayoutAnalysis" or "Page" (default: "LayoutAnalysis")
            - `ocr_strategy`: "Auto" or "All" (default: "Auto")
            - `high_resolution`: Whether to use high resolution processing (default: False)
        """
        super().__init__(provider_name, base_config)

        # Get API key
        self._api_key = self.base_config.get("api_key") or os.getenv("CHUNKR_API_KEY")
        if not self._api_key:
            raise ProviderConfigError(
                "Chunkr API key is required. Set CHUNKR_API_KEY environment variable or pass api_key in base_config."
            )

        # Configuration options
        self._segmentation_strategy = self.base_config.get("segmentation_strategy", "LayoutAnalysis")
        self._ocr_strategy = self.base_config.get("ocr_strategy", "Auto")
        self._high_resolution = self.base_config.get("high_resolution", False)

    async def _parse_document_async(self, file_path: str) -> dict[str, Any]:
        """
        Parse a document using Chunkr API (async).

        :param file_path: Path to the document file
        :return: Raw API response as dictionary
        :raises ProviderError: For any API errors
        """
        try:
            from chunkr_ai import Chunkr  # type: ignore[import-untyped]
            from chunkr_ai.models import (  # type: ignore[import-untyped]
                Configuration,
                OcrStrategy,
                SegmentationStrategy,
            )

            # Map string config values to enum values
            segmentation_map = {
                "layoutanalysis": SegmentationStrategy.LAYOUT_ANALYSIS,
                "layout_analysis": SegmentationStrategy.LAYOUT_ANALYSIS,
                "page": SegmentationStrategy.PAGE,
            }
            ocr_map = {
                "auto": OcrStrategy.AUTO,
                "all": OcrStrategy.ALL,
            }

            seg_strategy = segmentation_map.get(
                self._segmentation_strategy.lower(),
                SegmentationStrategy.LAYOUT_ANALYSIS,
            )
            ocr_strategy = ocr_map.get(
                self._ocr_strategy.lower(),
                OcrStrategy.AUTO,
            )

            # Initialize client
            client = Chunkr(api_key=self._api_key)

            try:
                # Configure for HTML output (tables are HTML by default in Chunkr)
                config = Configuration(
                    segmentation_strategy=seg_strategy,
                    ocr_strategy=ocr_strategy,
                    high_resolution=self._high_resolution,
                )

                # Upload and process
                # Note: The Chunkr SDK is async-native with an @anywhere() decorator.
                # We must call it directly as async (not via asyncio.to_thread) to avoid
                # race conditions with the SDK's global _sync_loop singleton.
                task = await client.upload(file_path, config)

                # poll() ensures task is complete (no-op if already complete)
                if hasattr(task, "poll") and callable(task.poll):
                    task = await task.poll()

                # Extract raw response
                if hasattr(task, "model_dump"):
                    raw_response = task.model_dump()
                elif hasattr(task, "dict"):
                    raw_response = task.dict()
                else:
                    # Manual extraction as fallback
                    raw_response = {
                        "task_id": getattr(task, "task_id", None),
                        "status": getattr(task, "status", None),
                        "output": getattr(task, "output", None),
                    }

                # Get HTML content (includes tables as HTML)
                try:
                    html_content = await task.html() if hasattr(task, "html") else ""
                except Exception:
                    html_content = ""
                raw_response["_html_content"] = html_content

                # Get markdown content as alternative
                try:
                    markdown_content = await task.markdown() if hasattr(task, "markdown") else ""
                except Exception:
                    markdown_content = ""
                raw_response["_markdown_content"] = markdown_content

                # Store configuration for reference
                raw_response["_config"] = {
                    "segmentation_strategy": self._segmentation_strategy,
                    "ocr_strategy": self._ocr_strategy,
                    "high_resolution": self._high_resolution,
                }

                result: dict[str, Any] = raw_response
                return result

            finally:
                # Close the client (async method with @anywhere decorator)
                await client.close()

        except ImportError as e:
            raise ProviderConfigError("chunkr-ai package not installed. Run: pip install chunkr-ai") from e
        except Exception as e:
            error_str = str(e).lower()
            transient_keywords = ["timeout", "network", "connection", "503", "502", "504"]
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during parsing: {e}") from e
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
            raise ProviderPermanentError(f"ChunkrProvider only supports PARSE product type, got {request.product_type}")

        started_at = datetime.now()

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        try:
            raw_output = self.run_async_from_sync(self._parse_document_async(str(file_path)))

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

        except (ProviderPermanentError, ProviderTransientError):
            raise
        except Exception as e:
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
                f"ChunkrProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        # Extract HTML content (preferred for table tests)
        html_content = raw_result.raw_output.get("_html_content", "")

        # Fallback: extract from output.chunks if _html_content not available
        if not html_content:
            output = raw_result.raw_output.get("output", {})
            chunks = output.get("chunks", [])
            # Concatenate HTML from all chunks/segments
            html_parts = []
            for chunk in chunks:
                segments = chunk.get("segments", [])
                for segment in segments:
                    html = segment.get("html", "")
                    if html:
                        html_parts.append(html)
            html_content = "\n".join(html_parts)

        # If still no HTML content, try markdown
        if not html_content:
            html_content = raw_result.raw_output.get("_markdown_content", "")

        # Final fallback: concatenate content from chunks
        if not html_content:
            output = raw_result.raw_output.get("output", {})
            chunks = output.get("chunks", [])
            content_parts = []
            for chunk in chunks:
                content = chunk.get("content", "")
                if content:
                    content_parts.append(content)
            html_content = "\n".join(content_parts)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],  # Chunkr doesn't provide per-page split by default
            markdown=html_content,  # HTML content goes here for table tests
            job_id=raw_result.raw_output.get("task_id"),
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
