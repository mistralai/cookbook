"""Provider for LlamaParse PARSE and LAYOUT_DETECTION."""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from llama_cloud import LlamaCloud

    _HAS_V2_SDK = True
except ImportError:
    _HAS_V2_SDK = False
from PIL import Image

from extract_bench.inference.layout_extraction import (
    extract_all_layouts_from_llamaparse_output,
)
from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.parse.llamaparse_v2_normalization import (
    build_pages_from_sdk_response_payload,
    build_parse_output_from_pages,
    extract_job_id_from_raw_payload,
    layout_pages_to_legacy_pages_payload,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType


@register_provider("llamaparse")
class LlamaParseProvider(Provider):
    """
    Provider for LlamaParse PARSE.

    This provider uses the LlamaParse API for parsing tasks.
    """

    CREDIT_RATE_USD = 0.00125  # $1.25 per 1,000 credits

    # Credits per page by tier
    _CREDITS_PER_PAGE = {
        "agentic": 10,
        "agentic_plus": 45,
        "cost_effective": 3,
    }

    # Parameters that are handled by the provider and should not be forwarded to the SDK
    _PROVIDER_ONLY_PARAMS = {"use_europe", "api_key", "base_url"}

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration. Provider-specific parameters:
            - `api_key`: LlamaCloud API key (defaults to LLAMA_CLOUD_API_KEY env var)
            - `base_url`: Override the LlamaParse API base URL (defaults to
              LLAMA_CLOUD_BASE_URL env var). When set, it takes precedence over
              use_europe and the default prod URL. Useful for
              custom deployments, e.g. http://localhost:8000.
            - `use_europe`: Use European Union (EU) region (default: False)

            All other parameters are forwarded directly to the V2 LlamaParse SDK.
            See LlamaParse SDK documentation for available options including:
            tier, version, disable_cache, parse_mode, model,
            specialized_chart_parsing_agentic, and many more.
        """
        super().__init__(provider_name, base_config)

        if not _HAS_V2_SDK:
            raise ProviderConfigError(
                "LlamaParse V2 provider requires llama-cloud>=1.4.1. Install it with: pip install 'llama-cloud>=1.4.1'"
            )

        self._credit_rate_usd = self.CREDIT_RATE_USD

        # Get API key - use EU key if in EU mode
        use_europe = self.base_config.get("use_europe", False)

        # An explicit base URL override wins over EU/prod selection.
        # Precedence: base_config["base_url"] -> LLAMA_CLOUD_BASE_URL env var.
        explicit_base_url = self.base_config.get("base_url") or os.getenv("LLAMA_CLOUD_BASE_URL")

        if explicit_base_url:
            # Custom deployment target. Such a target may accept any or
            # an empty key, so don't hard-fail when no API key is provided; fall
            # back to an empty string so the SDK client can still initialize.
            self._api_key = self.base_config.get("api_key") or os.getenv("LLAMA_CLOUD_API_KEY") or ""
            self._base_url = explicit_base_url
        elif use_europe:
            # EU region configuration
            eu_key = self.base_config.get("api_key") or os.getenv("LLAMA_CLOUD_EU_API_KEY")
            self._api_key = eu_key
            if not self._api_key:
                raise ProviderConfigError(
                    "LlamaCloud EU API key is required when use_europe is True. "
                    "Set LLAMA_CLOUD_EU_API_KEY environment variable or "
                    "pass api_key in base_config."
                )
            self._base_url = "https://api.cloud.eu.llamaindex.ai"
        else:
            self._api_key = self.base_config.get("api_key") or os.getenv("LLAMA_CLOUD_API_KEY")
            if not self._api_key:
                raise ProviderConfigError(
                    "LlamaCloud API key is required. "
                    "Set LLAMA_CLOUD_API_KEY environment variable or pass api_key in base_config."
                )
            self._base_url = None  # type: ignore[assignment]  # Use default production URL

        # Build SDK config from user config (excluding provider-only params)
        self._sdk_config: dict[str, Any] = {}
        for k, v in self.base_config.items():
            if k not in self._PROVIDER_ONLY_PARAMS:
                self._sdk_config[k] = v

    @property
    def credit_rate_usd(self) -> float | None:
        return self._credit_rate_usd

    def _image_to_temp_pdf(self, image_path: Path) -> tuple[str, tuple[int, int]]:
        """
        Convert an image file to a temporary PDF.

        :param image_path: Path to the image file
        :return: Tuple of (temp_pdf_path, (width, height))
        """
        # Load image
        image = Image.open(image_path)
        image_size = image.size  # (width, height)

        # Convert to RGB if necessary (PDF doesn't support RGBA)
        if image.mode == "RGBA":
            image = image.convert("RGB")  # type: ignore[assignment]
        elif image.mode != "RGB":
            image = image.convert("RGB")  # type: ignore[assignment]

        # Create temporary PDF file
        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # Save image as PDF
        image.save(temp_path, "PDF", resolution=100.0)

        return temp_path, image_size

    def _output_tables_as_markdown(self) -> bool:
        output_options = self._sdk_config.get("output_options")
        if isinstance(output_options, dict):
            markdown_options = output_options.get("markdown")
            if isinstance(markdown_options, dict):
                table_options = markdown_options.get("tables")
                if isinstance(table_options, dict):
                    markdown_flag = table_options.get("output_tables_as_markdown")
                    if isinstance(markdown_flag, bool):
                        return markdown_flag

        output_tables_as_html = self._sdk_config.get("output_tables_as_HTML")
        if isinstance(output_tables_as_html, bool):
            return not output_tables_as_html

        return False

    def _parse_pdf(self, pdf_path: str) -> dict[str, Any]:
        """
        Parse a PDF using LlamaCloud V2 SDK.

        :param pdf_path: Path to the PDF file
        :return: Raw API response as dictionary
        :raises ProviderError: For any API errors
        """
        job_id: str | None = None
        try:
            # Initialize LlamaCloud client
            client_kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url

            client = LlamaCloud(**client_kwargs)

            # Build V2 parse kwargs
            # Expand "items" (md + text + bboxes per page),
            # "text" (plain text fallback), and "metadata"
            parse_kwargs: dict[str, Any] = {
                "upload_file": pdf_path,
                "expand": ["items", "text", "metadata", "debug_logs"],
                # Default tier and version if not specified
                "tier": self._sdk_config.get("tier", "agentic"),
                "version": self._sdk_config.get("version", "latest"),
                "timeout": self._sdk_config.get("timeout", 600.0),
            }

            # Forward all remaining config keys directly to the V2 SDK
            for key, value in self._sdk_config.items():
                if key in ("tier", "version"):
                    continue  # Already handled above
                parse_kwargs[key] = value

            # Split parse into create + wait + get so we always have the job_id,
            # even when polling or retrieval fails.
            polling_timeout = parse_kwargs.pop("timeout")

            # Separate create-only kwargs from polling/get kwargs
            expand = parse_kwargs.pop("expand")
            create_kwargs = dict(parse_kwargs.items())

            job = client.parsing.create(**create_kwargs)
            job_id = job.id

            client.parsing.wait_for_completion(job_id, timeout=polling_timeout)
            result = client.parsing.get(job_id, expand=expand)
            payload = result.model_dump(mode="json", by_alias=True)

            # Extract debug_logs presigned URL from V2 expand response.
            content_meta = payload.get("result_content_metadata")
            if isinstance(content_meta, dict):
                debug_meta = content_meta.get("debug_logs")
                if isinstance(debug_meta, dict) and debug_meta.get("exists"):
                    presigned_url = debug_meta.get("presigned_url")
                    if isinstance(presigned_url, str) and presigned_url:
                        payload.setdefault("job_logs_url", presigned_url)

            return payload

        except Exception as e:
            # Include job_id in error messages if we got one
            job_id_str = f" (job_id={job_id})" if job_id else ""

            # Check if it's a transient error (network, timeout, etc.)
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str:
                raise ProviderRateLimitError(f"Rate limit exceeded during parsing{job_id_str}: {e}") from e
            transient_keywords = ["timeout", "network", "connection", "503", "502", "504"]
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during parsing{job_id_str}: {e}") from e
            raise ProviderPermanentError(f"Error during parsing{job_id_str}: {e}") from e

    def _fetch_job_logs_descriptor(self, client: "LlamaCloud", job_id: str) -> dict[str, Any] | None:
        """Fetch v1 parse job logs descriptor for a completed job.

        Calls:
          GET /api/v1/parsing/job/{job_id}/read/jobLogs.json

        Returns JSON payload with at least `url` and `expires_at` when available.
        Returns None for 404 / missing payload / transient issues.
        """
        try:
            response = client._client.get(f"/api/v1/parsing/job/{job_id}/read/jobLogs.json")
            if response.status_code == 404:
                return None
            response.raise_for_status()

            payload = response.json()
            if not isinstance(payload, dict):
                return None

            # Ensure dict[str, Any] shape and keep unknown fields for debugging.
            return {str(key): value for key, value in payload.items()}
        except Exception:
            # Logs endpoint should never break core parse execution.
            return None

    def _extract_num_pages(self, raw_output: dict[str, Any]) -> int | None:
        """Infer page count from v2 payload sections."""
        existing_pages = raw_output.get("num_pages")
        if isinstance(existing_pages, (int, float)) and int(existing_pages) > 0:
            return int(existing_pages)

        legacy_pages = raw_output.get("pages")
        if isinstance(legacy_pages, list) and legacy_pages:
            return len(legacy_pages)

        for section_key in ("items", "text", "metadata"):
            section_value = raw_output.get(section_key)
            if isinstance(section_value, dict):
                pages = section_value.get("pages")
                if isinstance(pages, list) and pages:
                    return len(pages)

        return None

    def _extract_token_usage(self, raw_output: dict[str, Any]) -> dict[str, int]:
        """Extract token usage from raw output if available.

        Token data may be present in:
        - usage.input_tokens / usage.output_tokens (common API pattern)
        - statistics.input_tokens / statistics.output_tokens
        - job.usage.input_tokens / job.usage.output_tokens
        - metadata.usage.* fields

        Returns dict with input_tokens, output_tokens, total_tokens if found.
        """
        tokens: dict[str, int] = {}

        # Check common locations for token data
        usage_sources = [
            raw_output.get("usage"),
            raw_output.get("statistics"),
            (raw_output.get("job") or {}).get("usage"),
            (raw_output.get("job") or {}).get("statistics"),
            (raw_output.get("metadata") or {}).get("usage"),
        ]

        for usage in usage_sources:
            if not isinstance(usage, dict):
                continue

            # Try common token field names
            input_keys = ["input_tokens", "prompt_tokens", "inputTokens", "promptTokens"]
            output_keys = ["output_tokens", "completion_tokens", "outputTokens", "completionTokens"]
            total_keys = ["total_tokens", "totalTokens"]

            for key in input_keys:
                val = usage.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    tokens["input_tokens"] = int(val)
                    break

            for key in output_keys:
                val = usage.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    tokens["output_tokens"] = int(val)
                    break

            for key in total_keys:
                val = usage.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    tokens["total_tokens"] = int(val)
                    break

            if tokens:
                break

        # Compute total if we have input and output but not total
        if "input_tokens" in tokens and "output_tokens" in tokens and "total_tokens" not in tokens:
            tokens["total_tokens"] = tokens["input_tokens"] + tokens["output_tokens"]

        return tokens

    def _attach_usage_metadata(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        """Attach bench usage metadata to raw payload for operational stats."""
        output = dict(raw_output)

        num_pages = self._extract_num_pages(output)
        if num_pages and num_pages > 0:
            output.setdefault("num_pages", num_pages)

            tier = self._sdk_config.get("tier", "")
            credits_per_page = self._CREDITS_PER_PAGE.get(str(tier), 0)
            if credits_per_page > 0:
                credits = num_pages * credits_per_page
                output.setdefault("credits_used", credits)
                cost_usd = float(credits) * self._credit_rate_usd
                output.setdefault("cost_usd", cost_usd)
                output.setdefault("cost_per_page_usd", cost_usd / float(num_pages))

        job = output.get("job")
        if isinstance(job, dict):
            job_id = job.get("id")
            if isinstance(job_id, str) and job_id:
                output.setdefault("job_id", job_id)

        # Extract token usage if available
        tokens = self._extract_token_usage(output)

        # Also check embedded token_usage from debug logs (populated by runner)
        token_usage = output.get("token_usage")
        if isinstance(token_usage, dict) and not tokens:
            for key in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens"):
                val = token_usage.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    tokens.setdefault(key, int(val))

        for key, value in tokens.items():
            output.setdefault(key, value)

        # Compute per-page token metrics if we have page count
        if num_pages and num_pages > 0:
            if "input_tokens" in tokens:
                output.setdefault("input_tokens_per_page", tokens["input_tokens"] / num_pages)
            if "output_tokens" in tokens:
                output.setdefault("output_tokens_per_page", tokens["output_tokens"] / num_pages)

        return output

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        :param pipeline: Pipeline specification
        :param request: Inference request
        :return: Raw inference result
        :raises ProviderError: For any provider-related failures
        """
        # Accept both PARSE and LAYOUT_DETECTION product types
        if request.product_type not in (ProductType.PARSE, ProductType.LAYOUT_DETECTION):
            raise ProviderPermanentError(
                f"LlamaParseProvider supports PARSE and LAYOUT_DETECTION product types, got {request.product_type}"
            )

        started_at = datetime.now()

        # Check if file exists
        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")

        # For image files, convert to temporary
        #  temp_pdf_path: str | None = None
        # if source_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".jfif"):
        #    temp_pdf_path, image_size = self._image_to_temp_pdf(source_path)
        #    parse_path = temp_pdf_path
        # else:
        #    parse_path = str(source_path)
        parse_path = str(source_path)

        try:
            # Run parsing with V2 SDK (synchronous)
            raw_output = self._attach_usage_metadata(self._parse_pdf(parse_path))

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
        Normalize raw inference result to produce typed output.

        Dispatches to the appropriate normalization method based on product_type.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        if raw_result.product_type == ProductType.PARSE:
            return self._normalize_parse(raw_result)
        elif raw_result.product_type == ProductType.LAYOUT_DETECTION:
            return self._normalize_layout_detection(raw_result)
        else:
            raise ProviderPermanentError(
                f"LlamaParseProvider supports PARSE and LAYOUT_DETECTION product types, got {raw_result.product_type}"
            )

    def _normalize_parse(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce ParseOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with ParseOutput
        """
        raw_output = self._attach_usage_metadata(raw_result.raw_output)
        try:
            pages = build_pages_from_sdk_response_payload(
                raw_payload=raw_output,
                output_tables_as_markdown=self._output_tables_as_markdown(),
            )
        except ValueError as exc:
            raise ProviderPermanentError(f"Failed to normalize LlamaParse SDK payload for parse output: {exc}") from exc

        output = build_parse_output_from_pages(
            pages_payload=pages,
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            job_id=extract_job_id_from_raw_payload(raw_output),
        )

        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )

    def _normalize_layout_detection(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce LayoutOutput.

        Extracts layout predictions from ALL pages' items[i].layoutAwareBbox.
        Each prediction includes a page number (1-indexed) for multi-page documents.
        Coordinates are in SDK's scaled space and must be scaled to original
        image dimensions for proper evaluation.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with LayoutOutput containing all pages
        """
        raw_output = self._attach_usage_metadata(raw_result.raw_output)
        try:
            pages = build_pages_from_sdk_response_payload(
                raw_payload=raw_output,
                output_tables_as_markdown=self._output_tables_as_markdown(),
            )
        except ValueError as exc:
            raise ProviderPermanentError(
                f"Failed to normalize LlamaParse SDK payload for layout output: {exc}"
            ) from exc

        parse_output = build_parse_output_from_pages(
            pages_payload=pages,
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            job_id=extract_job_id_from_raw_payload(raw_output),
        )
        pages_for_layout = layout_pages_to_legacy_pages_payload(parse_output.layout_pages)

        extraction_input: dict[str, Any] = {"pages": pages_for_layout}
        raw_image_width = raw_output.get("image_width")
        raw_image_height = raw_output.get("image_height")
        if isinstance(raw_image_width, (int, float)) and isinstance(raw_image_height, (int, float)):
            extraction_input["image_width"] = raw_image_width
            extraction_input["image_height"] = raw_image_height

        output = extract_all_layouts_from_llamaparse_output(
            raw_output=extraction_input,
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
        )

        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
