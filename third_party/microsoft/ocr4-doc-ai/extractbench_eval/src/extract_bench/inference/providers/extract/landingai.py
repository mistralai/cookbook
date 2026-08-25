"""Provider for Landing AI EXTRACT."""

import json
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from landingai_ade import LandingAIADE

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.citations import extract_landingai_field_citations
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType


def _to_plain_data(value: Any) -> Any:
    """Convert SDK response objects into JSON-serializable Python data."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {k: _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(v) for v in value]
    return value


def _coerce_positive(value: Any) -> float | None:
    """Return ``value`` as a positive float, or ``None`` if non-positive/invalid."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


@register_provider("landingai_extract")
class LandingAIExtractProvider(Provider):
    """
    Provider for Landing AI EXTRACT.

    This provider uses the Landing AI ADE API for extraction tasks.
    """

    # LandingAI Extract runs parse -> extract, and both calls bill credits. We
    # sum them and convert to USD at the Explore-plan rate, matching
    # ``LandingAIParseProvider``. See docs.landing.ai/ade/ade-pricing.
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
            - `model`: Extraction model version (optional)
            - `parse_model`: Model to use for parsing before extraction (default: "dpt-2-latest")
            - Any other extract parameters from Landing AI API
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
        self._model = self.base_config.get("model")

        # Initialize client
        self._client = LandingAIADE()

    def _extract_from_document(self, document_path: Path, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Extract data from a document using Landing AI API.

        Landing AI extract works with parse responses or markdown files.
        We first parse the document, then extract from the parse response.

        :param document_path: Path to the document file
        :param schema: JSON schema for extraction
        :return: Raw API response as dictionary
        :raises ProviderError: For any API errors
        """
        try:
            # First, parse the document
            parse_model = self.base_config.get("parse_model", "dpt-2-latest")
            parse_response = self._client.parse(
                document=document_path,
                model=parse_model,
            )

            # Get markdown from parse response
            markdown = parse_response.markdown if hasattr(parse_response, "markdown") else ""

            # Build extract parameters
            extract_kwargs = {}
            if self._model:
                extract_kwargs["model"] = self._model

            preserve_grounding = bool(self.base_config.get("preserve_grounding", False))

            # Add any other extract config parameters (excluding provider-only options)
            for k, v in self.base_config.items():
                if k not in ["api_key", "model", "parse_model", "preserve_grounding"]:
                    extract_kwargs[k] = v

            # Convert schema to JSON string if it's a dict
            schema_json = json.dumps(schema) if isinstance(schema, dict) else schema

            # Extract data from markdown using the extract method
            response = self._client.extract(schema=schema_json, markdown=markdown, **extract_kwargs)

            # Convert response to dictionary format
            result: dict[str, Any] = {
                "data": {},
            }

            # Extract the data from response
            # The extract method returns data in response.extraction
            if hasattr(response, "extraction"):
                extraction = response.extraction
                if hasattr(extraction, "model_dump"):
                    result["data"] = extraction.model_dump()
                elif hasattr(extraction, "dict"):
                    result["data"] = extraction.dict()
                elif isinstance(extraction, dict):
                    result["data"] = extraction
                else:
                    result["data"] = extraction
            elif hasattr(response, "data"):
                # Fallback to data attribute if extraction doesn't exist
                data = response.data
                if hasattr(data, "model_dump"):
                    result["data"] = data.model_dump()
                elif hasattr(data, "dict"):
                    result["data"] = data.dict()
                elif isinstance(data, dict):
                    result["data"] = data
                else:
                    result["data"] = data

            if preserve_grounding:
                # Preserve grounding-related metadata when the SDK exposes it. LandingAI
                # Extract can reference parse chunks; parse grounding carries the bboxes.
                for attr in ("extraction_metadata", "metadata", "job_id", "credits_used"):
                    if hasattr(response, attr):
                        result[attr] = _to_plain_data(getattr(response, attr))

                parse_payload: dict[str, Any] = {}
                for attr in ("grounding", "chunks", "metadata", "markdown"):
                    if hasattr(parse_response, attr):
                        parse_payload[attr] = _to_plain_data(getattr(parse_response, attr))
                if parse_payload:
                    result["parse_response"] = parse_payload
            else:
                if hasattr(response, "job_id"):
                    result["job_id"] = response.job_id
                if hasattr(response, "credits_used"):
                    result["credits_used"] = response.credits_used

            self._apply_cost_fields(result, parse_response, response)

            return result

        except Exception as e:
            # Classify the error: rate limits and transient network errors are
            # retryable; everything else is permanent.
            error_str = str(e).lower()
            rate_limit_keywords = ["429", "rate limit", "rate_limit", "too many requests"]
            transient_keywords = ["timeout", "network", "connection", "503", "502", "504"]
            if any(keyword in error_str for keyword in rate_limit_keywords):
                raise ProviderRateLimitError(f"Rate limited during extraction: {e}") from e
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during extraction: {e}") from e
            raise ProviderPermanentError(f"Error during extraction: {e}") from e

    def _apply_cost_fields(self, result: dict[str, Any], parse_response: Any, extract_response: Any) -> None:
        """Attach credit and USD cost fields to the raw output.

        LandingAI Extract runs as parse -> extract and both steps bill credits.
        Each SDK response reports its own credits under ``metadata.credit_usage``
        -- ``ExtractResponse`` has no top-level ``credits_used`` field -- while the
        page count is only on the parse metadata. We sum parse + extract credits
        so the reported cost reflects the full pipeline, then convert to USD at
        :attr:`CREDIT_RATE_USD`. When no credits are reported the cost fields are
        left unset rather than defaulting to zero.
        """
        parse_credits = None
        num_pages = None
        parse_meta = getattr(parse_response, "metadata", None)
        if parse_meta is not None:
            parse_credits = _coerce_positive(getattr(parse_meta, "credit_usage", None))
            num_pages = _coerce_positive(getattr(parse_meta, "page_count", None))

        # ExtractResponse exposes its credits on metadata.credit_usage (there is
        # no top-level ``credits_used`` attribute), same shape as parse metadata.
        extract_credits = None
        extract_meta = getattr(extract_response, "metadata", None)
        if extract_meta is not None:
            extract_credits = _coerce_positive(getattr(extract_meta, "credit_usage", None))

        if parse_credits is not None:
            result["parse_credits"] = parse_credits
        if extract_credits is not None:
            result["extract_credits_used"] = extract_credits

        total_credits = (parse_credits or 0.0) + (extract_credits or 0.0)
        if total_credits <= 0:
            return

        cost_usd = total_credits * self.CREDIT_RATE_USD
        result["credits_used"] = total_credits
        result["cost_usd"] = cost_usd
        if num_pages is not None:
            result["num_pages"] = num_pages
            result["cost_per_page_usd"] = cost_usd / num_pages

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        :param pipeline: Pipeline specification
        :param request: Inference request (must include schema_override for EXTRACT)
        :return: Raw inference result
        :raises ProviderError: For any provider-related failures
        """
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"LandingAIExtractProvider only supports EXTRACT product type, got {request.product_type}"
            )

        # Schema is required for extraction
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override"
            )

        started_at = datetime.now()

        # Check if file exists
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        try:
            # Run extraction
            raw_output = self._extract_from_document(
                document_path=file_path,
                schema=request.schema_override,
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
        Normalize raw inference result to produce ExtractOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        if raw_result.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"LandingAIExtractProvider only supports EXTRACT product type, got {raw_result.product_type}"
            )

        # Extract the structured data from the response
        extracted_data = raw_result.raw_output.get("data", {})

        # LandingAI's API ships chunk-level grounding for cover-page scalars
        # (one bbox shared across many fields). Pass pdf_path + extracted_data
        # so the citation builder can refine those coarse bboxes by text-
        # searching for the value within the chunk rectangle. See
        # ``extract_landingai_field_citations`` for details.
        refinement_data: Mapping[str, Any] | None = extracted_data if isinstance(extracted_data, Mapping) else None
        output = ExtractOutput(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data,
            field_citations=extract_landingai_field_citations(
                raw_result.raw_output,
                pdf_path=raw_result.request.source_file_path,
                extracted_data=refinement_data,
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
