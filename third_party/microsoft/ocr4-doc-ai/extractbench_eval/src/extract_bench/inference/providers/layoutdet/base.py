"""Base class for HuggingFace layout detection providers."""

import io
import os
from abc import abstractmethod
from datetime import datetime
from typing import Any

import requests
from PIL import Image

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.schemas.layout_detection_output import (
    LayoutDetectionModel,
    LayoutPrediction,
)
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.read()


class HFLayoutDetProvider(Provider):
    """
    Base class for HuggingFace layout detection inference providers.

    Subclasses must set:
    - endpoint_env_var: Env var naming the deployment to call (class attribute)
    - model_type: The LayoutDetectionModel enum value

    Subclasses must implement:
    - _parse_response(): Convert raw API response to list of predictions
    - normalize(): Convert raw result to LayoutOutput

    These models have no public hosted API. Each provider points at an
    inference endpoint you deploy yourself (HuggingFace Inference Endpoints,
    or any host speaking the same protocol), supplied through the subclass's
    ``endpoint_env_var`` or an ``endpoint_url`` pipeline config key.
    """

    # Subclasses must override these
    endpoint_url: str = ""
    endpoint_env_var: str = ""
    model_type: LayoutDetectionModel

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the HF layout detection provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `endpoint_url`: Self-hosted endpoint (defaults to the subclass's
              `endpoint_env_var` environment variable)
            - `hf_token`: HuggingFace token (defaults to HF_TOKEN env var)
            - `timeout`: Request timeout in seconds (default: 60)
        """
        super().__init__(provider_name, base_config)

        # Self-hosted endpoint: config `endpoint_url`, else the subclass's env
        # var. There is no default deployment to fall back on — these models
        # are not offered as a public hosted API by their authors.
        self.endpoint_url = (
            self.base_config.get("endpoint_url")
            or (os.getenv(self.endpoint_env_var) if self.endpoint_env_var else None)
            or self.endpoint_url
        )
        if not self.endpoint_url:
            env_hint = f" or the {self.endpoint_env_var} environment variable" if self.endpoint_env_var else ""
            raise ProviderConfigError(
                f"{type(self).__name__} requires a self-hosted inference endpoint: set "
                f"`endpoint_url` in the pipeline config{env_hint}."
            )

        # Get HF token
        self._hf_token = self.base_config.get("hf_token") or os.getenv("HF_TOKEN")
        if not self._hf_token:
            raise ProviderConfigError(
                "HuggingFace token is required. Set HF_TOKEN environment variable or pass hf_token in base_config."
            )

        # Get timeout (default 60 seconds)
        self._timeout = self.base_config.get("timeout", 60)

    def _call_endpoint(self, image: Image.Image) -> dict[str, Any]:
        """
        Call the HuggingFace inference endpoint with an image.

        :param image: PIL Image to send
        :return: Raw JSON response from the endpoint
        :raises ProviderError: For any API errors
        """
        if not self.endpoint_url:
            raise ProviderConfigError("Endpoint URL not configured")

        headers = {
            "Authorization": f"Bearer {self._hf_token}",
            "Content-Type": "image/png",
        }

        image_bytes = image_to_bytes(image, format="PNG")

        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                data=image_bytes,
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

        except requests.exceptions.Timeout as e:
            raise ProviderTransientError(f"Request timed out: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ProviderTransientError(f"Connection error: {e}") from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            if status_code == 429:
                raise ProviderRateLimitError(f"Rate limit exceeded: {e}") from e
            elif status_code and 500 <= status_code < 600:
                raise ProviderTransientError(f"Server error ({status_code}): {e}") from e
            elif status_code and 400 <= status_code < 500:
                raise ProviderPermanentError(f"Client error ({status_code}): {e}") from e
            else:
                raise ProviderPermanentError(f"HTTP error: {e}") from e
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error calling endpoint: {e}") from e

    @abstractmethod
    def _parse_response(self, response: dict[str, Any]) -> list[LayoutPrediction]:
        """
        Parse the raw API response into a list of layout predictions.

        Subclasses must implement this to handle model-specific response formats.

        :param response: Raw JSON response from the endpoint
        :return: List of layout predictions
        """
        raise NotImplementedError("Subclasses must implement _parse_response")

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run layout detection inference on an image.

        :param pipeline: Pipeline specification
        :param request: Inference request (source_file_path should be an image)
        :return: Raw inference result
        :raises ProviderError: For any provider-related failures
        """
        if request.product_type != ProductType.LAYOUT_DETECTION:
            raise ProviderPermanentError(
                f"{self.__class__.__name__} only supports LAYOUT_DETECTION product type, got {request.product_type}"
            )

        started_at = datetime.now()

        # Load the image
        try:
            image = Image.open(request.source_file_path)
            # Ensure image is in RGB mode for PNG conversion
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")  # type: ignore[assignment]
        except Exception as e:
            raise ProviderPermanentError(f"Failed to load image: {e}") from e

        # Get image dimensions
        image_width, image_height = image.size

        # Call the endpoint
        raw_response = self._call_endpoint(image)

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        # Store image dimensions in raw output for normalization
        raw_output = {
            "response": raw_response,
            "image_width": image_width,
            "image_height": image_height,
        }

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

    @abstractmethod
    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce LayoutOutput.

        Each subclass MUST implement this method to return normalized
        provider predictions in the common layout interface.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        raise NotImplementedError("Subclasses must implement normalize")
