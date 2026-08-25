"""Provider for PyPDF PARSE."""

from datetime import datetime
from pathlib import Path
from typing import Any

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.parse_output import PageIR, ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType


@register_provider("pypdf")
class PyPDFProvider(Provider):
    """
    Provider for PyPDF PARSE.

    Extracts embedded text from PDFs using pypdf library.
    Fast baseline with no OCR capabilities.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration (currently unused)
        """
        super().__init__(provider_name, base_config)

    def _extract_text(self, pdf_path: str) -> dict[str, Any]:
        """
        Extract text from PDF using pypdf.

        :param pdf_path: Path to the PDF file
        :return: Raw extraction result with page-level text
        :raises ProviderError: For any extraction errors
        """
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ProviderConfigError("pypdf package not installed. Run: pip install pypdf") from e

        try:
            reader = PdfReader(pdf_path)
            pages = []

            for page_index, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    pages.append(
                        {
                            "page_index": page_index,
                            "text": text,
                        }
                    )
                except Exception as e:
                    # If one page fails, log but continue
                    pages.append(
                        {
                            "page_index": page_index,
                            "text": "",
                            "error": str(e),
                        }
                    )

            return {
                "pages": pages,
                "num_pages": len(reader.pages),
                "metadata": reader.metadata if hasattr(reader, "metadata") else {},
            }

        except FileNotFoundError as e:
            raise ProviderPermanentError(f"PDF file not found: {pdf_path}") from e
        except Exception as e:
            error_str = str(e).lower()
            # pypdf can fail on encrypted PDFs or corrupted files
            if any(kw in error_str for kw in ["encrypted", "password", "corrupt"]):
                raise ProviderPermanentError(f"Cannot read PDF: {e}") from e
            raise ProviderPermanentError(f"Error extracting text: {e}") from e

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        :param pipeline: Pipeline specification
        :param request: Inference request
        :return: Raw inference result
        :raises ProviderError: For any provider-related failures
        """
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"PyPDFProvider only supports PARSE product type, got {request.product_type}")

        # Check file extension
        pdf_path = Path(request.source_file_path)
        if pdf_path.suffix.lower() != ".pdf":
            raise ProviderPermanentError(f"PyPDFProvider only supports .pdf files, got {pdf_path.suffix}")

        if not pdf_path.exists():
            raise ProviderPermanentError(f"PDF file not found: {pdf_path}")

        started_at = datetime.now()

        try:
            raw_output = self._extract_text(str(pdf_path))
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

        except (ProviderPermanentError, ProviderConfigError):
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
                f"PyPDFProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        # Extract page-level text
        pages: list[PageIR] = []
        page_texts = []

        for page_data in raw_result.raw_output.get("pages", []):
            page_index = page_data.get("page_index", 0)
            text = page_data.get("text", "")

            pages.append(PageIR(page_index=page_index, markdown=text))
            page_texts.append(text)

        # Concatenate all pages
        full_text = "\n\n".join(page_texts)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            markdown=full_text,
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
