"""Provider for PyMuPDF PARSE."""

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


@register_provider("pymupdf")
class PyMuPDFProvider(Provider):
    """
    Provider for PyMuPDF PARSE.

    Extracts embedded text from PDFs using PyMuPDF (fitz) library.
    Alternative to PyPDF for comparison benchmarking.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `text_format`: "text", "html", "dict", "json", "rawdict", "xml" (default: "text")
            - `flags`: Text extraction flags as integer (default: 0)
        """
        super().__init__(provider_name, base_config)
        self._text_format = self.base_config.get("text_format", "text")
        self._flags = self.base_config.get("flags", 0)

    def _extract_text(self, pdf_path: str) -> dict[str, Any]:
        """
        Extract text from PDF using PyMuPDF.

        :param pdf_path: Path to the PDF file
        :return: Raw extraction result with page-level text
        :raises ProviderError: For any extraction errors
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ProviderConfigError("pymupdf package not installed. Run: pip install pymupdf") from e

        try:
            doc = fitz.open(pdf_path)
            pages = []

            for page_index in range(len(doc)):
                page = doc[page_index]
                try:
                    # Extract text based on format
                    if self._text_format == "text":
                        text = page.get_text("text", flags=self._flags)
                    elif self._text_format == "html":
                        text = page.get_text("html", flags=self._flags)
                    elif self._text_format == "dict":
                        text = str(page.get_text("dict", flags=self._flags))
                    elif self._text_format == "json":
                        text = page.get_text("json", flags=self._flags)
                    elif self._text_format == "rawdict":
                        text = str(page.get_text("rawdict", flags=self._flags))
                    elif self._text_format == "xml":
                        text = page.get_text("xml", flags=self._flags)
                    else:
                        text = page.get_text("text", flags=self._flags)

                    pages.append(
                        {
                            "page_index": page_index,
                            "text": text,
                            "width": page.rect.width,
                            "height": page.rect.height,
                        }
                    )
                except Exception as e:
                    pages.append(
                        {
                            "page_index": page_index,
                            "text": "",
                            "error": str(e),
                        }
                    )

            # Get metadata
            metadata = doc.metadata or {}

            doc.close()

            return {
                "pages": pages,
                "num_pages": len(pages),
                "metadata": metadata,
                "text_format": self._text_format,
            }

        except FileNotFoundError as e:
            raise ProviderPermanentError(f"PDF file not found: {pdf_path}") from e
        except Exception as e:
            error_str = str(e).lower()
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
            raise ProviderPermanentError(
                f"PyMuPDFProvider only supports PARSE product type, got {request.product_type}"
            )

        # Check file extension
        pdf_path = Path(request.source_file_path)
        if pdf_path.suffix.lower() != ".pdf":
            raise ProviderPermanentError(f"PyMuPDFProvider only supports .pdf files, got {pdf_path.suffix}")

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
                f"PyMuPDFProvider only supports PARSE product type, got {raw_result.product_type}"
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
