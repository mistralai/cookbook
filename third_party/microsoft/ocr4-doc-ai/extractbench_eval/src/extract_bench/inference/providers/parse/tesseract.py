"""Provider for Tesseract OCR PARSE."""

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
from extract_bench.schemas.parse_output import PageIR, ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType


@register_provider("tesseract")
class TesseractProvider(Provider):
    """
    Provider for Tesseract OCR PARSE.

    Performs OCR on PDF pages and images using Tesseract.
    Handles scanned documents where embedded text is not available.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `lang`: Tesseract language code (default: "eng")
            - `config`: Tesseract config string (default: "")
            - `dpi`: DPI for PDF to image conversion (default: 300)
            - `output_type`: Tesseract output type -
              "text", "dict", "data", "boxes", "osd"
              (default: "text")
        """
        super().__init__(provider_name, base_config)
        self._lang = self.base_config.get("lang", "eng")
        self._config = self.base_config.get("config", "")
        self._dpi = self.base_config.get("dpi", 300)
        self._output_type = self.base_config.get("output_type", "text")

    def _ocr_pdf(self, pdf_path: str) -> dict[str, Any]:
        """
        Perform OCR on PDF pages.

        :param pdf_path: Path to the PDF file
        :return: Raw OCR result with page-level text
        :raises ProviderError: For any OCR errors
        """
        try:
            import pytesseract
            from pdf2image import convert_from_path
        except ImportError as e:
            missing_pkg = "pytesseract" if "pytesseract" in str(e) else "pdf2image"
            raise ProviderConfigError(
                f"{missing_pkg} package not installed. Run: pip install pytesseract pdf2image"
            ) from e

        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path, dpi=self._dpi)

            pages = []
            for page_index, image in enumerate(images):
                try:
                    # Perform OCR based on output type
                    if self._output_type == "text":
                        text = pytesseract.image_to_string(image, lang=self._lang, config=self._config)
                    elif self._output_type == "dict":
                        data = pytesseract.image_to_data(
                            image,
                            lang=self._lang,
                            config=self._config,
                            output_type=pytesseract.Output.DICT,
                        )
                        text = " ".join([word for word in data.get("text", []) if word.strip()])
                    elif self._output_type == "data":
                        text = pytesseract.image_to_data(image, lang=self._lang, config=self._config)
                    elif self._output_type == "boxes":
                        text = pytesseract.image_to_boxes(image, lang=self._lang, config=self._config)
                    elif self._output_type == "osd":
                        text = pytesseract.image_to_osd(image, config=self._config)
                    else:
                        text = pytesseract.image_to_string(image, lang=self._lang, config=self._config)

                    pages.append(
                        {
                            "page_index": page_index,
                            "text": text,
                            "width": image.width,
                            "height": image.height,
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

            return {
                "pages": pages,
                "num_pages": len(images),
                "config": {
                    "lang": self._lang,
                    "dpi": self._dpi,
                    "output_type": self._output_type,
                },
            }

        except FileNotFoundError as e:
            raise ProviderPermanentError(f"PDF file not found: {pdf_path}") from e
        except Exception as e:
            error_str = str(e).lower()
            # Check for transient errors
            if any(kw in error_str for kw in ["timeout", "memory", "resource"]):
                raise ProviderTransientError(f"Transient error during OCR: {e}") from e
            # Check for Tesseract installation issues
            if "tesseract" in error_str and any(kw in error_str for kw in ["not found", "not installed", "command"]):
                raise ProviderConfigError(
                    "Tesseract OCR engine not found. Please install Tesseract: "
                    "https://github.com/tesseract-ocr/tesseract"
                ) from e
            raise ProviderPermanentError(f"Error during OCR: {e}") from e

    def _ocr_image(self, image_path: str) -> dict[str, Any]:
        """
        Perform OCR on a single image.

        :param image_path: Path to the image file
        :return: Raw OCR result
        :raises ProviderError: For any OCR errors
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:
            missing_pkg = "pytesseract" if "pytesseract" in str(e) else "Pillow"
            raise ProviderConfigError(
                f"{missing_pkg} package not installed. Run: pip install pytesseract Pillow"
            ) from e

        try:
            image = Image.open(image_path)

            # Perform OCR
            if self._output_type == "text":
                text = pytesseract.image_to_string(image, lang=self._lang, config=self._config)
            elif self._output_type == "dict":
                data = pytesseract.image_to_data(
                    image, lang=self._lang, config=self._config, output_type=pytesseract.Output.DICT
                )
                text = " ".join([word for word in data.get("text", []) if word.strip()])
            elif self._output_type == "data":
                text = pytesseract.image_to_data(image, lang=self._lang, config=self._config)
            elif self._output_type == "boxes":
                text = pytesseract.image_to_boxes(image, lang=self._lang, config=self._config)
            elif self._output_type == "osd":
                text = pytesseract.image_to_osd(image, config=self._config)
            else:
                text = pytesseract.image_to_string(image, lang=self._lang, config=self._config)

            return {
                "pages": [
                    {
                        "page_index": 0,
                        "text": text,
                        "width": image.width,
                        "height": image.height,
                    }
                ],
                "num_pages": 1,
                "config": {
                    "lang": self._lang,
                    "output_type": self._output_type,
                },
            }

        except FileNotFoundError as e:
            raise ProviderPermanentError(f"Image file not found: {image_path}") from e
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["timeout", "memory", "resource"]):
                raise ProviderTransientError(f"Transient error during OCR: {e}") from e
            if "tesseract" in error_str and any(kw in error_str for kw in ["not found", "not installed", "command"]):
                raise ProviderConfigError(
                    "Tesseract OCR engine not found. Please install Tesseract: "
                    "https://github.com/tesseract-ocr/tesseract"
                ) from e
            raise ProviderPermanentError(f"Error during OCR: {e}") from e

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
                f"TesseractProvider only supports PARSE product type, got {request.product_type}"
            )

        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")

        # Check file extension
        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}
        if source_path.suffix.lower() not in supported_extensions:
            raise ProviderPermanentError(
                f"TesseractProvider only supports {supported_extensions}, got {source_path.suffix}"
            )

        started_at = datetime.now()

        try:
            # Route to appropriate OCR method
            if source_path.suffix.lower() == ".pdf":
                raw_output = self._ocr_pdf(str(source_path))
            else:
                raw_output = self._ocr_image(str(source_path))

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

        except (ProviderPermanentError, ProviderTransientError, ProviderConfigError):
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
                f"TesseractProvider only supports PARSE product type, got {raw_result.product_type}"
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
