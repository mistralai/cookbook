"""Two-stage Mistral OCR-4 extraction provider.

Stage 1 — Parse: Mistral OCR-4 (mistral-ocr-4-0) via Azure AI Foundry converts
the document to per-page markdown, with tables inlined.

Stage 2 — Extract: Mistral chat API (mistral-large-latest or mistral-medium-3-5)
uses the markdown text + JSON schema response_format to produce structured output.

Environment variables
---------------------
AZURE_MISTRAL_DOCUMENT_AI_KEY       Bearer key for the Azure AI Foundry OCR-4 endpoint
AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT  Full URL of the OCR endpoint, e.g.:
                                    https://<resource>.services.ai.azure.com/providers/mistral/azure/ocr
MISTRAL_API_KEY                     API key for the Mistral chat API (extract stage)

These can also be passed per-pipeline via the config dict (keys: ``ocr_api_key``,
``ocr_endpoint``, ``extract_api_key``).
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.direct_model_utils import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_INSTRUCTION,
    IMAGE_EXTENSIONS,
    add_additional_properties_false,
    normalize_extract_result,
    promote_repeated_structure,
)
from extract_bench.inference.providers.extract.parsed_text_source import (
    ParsedDocumentText,
    render_paged_markdown,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType

_MISTRAL_EXTRACT_API_URL = "https://api.mistral.ai/v1/chat/completions"

_MISTRAL_EXTRACT_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "mistral-large": (2.00, 6.00),
    "mistral-medium-3-5": (0.40, 2.00),
    "mistral-small": (0.10, 0.30),
}


def _pricing_for_extract_model(model: str) -> tuple[float, float]:
    matches = [(p, r) for p, r in _MISTRAL_EXTRACT_PRICING_PER_M.items() if model.startswith(p)]
    return max(matches, key=lambda x: len(x[0]))[1] if matches else (0.0, 0.0)


@register_provider("mistral_ocr4_extract")
class MistralOCR4ExtractProvider(Provider):
    """Two-stage document extraction: Mistral OCR-4 parse → Mistral LLM extract."""

    DEFAULT_EXTRACT_MODEL = "mistral-large-latest"

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        self._ocr_api_key = (
            self.base_config.get("ocr_api_key")
            or os.getenv("AZURE_MISTRAL_DOCUMENT_AI_KEY")
            or os.getenv("AZURE_API_KEY")
        )
        self._ocr_endpoint = (
            self.base_config.get("ocr_endpoint")
            or os.getenv("AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT")
            or os.getenv("AZURE_OCR_ENDPOINT")
        )
        if not self._ocr_api_key:
            raise ProviderConfigError(
                "OCR-4 Azure API key is required. Set AZURE_MISTRAL_DOCUMENT_AI_KEY or pass ocr_api_key in config."
            )
        if not self._ocr_endpoint:
            raise ProviderConfigError(
                "OCR-4 Azure endpoint is required. Set AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT or pass ocr_endpoint in config."
            )

        self._extract_api_key = self.base_config.get("extract_api_key") or os.getenv("MISTRAL_API_KEY")
        if not self._extract_api_key:
            raise ProviderConfigError(
                "Mistral API key is required for the extract stage. Set MISTRAL_API_KEY or pass extract_api_key in config."
            )

        self._extract_model: str = self.base_config.get("extract_model", self.DEFAULT_EXTRACT_MODEL)
        self._max_tokens: int = int(self.base_config.get("max_tokens", 32768))
        self._system_prompt: str = self.base_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self._user_instruction: str = self.base_config.get("user_instruction", DEFAULT_USER_INSTRUCTION)
        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))

        in_price, out_price = _pricing_for_extract_model(self._extract_model)
        self._input_price_per_1m: float = float(self.base_config.get("input_price_per_1m", in_price))
        self._output_price_per_1m: float = float(self.base_config.get("output_price_per_1m", out_price))

    def _prepare_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        schema = promote_repeated_structure(schema)
        if self._additional_properties_false:
            schema = add_additional_properties_false(schema)
        return schema

    # Azure AI Foundry OCR-4 accepts at most 30 pages per request.
    _OCR_PAGE_LIMIT = 30

    def _ocr_chunk(self, data: bytes, mime: str, page_offset: int) -> list[tuple[int, str]]:
        """Send one chunk (≤30 pages) to OCR-4 and return (1-based-page, markdown) tuples."""
        encoded = base64.standard_b64encode(data).decode("utf-8")
        payload = {
            "model": "mistral-ocr-4-0",
            "document": {
                "type": "document_url",
                "document_url": f"data:{mime};base64,{encoded}",
            },
        }
        headers = {
            "Authorization": f"Bearer {self._ocr_api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = httpx.post(self._ocr_endpoint, json=payload, headers=headers, timeout=300.0)
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise ProviderTransientError(f"OCR-4 request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 503, 502, 504):
                raise ProviderTransientError(f"OCR-4 transient HTTP error {e.response.status_code}: {e}") from e
            body = getattr(e.response, "text", "")[:200]
            raise ProviderPermanentError(f"OCR-4 HTTP error {e.response.status_code}: {body}") from e

        pages = resp.json().get("pages", [])
        result: list[tuple[int, str]] = []
        for i, p in enumerate(pages):
            text = p.get("markdown", "")
            for tbl in (p.get("tables") or []):
                tid = tbl.get("id", "")
                tcnt = (tbl.get("content") or "").strip()
                if tid and tcnt:
                    text = text.replace(f"[{tid}]({tid})", tcnt)
            result.append((page_offset + p.get("index", i) + 1, text))
        return result

    def _ocr_document(self, file_path: Path) -> ParsedDocumentText:
        """Run Mistral OCR-4 on a PDF or image; return per-page markdown text.

        PDFs longer than 30 pages are split into chunks before sending because
        the Azure AI Foundry OCR-4 endpoint enforces a 30-page limit per request.
        """
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            mime = "application/pdf"
        elif ext in IMAGE_EXTENSIONS:
            mime = IMAGE_EXTENSIONS[ext]
        else:
            raise ProviderPermanentError(
                f"MistralOCR4: supports PDF and {set(IMAGE_EXTENSIONS)}, got {ext!r}"
            )

        page_tuples: list[tuple[int, str]] = []

        if ext == ".pdf":
            from io import BytesIO
            from pypdf import PdfReader, PdfWriter

            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)

            for start in range(0, total_pages, self._OCR_PAGE_LIMIT):
                end = min(start + self._OCR_PAGE_LIMIT, total_pages)
                writer = PdfWriter()
                for page_idx in range(start, end):
                    writer.add_page(reader.pages[page_idx])
                buf = BytesIO()
                writer.write(buf)
                chunk_bytes = buf.getvalue()
                page_tuples.extend(self._ocr_chunk(chunk_bytes, mime, start))
        else:
            page_tuples.extend(self._ocr_chunk(file_path.read_bytes(), mime, 0))

        full_text = render_paged_markdown(page_tuples) if page_tuples else ""
        if not full_text.strip():
            raise ProviderPermanentError(f"OCR-4 produced no text for {file_path.name}")

        num_pages = len(page_tuples) or 1

        return ParsedDocumentText(
            text=full_text,
            num_pages=num_pages,
            parse_cost_usd=0.0,
            metadata={
                "type": "mistral_ocr4",
                "model": "mistral-ocr-4-0",
                "num_pages": num_pages,
            },
        )

    def _extract_structured(self, document_text: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Call Mistral chat API with JSON schema response_format to extract structured data."""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": document_text + "\n\n" + self._user_instruction,
            },
        ]
        payload: dict[str, Any] = {
            "model": self._extract_model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        headers = {
            "Authorization": f"Bearer {self._extract_api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = httpx.post(_MISTRAL_EXTRACT_API_URL, json=payload, headers=headers, timeout=300.0)
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise ProviderTransientError(f"Mistral extract request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 503, 502, 504):
                raise ProviderTransientError(
                    f"Mistral extract transient HTTP error {e.response.status_code}: {e}"
                ) from e
            raise ProviderPermanentError(
                f"Mistral extract HTTP error {e.response.status_code}: {e.response.text[:500]}"
            ) from e

        result = resp.json()
        raw_text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            s, e_idx = raw_text.find("{"), raw_text.rfind("}") + 1
            if s >= 0 and e_idx > s:
                try:
                    data = json.loads(raw_text[s:e_idx])
                except json.JSONDecodeError as exc:
                    raise ProviderPermanentError(
                        f"Mistral extract returned non-JSON despite structured-output request: {exc}"
                    ) from exc
            else:
                raise ProviderPermanentError(
                    f"Mistral extract returned non-JSON despite structured-output request: {raw_text[:200]}"
                )

        return {
            "data": data,
            "model": self._extract_model,
            "usage": usage,
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"MistralOCR4ExtractProvider only supports EXTRACT, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type."
            )

        started_at = datetime.now()
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        schema = self._prepare_schema(request.schema_override)

        try:
            parsed_doc = self._ocr_document(file_path)
            raw_output = self._extract_structured(parsed_doc.text, schema)
        except (ProviderPermanentError, ProviderTransientError, ProviderConfigError):
            raise
        except Exception as e:
            error_str = str(e).lower()
            transient_keywords = ("timeout", "network", "connection", "503", "502", "504", "429", "rate limit")
            if any(k in error_str for k in transient_keywords):
                raise ProviderTransientError(f"Transient error during OCR-4 extraction: {e}") from e
            raise ProviderPermanentError(f"Error during OCR-4 extraction: {e}") from e

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        usage = raw_output["usage"]
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        extract_cost_usd = (
            input_tokens / 1_000_000 * self._input_price_per_1m
            + output_tokens / 1_000_000 * self._output_price_per_1m
        )
        cost_usd = extract_cost_usd + parsed_doc.parse_cost_usd
        num_pages = parsed_doc.num_pages if parsed_doc.num_pages > 0 else 1

        raw_output.update(
            {
                "extract_cost_usd": extract_cost_usd,
                "parse_cost_usd": parsed_doc.parse_cost_usd,
                "parse_metadata": parsed_doc.metadata,
                "parsed_text": parsed_doc.text,
                "num_pages": num_pages,
                "cost_usd": cost_usd,
                "_config": {
                    "extract_model": self._extract_model,
                    "max_tokens": self._max_tokens,
                    "additional_properties_false": self._additional_properties_false,
                    "input_price_per_1m": self._input_price_per_1m,
                    "output_price_per_1m": self._output_price_per_1m,
                },
            }
        )
        if num_pages > 0:
            raw_output["cost_per_page_usd"] = cost_usd / num_pages

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

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        return normalize_extract_result(raw_result)
