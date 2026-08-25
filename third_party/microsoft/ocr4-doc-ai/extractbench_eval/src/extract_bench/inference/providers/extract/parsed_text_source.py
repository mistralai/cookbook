"""Pluggable parse-stage text sources for two-stage (parse → text extract) pipelines.

Two-stage direct-model extraction first converts the document into text via a
parse stage, then sends that text (instead of the raw file) to the model's
structured-output endpoint. The parse stage is pluggable so future experiments
can swap the parsing approach per-pipeline without touching the model
providers: implement ``ParseTextSource`` and route a new ``type`` value to it
in ``create_parse_text_source``.

Pipeline config shape (under the extract provider's ``parse_source`` key)::

    {"type": "llamaparse", "tier": "agentic", "version": "latest", ...}

Every key besides ``type`` is forwarded to the underlying parse provider's
base_config, so LlamaParse options (``api_key``,
``processing_options``, ...) work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from extract_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
)
from extract_bench.schemas.parse_output import ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType


@dataclass
class ParsedDocumentText:
    """Text rendering of a parsed document plus parse-stage accounting."""

    text: str
    num_pages: int
    parse_cost_usd: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ParseTextSource(Protocol):
    """Parse stage interface: document file → text for the extract model."""

    def parse(self, request: InferenceRequest) -> ParsedDocumentText: ...


def render_paged_markdown(pages: list[tuple[int, str]]) -> str:
    """Join per-page markdown with explicit page markers.

    ``pages`` is a list of ``(page_number, markdown)`` with 1-based page
    numbers. HTML-comment markers keep pagination visible to the model without
    colliding with document content rendered as markdown headings.
    """
    return "\n\n".join(f"<!-- Page {page_num} -->\n\n{markdown.strip()}" for page_num, markdown in pages)


class LlamaParseTextSource:
    """Runs LlamaParse and renders the result as per-page markdown."""

    DEFAULT_PARSE_CONFIG: dict[str, Any] = {
        "tier": "agentic",
        "version": "latest",
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self._parse_config: dict[str, Any] = {**self.DEFAULT_PARSE_CONFIG, **(config or {})}

    @property
    def parse_config(self) -> dict[str, Any]:
        return dict(self._parse_config)

    def parse(self, request: InferenceRequest) -> ParsedDocumentText:
        # Imported here so constructing extract providers doesn't require the
        # llama-cloud SDK unless the parsed_text input mode is actually used.
        from extract_bench.inference.providers.parse.llamaparse import LlamaParseProvider

        parse_provider = LlamaParseProvider("llamaparse", base_config=self.parse_config)
        parse_pipeline = PipelineSpec(
            pipeline_name="parsed_text_source__llamaparse",
            provider_name="llamaparse",
            product_type=ProductType.PARSE,
            config=self.parse_config,
        )
        parse_request = InferenceRequest(
            example_id=request.example_id,
            source_file_path=request.source_file_path,
            product_type=ProductType.PARSE,
        )
        raw_parse = parse_provider.run_inference(parse_pipeline, parse_request)
        parse_result = parse_provider.normalize(raw_parse)
        parse_output = parse_result.output
        if not isinstance(parse_output, ParseOutput):
            raise ProviderPermanentError(
                f"LlamaParseTextSource: expected ParseOutput from LlamaParse, got {type(parse_output).__name__}."
            )

        page_texts = [(p.page_index + 1, p.markdown) for p in parse_output.pages]
        if any(markdown.strip() for _, markdown in page_texts):
            text = render_paged_markdown(page_texts)
        elif parse_output.markdown.strip():
            text = parse_output.markdown
        else:
            raise ProviderPermanentError(
                f"LlamaParseTextSource: parse produced no text for {request.example_id} — nothing to extract from."
            )

        raw_output = raw_parse.raw_output
        num_pages = int(raw_output.get("num_pages") or len(parse_output.pages))
        metadata: dict[str, Any] = {
            "type": "llamaparse",
            "tier": self._parse_config.get("tier"),
            "version": self._parse_config.get("version"),
            "num_pages": num_pages,
        }
        for key in ("job_id", "credits_used"):
            if raw_output.get(key) is not None:
                metadata[key] = raw_output[key]

        return ParsedDocumentText(
            text=text,
            num_pages=num_pages,
            parse_cost_usd=float(raw_output.get("cost_usd") or 0.0),
            metadata=metadata,
        )


def create_parse_text_source(config: dict[str, Any] | None) -> ParseTextSource:
    """Build a ``ParseTextSource`` from pipeline config (default: LlamaParse agentic)."""
    cfg = dict(config or {})
    source_type = cfg.pop("type", "llamaparse")
    if source_type == "llamaparse":
        return LlamaParseTextSource(cfg)
    raise ProviderConfigError(
        f"Unknown parse_source type {source_type!r}. Available: 'llamaparse'. "
        "Add new types in parsed_text_source.create_parse_text_source."
    )
