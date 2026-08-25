"""Table→schema code-generation extraction provider.

Runs **LlamaParse Agentic first**, then has a model author a Python extractor
script over the parsed document's tables + the (ref-resolved) output schema; the
script is gated for validity and run over the full parsed document to produce the
extraction. Distinct from ``claude_code_extract`` (which hands the agent the raw
PDF): here the document is pre-parsed into a compressed table/prose context, the
schema is shown with $ref/anyOf resolved, and a validity gate enforces the shape
— so the provider drives its own tool-loop rather than shelling the CLI.

Internals (parser, document, schema utils, agent loop, harness) live in the
``table_codegen/`` subpackage.
"""

from __future__ import annotations

import json
import logging
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
from extract_bench.inference.providers.extract.direct_model_utils import (
    IMAGE_EXTENSIONS,
    normalize_extract_result,
    pricing_for_model,
)
from extract_bench.inference.providers.parse.llamaparse import LlamaParseProvider
from extract_bench.inference.providers.parse.llamaparse_v2_normalization import (
    extract_job_id_from_raw_payload,
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
from extract_bench.utils.gemini_pricing import (
    GEMINI_PRICING_PER_MILLION,
    gemini_context_cache_pricing_per_million,
)

from .table_codegen import Document, ItemsDocument, generate_extractor

logger = logging.getLogger(__name__)

# USD per 1M tokens (input, output). Configurable via base_config["pricing"].
# Gemini rates come from the canonical in-repo table (utils.gemini_pricing) — do
# not duplicate the numbers here so the two paths can't drift.
_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    # Claude Sonnet 5 introductory pricing ($2/$10 per MTok) is in effect through
    # 2026-08-31; it reverts to the standard $3/$15 after that — bump these then.
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-5.4": (2.5, 15.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    **GEMINI_PRICING_PER_MILLION,
}


def _provider_for_model(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt"):
        return "openai"
    if model.startswith("gemini"):
        return "google"
    raise ProviderConfigError(
        f"table_codegen_extract: cannot infer provider for model {model!r} (expected claude*/gpt*/gemini*)."
    )


# Ordered env-var candidates per LLM provider. Never fall back to *another*
# provider's key — handing an Anthropic key to the OpenAI client just 401s. The
# google chain is all-Google vars, so it does not cross provider boundaries.
_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


@register_provider("table_codegen_extract")
class TableCodegenExtractProvider(Provider):
    """LlamaParse-Agentic → table context → agent-authored extractor script → run.

    config keys: ``model`` (default claude-sonnet-4-6), ``max_turns`` (10),
    ``max_tokens`` (16000, per-response cap for the codegen model — field-dense
    schemas need long scripts; the loop warns the model and bails after two
    consecutive truncations),
    ``script_timeout_s`` (60, hard wall-clock cap for each sandboxed run of the
    generated script), ``document_mode`` (``"pages"`` default: per-page markdown
    Document; ``"items"``: structured-items Document built from the parse's
    typed item sequence — scripts iterate/filter items instead of regexing page
    text), ``review_loop`` (default False: one-shot submit; True adds the
    run->inspect->submit loop so the model sees its output before finalizing),
    ``coverage_warning`` (default False, requires review_loop: each run flags
    table rows whose values are not reflected in the output), ``output_summary``
    (default False, requires review_loop: each run appends per-field fill-rate +
    value-distribution of the model's own output for it to reconcile against the
    document), ``view_tools`` (default False, requires review_loop: adds
    view_document / view_output on-demand inspection tools), ``edit_tool`` (default
    False: adds str_replace for surgical script edits; works in either path),
    ``echo_run_output`` (default True; False requires review_loop: run_script returns
    a record-count stub instead of the full output, for use with view_tools),
    ``normalize_quotes`` (default False, items mode only: fold curly quotes to
    straight in the parsed items so brittle single-quote-form regexes match the whole
    doc), ``provenance`` (default False, items mode only: the script attaches a reserved
    _provenance:{"page": N} to each output record from the item's page_num — the gate
    tolerates the key and the accuracy metric ignores it; with output_summary on, the run
    summary also reports per-record page coverage), ``preview_max_items``
    (None default: review-loop runs echo the full output; an int caps each array
    for cheap smoke tests), ``parse_config`` (LlamaParse base_config; defaults to
    agentic/latest with product-default heuristics/HTML tables), ``pricing``
    (model -> [in,out] per-1M), and optional ``api_key`` (model API key; else
    from the provider-matched env var).
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_MAX_TURNS = 10
    DEFAULT_MAX_TOKENS = 16000
    DEFAULT_SCRIPT_TIMEOUT_S = 60.0
    # Forwarded to LlamaParseProvider base_config -> the V2 SDK parsing.create().
    # Product defaults apply beyond tier/version (parse heuristics stay ON);
    # tables default to HTML in the normalized page markdown (the parse provider's
    # _output_tables_as_markdown() is False unless asked otherwise), which is what
    # table_parser consumes.
    DEFAULT_PARSE_CONFIG: dict[str, Any] = {
        "tier": "agentic",
        "version": "latest",
    }

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._model: str = self.base_config.get("model", self.DEFAULT_MODEL)
        # Fail fast on unknown model families (also selects the API-key env var).
        self._llm_provider: str = _provider_for_model(self._model)
        self._document_mode: str = self.base_config.get("document_mode", "pages")
        if self._document_mode not in ("pages", "items"):
            raise ProviderConfigError(
                f"table_codegen_extract: document_mode must be 'pages' or 'items', got {self._document_mode!r}."
            )
        # Experiment flags (default OFF so the baseline pipelines are one-shot, matching
        # the original behaviour). review_loop adds the run->inspect->submit loop;
        # coverage_warning augments each run with a column-coverage missing-row check.
        self._review_loop: bool = bool(self.base_config.get("review_loop", False))
        self._coverage_warning: bool = bool(self.base_config.get("coverage_warning", False))
        if self._coverage_warning and not self._review_loop:
            raise ProviderConfigError(
                "table_codegen_extract: coverage_warning=True requires review_loop=True "
                "(it augments the run_script step, which only exists in the review loop)."
            )
        # Output-summary check: augment each run_script with per-field fill-rate +
        # value-distribution of the model's own output, for the model to reconcile
        # against the document (also review-loop only).
        self._output_summary: bool = bool(self.base_config.get("output_summary", False))
        if self._output_summary and not self._review_loop:
            raise ProviderConfigError(
                "table_codegen_extract: output_summary=True requires review_loop=True "
                "(it augments the run_script step, which only exists in the review loop)."
            )
        # On-demand inspection tools (review loop only): view_document / view_output.
        self._view_tools: bool = bool(self.base_config.get("view_tools", False))
        if self._view_tools and not self._review_loop:
            raise ProviderConfigError(
                "table_codegen_extract: view_tools=True requires review_loop=True "
                "(view_output re-reads the run_script output, which only exists in the review loop)."
            )
        # str_replace edit tool — works in either path (it edits the held script).
        self._edit_tool: bool = bool(self.base_config.get("edit_tool", False))
        # Whether run_script echoes the full output (default True). Turning it OFF only
        # affects the review loop (one-shot's submit never echoes anyway).
        self._echo_run_output: bool = bool(self.base_config.get("echo_run_output", True))
        if not self._echo_run_output and not self._review_loop:
            raise ProviderConfigError(
                "table_codegen_extract: echo_run_output=False requires review_loop=True "
                "(the run echo only exists in the review loop)."
            )
        # Surface the turn budget in each run_script result ("turn X of N — finalize while a
        # spare turn remains") so the model stops editing a correct output instead of running
        # to max_turns and being salvaged. Review-loop only (it rides the run_script result).
        self._turn_budget_hint: bool = bool(self.base_config.get("turn_budget_hint", False))
        if self._turn_budget_hint and not self._review_loop:
            raise ProviderConfigError(
                "table_codegen_extract: turn_budget_hint=True requires review_loop=True "
                "(it augments the run_script result, which only exists in the review loop)."
            )
        # Fold curly quotes -> straight in the parsed items (items mode only) so brittle
        # single-quote-form regexes match the whole doc, not just the straight-quoted subset.
        self._normalize_quotes: bool = bool(self.base_config.get("normalize_quotes", False))
        # Page-level per-record provenance (items mode only): the script attaches a reserved
        # _provenance:{"page": N} to each output record (from the item's page_num); the gate
        # tolerates the key and the accuracy metric ignores it. Works one-shot or in the review
        # loop; with output_summary on, the run summary also reports per-record page coverage.
        self._provenance: bool = bool(self.base_config.get("provenance", False))
        if self._provenance and self._document_mode != "items":
            raise ProviderConfigError(
                "table_codegen_extract: provenance=True requires document_mode='items' "
                "(page attribution rides on the parse's typed items, which only the items document exposes)."
            )
        self._preview_max_items: int | None = self.base_config.get("preview_max_items")
        # Gemini-only thinking effort ("low"/"medium"/"high"/"minimal"; None = the model's
        # default dynamic thinking). Ignored for anthropic/openai. Plumbed into _run_google.
        # gemini-3.x flash honors thinking_level, not the 2.5-era thinking_budget.
        self._thinking_level: str | None = self.base_config.get("thinking_level")
        # Anthropic 5-gen / 4.x reasoning effort ("low"/"medium"/"high"/"xhigh"/"max"; None =
        # the model's default, which is "high" on sonnet-5). GA via output_config.effort (no beta
        # header); ignored by gemini (uses thinking_level above) and openai. Plumbed to _run_anthropic.
        self._effort: str | None = self.base_config.get("effort")
        self._max_turns: int = int(self.base_config.get("max_turns", self.DEFAULT_MAX_TURNS))
        self._max_tokens: int = int(self.base_config.get("max_tokens", self.DEFAULT_MAX_TOKENS))
        self._script_timeout_s: float = float(self.base_config.get("script_timeout_s", self.DEFAULT_SCRIPT_TIMEOUT_S))
        self._parse_config: dict[str, Any] = {**self.DEFAULT_PARSE_CONFIG, **self.base_config.get("parse_config", {})}
        self._pricing: dict[str, tuple[float, float]] = {**_PRICING_PER_1M, **self.base_config.get("pricing", {})}
        # Model API key (provider-matched; no cross-provider fallback). None is
        # fine — the SDK raises its own clear auth error. Parse uses its own
        # LLAMA_CLOUD key. The google chain tries each Gemini var in order.
        self._api_key: str | None = self.base_config.get("api_key") or next(
            (v for var in _ENV_KEYS[self._llm_provider] if (v := os.getenv(var))), None
        )

    # ------------------------------------------------------------------ #
    def _parse_agentic(self, request: InferenceRequest) -> tuple[Document | ItemsDocument, float, int, str | None]:
        """Run LlamaParse Agentic and return (document, parse_cost_usd, num_pages,
        parse_job_id). The document is a markdown-page ``Document`` or, with
        ``document_mode="items"``, an ``ItemsDocument`` over the parse's typed
        item sequence."""
        parse_cfg = {**self._parse_config}
        if self.base_config.get("llama_cloud_api_key"):
            parse_cfg["api_key"] = self.base_config["llama_cloud_api_key"]
        parse_provider = LlamaParseProvider("llamaparse", base_config=parse_cfg)
        parse_pipeline = PipelineSpec(
            pipeline_name=f"{self.provider_name}__parse",
            provider_name="llamaparse",
            product_type=ProductType.PARSE,
            config=parse_cfg,
        )
        parse_request = InferenceRequest(
            example_id=request.example_id,
            source_file_path=request.source_file_path,
            product_type=ProductType.PARSE,
        )
        raw_parse = parse_provider.run_inference(parse_pipeline, parse_request)
        parse_result = parse_provider.normalize(raw_parse)
        parse_out = parse_result.output
        if not isinstance(parse_out, ParseOutput):
            raise ProviderPermanentError(
                f"table_codegen_extract: expected ParseOutput from LlamaParse, got {type(parse_out).__name__}."
            )
        document: Document | ItemsDocument
        if self._document_mode == "items":
            items_pages = (raw_parse.raw_output.get("items") or {}).get("pages")
            if not isinstance(items_pages, list) or not items_pages:
                raise ProviderPermanentError(
                    "table_codegen_extract: document_mode='items' but the parse result has no "
                    "items.pages — the LlamaParse call must expand 'items'."
                )
            document = ItemsDocument(items_pages, normalize_quotes=self._normalize_quotes)
            n_pages = len(document.pages)
        else:
            pages = [{"page_num": p.page_index + 1, "text": p.markdown} for p in parse_out.pages]
            document = Document(pages)
            n_pages = len(pages)
        parse_cost = float(raw_parse.raw_output.get("cost_usd") or 0.0)
        num_pages = int(raw_parse.raw_output.get("num_pages") or n_pages)
        parse_job_id = extract_job_id_from_raw_payload(raw_parse.raw_output)
        return document, parse_cost, num_pages, parse_job_id

    def _codegen_cost(self, usage: dict[str, Any]) -> float:
        cin, cout = pricing_for_model(self._model, self._pricing)  # longest-prefix match
        if not cin and not cout:
            logger.warning(
                "table_codegen_extract: no pricing for model %r — codegen cost will be reported as $0",
                self._model,
            )
            return 0.0
        n_in = float(usage.get("input", 0) or 0)
        n_out = float(usage.get("output", 0) or 0)
        read = float(usage.get("cache_read", 0) or 0)
        write = float(usage.get("cache_write", 0) or 0)
        # Gemini reports thinking tokens (thoughts_token_count) SEPARATELY from output and
        # bills them at the output rate. Defaults to 0 for anthropic/openai, whose own
        # output_tokens/completion_tokens already fold in any reasoning tokens.
        think = float(usage.get("thinking", 0) or 0)
        if self._llm_provider == "anthropic":
            # input_tokens EXCLUDES cache tokens; writes bill at 1.25x, reads at 0.1x.
            in_cost = (n_in + 1.25 * write + 0.1 * read) / 1e6 * cin
        elif self._llm_provider == "google":
            # Gemini prompt_token_count INCLUDES cached tokens; the cached subset bills
            # at the absolute cache-hit rate (≈0.1x input for flash-lite), the rest at
            # the standard input rate. Implicit caching has no write surcharge.
            cache_hit, _storage = gemini_context_cache_pricing_per_million(self._model)
            in_cost = ((n_in - read) * cin + read * cache_hit) / 1e6
        else:
            # OpenAI prompt_tokens INCLUDES cached tokens, billed at 0.5x.
            in_cost = ((n_in - read) + 0.5 * read) / 1e6 * cin
        return round(in_cost + (n_out + think) / 1e6 * cout, 6)

    # ------------------------------------------------------------------ #
    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"{type(self).__name__} only supports EXTRACT product type, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT. Provide a JSON schema in InferenceRequest.schema_override."
            )
        src = Path(request.source_file_path)
        if not src.exists():
            raise ProviderPermanentError(f"File not found: {src}")
        if src.suffix.lower() != ".pdf" and src.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ProviderPermanentError(
                f"{type(self).__name__} supports PDFs and {set(IMAGE_EXTENSIONS)}, got {src.suffix}"
            )

        started_at = datetime.now()

        # 1) LlamaParse Agentic FIRST — its parsed output is the pipeline's input.
        document, parse_cost, num_pages, parse_job_id = self._parse_agentic(request)
        if isinstance(document, ItemsDocument):
            if not document.items():
                logger.warning(
                    "table_codegen_extract: items document for %s has zero items — the agent "
                    "will see empty pages; check the parse items expand.",
                    request.example_id,
                )
        elif not document.tables():
            logger.warning(
                "table_codegen_extract: no HTML tables parsed from %s — the agent will see prose "
                "only; check the parse_config table output mode (tables must be HTML, not markdown).",
                request.example_id,
            )

        # 2) Agent authors + gates an extractor; run it over the full document.
        result = generate_extractor(
            document,
            request.schema_override,
            provider=self._llm_provider,
            model=self._model,
            max_turns=self._max_turns,
            max_tokens=self._max_tokens,
            script_timeout_s=self._script_timeout_s,
            api_key=self._api_key,
            review_loop=self._review_loop,
            coverage_warning=self._coverage_warning,
            output_summary=self._output_summary,
            view_tools=self._view_tools,
            edit_tool=self._edit_tool,
            echo_run_output=self._echo_run_output,
            preview_max_items=self._preview_max_items,
            provenance=self._provenance,
            turn_budget_hint=self._turn_budget_hint,
            thinking_level=self._thinking_level,
            effort=self._effort,
        )
        # Accept a clean submit OR a *_salvaged result (the agent ran out of turns but
        # had produced a viable run — see generate_extractor's salvage). Everything else
        # is transient, not permanent: a gate failure within max_turns is sampling noise
        # (draw-to-draw variance is large) — let the runner's retry take a fresh sample.
        accepted = result["status"] == "done" or str(result["status"]).endswith("_salvaged")
        if not accepted or not isinstance(result.get("output"), dict):
            raise ProviderTransientError(
                f"table_codegen_extract: agent did not produce a valid extraction "
                f"(status={result['status']}, turns={result.get('turns')})."
            )

        completed_at = datetime.now()
        usage = result.get("usage") or {}
        codegen_cost = self._codegen_cost(usage)
        cost_usd = round(parse_cost + codegen_cost, 6)
        raw_output: dict[str, Any] = {
            "data": result["output"],
            "model": self._model,
            "usage": usage,
            "num_pages": num_pages,
            "cost_usd": cost_usd,
            "cost_per_page_usd": (cost_usd / num_pages) if num_pages else None,
            "parse_cost_usd": parse_cost,
            "codegen_cost_usd": codegen_cost,
            # The evaluator copies parse_job_id into the report. parse_pages is the
            # exact document input (markdown pages or items pages, per document_mode)
            # — the agentic parse is nondeterministic across runs, so without it a
            # run's scripts can't be debugged or re-executed after the fact.
            "parse_job_id": parse_job_id,
            "parse_pages": document.to_pages(),
            "turns": result.get("turns"),
            # Surface the salvage distinctly: a salvaged result is the agent's last
            # validated run returned because it never reached submit — never a clean submit.
            "status": result["status"],
            "salvaged": bool(result.get("salvaged")),
            "script": result.get("script", ""),
            "_config": {
                "model": self._model,
                "max_turns": self._max_turns,
                "max_tokens": self._max_tokens,
                "script_timeout_s": self._script_timeout_s,
                "document_mode": self._document_mode,
                "review_loop": self._review_loop,
                "coverage_warning": self._coverage_warning,
                "output_summary": self._output_summary,
                "view_tools": self._view_tools,
                "edit_tool": self._edit_tool,
                "echo_run_output": self._echo_run_output,
                "normalize_quotes": self._normalize_quotes,
                "provenance": self._provenance,
                "turn_budget_hint": self._turn_budget_hint,
                "effort": self._effort,
                "preview_max_items": self._preview_max_items,
                "parse_config": self._parse_config,
            },
            "_trace_tail": (result.get("trace") or [])[-20:],
            # Full turn-by-turn trace + generated script, written out as sidecars by
            # the runner (_write_codegen_artifacts) and popped from raw.json. _trace_tail
            # above keeps only the last 20 entries, which drops the early turns where the
            # model chooses its approach — exactly what's needed to debug a collapse. Each
            # tool result is already capped (trace_result_limit) so the trace stays bounded.
            "_codegen_artifacts": {
                "trace_jsonl_lines": [
                    json.dumps(entry, ensure_ascii=False, default=str) for entry in (result.get("trace") or [])
                ],
                "script": result.get("script", "") or "",
            },
        }
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        return normalize_extract_result(raw_result)

    def cancel(self, example_id: str) -> bool:
        return False
