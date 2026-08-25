"""Shared helper to build operational RunStat entries from an InferenceResult."""

from extract_bench.schemas.evaluation import RunStat
from extract_bench.schemas.pipeline_io import InferenceResult

# Stat keys to look for in raw_output, with their units
_RAW_OUTPUT_STATS: list[tuple[str, str]] = [
    ("credits_used", "credits"),
    # Credit-billed pipelines (LlamaExtract) report their charge in credits and
    # bill the extract leg against `num_pages_billed` — the *effective* billed
    # pages, which a large schema pushes above the document's page count — while
    # the parse leg bills against pages parsed. Carrying both legs separately is
    # what makes a per-page cost reconstructable from the report.
    ("credits_per_page", "credits/page"),
    ("parse_credits", "credits"),
    ("extract_credits", "credits"),
    ("total_credits", "credits"),
    ("num_pages_billed", "pages"),
    ("cost_usd", "$"),
    ("cost_per_page_usd", "$/page"),
    ("input_cost_usd", "$"),
    ("tool_use_prompt_cost_usd", "$"),
    ("cached_input_cost_usd", "$"),
    ("output_and_thinking_cost_usd", "$"),
    ("cache_storage_cost_usd", "$"),
    # Non-token charges from server-side tool use, and the cache-write term some
    # model families bill separately. Without these the per-term cost breakdown
    # under-sums against cost_usd — by roughly a third on a GPT-5.6 tool run,
    # where the container fee plus cache writes carry that much of the bill.
    ("cache_write_cost_usd", "$"),
    ("container_cost_usd", "$"),
    ("container_time_cost_usd", "$"),
    ("tool_surcharge_usd", "$"),
    ("token_cost_usd", "$"),
    ("acu_content_extraction_cost_usd", "$"),
    ("acu_contextualization_cost_usd", "$"),
    # Token metrics (when available from provider)
    ("input_tokens", "tokens"),
    ("tool_use_prompt_tokens", "tokens"),
    ("cached_content_tokens", "tokens"),
    ("cache_write_tokens", "tokens"),
    ("output_tokens", "tokens"),
    ("total_tokens", "tokens"),
    ("thinking_tokens", "tokens"),
    ("num_api_calls", "calls"),
    # A tool pipeline where the model never called the tool is otherwise
    # indistinguishable in a report from one where it did.
    ("num_tool_calls", "calls"),
    ("num_containers", "containers"),
    ("num_regions", "regions"),
    ("input_tokens_per_page", "tokens/page"),
    ("tool_use_prompt_tokens_per_page", "tokens/page"),
    ("cached_content_tokens_per_page", "tokens/page"),
    ("output_tokens_per_page", "tokens/page"),
]


def build_operational_stats(inference_result: InferenceResult) -> list[RunStat]:
    """Build operational stats from an inference result.

    Extracts latency from the dedicated field and cost-related stats
    from raw_output (pre-computed by the provider).
    """
    stats: list[RunStat] = []
    raw = inference_result.raw_output

    # Latency (from dedicated field)
    if inference_result.latency_in_ms is not None:
        stats.append(RunStat(name="latency_ms", value=float(inference_result.latency_in_ms), unit="ms"))

    # Per-page latency (computed from latency and num_pages)
    num_pages = raw.get("num_pages")
    if inference_result.latency_in_ms is not None and isinstance(num_pages, (int, float)) and num_pages > 0:
        stats.append(
            RunStat(
                name="latency_ms_per_page",
                value=float(inference_result.latency_in_ms) / float(num_pages),
                unit="ms/page",
            )
        )

    # Cost and token stats (from raw_output, pre-computed by provider)
    for key, unit in _RAW_OUTPUT_STATS:
        value = raw.get(key)
        if value is not None:
            stats.append(RunStat(name=key, value=float(value), unit=unit))

    return stats
