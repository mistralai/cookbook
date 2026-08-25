"""Credit-based cost accounting for the LlamaExtract V2 pipelines.

LlamaExtract bills in credits rather than tokens, and a single extract job has
two separately-billed legs: the extract itself and the parse it runs first.
Rates are the published ones at developers.llamaindex.ai/llamaparse/general/pricing.

The two legs bill against *different* page counts, which is why they are kept
apart rather than summed into one all-in rate:

* parse bills every page it processed;
* extract bills ``num_pages_billed`` -- the *effective* billable pages, which a
  large schema pushes above the document's real page count because the schema is
  extracted in several passes.

Pricing is applied at normalize time, so re-normalizing a stored result
re-prices it without re-running inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

USD_PER_CREDIT = 1.25 / 1000

EXTRACT_TIER_CREDITS_PER_PAGE: dict[str, int] = {
    "cost_effective": 5,
    "agentic": 15,
    "agentic_plus": 50,
}
PARSE_TIER_CREDITS_PER_PAGE: dict[str, int] = {
    "fast": 1,
    "cost_effective": 3,
    "agentic": 10,
    "agentic_plus": 45,
}

# Extract tiers that always run a specific parse tier first, regardless of the
# tier name they share. `agentic_plus` extract runs an *agentic* parse (10
# credits/page), not an agentic_plus parse: billing is 50 per billable page for
# the extract plus 10 per parsed page. Pricing it as a flat all-in 50 drops the
# parse leg entirely.
DEFAULT_PARSE_TIER_FOR_EXTRACT_TIER: dict[str, str] = {
    "agentic_plus": "agentic",
}

# Large-schema bands: (max field count, billable-page multiplier). A schema with
# more fields is extracted in more passes and bills proportionally more effective
# pages. Only ever a FALLBACK -- a page count the job itself reported always wins.
LARGE_SCHEMA_BANDS: tuple[tuple[int, int], ...] = ((200, 1), (800, 2), (1600, 3), (2400, 4), (3200, 5))
# Top band edge. Counting stops here: every larger schema prices identically, so
# walking further buys nothing and is exponential on a `$ref` DAG.
MAX_SCHEMA_FIELDS = LARGE_SCHEMA_BANDS[-1][0]
# The premium is published for Agentic Plus alone. Estimating it for any other
# tier would invent a charge the platform never makes.
LARGE_SCHEMA_MULTIPLIER_TIERS = frozenset({"agentic_plus"})


def normalize_tier(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def page_count_from(raw_output: dict[str, Any]) -> float | None:
    """Pages the job says it processed, or None."""
    metadata = raw_output.get("metadata")
    if isinstance(metadata, dict):
        usage = metadata.get("usage")
        if isinstance(usage, dict):
            for key in ("num_pages_extracted", "num_pages", "page_count"):
                value = usage.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    return float(value)
    return None


def reported_billed_page_count(raw_output: dict[str, Any]) -> float | None:
    """Billable pages the job itself reported, or None.

    Read *only* out of the API's own ``metadata.usage`` block, never from a
    top-level key. Pricing runs again on every normalize, so reading back a
    top-level ``num_pages_billed`` would let an earlier pass's schema *estimate*
    be relabelled as a reported value and then frozen there -- a number we
    invented, presented as one the platform billed.
    """
    metadata = raw_output.get("metadata")
    if isinstance(metadata, dict):
        usage = metadata.get("usage")
        if isinstance(usage, dict):
            value = usage.get("num_pages_billed")
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return None


def local_page_count(file_path: Path | str | None) -> float | None:
    """Page count read off the source document, for jobs that report none.

    Imported lazily: the PDF readers live in the ``runners`` extra, and the
    core package must stay importable without it.
    """
    if not file_path:
        return None
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        return None
    try:
        import pypdf

        with path.open("rb") as fh:
            return float(len(pypdf.PdfReader(fh).pages))
    except Exception:  # noqa: BLE001 - a page count is best-effort, never fatal
        logger.debug("could not read a local page count from %s", path, exc_info=True)
        return None


def _resolve_schema_ref(node: Any, root: dict[str, Any], seen: frozenset[str]) -> tuple[Any, frozenset[str]]:
    """Follow local ``$ref`` pointers, refusing external refs and cycles."""
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen:
            return {}, seen
        seen = seen | {ref}
        target: Any = root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                return {}, seen
            target = target[part]
        node = target
    return node, seen


def schema_field_count(
    node: Any,
    root: dict[str, Any],
    seen: frozenset[str] = frozenset(),
    limit: int = MAX_SCHEMA_FIELDS,
) -> int:
    """Declared field count of a schema, capped at ``limit``.

    Each named property counts once. An array counts as one field and its item
    object's fields are counted once, not per row -- the schema is what drives
    extraction passes, so the row count is irrelevant. A nullable field written
    as ``anyOf: [T, null]`` counts as its widest branch, not the sum.

    Counting stops once ``limit`` is reached. ``seen`` only guards a ``$ref``
    repeating on the current path, so it stops cycles but not re-walking a shared
    sub-schema through every route that reaches it -- a plain ``$ref`` DAG, which
    is what generated schemas produce, costs exponential time without the cap.
    Every count past the top band prices identically, so stopping is free.
    """
    if limit <= 0:
        return 0

    node, seen = _resolve_schema_ref(node, root, seen)
    if not isinstance(node, dict):
        return 0

    total = 0

    def add(value: int) -> bool:
        """Accumulate, reporting whether the cap has been hit."""
        nonlocal total
        total += value
        return total >= limit

    props = node.get("properties")
    if isinstance(props, dict):
        for sub in props.values():
            if add(1 + schema_field_count(sub, root, seen, limit - total - 1)):
                return total

    items = node.get("items")
    if isinstance(items, dict):
        if add(schema_field_count(items, root, seen, limit - total)):
            return total
    elif isinstance(items, list):  # tuple-form arrays
        for sub in items:
            if add(schema_field_count(sub, root, seen, limit - total)):
                return total

    for key in ("anyOf", "oneOf"):
        branches = node.get(key)
        if isinstance(branches, list):
            widest = max((schema_field_count(b, root, seen, limit - total) for b in branches), default=0)
            if add(widest):
                return total

    all_of = node.get("allOf")
    if isinstance(all_of, list):
        for sub in all_of:
            if add(schema_field_count(sub, root, seen, limit - total)):
                return total

    return total


def large_schema_multiplier(data_schema: dict[str, Any] | None) -> tuple[int, int] | None:
    """``(field count, billable-page multiplier)``, or None with no usable schema."""
    if not isinstance(data_schema, dict) or not data_schema:
        return None
    fields = schema_field_count(data_schema, data_schema)
    if fields <= 0:
        return None
    for max_fields, multiplier in LARGE_SCHEMA_BANDS:
        if fields <= max_fields:
            return fields, multiplier
    # Counting caps at the top band, so this is reachable only if the bands and
    # the cap ever disagree. Bill at the highest published rate either way.
    return fields, LARGE_SCHEMA_BANDS[-1][1]


def pricing_for_configuration(configuration: dict[str, Any]) -> tuple[float, float] | None:
    """Per-page credit rates as ``(extract, parse)``, or None for an unknown tier."""
    extract_tier = normalize_tier(configuration.get("tier") or "cost_effective")
    default_parse_tier = DEFAULT_PARSE_TIER_FOR_EXTRACT_TIER.get(extract_tier, extract_tier)
    parse_tier = normalize_tier(configuration.get("parse_tier") or default_parse_tier)
    extract_credits = EXTRACT_TIER_CREDITS_PER_PAGE.get(extract_tier)
    parse_credits = PARSE_TIER_CREDITS_PER_PAGE.get(parse_tier)
    if extract_credits is None or parse_credits is None:
        return None
    return float(extract_credits), float(parse_credits)


def billed_page_count_for(
    raw_output: dict[str, Any],
    page_count: float | None,
    data_schema: dict[str, Any] | None,
    extract_tier: str,
) -> tuple[float, str, int | None] | None:
    """Billable pages as ``(pages, source, field count)``, preferring the job.

    The job's own ``num_pages_billed`` is authoritative and always wins for every
    tier: it is what the platform actually billed. Only when the response carries
    no usage block do we estimate from the schema's field count, because the
    large-schema premium is a published function of exactly that.

    The field count is returned so it can be recorded -- the band edges are our
    counting convention rather than a published rule, so anyone checking a
    disputed number needs to see whether the schema landed at 199 or 3500 fields.

    Estimation is tier-gated: the premium is published for Agentic Plus only, so
    estimating it elsewhere would invent a charge that is never made.
    """
    reported = reported_billed_page_count(raw_output)
    if reported is not None:
        return reported, "reported", None

    if page_count is None or extract_tier not in LARGE_SCHEMA_MULTIPLIER_TIERS:
        return None
    band = large_schema_multiplier(data_schema)
    if band is None:
        return None
    fields, multiplier = band
    return multiplier * page_count, "schema_estimate", fields


def apply_pricing_fields(
    raw_output: dict[str, Any],
    configuration: dict[str, Any],
    *,
    data_schema: dict[str, Any] | None = None,
    fallback_page_count: float | None = None,
) -> None:
    """Write credit/cost stats onto ``raw_output`` in place.

    Keys land where ``evaluation.stats.build_operational_stats`` picks them up,
    so they flow into the report as per-document operational stats.
    """
    # Page count and pricing resolve independently: a tier we cannot price must
    # not suppress num_pages, which the report uses on its own.
    page_count = page_count_from(raw_output)
    if page_count is None and fallback_page_count is not None and fallback_page_count > 0:
        page_count = float(fallback_page_count)
    if page_count is not None:
        raw_output["num_pages"] = page_count

    extract_tier = normalize_tier(configuration.get("tier") or "cost_effective")

    billed_pages: float | None = None
    billed = billed_page_count_for(raw_output, page_count, data_schema, extract_tier)
    if billed is not None:
        billed_pages, billed_source, billed_fields = billed
        raw_output["num_pages_billed"] = billed_pages
        raw_output["num_pages_billed_source"] = billed_source
        if billed_fields is not None:
            raw_output["schema_field_count"] = billed_fields

    pricing = pricing_for_configuration(configuration)
    if pricing is None:
        return
    extract_credits_per_page, parse_credits_per_page = pricing
    raw_output["extract_credits_per_page"] = extract_credits_per_page
    raw_output["parse_credits_per_page"] = parse_credits_per_page

    if page_count is None:
        return

    # Parse bills every page it processed; extract bills the effective billable
    # pages when known, falling back to parsed pages otherwise.
    extract_pages = billed_pages if billed_pages is not None else page_count
    extract_credits = extract_credits_per_page * extract_pages
    parse_credits = parse_credits_per_page * page_count
    credits_used = extract_credits + parse_credits

    raw_output["extract_credits"] = extract_credits
    raw_output["parse_credits"] = parse_credits
    raw_output["total_credits"] = credits_used
    raw_output["credits_used"] = credits_used
    # Effective all-in rate over *parsed* pages -- what $/page reporting
    # consumes. With billable pages in play this is no longer a flat constant.
    raw_output["credits_per_page"] = credits_used / page_count
    raw_output["cost_per_page_usd"] = (credits_used / page_count) * USD_PER_CREDIT
    raw_output["cost_usd"] = credits_used * USD_PER_CREDIT
