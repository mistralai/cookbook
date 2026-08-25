"""Document-pooled cost aggregation shared by every report surface.

Aggregation follows the ExtractBench paper: *every reported figure, for accuracy,
cost and latency, is a mean over documents in which each document counts once and
equally*. An overall number is therefore the mean over all scored documents in
the pool -- pooled across splits, **not** the mean of the per-split means. A
split holding 20 documents must not weigh the same as one holding 252.

Two figures are reported per scope, and they answer different questions:

* **total cost** -- the sum over documents. What the run actually charged.
* **per-page cost** -- the unweighted mean over documents of each document's own
  ``cost_usd / pages``. This is the comparison axis, and it is
  deliberately *not* ``total_cost / total_pages``, which would let long documents
  dominate the figure.

Documents whose provider reported no cost contribute to neither and are counted
in ``documents_without_cost`` so the denominator is always visible rather than
silently shrinking.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

# Per-document stat names written by `evaluation.stats.build_operational_stats`.
COST_TOTAL_STAT = "cost_usd"
COST_PER_PAGE_STAT = "cost_per_page_usd"


@dataclasses.dataclass(frozen=True)
class CostSummary:
    """Cost over one pool of documents, each counted once."""

    total_usd: float
    #: Unweighted mean over documents of per-document cost/page. None when no
    #: document in the pool reported one.
    mean_per_page_usd: float | None
    #: Unweighted mean over documents of per-document total cost.
    mean_per_document_usd: float | None
    #: Documents contributing a cost figure.
    documents: int
    #: Documents in the pool whose provider reported no cost at all.
    documents_without_cost: int

    @property
    def has_cost(self) -> bool:
        return self.documents > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "totalUsd": self.total_usd,
            "meanPerPageUsd": self.mean_per_page_usd,
            "meanPerDocumentUsd": self.mean_per_document_usd,
            "documents": self.documents,
            "documentsWithoutCost": self.documents_without_cost,
            "hasCost": self.has_cost,
        }


def _stat_value(stats: Iterable[Any], name: str) -> float | None:
    """Read one named stat off a per-example result (dict or RunStat)."""
    for stat in stats or ():
        stat_name = stat.get("name") if isinstance(stat, dict) else getattr(stat, "name", None)
        if stat_name != name:
            continue
        value = stat.get("value") if isinstance(stat, dict) else getattr(stat, "value", None)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _document_key(example: Any) -> str:
    """Stable per-document identity used to dedupe across splits."""
    for attr in ("test_id", "example_id"):
        value = example.get(attr) if isinstance(example, dict) else getattr(example, attr, None)
        if value:
            return str(value)
    return ""


def summarize_documents(examples: Iterable[Any]) -> CostSummary:
    """Pool cost over per-example results, counting each document once.

    Accepts the serialized ``per_example_results`` rows or ``EvaluationResult``
    objects. Later duplicates of a document are ignored rather than added, so
    pooling several splits can never double-count.
    """
    seen: set[str] = set()
    totals: list[float] = []
    per_page: list[float] = []
    without_cost = 0

    for example in examples:
        key = _document_key(example)
        if key and key in seen:
            continue
        if key:
            seen.add(key)

        success = example.get("success") if isinstance(example, dict) else getattr(example, "success", True)
        if not success:
            # A document with no result was never billed a measurable amount;
            # counting it as 0 would understate every per-page figure. It is
            # excluded from the cost mean and surfaced separately -- the value
            # metrics already penalize the failure itself.
            without_cost += 1
            continue

        stats = example.get("stats") if isinstance(example, dict) else getattr(example, "stats", None)
        total = _stat_value(stats or (), COST_TOTAL_STAT)
        page = _stat_value(stats or (), COST_PER_PAGE_STAT)
        if total is None and page is None:
            without_cost += 1
            continue
        if total is not None:
            totals.append(total)
        if page is not None:
            per_page.append(page)

    documents = max(len(totals), len(per_page))
    return CostSummary(
        total_usd=sum(totals),
        mean_per_page_usd=(sum(per_page) / len(per_page)) if per_page else None,
        mean_per_document_usd=(sum(totals) / len(totals)) if totals else None,
        documents=documents,
        documents_without_cost=without_cost,
    )


def summarize_splits(per_split_examples: dict[str, Iterable[Any]]) -> CostSummary:
    """Overall cost pooled across splits, each document counted once.

    Splits in this benchmark are disjoint by construction -- a document's split
    *is* its ``length:`` tag, which is single-valued -- but pooling dedupes on
    document identity anyway so a future overlapping slice cannot silently
    double-count. This is the overall figure: pooled over documents, not
    averaged over split means.
    """
    return summarize_documents(example for examples in per_split_examples.values() for example in examples)


def format_usd(value: float | None, *, cents: bool = False) -> str:
    """Render a cost for display, or an em dash when there is none."""
    if value is None:
        return "—"
    if cents:
        return f"{value * 100:.2f}¢"
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"
