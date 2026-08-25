"""Helpers for reshaping flat avg_/min_/max_ aggregate metric dicts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from extract_bench.analysis.metric_definitions import display_name, order_metrics

_AGG_PREFIXES = ("avg_", "min_", "max_")


def group_avg_min_max(
    flat: Mapping[str, float],
    *,
    exclude: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Group ``avg_X`` / ``min_X`` / ``max_X`` keys into ``{X: {avg, min, max}}``.

    Non-prefixed keys (``micro_*``, ``total_*``, ``example_count``, etc.) are
    ignored. Bases listed in ``exclude`` are skipped entirely.
    """
    skip = exclude or set()
    groups: dict[str, dict[str, float]] = {}
    for key, value in flat.items():
        for prefix in _AGG_PREFIXES:
            if key.startswith(prefix):
                base = key[len(prefix) :]
                if base in skip:
                    break
                groups.setdefault(base, {})[prefix.rstrip("_")] = float(value)
                break
    return groups


def to_agg_metric_records(
    flat: Mapping[str, float],
    *,
    exclude: set[str] | None = None,
    sort_by_avg: bool = False,
) -> list[dict[str, Any]]:
    """Build display records ``{name, displayName, avg, min, max}`` from a flat dict.

    When ``sort_by_avg`` is false (default), metrics follow
    :func:`order_metrics` (headline metrics first). When true, records are
    sorted by average descending — used by the aggregate metrics panel.
    """
    groups = group_avg_min_max(flat, exclude=exclude)
    if sort_by_avg:
        names = sorted(groups, key=lambda n: groups[n].get("avg", 0.0), reverse=True)
    else:
        names = order_metrics(groups.keys(), separator=False)

    records: list[dict[str, Any]] = []
    for name in names:
        vals = groups[name]
        records.append(
            {
                "name": name,
                "displayName": display_name(name),
                "avg": vals.get("avg", 0.0),
                "min": vals.get("min", 0.0),
                "max": vals.get("max", 0.0),
            }
        )
    return records
