"""Value primitives exposed to generated extractor code.

This is the *only* shared, domain- and doc-agnostic code an approach script
imports (besides the Table interface in ``table_parser``). It is the harness's
"standard library": small, pure value converters a model can rely on. Anything
domain- or doc-specific (table-role detection, statement constants, mapping
logic) lives inside each approach's own file, not here.
"""

from __future__ import annotations

import re


def money(cell: str) -> float | None:
    """'$1,086.44' / '-114.90' -> float ; '' / non-numeric -> None."""
    m = re.search(r"-?[\d,]+(?:\.\d+)?", cell)
    return float(m.group().replace(",", "")) if m else None


def iso_date(mmdd: str, year: int) -> str | None:
    """'08/17' + 2025 -> '2025-08-17' ; non-MM/DD -> None."""
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})", mmdd)
    return f"{year}-{int(m.group(1)):02d}-{int(m.group(2)):02d}" if m else None
