"""Regenerate the leaderboard tables in README.md from leaderboard.csv.

Two tables are generated:

    LEADERBOARD    unified value F1 (the headline metric)
    GROUNDING      word-level and page-level grounding F1, side by side

Run: uv run python scripts/update_readme.py
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "leaderboard.csv"
README_PATH = REPO_ROOT / "README.md"


# Grounding is only defined where a field's ground truth carries a verified box,
# so both grounding tables are scored over a smaller and differently distributed
# document set than the value F1 above — their columns do not reconcile with it.
GROUNDING_SCOPE_NOTE = "Scored only over the documents that carry verified box ground truth."


def fmt(v: str) -> str:
    return v if v else "—"


def fmt_cost(v: str) -> str:
    """Cost columns are stored in $/page; render as ¢/page."""
    return f"{float(v) * 100:.2f}¢" if v else "—"


def column_ranks(rows: list[dict], columns: tuple[str, ...]) -> dict[str, dict[float, str]]:
    """Return emphasis for the best and second-best distinct score in each column."""
    ranks = {}
    for column in columns:
        values = sorted({float(row[column]) for row in rows}, reverse=True)
        ranks[column] = dict(zip(values[:2], ("best", "second"), strict=False))
    return ranks


def fmt_ranked(value: str, rank: str | None, *, html: bool = False) -> str:
    """Render a score, emphasizing the best and second-best distinct values."""
    rendered = fmt(value)
    if rank == "best":
        return f"<strong>{rendered}</strong>" if html else f"**{rendered}**"
    if rank == "second":
        return f"<u>{rendered}</u>"
    return rendered


def table(header: str, aligns: str, rows: list[list[str]]) -> str:
    return "\n".join([header, aligns, *(f"| {' | '.join(r)} |" for r in rows)])


def value_table(rows: list[dict]) -> str:
    ranked = sorted(rows, key=lambda r: float(r["Overall"]), reverse=True)
    score_columns = ("Overall", "Short", "Medium", "Long")
    ranks = column_ranks(rows, score_columns)
    return table(
        "| Rank | Provider | Category | Overall | Short | Medium | Long | ¢ / Page |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
        [
            [
                str(i),
                r["Provider"],
                r["Category"],
                *(fmt_ranked(r[column], ranks[column].get(float(r[column]))) for column in score_columns),
                fmt_cost(r["Cost_Per_Page"]),
            ]
            for i, r in enumerate(ranked, 1)
        ],
    )


GROUNDING_PREFIXES = ("Word_Grounding", "Page_Grounding")
GROUNDING_SPANS = ("Word-level grounding F1", "Page-level grounding F1")


def _grounding_cells(row: dict, ranks: dict[str, dict[float, str]]) -> list[str]:
    """The eight grounding numbers, word block then page block."""
    return [
        fmt_ranked(row[key], ranks[key].get(float(row[key])), html=True)
        for prefix in GROUNDING_PREFIXES
        for key in (prefix, f"{prefix}_Short", f"{prefix}_Medium", f"{prefix}_Long")
    ]


def grounding_table(rows: list[dict]) -> str:
    """Both grounding metrics in one table, ranked by word-level overall.

    Written as HTML rather than a markdown table because the eight metric
    columns only stay readable under grouped headers, and markdown tables
    cannot span a header cell.

    Systems that score zero on *both* metrics collapse into a single trailing
    row: most systems return no source evidence at all, so listing them would be
    eight rows of zeros. A system scoring on one metric but not the other still
    gets its own row.
    """
    scoring = [r for r in rows if any(float(r[p]) > 0 for p in GROUNDING_PREFIXES)]
    scoring.sort(key=lambda r: float(r[GROUNDING_PREFIXES[0]]), reverse=True)
    zeroed = [r for r in rows if all(float(r[p]) == 0 for p in GROUNDING_PREFIXES)]
    score_columns = tuple(
        key
        for prefix in GROUNDING_PREFIXES
        for key in (prefix, f"{prefix}_Short", f"{prefix}_Medium", f"{prefix}_Long")
    )
    ranks = column_ranks(scoring, score_columns)

    span = "".join(f'<th colspan="4">{s}</th>' for s in GROUNDING_SPANS)
    sub = "".join(f'<th align="right">{c}</th>' for c in ("Overall", "Short", "Medium", "Long") * 2)
    body = [
        "    <tr>"
        f'<td align="right">{i}</td><td>{r["Provider"]}</td>'
        + "".join(f'<td align="right">{c}</td>' for c in _grounding_cells(r, ranks))
        + "</tr>"
        for i, r in enumerate(scoring, 1)
    ]
    if zeroed:
        body.append(
            '    <tr><td align="right">—</td>'
            f"<td><em>All {len(zeroed)} other systems</em></td>" + '<td align="right">0.00</td>' * 8 + "</tr>"
        )
    return "\n".join(
        [
            "<table>",
            "  <thead>",
            f'    <tr><th rowspan="2">Rank</th><th rowspan="2">Provider</th>{span}</tr>',
            f"    <tr>{sub}</tr>",
            "  </thead>",
            "  <tbody>",
            *body,
            "  </tbody>",
            "</table>",
        ]
    )


def replace_block(readme: str, name: str, body: str) -> str:
    start_marker, end_marker = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    start, end = readme.find(start_marker), readme.find(end_marker)
    if start == -1 or end == -1:
        raise SystemExit(f"Markers not found in README.md. Add {start_marker} and {end_marker}.")
    block = f"{start_marker}\n{body}\n{end_marker}"
    return readme[:start] + block + readme[end + len(end_marker) :]


def main() -> None:
    with CSV_PATH.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("Provider")]

    readme = README_PATH.read_text()
    readme = replace_block(
        readme,
        "LEADERBOARD",
        "**Unified value F1** — the headline metric. Every score is an unweighted mean over "
        "documents; each document counts once, whatever its length. For raw data including per-split "
        "precision and recall, cost, and latency, see [leaderboard.csv](leaderboard.csv). "
        "The best score in each Overall, Short, Medium, and Long column is **bold**; the second-best "
        "distinct score is <u>underlined</u>.\n\n" + value_table(rows),
    )
    readme = replace_block(
        readme,
        "GROUNDING",
        "**Grounding F1** — a field counts only when its value is accepted *and* it points at the "
        "right evidence: at word level the predicted box must overlap an accepted evidence box at "
        "IoU 0.5, at page level the cited page must be correct. "
        + GROUNDING_SCOPE_NOTE
        + "\n\n"
        + grounding_table(rows),
    )
    README_PATH.write_text(readme)
    print(f"Updated README.md from {CSV_PATH.name} ({len(rows)} systems)")


if __name__ == "__main__":
    main()
