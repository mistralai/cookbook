#!/usr/bin/env python3
"""Convert a JSONL robot-run ledger into a compact Markdown wiki page.

This vendored copy is intentionally small for the cookbook demo. It keeps the
core corpus-generation flow and drops the live ATF tooling dependencies.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKIPPED_ACTIONS = {"execute"}
MILESTONES = (25, 50, 75, 100)


def format_ts(raw: str) -> str:
    if not raw:
        return "unknown"
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    except ValueError:
        return raw


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if event.get("event_action") in SKIPPED_ACTIONS:
                continue
            events.append(event)
    return events


def event_text(event: dict[str, Any]) -> str:
    details = event.get("details") or {}
    for key in ("content", "prompt", "summary", "message", "status", "raw_rest"):
        value = details.get(key)
        if value not in (None, ""):
            return str(value)
    return str(event.get("raw_line") or "")


def segment_by_job(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pre_run: list[dict[str, Any]] = []
    job_number = 0

    for event in events:
        event_type = event.get("event_type")
        action = event.get("event_action")

        if event_type == "job" and action == "start":
            if current is None and pre_run:
                segments.append({"label": "Pre-run setup", "events": pre_run})
                pre_run = []
            job_number += 1
            current = {
                "label": f"Job {job_number}",
                "content": event_text(event),
                "status": "incomplete",
                "duration": "",
                "events": [event],
            }
            continue

        if current is None:
            pre_run.append(event)
            continue

        current["events"].append(event)
        if event_type == "job" and action == "end":
            details = event.get("details") or {}
            current["status"] = details.get("status", "complete")
            duration = details.get("duration_seconds")
            current["duration"] = f"{duration}s" if duration is not None else ""
            segments.append(current)
            current = None

    if current is not None:
        segments.append(current)
    if pre_run:
        segments.append({"label": "Post-run events", "events": pre_run})
    return segments


def drawing_milestones(events: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    next_index = 0
    for event in events:
        if event.get("event_type") != "drawing":
            continue

        details = event.get("details") or {}
        action = event.get("event_action")
        ts = format_ts(str(event.get("timestamp", "")))

        if action == "start":
            next_index = 0
            lines.append(f"- `{ts}` Drawing started: {event_text(event)}")
        elif action == "progress":
            percent = float(details.get("percent_complete") or 0)
            while next_index < len(MILESTONES) and percent >= MILESTONES[next_index]:
                milestone = MILESTONES[next_index]
                move_current = details.get("move_current")
                move_total = details.get("move_total")
                move_text = (
                    f" ({move_current}/{move_total} moves)"
                    if move_current is not None and move_total is not None
                    else ""
                )
                lines.append(f"- `{ts}` {milestone}% complete{move_text}")
                next_index += 1
    return lines


def render_event(event: dict[str, Any]) -> str | None:
    event_type = event.get("event_type", "event")
    action = event.get("event_action", "")
    category = event.get("event_category", "")

    if event_type == "drawing":
        return None

    label = {
        ("job", "start"): "JOB START",
        ("job", "end"): "JOB END",
        ("ai", "request"): "AI REQUEST",
        ("ai", "usage"): "AI TOKEN USAGE",
        ("voice", "intro"): "VOICE INTRO",
        ("voice", "outro"): "VOICE OUTRO",
        ("system", "ready"): "SYSTEM READY",
        ("system", "obs"): "OBS",
        ("error", "warning"): "WARNING",
    }.get((event_type, action), f"{event_type.upper()} {action.upper()}".strip())

    suffix = f" [{category}]" if category and category.upper() not in label else ""
    ts = format_ts(str(event.get("timestamp", "")))
    return f"- `{ts}` **{label}{suffix}** {event_text(event)}"


def build_markdown(events: list[dict[str, Any]], title: str) -> str:
    if not events:
        return f"# {title}\n\nNo events found.\n"

    dates = sorted({str(event.get("timestamp", ""))[:10] for event in events})
    counts = Counter(str(event.get("event_type", "event")) for event in events)
    segments = segment_by_job(events)

    lines = [
        f"# {title}",
        "",
        "## Corpus Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Date range | {dates[0]} to {dates[-1]} |",
        f"| Events retained | {len(events)} |",
        f"| Job segments | {sum(1 for segment in segments if segment['label'].startswith('Job'))} |",
    ]
    for event_type, count in sorted(counts.items()):
        lines.append(f"| {event_type} events | {count} |")
    lines.append("")

    lines.extend(["## Timeline", ""])
    for segment in segments:
        lines.extend([f"### {segment['label']}", ""])
        if "content" in segment:
            lines.extend(
                [
                    "| Field | Value |",
                    "|---|---|",
                    f"| Content | {segment.get('content', '')} |",
                    f"| Status | {segment.get('status', '')} |",
                    f"| Duration | {segment.get('duration', '')} |",
                    "",
                ]
            )

        milestones = drawing_milestones(segment["events"])
        if milestones:
            lines.extend(["#### Drawing Milestones", "", *milestones, ""])

        rendered = [line for event in segment["events"] if (line := render_event(event))]
        if rendered:
            lines.extend(["#### Ledger Events", "", *rendered, ""])

    ai_prompts = [
        event_text(event)
        for event in events
        if event.get("event_type") == "ai" and event.get("event_action") == "request"
    ]
    lines.extend(["## Query Seeds", ""])
    if ai_prompts:
        lines.extend(f"- AI prompt: {prompt}" for prompt in ai_prompts)
    else:
        lines.append("- No AI prompts recorded.")
    lines.extend(
        [
            "",
            f"*Generated by `ledger_to_md.py` at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.*",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--title", default="Sample Robot Run")
    args = parser.parse_args()

    events = load_events(args.input)
    markdown = build_markdown(events, args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output} from {len(events)} retained events")


if __name__ == "__main__":
    main()
