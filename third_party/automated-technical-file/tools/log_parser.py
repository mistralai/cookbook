#!/usr/bin/env python3
"""
Robot Arm Log Parser for ATF (Automated Technical File)

Parses raw robot arm logs into a normalized event structure.
These logs contain robot arm operations, drawing commands, job lifecycle events,
and system status messages.

Usage:
    python log_parser.py <log_path> [--output output.jsonl]
    python log_parser.py --directory <log_dir> --output output.jsonl
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class NormalizedEvent:
    """Normalized event schema for robot arm logs."""
    # Core identification
    event_id: str
    timestamp: str
    timestamp_epoch: float

    # Event classification
    event_type: str           # e.g., "system", "job", "drawing", "motion", "voice", "error"
    event_category: str       # e.g., "JOB", "DRAWING", "GCODE", "PEN", "NARRATION", "CHECK"
    event_action: str         # e.g., "start", "end", "down", "up", "ready", "failed"

    # Context and provenance
    source_file: str
    line_number: int
    raw_line: str

    # Event-specific payloads
    details: Dict[str, Any]

    # Metadata
    parsed_at: str
    parser_version: str = "1.0.0"


# Event type mappings based on log analysis
EVENT_CATEGORY_MAP = {
    # Job lifecycle
    "JOB START": ("job", "JOB", "start"),
    "JOB END": ("job", "JOB", "end"),

    # Drawing operations
    "PEN DOWN": ("drawing", "PEN", "down"),
    "PEN UP": ("drawing", "PEN", "up"),
    "DRAWING": ("drawing", "DRAWING", "progress"),
    "DRAW START": ("drawing", "DRAW", "start"),
    "DRAW END": ("drawing", "DRAW", "end"),
    "BUSY": ("job", "BUSY", "active"),

    # G-code execution
    "GCODE": ("motion", "GCODE", "execute"),

    # Voice and narration
    "NARRATION": ("voice", "NARRATION", "status"),
    "VOICE INTRO": ("voice", "VOICE", "intro"),
    "VOICE OUTRO": ("voice", "VOICE", "outro"),
    "WARNING TONE": ("voice", "WARNING", "play"),

    # System checks
    "CHECK": ("system", "CHECK", "status"),
    "PORT": ("system", "PORT", "status"),

    # Media and streaming
    "OBS": ("system", "OBS", "status"),
    "CAFFEINATE": ("system", "CAFFEINATE", "status"),

    # AI interactions
    "AI TOKENS": ("ai", "AI", "usage"),
    "SKETCH": ("ai", "SKETCH", "request"),
    "SKETCH PREVIEW": ("ai", "SKETCH", "preview"),
    "APPROVE": ("ai", "APPROVE", "confirm"),

    # Errors
    "ERROR": ("error", "ERROR", "occurred"),
    "MARKERS ERROR": ("error", "MARKERS", "failed"),
    "SUBPROCESS": ("error", "SUBPROCESS", "issue"),

    # Other
    "STOP": ("system", "STOP", "signal"),
    "Calibration": ("system", "CALIBRATION", "run"),
    "LOCK": ("system", "LOCK", "status"),
    "DRY": ("system", "DRY", "mode"),
}

# Regex patterns for extracting details from event lines
DETAIL_PATTERNS = {
    "JOB START": re.compile(r'action=(\w+).*content=[\'"]([^\'"]+)[\'"]'),
    "JOB END": re.compile(r'status=(\w+).*duration=([\d.]+)s'),
    "DRAWING": re.compile(r'(\d+)/(\d+) moves \((\d+)%\)*\s*(?:.\s*)?X:([+-]?[\d.]+)\s*Y:([+-]?[\d.]+)mm'),
    "PEN DOWN": re.compile(r'drawing line (\d+)/(\d+)'),
    "AI TOKENS": re.compile(r'TOKENS — input=(\d+) output=(\d+) total=(\d+)'),
    "SKETCH": re.compile(r'prompt=[\'"]([^\'"]+)[\'"]'),
    "CHECK": re.compile(r'(.+)'),
    "PORT": re.compile(r'(.+)'),
    "OBS": re.compile(r'(.+)'),
    "CAFFEINATE": re.compile(r'(.+)'),
    "VOICE INTRO": re.compile(r'(.+)'),
    "VOICE OUTRO": re.compile(r'(.+)'),
    "ERROR": re.compile(r'(.+)'),
    "NARRATION": re.compile(r'(.+)'),
}


def parse_timestamp(ts_str: str) -> tuple:
    """Parse timestamp string into datetime and epoch."""
    try:
        dt = datetime.strptime(ts_str.strip('[]'), "%Y-%m-%d %H:%M:%S")
        epoch = dt.timestamp()
        return dt.isoformat() + "Z", epoch
    except ValueError:
        return ts_str, 0.0


def extract_event_key(line: str) -> str:
    """Extract the event key (part after timestamp before separator)."""
    match = re.match(r'\[.*?\]\s+([A-Za-z][A-Za-z0-9 ]+)', line)
    if match:
        return match.group(1).strip()
    return "UNKNOWN"


def get_event_classification(event_key: str) -> tuple:
    """Map event key to normalized classification."""
    upper_key = event_key.upper()
    if event_key in EVENT_CATEGORY_MAP:
        return EVENT_CATEGORY_MAP[event_key]
    for key, value in EVENT_CATEGORY_MAP.items():
        if key.startswith(upper_key) or upper_key.startswith(key.replace(" ", "")):
            return value
    return ("unknown", "UNKNOWN", "unknown")


def extract_details(event_key: str, line: str) -> Dict[str, Any]:
    """Extract structured details from the log line."""
    details = {}

    match = re.match(r'\[.*?\]\s+[A-Za-z][A-Za-z0-9 ]*\s*(.*)', line)
    if not match:
        return details

    rest = match.group(1).strip()
    rest = re.sub(r'[\x80-\xFF]', '|', rest)

    upper_key = event_key.upper()

    try:
        if upper_key == "JOB START":
            m = re.search(r'action=(\w+)', rest)
            if m:
                details["action"] = m.group(1)
            m = re.search(r'content=[\'"]([^\'"]+)[\'"]', rest)
            if m:
                details["content"] = m.group(1)
            m = re.search(r'size=([\d.]+)', rest)
            if m:
                details["size"] = float(m.group(1))

        elif upper_key == "JOB END":
            m = re.search(r'status=(\w+)', rest)
            if m:
                details["status"] = m.group(1)
            m = re.search(r'duration=([\d.]+)s', rest)
            if m:
                details["duration_seconds"] = float(m.group(1))

        elif upper_key == "DRAWING":
            m = re.search(r'(\d+)/(\d+) moves', rest)
            if m:
                details["move_current"] = int(m.group(1))
                details["move_total"] = int(m.group(2))
                details["percent_complete"] = int(m.group(1)) / int(m.group(2)) * 100
            m = re.search(r'X:([+-]?[\d.]+)', rest)
            if m:
                details["x_mm"] = float(m.group(1))
            m = re.search(r'Y:([+-]?[\d.]+)mm', rest)
            if m:
                details["y_mm"] = float(m.group(1))

        elif upper_key == "PEN DOWN":
            m = re.search(r'drawing line (\d+)/(\d+)', rest)
            if m:
                details["line_number"] = int(m.group(1))
                details["line_total"] = int(m.group(2))

        elif upper_key == "AI TOKENS":
            m = re.search(r'input=(\d+) output=(\d+) total=(\d+)', rest)
            if m:
                details["input_tokens"] = int(m.group(1))
                details["output_tokens"] = int(m.group(2))
                details["total_tokens"] = int(m.group(3))

        elif upper_key == "SKETCH":
            m = re.search(r'prompt=[\'"]([^\'"]+)[\'"]', rest)
            if m:
                details["prompt"] = m.group(1)

        elif upper_key in ["NARRATION", "VOICE INTRO", "VOICE OUTRO", "WARNING TONE"]:
            parts = rest.split('|')
            if len(parts) > 1:
                details["message"] = parts[-1].strip()
            else:
                details["message"] = rest.strip()

        elif upper_key in ["CHECK", "PORT", "OBS", "CAFFEINATE", "ERROR"]:
            parts = rest.split('|')
            if len(parts) > 1:
                details["status"] = parts[-1].strip()
            else:
                details["status"] = rest.strip()
    except Exception as e:
        details["parse_error"] = str(e)

    details["raw_rest"] = rest

    return details


def generate_event_id(source_file: str, line_number: int, timestamp: str) -> str:
    """Generate a unique event ID."""
    import hashlib
    unique_str = f"{source_file}:{line_number}:{timestamp}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:16]


def parse_line(line: str, source_file: str, line_number: int) -> Optional[NormalizedEvent]:
    """Parse a single log line into a normalized event."""
    line = line.strip()
    if not line:
        return None

    ts_match = re.match(r'\[([^\]]+)\]', line)
    if not ts_match:
        return None

    timestamp_str = ts_match.group(1)
    iso_timestamp, epoch = parse_timestamp(timestamp_str)

    event_key = extract_event_key(line)
    if not event_key:
        return None

    event_type, event_category, event_action = get_event_classification(event_key)
    details = extract_details(event_key, line)
    details["event_key_raw"] = event_key

    event = NormalizedEvent(
        event_id=generate_event_id(source_file, line_number, iso_timestamp),
        timestamp=iso_timestamp,
        timestamp_epoch=epoch,
        event_type=event_type,
        event_category=event_category,
        event_action=event_action,
        source_file=source_file,
        line_number=line_number,
        raw_line=line,
        details=details,
        parsed_at=datetime.utcnow().isoformat() + "Z",
        parser_version="1.0.0"
    )

    return event


def parse_file(file_path: str) -> List[NormalizedEvent]:
    """Parse a log file and return all events."""
    events = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            event = parse_line(line, str(file_path), line_num)
            if event:
                events.append(event)

    return events


def parse_directory(directory: str) -> List[NormalizedEvent]:
    """Parse all log files in a directory."""
    all_events = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.log') or file.endswith('.txt'):
                file_path = os.path.join(root, file)
                print(f"Parsing: {file_path}", file=sys.stderr)
                events = parse_file(file_path)
                all_events.extend(events)
                print(f"  Found {len(events)} events", file=sys.stderr)

    return all_events


def save_events(events: List[NormalizedEvent], output_path: str):
    """Save events to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for event in events:
            event_dict = asdict(event)
            f.write(json.dumps(event_dict) + '\n')


def load_events(input_path: str) -> List[NormalizedEvent]:
    """Load events from JSONL file."""
    events = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                event_dict = json.loads(line)
                event = NormalizedEvent(**event_dict)
                events.append(event)
    return events


def print_stats(events: List[NormalizedEvent]):
    """Print statistics about parsed events."""
    print("\n=== Parser Statistics ===", file=sys.stderr)
    print(f"Total events parsed: {len(events)}", file=sys.stderr)

    category_counts = {}
    for event in events:
        cat = event.event_category
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\nBy Category:", file=sys.stderr)
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}", file=sys.stderr)

    type_counts = {}
    for event in events:
        etype = event.event_type
        type_counts[etype] = type_counts.get(etype, 0) + 1

    print("\nBy Type:", file=sys.stderr)
    for etype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {etype}: {count}", file=sys.stderr)

    if events:
        timestamps = [e.timestamp for e in events if e.timestamp]
        if timestamps:
            print(f"\nTime range: {min(timestamps)} to {max(timestamps)}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Parse robot arm logs into normalized events"
    )
    parser.add_argument(
        'path',
        nargs='?',
        help='Path to log file or directory'
    )
    parser.add_argument(
        '--directory', '-d',
        help='Directory containing log files'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print parsing statistics'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test parsing on a sample of lines'
    )

    args = parser.parse_args()

    input_path = args.directory if args.directory else args.path
    if not input_path:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if os.path.isdir(input_path):
        events = parse_directory(input_path)
    else:
        events = parse_file(input_path)

    if args.output:
        save_events(events, args.output)
        print(f"Saved {len(events)} events to {args.output}", file=sys.stderr)

    if args.stats:
        print_stats(events)

    if args.test:
        print("\n=== Sample Events ===", file=sys.stderr)
        for event in events[:5]:
            print(json.dumps(asdict(event), indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
