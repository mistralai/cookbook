"""Path-based helpers for extract_field test rules.

Field paths use dotted + bracketed notation matching the annotator tree
and the source-export corpora, e.g.:

    "po_number"
    "buyer.company"
    "line_items[0].description"
    "employees[12].name"
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_TOKEN_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")

_MISSING = object()


def parse_field_path(path: str) -> list[str | int]:
    """Turn "line_items[0].description" into ["line_items", 0, "description"]."""
    if not path:
        raise ValueError("field_path must be non-empty")
    tokens: list[str | int] = []
    for match in _TOKEN_RE.finditer(path):
        key, index = match.groups()
        if key is not None:
            tokens.append(key)
        else:
            tokens.append(int(index))
    return tokens


def set_path(
    target: dict[str, Any] | list[Any],
    tokens: Iterable[str | int],
    value: Any,
) -> None:
    """Set a value at a parsed path, creating intermediate containers as needed.

    Rules:
      - string token → require dict, create/descend into key.
      - integer token → require list, pad with None up to index, create/descend.
      - At final token, assign.

    Raises TypeError if a container shape mismatch is encountered.
    """
    tokens_list = list(tokens)
    cursor: Any = target
    for i, tok in enumerate(tokens_list):
        is_last = i == len(tokens_list) - 1
        if isinstance(tok, str):
            if not isinstance(cursor, dict):
                raise TypeError(f"expected dict for key {tok!r} at position {i}, got {type(cursor).__name__}")
            if is_last:
                cursor[tok] = value
                return
            next_tok = tokens_list[i + 1]
            default: Any = [] if isinstance(next_tok, int) else {}
            cursor.setdefault(tok, default)
            cursor = cursor[tok]
        else:  # int
            if not isinstance(cursor, list):
                raise TypeError(f"expected list for index {tok} at position {i}, got {type(cursor).__name__}")
            while len(cursor) <= tok:
                cursor.append(None)
            if is_last:
                cursor[tok] = value
                return
            next_tok = tokens_list[i + 1]
            default = [] if isinstance(next_tok, int) else {}
            if cursor[tok] is None:
                cursor[tok] = default
            cursor = cursor[tok]


def get_path(source: Any, tokens: Iterable[str | int], default: Any = None) -> Any:
    """Return the value at a parsed path or `default` if any segment is missing."""
    cursor: Any = source
    for tok in tokens:
        if isinstance(tok, str):
            if not isinstance(cursor, dict) or tok not in cursor:
                return default
            cursor = cursor[tok]
        else:
            if not isinstance(cursor, list) or tok < 0 or tok >= len(cursor):
                return default
            cursor = cursor[tok]
    return cursor


def inflate_expected_output(rules: Iterable[Any]) -> dict[str, Any]:
    """Rebuild the nested expected_output dict from a flat list of extract_field rules.

    Only rules whose `field_path` parses without error are used. Duplicate paths
    (multiple evidence rules on the same cell — should only happen for historical
    reasons) keep the first non-None value.
    """
    out: dict[str, Any] = {}
    for rule in rules:
        field_path = getattr(rule, "field_path", None)
        if field_path is None and isinstance(rule, dict):
            field_path = rule.get("field_path")
        if not field_path:
            continue
        if isinstance(rule, dict):
            value = rule.get("expected_value")
        else:
            value = getattr(rule, "expected_value", None)
        try:
            tokens = parse_field_path(field_path)
        except ValueError:
            continue
        try:
            existing = get_path(out, tokens, default=_MISSING)
        except Exception:
            existing = _MISSING
        if existing is _MISSING or existing is None:
            set_path(out, tokens, value)
    return out


def validate_rules_match_expected_output(
    rules: Iterable[Any],
    expected_output: dict[str, Any] | None,
) -> list[str]:
    """Return a list of drift messages; empty list == exact match."""
    if expected_output is None:
        return []
    inflated = inflate_expected_output(rules)
    return _compare_json(inflated, expected_output, path="")


def _compare_json(a: Any, b: Any, *, path: str) -> list[str]:
    drifts: list[str] = []
    if type(a) is not type(b) and not (a is None or b is None):
        drifts.append(f"{path or '<root>'}: type mismatch ({type(a).__name__} vs {type(b).__name__})")
        return drifts
    if isinstance(a, dict) and isinstance(b, dict):
        for key in set(a) | set(b):
            sub_path = f"{path}.{key}" if path else key
            if key not in a:
                drifts.append(f"{sub_path}: missing in rules-derived")
            elif key not in b:
                drifts.append(f"{sub_path}: missing in expected_output")
            else:
                drifts.extend(_compare_json(a[key], b[key], path=sub_path))
        return drifts
    if isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            sub_path = f"{path}[{i}]"
            if i >= len(a):
                drifts.append(f"{sub_path}: missing in rules-derived")
            elif i >= len(b):
                drifts.append(f"{sub_path}: missing in expected_output")
            else:
                drifts.extend(_compare_json(a[i], b[i], path=sub_path))
        return drifts
    if a != b:
        drifts.append(f"{path or '<root>'}: value mismatch ({a!r} vs {b!r})")
    return drifts
