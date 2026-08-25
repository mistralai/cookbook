"""Shared helpers for deterministic test-rule identifiers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_rule_signature(rule: dict[str, Any]) -> str:
    """Return a canonical JSON signature for a rule without its id."""
    payload = dict(rule)
    payload.pop("id", None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_rule_id(rule: dict[str, Any], hash_len: int) -> str:
    """Compute the deterministic rule id used by `scripts/assign_rule_ids.py`."""
    signature = canonical_rule_signature(rule)
    page = rule.get("page")
    page_prefix = str(page) if page is not None else ""
    payload = f"{page_prefix}\u0000{signature}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:hash_len]
