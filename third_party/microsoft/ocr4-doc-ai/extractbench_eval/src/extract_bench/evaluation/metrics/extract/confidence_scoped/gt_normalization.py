"""Declaration-driven GT/schema normalization for lifted repeated structures.

The submission path copies ``repeated_structure`` entries into top-level
``properties``, so extraction emits those keys top-level while some datasets
author the same rows nested under a parent array. Normalizing both sides here
keeps ``schema_valid`` truthful and lets alignment compare like shapes; no-op
without a declaration.
"""

from __future__ import annotations

from typing import Any

# Every identity spelling the alignment layer honors; the lift must strip the
# same set (plus provider-only gates) so the vocabularies cannot drift.
REPEATED_STRUCTURE_IDENTITY_SPELLINGS = ("identity_key", "identity_keys", "match_by")
_REPEATED_STRUCTURE_ONLY_KEYS = (*REPEATED_STRUCTURE_IDENTITY_SPELLINGS, "hallucination_gate")


def lift_repeated_structure_schema(schema: Any) -> Any:
    """Mirror the submission-side transform: copy ``repeated_structure``
    entries into top-level ``properties`` (minus provider-only keys) so
    schema_valid/identity lookups see the fields the extraction was actually
    asked for. The ``repeated_structure`` key itself is preserved."""
    if not isinstance(schema, dict):
        return schema
    repeated = schema.get("repeated_structure")
    if not isinstance(repeated, dict) or not repeated:
        return schema
    out = dict(schema)
    properties = dict(out.get("properties") or {})
    for name, definition in repeated.items():
        if isinstance(definition, dict) and name not in properties:
            properties[name] = {
                key: value for key, value in definition.items() if key not in _REPEATED_STRUCTURE_ONLY_KEYS
            }
    out["properties"] = properties
    return out


def lift_repeated_structure_gt(gt_doc: Any, schema: Any) -> Any:
    """MOVE nested rows of each declared key to the GT top level (parent-row
    order preserved, nested copies removed); no-op when already top-level."""
    if not isinstance(gt_doc, dict) or not isinstance(schema, dict):
        return gt_doc
    repeated = schema.get("repeated_structure")
    if not isinstance(repeated, dict) or not repeated:
        return gt_doc
    out: dict[str, Any] | None = None
    for key in repeated.keys():
        if not isinstance(key, str) or key in gt_doc:
            continue
        lifted_rows: list[Any] = []
        parents_with_key: list[str] = []
        for parent_key, parent_value in gt_doc.items():
            if not isinstance(parent_value, list):
                continue
            if any(isinstance(row, dict) and isinstance(row.get(key), list) for row in parent_value):
                parents_with_key.append(parent_key)
                for row in parent_value:
                    if isinstance(row, dict) and isinstance(row.get(key), list):
                        lifted_rows.extend(row[key])
        if not lifted_rows:
            continue
        if out is None:
            out = dict(gt_doc)
        for parent_key in parents_with_key:
            out[parent_key] = [
                ({k: v for k, v in row.items() if k != key} if isinstance(row, dict) else row)
                for row in out[parent_key]
            ]
        out[key] = lifted_rows
    return out if out is not None else gt_doc
