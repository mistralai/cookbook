"""Schema normalization + field-check helpers for the code-gen extractor.

Pydantic-style JSON schemas express nullable fields as ``anyOf: [{type: X}, {type:
null}]`` (or ``oneOf``), nested models as ``$defs`` + ``$ref`` (sometimes wrapped
in a field-level ``allOf``, Pydantic-v1 style), and ``dict[str, T]`` fields as
``additionalProperties``. Left unresolved, the prompt renderer shows ``array of
any`` and the validity gate either silently passes any field under such a node or
— worse, for ``additionalProperties`` — rejects every key. These helpers collapse
all of those shapes so the renderer and gate can descend into the real structure:

  * ``resolve_refs`` — inline local ``$ref``/``$defs`` (cycle-guarded).
  * ``_effective`` — merge ``allOf`` and collapse ``anyOf``/``oneOf`` wrappers.
  * ``render_output_schema`` — render a schema as an indented field tree.
  * ``unknown_output_fields`` / ``missing_required_fields`` — gate checks.

``additionalProperties`` semantics (a deliberate deviation from the JSON-Schema
default, because the gate's purpose is to force schema adherence): absent or
``false`` → unknown keys are flagged; a schema dict → keys are allowed and each
*value* is checked against that subschema; ``true`` → keys are allowed unchecked.

Ported from doc-extraction-gt (generalized/phase2_harness/generate_phase2.py +
extract_harness/generate_extract.py); self-contained (stdlib only).
"""

from __future__ import annotations

import re
from typing import Any

# Reserved output key for source attribution (page-level provenance). It is NOT
# part of any schema and is never schema-scored; the validity gate only *tolerates*
# it (via ``unknown_output_fields(..., allow_keys={PROVENANCE_KEY})``) when the
# provenance experiment is on. See the page-provenance plan/journal.
PROVENANCE_KEY = "_provenance"


def _effective(schema: dict[str, Any]) -> dict[str, Any]:
    """Collapse Pydantic/JSON-Schema combinator wrappers into one effective node.

    ``allOf`` (every branch applies — Pydantic v1 wraps a ``$ref`` plus a
    field-level ``description`` this way) is merged first: branch ``properties``
    are unioned, ``required`` concatenated (deduped), and the outer node's own
    keys (e.g. ``description``) win. ``anyOf``/``oneOf`` (nullable wrappers and
    unions — for gating purposes exclusive-or behaves like or) are then collapsed:
    the real structure (``properties``/``items``/``enum``/``additionalProperties``)
    lives in the non-null branch and there is no top-level ``type``; this merges
    that structure with the outer ``description``/``title`` and records the union
    of branch types (incl. ``null``) as ``type``. Idempotent on plain schemas; for
    a true multi-type union it takes the first branch that carries structure.
    """
    if not isinstance(schema, dict):
        return schema
    if "allOf" in schema:
        merged_all: dict[str, Any] = {k: v for k, v in schema.items() if k != "allOf"}
        props: dict[str, Any] = dict(merged_all.get("properties") or {})
        required: list[str] = list(merged_all.get("required") or [])
        for b in schema["allOf"]:
            if not isinstance(b, dict):
                continue
            eb = _effective(b)
            props.update(eb.get("properties") or {})
            required += [r for r in eb.get("required") or [] if r not in required]
            for key in ("type", "items", "enum", "additionalProperties"):
                if key in eb and key not in merged_all:
                    merged_all[key] = eb[key]
            if "description" not in merged_all and eb.get("description"):
                merged_all["description"] = eb["description"]
        if props:
            merged_all["properties"] = props
        if required:
            merged_all["required"] = required
        schema = merged_all
    union_key = "anyOf" if "anyOf" in schema else "oneOf" if "oneOf" in schema else None
    if union_key is None:
        return schema
    merged: dict[str, Any] = {k: v for k, v in schema.items() if k != union_key}
    types: list[str] = []
    for b in schema[union_key]:
        if not isinstance(b, dict):
            continue
        b = _effective(b)  # a branch may itself wrap allOf/$ref structure
        bt = b.get("type")
        for t in bt if isinstance(bt, list) else [bt]:
            if t and t not in types:
                types.append(t)
        if b.get("type") != "null":
            for key in ("properties", "items", "required", "enum", "additionalProperties"):
                if key in b and key not in merged:
                    merged[key] = b[key]
            if "description" not in merged and b.get("description"):
                merged["description"] = b["description"]
    if types:
        merged["type"] = types
    return merged


def resolve_refs(schema: Any, root: dict[str, Any] | None = None, _seen: frozenset[str] = frozenset()) -> Any:
    """Inline all local ``$ref`` pointers (``#/$defs/X`` or ``#/definitions/X``).

    Pydantic emits nested models as a top-level ``$defs`` block plus
    ``{"$ref": "#/$defs/Model"}`` at the use site. Without resolution the renderer
    shows ``array of any`` and the gate can't enforce anything. Cycle-guarded: a
    ``$ref`` already being expanded on the current path resolves to ``{}``. ``$ref``
    siblings (e.g. a co-located ``description``) are preserved and win.
    """
    if root is None and isinstance(schema, dict):
        root = schema
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            if ref in _seen:
                return {}
            target: Any = root
            for part in ref[2:].split("/"):
                target = target.get(part, {}) if isinstance(target, dict) else {}
            resolved = resolve_refs(target, root, _seen | {ref})
            out: dict[str, Any] = dict(resolved) if isinstance(resolved, dict) else {}
            out.update({k: resolve_refs(v, root, _seen) for k, v in schema.items() if k != "$ref"})
            return out
        return {k: resolve_refs(v, root, _seen) for k, v in schema.items()}
    if isinstance(schema, list):
        return [resolve_refs(v, root, _seen) for v in schema]
    return schema


def _typestr(s: dict[str, Any]) -> str:
    s = _effective(s)
    t = s.get("type")
    return "|".join(x for x in t if x) if isinstance(t, list) else (t or "any")


def render_output_schema(
    schema: dict[str, Any], name: str | None = None, indent: int = 0, max_desc: int | None = None
) -> str:
    """Render a JSON schema as an indented field tree. ``max_desc`` caps each
    field description (whitespace-collapsed to one line); ``None`` = full text
    (descriptions often carry the extraction conventions, so show them in full)."""
    schema = _effective(schema)
    pad = "    " * indent
    raw_types = schema.get("type")
    types = raw_types if isinstance(raw_types, list) else [raw_types]
    desc = schema.get("description")
    if desc:
        d = re.sub(r"\s+", " ", desc).strip()
        dtxt = f"  — {d if max_desc is None else d[:max_desc]}"
    else:
        dtxt = ""
    out: list[str] = []
    if "object" in types:
        child = indent
        if name is not None:
            out.append(f"{pad}{name}: object{dtxt}")
            child = indent + 1
        for k, v in (schema.get("properties") or {}).items():
            out.append(render_output_schema(v, k, child, max_desc))
        addl = schema.get("additionalProperties")
        if isinstance(addl, dict) and addl:  # dict[str, T]: show the value structure
            out.append(render_output_schema(addl, "<any key>", child, max_desc))
    elif "array" in types:
        items = _effective(schema.get("items") or {})
        raw_itypes = items.get("type")
        itypes = raw_itypes if isinstance(raw_itypes, list) else [raw_itypes]
        kind = "object, each:" if "object" in itypes else _typestr(items)
        out.append(f"{pad}{name}: array of {kind}{dtxt}")
        if "object" in itypes:
            for k, v in (items.get("properties") or {}).items():
                out.append(render_output_schema(v, k, indent + 1, max_desc))
    else:
        out.append(f"{pad}{name}: {_typestr(schema)}{dtxt}")
    return "\n".join(out)


def unknown_output_fields(
    value: Any, schema: dict[str, Any], path: str = "", allow_keys: frozenset[str] = frozenset()
) -> list[str]:
    """Paths of output fields not present in the schema (recurses objects + array
    items; resolves combinators so it descends into nested structure). Keys not in
    ``properties`` are flagged unless ``additionalProperties`` is ``true`` (allowed
    unchecked) or a schema dict (allowed; the value is checked against it) — see
    the module docstring for the deliberate strict-by-default choice. ``allow_keys``
    are reserved keys (e.g. ``PROVENANCE_KEY``) tolerated at every object level and
    never recursed into — empty by default, so the gate is unchanged unless a caller
    opts a key in."""
    schema = _effective(schema)
    raw_types = schema.get("type")
    types = raw_types if isinstance(raw_types, list) else [raw_types]
    errs: list[str] = []
    if "object" in types and isinstance(value, dict):
        props = schema.get("properties") or {}
        addl = schema.get("additionalProperties")
        for k, v in value.items():
            if k in props:
                errs += unknown_output_fields(v, props[k], f"{path}/{k}", allow_keys)
            elif k in allow_keys:
                continue  # reserved key (e.g. _provenance) — tolerated, not scored
            elif isinstance(addl, dict):
                errs += unknown_output_fields(v, addl, f"{path}/{k}", allow_keys)
            elif addl is not True:
                errs.append(f"{path}/{k}")
    elif "array" in types and isinstance(value, list):
        items = _effective(schema.get("items") or {})
        for i, el in enumerate(value):
            errs += unknown_output_fields(el, items, f"{path}[{i}]", allow_keys)
    return errs


def missing_required_fields(value: Any, schema: dict[str, Any], path: str = "") -> list[str]:
    """Paths of ``required`` keys absent from the corresponding object (recurses;
    presence-only — a null value satisfies ``required``, per JSON-Schema; enum /
    non-null constraints are out of scope)."""
    schema = _effective(schema)
    raw_types = schema.get("type")
    types = raw_types if isinstance(raw_types, list) else [raw_types]
    errs: list[str] = []
    if "object" in types and isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errs.append(f"{path}/{req}")
        props = schema.get("properties") or {}
        addl = schema.get("additionalProperties")
        for k, v in value.items():
            if k in props:
                errs += missing_required_fields(v, props[k], f"{path}/{k}")
            elif isinstance(addl, dict):
                errs += missing_required_fields(v, addl, f"{path}/{k}")
    elif "array" in types and isinstance(value, list):
        items = _effective(schema.get("items") or {})
        for i, el in enumerate(value):
            errs += missing_required_fields(el, items, f"{path}[{i}]")
    return errs


def _matches_type(value: Any, type_name: str) -> bool:
    """One JSON-Schema type token vs a Python value. bool is excluded from
    number/integer (Python bool subclasses int; JSON booleans aren't numbers);
    integer accepts integral floats per JSON-Schema. Unknown tokens are
    permissive."""
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if type_name == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "null":
        return value is None
    return True


def invalid_output_values(value: Any, schema: dict[str, Any], path: str = "") -> list[str]:
    """Violation messages (``path: detail``) for non-null values that don't match
    the schema's declared ``type`` or ``enum`` (recurses objects + array items;
    resolves combinators). Null is exempt everywhere — the gate reads null as
    "not found", consistent with ``missing_required_fields``' presence-only
    stance. A value of the wrong container type yields one error and is not
    recursed into."""
    schema = _effective(schema)
    if value is None:
        return []
    where = path or "/"
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        return [f"{where}: {value!r} is not one of the allowed values {enum!r}"]
    raw_types = schema.get("type")
    types = [t for t in (raw_types if isinstance(raw_types, list) else [raw_types]) if isinstance(t, str)]
    if types and not any(_matches_type(value, t) for t in types):
        shown = repr(value)
        return [f"{where}: expected {'/'.join(types)}, got {type(value).__name__} ({shown[:60]})"]
    errs: list[str] = []
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        addl = schema.get("additionalProperties")
        for k, v in value.items():
            if k in props:
                errs += invalid_output_values(v, props[k], f"{path}/{k}")
            elif isinstance(addl, dict):
                errs += invalid_output_values(v, addl, f"{path}/{k}")
    elif isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, el in enumerate(value):
                errs += invalid_output_values(el, items, f"{path}[{i}]")
    return errs
