from __future__ import annotations

import re
from typing import Any


def path_tokens(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for part in path.split("."):
        if not part:
            continue
        match = re.match(r"^([^\[]+)", part)
        if match:
            tokens.append(match.group(1))
        for index in re.findall(r"\[(\d+)\]", part):
            tokens.append(int(index))
    return tokens


def format_path(tokens: list[str | int]) -> str:
    out = ""
    for token in tokens:
        if isinstance(token, int):
            out = f"{out}[{token}]"
        elif out:
            out = f"{out}.{token}"
        else:
            out = token
    return out


def value_at(source: Any, tokens: list[str | int]) -> tuple[bool, Any]:
    cursor = source
    for token in tokens:
        if isinstance(token, int):
            if not isinstance(cursor, list) or token >= len(cursor):
                return False, None
            cursor = cursor[token]
        else:
            if not isinstance(cursor, dict) or token not in cursor:
                return False, None
            cursor = cursor[token]
    return True, cursor


def iter_leaf_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        leaves: list[tuple[str, Any]] = []
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else key
            leaves.extend(iter_leaf_values(child, child_prefix))
        return leaves
    if isinstance(value, list):
        leaves = []
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            leaves.extend(iter_leaf_values(child, child_prefix))
        return leaves
    return [(prefix, value)] if prefix else []


def iter_array_values(value: Any, prefix: str = "") -> list[tuple[str, list[Any]]]:
    if isinstance(value, dict):
        arrays: list[tuple[str, list[Any]]] = []
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else key
            arrays.extend(iter_array_values(child, child_prefix))
        return arrays
    if isinstance(value, list):
        arrays = [(prefix, value)] if prefix else []
        for index, child in enumerate(value):
            arrays.extend(iter_array_values(child, f"{prefix}[{index}]"))
        return arrays
    return []


def resolve_schema_ref(root: Any, node: Any) -> Any:
    seen: set[str] = set()
    while isinstance(node, dict) and isinstance(node.get("$ref"), str):
        ref = node["$ref"]
        if ref in seen or not ref.startswith("#/"):
            return node
        seen.add(ref)
        cursor = root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(cursor, dict) or part not in cursor:
                return node
            cursor = cursor[part]
        node = cursor
    return node


def concrete_schema_node(root: Any, node: Any) -> Any:
    node = resolve_schema_ref(root, node)
    while isinstance(node, dict):
        variants = node.get("anyOf") or node.get("oneOf")
        if not isinstance(variants, list):
            break
        candidates = [
            resolve_schema_ref(root, candidate)
            for candidate in variants
            if not (isinstance(candidate, dict) and candidate.get("type") == "null")
        ]
        if not candidates:
            break
        preferred = None
        for candidate in candidates:
            candidate = resolve_schema_ref(root, candidate)
            if isinstance(candidate, dict) and (
                candidate.get("type") in {"object", "array"} or "properties" in candidate or "items" in candidate
            ):
                preferred = candidate
                break
        node = preferred if preferred is not None else candidates[0]
        node = resolve_schema_ref(root, node)
    return node


def schema_at_path(schema: Any, tokens: list[str | int]) -> Any:
    cursor = schema
    for token in tokens:
        cursor = concrete_schema_node(schema, cursor)
        if not isinstance(cursor, dict):
            return None
        if isinstance(token, int):
            cursor = concrete_schema_node(schema, cursor.get("items"))
            continue
        if cursor.get("type") == "array" or ("items" in cursor and "properties" not in cursor):
            cursor = concrete_schema_node(schema, cursor.get("items"))
            if not isinstance(cursor, dict):
                return None
        props = cursor.get("properties")
        if not isinstance(props, dict) or token not in props:
            return None
        cursor = props[token]
    return concrete_schema_node(schema, cursor)


def schema_description(schema: Any, path: str) -> str:
    node = schema_at_path(schema, path_tokens(path))
    if isinstance(node, dict):
        return str(node.get("description") or "")
    return ""


def schema_path_exists(schema: Any, path: str) -> bool:
    return schema_at_path(schema, path_tokens(path)) is not None


def is_array_schema(schema: Any, path: str) -> bool:
    node = schema_at_path(schema, path_tokens(path))
    return isinstance(node, dict) and (node.get("type") == "array" or "items" in node)


def field_name(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path).split(".")[-1]


def parent_path_for_array_item(path: str) -> tuple[str, int, str] | None:
    tokens = path_tokens(path)
    for pos in range(len(tokens) - 1, -1, -1):
        index = tokens[pos]
        if isinstance(index, int):
            parent = format_path(tokens[:pos])
            suffix = format_path(tokens[pos + 1 :])
            return parent, index, suffix
    return None


def path_without_indices(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)
