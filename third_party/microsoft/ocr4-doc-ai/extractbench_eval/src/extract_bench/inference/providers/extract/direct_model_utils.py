"""Shared helpers for model-native direct extraction providers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from extract_bench.inference.providers.base import ProviderPermanentError
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline_io import InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType

IMAGE_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are extracting structured data from a document according to the "
    "provided JSON schema. Return only the JSON that matches the schema. Use "
    "null for fields not present in the document. When the schema includes a "
    "list field, populate every relevant row visible in the document - do not "
    "return an empty list when rows are present."
)

DEFAULT_USER_INSTRUCTION = (
    "Extract every field from the attached document according to the schema. "
    "Return JSON only. Use null for fields not present in the document. "
    "Whenever the schema declares a list field, enumerate every row visible in "
    "the document - do not collapse rows or return an empty list when rows are "
    "present."
)


def promote_repeated_structure(schema: dict[str, Any]) -> dict[str, Any]:
    """Promote ``repeated_structure[name]`` entries into ``properties[name]``.

    LlamaExtract schemas declare repeated array nodes under a top-level
    ``repeated_structure`` key. Standard JSON Schema consumers (Anthropic,
    OpenAI, Gemini) either reject the unknown key or silently ignore it,
    which drops the repeated fields. This adapter rewrites the schema into
    plain JSON Schema by copying each entry into ``properties`` and
    dropping the ``repeated_structure`` key.
    """
    if not isinstance(schema, dict):
        return schema
    repeated_structure = schema.get("repeated_structure")
    if not isinstance(repeated_structure, dict) or not repeated_structure:
        return schema
    out = dict(schema)
    properties = dict(out.get("properties") or {})
    for name, definition in repeated_structure.items():
        if isinstance(definition, dict) and name not in properties:
            properties[name] = definition
    out["properties"] = properties
    out.pop("repeated_structure", None)
    return out


def add_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a schema with every object closed via ``additionalProperties: false``."""
    out = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                for value in (node.get("properties") or {}).values():
                    walk(value)
            if "items" in node:
                walk(node["items"])
            for key in ("anyOf", "oneOf", "allOf"):
                for branch in node.get(key, []) or []:
                    walk(branch)
            for key in ("$defs", "definitions"):
                for definition in (node.get(key) or {}).values():
                    walk(definition)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(out)
    return out


def force_all_properties_required(schema: dict[str, Any]) -> dict[str, Any]:
    """Mark every property on every object as ``required``.

    Anthropic's structured-output grammar compiler caps the number of
    optional properties per object at 24. Schemas that already union each
    property with ``null`` (the LlamaParse extraction convention) can sidestep
    this by listing every property as required; the model still returns
    ``null`` for fields absent from the document, just inside a non-optional
    slot.
    """
    out = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict) and properties:
                node["required"] = list(properties.keys())
                for value in properties.values():
                    walk(value)
            if "items" in node:
                walk(node["items"])
            for key in ("anyOf", "oneOf", "allOf"):
                for branch in node.get(key, []) or []:
                    walk(branch)
            for key in ("$defs", "definitions"):
                for definition in (node.get(key) or {}).values():
                    walk(definition)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(out)
    return out


def pricing_for_model(model: str, pricing: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """Return the longest-prefix pricing match for ``model``."""
    matches = [(prefix, rate) for prefix, rate in pricing.items() if model.startswith(prefix)]
    return max(matches, key=lambda item: len(item[0]))[1] if matches else (0.0, 0.0)


def page_count(source_path: Path) -> int:
    """Return the direct-extract page count for an image/PDF source."""
    if source_path.suffix.lower() in IMAGE_EXTENSIONS:
        return 1
    try:
        return len(PdfReader(str(source_path)).pages)
    except Exception:
        return 0


def normalize_extract_result(raw_result: RawInferenceResult) -> InferenceResult:
    """Normalize direct model extraction output into the shared ``ExtractOutput`` shape."""
    if raw_result.product_type != ProductType.EXTRACT:
        raise ProviderPermanentError(
            f"Direct extract normalization only supports EXTRACT, got {raw_result.product_type}"
        )

    output = ExtractOutput(
        task_type="extract",
        example_id=raw_result.request.example_id,
        pipeline_name=raw_result.pipeline_name,
        extracted_data=raw_result.raw_output.get("data") or {},
        field_citations=[],
    )

    return InferenceResult(
        request=raw_result.request,
        pipeline_name=raw_result.pipeline_name,
        product_type=raw_result.product_type,
        raw_output=raw_result.raw_output,
        output=output,
        started_at=raw_result.started_at,
        completed_at=raw_result.completed_at,
        latency_in_ms=raw_result.latency_in_ms,
    )
