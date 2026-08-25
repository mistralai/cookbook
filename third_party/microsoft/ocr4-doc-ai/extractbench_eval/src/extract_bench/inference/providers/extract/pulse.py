"""Provider for Pulse EXTRACT.

Pulse's current structured extraction flow is two-step:

1. POST /extract with the source file to create a saved extraction.
2. POST /schema with that extraction_id plus the benchmark JSON schema.

The older ``structured_output`` option on /extract is deprecated, so this
provider follows the documented Extract -> Schema path.
"""

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

_API_BASE_URL = "https://api.runpulse.com"
_UNSUPPORTED_SCHEMA_KEYS = {"$defs", "$schema", "definitions", "default", "repeated_structure"}


@register_provider("pulse_extract")
class PulseExtractProvider(Provider):
    """Provider for Pulse structured extraction via REST."""

    CREDIT_RATE_USD = 0.015

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        api_key = self.base_config.get("api_key") or os.getenv("PULSE_API_KEY")
        if not api_key or not isinstance(api_key, str):
            raise ProviderConfigError(
                "Pulse API key is required. Set PULSE_API_KEY environment variable or pass api_key in base_config."
            )
        self._api_key = api_key

        self._api_base_url = str(self.base_config.get("api_base_url", _API_BASE_URL)).rstrip("/")
        self._model: str | None = self.base_config.get("model")
        self._pages: str | None = self.base_config.get("pages")
        self._timeout = float(self.base_config.get("timeout", 600))
        self._schema_prompt: str | None = self.base_config.get("schema_prompt")
        self._schema_effort = bool(self.base_config.get("effort", self.base_config.get("schema_effort", False)))
        self._estimate_schema_cost = bool(self.base_config.get("estimate_schema_cost", True))

        figure_processing = self.base_config.get("figure_processing")
        if figure_processing is not None and not isinstance(figure_processing, dict):
            raise ProviderConfigError("figure_processing must be a dict")
        self._figure_processing: dict[str, Any] | None = figure_processing

        extensions = self.base_config.get("extensions")
        if extensions is not None and not isinstance(extensions, dict):
            raise ProviderConfigError("extensions must be a dict")
        self._extensions: dict[str, Any] | None = extensions

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key}

    def _handle_response(self, response: requests.Response, *, context: str) -> dict[str, Any]:
        if response.status_code == 401:
            raise ProviderConfigError(f"Pulse auth failed during {context} (401): {response.text[:300]}")
        if response.status_code == 429:
            raise ProviderRateLimitError(f"Pulse rate limit during {context} (429): {response.text[:300]}")
        if response.status_code in (502, 503, 504):
            raise ProviderTransientError(
                f"Pulse transient during {context} ({response.status_code}): {response.text[:300]}"
            )
        if response.status_code >= 400:
            raise ProviderPermanentError(f"Pulse {context} failed ({response.status_code}): {response.text[:300]}")

        try:
            raw = response.json()
        except ValueError as e:
            raise ProviderPermanentError(f"Pulse returned non-JSON response during {context}: {e}") from e
        if not isinstance(raw, dict):
            raise ProviderPermanentError(f"Pulse returned unexpected {context} response type: {type(raw).__name__}")
        return raw

    def _fetch_large_result(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not raw.get("is_url") or not raw.get("url"):
            return raw

        response = requests.get(str(raw["url"]), headers=self._headers(), timeout=self._timeout)
        result = self._handle_response(response, context="large-result download")
        for key in ("plan_info", "plan-info", "credits_used", "page_count"):
            if key in raw and key not in result:
                result[key] = raw[key]
        return result

    def _build_extract_fields(self) -> list[tuple[str, tuple[None, str]]]:
        fields: list[tuple[str, tuple[None, str]]] = []

        def add(name: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, bool):
                fields.append((name, (None, "true" if value else "false")))
            elif isinstance(value, (dict, list)):
                fields.append((name, (None, json.dumps(value))))
            else:
                fields.append((name, (None, str(value))))

        add("model", self._model)
        add("pages", self._pages)
        add("figure_processing", self._figure_processing)
        add("extensions", self._extensions)
        return fields

    def _extract_file(self, file_path: Path) -> dict[str, Any]:
        with file_path.open("rb") as f:
            files: list[tuple[str, Any]] = [("file", (file_path.name, f, _content_type_for_path(file_path)))]
            files.extend(self._build_extract_fields())
            response = requests.post(
                f"{self._api_base_url}/extract",
                headers=self._headers(),
                files=files,
                timeout=self._timeout,
            )

        raw = self._handle_response(response, context="extract")
        return self._fetch_large_result(raw)

    def _apply_schema(self, extraction_id: str, schema: dict[str, Any]) -> dict[str, Any]:
        schema_config: dict[str, Any] = {
            "input_schema": _adapt_schema_for_pulse(schema),
        }
        if self._schema_prompt:
            schema_config["schema_prompt"] = self._schema_prompt
        if self._schema_effort:
            schema_config["effort"] = True

        response = requests.post(
            f"{self._api_base_url}/schema",
            headers={**self._headers(), "Content-Type": "application/json"},
            json={
                "extraction_id": extraction_id,
                "schema_config": schema_config,
            },
            timeout=self._timeout,
        )
        raw = self._handle_response(response, context="schema")
        return self._fetch_large_result(raw)

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"PulseExtractProvider only supports EXTRACT product type, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override"
            )

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        started_at = datetime.now()
        try:
            extract_raw = self._extract_file(file_path)
            extraction_id = extract_raw.get("extraction_id")
            if not isinstance(extraction_id, str) or not extraction_id:
                raise ProviderPermanentError(f"Pulse /extract response did not include extraction_id: {extract_raw}")

            schema_raw = self._apply_schema(extraction_id=extraction_id, schema=request.schema_override)
        except (
            ProviderPermanentError,
            ProviderTransientError,
            ProviderConfigError,
            ProviderRateLimitError,
        ):
            raise
        except requests.Timeout as e:
            raise ProviderTransientError(f"Pulse request timed out: {e}") from e
        except requests.ConnectionError as e:
            raise ProviderTransientError(f"Pulse connection error: {e}") from e
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during Pulse extraction: {e}") from e

        raw_output = {
            "extract": extract_raw,
            "schema": schema_raw,
            "_config": {
                "model": self._model,
                "pages": self._pages,
                "figure_processing": self._figure_processing,
                "extensions": self._extensions,
                "schema_prompt": self._schema_prompt,
                "effort": self._schema_effort,
                "estimate_schema_cost": self._estimate_schema_cost,
            },
        }
        _apply_usage_cost_fields(raw_output)

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=latency_ms,
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"PulseExtractProvider only supports EXTRACT product type, got {raw_result.product_type}"
            )

        schema_output = _as_mapping(_as_mapping(raw_result.raw_output.get("schema")).get("schema_output"))
        values = schema_output.get("values", {})
        extracted_data = values if isinstance(values, (dict, list)) else {}

        # Pulse reports per-field citations as element-anchor ids (e.g. "tbl-1-r25c3",
        # "txt-2"), not geometry. The coordinates live separately in the /extract
        # payload's bounding_boxes table. Resolve each anchor id to the box Pulse
        # already reported for it, then reuse the shared citation collector. Boxes
        # are passed through unchanged (no rescaling or reprojection).
        anchor_index = _build_pulse_anchor_index(
            _as_mapping(raw_result.raw_output.get("extract")).get("bounding_boxes")
        )
        resolved_citations = _resolve_pulse_citation_anchors(schema_output.get("citations"), anchor_index)
        citations = _extract_pulse_field_citations(resolved_citations)

        output = ExtractOutput(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data if isinstance(extracted_data, (dict, list)) else {},
            field_citations=citations,
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


def _content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        return f"image/{'jpeg' if suffix in {'.jpg', '.jpeg'} else suffix.lstrip('.')}"
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return "application/octet-stream"


def _adapt_schema_for_pulse(schema: Any) -> Any:
    """Normalize benchmark JSON Schema to the subset Pulse strict mode accepts.

    Pulse follows JSON Schema, but its strict-mode validator currently rejects
    slash/quote literals in places the model may need to reproduce. Keep the
    extraction shape intact while removing unsupported description characters,
    inlining local refs, collapsing nullable unions, and dropping unsupported
    schema metadata/defaults.
    """
    if not isinstance(schema, Mapping):
        if isinstance(schema, list):
            return [_adapt_schema_for_pulse(item) for item in schema]
        return schema

    schema = _promote_repeated_structure(dict(schema))

    def resolve_json_pointer(ref: str) -> Mapping[str, Any] | None:
        if not ref.startswith("#/"):
            return None
        current: Any = schema
        for raw_part in ref[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if not isinstance(current, Mapping):
                return None
            current = current.get(part)
        return current if isinstance(current, Mapping) else None

    def is_null_schema(node: Any) -> bool:
        return isinstance(node, Mapping) and node.get("type") == "null"

    def resolve_ref_node(node: Mapping[str, Any]) -> dict[str, Any]:
        ref = node.get("$ref")
        if isinstance(ref, str):
            resolved = resolve_json_pointer(ref)
            if resolved is not None:
                merged = dict(resolved)
                for key, value in node.items():
                    if key != "$ref":
                        merged[key] = value
                return merged
        return dict(node)

    def normalize_nullable_schema(node: Mapping[str, Any]) -> dict[str, Any]:
        node = dict(node)
        node_type = node.get("type")
        if isinstance(node_type, list):
            non_null_types = [value for value in node_type if value != "null"]
            if len(non_null_types) == 1:
                node["type"] = non_null_types[0]

        for union_key in ("anyOf", "oneOf"):
            options = node.get(union_key)
            if not isinstance(options, list):
                continue
            non_null_options = [option for option in options if not is_null_schema(option)]
            if len(non_null_options) != 1 or len(non_null_options) == len(options):
                continue

            result = dict(non_null_options[0])
            for key, value in node.items():
                if key != union_key and key not in result:
                    result[key] = value
            return result
        return node

    def adapt_node(node: Any) -> Any:
        if isinstance(node, Mapping):
            node = resolve_ref_node(node)
            node = normalize_nullable_schema(node)
            node = resolve_ref_node(node)

            out: dict[str, Any] = {}
            enum_values = node.get("enum")
            strip_enum = _has_pulse_unsafe_enum_literal(enum_values)
            for key, value in node.items():
                if key in _UNSUPPORTED_SCHEMA_KEYS:
                    continue
                if key == "$ref":
                    continue
                if key == "enum" and strip_enum:
                    continue
                if key == "description" and strip_enum and isinstance(value, str):
                    out[key] = _sanitize_pulse_description(
                        f"{value} Allowed values: {_format_enum_values(enum_values)}."
                    )
                elif key == "description" and isinstance(value, str):
                    out[key] = _sanitize_pulse_description(value)
                else:
                    out[key] = adapt_node(value)

            if strip_enum and "description" not in out:
                out["description"] = _sanitize_pulse_description(f"Allowed values: {_format_enum_values(enum_values)}.")
            return out
        if isinstance(node, list):
            return [adapt_node(item) for item in node]
        return node

    return adapt_node(schema)


def _promote_repeated_structure(schema: dict[str, Any]) -> dict[str, Any]:
    repeated_structure = schema.get("repeated_structure")
    if not isinstance(repeated_structure, Mapping):
        return schema

    out = dict(schema)
    properties = dict(out.get("properties") or {})
    for name, definition in repeated_structure.items():
        if isinstance(definition, Mapping) and name not in properties:
            properties[name] = dict(definition)
    out["properties"] = properties
    return out


def _has_pulse_unsafe_enum_literal(enum_values: Any) -> bool:
    if not isinstance(enum_values, Sequence) or isinstance(enum_values, (str, bytes, bytearray)):
        return False
    return any(isinstance(value, str) and any(char in value for char in ('"', "'", "/")) for value in enum_values)


def _format_enum_values(enum_values: Any) -> str:
    if not isinstance(enum_values, Sequence) or isinstance(enum_values, (str, bytes, bytearray)):
        return ""
    return ", ".join("null" if value is None else str(value) for value in enum_values)


def _sanitize_pulse_description(description: str) -> str:
    translation = str.maketrans(
        {
            '"': "",
            "'": "",
            "/": " or ",
            "\u2018": "",
            "\u2019": "",
            "\u201c": "",
            "\u201d": "",
        }
    )
    return " ".join(description.translate(translation).split())


def _apply_usage_cost_fields(raw_output: dict[str, Any]) -> None:
    extract = _as_mapping(raw_output.get("extract"))
    schema = _as_mapping(raw_output.get("schema"))

    page_count = _coerce_float(extract.get("page_count"))
    plan_info = _as_mapping(extract.get("plan_info") or extract.get("plan-info"))
    pages_used = _coerce_float(plan_info.get("pages_used"))
    if pages_used is None:
        pages_used = page_count

    extract_credits = _coerce_float(extract.get("credits_used"))
    schema_credits = _coerce_float(schema.get("credits_used"))
    config = _as_mapping(raw_output.get("_config"))
    estimate_schema_cost = bool(config.get("estimate_schema_cost", True))
    if schema_credits is None and pages_used is not None and estimate_schema_cost and config.get("effort"):
        schema_credits = pages_used * 4
    elif schema_credits is None and pages_used is not None and estimate_schema_cost:
        schema_credits = pages_used

    if pages_used is not None:
        raw_output["num_pages"] = pages_used
    if extract_credits is not None:
        raw_output["extract_credits_used"] = extract_credits
    if schema_credits is not None:
        raw_output["schema_credits_used_estimated"] = schema_credits

    total_credits = (extract_credits or 0.0) + (schema_credits or 0.0)
    if total_credits > 0:
        raw_output["credits_used"] = total_credits
        raw_output["cost_usd"] = total_credits * PulseExtractProvider.CREDIT_RATE_USD
        if pages_used and pages_used > 0:
            raw_output["cost_per_page_usd"] = raw_output["cost_usd"] / pages_used


def _extract_pulse_field_citations(citations: Any) -> list[FieldCitation]:
    return _dedupe(_collect_citations(citations, path=[]))


def _build_pulse_anchor_index(bounding_boxes: Any) -> dict[str, dict[str, Any]]:
    """Map Pulse element-anchor ids to the box Pulse reported for them.

    Pulse's citation payload references elements by id (``tbl-<n>-r<r>c<c>`` for
    table cells, ``txt-<n>`` for text blocks). The geometry for those ids lives in
    the /extract ``bounding_boxes`` payload, keyed the same way. Build a lookup so
    anchor citations can be grounded. Boxes are copied verbatim — the shared
    collector already understands the 8-value normalized-polygon shape, so no
    coordinate transform happens here.
    """
    boxes = _as_mapping(bounding_boxes)
    index: dict[str, dict[str, Any]] = {}
    for table in _as_sequence(boxes.get("Tables")):
        for cell in _as_sequence(_as_mapping(table).get("cell_data")):
            _register_pulse_anchor(index, cell)
    for block in _as_sequence(boxes.get("Text")):
        _register_pulse_anchor(index, block)
    return index


def _register_pulse_anchor(index: dict[str, dict[str, Any]], node: Any) -> None:
    node = _as_mapping(node)
    anchor = node.get("id")
    if not isinstance(anchor, str) or not anchor:
        return

    # Table cells expose location.coordinates; text blocks expose bounding_box.
    location = _as_mapping(node.get("location"))
    coordinates = location.get("coordinates") if location else node.get("bounding_box")
    if not isinstance(coordinates, Sequence) or isinstance(coordinates, (str, bytes, bytearray)):
        return

    resolved: dict[str, Any] = {"polygon": list(coordinates)}
    page = _coerce_int(location.get("page")) or _coerce_int(node.get("page")) or _coerce_int(node.get("page_number"))
    if page is not None:
        resolved["page"] = page
    for text_key in ("text", "content"):
        text = node.get(text_key)
        if isinstance(text, str) and text:
            resolved["text"] = text
            break
    confidence = _coerce_probability(node.get("confidence") or node.get("average_word_confidence"))
    if confidence is not None:
        resolved["confidence"] = confidence
    index[anchor] = resolved


def _resolve_pulse_citation_anchors(node: Any, index: dict[str, dict[str, Any]]) -> Any:
    """Replace each anchor-id leaf with its resolved box, preserving field paths.

    The citation tree mirrors the extracted-values tree, so keeping its dict/list
    shape lets the shared collector derive field paths like ``holdings[0].cusip``.
    Anchors that are empty or absent from the index resolve to ``None`` so no
    citation is emitted for that field.
    """
    if isinstance(node, str):
        resolved = index.get(node)
        return dict(resolved) if resolved is not None else None
    if isinstance(node, Mapping):
        if _looks_like_citation(node):
            return dict(node)
        return {
            key: _resolve_pulse_citation_anchors(value, index) for key, value in node.items() if isinstance(key, str)
        }
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        return [_resolve_pulse_citation_anchors(item, index) for item in node]
    return None


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _collect_citations(node: Any, *, path: list[str]) -> list[FieldCitation]:
    if isinstance(node, Mapping):
        field_path = _format_field_path(path)
        if field_path and _looks_like_citation(node):
            citation = _citation_from_node(field_path, node)
            return [citation] if citation is not None else []

        citations: list[FieldCitation] = []
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            citations.extend(_collect_citations(value, path=[*path, key]))
        return citations

    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        citations = []
        for index, item in enumerate(node):
            citations.extend(_collect_citations(item, path=[*path, f"[{index}]"]))
        return citations

    return []


def _looks_like_citation(node: Mapping[str, Any]) -> bool:
    return any(key in node for key in ("bbox", "bounding_box", "boundingBox", "polygon", "page", "page_number"))


def _citation_from_node(field_path: str, node: Mapping[str, Any]) -> FieldCitation | None:
    page = _coerce_int(node.get("page")) or _coerce_int(node.get("page_number")) or 1
    bbox, polygon = _extract_bbox_and_polygon(node)
    if bbox is None and not _has_page_only_citation(node):
        return None

    return FieldCitation(
        field_path=field_path,
        page=page,
        bbox=bbox,
        polygon=polygon,
        reference_text=_reference_text(node),
        confidence=_coerce_probability(node.get("confidence") or node.get("score")),
        source="pulse",
        metadata=_compact_metadata(node),
    )


def _extract_bbox_and_polygon(node: Mapping[str, Any]) -> tuple[list[float] | None, list[list[float]] | None]:
    raw = node.get("bbox", node.get("bounding_box", node.get("boundingBox", node.get("polygon"))))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None, None

    values = [_coerce_float(value) for value in raw]
    if any(value is None for value in values):
        return None, None
    coords = [float(value) for value in values if value is not None]

    if len(coords) == 8:
        points = [[coords[index], coords[index + 1]] for index in range(0, 8, 2)]
        if not _all_normalized(coords):
            return None, None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        left = min(xs)
        top = min(ys)
        bbox = [left, top, max(xs) - left, max(ys) - top]
        return _round_bbox(bbox), [[round(point[0], 8), round(point[1], 8)] for point in points]

    if len(coords) == 4:
        x1, y1, third, fourth = coords
        if not _all_normalized(coords):
            return None, None
        # Pulse schema examples use [x1, y1, x2, y2]. If the last two
        # coordinates cannot be a lower-right corner, fall back to xywh.
        if third > x1 and fourth > y1:
            return _round_bbox([x1, y1, third - x1, fourth - y1]), None
        return _round_bbox([x1, y1, third, fourth]), None

    return None, None


def _has_page_only_citation(node: Mapping[str, Any]) -> bool:
    return "page" in node or "page_number" in node


def _reference_text(node: Mapping[str, Any]) -> str | None:
    for key in ("text", "content", "value", "reference_text", "referenceText", "quote"):
        value = node.get(key)
        if isinstance(value, str):
            return value
    return None


def _compact_metadata(node: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = {
        key: value
        for key, value in node.items()
        if key not in {"bbox", "bounding_box", "boundingBox", "polygon", "page", "page_number"}
    }
    return dict(metadata) if metadata else None


def _format_field_path(path: list[str]) -> str:
    rendered = ""
    for token in path:
        if token.startswith("[") and token.endswith("]"):
            rendered += token
        elif rendered:
            rendered += "." + token
        else:
            rendered = token
    return rendered


def _dedupe(citations: list[FieldCitation]) -> list[FieldCitation]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[FieldCitation] = []
    for citation in citations:
        key = (
            citation.field_path,
            citation.page,
            tuple(citation.bbox) if citation.bbox is not None else None,
            citation.reference_text,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def _round_bbox(bbox: list[float]) -> list[float] | None:
    if not _valid_normalized_bbox(bbox):
        return None
    return [round(value, 8) for value in bbox]


def _valid_normalized_bbox(bbox: list[float]) -> bool:
    if len(bbox) != 4:
        return False
    x, y, width, height = bbox
    return (
        0 <= x <= 1
        and 0 <= y <= 1
        and 0 < width <= 1
        and 0 < height <= 1
        and x + width <= 1.000001
        and y + height <= 1.000001
    )


def _all_normalized(values: list[float]) -> bool:
    return all(0 <= value <= 1 for value in values)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_probability(value: Any) -> float | None:
    score = _coerce_float(value)
    if score is None or not 0.0 <= score <= 1.0:
        return None
    return score


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None
