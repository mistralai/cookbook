"""Provider for Azure Content Understanding GA EXTRACT."""

import base64
import hashlib
import json
import mimetypes
import os
import re
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

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

_DEFAULT_API_VERSION = "2025-11-01"
_TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}
_TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_SOURCE_REGION_RE = re.compile(r"D\(([^)]*)\)")
_DEFAULT_DOCUMENT_PAGE_MINIMAL_PRICE_PER_1000 = 0.01
_DEFAULT_DOCUMENT_PAGE_BASIC_PRICE_PER_1000 = 1.0
_DEFAULT_DOCUMENT_PAGE_STANDARD_PRICE_PER_1000 = 5.0
_DEFAULT_CONTEXTUALIZATION_PRICE_PER_1M = 1.0
_DEFAULT_COMPLETION_MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
}


@register_provider("azure_content_understanding")
class AzureContentUnderstandingProvider(Provider):
    """Run Azure Content Understanding analyzer-backed extraction."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        endpoint = self.base_config.get("endpoint") or os.getenv("AZURE_CONTENT_UNDERSTANDING_ENDPOINT")
        api_key = self.base_config.get("api_key") or os.getenv("AZURE_CONTENT_UNDERSTANDING_KEY")
        if not endpoint:
            raise ProviderConfigError(
                "Azure Content Understanding endpoint is required. "
                "Set AZURE_CONTENT_UNDERSTANDING_ENDPOINT or pass endpoint in config."
            )
        if not api_key:
            raise ProviderConfigError(
                "Azure Content Understanding API key is required. "
                "Set AZURE_CONTENT_UNDERSTANDING_KEY or pass api_key in config."
            )

        self._endpoint = _content_understanding_endpoint(str(endpoint)).rstrip("/")
        self._api_key = str(api_key)
        self._api_version = str(self.base_config.get("api_version", _DEFAULT_API_VERSION))
        self._timeout = float(self.base_config.get("timeout", 300))
        self._poll_interval = float(self.base_config.get("poll_interval", 2))
        self._max_polls = int(self.base_config.get("max_polls", 300))
        self._session = requests.Session()

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"AzureContentUnderstandingProvider only supports EXTRACT, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError("schema_override is required for EXTRACT. Provide a JSON schema.")

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        started_at = datetime.now()
        try:
            analyzer_id, analyzer_payload, field_name_map = self._ensure_analyzer(pipeline, request.schema_override)
            raw_output = self._analyze_binary(analyzer_id, file_path)
            raw_output["_analyzer_id"] = analyzer_id
            raw_output["_analyzer_payload"] = analyzer_payload
            raw_output["_field_name_map"] = field_name_map
            raw_output["_config"] = _public_config(pipeline.config)
            _apply_usage_cost_fields(raw_output, pipeline.config)

            completed_at = datetime.now()
            return RawInferenceResult(
                request=request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=request.product_type,
                raw_output=raw_output,
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
            )
        except (ProviderPermanentError, ProviderRateLimitError, ProviderTransientError):
            raise
        except Exception as exc:
            raise ProviderPermanentError(
                f"Unexpected error during Azure Content Understanding extraction: {exc}"
            ) from exc

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"AzureContentUnderstandingProvider only supports EXTRACT, got {raw_result.product_type}"
            )

        result = _analysis_result(raw_result.raw_output)
        content = _first_document_content(result)
        fields = content.get("fields") if isinstance(content, Mapping) else {}
        fields = _restore_field_names(fields, raw_result.raw_output.get("_field_name_map"))
        extracted_data = _field_value({"type": "object", "valueObject": fields})
        page_sizes = _page_sizes(content)
        document_text = _document_text(content)
        citations = _field_citations(fields, page_sizes, document_text=document_text)

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

    def _ensure_analyzer(
        self, pipeline: PipelineSpec, schema: dict[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        analyzer_payload, field_name_map = _build_analyzer_payload_and_field_map(schema, pipeline.config)
        config_hash = hashlib.sha256(
            json.dumps(analyzer_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        prefix = _analyzer_id_part(str(pipeline.config.get("analyzer_id_prefix", pipeline.pipeline_name)))[:32]
        analyzer_id = f"bench{prefix}{config_hash}"[:64]

        url = self._url(f"/contentunderstanding/analyzers/{analyzer_id}", {"allowReplace": "true"})
        response = self._request(
            "PUT",
            url,
            json=analyzer_payload,
            headers={"Content-Type": "application/json"},
            context="create analyzer",
        )
        operation_url = response.headers.get("Operation-Location")
        body = _response_json(response)
        if operation_url:
            self._poll_operation(operation_url, context=f"create analyzer {analyzer_id}")
        elif str(body.get("status", "")).lower() not in {"ready", "succeeded"}:
            self._poll_analyzer_ready(analyzer_id)

        return analyzer_id, analyzer_payload, field_name_map

    def _analyze_binary(self, analyzer_id: str, file_path: Path) -> dict[str, Any]:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        url = self._url(f"/contentunderstanding/analyzers/{analyzer_id}:analyze")
        payload = {
            "inputs": [
                {
                    "name": file_path.name,
                    "mimeType": mime_type,
                    "data": base64.b64encode(file_path.read_bytes()).decode("ascii"),
                }
            ]
        }
        response = self._request(
            "POST",
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            context="analyze",
        )
        operation_url = response.headers.get("Operation-Location")
        body = _response_json(response)
        if not operation_url:
            return body
        return self._poll_operation(operation_url, context=f"analyze {file_path.name}")

    def _poll_analyzer_ready(self, analyzer_id: str) -> None:
        url = self._url(f"/contentunderstanding/analyzers/{analyzer_id}")
        for _ in range(self._max_polls):
            response = self._request("GET", url, context=f"get analyzer {analyzer_id}")
            body = _response_json(response)
            status = str(body.get("status", "")).lower()
            if status == "ready":
                return
            if status == "failed":
                raise ProviderPermanentError(f"Azure Content Understanding analyzer {analyzer_id} failed: {body}")
            time.sleep(self._poll_interval)
        raise ProviderTransientError(f"Timed out waiting for Azure Content Understanding analyzer {analyzer_id}")

    def _poll_operation(self, operation_url: str, *, context: str) -> dict[str, Any]:
        for _ in range(self._max_polls):
            response = self._request("GET", operation_url, context=context)
            body = _response_json(response)
            status = str(body.get("status", "")).lower()
            if status in _TERMINAL_STATUSES:
                if status == "succeeded":
                    return body
                raise ProviderPermanentError(f"Azure Content Understanding {context} failed: {body}")
            time.sleep(self._poll_interval)
        raise ProviderTransientError(f"Timed out waiting for Azure Content Understanding {context}")

    def _url(self, path: str, extra_params: Mapping[str, str] | None = None) -> str:
        params = {"api-version": self._api_version}
        if extra_params:
            params.update(extra_params)
        return f"{self._endpoint}{path}?{urlencode(params)}"

    def _request(self, method: str, url: str, *, context: str, **kwargs: Any) -> requests.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Ocp-Apim-Subscription-Key"] = self._api_key
        try:
            response = self._session.request(method, url, headers=headers, timeout=self._timeout, **kwargs)
        except requests.RequestException as exc:
            raise ProviderTransientError(f"Azure Content Understanding {context} request failed: {exc}") from exc

        if response.status_code == 429:
            raise ProviderRateLimitError(f"Azure Content Understanding rate limited during {context}: {response.text}")
        if response.status_code in _TRANSIENT_STATUS_CODES:
            raise ProviderTransientError(
                f"Azure Content Understanding transient error during {context}: {response.status_code} {response.text}"
            )
        if response.status_code >= 400:
            raise ProviderPermanentError(
                f"Azure Content Understanding error during {context}: {response.status_code} {response.text}"
            )
        return response


def _public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    hidden = {"api_key", "endpoint", "timeout", "poll_interval", "max_polls"}
    return {key: value for key, value in config.items() if key not in hidden}


def _content_understanding_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.hostname is None:
        return endpoint
    return urlunparse(parsed._replace(path="", params="", query="", fragment=""))


def _apply_usage_cost_fields(raw_output: dict[str, Any], config: Mapping[str, Any]) -> None:
    usage = raw_output.get("usage")
    if not isinstance(usage, Mapping):
        return

    breakdown = _usage_cost_breakdown(usage, config)
    if not breakdown:
        return

    total_cost = float(breakdown["total_cost_usd"])
    raw_output["acu_cost_breakdown"] = breakdown
    raw_output["cost_usd"] = total_cost
    raw_output["input_cost_usd"] = breakdown["llm_input_cost_usd"]
    raw_output["output_and_thinking_cost_usd"] = breakdown["llm_output_cost_usd"]
    raw_output["acu_content_extraction_cost_usd"] = breakdown["content_extraction_cost_usd"]
    raw_output["acu_contextualization_cost_usd"] = breakdown["contextualization_cost_usd"]
    page_count = _usage_document_pages(usage)
    if page_count > 0:
        raw_output["num_pages"] = page_count
        raw_output["cost_per_page_usd"] = total_cost / page_count

    tokens = usage.get("tokens")
    if isinstance(tokens, Mapping):
        raw_output["input_tokens"] = _usage_token_count(tokens, "-input")
        raw_output["output_tokens"] = _usage_token_count(tokens, "-output")
        raw_output["total_tokens"] = raw_output["input_tokens"] + raw_output["output_tokens"]


def _usage_cost_breakdown(usage: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, float]:
    page_standard = _number(usage.get("documentPagesStandard"))
    page_basic = _number(usage.get("documentPagesBasic"))
    page_minimal = _number(usage.get("documentPagesMinimal"))
    page_standard_rate = _float_config(
        config,
        "document_page_standard_price_per_1000",
        _DEFAULT_DOCUMENT_PAGE_STANDARD_PRICE_PER_1000,
    )
    page_basic_rate = _float_config(
        config,
        "document_page_basic_price_per_1000",
        _DEFAULT_DOCUMENT_PAGE_BASIC_PRICE_PER_1000,
    )
    page_minimal_rate = _float_config(
        config,
        "document_page_minimal_price_per_1000",
        _DEFAULT_DOCUMENT_PAGE_MINIMAL_PRICE_PER_1000,
    )
    content_extraction_cost = (
        page_standard / 1000.0 * page_standard_rate
        + page_basic / 1000.0 * page_basic_rate
        + page_minimal / 1000.0 * page_minimal_rate
    )

    contextualization_tokens = _number(usage.get("contextualizationTokens", usage.get("contextualizationToken")))
    contextualization_rate = _float_config(
        config,
        "contextualization_price_per_1m",
        _DEFAULT_CONTEXTUALIZATION_PRICE_PER_1M,
    )
    contextualization_cost = contextualization_tokens / 1_000_000.0 * contextualization_rate

    input_tokens = 0.0
    output_tokens = 0.0
    input_cost = 0.0
    output_cost = 0.0
    tokens = usage.get("tokens")
    if isinstance(tokens, Mapping):
        for key, value in tokens.items():
            token_count = _number(value)
            if token_count <= 0:
                continue
            token_key = str(key)
            if token_key.endswith("-input"):
                model = token_key[: -len("-input")]
                input_rate, _ = _completion_model_rates(model, config)
                input_tokens += token_count
                input_cost += token_count / 1_000_000.0 * input_rate
            elif token_key.endswith("-output"):
                model = token_key[: -len("-output")]
                _, output_rate = _completion_model_rates(model, config)
                output_tokens += token_count
                output_cost += token_count / 1_000_000.0 * output_rate

    total_cost = content_extraction_cost + contextualization_cost + input_cost + output_cost
    return {
        "total_cost_usd": total_cost,
        "content_extraction_cost_usd": content_extraction_cost,
        "contextualization_cost_usd": contextualization_cost,
        "llm_input_cost_usd": input_cost,
        "llm_output_cost_usd": output_cost,
        "document_pages_standard": page_standard,
        "document_pages_basic": page_basic,
        "document_pages_minimal": page_minimal,
        "contextualization_tokens": contextualization_tokens,
        "llm_input_tokens": input_tokens,
        "llm_output_tokens": output_tokens,
    }


def _usage_document_pages(usage: Mapping[str, Any]) -> float:
    return (
        _number(usage.get("documentPagesStandard"))
        + _number(usage.get("documentPagesBasic"))
        + _number(usage.get("documentPagesMinimal"))
    )


def _usage_token_count(tokens: Mapping[str, Any], suffix: str) -> int:
    return int(sum(_number(value) for key, value in tokens.items() if str(key).endswith(suffix)))


def _completion_model_rates(model: str, config: Mapping[str, Any]) -> tuple[float, float]:
    pricing = config.get("completion_model_pricing_per_1m")
    if isinstance(pricing, Mapping):
        value = pricing.get(model)
        if isinstance(value, Mapping):
            return _number(value.get("input")), _number(value.get("output"))
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return _number(value[0]), _number(value[1])

    for prefix, rates in sorted(_DEFAULT_COMPLETION_MODEL_PRICING_PER_1M.items(), key=lambda item: -len(item[0])):
        if model.startswith(prefix):
            return rates
    return (0.0, 0.0)


def _float_config(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key)
    return _number(value) if value is not None else default


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _build_analyzer_payload(schema: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    payload, _ = _build_analyzer_payload_and_field_map(schema, config)
    return payload


def _build_analyzer_payload_and_field_map(
    schema: dict[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    fields, field_name_map = _schema_fields(schema, str(config.get("field_method", "extract")))
    payload = {
        "baseAnalyzerId": config.get("base_analyzer_id", "prebuilt-document"),
        "config": _content_config(config),
        "fieldSchema": {
            "name": str(config.get("field_schema_name", "BenchExtractSchema")),
            "description": str(config.get("field_schema_description", "Benchmark extraction schema")),
            "fields": fields,
            "definitions": {},
        },
    }
    models = config.get("models")
    if isinstance(models, Mapping):
        payload["models"] = {str(key): str(value) for key, value in models.items()}
    return payload, field_name_map


def _analyzer_id_part(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", value) or "acu"


def _content_config(config: Mapping[str, Any]) -> dict[str, Any]:
    annotation_format = "markdown" if bool(config.get("enable_annotations", True)) else "none"
    return {
        "enableOcr": bool(config.get("enable_ocr", True)),
        "enableLayout": bool(config.get("enable_layout", True)),
        "enableFormula": bool(config.get("enable_formula", True)),
        "enableBarcode": bool(config.get("enable_barcode", True)),
        "returnDetails": bool(config.get("return_details", True)),
        "omitContent": bool(config.get("omit_content", False)),
        "tableFormat": str(config.get("table_format", "html")),
        "enableFigureDescription": bool(config.get("enable_figure_description", False)),
        "enableFigureAnalysis": bool(config.get("enable_figure_analysis", False)),
        "annotationFormat": annotation_format,
        "enableSegment": bool(config.get("enable_segment", False)),
        "segmentPerPage": bool(config.get("segment_per_page", False)),
        "estimateFieldSourceAndConfidence": bool(config.get("estimate_field_source_and_confidence", True)),
    }


def _schema_fields(schema: Mapping[str, Any], field_method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_properties = schema.get("properties")
    properties = raw_properties if isinstance(raw_properties, Mapping) else {}
    fields: dict[str, Any] = {}
    field_name_map: dict[str, Any] = {}
    used_names: set[str] = set()
    for name, prop in properties.items():
        original_name = str(name)
        safe_name = _safe_field_name(original_name, used_names)
        node, children = _schema_node(prop, field_method)
        fields[safe_name] = node
        field_name_map[safe_name] = {"name": original_name, "children": children}
    return fields, field_name_map


def _schema_node(node: Any, field_method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(node, Mapping):
        return {"type": "string", "method": field_method}, {}

    node_type = _normalize_schema_type(node.get("type"))
    out: dict[str, Any] = {"type": node_type}
    field_name_map: dict[str, Any] = {}
    if isinstance(node.get("description"), str):
        out["description"] = node["description"]
    if isinstance(node.get("enum"), list):
        out["enum"] = [str(value) for value in node["enum"]]

    if node_type == "object":
        raw_properties = node.get("properties")
        properties = raw_properties if isinstance(raw_properties, Mapping) else {}
        out["properties"], field_name_map = _schema_fields({"properties": properties}, field_method)
    elif node_type == "array":
        out["items"], field_name_map = _schema_node(node.get("items", {"type": "string"}), field_method)
    else:
        out["method"] = field_method
        out["estimateSourceAndConfidence"] = True
    return out, field_name_map


def _safe_field_name(value: str, used_names: set[str]) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value)
    candidate = "".join(part[:1].upper() + part[1:] for part in parts) or "Field"
    if not candidate[0].isalpha():
        candidate = f"Field{candidate}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    suffix = hashlib.sha1(value.encode()).hexdigest()[:8]
    deduped = f"{candidate}{suffix}"
    counter = 2
    while deduped in used_names:
        deduped = f"{candidate}{suffix}{counter}"
        counter += 1
    used_names.add(deduped)
    return deduped


def _normalize_schema_type(value: Any) -> str:
    if isinstance(value, list):
        non_null = [item for item in value if item != "null"]
        value = non_null[0] if non_null else "string"
    if value in {"string", "number", "integer", "boolean", "array", "object"}:
        return str(value)
    return "string"


def _response_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {"value": payload}


def _analysis_result(raw_output: Mapping[str, Any]) -> dict[str, Any]:
    result = raw_output.get("result", raw_output)
    if isinstance(result, Mapping):
        return dict(result)
    return {}


def _first_document_content(result: Mapping[str, Any]) -> Mapping[str, Any]:
    contents = result.get("contents")
    if isinstance(contents, list):
        for content in contents:
            if isinstance(content, Mapping) and content.get("kind") == "document":
                return content
        for content in contents:
            if isinstance(content, Mapping):
                return content
    return {}


def _field_value(field: Any) -> Any:
    if not isinstance(field, Mapping):
        return field
    field_type = field.get("type")
    if field_type == "object":
        value = field.get("valueObject")
        if isinstance(value, Mapping):
            return {str(key): _field_value(child) for key, child in value.items()}
        return {}
    if field_type == "array":
        value = field.get("valueArray")
        if isinstance(value, list):
            return [_field_value(item) for item in value]
        return []
    for key in ("valueString", "valueNumber", "valueInteger", "valueBoolean", "valueDate", "valueTime", "valueJson"):
        if key in field:
            return field[key]
    return None


def _restore_field_names(fields: Any, field_name_map: Any) -> Any:
    if not isinstance(fields, Mapping) or not isinstance(field_name_map, Mapping):
        return fields
    restored: dict[str, Any] = {}
    for safe_name, field in fields.items():
        mapping = field_name_map.get(str(safe_name))
        original_name = mapping.get("name") if isinstance(mapping, Mapping) else safe_name
        children = mapping.get("children") if isinstance(mapping, Mapping) else None
        restored[str(original_name)] = _restore_child_field_names(field, children)
    return restored


def _restore_child_field_names(field: Any, field_name_map: Any) -> Any:
    if not isinstance(field, Mapping):
        return field
    restored = dict(field)
    if restored.get("type") == "object" and isinstance(restored.get("valueObject"), Mapping):
        restored["valueObject"] = _restore_field_names(restored["valueObject"], field_name_map)
    elif restored.get("type") == "array" and isinstance(restored.get("valueArray"), list):
        restored["valueArray"] = [_restore_child_field_names(item, field_name_map) for item in restored["valueArray"]]
    return restored


def _page_sizes(content: Mapping[str, Any]) -> dict[int, tuple[float, float]]:
    sizes: dict[int, tuple[float, float]] = {}
    pages = content.get("pages")
    if not isinstance(pages, list):
        return sizes
    for page in pages:
        if not isinstance(page, Mapping):
            continue
        page_number = page.get("pageNumber")
        width = page.get("width")
        height = page.get("height")
        if isinstance(page_number, int) and isinstance(width, (int, float)) and isinstance(height, (int, float)):
            sizes[page_number] = (float(width), float(height))
    return sizes


def _document_text(content: Mapping[str, Any]) -> str | None:
    for key in ("markdown", "text", "content"):
        value = content.get(key)
        if isinstance(value, str):
            return value
    return None


def _field_citations(
    fields: Any,
    page_sizes: Mapping[int, tuple[float, float]],
    path: str = "",
    *,
    document_text: str | None = None,
) -> list[FieldCitation]:
    citations: list[FieldCitation] = []
    if not isinstance(fields, Mapping):
        return citations
    for name, field in fields.items():
        field_path = f"{path}.{name}" if path else str(name)
        citations.extend(_field_citations_for_field(field_path, field, page_sizes, document_text=document_text))
    return citations


def _field_citations_for_field(
    field_path: str,
    field: Any,
    page_sizes: Mapping[int, tuple[float, float]],
    *,
    document_text: str | None = None,
) -> list[FieldCitation]:
    if not isinstance(field, Mapping):
        return []
    field_type = field.get("type")
    if field_type == "object" and isinstance(field.get("valueObject"), Mapping):
        return _field_citations(field["valueObject"], page_sizes, field_path, document_text=document_text)
    if field_type == "array" and isinstance(field.get("valueArray"), list):
        citations: list[FieldCitation] = []
        for index, item in enumerate(field["valueArray"]):
            citations.extend(
                _field_citations_for_field(f"{field_path}[{index}]", item, page_sizes, document_text=document_text)
            )
        return citations
    citation = _citation_from_field(field_path, field, page_sizes, document_text=document_text)
    return [citation] if citation is not None else []


def _citation_from_field(
    field_path: str,
    field: Mapping[str, Any],
    page_sizes: Mapping[int, tuple[float, float]],
    *,
    document_text: str | None = None,
) -> FieldCitation | None:
    source = field.get("source")
    if not isinstance(source, str) or not source:
        return None
    selected_region_indexes, reference_text = _selected_source_region_indexes(field, document_text)
    page, bbox = _parse_source_bbox(source, page_sizes, selected_region_indexes=selected_region_indexes)
    return FieldCitation(
        field_path=field_path,
        page=page or 1,
        bbox=bbox,
        reference_text=reference_text or _reference_text(field),
        confidence=float(field["confidence"]) if isinstance(field.get("confidence"), (int, float)) else None,
        source="azure_content_understanding",
        metadata={"source": source, "spans": field.get("spans")},
    )


def _reference_text(field: Mapping[str, Any]) -> str | None:
    value = _field_value(field)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _parse_source_bbox(
    source: str,
    page_sizes: Mapping[int, tuple[float, float]],
    *,
    selected_region_indexes: set[int] | None = None,
) -> tuple[int | None, list[float] | None]:
    regions: list[tuple[int, float, float, float, float]] = []
    for index, match in enumerate(_SOURCE_REGION_RE.finditer(source)):
        if selected_region_indexes is not None and index not in selected_region_indexes:
            continue
        region = _source_region_bbox(match.group(1))
        if region is not None:
            regions.append(region)
    if not regions:
        region = _source_region_bbox(source)
        if region is not None:
            regions.append(region)
    if not regions:
        return None, None

    page = regions[0][0]
    same_page_regions = [region for region in regions if region[0] == page]
    x0 = min(region[1] for region in same_page_regions)
    y0 = min(region[2] for region in same_page_regions)
    x1 = max(region[3] for region in same_page_regions)
    y1 = max(region[4] for region in same_page_regions)
    width, height = page_sizes.get(page, (1.0, 1.0))
    if width > 0 and height > 0 and (x1 > 1.0 or y1 > 1.0):
        x0, x1 = x0 / width, x1 / width
        y0, y1 = y0 / height, y1 / height
    x0 = min(1.0, max(0.0, x0))
    y0 = min(1.0, max(0.0, y0))
    x1 = min(1.0, max(0.0, x1))
    y1 = min(1.0, max(0.0, y1))
    return page, [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def _selected_source_region_indexes(
    field: Mapping[str, Any], document_text: str | None
) -> tuple[set[int] | None, str | None]:
    span_texts = _field_span_texts(field, document_text)
    if not span_texts:
        return None, None

    value = _field_value(field)
    value_text = _reference_text(field)
    selected: set[int] = set()
    selected_texts: list[str] = []
    for index, text in span_texts:
        if _span_matches_value(text, value, value_text):
            selected.add(index)
            selected_texts.append(text.strip())

    if not selected:
        return None, None
    reference_text = " ".join(text for text in selected_texts if text)
    return selected, reference_text or None


def _field_span_texts(field: Mapping[str, Any], document_text: str | None) -> list[tuple[int, str]]:
    if not document_text:
        return []
    spans = field.get("spans")
    if not isinstance(spans, list):
        return []
    out: list[tuple[int, str]] = []
    for index, span in enumerate(spans):
        if not isinstance(span, Mapping):
            continue
        offset = span.get("offset")
        length = span.get("length")
        if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length <= 0:
            continue
        text = document_text[offset : offset + length]
        if text:
            out.append((index, text))
    return out


def _span_matches_value(span_text: str, value: Any, value_text: str | None) -> bool:
    span_norm = _match_text(span_text)
    if not span_norm:
        return False
    if isinstance(value, bool):
        accepted = {"true", "yes", "y", "checked"} if value else {"false", "no", "n", "unchecked"}
        return span_norm in accepted
    if value_text is None:
        return False
    value_norm = _match_text(value_text)
    if not value_norm:
        return False
    return span_norm == value_norm or span_norm in value_norm or value_norm in span_norm


def _match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _source_region_bbox(source_region: str) -> tuple[int, float, float, float, float] | None:
    numbers = [float(match.group(0)) for match in _NUMBER_RE.finditer(source_region)]
    if len(numbers) < 5:
        return None
    page = int(numbers[0])
    coords = numbers[1:]
    if len(coords) >= 8:
        xs = coords[0::2]
        ys = coords[1::2]
        return page, min(xs), min(ys), max(xs), max(ys)
    x0, y0, x1, y1 = coords[:4]
    return page, x0, y0, x1, y1
