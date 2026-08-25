"""Provider for NuExtract3 structured extraction on the self-hosted vLLM server.

NuExtract3 (numind/NuExtract3, a 4B Qwen3.5-based VLM) does structured
extraction natively: given document images + a NuExtract *template* it emits a
JSON object shaped like that template. It reuses the same deployed vLLM endpoint
as the parse pipeline — extraction is just a different ``chat_template_kwargs``
(a ``template`` instead of ``mode="markdown"``).

The bench supplies a JSON Schema (``request.schema_override``). NuExtract expects
its own template format whose leaves are type names (``"string"``, ``"integer"``,
``"number"``, ``"boolean"``, ``"date"``…) and whose enums are lists of options,
so we convert the JSON Schema to a NuExtract template locally.

Like lift, NuExtract3 emits schema-shaped JSON with no per-field citations /
bboxes, so ``field_citations`` is always empty (evidence-bbox metrics are N/A);
the bench scores the extracted values.
"""

import asyncio
import base64
import io
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

DEFAULT_SERVED_MODEL_NAME = "nuextract3"

_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

_FORMAT_TO_NUEXTRACT_TYPE = {
    "string": "string",
    "verbatim-string": "verbatim-string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "date": "date",
    "time": "time",
    "date-time": "date-time",
    "duration": "duration",
    "country-code-ISO_3166-1_2chars": "country",
    "country-code": "country",
    "currency-code-ISO_4217_3chars": "currency",
    "currency-code": "currency",
    "language-code-ISO_639-3_3chars": "language",
    "language-code": "language",
    "language-tag-IETF-BCP-47": "language-tag",
    "language-tag": "language-tag",
    "script-code-ISO_15924-4chars": "script",
    "script-code": "script",
    "iri": "url",
    "uri": "url",
    "url": "url",
    "idn-email": "email-address",
    "email": "email-address",
    "email-address": "email-address",
    "phone-number-E.164": "phone-number",
    "phone-number": "phone-number",
    "iban-ISO_13616-1": "iban",
    "bice-code-ISO_9362": "bic",
    "ucum-unit-code": "unit-code",
}
_REGION_COUNTRIES = {
    "US",
    "FR",
    "IE",
    "GB",
    "IT",
    "ES",
    "DE",
    "PT",
    "CA",
    "MX",
    "BR",
    "AU",
    "JP",
    "KR",
    "CN",
    "IN",
    "VN",
    "TH",
    "RU",
    "PL",
}


def _repair_truncated_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort recovery of a truncated / trailing-garbage JSON object.

    Large extractions (e.g. a 13F with hundreds of holdings) can exceed the
    output-token budget and get cut mid-array, and models occasionally append
    junk after a valid object. We recover the top-level fields plus every
    complete array element before the cut by truncating at successive value/
    container boundaries (latest first), closing any still-open brackets, and
    returning the first prefix that parses to an object. Returns ``None`` if
    nothing parseable can be salvaged.
    """
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]

    # Collect candidate cut points: the index just after every completed string
    # or closed container — i.e. positions we can truncate at and then re-close.
    boundaries: list[int] = []
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                boundaries.append(i + 1)
            continue
        if ch == '"':
            in_str = True
        elif ch in "}]":
            boundaries.append(i + 1)

    # Try the latest boundaries first; the valid cut is usually within a handful
    # of the truncation point. Cap attempts so a pathological output stays cheap.
    for cut in list(reversed(boundaries))[:300]:
        prefix = s[:cut]
        stack: list[str] = []
        p_in_str = False
        p_esc = False
        for ch in prefix:
            if p_in_str:
                if p_esc:
                    p_esc = False
                elif ch == "\\":
                    p_esc = True
                elif ch == '"':
                    p_in_str = False
                continue
            if ch == '"':
                p_in_str = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]" and stack:
                stack.pop()
        candidate = re.sub(r"[,\s]+$", "", prefix) + "".join(reversed(stack))
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _json_schema_to_nuextract_template_and_instructions(schema: Any) -> tuple[Any, str]:
    """Convert JSON Schema to a NuExtract template and description instructions.

    This ports Numind's schema sanitization and template conversion, excluding its
    JSON Schema validation, instance adaptation, and conversion-status reporting.
    """
    if not isinstance(schema, dict):
        raise ProviderPermanentError("JSON Schema must be an object")

    omitted = object()
    composition_descriptions_key = "x-nuextract-composition-descriptions"
    annotation_keys = {
        "title",
        "description",
        "default",
        "examples",
        "format",
        "x-verbatim",
        composition_descriptions_key,
    }
    structural_keys = {
        "type",
        "properties",
        "patternProperties",
        "additionalProperties",
        "required",
        "items",
        "prefixItems",
        "enum",
        "allOf",
        "anyOf",
        "oneOf",
    }

    def annotations(node: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in node.items() if key in annotation_keys}

    def resolve_ref(ref: Any) -> dict[str, Any]:
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise ValueError(f"Only local JSON Schema references are supported: {ref!r}")
        current: Any = schema
        for raw_part in ref[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                raise ValueError(f"Could not resolve JSON Schema reference: {ref}")
        if not isinstance(current, dict):
            raise ValueError(f"JSON Schema reference does not point to an object: {ref}")
        return current

    def decode_leaf(node: dict[str, Any]) -> Any:
        if "enum" in node:
            values = node["enum"]
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError(f"Unsupported enum node: {node}")
            return values

        node_type = node.get("type")
        if not isinstance(node_type, str):
            raise TypeError(f"Invalid schema leaf: missing string 'type'. Node: {node}")
        decoded_type = node_type
        if node_type == "string":
            for annotation in (node.get("format"), node.get("description")):
                if isinstance(annotation, str) and annotation in _FORMAT_TO_NUEXTRACT_TYPE:
                    decoded_type = _FORMAT_TO_NUEXTRACT_TYPE[annotation]
                    break
                if isinstance(annotation, str):
                    region = annotation.removeprefix("region-code-ISO_3166-2:")
                    if region in _REGION_COUNTRIES:
                        decoded_type = f"region:{region}"
                        break
        if node.get("x-verbatim") and decoded_type != "string":
            raise ValueError(f"x-verbatim is only supported for string leaves. Node: {node}")
        return f"verbatim-{decoded_type}" if node.get("x-verbatim") else decoded_type

    def process(node: dict[str, Any]) -> Any:
        if "enum" in node:
            return decode_leaf(node)
        if "anyOf" in node:
            branch = next(value for value in node["anyOf"] if value.get("type") != "null")
            return process(branch)
        node_type = node["type"]
        if node_type == "array":
            item = process(node["items"])
            return omitted if item is omitted else [item]
        if node_type != "object":
            return decode_leaf(node)
        converted = {}
        for key, value in node.get("properties", {}).items():
            processed_value = process(value)
            if processed_value is not omitted:
                converted[key] = processed_value
        return converted or omitted

    def merge_objects(nodes: list[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {"type": "object"}
        properties: dict[str, Any] = {}
        required: list[str] = []
        for node in nodes:
            for key in ("title", "default", "examples"):
                if key in node and key not in merged:
                    merged[key] = node[key]
            for key, value in node.get("properties", {}).items():
                properties[key] = merge_nodes([properties[key], value]) if key in properties else value
            for key in node.get("required", []):
                if isinstance(key, str) and key not in required:
                    required.append(key)
        if properties:
            merged["properties"] = properties
        kept_required = [key for key in required if key in properties]
        if kept_required:
            merged["required"] = kept_required
        return merged

    def merge_nodes(nodes: list[dict[str, Any]]) -> dict[str, Any]:
        if not nodes:
            raise ValueError("Cannot merge an empty list of schema nodes")
        if all(node.get("type") == "object" for node in nodes):
            merged = merge_objects(nodes)
        elif all(node.get("type") == "array" and isinstance(node.get("items"), dict) for node in nodes):
            merged = {"type": "array", "items": merge_nodes([node["items"] for node in nodes])}
        else:
            templates = [process(node) for node in nodes]
            if any(template != templates[0] for template in templates):
                raise ValueError(f"Composition contains incompatible schemas: {nodes}")
            merged = deepcopy(nodes[0])

        descriptions = []
        for node in nodes:
            candidates = [node.get("description"), *node.get(composition_descriptions_key, [])]
            for description in candidates:
                if isinstance(description, str) and description not in descriptions:
                    descriptions.append(description)
        merged.pop("description", None)
        merged.pop(composition_descriptions_key, None)
        if descriptions:
            merged[composition_descriptions_key] = descriptions
        return merged

    def sanitize(raw_node: Any, ref_stack: frozenset[str] = frozenset()) -> dict[str, Any]:
        if not isinstance(raw_node, dict):
            raise TypeError(f"Invalid schema segment: expected object node. Node: {raw_node}")
        node = dict(raw_node)
        node_type = node.get("type")
        if isinstance(node_type, list):
            if not all(isinstance(value, str) for value in node_type):
                raise ValueError(f"Invalid list 'type'. Node: {node}")
            non_null_types = [value for value in node_type if value != "null"]
            if len(node_type) == 1:
                node["type"] = node_type[0]
            elif len(non_null_types) == 1 and len(non_null_types) != len(node_type):
                node["type"] = non_null_types[0]
            elif not non_null_types:
                raise ValueError(f"Unsupported null-only schema node: {node}")
            else:
                raise ValueError(f"Unsupported type union: {node}")

        if "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str):
                raise ValueError(f"Invalid $ref value: {ref}")
            if ref in ref_stack:
                raise ValueError(f"Cyclic $ref detected: {ref}")
            referenced = sanitize(resolve_ref(ref), ref_stack | {ref})
            siblings = {key: value for key, value in node.items() if key in structural_keys}
            nodes = [referenced]
            if siblings:
                siblings.setdefault("type", referenced.get("type"))
                nodes.append(sanitize(siblings, ref_stack | {ref}))
            return {**merge_nodes(nodes), **annotations(node)}

        if "allOf" in node:
            branches = node["allOf"]
            if not isinstance(branches, list):
                raise ValueError(f"Invalid allOf node: {node}")
            sibling_structure = {
                key: value for key, value in node.items() if key in structural_keys - {"allOf", "anyOf", "oneOf"}
            }
            sanitized = [sanitize(branch, ref_stack) for branch in [*branches, sibling_structure] if branch]
            if not sanitized:
                raise ValueError(f"Empty allOf node: {node}")
            merged = merge_nodes(sanitized)
            node_annotations = annotations(node)
            if merged.get("type") == "object":
                node_annotations.pop("format", None)
                node_annotations.pop("x-verbatim", None)
            return {**merged, **node_annotations}

        if "enum" in node:
            values = node["enum"]
            if not isinstance(values, list) or not all(value is None or isinstance(value, str) for value in values):
                raise ValueError(f"Unsupported enum node: {node}")
            non_null_values = [value for value in values if value is not None]
            if len(non_null_values) == 1 and len(values) > 1:
                return {**annotations(node), "type": "string"}
            if len(non_null_values) < 2:
                raise ValueError(f"Unsupported enum node: {node}")
            return {**annotations(node), "enum": non_null_values}

        if "oneOf" in node:
            branches = node["oneOf"]
            if not isinstance(branches, list):
                raise ValueError(f"Invalid oneOf node: {node}")
            sibling_structure = {
                key: value for key, value in node.items() if key in structural_keys - {"allOf", "anyOf", "oneOf"}
            }
            normalized = annotations(node)
            normalized["anyOf"] = [
                {**sibling_structure, **branch}
                if sibling_structure.get("type") == "array"
                else {"allOf": [sibling_structure, branch]}
                if sibling_structure
                else branch
                for branch in branches
            ]
            return sanitize(normalized, ref_stack)

        if "anyOf" in node:
            branches = node["anyOf"]
            if not isinstance(branches, list):
                raise ValueError(f"Invalid anyOf node: {node}")
            sibling_structure = {
                key: value for key, value in node.items() if key in structural_keys - {"allOf", "anyOf", "oneOf"}
            }
            non_null = [
                {**sibling_structure, **branch}
                if sibling_structure.get("type") == "array" and isinstance(branch, dict)
                else {"allOf": [sibling_structure, branch]}
                if sibling_structure
                else branch
                for branch in branches
                if not (isinstance(branch, dict) and branch.get("type") == "null")
            ]
            if len(non_null) == 1 and len(non_null) != len(branches):
                return {**annotations(node), "anyOf": [sanitize(non_null[0], ref_stack), {"type": "null"}]}
            if not non_null:
                raise ValueError(f"Unsupported null-only union node: {node}")
            raise ValueError("Ambiguous unions cannot be represented by one NuExtract template")

        node_type = node.get("type")
        if node_type == "null":
            raise ValueError(f"Unsupported null-only schema node: {node}")
        if node_type == "array":
            prefix_items = node.get("prefixItems")
            if prefix_items is not None:
                if not isinstance(prefix_items, list):
                    raise ValueError(f"Invalid prefixItems: {node}")
                max_items = node.get("maxItems")
                reachable = min(len(prefix_items), max_items) if isinstance(max_items, int) else len(prefix_items)
                items = [sanitize(value, ref_stack) for value in prefix_items[:reachable]]
                trailing_reachable = not isinstance(max_items, int) or max_items > len(prefix_items)
                remaining = node.get("items", True)
                if isinstance(remaining, dict) and trailing_reachable:
                    items.append(sanitize(remaining, ref_stack))
                elif remaining is not False and trailing_reachable:
                    raise ValueError(f"Unsupported tuple with unconstrained trailing items: {node}")
                if not items:
                    raise ValueError(f"Unsupported empty tuple node: {node}")
                return {**annotations(node), "type": "array", "items": merge_nodes(items)}
            if "items" not in node:
                raise ValueError(f"Unsupported array node without items: {node}")
            return {**annotations(node), "type": "array", "items": sanitize(node["items"], ref_stack)}

        if node_type is not None and node_type != "object":
            decode_leaf(node)
            return {**annotations(node), "type": node_type}

        if node_type == "object" or any(
            key in node for key in ("properties", "patternProperties", "additionalProperties")
        ):
            properties = node.get("properties", {})
            if not isinstance(properties, dict):
                raise ValueError(f"Object properties must be an object: {node}")
            if (isinstance(node.get("patternProperties"), dict) and node["patternProperties"]) or isinstance(
                node.get("additionalProperties"), dict
            ):
                raise ValueError("Dynamic object keys cannot be represented by a NuExtract template")
            sanitized_properties = {key: sanitize(value, ref_stack) for key, value in properties.items()}
            result = {**annotations(node), "type": "object"}
            if sanitized_properties:
                result["properties"] = sanitized_properties
            required = node.get("required")
            if isinstance(required, list):
                kept = [key for key in required if isinstance(key, str) and key in sanitized_properties]
                if kept:
                    result["required"] = kept
            return result
        raise ValueError(f"Schema node has no supported structural keyword: {node}")

    def description_lines(node: Any, path: str = "$") -> list[str]:
        if not isinstance(node, dict):
            return []
        result = []
        seen_descriptions: set[str] = set()
        candidates = [node.get("description"), *node.get(composition_descriptions_key, [])]
        for description in candidates:
            if isinstance(description, str) and description not in seen_descriptions:
                result.append(f"{path}: {description}")
                seen_descriptions.add(description)
        if "anyOf" in node:
            branch = next(value for value in node["anyOf"] if value.get("type") != "null")
            result.extend(description_lines(branch, path))
        elif node.get("type") == "object":
            for key, value in node.get("properties", {}).items():
                result.extend(description_lines(value, f"{path}.{key}"))
        elif node.get("type") == "array":
            result.extend(description_lines(node["items"], f"{path}[]"))
        return result

    try:
        compatible_schema = sanitize(schema)
        template = process(compatible_schema)
        if template is omitted:
            raise ValueError("Root schema contains no supported template fields")
        descriptions = description_lines(compatible_schema)
    except (KeyError, TypeError, ValueError) as e:
        raise ProviderPermanentError(f"Could not convert JSON Schema to a NuExtract template: {e}") from e
    return template, "\n".join(descriptions)


@register_provider("nuextract3_extract")
class NuExtract3ExtractProvider(Provider):
    """
    Provider for NuExtract3 structured extraction (direct vLLM, self-hosted).

    Configuration options:
        - server_url (str, required): self-hosted vLLM server URL (the same endpoint
          the nuextract3 parse pipeline uses).
        - model (str, default="nuextract3"): served model name.
        - timeout (int, default=1800): per-request timeout in seconds (large
          extractions generate a lot of tokens and can run for many minutes).
        - dpi (int, default=150): DPI for PDF-to-image rendering.
        - max_pages (int, default=100): cap on rendered pages per document (keeps
          the request within the server's image / context limits).
        - max_tokens (int, default=100000): max output tokens — large enough for
          a long holdings array (server context is 262144).
        - temperature (float, default=0.2): sampling temperature (non-thinking).
        - enable_thinking (bool, default=False): NuExtract reasoning mode.
        - api_key_env (str, default="VLLM_API_KEY"): env var for the API key.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        # No default endpoint: the deployment is yours. `endpoint_env_var` lets
        # the pipeline name the env var that carries its URL.
        server_url = self.base_config.get("server_url")
        endpoint_env_var = self.base_config.get("endpoint_env_var")
        if not server_url and endpoint_env_var:
            import os as _os

            server_url = _os.environ.get(str(endpoint_env_var), "")
        if not server_url:
            raise ProviderConfigError(
                "nuextract3_extract provider requires 'server_url' in config"
                + (f" or the {endpoint_env_var} environment variable." if endpoint_env_var else ".")
            )
        self._server_url: str = str(server_url)

        self._model = self.base_config.get("model", DEFAULT_SERVED_MODEL_NAME)
        self._timeout = int(self.base_config.get("timeout", 1800))
        self._dpi = int(self.base_config.get("dpi", 150))
        self._max_pages = int(self.base_config.get("max_pages", 100))
        self._max_tokens = int(self.base_config.get("max_tokens", 100000))
        self._temperature = float(self.base_config.get("temperature", 0.2))
        self._enable_thinking = bool(self.base_config.get("enable_thinking", False))

        import os

        api_key_env = self.base_config.get("api_key_env", "VLLM_API_KEY")
        self._api_key = os.environ.get(api_key_env, "")

    # ------------------------------------------------------------------
    # Image rendering
    # ------------------------------------------------------------------

    def _render_images_b64(self, file_path: Path) -> list[str]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            try:
                from pdf2image import convert_from_path

                images = convert_from_path(file_path, dpi=self._dpi)
            except ImportError as e:
                raise ProviderPermanentError("pdf2image is required.") from e
            except Exception as e:
                raise ProviderPermanentError(f"Error converting PDF to image: {e}") from e
            if not images:
                raise ProviderPermanentError(f"No pages found in PDF: {file_path}")
            images = images[: self._max_pages]
            out: list[str] = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                out.append(base64.b64encode(buf.getvalue()).decode())
            return out

        if suffix in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
            return [base64.b64encode(file_path.read_bytes()).decode()]

        raise ProviderPermanentError(
            f"Unsupported file type: {suffix}. Supported: .pdf, .png, .jpg, .jpeg, .webp, .tiff, .bmp"
        )

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    async def _extract_async(
        self,
        images_b64: list[str],
        template: dict[str, Any],
        instructions: str,
    ) -> dict[str, Any]:
        api_url = f"{self._server_url.rstrip('/')}/v1/chat/completions"

        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}} for b64 in images_b64]
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
            # NuExtract selects structured-extraction by passing a template
            # (vLLM OpenAI extension — top-level request field).
            "chat_template_kwargs": {
                "template": json.dumps(template),
                "instructions": instructions,
                "enable_thinking": self._enable_thinking,
            },
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    if resp.status in (408, 429, 502, 503, 504):
                        raise ProviderTransientError(f"HTTP {resp.status}: {error_text[:200]}")
                    raise ProviderPermanentError(f"HTTP {resp.status}: {error_text[:200]}")

                result: dict[str, Any] = await resp.json()

        try:
            raw_content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderPermanentError(f"Invalid response format: {e}") from e
        if not raw_content:
            raise ProviderPermanentError("Empty content response from API")

        return {
            "content": str(raw_content),
            "template": template,
            "instructions": instructions,
            "_config": {
                "server_url": self._server_url,
                "model": self._model,
                "dpi": self._dpi,
            },
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(f"NuExtract3ExtractProvider only supports EXTRACT, got {request.product_type}")
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override"
            )

        started_at = datetime.now()

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        template, instructions = _json_schema_to_nuextract_template_and_instructions(request.schema_override)
        if not isinstance(template, dict):
            raise ProviderPermanentError("Top-level schema must be an object (produced a non-dict template)")

        try:
            images_b64 = self._render_images_b64(file_path)
            raw_output = asyncio.run(self._extract_async(images_b64, template, instructions))

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

        except (ProviderPermanentError, ProviderTransientError):
            raise
        except TimeoutError as e:
            raise ProviderTransientError(f"Request timed out after {self._timeout}s") from e
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    # ------------------------------------------------------------------
    # normalize
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_extraction(content: str) -> dict[str, Any]:
        """Parse the model's JSON extraction, never raising.

        Tries a strict parse (raw + de-fenced), then best-effort salvage of a
        truncated/garbage object. Returns ``{}`` if nothing usable is found, so
        a single pathological document (truncated huge array, repetition loop)
        is scored on whatever it produced rather than hard-failing the whole run.
        """
        text = _THINK_RE.sub("", content).strip()
        candidates = [text]
        fence = _FENCE_RE.search(text)
        if fence:
            candidates.append(fence.group(1))

        for cand in candidates:
            try:
                parsed = json.loads(cand)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed

        # Salvage: recover top-level fields + complete array elements before a cut.
        for cand in candidates:
            repaired = _repair_truncated_json_object(cand)
            if repaired is not None:
                return repaired
        return {}

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"NuExtract3ExtractProvider only supports EXTRACT, got {raw_result.product_type}"
            )

        content = raw_result.raw_output.get("content", "")
        extracted_data = self._parse_extraction(content) if content else {}

        output = ExtractOutput(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data,
            field_citations=[],  # NuExtract3 emits no per-field citations / bboxes
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
