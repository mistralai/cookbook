"""Helpers for Gemini Agentic Vision parse-with-layout flows."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from PIL import Image

from extract_bench.inference.providers.base import ProviderPermanentError, ProviderTransientError
from extract_bench.inference.providers.parse._layout_utils import LABEL_MAP, items_to_markdown
from extract_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, ParseLayoutPageIR

logger = logging.getLogger(__name__)

TRANSIENT_ERROR_KEYWORDS = ("timeout", "connection", "network")
RATE_LIMIT_ERROR_KEYWORDS = ("rate_limit", "rate limit", "429", "resource_exhausted")

CORE11_LABELS = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]

SYSTEM_PROMPT_AGENTIC_VISION = (
    "You are a document parser. Convert a document page image into clean, well-structured markdown "
    "with layout grounding.\n\n"
    "Rules:\n"
    "- Preserve reading order.\n"
    "- Preserve document structure, including headings, lists, formulas, captions, and tables.\n"
    "- Use HTML tables for tabular data.\n"
    "- For figures or pictures, describe them briefly in square brackets like [Figure: description].\n"
    "- Do not add commentary outside the requested wrapped content.\n"
    "- Wrap each layout element in a <div> tag with a data-bbox and data-label attribute.\n"
    '- data-bbox must use Gemini native coordinates: "[y_min, x_min, y_max, x_max]".\n'
    "- Coordinates must be normalized to 0..1000 relative to the full original page image.\n"
    "- data-label must be one of: Caption, Footnote, Formula, List-item, Page-footer, "
    "Page-header, Picture, Section-header, Table, Text, Title.\n"
    "- Every piece of content must be inside exactly one <div> wrapper.\n"
    "- Start from the full page image and preserve reading order from that full-page view.\n"
    "- First try to read and ground content from the full page image.\n"
    "- If you zoom, crop, rotate, or enhance the page using code execution, always convert the final box "
    "back to the full original page image coordinate system before returning data-bbox.\n"
    "- Use code execution only if text is too small, dense, rotated, low-contrast, or ambiguous at full-page "
    "scale, or if the bounding box would otherwise be unreliable. Do not guess.\n"
    "- If only one region is ambiguous, inspect only that region. Prefer the smallest crop or zoom needed.\n"
    "- Do not crop or zoom the whole page by default.\n"
    "- Use code execution only for visual inspection, cropping, rotation, or measurement. Do not use it to "
    "construct Python dictionaries, lists, or JSON for the final answer.\n"
    "- Every returned data-bbox must refer to the original full-page coordinate frame, never the crop frame.\n"
    "- After inspection, return the final wrapped markdown as assistant text. If you must use code for the final "
    "step, print only one raw triple-quoted string containing the wrapped markdown.\n"
)

USER_PROMPT_AGENTIC_VISION_PREFIX = (
    "Parse this document page and output its content as clean markdown, with each layout element wrapped in a "
    '<div data-bbox="[y_min,x_min,y_max,x_max]" data-label="Category"> tag. '
    "Use Gemini native bbox order [y_min, x_min, y_max, x_max], normalized to 0..1000 on the full page image.\n"
    "Use HTML tables for any tabular data.\n"
    "For Title and Section-header items, output only the heading text inside the wrapper, "
    "not markdown heading markers.\n"
    "For Formula items, output only the formula content inside the wrapper, not $$ fences.\n"
    "Start from the full page image and only zoom or crop when needed for a specific ambiguous region.\n"
    "Use code execution only to inspect the page. Do not return screenshots, plots, or other artifacts.\n"
    "Use code execution only if text is too small, dense, rotated, low-contrast, or ambiguous at full-page scale, "
    "or if the bbox would otherwise be unreliable.\n"
    "Prefer the smallest crop or zoom needed and do not zoom the whole page by default.\n"
    "Do not use code execution to build Python dictionaries, lists, or JSON for the final answer.\n"
    "Every returned data-bbox must be mapped back to the original full page image, normalized to 0..1000.\n"
    "After inspection, return the wrapped markdown as assistant text. If you must use code for the final step, "
    "print only one raw triple-quoted string containing the wrapped markdown and nothing else.\n"
    "Output ONLY the wrapped content, no explanations.\n"
)

RETRY_PROMPT_RECITATION = (
    "Retry mode: the previous attempt returned no final wrapped markdown and triggered recitation-style behavior.\n"
    "Do not rely on memorized text, web recall, citations, URLs, or external sources.\n"
    "Read only from the attached page image.\n"
    "If you use code execution, execute the final code and print the complete wrapped markdown as a single raw "
    'triple-quoted string containing <div data-bbox="[y_min,x_min,y_max,x_max]" ...> wrappers.\n'
    "Do not stop after writing planning code. The executed code must print the final wrapped markdown.\n"
    "Do not emit citations, URLs, commentary, or any text outside the wrapped markdown.\n"
)

RETRY_PROMPT_EMPTY_OUTPUT = (
    "Retry mode: the previous attempt returned code or citations but no usable wrapped markdown.\n"
    "You must finish this retry with actual wrapped markdown output, not just planning code.\n"
    "If needed, use code execution to inspect crops, then print the complete wrapped markdown as one raw "
    "triple-quoted string.\n"
)

RETRY_PROMPT_FINAL_ONLY = (
    "Final retry: do not write planning code, helper code, comments, crop definitions, or analysis.\n"
    "Return the final wrapped markdown now.\n"
    "Preferred form: execute exactly one print(r'''...''') statement containing the complete wrapped markdown.\n"
    "Alternative form: return the wrapped markdown directly as assistant text.\n"
    "Do not output anything except the wrapped markdown itself.\n"
)

_PATTERN_BBOX_FIRST = re.compile(
    r'<div\s+[^>]*?data-bbox=["\'](\[[^\]]+\])["\'][^>]*?data-label=["\']([^"\']+)["\'][^>]*?>'
    r"([\s\S]*?)</div>",
    re.IGNORECASE,
)
_PATTERN_LABEL_FIRST = re.compile(
    r'<div\s+[^>]*?data-label=["\']([^"\']+)["\'][^>]*?data-bbox=["\'](\[[^\]]+\])["\'][^>]*?>'
    r"([\s\S]*?)</div>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AgenticVisionCacheInfo:
    """Resolved explicit cache metadata for a run."""

    name: str
    display_name: str
    token_count: int
    ttl_seconds: int
    storage_cost_usd: float
    created: bool


@dataclass(frozen=True)
class AgenticVisionPageResponse:
    """Parsed wrapped layout output for one page."""

    raw_content: str
    items: list[dict[str, Any]]


@dataclass
class AgenticVisionPageResult:
    """Per-page parse result plus serialized API call traces."""

    page_index: int
    width: int
    height: int
    image_mime_type: str
    items: list[dict[str, Any]]
    markdown: str
    raw_content: str
    thought_summaries: list[str]
    thought_signatures: list[str]
    generated_code: list[dict[str, Any]]
    code_execution_results: list[dict[str, Any]]
    api_calls: list[dict[str, Any]]


def estimate_text_tokens(text: str) -> int:
    """Very rough token estimate for cache gating without extra API calls."""
    return max(1, len(text) // 4)


def build_page_prompt_suffix(page_width: int, page_height: int) -> str:
    """Return page-specific prompt instructions kept separate from the cached prefix."""
    return (
        f"Page image dimensions: {page_width}x{page_height} pixels.\n"
        "The attached page image is the original full-page reference frame.\n"
        "If you use code execution to zoom or crop, convert final boxes back to the original full page before "
        "returning data-bbox.\n"
    )


def identify_part_kind(part: Any) -> str:
    """Return a stable kind string for a Gemini content part."""
    if getattr(part, "executable_code", None) is not None:
        return "executable_code"
    if getattr(part, "code_execution_result", None) is not None:
        return "code_execution_result"
    if getattr(part, "inline_data", None) is not None:
        return "inline_data"
    if getattr(part, "file_data", None) is not None:
        return "file_data"
    if getattr(part, "function_call", None) is not None:
        return "function_call"
    if getattr(part, "function_response", None) is not None:
        return "function_response"
    if getattr(part, "text", None) is not None:
        return "thought_text" if getattr(part, "thought", False) else "text"
    return "unknown"


def summarize_part_for_request(part: Any) -> dict[str, Any]:
    """Serialize a request part without embedding large binary payloads."""
    kind = identify_part_kind(part)
    summary: dict[str, Any] = {"kind": kind}
    text = getattr(part, "text", None)
    if text is not None:
        summary["text"] = text
    inline_data = getattr(part, "inline_data", None)
    if inline_data is not None:
        summary["inline_data"] = {
            "mime_type": getattr(inline_data, "mime_type", None),
            "data_size_bytes": len(getattr(inline_data, "data", b"") or b""),
        }
    file_data = getattr(part, "file_data", None)
    if file_data is not None:
        summary["file_data"] = {
            "mime_type": getattr(file_data, "mime_type", None),
            "file_uri": getattr(file_data, "file_uri", None),
        }
    if getattr(part, "thought", False):
        summary["thought"] = True
    thought_signature = getattr(part, "thought_signature", None)
    if thought_signature:
        summary["thought_signature"] = normalize_signature(thought_signature)
    return summary


def serialize_part(part: Any, part_index: int) -> dict[str, Any]:
    """Serialize a Gemini content part into a JSON-safe dict."""
    serialized = safe_model_dump(part)
    if serialized is None:
        serialized = {}
    if not isinstance(serialized, dict):
        serialized = {"value": serialized}
    serialized["kind"] = identify_part_kind(part)
    serialized["part_index"] = part_index
    thought_signature = getattr(part, "thought_signature", None)
    if thought_signature:
        serialized["thought_signature"] = normalize_signature(thought_signature)
    return serialized


def safe_model_dump(value: Any) -> Any:
    """Best-effort JSON-safe serializer for SDK objects."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", exclude_none=True)
        except TypeError:
            return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {k: safe_model_dump(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [safe_model_dump(v) for v in value]
    if isinstance(value, tuple):
        return [safe_model_dump(v) for v in value]
    if isinstance(value, bytes):
        return value.hex()
    return value


def normalize_signature(signature: Any) -> str:
    """Return a stable string representation for a thought signature payload."""
    if isinstance(signature, bytes):
        return signature.hex()
    return str(signature)


def extract_candidate_parts(response: Any) -> list[Any]:
    """Extract parts from the first candidate content."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None)
    return list(parts or [])


def extract_finish_reason(response: Any) -> str | None:
    """Extract the first candidate finish reason, if any."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    finish_reason = getattr(candidates[0], "finish_reason", None)
    return str(finish_reason) if finish_reason else None


def is_recitation_finish_reason(finish_reason: str | None) -> bool:
    """Return whether a finish reason represents Gemini's RECITATION stop."""
    return bool(finish_reason and "RECITATION" in finish_reason.upper())


def response_has_citations(response: Any) -> bool:
    """Return whether the first candidate carries citation metadata."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return False
    citation_metadata = getattr(candidates[0], "citation_metadata", None)
    citations = getattr(citation_metadata, "citations", None)
    return bool(citations)


def build_retry_instruction(response: Any, last_error: str, *, attempt: int) -> str | None:
    """Return an adaptive retry instruction for recitation and empty-output failures."""
    finish_reason = extract_finish_reason(response)
    parts = extract_candidate_parts(response)
    has_code = any(getattr(part, "executable_code", None) is not None for part in parts)
    has_code_output = any(getattr(getattr(part, "code_execution_result", None), "output", None) for part in parts)

    instructions: list[str] = []
    if is_recitation_finish_reason(finish_reason) or response_has_citations(response):
        instructions.append(RETRY_PROMPT_RECITATION)
    if "No wrapped layout payload found" in last_error or "No valid wrapped layout payload found" in last_error:
        instructions.append(RETRY_PROMPT_EMPTY_OUTPUT)
    if has_code and not has_code_output:
        instructions.append(
            "The previous attempt produced executable code but no printed final answer. "
            "This retry must execute code that prints the final wrapped markdown."
        )
    if attempt >= 2 and has_code and not has_code_output:
        instructions.append(RETRY_PROMPT_FINAL_ONLY)

    if not instructions:
        return None
    return "\n".join(instructions)


def extract_serialized_response_parts(response: Any) -> list[dict[str, Any]]:
    """Serialize all first-candidate parts."""
    return [serialize_part(part, idx) for idx, part in enumerate(extract_candidate_parts(response))]


def extract_thought_summaries(response: Any) -> list[str]:
    """Extract exposed thought summary text parts."""
    summaries: list[str] = []
    for part in extract_candidate_parts(response):
        if getattr(part, "thought", False) and getattr(part, "text", None):
            summaries.append(str(part.text))
    return summaries


def extract_thought_signatures(response: Any) -> list[str]:
    """Extract exposed thought signatures from all parts."""
    signatures: list[str] = []
    for part in extract_candidate_parts(response):
        thought_signature = getattr(part, "thought_signature", None)
        if thought_signature:
            signatures.append(normalize_signature(thought_signature))
    return signatures


def extract_generated_code(response: Any) -> list[dict[str, Any]]:
    """Extract generated code parts."""
    code_parts: list[dict[str, Any]] = []
    for idx, part in enumerate(extract_candidate_parts(response)):
        executable_code = getattr(part, "executable_code", None)
        if executable_code is not None:
            payload = safe_model_dump(executable_code)
            if isinstance(payload, dict):
                payload["part_index"] = idx
            code_parts.append(payload)
    return code_parts


def extract_code_execution_results(response: Any) -> list[dict[str, Any]]:
    """Extract code execution result parts."""
    results: list[dict[str, Any]] = []
    for idx, part in enumerate(extract_candidate_parts(response)):
        execution_result = getattr(part, "code_execution_result", None)
        if execution_result is not None:
            payload = safe_model_dump(execution_result)
            if isinstance(payload, dict):
                payload["part_index"] = idx
            results.append(payload)
    return results


def extract_final_text(response: Any) -> str:
    """Extract the last non-thought text part from a response."""
    final_text = ""
    for part in extract_candidate_parts(response):
        text = getattr(part, "text", None)
        if text and not getattr(part, "thought", False):
            final_text = str(text)
    return final_text


def _candidate_layout_payloads(response: Any) -> list[str]:
    """Return possible final wrapped-markdown payloads from text and code-execution outputs."""
    payloads: list[str] = []
    final_text = extract_final_text(response)
    if final_text:
        payloads.append(final_text)

    for part in reversed(extract_candidate_parts(response)):
        execution_result = getattr(part, "code_execution_result", None)
        output = getattr(execution_result, "output", None) if execution_result is not None else None
        if output:
            payloads.append(str(output))
    return payloads


def _normalize_bbox_2d(value: object) -> list[int]:
    """Normalize a candidate bbox payload into Gemini-native integer coordinates."""
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox must be a list of four coordinates")
    coords = [int(round(float(v))) for v in value]
    y_min, x_min, y_max, x_max = coords
    if min(coords) < 0 or max(coords) > 1000:
        raise ValueError("bbox coordinates must be in the 0..1000 range")
    if y_max < y_min or x_max < x_min:
        raise ValueError("bbox must be ordered as [y_min, x_min, y_max, x_max]")
    return coords


def parse_agentic_layout_blocks(content: str) -> AgenticVisionPageResponse:
    """Parse wrapped layout blocks using Gemini-native y-first bbox ordering."""
    raw_matches: list[tuple[int, list[int], str, str, str]] = []

    for match in _PATTERN_BBOX_FIRST.finditer(content):
        try:
            bbox = _normalize_bbox_2d(json.loads(match.group(1)))
        except Exception:
            continue
        raw_matches.append((match.start(), bbox, match.group(2), match.group(3).strip(), match.group(0).strip()))

    for match in _PATTERN_LABEL_FIRST.finditer(content):
        try:
            bbox = _normalize_bbox_2d(json.loads(match.group(2)))
        except Exception:
            continue
        raw_matches.append((match.start(), bbox, match.group(1), match.group(3).strip(), match.group(0).strip()))

    raw_matches.sort(key=lambda item: item[0])

    items: list[dict[str, Any]] = []
    wrapper_blocks: list[str] = []
    seen_positions: set[int] = set()
    for pos, bbox, label, text, full_block in raw_matches:
        if pos in seen_positions:
            continue
        seen_positions.add(pos)
        items.append(
            {
                "bbox_2d": bbox,
                "label": normalize_label(label),
                "text": text,
            }
        )
        wrapper_blocks.append(full_block)

    return AgenticVisionPageResponse(raw_content="\n\n".join(wrapper_blocks), items=items)


def parse_page_response(response: Any) -> AgenticVisionPageResponse:
    """Parse the final wrapped layout response from a Gemini response."""
    errors: list[str] = []
    for payload in _candidate_layout_payloads(response):
        parsed = parse_agentic_layout_blocks(payload)
        if parsed.items:
            return parsed
        errors.append("No wrapped layout blocks found")

    if errors:
        raise ValueError(f"No valid wrapped layout payload found in Gemini response: {errors[-1]}")
    raise ValueError("No wrapped layout payload found in Gemini response")


def normalize_label(label: str) -> str:
    """Canonicalize a raw label string into the benchmark label set."""
    return LABEL_MAP.get(label.lower(), label)


def infer_item_type(label: str) -> str:
    """Infer normalized item type from a Core11 label."""
    norm_label = label.lower()
    if norm_label == "table":
        return "table"
    if norm_label in ("picture", "figure"):
        return "image"
    return "text"


def bbox_2d_to_xyxy(bbox_2d: list[int]) -> list[int]:
    """Convert Gemini-native [y_min, x_min, y_max, x_max] to x-first [x1, y1, x2, y2]."""
    y_min, x_min, y_max, x_max = bbox_2d
    return [x_min, y_min, x_max, y_max]


def build_layout_pages_from_agentic_items(
    items_data: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    page_number: int,
) -> tuple[str, list[ParseLayoutPageIR]]:
    """Convert wrapped Agentic Vision items to page markdown and ParseLayoutPageIR."""
    if not items_data or not image_width or not image_height:
        return "", []

    markdown = items_to_markdown(items_data)
    layout_items: list[LayoutItemIR] = []
    for item in items_data:
        try:
            bbox_2d = _normalize_bbox_2d(item.get("bbox_2d", []))
        except (TypeError, ValueError):
            continue
        x1, y1, x2, y2 = bbox_2d_to_xyxy(bbox_2d)
        text = str(item.get("text", ""))
        label = normalize_label(str(item.get("label", "Text")))
        item_type = infer_item_type(label)
        seg = LayoutSegmentIR(
            x=x1 / 1000.0,
            y=y1 / 1000.0,
            w=(x2 - x1) / 1000.0,
            h=(y2 - y1) / 1000.0,
            confidence=1.0,
            label=label,
        )
        layout_items.append(
            LayoutItemIR(
                type=item_type,
                md=text,
                html=text if item_type == "table" else "",
                value=text,
                bbox=seg,
                layout_segments=[seg],
            )
        )

    return markdown, [
        ParseLayoutPageIR(
            page_number=page_number,
            width=float(image_width),
            height=float(image_height),
            md=markdown,
            items=layout_items,
        )
    ]


class GoogleAgenticVisionRunner:
    """One-call-per-page Agentic Vision runner with optional explicit prefix caching."""

    def __init__(
        self,
        *,
        client: Any,
        types_module: Any,
        model: str,
        max_output_tokens: int,
        thinking_level: str | None,
        enable_explicit_context_cache: bool,
        context_cache_ttl_seconds: int,
        min_cacheable_tokens: int,
        input_cost_per_million: float,
        cache_hit_cost_per_million: float,
        cache_storage_cost_per_million_token_hour: float,
        expected_page_calls: int,
    ) -> None:
        self._client = client
        self._types = types_module
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._thinking_level = thinking_level
        self._enable_explicit_context_cache = enable_explicit_context_cache
        self._context_cache_ttl_seconds = context_cache_ttl_seconds
        self._min_cacheable_tokens = min_cacheable_tokens
        self._input_cost_per_million = input_cost_per_million
        self._cache_hit_cost_per_million = cache_hit_cost_per_million
        self._cache_storage_cost_per_million_token_hour = cache_storage_cost_per_million_token_hour
        self._expected_page_calls = expected_page_calls
        self._cache_info: AgenticVisionCacheInfo | None = None
        self._cache_error: str | None = None

    @property
    def cache_info(self) -> AgenticVisionCacheInfo | None:
        return self._cache_info

    @property
    def cache_error(self) -> str | None:
        return self._cache_error

    def _maybe_create_prefix_cache(self) -> AgenticVisionCacheInfo | None:
        if self._cache_info is not None:
            return self._cache_info
        if not self._enable_explicit_context_cache:
            return None
        if self._expected_page_calls < 2:
            return None

        estimated_tokens = estimate_text_tokens(SYSTEM_PROMPT_AGENTIC_VISION + USER_PROMPT_AGENTIC_VISION_PREFIX)
        if estimated_tokens < self._min_cacheable_tokens:
            logger.info(
                "Agentic Vision prompt estimate (%s) is below min_cacheable_tokens (%s); "
                "attempting cache creation anyway because Gemini cache tokenization can exceed the heuristic.",
                estimated_tokens,
                self._min_cacheable_tokens,
            )

        display_name = (
            "extract-bench-gemini-agentic-vision-prefix-"
            + hashlib.sha256(
                f"{self._model}|{SYSTEM_PROMPT_AGENTIC_VISION}|{USER_PROMPT_AGENTIC_VISION_PREFIX}".encode()
            ).hexdigest()[:16]
        )

        try:
            cache = self._client.caches.create(
                model=self._model,
                config=self._types.CreateCachedContentConfig(
                    display_name=display_name,
                    system_instruction=SYSTEM_PROMPT_AGENTIC_VISION,
                    contents=[
                        self._types.Content(
                            role="user",
                            parts=[self._types.Part.from_text(text=USER_PROMPT_AGENTIC_VISION_PREFIX)],
                        )
                    ],
                    tools=[self._types.Tool(code_execution=self._types.ToolCodeExecution())],
                    ttl=timedelta(seconds=self._context_cache_ttl_seconds),
                ),
            )
        except Exception as exc:
            logger.warning("Failed to create Gemini context cache for Agentic Vision: %s", exc)
            self._cache_error = str(exc)
            return None

        token_count = int(getattr(getattr(cache, "usage_metadata", None), "total_token_count", 0) or 0)
        ttl_hours = self._context_cache_ttl_seconds / 3600.0
        storage_cost_usd = (
            token_count * self._cache_storage_cost_per_million_token_hour * ttl_hours / 1_000_000
            if token_count > 0
            else 0.0
        )
        self._cache_info = AgenticVisionCacheInfo(
            name=str(getattr(cache, "name", "")),
            display_name=display_name,
            token_count=token_count,
            ttl_seconds=self._context_cache_ttl_seconds,
            storage_cost_usd=storage_cost_usd,
            created=True,
        )
        return self._cache_info

    def _build_generation_config(self, cache_name: str | None) -> Any:
        config = self._types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=self._max_output_tokens,
            tools=[self._types.Tool(code_execution=self._types.ToolCodeExecution())],
        )
        if cache_name:
            config.cached_content = cache_name
        else:
            config.system_instruction = SYSTEM_PROMPT_AGENTIC_VISION
        if self._thinking_level is not None:
            config.thinking_config = self._types.ThinkingConfig(
                thinking_level=self._thinking_level,
                include_thoughts=True,
            )
        return config

    def _build_contents(
        self,
        *,
        image_bytes: bytes,
        image_mime_type: str,
        page_width: int,
        page_height: int,
        use_cached_prefix: bool,
        retry_instruction: str | None = None,
    ) -> list[Any]:
        parts = []
        if not use_cached_prefix:
            parts.append(self._types.Part.from_text(text=USER_PROMPT_AGENTIC_VISION_PREFIX))
        parts.append(self._types.Part.from_text(text=build_page_prompt_suffix(page_width, page_height)))
        if retry_instruction:
            parts.append(self._types.Part.from_text(text=retry_instruction))
        parts.append(self._types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type))
        return [self._types.Content(role="user", parts=parts)]

    def parse_page(
        self,
        *,
        page_index: int,
        image: Image.Image,
        image_bytes: bytes,
        image_mime_type: str,
        max_attempts: int = 3,
    ) -> AgenticVisionPageResult:
        """Run one Agentic Vision page parse with retry on malformed final wrapped output."""
        cache_info = self._maybe_create_prefix_cache()
        use_cached_prefix = cache_info is not None
        cache_name = cache_info.name if cache_info is not None else None

        api_calls: list[dict[str, Any]] = []
        last_error = "No attempts executed"
        retry_instruction: str | None = None
        width, height = image.size

        for attempt in range(1, max_attempts + 1):
            contents = self._build_contents(
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                page_width=width,
                page_height=height,
                use_cached_prefix=use_cached_prefix,
                retry_instruction=retry_instruction,
            )
            request_summary = {
                "system_instruction": SYSTEM_PROMPT_AGENTIC_VISION if not use_cached_prefix else None,
                "user_prompt_prefix": None if use_cached_prefix else USER_PROMPT_AGENTIC_VISION_PREFIX,
                "page_prompt_suffix": build_page_prompt_suffix(width, height),
                "retry_instruction": retry_instruction,
                "used_cached_content": bool(cache_name),
                "cache_name": cache_name,
                "contents": [
                    {
                        "role": getattr(content, "role", None),
                        "parts": [summarize_part_for_request(part) for part in getattr(content, "parts", []) or []],
                    }
                    for content in contents
                ],
            }

            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=self._build_generation_config(cache_name),
                )
            except Exception as exc:
                raise classify_gemini_api_exception(exc) from exc

            usage = extract_usage_from_response(response)
            response_parts = extract_serialized_response_parts(response)
            thought_summaries = extract_thought_summaries(response)
            thought_signatures = extract_thought_signatures(response)
            generated_code = extract_generated_code(response)
            execution_results = extract_code_execution_results(response)
            final_text = extract_final_text(response)

            try:
                parsed = parse_page_response(response)
            except Exception as exc:
                last_error = str(exc)
                parsed = None

            call_record = {
                "page_index": page_index,
                "attempt": attempt,
                "request": request_summary,
                "response": safe_model_dump(response),
                "response_parts": response_parts,
                "usage": usage,
                "final_text": final_text,
                "cost_usd": 0.0,
            }
            api_calls.append(call_record)

            if parsed is not None:
                markdown = items_to_markdown(parsed.items)
                return AgenticVisionPageResult(
                    page_index=page_index,
                    width=width,
                    height=height,
                    image_mime_type=image_mime_type,
                    items=parsed.items,
                    markdown=markdown,
                    raw_content=parsed.raw_content,
                    thought_summaries=thought_summaries,
                    thought_signatures=thought_signatures,
                    generated_code=generated_code,
                    code_execution_results=execution_results,
                    api_calls=api_calls,
                )

            retry_instruction = build_retry_instruction(response, last_error, attempt=attempt)

        raise ProviderPermanentError(
            f"Failed to obtain valid Agentic Vision wrapped layout output after {max_attempts} attempts: {last_error}",
            debug_payload={
                "mode": "parse_with_layout_agentic_vision",
                "page_index": page_index,
                "page_width": width,
                "page_height": height,
                "image_mime_type": image_mime_type,
                "api_calls": api_calls,
                "last_error": last_error,
            },
        )


def extract_usage_from_response(response: Any) -> dict[str, int]:
    """Extract all usage buckets relevant to Agentic Vision accounting."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {
            "input_tokens": 0,
            "tool_use_prompt_tokens": 0,
            "cached_content_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
        }
    return {
        "input_tokens": int(getattr(meta, "prompt_token_count", 0) or 0),
        "tool_use_prompt_tokens": int(getattr(meta, "tool_use_prompt_token_count", 0) or 0),
        "cached_content_tokens": int(getattr(meta, "cached_content_token_count", 0) or 0),
        "output_tokens": int(getattr(meta, "candidates_token_count", 0) or 0),
        "thinking_tokens": int(getattr(meta, "thoughts_token_count", 0) or 0),
        "total_tokens": int(getattr(meta, "total_token_count", 0) or 0),
    }


def classify_gemini_api_exception(exc: Exception) -> Exception:
    """Classify raw SDK exceptions into retryable provider errors when possible."""
    error_str = str(exc).lower()
    if any(keyword in error_str for keyword in TRANSIENT_ERROR_KEYWORDS):
        return ProviderTransientError(f"Transient error calling Gemini API: {exc}")
    if any(keyword in error_str for keyword in RATE_LIMIT_ERROR_KEYWORDS):
        return ProviderTransientError(f"Rate limited: {exc}")
    return ProviderPermanentError(f"Error calling Gemini API: {exc}")
