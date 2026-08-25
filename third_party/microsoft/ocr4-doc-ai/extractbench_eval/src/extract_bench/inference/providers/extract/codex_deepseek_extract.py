"""Codex CLI extraction with DeepSeek as the Codex model provider."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

from extract_bench.inference.providers.base import ProviderConfigError
from extract_bench.inference.providers.extract.codex_code_extract import CodexCodeExtractProvider
from extract_bench.inference.providers.extract.deepseek_extract import (
    _DEEPSEEK_BASE_URL,
    _DEEPSEEK_EXTRACT_PRICING_PER_M,
    _FIREWORKS_BASE_URL,
    _FIREWORKS_DEEPSEEK_V4_PRO_MODEL,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult

_PROXY_ENV_KEY = "CODEX_DEEPSEEK_PROXY_KEY"
_PROXY_BEARER = "local-codex-deepseek-proxy"
_GLM_PROXY_ENV_KEY = "CODEX_GLM_PROXY_KEY"
_GLM_PROXY_BEARER = "local-codex-glm-proxy"
_GLM_API_BASE_URL = "https://api.fireworks.ai/inference/v1"
_GLM_5_2_MODEL = "accounts/fireworks/models/glm-5p2"
_SUPPORTED_CODEX_TOOL_NAMES = {"exec_command", "write_stdin"}

# USD per million tokens: input, cached input, output.
_GLM_EXTRACT_PRICING_PER_M: dict[str, tuple[float, float, float]] = {
    _GLM_5_2_MODEL: (1.40, 0.26, 4.40),
}


class _DaemonThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False


@dataclass
class _ProxyUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    requests: int = 0
    cost_usd: float = 0.0
    last_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: dict[str, int], cost_usd: float) -> None:
        with self._lock:
            self.input_tokens += int(usage.get("input_tokens", 0) or 0)
            self.cached_input_tokens += int(usage.get("cached_input_tokens", 0) or 0)
            self.output_tokens += int(usage.get("output_tokens", 0) or 0)
            self.reasoning_output_tokens += int(usage.get("reasoning_output_tokens", 0) or 0)
            self.requests += 1
            self.cost_usd += cost_usd

    def set_error(self, error: str) -> None:
        with self._lock:
            self.last_error = error

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "input_tokens": self.input_tokens,
                "cached_input_tokens": self.cached_input_tokens,
                "output_tokens": self.output_tokens,
                "reasoning_output_tokens": self.reasoning_output_tokens,
                "requests": self.requests,
                "cost_usd": self.cost_usd,
                "last_error": self.last_error,
            }


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(p for p in parts if p)


def _responses_input_to_chat_messages(instructions: str | None, input_items: Any) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    if not isinstance(input_items, list):
        if input_items:
            messages.append({"role": "user", "content": str(input_items)})
        return messages

    for item in input_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            role = item.get("role")
            if role in ("developer", "system"):
                role = "system"
            elif role not in ("user", "assistant"):
                role = "user"
            messages.append({"role": role, "content": _extract_text(item.get("content"))})
        elif item_type == "function_call":
            call_id = str(item.get("call_id") or item.get("id") or f"call_{len(messages)}")
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name") or "",
                                "arguments": item.get("arguments") or "{}",
                            },
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or item.get("id") or ""),
                    "content": _extract_text(item.get("output") or item.get("content")),
                }
            )

    return messages


def _responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        if tool.get("name") not in _SUPPORTED_CODEX_TOOL_NAMES:
            continue
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name") or "",
                    "description": tool.get("description") or "",
                    "parameters": parameters,
                },
            }
        )
    return out


def _usage_from_chat_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "total_tokens": 0,
        }
    input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
    output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
    total_tokens = getattr(usage, "total_tokens", None) or input_tokens + output_tokens
    cached_input_tokens = getattr(usage, "prompt_cache_hit_tokens", None) or 0
    reasoning_tokens = 0

    for details_name in ("prompt_tokens_details", "input_tokens_details"):
        details = getattr(usage, details_name, None)
        if details is None:
            continue
        cached_input_tokens = (
            getattr(details, "cached_tokens", None)
            or getattr(details, "cached_input_tokens", None)
            or cached_input_tokens
        )

    for details_name in ("completion_tokens_details", "output_tokens_details"):
        details = getattr(usage, details_name, None)
        if details is None:
            continue
        reasoning_tokens = (
            getattr(details, "reasoning_tokens", None)
            or getattr(details, "reasoning_output_tokens", None)
            or reasoning_tokens
        )

    return {
        "input_tokens": int(input_tokens),
        "cached_input_tokens": int(cached_input_tokens),
        "output_tokens": int(output_tokens),
        "reasoning_output_tokens": int(reasoning_tokens),
        "total_tokens": int(total_tokens),
    }


class _DeepSeekResponsesProxy:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int | None,
        thinking_type: str | None,
        pricing: tuple[float, float, float],
        timeout_s: float,
        reasoning_effort: str | None = None,
        display_name: str = "DeepSeek Proxy",
        description: str = "DeepSeek model exposed through a local Codex Responses proxy.",
        context_window: int = 272000,
        comp_hash: str = "deepseek-proxy",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.thinking_type = thinking_type
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s
        self.display_name = display_name
        self.description = description
        self.context_window = context_window
        self.comp_hash = comp_hash
        self.input_price_per_1m, self.cached_input_price_per_1m, self.output_price_per_1m = pricing
        self.usage = _ProxyUsage()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("proxy is not started")
        address = self._server.server_address
        if not isinstance(address, tuple) or len(address) < 2:
            raise RuntimeError("unexpected proxy server address")
        port = int(address[1])
        return f"http://127.0.0.1:{port}/v1"

    def start(self) -> None:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                proxy._handle_get(self)

            def do_POST(self) -> None:  # noqa: N802
                proxy._handle_post(self)

            def log_message(self, *_args: Any) -> None:
                return

        self._server = _DaemonThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(handler.path)
        if parsed.path != "/v1/models":
            self._send_json(handler, 404, {"error": {"message": "not found"}})
            return
        self._send_json(
            handler,
            200,
            {
                "models": [
                    {
                        "slug": self.model,
                        "display_name": self.display_name,
                        "description": self.description,
                        "default_reasoning_level": "none",
                        "supported_reasoning_levels": [],
                        "shell_type": "shell_command",
                        "visibility": "list",
                        "supported_in_api": True,
                        "priority": 0,
                        "additional_speed_tiers": [],
                        "service_tiers": [],
                        "availability_nux": {"message": ""},
                        "upgrade": None,
                        "base_instructions": "",
                        "model_messages": {},
                        "supports_reasoning_summaries": False,
                        "default_reasoning_summary": "none",
                        "support_verbosity": False,
                        "default_verbosity": "low",
                        "apply_patch_tool_type": "freeform",
                        "web_search_tool_type": "text_and_image",
                        "truncation_policy": {"mode": "tokens", "limit": 10000},
                        "supports_parallel_tool_calls": False,
                        "supports_image_detail_original": False,
                        "context_window": self.context_window,
                        "max_context_window": self.context_window,
                        "comp_hash": self.comp_hash,
                        "effective_context_window_percent": 95,
                        "experimental_supported_tools": [],
                        "input_modalities": ["text"],
                        "supports_search_tool": False,
                        "use_responses_lite": False,
                        "id": self.model,
                    }
                ]
            },
        )

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(handler.path)
        if parsed.path != "/v1/responses":
            self._send_json(handler, 404, {"error": {"message": "not found"}})
            return
        try:
            length = int(handler.headers.get("content-length") or 0)
            body = json.loads(handler.rfile.read(length))
            response = self._call_deepseek(body)
            self._send_sse_response(handler, response)
        except Exception as e:  # noqa: BLE001 - preserve the provider error in Codex stderr/retry logs.
            self.usage.set_error(str(e))
            self._send_json(handler, 500, {"error": {"message": str(e)}})

    def _call_deepseek(self, body: dict[str, Any]) -> Any:
        messages = _responses_input_to_chat_messages(body.get("instructions"), body.get("input"))
        tools = _responses_tools_to_chat_tools(body.get("tools"))
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.thinking_type and "fireworks.ai" not in self.base_url:
            kwargs["extra_body"] = {"thinking": {"type": self.thinking_type}}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s, max_retries=0)
        try:
            return client.chat.completions.create(**kwargs)
        finally:
            client.close()

    def _send_sse_response(self, handler: BaseHTTPRequestHandler, chat_response: Any) -> None:
        choice = chat_response.choices[0]
        message = choice.message
        usage = _usage_from_chat_response(chat_response)
        self.usage.add(usage, self._estimate_cost_usd(usage))

        response_id = getattr(chat_response, "id", None) or f"resp_{time.time_ns()}"
        output_items = self._response_output_items(message)

        handler.send_response(200)
        handler.send_header("content-type", "text/event-stream")
        handler.send_header("cache-control", "no-cache")
        handler.end_headers()

        self._send_sse(
            handler,
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": self.model,
                    "output": [],
                },
            },
        )

        for output_index, item in enumerate(output_items):
            if item["type"] == "function_call":
                self._send_function_call_events(handler, output_index, item)
            else:
                self._send_message_events(handler, output_index, item)

        self._send_sse(
            handler,
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "model": self.model,
                    "output": output_items,
                    "usage": {
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": usage["output_tokens"],
                        "total_tokens": usage["total_tokens"],
                        "input_tokens_details": {"cached_tokens": usage["cached_input_tokens"]},
                        "output_tokens_details": {"reasoning_tokens": usage["reasoning_output_tokens"]},
                    },
                },
            },
        )
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()

    @staticmethod
    def _response_output_items(message: Any) -> list[dict[str, Any]]:
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            items: list[dict[str, Any]] = []
            for idx, tool_call in enumerate(tool_calls):
                function = getattr(tool_call, "function", None)
                call_id = getattr(tool_call, "id", None) or f"call_{idx}"
                items.append(
                    {
                        "id": f"fc_{call_id}",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": call_id,
                        "name": getattr(function, "name", "") if function is not None else "",
                        "arguments": getattr(function, "arguments", "{}") if function is not None else "{}",
                    }
                )
            return items

        text = getattr(message, "content", None) or ""
        return [
            {
                "id": f"msg_{time.time_ns()}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ]

    def _send_function_call_events(
        self,
        handler: BaseHTTPRequestHandler,
        output_index: int,
        item: dict[str, Any],
    ) -> None:
        item_id = item["id"]
        arguments = item.get("arguments") or "{}"
        self._send_sse(
            handler,
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {**item, "status": "in_progress", "arguments": ""},
            },
        )
        self._send_sse(
            handler,
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "output_index": output_index,
                "item_id": item_id,
                "delta": arguments,
            },
        )
        self._send_sse(
            handler,
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "output_index": output_index,
                "item_id": item_id,
                "arguments": arguments,
            },
        )
        self._send_sse(
            handler,
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item,
            },
        )

    def _send_message_events(self, handler: BaseHTTPRequestHandler, output_index: int, item: dict[str, Any]) -> None:
        item_id = item["id"]
        text = _extract_text(item.get("content"))
        self._send_sse(
            handler,
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {**item, "status": "in_progress", "content": []},
            },
        )
        self._send_sse(
            handler,
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "output_index": output_index,
                "item_id": item_id,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            },
        )
        self._send_sse(
            handler,
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "output_index": output_index,
                "item_id": item_id,
                "content_index": 0,
                "delta": text,
            },
        )
        self._send_sse(
            handler,
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "output_index": output_index,
                "item_id": item_id,
                "content_index": 0,
                "text": text,
            },
        )
        self._send_sse(
            handler,
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "output_index": output_index,
                "item_id": item_id,
                "content_index": 0,
                "part": {"type": "output_text", "text": text},
            },
        )
        self._send_sse(
            handler,
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item,
            },
        )

    def _estimate_cost_usd(self, usage: dict[str, int]) -> float:
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
        return (
            uncached_input_tokens / 1_000_000 * self.input_price_per_1m
            + cached_input_tokens / 1_000_000 * self.cached_input_price_per_1m
            + output_tokens / 1_000_000 * self.output_price_per_1m
        )

    @staticmethod
    def _send_json(handler: BaseHTTPRequestHandler, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        handler.send_response(status)
        handler.send_header("content-type", "application/json")
        handler.send_header("content-length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    @staticmethod
    def _send_sse(handler: BaseHTTPRequestHandler, event: str, data: dict[str, Any]) -> None:
        handler.wfile.write(f"event: {event}\n".encode())
        handler.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
        handler.wfile.flush()


@register_provider("codex_deepseek_extract")
class CodexDeepSeekExtractProvider(CodexCodeExtractProvider):
    """Use Codex CLI with a DeepSeek-backed Responses-compatible model provider."""

    DEFAULT_MODEL = "deepseek-v4-flash"
    DEFAULT_MAX_TOKENS = None

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        cfg = dict(base_config or {})
        self._deepseek_base_url: str = cfg.get(
            "base_url",
            _FIREWORKS_BASE_URL
            if str(cfg.get("model", self.DEFAULT_MODEL)).startswith("accounts/fireworks/models/")
            else _DEEPSEEK_BASE_URL,
        )
        self._uses_fireworks = "fireworks.ai" in self._deepseek_base_url
        self._deepseek_api_key: str | None = cfg.get("deepseek_api_key") or cfg.get("api_key")
        if self._deepseek_api_key is None:
            import os

            self._deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self._deepseek_api_key:
            if self._uses_fireworks:
                raise ProviderConfigError("DEEPSEEK_API_KEY is required for Fireworks-backed codex_deepseek_extract.")
            raise ProviderConfigError("DeepSeek API key is required for codex_deepseek_extract.")

        self._deepseek_thinking_type: str | None = cfg.get("thinking_type")
        self._deepseek_reasoning_effort: str | None = cfg.get("reasoning_effort")
        self._deepseek_timeout_s = float(cfg.get("deepseek_timeout_s", 120.0))
        max_tokens_cfg = cfg.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        self._deepseek_max_tokens: int | None = int(max_tokens_cfg) if max_tokens_cfg is not None else None
        self._thread_state = threading.local()
        self._last_proxy_usage: dict[str, Any] = {}

        parent_config = dict(cfg)
        parent_config.pop("api_key", None)
        parent_config.pop("deepseek_api_key", None)
        parent_config["model"] = cfg.get("model", self.DEFAULT_MODEL)
        # Keep the model/provider settings isolated to this invocation.
        parent_config["ignore_user_config"] = True
        super().__init__(provider_name, parent_config)

    @staticmethod
    def _pricing_for_model(model: str) -> tuple[float, float, float]:
        matches = [
            (prefix, rates) for prefix, rates in _DEEPSEEK_EXTRACT_PRICING_PER_M.items() if model.startswith(prefix)
        ]
        return max(matches, key=lambda item: len(item[0]))[1] if matches else (0.0, 0.0, 0.0)

    def _extra_subprocess_env(self) -> dict[str, str]:
        env = {_PROXY_ENV_KEY: _PROXY_BEARER}
        codex_home = getattr(self._thread_state, "codex_home", None)
        if codex_home:
            env["CODEX_HOME"] = codex_home
        return env

    def _build_cmd(
        self,
        *,
        workdir: Path,
        last_message_path: Path,
        image_path: Path | None = None,
    ) -> list[str]:
        proxy_url = getattr(self._thread_state, "proxy_url", None)
        if not proxy_url:
            raise RuntimeError("codex_deepseek_extract proxy URL was not initialized")
        codex_home = workdir / ".codex-home"
        codex_home.mkdir(parents=True, exist_ok=True)
        self._thread_state.codex_home = str(codex_home)
        cmd = super()._build_cmd(workdir=workdir, last_message_path=last_message_path, image_path=image_path)
        insert_at = cmd.index("--output-last-message")
        proxy_overrides = [
            "-c",
            'model_provider="deepseek_proxy"',
            "-c",
            'model_providers.deepseek_proxy.name="DeepSeek Proxy"',
            "-c",
            f'model_providers.deepseek_proxy.base_url="{proxy_url}"',
            "-c",
            f'model_providers.deepseek_proxy.env_key="{_PROXY_ENV_KEY}"',
            "-c",
            'model_providers.deepseek_proxy.wire_api="responses"',
        ]
        cmd[insert_at:insert_at] = proxy_overrides
        return cmd

    def _pricing_snapshot(self, usage: dict[str, int]) -> dict[str, Any]:
        return {
            "pricing_basis": "deepseek_api_standard_via_codex_model_provider",
            "input_price_per_1m": self._input_price_per_1m,
            "cached_input_price_per_1m": self._cached_input_price_per_1m,
            "output_price_per_1m": self._output_price_per_1m,
        }

    def _config_snapshot(self) -> dict[str, Any]:
        snapshot = super()._config_snapshot()
        snapshot.update(
            {
                "provider": "codex_deepseek_extract",
                "model_provider": "deepseek_proxy",
                "deepseek_base_url": self._deepseek_base_url,
                "deepseek_thinking_type": self._deepseek_thinking_type,
                "deepseek_reasoning_effort": self._deepseek_reasoning_effort,
                "deepseek_max_tokens": self._deepseek_max_tokens,
                "deepseek_timeout_s": self._deepseek_timeout_s,
                "pricing_basis": "deepseek_api_standard_via_codex_model_provider",
            }
        )
        return snapshot

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        proxy = _DeepSeekResponsesProxy(
            api_key=self._deepseek_api_key or "",
            base_url=self._deepseek_base_url,
            model=self._model,
            max_tokens=self._deepseek_max_tokens,
            thinking_type=self._deepseek_thinking_type,
            reasoning_effort=self._deepseek_reasoning_effort,
            pricing=(self._input_price_per_1m, self._cached_input_price_per_1m, self._output_price_per_1m),
            timeout_s=self._deepseek_timeout_s,
            display_name="DeepSeek V4 Pro Fireworks Proxy"
            if self._model == _FIREWORKS_DEEPSEEK_V4_PRO_MODEL
            else "DeepSeek Proxy",
            description="DeepSeek model exposed through a Fireworks-backed local Codex Responses proxy."
            if self._uses_fireworks
            else "DeepSeek model exposed through a local Codex Responses proxy.",
        )
        proxy.start()
        try:
            self._thread_state.proxy_url = proxy.url
            raw = super().run_inference(pipeline, request)
            self._last_proxy_usage = proxy.usage.snapshot()
            raw.raw_output["model_provider"] = "deepseek_proxy"
            raw.raw_output["deepseek_model_cost_usd"] = raw.raw_output.get("cost_usd")
            raw.raw_output["_proxy_usage"] = self._last_proxy_usage
            return raw
        finally:
            self._thread_state.proxy_url = None
            self._thread_state.codex_home = None
            proxy.stop()


@register_provider("codex_glm_extract")
class CodexGLMExtractProvider(CodexCodeExtractProvider):
    """Use Codex CLI with a GLM-backed Responses-compatible model provider."""

    DEFAULT_MODEL = _GLM_5_2_MODEL
    DEFAULT_MAX_TOKENS = None

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        cfg = dict(base_config or {})
        self._glm_api_key: str | None = cfg.get("glm_api_key") or cfg.get("api_key")
        if self._glm_api_key is None:
            import os

            self._glm_api_key = os.getenv("GLM_API_KEY")
        if not self._glm_api_key:
            raise ProviderConfigError("GLM API key is required for codex_glm_extract.")

        self._glm_base_url: str = cfg.get("base_url", _GLM_API_BASE_URL)
        self._glm_reasoning_effort: str | None = cfg.get("reasoning_effort")
        self._glm_timeout_s = float(cfg.get("glm_timeout_s", 120.0))
        max_tokens_cfg = cfg.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        self._glm_max_tokens: int | None = int(max_tokens_cfg) if max_tokens_cfg is not None else None
        self._thread_state = threading.local()
        self._last_proxy_usage: dict[str, Any] = {}

        parent_config = dict(cfg)
        parent_config.pop("api_key", None)
        parent_config.pop("glm_api_key", None)
        parent_config["model"] = cfg.get("model", self.DEFAULT_MODEL)
        parent_config["ignore_user_config"] = True
        super().__init__(provider_name, parent_config)

    @staticmethod
    def _pricing_for_model(model: str) -> tuple[float, float, float]:
        matches = [(prefix, rates) for prefix, rates in _GLM_EXTRACT_PRICING_PER_M.items() if model.startswith(prefix)]
        return max(matches, key=lambda item: len(item[0]))[1] if matches else (0.0, 0.0, 0.0)

    def _extra_subprocess_env(self) -> dict[str, str]:
        env = {_GLM_PROXY_ENV_KEY: _GLM_PROXY_BEARER}
        codex_home = getattr(self._thread_state, "codex_home", None)
        if codex_home:
            env["CODEX_HOME"] = codex_home
        return env

    def _build_cmd(
        self,
        *,
        workdir: Path,
        last_message_path: Path,
        image_path: Path | None = None,
    ) -> list[str]:
        proxy_url = getattr(self._thread_state, "proxy_url", None)
        if not proxy_url:
            raise RuntimeError("codex_glm_extract proxy URL was not initialized")
        codex_home = workdir / ".codex-home"
        codex_home.mkdir(parents=True, exist_ok=True)
        self._thread_state.codex_home = str(codex_home)
        cmd = super()._build_cmd(workdir=workdir, last_message_path=last_message_path, image_path=image_path)
        insert_at = cmd.index("--output-last-message")
        proxy_overrides = [
            "-c",
            'model_provider="glm_proxy"',
            "-c",
            'model_providers.glm_proxy.name="GLM Proxy"',
            "-c",
            f'model_providers.glm_proxy.base_url="{proxy_url}"',
            "-c",
            f'model_providers.glm_proxy.env_key="{_GLM_PROXY_ENV_KEY}"',
            "-c",
            'model_providers.glm_proxy.wire_api="responses"',
        ]
        cmd[insert_at:insert_at] = proxy_overrides
        return cmd

    def _pricing_snapshot(self, usage: dict[str, int]) -> dict[str, Any]:
        return {
            "pricing_basis": "glm_5_2_serverless_via_codex_model_provider",
            "input_price_per_1m": self._input_price_per_1m,
            "cached_input_price_per_1m": self._cached_input_price_per_1m,
            "output_price_per_1m": self._output_price_per_1m,
        }

    def _config_snapshot(self) -> dict[str, Any]:
        snapshot = super()._config_snapshot()
        snapshot.update(
            {
                "provider": "codex_glm_extract",
                "model_provider": "glm_proxy",
                "glm_base_url": self._glm_base_url,
                "glm_reasoning_effort": self._glm_reasoning_effort,
                "glm_max_tokens": self._glm_max_tokens,
                "glm_timeout_s": self._glm_timeout_s,
                "pricing_basis": "glm_5_2_serverless_via_codex_model_provider",
            }
        )
        return snapshot

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        proxy = _DeepSeekResponsesProxy(
            api_key=self._glm_api_key or "",
            base_url=self._glm_base_url,
            model=self._model,
            max_tokens=self._glm_max_tokens,
            thinking_type=None,
            reasoning_effort=self._glm_reasoning_effort,
            pricing=(self._input_price_per_1m, self._cached_input_price_per_1m, self._output_price_per_1m),
            timeout_s=self._glm_timeout_s,
            display_name="GLM 5.2 Proxy",
            description="GLM model exposed through a local Codex Responses proxy.",
            context_window=1_040_000,
            comp_hash="glm-5-2-proxy",
        )
        proxy.start()
        try:
            self._thread_state.proxy_url = proxy.url
            raw = super().run_inference(pipeline, request)
            self._last_proxy_usage = proxy.usage.snapshot()
            raw.raw_output["model_provider"] = "glm_proxy"
            raw.raw_output["glm_model_cost_usd"] = raw.raw_output.get("cost_usd")
            raw.raw_output["_proxy_usage"] = self._last_proxy_usage
            return raw
        finally:
            self._thread_state.proxy_url = None
            self._thread_state.codex_home = None
            proxy.stop()
