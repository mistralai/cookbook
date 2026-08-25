"""Agentic extraction provider that shells out to the Codex CLI (`codex exec`).

This mirrors the Claude Code extract provider's harness shape: each document is
copied into an isolated temporary workdir, the agent is asked to inspect that
local file with shell tools and write ``./output.json``, and the result is
normalized through the standard Extract output path. Codex differs in one
important way: the CLI does not expose a native dollar budget flag or a terminal
``total_cost_usd`` event, so cost is estimated post-hoc from the
``turn.completed.usage`` token counts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.direct_model_utils import (
    IMAGE_EXTENSIONS,
    add_additional_properties_false,
    force_all_properties_required,
    normalize_extract_result,
    page_count,
    promote_repeated_structure,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

_TRACE_TAIL_EVENTS = 20

_RATE_LIMIT_KEYWORDS = ("rate limit", "rate_limit", "429", "quota")
_TRANSIENT_KEYWORDS = ("overloaded", "high demand", "529", "503", "502", "504", "timeout", "timed out")
_SANDBOX_FAILURE_KEYWORDS = (
    "bwrap: creating new namespace failed",
    "max_user_namespaces",
    "max_mnt_namespaces",
)

_DEFAULT_DISABLED_FEATURES = (
    "plugins",
    "apps",
    "memories",
    "chronicle",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "image_generation",
    "multi_agent",
    "workspace_dependencies",
)

# USD per million tokens (input, cached input, output); estimates use Codex CLI JSONL usage.
_OPENAI_CODEX_PRICING_PER_M: dict[str, tuple[float, float, float]] = {
    "gpt-5.6-sol": (5.00, 0.50, 30.00),
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4": (2.50, 0.25, 15.00),
}

_LONG_CONTEXT_THRESHOLD_TOKENS = 272_000
_LONG_CONTEXT_MODELS = ("gpt-5.6-sol", "gpt-5.5", "gpt-5.4")


@register_provider("codex_code_extract")
class CodexCodeExtractProvider(Provider):
    """Agentic extraction by shelling out to the Codex CLI (`codex exec`)."""

    DEFAULT_MODEL = "gpt-5.4"
    DEFAULT_TIMEOUT_S = 1200
    DEFAULT_MAX_COST_USD = 5.00

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._model: str = self.base_config.get("model", self.DEFAULT_MODEL)
        self._timeout_s: int = int(self.base_config.get("timeout_s", self.DEFAULT_TIMEOUT_S))
        self._max_cost_usd: float | None = self.base_config.get("max_cost_usd", self.DEFAULT_MAX_COST_USD)
        self._codex_bin: str = self.base_config.get("codex_bin", "codex")
        # SECURITY: the CLI runs on the host with `--ask-for-approval never`, so
        # `sandbox` is the only thing between a benchmark document and your machine.
        # `workspace-write` confines writes to the temp workdir. Treat benchmark PDFs
        # as untrusted input — prompt injection can drive the agent.
        self._sandbox: str = self.base_config.get("sandbox", "workspace-write")
        self._reasoning_effort: str | None = self.base_config.get("reasoning_effort", self.base_config.get("effort"))
        self._extra_flags: list[str] = list(self.base_config.get("extra_flags", []))
        self._config_overrides: list[str] = list(self.base_config.get("config_overrides", []))
        self._ignore_user_config: bool = bool(self.base_config.get("ignore_user_config", True))
        self._ignore_rules: bool = bool(self.base_config.get("ignore_rules", True))
        self._disabled_features: list[str] = list(self.base_config.get("disabled_features", _DEFAULT_DISABLED_FEATURES))
        self._attach_images: bool = bool(self.base_config.get("attach_images", True))
        self._require_shell_access: bool = bool(self.base_config.get("require_shell_access", True))

        self._promote_repeated: bool = bool(self.base_config.get("promote_repeated_structure", True))
        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))
        # Match claude_code_extract by default: preserve the benchmark schema's
        # own required fields instead of forcing every nullable/optional field
        # to be required. Direct model providers can still opt into this.
        self._all_properties_required: bool = bool(self.base_config.get("all_properties_required", False))

        self._api_key: str | None = self.base_config.get("api_key") or os.getenv("CODEX_API_KEY")
        self._codex_home: str | None = self.base_config.get("codex_home")

        default_input, default_cached_input, default_output = self._pricing_for_model(self._model)
        self._input_price_per_1m: float = float(self.base_config.get("input_price_per_1m", default_input))
        self._cached_input_price_per_1m: float = float(
            self.base_config.get("cached_input_price_per_1m", default_cached_input)
        )
        self._output_price_per_1m: float = float(self.base_config.get("output_price_per_1m", default_output))

        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._procs_lock = threading.Lock()

    @staticmethod
    def _pricing_for_model(model: str) -> tuple[float, float, float]:
        matches = [(prefix, rates) for prefix, rates in _OPENAI_CODEX_PRICING_PER_M.items() if model.startswith(prefix)]
        return max(matches, key=lambda item: len(item[0]))[1] if matches else (0.0, 0.0, 0.0)

    @staticmethod
    def _uses_long_context_pricing(model: str, usage: dict[str, int]) -> bool:
        """OpenAI applies a full-session uplift above the 272K input-token threshold."""
        if not any(model == prefix or model.startswith(f"{prefix}-") for prefix in _LONG_CONTEXT_MODELS):
            return False
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        return input_tokens > _LONG_CONTEXT_THRESHOLD_TOKENS

    def _prepare_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if self._promote_repeated:
            schema = promote_repeated_structure(schema)
        if self._additional_properties_false:
            schema = add_additional_properties_false(schema)
        if self._all_properties_required:
            schema = force_all_properties_required(schema)
        return schema

    def _build_prompt(self, schema: dict[str, Any], staged_name: str) -> str:
        schema_json = json.dumps(schema, indent=2)
        return (
            f"Extract structured data from the document file `./{staged_name}` in the current directory.\n\n"
            "Use only local file inspection and shell commands. Do not use web search, network calls, "
            "browser tools, or external services. Temporary scratch files inside the current directory are OK.\n\n"
            "Return a single JSON object conforming to this schema as your final answer:\n\n"
            f"```json\n{schema_json}\n```\n\n"
            "Rules:\n"
            "- Use null for fields not present in the document.\n"
            "- For list/array fields, enumerate every relevant row visible in the document; never collapse rows.\n"
            "- For large regular tables, prefer writing and running a local script to parse/enumerate rows.\n"
            "- For forms, prefer direct field extraction from the document content.\n"
            "- Write the resulting JSON object to ./output.json and validate that it is valid JSON before stopping.\n"
            "- Do not print the JSON to your assistant output."
        )

    def _build_cmd(
        self,
        *,
        workdir: Path,
        last_message_path: Path,
        image_path: Path | None = None,
    ) -> list[str]:
        cmd = [
            self._codex_bin,
            "--ask-for-approval",
            "never",
            "--sandbox",
            self._sandbox,
            "--cd",
            str(workdir),
        ]
        for feature in self._disabled_features:
            cmd += ["--disable", feature]

        cmd += [
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--model",
            self._model,
        ]
        if self._ignore_user_config:
            cmd.append("--ignore-user-config")
        if self._ignore_rules:
            cmd.append("--ignore-rules")
        if self._reasoning_effort:
            cmd += ["-c", f'model_reasoning_effort="{self._reasoning_effort}"']
        for override in self._config_overrides:
            cmd += ["-c", override]
        if image_path is not None:
            cmd += ["--image", str(image_path)]
        cmd += ["--output-last-message", str(last_message_path), *self._extra_flags, "-"]
        return cmd

    def _run_cli(
        self,
        cmd: list[str],
        prompt: str,
        workdir: Path,
        example_id: str,
        lines: list[str],
    ) -> tuple[int, str]:
        env = {**os.environ}
        env.update(self._extra_subprocess_env())
        if self._api_key:
            env["CODEX_API_KEY"] = self._api_key
        if self._codex_home:
            env["CODEX_HOME"] = self._codex_home

        stderr_path = workdir / "codex_stderr.log"
        try:
            errf: Any = open(stderr_path, "w", encoding="utf-8")
        except OSError:
            errf = None

        try:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(workdir),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=errf if errf is not None else subprocess.DEVNULL,
                    text=True,
                    env=env,
                )
            except FileNotFoundError as e:
                raise ProviderConfigError(
                    f"Codex CLI not found (tried '{self._codex_bin}'). Install Codex and ensure it is on PATH: {e}"
                ) from e

            with self._procs_lock:
                self._procs[example_id] = proc

            deadline = time.monotonic() + self._timeout_s

            try:
                if proc.stdin is not None:
                    try:
                        proc.stdin.write(prompt)
                        proc.stdin.close()
                    except BrokenPipeError:
                        pass

                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    if time.monotonic() > deadline:
                        self._terminate(proc)
                        raise subprocess.TimeoutExpired(cmd, self._timeout_s)
                    line = raw_line.rstrip("\n")
                    if line:
                        lines.append(line)

                try:
                    remaining = max(1.0, deadline - time.monotonic())
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    self._terminate(proc)
                    raise
            except subprocess.TimeoutExpired as e:
                raise ProviderTransientError(
                    f"codex_code_extract timed out after {self._timeout_s}s; subprocess killed."
                ) from e
            finally:
                with self._procs_lock:
                    self._procs.pop(example_id, None)

            return int(proc.returncode if proc.returncode is not None else 0), self._read_stderr_tail(stderr_path)
        finally:
            if errf is not None:
                errf.close()

    def _extra_subprocess_env(self) -> dict[str, str]:
        return {}

    @staticmethod
    def _read_stderr_tail(path: Path, max_chars: int = 4000) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-max_chars:]

    @staticmethod
    def _terminate(proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    @staticmethod
    def _json_events(lines: list[str]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    @classmethod
    def _usage_from_events(cls, lines: list[str]) -> dict[str, int]:
        usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "total_tokens": 0,
        }
        for event in cls._json_events(lines):
            if event.get("type") != "turn.completed":
                continue
            event_usage = event.get("usage") or {}
            if not isinstance(event_usage, dict):
                continue
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
                value = event_usage.get(key)
                if isinstance(value, int):
                    usage[key] += value
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return usage

    @staticmethod
    def _trace_tail(lines: list[str]) -> list[Any]:
        tail: list[Any] = []
        for line in lines[-_TRACE_TAIL_EVENTS:]:
            try:
                tail.append(json.loads(line))
            except json.JSONDecodeError:
                tail.append(line)
        return tail

    @classmethod
    def _error_messages(cls, lines: list[str]) -> list[str]:
        messages: list[str] = []
        for event in cls._json_events(lines):
            if event.get("type") == "error":
                message = event.get("message")
                if isinstance(message, str) and message:
                    messages.append(message)
                continue
            if event.get("type") == "turn.failed":
                error = event.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
                    if isinstance(message, str) and message:
                        messages.append(message)
                continue
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") != "error":
                continue
            message = item.get("message")
            if isinstance(message, str) and message:
                messages.append(message)
        return messages

    @classmethod
    def _command_execution_diagnostics(cls, lines: list[str]) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "completed": 0,
            "succeeded": 0,
            "failed": 0,
            "failures": [],
        }
        failures: list[dict[str, Any]] = []
        for event in cls._json_events(lines):
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") != "command_execution":
                continue
            diagnostics["completed"] += 1
            exit_code = item.get("exit_code")
            if exit_code == 0:
                diagnostics["succeeded"] += 1
                continue
            diagnostics["failed"] += 1
            output = item.get("aggregated_output")
            failures.append(
                {
                    "command": item.get("command"),
                    "exit_code": exit_code,
                    "status": item.get("status"),
                    "output_tail": output[-1000:] if isinstance(output, str) else None,
                }
            )
        diagnostics["failures"] = failures[:5]
        return diagnostics

    @classmethod
    def _assistant_messages(cls, lines: list[str]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for event in cls._json_events(lines):
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") != "agent_message":
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                messages.append({"role": "assistant", "content": text})
        return messages

    @classmethod
    def _tool_trace_events(cls, lines: list[str]) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for event in cls._json_events(lines):
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "command_execution":
                exit_code = item.get("exit_code")
                tools.append(
                    {
                        "name": "command_execution",
                        "input": {"command": item.get("command")},
                        "output": item.get("aggregated_output"),
                        "metadata": {
                            "item_id": item.get("id"),
                            "status": item.get("status"),
                            "exit_code": exit_code,
                        },
                        "error": (
                            f"command exited {exit_code}" if isinstance(exit_code, int) and exit_code != 0 else None
                        ),
                    }
                )
            elif item_type == "mcp_tool_call":
                status = item.get("status")
                error = item.get("error")
                result = item["result"] if "result" in item else item.get("output")
                tools.append(
                    {
                        "name": str(item.get("tool") or item.get("name") or item_type),
                        "input": item.get("arguments") or item.get("input"),
                        "output": result,
                        "metadata": {
                            "item_id": item.get("id"),
                            "server": item.get("server"),
                            "status": status,
                        },
                        "error": str(error) if error else ("MCP tool call failed" if status == "failed" else None),
                    }
                )
        return tools

    def _raise_for_sandbox_failures(self, lines: list[str]) -> None:
        if not self._require_shell_access:
            return
        diagnostics = self._command_execution_diagnostics(lines)
        if diagnostics["completed"] == 0 or diagnostics["succeeded"] > 0:
            return
        failure_text = "\n".join(
            str(failure.get("output_tail") or "") for failure in diagnostics["failures"] if isinstance(failure, dict)
        ).lower()
        if any(keyword in failure_text for keyword in _SANDBOX_FAILURE_KEYWORDS):
            raise ProviderTransientError(
                "codex_code_extract could not execute local shell commands because the Codex CLI sandbox failed "
                "to create its namespace. Configure this pipeline with sandbox='danger-full-access' on externally "
                "sandboxed benchmark workers."
            )

    def _raise_for_status(self, returncode: int, lines: list[str], stderr_tail: str = "") -> None:
        if returncode == 0:
            return
        messages = self._error_messages(lines)
        message = messages[-1] if messages else stderr_tail
        lowered = message.lower()
        if any(keyword in lowered for keyword in _RATE_LIMIT_KEYWORDS):
            raise ProviderRateLimitError(f"codex_code_extract rate-limited: {message}")
        if any(keyword in lowered for keyword in _TRANSIENT_KEYWORDS):
            raise ProviderTransientError(f"codex_code_extract transient CLI error: {message}")
        raise ProviderPermanentError(f"codex_code_extract CLI exited {returncode}: {message}")

    @staticmethod
    def _read_output_json(output_path: Path) -> dict[str, Any]:
        if not output_path.exists():
            raise ProviderPermanentError("codex_code_extract: CLI completed but final output JSON was not written.")
        try:
            data = json.loads(output_path.read_text())
        except json.JSONDecodeError as e:
            raise ProviderPermanentError(f"codex_code_extract: final output is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ProviderPermanentError(
                f"codex_code_extract: final output must be a JSON object, got {type(data).__name__}."
            )
        return data

    def _failure_debug_payload(
        self,
        *,
        lines: list[str],
        stderr_tail: str,
        last_message_path: Path,
        output_path: Path,
        returncode: int | None,
        cmd: list[str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "returncode": returncode,
            "stderr_tail": stderr_tail,
            "last_message": self._read_stderr_tail(last_message_path),
            "trace_jsonl_lines": lines,
            "trace_tail": self._trace_tail(lines),
            "command_execution": self._command_execution_diagnostics(lines),
            "config": self._config_snapshot(),
            "cmd": cmd,
        }
        if output_path.exists():
            output_text = self._read_stderr_tail(output_path, max_chars=100_000)
            payload["output_text_tail"] = output_text
            try:
                payload["output_json"] = json.loads(output_path.read_text())
            except json.JSONDecodeError:
                pass
        return payload

    def _estimate_cost_usd(self, usage: dict[str, int]) -> float:
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
        input_rate = self._input_price_per_1m
        cached_input_rate = self._cached_input_price_per_1m
        output_rate = self._output_price_per_1m
        if self._uses_long_context_pricing(self._model, usage):
            input_rate *= 2
            cached_input_rate *= 2
            output_rate *= 1.5
        return (
            uncached_input_tokens / 1_000_000 * input_rate
            + cached_input_tokens / 1_000_000 * cached_input_rate
            + output_tokens / 1_000_000 * output_rate
        )

    def _pricing_snapshot(self, usage: dict[str, int]) -> dict[str, Any]:
        long_context = self._uses_long_context_pricing(self._model, usage)
        input_rate = self._input_price_per_1m * (2 if long_context else 1)
        cached_input_rate = self._cached_input_price_per_1m * (2 if long_context else 1)
        output_rate = self._output_price_per_1m * (1.5 if long_context else 1)
        return {
            "pricing_basis": "openai_api_standard",
            "long_context_applied": long_context,
            "long_context_threshold_tokens": _LONG_CONTEXT_THRESHOLD_TOKENS,
            "input_price_per_1m": input_rate,
            "cached_input_price_per_1m": cached_input_rate,
            "output_price_per_1m": output_rate,
        }

    def _config_snapshot(self) -> dict[str, Any]:
        return {
            "model": self._model,
            "timeout_s": self._timeout_s,
            "max_cost_usd": self._max_cost_usd,
            "sandbox": self._sandbox,
            "reasoning_effort": self._reasoning_effort,
            "disabled_features": self._disabled_features,
            "require_shell_access": self._require_shell_access,
            "ignore_user_config": self._ignore_user_config,
            "ignore_rules": self._ignore_rules,
            "attach_images": self._attach_images,
            "promote_repeated_structure": self._promote_repeated,
            "additional_properties_false": self._additional_properties_false,
            "all_properties_required": self._all_properties_required,
            "input_price_per_1m": self._input_price_per_1m,
            "cached_input_price_per_1m": self._cached_input_price_per_1m,
            "output_price_per_1m": self._output_price_per_1m,
            "pricing_basis": "openai_api_standard",
            "long_context_threshold_tokens": _LONG_CONTEXT_THRESHOLD_TOKENS,
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"{type(self).__name__} only supports EXTRACT product type, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override."
            )

        src = Path(request.source_file_path)
        if not src.exists():
            raise ProviderPermanentError(f"File not found: {src}")
        ext = src.suffix.lower()
        if ext != ".pdf" and ext not in IMAGE_EXTENSIONS:
            raise ProviderPermanentError(
                f"{type(self).__name__} supports PDFs and {set(IMAGE_EXTENSIONS)}, got {src.suffix}"
            )

        started_at = datetime.now()
        schema = self._prepare_schema(request.schema_override)
        num_pages = page_count(src)

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            staged_name = f"input{ext}"
            staged_path = workdir / staged_name
            schema_path = workdir / "schema.json"
            output_path = workdir / "output.json"
            last_message_path = workdir / "last_message.txt"
            shutil.copy2(src, staged_path)
            schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

            prompt = self._build_prompt(schema, staged_name)
            image_path = staged_path if self._attach_images and ext in IMAGE_EXTENSIONS else None
            cmd = self._build_cmd(
                workdir=workdir,
                last_message_path=last_message_path,
                image_path=image_path,
            )
            lines: list[str] = []
            returncode: int | None = None
            stderr_tail = ""
            try:
                returncode, stderr_tail = self._run_cli(cmd, prompt, workdir, request.example_id, lines)
                self._raise_for_status(returncode, lines, stderr_tail)
                self._raise_for_sandbox_failures(lines)
                data = self._read_output_json(output_path)
                last_message_tail = self._read_stderr_tail(last_message_path)
            except ProviderError as exc:
                payload = self._failure_debug_payload(
                    lines=lines,
                    stderr_tail=stderr_tail,
                    last_message_path=last_message_path,
                    output_path=output_path,
                    returncode=returncode,
                    cmd=cmd,
                )
                if isinstance(exc.debug_payload, dict):
                    payload.update(exc.debug_payload)
                exc.debug_payload = payload
                raise

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        usage = self._usage_from_events(lines)
        cost_usd = self._estimate_cost_usd(usage)
        raw_output: dict[str, Any] = {
            "data": data,
            "model": self._model,
            "usage": usage,
            "num_pages": num_pages,
            "cost_usd": cost_usd,
            "cost_per_page_usd": (cost_usd / num_pages) if num_pages else None,
            "pricing": self._pricing_snapshot(usage),
            "cost_exceeded_budget": self._max_cost_usd is not None and cost_usd > self._max_cost_usd,
            "_config": self._config_snapshot(),
            "_command_execution": self._command_execution_diagnostics(lines),
            "_last_message_tail": last_message_tail,
            "_trace_tail": self._trace_tail(lines),
            "_codex_artifacts": {
                "output_json": data,
                "trace_jsonl_lines": lines,
                "stderr_tail": stderr_tail,
                "last_message": last_message_tail,
            },
        }

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
        return normalize_extract_result(raw_result)

    def cancel(self, example_id: str) -> bool:
        with self._procs_lock:
            proc = self._procs.get(example_id)
        if proc is None:
            return False
        self._terminate(proc)
        return True
