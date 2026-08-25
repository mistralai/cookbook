"""Agentic extraction provider that shells out to the Claude Code CLI (`claude -p`).

Mirrors the pure-``claude -p`` (no LlamaExtract skill) condition from
``agent_extract_experiments``: the document is staged in a temp workdir, Claude is
prompted to read it agentically and write ``./output.json`` against the benchmark
schema, and the extraction + cost are read back from the ``stream-json`` trace.

The CLI's terminal ``{"type":"result"}`` event carries ``total_cost_usd``, so no
pricing table is needed. Per-document spend is bounded by three layered controls,
all driven by ``max_cost_usd`` / ``timeout_s``:

1. ``--max-budget-usd`` — native CLI hard dollar cap (the deterministic backstop;
   the CLI itself stops spending past it). Passed only when ``max_cost_usd`` is set.
2. ``timeout_s`` — wall-clock kill enforced by the provider.
3. streaming ``max_cost_usd`` guard — the provider reads the ``stream-json`` stdout
   incrementally and terminates the subprocess the instant accumulated interim cost
   crosses the cap (belt-and-suspenders over (1), plus ``cost_exceeded_budget``
   reporting on the terminal event).

Note: CLI v2.1.x dropped ``--max-turns``; ``--max-budget-usd`` is the supported
spend backstop, so the loop is bounded by dollars rather than turn count.
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
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.direct_model_utils import (
    IMAGE_EXTENSIONS,
    add_additional_properties_false,
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

# Number of trailing stream-json events to keep in raw_output for debugging.
_TRACE_TAIL_EVENTS = 20

# Cap each logged Bash command so heredoc-embedded scripts / JSON dumps don't bloat
# raw.json (we log what the agent *runs*, not the full bodies).
_MAX_BASH_CMD_CHARS = 2000

# Substrings (in the result event's subtype/message) that classify a CLI error.
_RATE_LIMIT_KEYWORDS = ("rate limit", "rate_limit", "429", "quota")
_TRANSIENT_KEYWORDS = ("overloaded", "529", "503", "502", "504", "timeout", "timed out")


@register_provider("claude_code_extract")
class ClaudeCodeExtractProvider(Provider):
    """Agentic extraction by shelling out to the Claude Code CLI (`claude -p`).

    Stages the document in a temp workdir, prompts the agent to read it and write
    ./output.json against the benchmark schema, and reads the extraction + cost
    back from the stream-json trace. Mirrors the pure-`claude -p` (no LlamaExtract
    skill) condition from agent_extract_experiments.
    """

    DEFAULT_MODEL = "claude-opus-4-8"
    DEFAULT_TIMEOUT_S = 1200  # 20 min/doc; overridable via config["timeout_s"]

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._model: str = self.base_config.get("model", self.DEFAULT_MODEL)
        self._timeout_s: int = int(self.base_config.get("timeout_s", self.DEFAULT_TIMEOUT_S))
        # Cost controls (see module docstring):
        self._max_cost_usd: float | None = self.base_config.get("max_cost_usd")
        self._claude_bin: str = self.base_config.get("claude_bin", "claude")
        self._extra_flags: list[str] = list(self.base_config.get("extra_flags", []))
        # --bare: minimal mode — skip hooks/plugins/auto-memory and CLAUDE.md
        # auto-discovery for a clean, reproducible benchmark run (the faithful
        # `no_skill` flag set). Auth becomes strictly ANTHROPIC_API_KEY/apiKeyHelper.
        self._bare: bool = bool(self.base_config.get("bare", False))
        # --effort: thinking/effort level (low|medium|high|xhigh|max). Sonnet runs
        # unbounded extended thinking at the CLI default and never converges on dense
        # docs within the wall-clock timeout; "low" makes it converge. None → omit
        # the flag (use the CLI default, which is what Opus runs at).
        self._effort: str | None = self.base_config.get("effort")
        # Schema-prep toggles (reuse direct_model_utils), default to the
        # anthropic_direct-proven combination.
        self._promote_repeated: bool = bool(self.base_config.get("promote_repeated_structure", True))
        self._additional_properties_false: bool = bool(self.base_config.get("additional_properties_false", True))
        # Track live subprocesses for cancel(): example_id -> Popen
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._procs_lock = threading.Lock()
        # API key is optional here: claude may be authed via a logged-in session.
        self._api_key = self.base_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")

    # ------------------------------------------------------------------
    # Schema + prompt + command construction
    # ------------------------------------------------------------------
    def _prepare_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        if self._promote_repeated:
            schema = promote_repeated_structure(schema)
        if self._additional_properties_false:
            schema = add_additional_properties_false(schema)
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

    def _build_cmd(self, prompt: str, mcp_config_path: Path) -> list[str]:
        # ---------------------------------------------------------------
        # SECURITY: this runs UNSANDBOXED, by design. `bypassPermissions`
        # lets the agent execute shell commands on the host without
        # prompting; `--bare` is a reproducibility flag (no hooks/plugins/
        # CLAUDE.md), NOT a sandbox. Restricting permissions here has been
        # left alone deliberately: the agent needs free rein to read and
        # convert the document to reach paper-comparable scores, and a
        # half-configured sandbox silently degrades the result instead of
        # failing loudly.
        #
        # Consequences for anyone running this pipeline:
        #   - benchmark PDFs are untrusted input; a prompt-injection payload
        #     in a document can drive the agent's shell
        #   - the workdir is a temp dir, but the agent is not confined to it
        #   - WebFetch/WebSearch are disabled below, which is NOT egress
        #     control — the agent can still reach the network via Bash
        # Run it in a container/VM if that matters to you. Using it as-is is
        # an accepted, at-your-own-risk trade for benchmark fidelity.
        # ---------------------------------------------------------------
        cmd = [
            self._claude_bin,
            "--print",
            *(["--bare"] if self._bare else []),
            "--model",
            self._model,
            "--permission-mode",
            "bypassPermissions",
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--mcp-config",
            str(mcp_config_path),
            "--disallowedTools",
            "WebFetch WebSearch",
        ]
        # Bound thinking/effort when configured (Sonnet needs "low" to converge).
        if self._effort:
            cmd += ["--effort", str(self._effort)]
        # Native CLI hard dollar cap (deterministic spend backstop). CLI v2.1.x
        # has no --max-turns, so the loop is bounded by dollars instead.
        if self._max_cost_usd is not None:
            cmd += ["--max-budget-usd", str(self._max_cost_usd)]
        # Terminate option parsing before the positional prompt. Claude CLI
        # options such as --disallowedTools accept a variadic value list and
        # otherwise consume the prompt when no later option follows them.
        cmd += [*self._extra_flags, "--", prompt]
        return cmd

    # ------------------------------------------------------------------
    # Subprocess execution + streaming budget guard
    # ------------------------------------------------------------------
    def _run_cli(self, cmd: list[str], workdir: Path, example_id: str, lines: list[str]) -> tuple[int, str]:
        """Run the CLI, streaming stdout and enforcing the cost/time budgets.

        Appends stream-json stdout lines into the caller-owned ``lines`` buffer
        (so a partial trace survives timeout/budget kills) and returns
        ``(returncode, stderr_tail)``. Raises:
        - ``ProviderConfigError`` if the ``claude`` binary is missing.
        - ``ProviderTransientError`` on wall-clock timeout.
        - ``ProviderPermanentError`` if the streaming cost crosses ``max_cost_usd``.
        """
        env = {**os.environ}
        if self._api_key:
            env["ANTHROPIC_API_KEY"] = self._api_key

        # stderr -> file sink, never a PIPE. With --verbose always on, the CLI can
        # emit more diagnostics than the OS pipe buffer (~64 KB) holds; an undrained
        # stderr PIPE would block the child mid-write, stall our stdout loop, and
        # hang until the deadline. A file has no such limit. Mirrors the local
        # llamaparse/llamaextract subprocess providers. We keep the tail for debugging.
        stderr_path = workdir / "claude_stderr.log"
        try:
            errf: Any = open(stderr_path, "w", encoding="utf-8")
        except OSError:
            errf = None

        try:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(workdir),
                    stdout=subprocess.PIPE,
                    stderr=errf if errf is not None else subprocess.DEVNULL,
                    text=True,
                    env=env,
                )
            except FileNotFoundError as e:
                raise ProviderConfigError(
                    f"claude CLI not found (tried '{self._claude_bin}'). Install "
                    "@anthropic-ai/claude-code and ensure it is on PATH: {e}".format(e=e)
                ) from e

            with self._procs_lock:
                self._procs[example_id] = proc

            running_cost: float | None = None
            deadline = time.monotonic() + self._timeout_s

            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    if time.monotonic() > deadline:
                        self._terminate(proc)
                        raise subprocess.TimeoutExpired(cmd, self._timeout_s)

                    line = raw_line.rstrip("\n")
                    if not line:
                        continue
                    lines.append(line)

                    event_cost, event_type = self._event_cost_and_type(line)
                    if event_cost is not None:
                        running_cost = event_cost if running_cost is None else max(running_cost, event_cost)
                        # Only the streaming (interim) guard kills mid-run. The terminal
                        # ``result`` event is post-hoc — by then the spend is already
                        # incurred, so it is reported via ``cost_exceeded_budget`` instead
                        # of triggering a pointless kill.
                        if (
                            event_type != "result"
                            and self._max_cost_usd is not None
                            and running_cost > self._max_cost_usd
                        ):
                            self._terminate(proc)
                            raise ProviderPermanentError(
                                "claude_code_extract exceeded max_cost_usd budget "
                                f"${self._max_cost_usd:.4f} (spent ${running_cost:.4f}); "
                                "subprocess terminated mid-run. Not retriable."
                            )

                # Stream drained naturally; reap the process and surface a hang as a timeout.
                try:
                    remaining = max(1.0, deadline - time.monotonic())
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    self._terminate(proc)
                    raise
            except subprocess.TimeoutExpired as e:
                raise ProviderTransientError(
                    f"claude_code_extract timed out after {self._timeout_s}s; subprocess killed."
                ) from e
            finally:
                with self._procs_lock:
                    self._procs.pop(example_id, None)

            return int(proc.returncode or 0), self._read_stderr_tail(stderr_path)
        finally:
            if errf is not None:
                errf.close()

    @staticmethod
    def _read_stderr_tail(path: Path, max_chars: int = 4000) -> str:
        """Return the last ``max_chars`` of the captured stderr log (for debugging)."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-max_chars:]

    @staticmethod
    def _terminate(proc: subprocess.Popen[str]) -> None:
        """Terminate a subprocess, escalating to kill after a short grace period."""
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    @staticmethod
    def _event_cost_and_type(line: str) -> tuple[float | None, str | None]:
        """Return ``(total_cost_usd, event_type)`` for a stream-json line.

        ``total_cost_usd`` is the CLI's cumulative running cost; either field is
        ``None`` when absent or the line is not a JSON object.
        """
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None, None
        if not isinstance(event, dict):
            return None, None
        event_type = event.get("type")
        event_type = event_type if isinstance(event_type, str) else None
        cost = event.get("total_cost_usd")
        if isinstance(cost, (int, float)):
            return float(cost), event_type
        return None, event_type

    @staticmethod
    def _parse_result_event(lines: list[str]) -> dict[str, Any] | None:
        """Return the last terminal ``{"type":"result"}`` event from the stream."""
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                return event
        return None

    @staticmethod
    def _trace_tail(lines: list[str]) -> list[Any]:
        """Return the last few parsed events (or raw lines) for debugging."""
        tail: list[Any] = []
        for line in lines[-_TRACE_TAIL_EVENTS:]:
            try:
                tail.append(json.loads(line))
            except json.JSONDecodeError:
                tail.append(line)
        return tail

    @staticmethod
    def _tool_usage(lines: list[str]) -> dict[str, Any]:
        """Distill the agent's tool CALLS (never outputs) from the stream-json trace.

        Returns ``tool_counts`` (per tool-name totals across all tools) and
        ``bash_commands`` (every Bash command string, in order, each truncated to
        ``_MAX_BASH_CMD_CHARS``). Only ``tool_use`` blocks are read — tool *results*
        (file dumps, command stdout) are deliberately never logged, and long
        heredoc commands are capped, so this stays small. Lands in ``raw_output`` so
        it persists alongside ``<id>.raw.json``: an auditable record of what the
        agent ran on each document, without the bulky outputs.
        """
        tool_counts: dict[str, int] = {}
        bash_commands: list[str] = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "assistant":
                continue
            for block in (event.get("message") or {}).get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = str(block.get("name") or "")
                tool_counts[name] = tool_counts.get(name, 0) + 1
                if name == "Bash":
                    command = str((block.get("input") or {}).get("command") or "")
                    if len(command) > _MAX_BASH_CMD_CHARS:
                        command = command[:_MAX_BASH_CMD_CHARS] + f"...[+{len(command) - _MAX_BASH_CMD_CHARS} chars]"
                    bash_commands.append(command)
        return {"tool_counts": tool_counts, "bash_commands": bash_commands}

    def _raise_for_status(
        self,
        returncode: int,
        result_event: dict[str, Any] | None,
        lines: list[str],
        stderr_tail: str = "",
    ) -> None:
        """Classify CLI completion into the provider error hierarchy."""
        stderr_suffix = f" stderr: {stderr_tail}" if stderr_tail.strip() else ""
        if result_event is None:
            tail = self._trace_tail(lines)
            raise ProviderPermanentError(
                f"claude_code_extract: no terminal 'result' event in CLI output "
                f"(exit={returncode}). Trace tail: {tail}.{stderr_suffix}"
            )

        if result_event.get("is_error"):
            subtype = str(result_event.get("subtype") or "")
            message = str(result_event.get("result") or result_event.get("error") or "")
            lowered = f"{subtype} {message}".lower()
            if any(keyword in lowered for keyword in _RATE_LIMIT_KEYWORDS):
                raise ProviderRateLimitError(f"claude_code_extract rate-limited: {subtype or message}")
            if any(keyword in lowered for keyword in _TRANSIENT_KEYWORDS):
                raise ProviderTransientError(f"claude_code_extract transient CLI error: {subtype or message}")
            raise ProviderPermanentError(f"claude_code_extract CLI error: {subtype or message}")

        if returncode != 0:
            raise ProviderPermanentError(
                f"claude_code_extract: CLI exited {returncode} without an error result event.{stderr_suffix}"
            )

    @staticmethod
    def _read_output_json(workdir: Path) -> dict[str, Any]:
        output_path = workdir / "output.json"
        if not output_path.exists():
            raise ProviderPermanentError("claude_code_extract: CLI completed but ./output.json was not written.")
        try:
            data = json.loads(output_path.read_text())
        except json.JSONDecodeError as e:
            raise ProviderPermanentError(f"claude_code_extract: ./output.json is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ProviderPermanentError(
                f"claude_code_extract: ./output.json must be a JSON object, got {type(data).__name__}."
            )
        return data

    def _config_snapshot(self) -> dict[str, Any]:
        return {
            "model": self._model,
            "timeout_s": self._timeout_s,
            "max_cost_usd": self._max_cost_usd,
            "bare": self._bare,
            "effort": self._effort,
            "promote_repeated_structure": self._promote_repeated,
            "additional_properties_false": self._additional_properties_false,
        }

    # ------------------------------------------------------------------
    # Provider API
    # ------------------------------------------------------------------
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
            shutil.copy2(src, workdir / staged_name)
            mcp_config = workdir / "mcp_config.json"
            mcp_config.write_text('{"mcpServers": {}}')

            prompt = self._build_prompt(schema, staged_name)
            cmd = self._build_cmd(prompt, mcp_config)
            lines: list[str] = []
            returncode, stderr_tail = self._run_cli(cmd, workdir, request.example_id, lines)
            result_event = self._parse_result_event(lines)
            self._raise_for_status(returncode, result_event, lines, stderr_tail)
            data = self._read_output_json(workdir)

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        usage = (result_event or {}).get("usage", {}) or {}
        cost_usd = (result_event or {}).get("total_cost_usd")
        raw_output: dict[str, Any] = {
            "data": data,
            "model": self._model,
            "usage": usage,
            "num_pages": num_pages,
            "cost_usd": cost_usd,
            "cost_per_page_usd": (cost_usd / num_pages) if (cost_usd and num_pages) else None,
            "cost_exceeded_budget": (
                self._max_cost_usd is not None and cost_usd is not None and cost_usd > self._max_cost_usd
            ),
            "_config": self._config_snapshot(),
            # Auditable record of what the agent actually ran (tools + every Bash
            # command), persisted with the raw.json.
            **self._tool_usage(lines),
            "_trace_tail": self._trace_tail(lines),
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
