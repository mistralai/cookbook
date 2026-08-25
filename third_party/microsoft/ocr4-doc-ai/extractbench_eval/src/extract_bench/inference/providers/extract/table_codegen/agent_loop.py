"""Provider-agnostic tool-use agent loop (Anthropic + OpenAI + Google/Gemini).

Lightweight and reusable across the table->schema generation harnesses. A
:class:`Tool` is ``{name, description, input_schema (JSON Schema), handler}``;
``run_agent`` drives a tool-use loop against any of the three providers with one
shared interface. A handler raises :class:`ToolLoopDone` to end the loop early
(e.g. a successful submit), carrying a final payload.

OpenAI uses chat.completions (fully stateless — the full message history is
re-sent each turn), which sidesteps the org's ZDR limitation on
``previous_response_id`` chaining. Anthropic uses the Messages API with the
native tool_use / tool_result blocks. Google/Gemini uses
``models.generate_content`` with genai function-calling (also stateless — the
full ``contents`` history is re-sent each turn), translating each :class:`Tool`'s
JSON-Schema into a ``FunctionDeclaration`` and round-tripping ``function_call`` ->
``function_response`` parts.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# API keys come from the environment (the bench loads .env on CLI startup) or are
# passed explicitly by the provider; the SDK clients read ANTHROPIC_API_KEY /
# OPENAI_API_KEY from os.environ when api_key is None.


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema for the tool's parameters
    handler: Callable[[dict[str, Any]], str]


class ToolLoopDone(Exception):
    """Raised by a handler to stop the loop, carrying a final payload."""

    def __init__(self, payload: Any = None):
        super().__init__("tool loop done")
        self.payload = payload


# Appended to tool results (and sent as the user message when no tool call
# survived) whenever a response was cut off at the output-token limit. The
# model cannot see stop_reason — without an explicit message it just retries
# the same too-long script and the loop burns to max_turns.
_TRUNCATION_WARNING = (
    "WARNING: your response hit the output-token limit and was TRUNCATED — the "
    "tool input may have arrived empty or incomplete. Do NOT retry the same "
    "thing. Write the script more compactly: no prose preamble, no comments, "
    "short variable names, factor repeated field-mapping into loops/dicts."
)

# Two truncations in a row means the model cannot fit its approach in the
# output budget even after being told — bail (status "output_truncated") so a
# retry gets a fresh draw instead of burning the remaining turns.
_MAX_CONSECUTIVE_TRUNCATIONS = 2


def run_agent(
    provider: str,
    model: str,
    system: str,
    tools: list[Tool],
    *,
    max_turns: int = 10,
    max_tokens: int = 16000,
    verbose: bool = False,
    api_key: str | None = None,
    on_turn_start: Callable[[int], None] | None = None,
    trace_result_limit: int | None = 2000,
    thinking_level: str | None = None,
    turn_note: Callable[[int, int], str | None] | None = None,
    effort: str | None = None,
) -> dict[str, Any]:
    """Drive a tool-use loop. ``on_turn_start(turn_index)`` is called at the top
    of each turn before the model is queried — handlers that need to tell turns
    apart (e.g. a gate that must refuse a same-turn run+submit) read the index it
    publishes. ``trace_result_limit`` clips each tool result stored in the trace
    (default 2000 chars); pass ``None`` to capture full results for offline review.
    ``thinking_level`` (Google/Gemini only; ignored by anthropic/openai) sets the
    Gemini thinking effort — ``"low"`` / ``"medium"`` / ``"high"`` / ``"minimal"``;
    ``None`` leaves the model's default dynamic thinking untouched. (gemini-3.x flash
    honors ``thinking_level``, not the 2.5-era ``thinking_budget``.)
    ``turn_note(turn_index, max_turns)`` (optional) returns a short string appended to
    each turn's tool-result message — used to surface a live turn-budget reminder every
    turn; return ``None``/empty to add nothing that turn."""
    tmap = {t.name: t for t in tools}
    if provider == "anthropic":
        return _run_anthropic(
            model,
            system,
            tools,
            tmap,
            max_turns,
            max_tokens,
            verbose,
            api_key,
            on_turn_start,
            trace_result_limit,
            turn_note,
            effort,
        )
    if provider == "openai":
        return _run_openai(
            model, system, tools, tmap, max_turns, verbose, api_key, on_turn_start, trace_result_limit, turn_note
        )
    if provider == "google":
        return _run_google(
            model,
            system,
            tools,
            tmap,
            max_turns,
            max_tokens,
            verbose,
            api_key,
            on_turn_start,
            trace_result_limit,
            thinking_level,
            turn_note,
        )
    raise ValueError(f"unknown provider: {provider!r} (expected 'anthropic', 'openai', or 'google')")


def _call(tmap: dict[str, Tool], name: str, args: dict[str, Any], verbose: bool) -> str:
    if verbose:
        print(f"  · {name}({json.dumps(args)[:140]})")
    if name not in tmap:
        return f"tool error: unknown tool {name!r}"
    return tmap[name].handler(args)  # ToolLoopDone propagates to the loop


def _safe_args(raw: str | None) -> dict[str, Any]:
    """Parse OpenAI tool-call arguments; truncated responses (finish_reason ==
    "length") carry malformed JSON — treat as empty args rather than crashing."""
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _clip(text: str, limit: int | None) -> str:
    """Trim a tool result for the trace. ``limit is None`` keeps the full text —
    used when capturing complete traces for offline review (the run_script output
    the model inspects can be large)."""
    return text if limit is None else text[:limit]


def _run_anthropic(
    model: str,
    system: str,
    tools: list[Tool],
    tmap: dict[str, Tool],
    max_turns: int,
    max_tokens: int,
    verbose: bool,
    api_key: str | None,
    on_turn_start: Callable[[int], None] | None = None,
    trace_result_limit: int | None = 2000,
    turn_note: Callable[[int, int], str | None] | None = None,
    effort: str | None = None,
) -> dict[str, Any]:
    import anthropic
    from anthropic.types import TextBlock, ToolUseBlock

    client = anthropic.Anthropic(api_key=api_key)
    spec: list[dict[str, Any]] = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools
    ]
    # One cache breakpoint on the system prompt (the big document+schema context):
    # turns 2+ read it from cache instead of re-paying full input price.
    system_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Begin."}]
    trace: list[dict[str, Any]] = []
    usage: dict[str, int] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    t0 = time.time()

    def fin(d: dict[str, Any]) -> dict[str, Any]:
        d["usage"], d["latency_s"] = usage, round(time.time() - t0, 2)
        return d

    # 5-gen (sonnet-5 / opus 4.x) + sonnet-4.6 honor output_config.effort (GA, no beta
    # header; default "high"). Omitted when None so existing pipelines keep the default.
    extra: dict[str, Any] = {"output_config": {"effort": effort}} if effort is not None else {}
    truncations = 0  # consecutive max_tokens-truncated responses
    for turn in range(max_turns):
        if on_turn_start is not None:
            on_turn_start(turn)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,  # type: ignore[arg-type]
            tools=spec,  # type: ignore[arg-type]
            messages=messages,  # type: ignore[arg-type]
            **extra,
        )
        u = resp.usage
        usage["input"] += u.input_tokens or 0
        usage["output"] += u.output_tokens or 0
        usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        messages.append({"role": "assistant", "content": resp.content})
        text = "".join(b.text for b in resp.content if isinstance(b, TextBlock))
        tool_uses = [b for b in resp.content if isinstance(b, ToolUseBlock)]
        truncated = resp.stop_reason == "max_tokens"
        truncations = truncations + 1 if truncated else 0
        trace.append(
            {
                "turn": turn,
                "role": "assistant",
                "text": text,
                "stop_reason": resp.stop_reason,
                "tool_calls": [{"name": tu.name, "input": dict(tu.input)} for tu in tool_uses],
            }
        )
        if truncated:
            trace.append(
                {
                    "turn": turn,
                    "role": "system",
                    "note": f"response truncated at max_tokens={max_tokens} — tool input may be incomplete",
                }
            )
        if verbose and text.strip():
            print("  [model]", text.strip()[:240])
        if truncations >= _MAX_CONSECUTIVE_TRUNCATIONS:
            # The model can't fit its approach in the output budget even after
            # an explicit warning — stop here so a retry gets a fresh draw.
            return fin(
                {"status": "output_truncated", "stop_reason": resp.stop_reason, "turns": turn + 1, "trace": trace}
            )
        if not tool_uses:
            if truncated:
                # Cut off before any tool call survived — tell the model and
                # let it try again more compactly.
                messages.append({"role": "user", "content": _TRUNCATION_WARNING})
                continue
            return fin(
                {"status": "stopped_no_tool", "stop_reason": resp.stop_reason, "turns": turn + 1, "trace": trace}
            )
        results: list[dict[str, Any]] = []
        for tu in tool_uses:
            try:
                out = _call(tmap, tu.name, dict(tu.input), verbose)
            except ToolLoopDone as d:
                trace.append({"turn": turn, "role": "tool", "name": tu.name, "result": "<submit accepted — done>"})
                return fin({"status": "done", "payload": d.payload, "turns": turn + 1, "trace": trace})
            if truncated:
                # The model never sees stop_reason — say it in-band, or it
                # will retry the same too-long script until max_turns.
                out = f"{out}\n\n{_TRUNCATION_WARNING}"
            trace.append({"turn": turn, "role": "tool", "name": tu.name, "result": _clip(out, trace_result_limit)})
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out})
        if turn_note is not None and (note := turn_note(turn, max_turns)):
            results.append({"type": "text", "text": note})
        messages.append({"role": "user", "content": results})
    return fin({"status": "max_turns", "turns": max_turns, "trace": trace})


def _run_openai(
    model: str,
    system: str,
    tools: list[Tool],
    tmap: dict[str, Tool],
    max_turns: int,
    verbose: bool,
    api_key: str | None,
    on_turn_start: Callable[[int], None] | None = None,
    trace_result_limit: int | None = 2000,
    turn_note: Callable[[int, int], str | None] | None = None,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    spec: list[dict[str, Any]] = [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
        for t in tools
    ]
    messages: list[Any] = [{"role": "system", "content": system}, {"role": "user", "content": "Begin."}]
    trace: list[dict[str, Any]] = []
    usage: dict[str, int] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    t0 = time.time()

    def fin(d: dict[str, Any]) -> dict[str, Any]:
        d["usage"], d["latency_s"] = usage, round(time.time() - t0, 2)
        return d

    truncations = 0  # consecutive length-truncated responses
    for turn in range(max_turns):
        if on_turn_start is not None:
            on_turn_start(turn)
        resp = client.chat.completions.create(  # type: ignore[call-overload]
            model=model, messages=messages, tools=spec, tool_choice="auto"
        )
        if resp.usage:
            usage["input"] += resp.usage.prompt_tokens or 0
            usage["output"] += resp.usage.completion_tokens or 0
            details = getattr(resp.usage, "prompt_tokens_details", None)
            usage["cache_read"] += getattr(details, "cached_tokens", 0) or 0
        msg = resp.choices[0].message
        messages.append(msg)
        calls = msg.tool_calls or []
        finish_reason = resp.choices[0].finish_reason
        truncated = finish_reason == "length"
        truncations = truncations + 1 if truncated else 0
        trace.append(
            {
                "turn": turn,
                "role": "assistant",
                "text": msg.content or "",
                "stop_reason": finish_reason,
                "tool_calls": [{"name": tc.function.name, "input": _safe_args(tc.function.arguments)} for tc in calls],
            }
        )
        if verbose and msg.content:
            print("  [model]", msg.content.strip()[:240])
        if truncations >= _MAX_CONSECUTIVE_TRUNCATIONS:
            return fin({"status": "output_truncated", "stop_reason": finish_reason, "turns": turn + 1, "trace": trace})
        if not calls:
            if truncated:
                messages.append({"role": "user", "content": _TRUNCATION_WARNING})
                continue
            return fin({"status": "stopped_no_tool", "stop_reason": finish_reason, "turns": turn + 1, "trace": trace})
        for tc in calls:
            args = _safe_args(tc.function.arguments)
            try:
                out = _call(tmap, tc.function.name, args, verbose)
            except ToolLoopDone as d:
                trace.append(
                    {"turn": turn, "role": "tool", "name": tc.function.name, "result": "<submit accepted — done>"}
                )
                return fin({"status": "done", "payload": d.payload, "turns": turn + 1, "trace": trace})
            if truncated:
                out = f"{out}\n\n{_TRUNCATION_WARNING}"
            trace.append(
                {"turn": turn, "role": "tool", "name": tc.function.name, "result": _clip(out, trace_result_limit)}
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
        if turn_note is not None and (note := turn_note(turn, max_turns)):
            messages.append({"role": "user", "content": note})
    return fin({"status": "max_turns", "turns": max_turns, "trace": trace})


def _run_google(
    model: str,
    system: str,
    tools: list[Tool],
    tmap: dict[str, Tool],
    max_turns: int,
    max_tokens: int,
    verbose: bool,
    api_key: str | None,
    on_turn_start: Callable[[int], None] | None = None,
    trace_result_limit: int | None = 2000,
    thinking_level: str | None = None,
    turn_note: Callable[[int, int], str | None] | None = None,
) -> dict[str, Any]:
    """Google/Gemini tool-use loop via genai ``models.generate_content``.

    Stateless like the OpenAI loop: the full ``contents`` history is re-sent each
    turn (so accumulated ``prompt_token_count`` already includes any implicit cache
    hits, billed at the cache-hit rate by the provider's cost branch). Each ``Tool``
    becomes a ``FunctionDeclaration`` (raw JSON-Schema via ``parameters_json_schema``;
    no-arg tools omit parameters), the system prompt rides Gemini's dedicated
    ``system_instruction`` field, and each ``function_call`` part is run through the
    same ``_call`` handler then answered with a ``function_response`` part — keeping
    the trace shape identical to the other two loops so the runner's sidecar writer
    and the salvage path work unchanged."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def _fdecl(t: Tool) -> Any:
        # genai accepts a raw JSON-Schema dict via parameters_json_schema; a no-arg
        # tool (run_script/submit have empty `properties`) omits parameters entirely
        # rather than declare an empty object, which some Gemini versions reject.
        props = (t.input_schema or {}).get("properties") or {}
        if props:
            return types.FunctionDeclaration(
                name=t.name, description=t.description, parameters_json_schema=t.input_schema
            )
        return types.FunctionDeclaration(name=t.name, description=t.description)

    tool_spec = types.Tool(function_declarations=[_fdecl(t) for t in tools])
    gen_config_kwargs: dict[str, Any] = {
        "tools": [tool_spec],
        "system_instruction": system,  # Gemini's dedicated field — NOT a message
        "max_output_tokens": max_tokens,
    }
    if thinking_level is not None:
        # Set the Gemini thinking effort ("low"/"medium"/"high"/"minimal"). Left unset,
        # 3.x flash thinks dynamically (13k-26k tokens/run on these docs). gemini-3.x
        # flash honors thinking_level, NOT the 2.5-era thinking_budget (which it ignores).
        # Plumbed from the provider's `thinking_level` config; coerced to the enum the SDK
        # stub expects (it accepts the string at runtime, but mypy wants ThinkingLevel).
        gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel(thinking_level.upper())
        )
    gen_config = types.GenerateContentConfig(**gen_config_kwargs)
    contents: list[Any] = [types.Content(role="user", parts=[types.Part.from_text(text="Begin.")])]
    trace: list[dict[str, Any]] = []
    # `thinking` tracks thoughts_token_count separately from `output`
    # (candidates_token_count) — Gemini reports the two apart and bills thinking at the
    # output rate, so the provider's cost branch adds it to output (see _codegen_cost).
    usage: dict[str, int] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "thinking": 0}
    t0 = time.time()

    def fin(d: dict[str, Any]) -> dict[str, Any]:
        d["usage"], d["latency_s"] = usage, round(time.time() - t0, 2)
        return d

    truncations = 0  # consecutive MAX_TOKENS-truncated responses
    for turn in range(max_turns):
        if on_turn_start is not None:
            on_turn_start(turn)
        resp = client.models.generate_content(model=model, contents=contents, config=gen_config)
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            # prompt_token_count INCLUDES cached tokens (like OpenAI prompt_tokens);
            # cached_content_token_count is the cached subset. Implicit caching has no
            # write surcharge, so cache_write stays 0.
            usage["input"] += getattr(meta, "prompt_token_count", 0) or 0
            usage["output"] += getattr(meta, "candidates_token_count", 0) or 0
            usage["cache_read"] += getattr(meta, "cached_content_token_count", 0) or 0
            usage["thinking"] += getattr(meta, "thoughts_token_count", 0) or 0
        candidates = getattr(resp, "candidates", None) or []
        candidate = candidates[0] if candidates else None
        content = getattr(candidate, "content", None) if candidate is not None else None
        if content is not None:
            contents.append(content)
        parts = getattr(content, "parts", None) or []
        text = "".join(p.text for p in parts if getattr(p, "text", None))
        fcalls = [p.function_call for p in parts if getattr(p, "function_call", None)]
        finish_reason = getattr(candidate, "finish_reason", None) if candidate is not None else None
        fr_name = getattr(finish_reason, "name", None) or (str(finish_reason) if finish_reason is not None else None)
        truncated = fr_name == "MAX_TOKENS"
        truncations = truncations + 1 if truncated else 0
        trace.append(
            {
                "turn": turn,
                "role": "assistant",
                "text": text,
                "stop_reason": fr_name,
                "tool_calls": [{"name": fc.name, "input": dict(fc.args or {})} for fc in fcalls],
            }
        )
        if verbose and text.strip():
            print("  [model]", text.strip()[:240])
        if truncations >= _MAX_CONSECUTIVE_TRUNCATIONS:
            return fin({"status": "output_truncated", "stop_reason": fr_name, "turns": turn + 1, "trace": trace})
        if not fcalls:
            if truncated:
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=_TRUNCATION_WARNING)]))
                continue
            return fin({"status": "stopped_no_tool", "stop_reason": fr_name, "turns": turn + 1, "trace": trace})
        response_parts: list[Any] = []
        for fc in fcalls:
            try:
                out = _call(tmap, fc.name, dict(fc.args or {}), verbose)
            except ToolLoopDone as d:
                trace.append({"turn": turn, "role": "tool", "name": fc.name, "result": "<submit accepted — done>"})
                return fin({"status": "done", "payload": d.payload, "turns": turn + 1, "trace": trace})
            if truncated:
                out = f"{out}\n\n{_TRUNCATION_WARNING}"
            trace.append({"turn": turn, "role": "tool", "name": fc.name, "result": _clip(out, trace_result_limit)})
            response_parts.append(types.Part.from_function_response(name=fc.name, response={"result": out}))
        contents.append(types.Content(role="user", parts=response_parts))
        if turn_note is not None and (note := turn_note(turn, max_turns)):
            # Gemini rejects a single Content that mixes function_response parts with a text
            # part — send the turn-budget note as its own follow-up user Content instead.
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=note)]))
    return fin({"status": "max_turns", "turns": max_turns, "trace": trace})
