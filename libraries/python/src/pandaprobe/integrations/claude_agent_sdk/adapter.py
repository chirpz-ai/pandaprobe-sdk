"""Claude Agent SDK integration adapter for PandaProbe tracing.

Instruments the Claude Agent SDK by monkey-patching ``ClaudeSDKClient`` methods
via ``wrapt`` and injecting tool-tracing hooks.  Each ``receive_response``
invocation produces a full ``TraceData`` with nested ``SpanData`` objects for
the agent conversation, LLM calls, and tool executions.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pandaprobe.integrations._base import BaseIntegrationAdapter
from pandaprobe.integrations.claude_agent_sdk.utils import (
    extract_model_parameters,
    extract_thinking_from_content,
    extract_token_usage,
    flatten_content_blocks,
    normalize_content_to_text,
    safe_serialize,
    serialize_tool_response,
    strip_thinking_blocks,
)
from pandaprobe.schemas import SpanData, SpanKind, SpanStatusCode, TraceData, TraceStatus
from pandaprobe.tracing.session import get_current_session_id, get_current_user_id
from pandaprobe.validation import extract_last_user_message

logger = logging.getLogger("pandaprobe")

_patched = False

# ---------------------------------------------------------------------------
# Per-invocation trace state
# ---------------------------------------------------------------------------

_current_trace_state: ContextVar[_TraceState | None] = ContextVar("_pandaprobe_claude_trace_state", default=None)
_current_span_id: ContextVar[str | None] = ContextVar("_pandaprobe_claude_current_span", default=None)


@dataclass
class _TraceState:
    """Per-invocation state accumulated during a single receive_response call."""

    adapter: ClaudeAgentSDKAdapter
    spans: dict[str, SpanData] = field(default_factory=dict)
    root_span_id: str | None = None
    agent_span_id: str | None = None
    current_llm_span_id: str | None = None
    trace_input: Any = None
    trace_output: Any = None
    chain_output: Any = None
    trace_started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    collected_messages: list[dict[str, Any]] = field(default_factory=list)
    subagent_spans: dict[str, str] = field(default_factory=dict)
    tool_spans: dict[str, str] = field(default_factory=dict)
    pending_thinking: str | None = None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ClaudeAgentSDKAdapter(BaseIntegrationAdapter):
    """PandaProbe integration adapter for the Claude Agent SDK.

    Usage::

        from pandaprobe.integrations.claude_agent_sdk import ClaudeAgentSDKAdapter

        adapter = ClaudeAgentSDKAdapter(tags=["example"])
        adapter.instrument()

        # All subsequent ClaudeSDKClient interactions are now traced automatically.
    """

    def instrument(self) -> bool:
        """Apply monkey-patches to Claude Agent SDK internals.

        Returns ``True`` if patching succeeded, ``False`` if dependencies are
        missing.  The method is idempotent -- calling it multiple times is safe.
        """
        global _patched

        _store_adapter(self)

        if _patched:
            return True

        try:
            import claude_agent_sdk  # noqa: F401
            from wrapt import wrap_function_wrapper  # noqa: F401
        except ImportError as exc:
            logger.warning("PandaProbe Claude Agent SDK integration: missing dependency — %s", exc)
            return False

        # Required wrap: receive_response is the main trace entry point
        try:
            wrap_function_wrapper("claude_agent_sdk", "ClaudeSDKClient.receive_response", _wrap_receive_response)
        except Exception as exc:
            logger.error("PandaProbe Claude Agent SDK: failed to wrap receive_response — %s", exc)
            return False

        # Optional wraps -- fail gracefully
        optional_wraps = [
            ("claude_agent_sdk", "ClaudeSDKClient.query", _wrap_client_query),
            ("claude_agent_sdk", "ClaudeSDKClient.__init__", _wrap_client_init),
            ("claude_agent_sdk", "query", _wrap_standalone_query),
        ]
        for module, name, wrapper in optional_wraps:
            try:
                wrap_function_wrapper(module, name, wrapper)
            except Exception as exc:
                logger.warning("PandaProbe: failed to wrap %s — %s", name, exc)

        _patched = True
        return True

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize_trace(self, state: _TraceState, *, error: bool = False) -> None:
        """Assemble a TraceData from collected spans and submit it."""
        try:
            client = self._resolve_client()
            spans = list(state.spans.values())
            session_id = self._session_id if self._session_id is not None else get_current_session_id()
            user_id = self._user_id if self._user_id is not None else get_current_user_id()

            trace = TraceData(
                name="ClaudeAgentSDK",
                status=TraceStatus.ERROR if error else TraceStatus.COMPLETED,
                input=state.trace_input,
                output=state.trace_output,
                metadata=dict(self._metadata),
                started_at=state.trace_started_at,
                ended_at=datetime.now(timezone.utc),
                session_id=session_id,
                user_id=user_id,
                tags=list(self._tags),
                spans=spans,
            )
            client.log_trace(trace)
        except Exception as exc:
            logger.error("PandaProbe Claude Agent SDK adapter failed to submit trace: %s", exc)


# ---------------------------------------------------------------------------
# Module-level adapter storage
# ---------------------------------------------------------------------------

_active_adapter: ClaudeAgentSDKAdapter | None = None

# Module-level trace state fallback for hooks (contextvars may not propagate
# across anyio threads used by the Claude Agent SDK subprocess).
_active_trace_state: _TraceState | None = None


def _store_adapter(adapter: ClaudeAgentSDKAdapter) -> None:
    global _active_adapter
    _active_adapter = adapter


def _get_adapter() -> ClaudeAgentSDKAdapter | None:
    return _active_adapter


def _set_active_state(state: _TraceState | None) -> None:
    global _active_trace_state
    _active_trace_state = state


def _get_active_state() -> _TraceState | None:
    return _current_trace_state.get(None) or _active_trace_state


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def _set_trace_state(state: _TraceState) -> Token[_TraceState | None]:
    return _current_trace_state.set(state)


def _get_trace_state() -> _TraceState | None:
    return _current_trace_state.get(None)


def _set_current_span(span_id: str | None) -> Token[str | None]:
    return _current_span_id.set(span_id)


def _get_current_span() -> str | None:
    return _current_span_id.get(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_thinking_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of messages with thinking blocks stripped from all assistant content."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content")
            stripped = strip_thinking_blocks(content) if isinstance(content, list) else content
            new_msg = dict(msg)
            new_msg["content"] = stripped
            result.append(new_msg)
        else:
            result.append(msg)
    return result


def _is_thinking_only(content: Any) -> bool:
    """Return True if content consists entirely of thinking blocks."""
    if not isinstance(content, list):
        return False
    if not content:
        return False
    for block in content:
        block_type = type(block).__name__
        if isinstance(block, dict):
            if block.get("type") != "thinking":
                return False
        elif block_type == "ThinkingBlock":
            continue
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# Wrapper: ClaudeSDKClient.__init__  →  inject tool hooks + init history
# ---------------------------------------------------------------------------


def _wrap_client_init(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> None:
    wrapped(*args, **kwargs)
    instance._pandaprobe_history = []
    _inject_tracing_hooks(instance)


def _inject_tracing_hooks(client: Any) -> None:
    """Inject PandaProbe tool-tracing hooks into the client's options."""
    options = getattr(client, "options", None)
    if options is None:
        return

    if not hasattr(options, "hooks"):
        return

    if options.hooks is None:
        options.hooks = {}

    for event in ("PreToolUse", "PostToolUse", "PostToolUseFailure"):
        if event not in options.hooks:
            options.hooks[event] = []

    try:
        from claude_agent_sdk import HookMatcher  # type: ignore[import-not-found]

        options.hooks["PreToolUse"].insert(0, HookMatcher(matcher=None, hooks=[_pre_tool_use_hook]))
        options.hooks["PostToolUse"].insert(0, HookMatcher(matcher=None, hooks=[_post_tool_use_hook]))
        options.hooks["PostToolUseFailure"].insert(0, HookMatcher(matcher=None, hooks=[_post_tool_use_failure_hook]))
    except ImportError:
        logger.warning("PandaProbe: failed to import HookMatcher from claude_agent_sdk")
    except Exception as exc:
        logger.warning("PandaProbe: failed to inject tracing hooks — %s", exc)


# ---------------------------------------------------------------------------
# Wrapper: ClaudeSDKClient.query  →  capture prompt & start time
# ---------------------------------------------------------------------------


async def _wrap_client_query(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    prompt = args[0] if args else kwargs.get("prompt")

    if isinstance(prompt, str):
        instance._pandaprobe_prompt = prompt
    else:
        instance._pandaprobe_prompt = None

    instance._pandaprobe_start_time = datetime.now(timezone.utc)

    return await wrapped(*args, **kwargs)


# ---------------------------------------------------------------------------
# Wrapper: ClaudeSDKClient.receive_response  →  root CHAIN/AGENT spans + trace lifecycle
# ---------------------------------------------------------------------------


async def _wrap_receive_response(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    adapter = _get_adapter()
    if adapter is None:
        async for msg in wrapped(*args, **kwargs):
            yield msg
        return

    state = _TraceState(adapter=adapter)
    state_token = _set_trace_state(state)
    _set_active_state(state)

    prompt = getattr(instance, "_pandaprobe_prompt", None)
    start_time = getattr(instance, "_pandaprobe_start_time", None) or state.trace_started_at
    state.trace_started_at = start_time

    options = getattr(instance, "options", None)
    system_prompt = None
    model_name = None
    model_params = None
    if options:
        sp = getattr(options, "system_prompt", None)
        if isinstance(sp, str):
            system_prompt = sp
        elif isinstance(sp, dict) and sp.get("type") == "preset":
            system_prompt = f"preset: {sp.get('preset', 'claude_code')}"
            if "append" in sp:
                system_prompt += f"\n{sp['append']}"
        model_name = getattr(options, "model", None)
        model_params = extract_model_parameters(options)

    # -- Build full conversation history from client state ---------------------
    history: list[dict[str, Any]] = getattr(instance, "_pandaprobe_history", None) or []

    # Seed collected_messages with system prompt + prior history + current user message
    initial_messages: list[dict[str, Any]] = []
    if system_prompt:
        initial_messages.append({"role": "system", "content": system_prompt})
    initial_messages.extend(history)
    if isinstance(prompt, str) and prompt:
        initial_messages.append({"role": "user", "content": prompt})

    state.trace_input = extract_last_user_message({"messages": list(initial_messages)}) if initial_messages else None
    state.collected_messages = list(initial_messages)

    # -- CHAIN span (root) -----------------------------------------------------
    chain_span_id = str(uuid4())
    state.root_span_id = chain_span_id
    chain_span = SpanData(
        span_id=chain_span_id,
        parent_span_id=None,
        name="ClaudeAgentSDK",
        kind=SpanKind.CHAIN,
        input={"messages": _strip_thinking_from_messages(initial_messages)} if initial_messages else state.trace_input,
        started_at=start_time,
    )
    state.spans[chain_span_id] = chain_span

    # -- AGENT span (child of CHAIN) -------------------------------------------
    agent_span_id = str(uuid4())
    state.agent_span_id = agent_span_id
    parent_span_token = _set_current_span(agent_span_id)

    agent_span = SpanData(
        span_id=agent_span_id,
        parent_span_id=chain_span_id,
        name="ClaudeAgentSDK",
        kind=SpanKind.AGENT,
        input={"messages": _strip_thinking_from_messages(initial_messages)} if initial_messages else None,
        model=model_name,
        model_parameters=model_params,
        started_at=start_time,
    )
    state.spans[agent_span_id] = agent_span

    llm_start_time = start_time
    has_error = False

    try:
        async for msg in wrapped(*args, **kwargs):
            msg_type = type(msg).__name__

            if msg_type == "AssistantMessage":
                _handle_assistant_message(msg, state, model_name, model_params, llm_start_time)

            elif msg_type == "UserMessage":
                llm_start_time = _handle_user_message(msg, state)

            elif msg_type == "ResultMessage":
                _handle_result_message(msg, state)

            yield msg

    except Exception as exc:
        has_error = True
        chain_span.error = str(exc)
        chain_span.status = SpanStatusCode.ERROR
        agent_span.error = str(exc)
        agent_span.status = SpanStatusCode.ERROR
        raise

    finally:
        now = datetime.now(timezone.utc)

        # CHAIN output = full conversation with thinking stripped
        if state.collected_messages:
            chain_span.output = {"messages": _strip_thinking_from_messages(state.collected_messages)}

        # AGENT output = only the last assistant message (with thinking stripped)
        last_assistant = None
        for m in reversed(state.collected_messages):
            if m.get("role") == "assistant":
                stripped_content = (
                    strip_thinking_blocks(m.get("content")) if isinstance(m.get("content"), list) else m.get("content")
                )
                last_assistant = {"role": "assistant", "content": stripped_content}
                break
        if last_assistant:
            agent_span.output = {"messages": [last_assistant]}
            state.trace_output = {"messages": [last_assistant]}

        if not has_error:
            agent_span.status = SpanStatusCode.OK
            chain_span.status = SpanStatusCode.OK
        agent_span.ended_at = now
        chain_span.ended_at = now

        # Update persistent conversation history on the client instance
        # (exclude system prompt — it's re-added each time from options)
        new_messages_for_history = [m for m in state.collected_messages if m.get("role") != "system"]
        # Only keep messages added in THIS turn (after the history we already had)
        history_len = len(history)
        new_turn_messages = new_messages_for_history[history_len:]
        if hasattr(instance, "_pandaprobe_history"):
            instance._pandaprobe_history.extend(new_turn_messages)
        else:
            instance._pandaprobe_history = new_turn_messages

        adapter._finalize_trace(state, error=has_error)

        _current_trace_state.reset(state_token)
        _current_span_id.reset(parent_span_token)
        _set_active_state(None)


# ---------------------------------------------------------------------------
# Message handlers (called from _wrap_receive_response)
# ---------------------------------------------------------------------------


def _handle_assistant_message(
    msg: Any,
    state: _TraceState,
    model_name: str | None,
    model_params: dict[str, Any] | None,
    llm_start_time: datetime,
) -> None:
    """Process an AssistantMessage.

    The Claude SDK with thinking enabled sends TWO AssistantMessages per turn:
    1. A thinking-only message (only ThinkingBlock content)
    2. The actual text response (TextBlock + optionally ToolUseBlock)

    We buffer the thinking from (1) and only create an LLM span when (2)
    arrives, attaching the buffered thinking as ``reasoning_summary``.
    """
    parent_tool_use_id = getattr(msg, "parent_tool_use_id", None)
    model = getattr(msg, "model", None) or model_name
    content = getattr(msg, "content", [])
    error = getattr(msg, "error", None)

    # Flatten content blocks to dicts for processing
    flattened = flatten_content_blocks(content)

    # Check if this is a thinking-only message — if so, buffer it and skip span creation
    if _is_thinking_only(content) or _is_thinking_only(flattened):
        thinking = extract_thinking_from_content(content) or extract_thinking_from_content(flattened)
        if thinking:
            state.pending_thinking = thinking
        return

    # Determine parent: subagent AGENT span or main AGENT span
    if parent_tool_use_id and parent_tool_use_id in state.subagent_spans:
        parent_id = state.subagent_spans[parent_tool_use_id]
    else:
        parent_id = state.agent_span_id

    # Extract thinking from this message (if any) + combine with pending
    thinking = extract_thinking_from_content(content) or extract_thinking_from_content(flattened)
    if state.pending_thinking:
        if thinking:
            thinking = state.pending_thinking + "\n\n" + thinking
        else:
            thinking = state.pending_thinking
        state.pending_thinking = None

    # Strip thinking from visible content
    stripped = strip_thinking_blocks(flattened)

    # Build assistant message dict
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": stripped}

    # Attach tool_calls if present
    tool_calls = [
        {"name": b.get("name", ""), "arguments": b.get("input", {})}
        for b in flattened
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls

    # Build LLM span input from messages collected so far (with thinking stripped)
    llm_input = {"messages": _strip_thinking_from_messages(state.collected_messages)}

    # Add assistant message to collected messages (with raw content for history)
    state.collected_messages.append(assistant_msg)

    llm_span_id = str(uuid4())
    state.current_llm_span_id = llm_span_id
    _set_current_span(llm_span_id)

    span = SpanData(
        span_id=llm_span_id,
        parent_span_id=parent_id,
        name=model or "claude",
        kind=SpanKind.LLM,
        input=llm_input,
        output={"messages": [assistant_msg]},
        model=model,
        model_parameters=model_params,
        started_at=llm_start_time,
        ended_at=datetime.now(timezone.utc),
        status=SpanStatusCode.OK,
    )

    if thinking:
        span.metadata["reasoning_summary"] = thinking

    if error:
        span.status = SpanStatusCode.ERROR
        span.error = str(error)

    state.spans[llm_span_id] = span

    # Track subagent tool uses for parent-child nesting
    for block in flattened:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        tool_use_id = block.get("id")
        tool_name = block.get("name", "")
        if tool_use_id and tool_name == "Agent" and not parent_tool_use_id:
            _create_subagent_span(tool_use_id, block, state)


def _create_subagent_span(tool_use_id: str, block: dict[str, Any], state: _TraceState) -> None:
    """Create an AGENT span for a subagent invocation."""
    tool_input = block.get("input", {})
    subagent_name = "unknown-agent"
    if isinstance(tool_input, dict):
        desc = tool_input.get("description", "")
        subagent_name = tool_input.get("agent_name") or (desc.split()[0] if desc else None) or "unknown-agent"

    subagent_span_id = str(uuid4())
    span = SpanData(
        span_id=subagent_span_id,
        parent_span_id=state.agent_span_id,
        name=subagent_name,
        kind=SpanKind.AGENT,
        input=safe_serialize(tool_input) if tool_input else None,
        started_at=datetime.now(timezone.utc),
    )
    state.spans[subagent_span_id] = span
    state.subagent_spans[tool_use_id] = subagent_span_id


def _handle_user_message(msg: Any, state: _TraceState) -> datetime:
    """Process a UserMessage: record tool results or user text. Returns next llm_start_time."""
    content = getattr(msg, "content", None)
    parent_tool_use_id = getattr(msg, "parent_tool_use_id", None)

    if parent_tool_use_id:
        tool_result = getattr(msg, "tool_use_result", None)
        flattened = flatten_content_blocks(content) if content else []
        result_text = ""
        if tool_result is not None:
            result_text = serialize_tool_response(tool_result)
        elif flattened:
            texts = []
            for b in flattened:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    texts.append(b.get("content", ""))
                elif isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text", ""))
            result_text = "\n".join(texts) if texts else ""

        state.collected_messages.append(
            {
                "role": "tool",
                "content": result_text,
                "tool_call_id": parent_tool_use_id,
            }
        )
    else:
        text = normalize_content_to_text(content) if content is not None else None
        state.collected_messages.append({"role": "user", "content": text})

    # Reset current span to agent so next LLM span parents correctly
    if state.agent_span_id:
        _set_current_span(state.agent_span_id)

    return datetime.now(timezone.utc)


def _handle_result_message(msg: Any, state: _TraceState) -> None:
    """Process a ResultMessage: extract token usage and set trace output."""
    usage_data = getattr(msg, "usage", None)
    if usage_data and state.current_llm_span_id:
        token_usage = extract_token_usage(usage_data)
        if token_usage:
            llm_span = state.spans.get(state.current_llm_span_id)
            if llm_span:
                llm_span.token_usage = token_usage

    # Close any open subagent spans
    now = datetime.now(timezone.utc)
    for sub_span_id in state.subagent_spans.values():
        sub_span = state.spans.get(sub_span_id)
        if sub_span and sub_span.ended_at is None:
            sub_span.ended_at = now
            if sub_span.status == SpanStatusCode.UNSET:
                sub_span.status = SpanStatusCode.OK


# ---------------------------------------------------------------------------
# Wrapper: standalone query() function  →  trace entry point
# ---------------------------------------------------------------------------


async def _wrap_standalone_query(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    adapter = _get_adapter()
    if adapter is None:
        async for msg in wrapped(*args, **kwargs):
            yield msg
        return

    state = _TraceState(adapter=adapter)
    state_token = _set_trace_state(state)
    _set_active_state(state)

    prompt = kwargs.get("prompt") or (args[0] if args else None)
    prompt_text = prompt if isinstance(prompt, str) else None

    options = kwargs.get("options")
    system_prompt = None
    model_name = None
    model_params = None
    if options:
        sp = getattr(options, "system_prompt", None)
        if isinstance(sp, str):
            system_prompt = sp
        elif isinstance(sp, dict) and sp.get("type") == "preset":
            system_prompt = f"preset: {sp.get('preset', 'claude_code')}"
            if "append" in sp:
                system_prompt += f"\n{sp['append']}"
        model_name = getattr(options, "model", None)
        model_params = extract_model_parameters(options)

    initial_messages: list[dict[str, Any]] = []
    if system_prompt:
        initial_messages.append({"role": "system", "content": system_prompt})
    if prompt_text:
        initial_messages.append({"role": "user", "content": prompt_text})

    state.trace_input = extract_last_user_message({"messages": list(initial_messages)}) if initial_messages else None
    state.collected_messages = list(initial_messages)

    chain_span_id = str(uuid4())
    state.root_span_id = chain_span_id
    chain_span = SpanData(
        span_id=chain_span_id,
        parent_span_id=None,
        name="ClaudeAgentSDK",
        kind=SpanKind.CHAIN,
        input={"messages": list(initial_messages)} if initial_messages else state.trace_input,
        started_at=state.trace_started_at,
    )
    state.spans[chain_span_id] = chain_span

    agent_span_id = str(uuid4())
    state.agent_span_id = agent_span_id
    parent_span_token = _set_current_span(agent_span_id)

    agent_span = SpanData(
        span_id=agent_span_id,
        parent_span_id=chain_span_id,
        name="ClaudeAgentSDK",
        kind=SpanKind.AGENT,
        input={"messages": list(initial_messages)} if initial_messages else None,
        model=model_name,
        model_parameters=model_params,
        started_at=state.trace_started_at,
    )
    state.spans[agent_span_id] = agent_span

    llm_start_time = state.trace_started_at
    has_error = False

    try:
        async for msg in wrapped(*args, **kwargs):
            msg_type = type(msg).__name__

            if msg_type == "AssistantMessage":
                _handle_assistant_message(msg, state, model_name, model_params, llm_start_time)
            elif msg_type == "UserMessage":
                llm_start_time = _handle_user_message(msg, state)
            elif msg_type == "ResultMessage":
                _handle_result_message(msg, state)

            yield msg

    except Exception as exc:
        has_error = True
        chain_span.error = str(exc)
        chain_span.status = SpanStatusCode.ERROR
        agent_span.error = str(exc)
        agent_span.status = SpanStatusCode.ERROR
        raise

    finally:
        now = datetime.now(timezone.utc)

        if state.collected_messages:
            chain_span.output = {"messages": _strip_thinking_from_messages(state.collected_messages)}

        last_assistant = None
        for m in reversed(state.collected_messages):
            if m.get("role") == "assistant":
                stripped_content = (
                    strip_thinking_blocks(m.get("content")) if isinstance(m.get("content"), list) else m.get("content")
                )
                last_assistant = {"role": "assistant", "content": stripped_content}
                break
        if last_assistant:
            agent_span.output = {"messages": [last_assistant]}
            state.trace_output = {"messages": [last_assistant]}

        if not has_error:
            agent_span.status = SpanStatusCode.OK
            chain_span.status = SpanStatusCode.OK
        agent_span.ended_at = now
        chain_span.ended_at = now

        adapter._finalize_trace(state, error=has_error)
        _current_trace_state.reset(state_token)
        _current_span_id.reset(parent_span_token)
        _set_active_state(None)


# ---------------------------------------------------------------------------
# Tool hooks (injected via _wrap_client_init)
# ---------------------------------------------------------------------------


async def _pre_tool_use_hook(input_data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
    """Create a TOOL span when a tool execution starts."""
    if not tool_use_id:
        return {}

    state = _get_active_state()
    if state is None:
        return {}

    tool_name = (
        input_data.get("tool_name", "unknown_tool")
        if isinstance(input_data, dict)
        else str(getattr(input_data, "tool_name", "unknown_tool"))
    )
    tool_input = (
        input_data.get("tool_input", {}) if isinstance(input_data, dict) else getattr(input_data, "tool_input", {})
    )

    parent_id = _get_current_span() or state.agent_span_id

    span_id = str(uuid4())
    span = SpanData(
        span_id=span_id,
        parent_span_id=parent_id,
        name=tool_name,
        kind=SpanKind.TOOL,
        input=safe_serialize(tool_input) if tool_input else None,
        started_at=datetime.now(timezone.utc),
    )
    state.spans[span_id] = span
    state.tool_spans[tool_use_id] = span_id

    return {}


async def _post_tool_use_hook(input_data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
    """End a TOOL span when a tool execution completes."""
    if not tool_use_id:
        return {}

    state = _get_active_state()
    if state is None:
        return {}

    span_id = state.tool_spans.pop(tool_use_id, None)
    if not span_id:
        return {}

    span = state.spans.get(span_id)
    if not span:
        return {}

    tool_response = (
        input_data.get("tool_response") if isinstance(input_data, dict) else getattr(input_data, "tool_response", None)
    )

    span.output = serialize_tool_response(tool_response)
    span.status = SpanStatusCode.OK
    span.ended_at = datetime.now(timezone.utc)

    is_error = False
    if isinstance(tool_response, dict):
        is_error = tool_response.get("is_error", False)
    if is_error:
        span.status = SpanStatusCode.ERROR
        span.error = span.output if isinstance(span.output, str) else json.dumps(span.output)

    return {}


async def _post_tool_use_failure_hook(input_data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
    """End a TOOL span when a tool execution fails."""
    if not tool_use_id:
        return {}

    state = _get_active_state()
    if state is None:
        return {}

    span_id = state.tool_spans.pop(tool_use_id, None)
    if not span_id:
        return {}

    span = state.spans.get(span_id)
    if not span:
        return {}

    error_msg = (
        input_data.get("error", "Unknown error")
        if isinstance(input_data, dict)
        else str(getattr(input_data, "error", "Unknown error"))
    )

    span.output = {"error": error_msg}
    span.error = str(error_msg)
    span.status = SpanStatusCode.ERROR
    span.ended_at = datetime.now(timezone.utc)

    return {}
