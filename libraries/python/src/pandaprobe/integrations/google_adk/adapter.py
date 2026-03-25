"""Google ADK integration adapter for PandaProbe tracing.

Instruments Google ADK by monkey-patching key internal methods via ``wrapt``.
Each ``Runner.run_async`` invocation produces a full ``TraceData`` with nested
``SpanData`` objects for agents, LLM calls, and tool executions.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pandaprobe.integrations._base import BaseIntegrationAdapter
from pandaprobe.integrations.google_adk.utils import (
    _content_to_message,
    extract_model_name,
    extract_model_parameters,
    extract_text_from_content,
    extract_token_usage,
    normalize_contents_to_messages,
    normalize_llm_response_to_messages,
    safe_serialize,
)
from pandaprobe.schemas import SpanData, SpanKind, SpanStatusCode, TraceData, TraceStatus
from pandaprobe.tracing.session import get_current_session_id, get_current_user_id
from pandaprobe.validation import extract_last_user_message

logger = logging.getLogger("pandaprobe")

_patched = False

# ---------------------------------------------------------------------------
# Per-invocation trace state
# ---------------------------------------------------------------------------

_current_trace_state: ContextVar[_TraceState | None] = ContextVar("_pandaprobe_adk_trace_state", default=None)
_current_span_id: ContextVar[str | None] = ContextVar("_pandaprobe_adk_current_span", default=None)


@dataclass
class _TraceState:
    """Per-invocation state accumulated during a single Runner.run_async call."""

    adapter: GoogleADKAdapter
    spans: dict[str, SpanData] = field(default_factory=dict)
    root_span_id: str | None = None
    trace_input: Any = None
    trace_output: Any = None
    chain_output: Any = None
    trace_started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class GoogleADKAdapter(BaseIntegrationAdapter):
    """PandaProbe integration adapter for Google ADK.

    Usage::

        from pandaprobe.integrations.google_adk import GoogleADKAdapter

        adapter = GoogleADKAdapter(tags=["example"])
        adapter.instrument()

        # All subsequent Runner.run_async() calls are now traced automatically.
    """

    def instrument(self) -> bool:
        """Apply monkey-patches to Google ADK internals.

        Returns ``True`` if patching succeeded, ``False`` if dependencies are
        missing.  The method is idempotent — calling it multiple times is safe.
        """
        global _patched
        if _patched:
            return True

        try:
            import google.adk  # noqa: F401
            from wrapt import wrap_function_wrapper
        except ImportError as exc:
            logger.warning("PandaProbe Google ADK integration: missing dependency — %s", exc)
            return False

        _store_adapter(self)

        wraps = [
            ("google.adk.runners", "Runner.run_async", _wrap_runner_run_async),
            ("google.adk.agents.base_agent", "BaseAgent.run_async", _wrap_agent_run_async),
            (
                "google.adk.flows.llm_flows.base_llm_flow",
                "BaseLlmFlow._call_llm_async",
                _wrap_llm_call_async,
            ),
            ("google.adk.tools.base_tool", "BaseTool.run_async", _wrap_tool_run_async),
            ("google.adk.tools.function_tool", "FunctionTool.run_async", _wrap_tool_run_async),
        ]

        for module, name, wrapper in wraps:
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
                name="GoogleADK",
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
            logger.error("PandaProbe Google ADK adapter failed to submit trace: %s", exc)


# ---------------------------------------------------------------------------
# Module-level adapter storage
# ---------------------------------------------------------------------------

_active_adapter: GoogleADKAdapter | None = None


def _store_adapter(adapter: GoogleADKAdapter) -> None:
    global _active_adapter
    _active_adapter = adapter


def _get_adapter() -> GoogleADKAdapter | None:
    return _active_adapter


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
# Helper: build messages from session events
# ---------------------------------------------------------------------------


def _build_messages_from_session(ctx: Any) -> list[dict[str, Any]]:
    """Build an ordered list of message dicts from ADK session events."""
    session = getattr(ctx, "session", None)
    if not session:
        return []
    events = getattr(session, "events", None) or []
    messages: list[dict[str, Any]] = []
    for event in events:
        content = getattr(event, "content", None)
        if content is None:
            continue
        converted = _content_to_message(content)
        if isinstance(converted, list):
            messages.extend(converted)
        else:
            messages.append(converted)
    return messages


# ---------------------------------------------------------------------------
# Wrapper: Runner.run_async  →  root CHAIN span + trace lifecycle
# ---------------------------------------------------------------------------


async def _wrap_runner_run_async(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    adapter = _get_adapter()
    if adapter is None:
        async for event in wrapped(*args, **kwargs):
            yield event
        return

    state = _TraceState(adapter=adapter)
    state_token = _set_trace_state(state)

    span_id = str(uuid4())
    parent_span_token = _set_current_span(span_id)
    state.root_span_id = span_id

    trace_input_text = None
    new_message = kwargs.get("new_message")
    if new_message is not None:
        trace_input_text = extract_text_from_content(new_message)

    if trace_input_text:
        state.trace_input = extract_last_user_message({"messages": [{"role": "user", "content": trace_input_text}]})

    root_span = SpanData(
        span_id=span_id,
        parent_span_id=None,
        name=getattr(instance, "app_name", None) or "GoogleADK",
        kind=SpanKind.CHAIN,
        input=state.trace_input,
        started_at=state.trace_started_at,
    )
    state.spans[span_id] = root_span

    has_error = False
    try:
        async for event in wrapped(*args, **kwargs):
            content = getattr(event, "content", None)
            if content is not None:
                text = extract_text_from_content(content)
                if text:
                    state.trace_output = {"messages": [{"role": "assistant", "content": text}]}
            yield event
    except Exception as exc:
        has_error = True
        root_span.error = str(exc)
        root_span.status = SpanStatusCode.ERROR
        raise
    finally:
        root_span.output = state.chain_output or state.trace_output
        if not has_error:
            root_span.status = SpanStatusCode.OK
        root_span.ended_at = datetime.now(timezone.utc)

        adapter._finalize_trace(state, error=has_error)

        _current_trace_state.reset(state_token)
        _current_span_id.reset(parent_span_token)


# ---------------------------------------------------------------------------
# Wrapper: BaseAgent.run_async  →  AGENT span
# ---------------------------------------------------------------------------


async def _wrap_agent_run_async(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    state = _get_trace_state()
    if state is None:
        async for event in wrapped(*args, **kwargs):
            yield event
        return

    span_id = str(uuid4())
    parent_id = _get_current_span()
    parent_span_token = _set_current_span(span_id)

    agent_name = getattr(instance, "name", None) or type(instance).__name__

    ctx = args[0] if args else kwargs.get("ctx")
    session_messages = _build_messages_from_session(ctx) if ctx else []

    instruction = getattr(instance, "instruction", None)
    if isinstance(instruction, str) and instruction.strip():
        session_messages = [{"role": "system", "content": instruction}] + session_messages

    span_input: Any = {"messages": session_messages} if session_messages else None

    if session_messages and state.root_span_id:
        root_span = state.spans.get(state.root_span_id)
        if root_span:
            root_span.input = {"messages": list(session_messages)}

    span = SpanData(
        span_id=span_id,
        parent_span_id=parent_id,
        name=agent_name,
        kind=SpanKind.AGENT,
        input=span_input,
        started_at=datetime.now(timezone.utc),
    )
    state.spans[span_id] = span

    has_error = False
    final_text: str | None = None
    try:
        async for event in wrapped(*args, **kwargs):
            content = getattr(event, "content", None)
            if content is not None:
                text = extract_text_from_content(content)
                if text:
                    final_text = text
            yield event
    except Exception as exc:
        has_error = True
        span.error = str(exc)
        span.status = SpanStatusCode.ERROR
        raise
    finally:
        if final_text:
            span.output = {"messages": [{"role": "assistant", "content": final_text}]}

        if ctx and state.root_span_id:
            updated_messages = _build_messages_from_session(ctx)
            if updated_messages:
                state.chain_output = {"messages": updated_messages}

        if not has_error:
            span.status = SpanStatusCode.OK
        span.ended_at = datetime.now(timezone.utc)
        _current_span_id.reset(parent_span_token)


# ---------------------------------------------------------------------------
# Wrapper: BaseLlmFlow._call_llm_async  →  LLM span
# ---------------------------------------------------------------------------


async def _wrap_llm_call_async(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    state = _get_trace_state()
    if state is None:
        async for event in wrapped(*args, **kwargs):
            yield event
        return

    span_id = str(uuid4())
    parent_id = _get_current_span()
    parent_span_token = _set_current_span(span_id)

    llm_request = args[1] if len(args) > 1 else kwargs.get("llm_request")
    model_name = extract_model_name(llm_request) if llm_request else None
    model_params = extract_model_parameters(llm_request) if llm_request else None

    span_input: Any = None
    if llm_request:
        contents = getattr(llm_request, "contents", None)
        config = getattr(llm_request, "config", None)
        sys_instruction = getattr(config, "system_instruction", None) if config else None
        span_input = normalize_contents_to_messages(contents, sys_instruction)

    span = SpanData(
        span_id=span_id,
        parent_span_id=parent_id,
        name=model_name or "google_adk_llm",
        kind=SpanKind.LLM,
        input=span_input,
        model=model_name,
        model_parameters=model_params,
        started_at=datetime.now(timezone.utc),
    )
    state.spans[span_id] = span

    first_token_recorded = False
    last_event = None
    event_with_content = None
    has_error = False

    try:
        async for event in wrapped(*args, **kwargs):
            is_partial = getattr(event, "partial", False)
            if not first_token_recorded and is_partial:
                first_token_recorded = True
                span.completion_start_time = datetime.now(timezone.utc)

            last_event = event
            if hasattr(event, "content") and event.content is not None:
                event_with_content = event
            yield event
    except Exception as exc:
        has_error = True
        span.error = str(exc)
        span.status = SpanStatusCode.ERROR
        raise
    finally:
        content_source = event_with_content or last_event
        if content_source:
            response_content = getattr(content_source, "content", None)
            if response_content is not None:
                span.output = normalize_llm_response_to_messages(response_content)

            token_usage = extract_token_usage(content_source)
            if token_usage:
                span.token_usage = token_usage

        if not has_error:
            span.status = SpanStatusCode.OK
        span.ended_at = datetime.now(timezone.utc)
        _current_span_id.reset(parent_span_token)


# ---------------------------------------------------------------------------
# Wrapper: BaseTool.run_async / FunctionTool.run_async  →  TOOL span
# ---------------------------------------------------------------------------


async def _wrap_tool_run_async(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    state = _get_trace_state()
    if state is None:
        return await wrapped(*args, **kwargs)

    span_id = str(uuid4())
    parent_id = _get_current_span()
    parent_span_token = _set_current_span(span_id)

    tool_name = getattr(instance, "name", None) or type(instance).__name__
    tool_args = kwargs.get("args") or (args[0] if args else {})
    span_input = safe_serialize(tool_args) if tool_args else None

    span = SpanData(
        span_id=span_id,
        parent_span_id=parent_id,
        name=tool_name,
        kind=SpanKind.TOOL,
        input=span_input,
        started_at=datetime.now(timezone.utc),
    )
    state.spans[span_id] = span

    try:
        result = await wrapped(*args, **kwargs)
        if result is not None:
            span.output = safe_serialize(result)
        span.status = SpanStatusCode.OK
        return result
    except Exception as exc:
        span.error = str(exc)
        span.status = SpanStatusCode.ERROR
        raise
    finally:
        span.ended_at = datetime.now(timezone.utc)
        _current_span_id.reset(parent_span_token)
