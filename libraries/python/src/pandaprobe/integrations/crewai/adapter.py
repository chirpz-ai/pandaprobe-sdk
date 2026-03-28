"""CrewAI integration adapter for PandaProbe tracing.

Instruments CrewAI by monkey-patching key methods via ``wrapt``.
Each ``Crew.kickoff()`` invocation produces a full ``TraceData`` with nested
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
from pandaprobe.integrations.crewai.utils import (
    build_agent_system_message,
    build_crew_system_message,
    extract_model_name,
    extract_model_parameters,
    extract_reasoning_from_messages,
    extract_token_usage,
    normalize_messages,
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

_current_trace_state: ContextVar[_TraceState | None] = ContextVar("_pandaprobe_crewai_trace_state", default=None)
_current_span_id: ContextVar[str | None] = ContextVar("_pandaprobe_crewai_current_span", default=None)


@dataclass
class _AgentRecord:
    """Captures one agent's task description and output for context propagation."""

    agent: str
    task: str
    output: str


@dataclass
class _TraceState:
    """Per-invocation state accumulated during a single Crew.kickoff call."""

    adapter: CrewAIAdapter
    spans: dict[str, SpanData] = field(default_factory=dict)
    root_span_id: str | None = None
    trace_input: Any = None
    trace_output: Any = None
    crew_system_message: str | None = None
    agent_records: list[_AgentRecord] = field(default_factory=list)
    trace_started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class CrewAIAdapter(BaseIntegrationAdapter):
    """PandaProbe integration adapter for CrewAI.

    Usage::

        from pandaprobe.integrations.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(tags=["example"])
        adapter.instrument()

        # All subsequent Crew.kickoff() calls are now traced automatically.
    """

    def instrument(self) -> bool:
        """Apply monkey-patches to CrewAI internals.

        Returns ``True`` if patching succeeded, ``False`` if dependencies are
        missing.  The method is idempotent — calling it multiple times is safe.
        """
        global _patched

        _store_adapter(self)

        if _patched:
            return True

        try:
            import crewai  # noqa: F401
            from wrapt import wrap_function_wrapper
        except ImportError as exc:
            logger.warning("PandaProbe CrewAI integration: missing dependency — %s", exc)
            return False

        # Required wrap: kickoff is the main trace entry point
        try:
            wrap_function_wrapper("crewai", "Crew.kickoff", _wrap_kickoff)
        except Exception as exc:
            logger.error("PandaProbe CrewAI: failed to wrap Crew.kickoff — %s", exc)
            return False

        # Optional wraps — fail gracefully
        optional_wraps: list[tuple[str, str, Any]] = [
            ("crewai", "Agent.execute_task", _wrap_agent_execute_task),
            ("crewai", "Crew.kickoff_for_each", _wrap_kickoff),
        ]

        # LLM.call wraps: native providers (returned by LLM.__new__ factory)
        # plus the LiteLLM fallback on the LLM class itself.
        llm_wraps: list[tuple[str, str]] = [
            ("crewai.llms.providers.openai.completion", "OpenAICompletion.call"),
            ("crewai.llms.providers.anthropic.completion", "AnthropicCompletion.call"),
            ("crewai.llms.providers.gemini.completion", "GeminiCompletion.call"),
            ("crewai", "LLM.call"),
        ]
        for module, name in llm_wraps:
            optional_wraps.append((module, name, _wrap_llm_call))

        # Tool execution (standalone function)
        optional_wraps.append(
            ("crewai.agents.crew_agent_executor", "execute_tool_and_check_finality", _wrap_tool_execute),
        )

        for module, name, wrapper in optional_wraps:
            try:
                wrap_function_wrapper(module, name, wrapper)
            except Exception as exc:
                logger.debug("PandaProbe: failed to wrap %s.%s — %s", module, name, exc)

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
                name="CrewAI",
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
            logger.error("PandaProbe CrewAI adapter failed to submit trace: %s", exc)


# ---------------------------------------------------------------------------
# Module-level adapter storage
# ---------------------------------------------------------------------------

_active_adapter: CrewAIAdapter | None = None


def _store_adapter(adapter: CrewAIAdapter) -> None:
    global _active_adapter
    _active_adapter = adapter


def _get_adapter() -> CrewAIAdapter | None:
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
# Helper: build trace-level input / output
# ---------------------------------------------------------------------------


def _build_crew_input(instance: Any, kwargs: Any) -> dict[str, Any] | None:
    """Build CHAIN-level input from the Crew instance.

    Returns ``{"messages": [system?, user]}`` with the crew config as a
    system message and the task descriptions as the user message.
    """
    inputs = kwargs.get("inputs") or {}

    tasks = getattr(instance, "tasks", None) or []
    task_descriptions: list[str] = []
    for task in tasks:
        desc = getattr(task, "description", None)
        if desc:
            task_descriptions.append(str(desc))

    # User content: kickoff inputs or concatenated task descriptions
    user_content: str | None = None
    if isinstance(inputs, dict) and inputs:
        user_content = "\n".join(f"{k}: {v}" for k, v in inputs.items())
    elif task_descriptions:
        user_content = "\n".join(task_descriptions)

    if not user_content:
        return None

    messages: list[dict[str, Any]] = []

    crew_system = build_crew_system_message(instance)
    if crew_system:
        messages.append({"role": "system", "content": crew_system})

    messages.append({"role": "user", "content": user_content})
    return {"messages": messages}


def _extract_crew_output(result: Any) -> dict[str, Any] | None:
    """Extract output from CrewOutput for trace-level output."""
    if result is None:
        return None

    raw = getattr(result, "raw", None)
    if raw:
        return {"messages": [{"role": "assistant", "content": str(raw)}]}

    if isinstance(result, str):
        return {"messages": [{"role": "assistant", "content": result}]}

    return safe_serialize(result)


# ---------------------------------------------------------------------------
# Wrapper: Crew.kickoff  →  root CHAIN span + trace lifecycle
# ---------------------------------------------------------------------------


def _wrap_kickoff(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    adapter = _get_adapter()
    if adapter is None:
        return wrapped(*args, **kwargs)

    state = _TraceState(adapter=adapter)
    state_token = _set_trace_state(state)

    span_id = str(uuid4())
    parent_span_token = _set_current_span(span_id)
    state.root_span_id = span_id

    crew_system = build_crew_system_message(instance)
    state.crew_system_message = crew_system

    trace_input = _build_crew_input(instance, kwargs)
    state.trace_input = extract_last_user_message(trace_input) if trace_input else None

    root_span = SpanData(
        span_id=span_id,
        parent_span_id=None,
        name="CrewAI",
        kind=SpanKind.CHAIN,
        input=trace_input,
        started_at=state.trace_started_at,
    )
    state.spans[span_id] = root_span

    has_error = False
    try:
        result = wrapped(*args, **kwargs)

        output = _extract_crew_output(result)
        state.trace_output = output

        return result
    except Exception as exc:
        has_error = True
        root_span.error = str(exc)
        root_span.status = SpanStatusCode.ERROR
        raise
    finally:
        # Build CHAIN output from accumulated agent records
        chain_output = _build_chain_output(state)
        root_span.output = chain_output or state.trace_output

        if not has_error:
            root_span.status = SpanStatusCode.OK
        root_span.ended_at = datetime.now(timezone.utc)

        adapter._finalize_trace(state, error=has_error)

        _current_trace_state.reset(state_token)
        _current_span_id.reset(parent_span_token)


def _build_chain_output(state: _TraceState) -> dict[str, Any] | None:
    """Assemble CHAIN output from crew system message + agent conversation history."""
    if not state.agent_records:
        return None

    messages: list[dict[str, Any]] = []
    if state.crew_system_message:
        messages.append({"role": "system", "content": state.crew_system_message})

    for rec in state.agent_records:
        messages.append({"role": "user", "content": rec.task})
        messages.append({"role": "assistant", "content": rec.output})

    return {"messages": messages} if messages else None


# ---------------------------------------------------------------------------
# Wrapper: Agent.execute_task  →  AGENT span
# ---------------------------------------------------------------------------


def _wrap_agent_execute_task(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    state = _get_trace_state()
    if state is None:
        return wrapped(*args, **kwargs)

    span_id = str(uuid4())
    parent_id = _get_current_span()
    parent_span_token = _set_current_span(span_id)

    agent_role = getattr(instance, "role", None) or type(instance).__name__

    # Extract task description from first positional arg
    task = args[0] if args else kwargs.get("task")
    task_desc = getattr(task, "description", None) if task else None
    expected_output = getattr(task, "expected_output", None) if task else None

    task_text = str(task_desc) if task_desc else ""

    # Build input messages: system (backstory) + prior agent context + task
    input_messages: list[dict[str, Any]] = []

    system_content = build_agent_system_message(instance)
    if system_content:
        input_messages.append({"role": "system", "content": system_content})

    for rec in state.agent_records:
        input_messages.append({"role": "assistant", "content": f"[{rec.agent}]: {rec.output}"})

    if task_text:
        input_messages.append({"role": "user", "content": task_text})

    span_input = {"messages": input_messages} if input_messages else None

    # Extract model name from agent's LLM (for display only, not on span)
    llm = getattr(instance, "llm", None)
    model_name = extract_model_name(llm)

    metadata: dict[str, Any] = {}
    if expected_output:
        metadata["expected_output"] = str(expected_output)

    span = SpanData(
        span_id=span_id,
        parent_span_id=parent_id,
        name=agent_role,
        kind=SpanKind.AGENT,
        input=span_input,
        model=model_name,
        metadata=metadata,
        started_at=datetime.now(timezone.utc),
    )
    state.spans[span_id] = span

    has_error = False
    try:
        result = wrapped(*args, **kwargs)

        result_str: str | None = None
        if result is not None:
            result_str = getattr(result, "raw", None) or str(result)
            span.output = {"messages": [{"role": "assistant", "content": result_str}]}

        # Record this agent's work for subsequent agents and CHAIN output
        if result_str and task_text:
            state.agent_records.append(_AgentRecord(agent=agent_role, task=task_text, output=result_str))

        return result
    except Exception as exc:
        has_error = True
        span.error = str(exc)
        span.status = SpanStatusCode.ERROR
        raise
    finally:
        if not has_error:
            span.status = SpanStatusCode.OK
        span.ended_at = datetime.now(timezone.utc)
        _current_span_id.reset(parent_span_token)


# ---------------------------------------------------------------------------
# Wrapper: LLM.call (native providers + LiteLLM fallback)  →  LLM span
# ---------------------------------------------------------------------------


def _wrap_llm_call(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    state = _get_trace_state()
    if state is None:
        return wrapped(*args, **kwargs)

    span_id = str(uuid4())
    parent_id = _get_current_span()
    parent_span_token = _set_current_span(span_id)

    model_name = extract_model_name(instance)
    model_params = extract_model_parameters(instance)

    # Build LLM input from messages argument
    messages = args[0] if args else kwargs.get("messages")
    span_input: Any = None
    if messages is not None:
        if isinstance(messages, list):
            span_input = normalize_messages(messages)
        elif isinstance(messages, str):
            span_input = {"messages": [{"role": "user", "content": messages}]}

    # Extract reasoning from input messages before stripping
    reasoning: str | None = None
    if isinstance(messages, list):
        reasoning = extract_reasoning_from_messages(messages)

    span = SpanData(
        span_id=span_id,
        parent_span_id=parent_id,
        name=model_name or "crewai_llm",
        kind=SpanKind.LLM,
        input=span_input,
        model=model_name,
        model_parameters=model_params,
        started_at=datetime.now(timezone.utc),
    )
    state.spans[span_id] = span

    has_error = False
    try:
        result = wrapped(*args, **kwargs)

        if result is not None:
            if isinstance(result, str):
                span.output = {"messages": [{"role": "assistant", "content": result}]}
            elif isinstance(result, dict):
                content = result.get("content") or result.get("text") or str(result)
                span.output = {"messages": [{"role": "assistant", "content": content}]}
            else:
                raw = getattr(result, "text", None) or getattr(result, "content", None)
                if raw:
                    span.output = {"messages": [{"role": "assistant", "content": str(raw)}]}
                else:
                    span.output = {"messages": [{"role": "assistant", "content": str(result)}]}

        if reasoning:
            span.metadata["reasoning_summary"] = reasoning

        span.token_usage = extract_token_usage(instance)

        return result
    except Exception as exc:
        has_error = True
        span.error = str(exc)
        span.status = SpanStatusCode.ERROR
        raise
    finally:
        if not has_error:
            span.status = SpanStatusCode.OK
        span.ended_at = datetime.now(timezone.utc)
        _current_span_id.reset(parent_span_token)


# ---------------------------------------------------------------------------
# Wrapper: execute_tool_and_check_finality  →  TOOL span
# ---------------------------------------------------------------------------


def _wrap_tool_execute(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    state = _get_trace_state()
    if state is None:
        return wrapped(*args, **kwargs)

    span_id = str(uuid4())
    parent_id = _get_current_span()
    parent_span_token = _set_current_span(span_id)

    # First positional arg is an AgentAction with .tool and .tool_input
    agent_action = args[0] if args else kwargs.get("agent_action")
    tool_name = getattr(agent_action, "tool", None) or "unknown_tool"
    raw_input = getattr(agent_action, "tool_input", None)
    tool_input = safe_serialize(raw_input) if raw_input is not None else None

    span = SpanData(
        span_id=span_id,
        parent_span_id=parent_id,
        name=str(tool_name),
        kind=SpanKind.TOOL,
        input=tool_input,
        started_at=datetime.now(timezone.utc),
    )
    state.spans[span_id] = span

    has_error = False
    try:
        result = wrapped(*args, **kwargs)

        # ToolResult is a dataclass with .result (str) and .result_as_answer (bool)
        if result is not None:
            tool_output = getattr(result, "result", None)
            if tool_output is not None:
                span.output = str(tool_output)
            else:
                span.output = str(result)

        return result
    except Exception as exc:
        has_error = True
        span.error = str(exc)
        span.status = SpanStatusCode.ERROR
        raise
    finally:
        if not has_error:
            span.status = SpanStatusCode.OK
        span.ended_at = datetime.now(timezone.utc)
        _current_span_id.reset(parent_span_token)
