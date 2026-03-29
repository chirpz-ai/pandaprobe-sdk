"""OpenAI Agents SDK integration adapter for PandaProbe tracing.

Instruments the OpenAI Agents SDK by implementing its first-class
``TracingProcessor`` callback interface.  Each ``Runner.run`` invocation
produces a full ``TraceData`` with nested ``SpanData`` objects for agents,
LLM calls (Responses API and Chat Completions), tool invocations, handoffs,
and guardrails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pandaprobe.integrations._base import BaseIntegrationAdapter, safe_serialize
from pandaprobe.integrations.openai_agents.utils import (
    extract_generation_model_parameters,
    extract_reasoning,
    extract_response_model_parameters,
    extract_token_usage,
    normalize_generation_input,
    normalize_generation_output,
    normalize_response_input,
    normalize_response_output,
    serialize_tool_io,
)
from pandaprobe.schemas import SpanData, SpanKind, SpanStatusCode, TraceData, TraceStatus
from pandaprobe.tracing.session import get_current_session_id, get_current_user_id
from pandaprobe.validation import extract_last_user_message

logger = logging.getLogger("pandaprobe")

_patched = False

# ---------------------------------------------------------------------------
# Per-invocation trace state
# ---------------------------------------------------------------------------

_trace_states: dict[str, _TraceState] = {}


@dataclass
class _TraceState:
    """Per-invocation state accumulated during a single Runner.run call."""

    adapter: OpenAIAgentsAdapter
    spans: dict[str, SpanData] = field(default_factory=dict)
    root_span_id: str | None = None
    trace_input: Any = None
    trace_output: Any = None
    chain_input: Any = None
    chain_output: Any = None
    sdk_to_pandaprobe: dict[str, str] = field(default_factory=dict)
    trace_id: str | None = None
    trace_started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class OpenAIAgentsAdapter(BaseIntegrationAdapter):
    """PandaProbe integration adapter for the OpenAI Agents SDK.

    Usage::

        from pandaprobe.integrations.openai_agents import OpenAIAgentsAdapter

        adapter = OpenAIAgentsAdapter(tags=["example"])
        adapter.instrument()

        # All subsequent Runner.run() calls are now traced automatically.
    """

    def instrument(self) -> bool:
        """Register a ``TracingProcessor`` with the OpenAI Agents SDK.

        Returns ``True`` if registration succeeded, ``False`` if dependencies
        are missing.  The method is idempotent — calling it multiple times is
        safe.
        """
        global _patched

        _store_adapter(self)

        if _patched:
            return True

        try:
            from agents import tracing as agents_tracing  # type: ignore[import-not-found]
        except ImportError as exc:
            logger.warning("PandaProbe OpenAI Agents SDK integration: missing dependency — %s", exc)
            return False

        class _PandaProbeProcessor(agents_tracing.TracingProcessor):
            """Bridge between OpenAI Agents SDK tracing and PandaProbe."""

            def on_trace_start(self, trace: Any) -> None:
                _on_trace_start(trace)

            def on_trace_end(self, trace: Any) -> None:
                _on_trace_end(trace)

            def on_span_start(self, span: Any) -> None:
                _on_span_start(span)

            def on_span_end(self, span: Any) -> None:
                _on_span_end(span)

            def shutdown(self) -> None:
                pass

            def force_flush(self) -> None:
                pass

        agents_tracing.add_trace_processor(_PandaProbeProcessor())

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
                name="OpenAIAgentsSDK",
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
            logger.error("PandaProbe OpenAI Agents SDK adapter failed to submit trace: %s", exc)


# ---------------------------------------------------------------------------
# Module-level adapter storage
# ---------------------------------------------------------------------------

_active_adapter: OpenAIAgentsAdapter | None = None


def _store_adapter(adapter: OpenAIAgentsAdapter) -> None:
    global _active_adapter
    _active_adapter = adapter


def _get_adapter() -> OpenAIAgentsAdapter | None:
    return _active_adapter


# ---------------------------------------------------------------------------
# Span kind resolution
# ---------------------------------------------------------------------------

_SPAN_TYPE_TO_KIND: dict[str, SpanKind] = {
    "agent": SpanKind.AGENT,
    "handoff": SpanKind.AGENT,
    "response": SpanKind.LLM,
    "generation": SpanKind.LLM,
    "function": SpanKind.TOOL,
    "guardrail": SpanKind.OTHER,
    "custom": SpanKind.OTHER,
}


def _resolve_span_kind(span_data: Any) -> SpanKind:
    """Determine the PandaProbe SpanKind from an SDK span_data object."""
    span_type = getattr(span_data, "type", None)
    if span_type and isinstance(span_type, str):
        return _SPAN_TYPE_TO_KIND.get(span_type, SpanKind.OTHER)
    return SpanKind.OTHER


def _resolve_span_name(span: Any) -> str:
    """Derive a human-readable name from an SDK span."""
    span_data = getattr(span, "span_data", None)
    if span_data is not None:
        name = getattr(span_data, "name", None)
        if name:
            return str(name)

    name = getattr(span, "name", None)
    if name:
        return str(name)

    span_type = getattr(span_data, "type", None) if span_data else None
    if span_type:
        return str(span_type).capitalize()

    return "Span"


# ---------------------------------------------------------------------------
# TracingProcessor event handlers
# ---------------------------------------------------------------------------


def _on_trace_start(trace: Any) -> None:
    """Handle trace start: create _TraceState and root CHAIN span."""
    adapter = _get_adapter()
    if adapter is None:
        return

    try:
        trace_id = getattr(trace, "trace_id", None)
        if trace_id is None:
            return

        trace_id = str(trace_id)

        state = _TraceState(adapter=adapter, trace_id=trace_id)

        root_span_id = str(uuid4())
        state.root_span_id = root_span_id

        trace_name = getattr(trace, "name", None) or "OpenAIAgentsSDK"

        root_span = SpanData(
            span_id=root_span_id,
            parent_span_id=None,
            name=str(trace_name),
            kind=SpanKind.CHAIN,
            started_at=state.trace_started_at,
        )
        state.spans[root_span_id] = root_span
        _trace_states[trace_id] = state
    except Exception as exc:
        logger.debug("PandaProbe OpenAI Agents: error in on_trace_start — %s", exc)


def _on_trace_end(trace: Any) -> None:
    """Handle trace end: finalize root CHAIN span and submit trace."""
    try:
        trace_id = str(getattr(trace, "trace_id", ""))
        state = _trace_states.pop(trace_id, None)
        if state is None:
            return

        now = datetime.now(timezone.utc)
        root_span = state.spans.get(state.root_span_id) if state.root_span_id else None

        has_error = any(s.status == SpanStatusCode.ERROR for s in state.spans.values())

        if root_span is not None:
            root_span.input = state.chain_input
            root_span.output = state.chain_output
            root_span.ended_at = now
            root_span.status = SpanStatusCode.ERROR if has_error else SpanStatusCode.OK

        state.adapter._finalize_trace(state, error=has_error)
    except Exception as exc:
        logger.debug("PandaProbe OpenAI Agents: error in on_trace_end — %s", exc)


def _on_span_start(span: Any) -> None:
    """Handle span start: create SpanData, resolve parent, register ID mapping."""
    try:
        trace_id = str(getattr(span, "trace_id", ""))
        state = _trace_states.get(trace_id)
        if state is None:
            return

        sdk_span_id = str(getattr(span, "span_id", ""))
        if not sdk_span_id:
            return

        span_data = getattr(span, "span_data", None)
        kind = _resolve_span_kind(span_data)
        name = _resolve_span_name(span)

        pp_span_id = str(uuid4())
        state.sdk_to_pandaprobe[sdk_span_id] = pp_span_id

        sdk_parent_id = getattr(span, "parent_id", None)
        if sdk_parent_id is not None:
            sdk_parent_id = str(sdk_parent_id)

        if sdk_parent_id is None or sdk_parent_id == trace_id:
            parent_pp_id = state.root_span_id
        else:
            parent_pp_id = state.sdk_to_pandaprobe.get(sdk_parent_id, state.root_span_id)

        started_at = None
        raw_started = getattr(span, "started_at", None)
        if raw_started:
            try:
                started_at = datetime.fromisoformat(str(raw_started))
            except (ValueError, TypeError):
                pass
        if started_at is None:
            started_at = datetime.now(timezone.utc)

        pp_span = SpanData(
            span_id=pp_span_id,
            parent_span_id=parent_pp_id,
            name=name,
            kind=kind,
            started_at=started_at,
        )
        state.spans[pp_span_id] = pp_span
    except Exception as exc:
        logger.debug("PandaProbe OpenAI Agents: error in on_span_start — %s", exc)


def _on_span_end(span: Any) -> None:
    """Handle span end: fill in output, token usage, model params, reasoning, error."""
    try:
        trace_id = str(getattr(span, "trace_id", ""))
        state = _trace_states.get(trace_id)
        if state is None:
            return

        sdk_span_id = str(getattr(span, "span_id", ""))
        pp_span_id = state.sdk_to_pandaprobe.get(sdk_span_id)
        if pp_span_id is None:
            return

        pp_span = state.spans.get(pp_span_id)
        if pp_span is None:
            return

        ended_at = None
        raw_ended = getattr(span, "ended_at", None)
        if raw_ended:
            try:
                ended_at = datetime.fromisoformat(str(raw_ended))
            except (ValueError, TypeError):
                pass
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)
        pp_span.ended_at = ended_at

        error = getattr(span, "error", None)
        if error:
            pp_span.error = str(error)
            pp_span.status = SpanStatusCode.ERROR
        else:
            pp_span.status = SpanStatusCode.OK

        span_data = getattr(span, "span_data", None)
        if span_data is None:
            return

        span_type = getattr(span_data, "type", None)

        if span_type == "response":
            _fill_response_span(pp_span, span_data, state)
        elif span_type == "generation":
            _fill_generation_span(pp_span, span_data, state)
        elif span_type == "function":
            _fill_function_span(pp_span, span_data)
        elif span_type == "agent":
            _fill_agent_span(pp_span, span_data)
        elif span_type == "handoff":
            _fill_handoff_span(pp_span, span_data)
        elif span_type == "guardrail":
            _fill_guardrail_span(pp_span, span_data)
        elif span_type == "custom":
            _fill_custom_span(pp_span, span_data)

    except Exception as exc:
        logger.debug("PandaProbe OpenAI Agents: error in on_span_end — %s", exc)


# ---------------------------------------------------------------------------
# Span-type-specific data extraction
# ---------------------------------------------------------------------------


def _propagate_llm_to_parent_agent(pp_span: SpanData, state: _TraceState) -> None:
    """Propagate an LLM span's input/output to its parent AGENT span.

    The first child LLM span's input becomes the agent's input.
    The last child LLM span's output always updates the agent's output.
    """
    parent_id = str(pp_span.parent_span_id) if pp_span.parent_span_id else None
    if not parent_id:
        return
    parent_span = state.spans.get(parent_id)
    if parent_span is None or parent_span.kind != SpanKind.AGENT:
        return
    if parent_span.input is None and pp_span.input:
        parent_span.input = pp_span.input
    if pp_span.output:
        parent_span.output = pp_span.output


def _update_chain_and_trace_io(pp_span: SpanData, state: _TraceState) -> None:
    """Update chain-level and trace-level I/O from an LLM span.

    Chain I/O: the first LLM span's full input and a combined
    (input messages + output messages) for output, giving the full
    conversation including the new assistant response.

    Trace I/O: trimmed — last user message for input, last assistant
    output for output.
    """
    if state.chain_input is None and pp_span.input:
        state.chain_input = pp_span.input

    chain_messages: list[dict[str, Any]] = []
    if pp_span.input and isinstance(pp_span.input, dict):
        chain_messages.extend(pp_span.input.get("messages", []))
    if pp_span.output and isinstance(pp_span.output, dict):
        chain_messages.extend(pp_span.output.get("messages", []))
    if chain_messages:
        state.chain_output = {"messages": chain_messages}
    elif pp_span.output:
        state.chain_output = pp_span.output

    if state.trace_input is None and pp_span.input:
        state.trace_input = extract_last_user_message(pp_span.input)
    if pp_span.output:
        state.trace_output = pp_span.output


def _fill_response_span(pp_span: SpanData, span_data: Any, state: _TraceState) -> None:
    """Extract rich data from a ResponseSpanData."""
    pp_span.input = normalize_response_input(span_data)
    pp_span.output = normalize_response_output(span_data)

    response = getattr(span_data, "response", None)
    if response is not None:
        model = getattr(response, "model", None)
        if model:
            pp_span.model = str(model)
            pp_span.name = str(model)

        usage = getattr(response, "usage", None)
        if usage:
            pp_span.token_usage = extract_token_usage(usage)

        pp_span.model_parameters = extract_response_model_parameters(response)

        reasoning = extract_reasoning(response)
        if reasoning:
            pp_span.metadata["reasoning_summary"] = reasoning

    _propagate_llm_to_parent_agent(pp_span, state)
    _update_chain_and_trace_io(pp_span, state)


def _fill_generation_span(pp_span: SpanData, span_data: Any, state: _TraceState) -> None:
    """Extract data from a GenerationSpanData."""
    pp_span.input = normalize_generation_input(span_data)
    pp_span.output = normalize_generation_output(span_data)

    model = getattr(span_data, "model", None)
    if model:
        pp_span.model = str(model)
        pp_span.name = str(model)

    usage = getattr(span_data, "usage", None)
    if usage:
        pp_span.token_usage = extract_token_usage(usage)

    pp_span.model_parameters = extract_generation_model_parameters(span_data)

    _propagate_llm_to_parent_agent(pp_span, state)
    _update_chain_and_trace_io(pp_span, state)


def _fill_function_span(pp_span: SpanData, span_data: Any) -> None:
    """Extract data from a FunctionSpanData."""
    raw_input = getattr(span_data, "input", None)
    if raw_input is not None:
        pp_span.input = serialize_tool_io(raw_input)

    raw_output = getattr(span_data, "output", None)
    if raw_output is not None:
        pp_span.output = serialize_tool_io(raw_output)

    name = getattr(span_data, "name", None)
    if name:
        pp_span.name = str(name)

    mcp_data = getattr(span_data, "mcp_data", None)
    if mcp_data:
        pp_span.metadata["mcp_data"] = safe_serialize(mcp_data)


def _fill_agent_span(pp_span: SpanData, span_data: Any) -> None:
    """Extract metadata from an AgentSpanData."""
    tools = getattr(span_data, "tools", None)
    if tools:
        pp_span.metadata["tools"] = safe_serialize(tools)

    handoffs = getattr(span_data, "handoffs", None)
    if handoffs:
        pp_span.metadata["handoffs"] = safe_serialize(handoffs)

    output_type = getattr(span_data, "output_type", None)
    if output_type:
        pp_span.metadata["output_type"] = str(output_type)


def _fill_handoff_span(pp_span: SpanData, span_data: Any) -> None:
    """Extract data from a HandoffSpanData."""
    from_agent = getattr(span_data, "from_agent", None)
    to_agent = getattr(span_data, "to_agent", None)

    pp_span.input = {"from_agent": str(from_agent) if from_agent else None}
    pp_span.output = {"to_agent": str(to_agent) if to_agent else None}
    pp_span.metadata["from_agent"] = str(from_agent) if from_agent else None
    pp_span.metadata["to_agent"] = str(to_agent) if to_agent else None


def _fill_guardrail_span(pp_span: SpanData, span_data: Any) -> None:
    """Extract data from a GuardrailSpanData."""
    triggered = getattr(span_data, "triggered", None)
    pp_span.metadata["triggered"] = triggered


def _fill_custom_span(pp_span: SpanData, span_data: Any) -> None:
    """Extract data from a CustomSpanData."""
    data = getattr(span_data, "data", None)
    if data:
        pp_span.metadata["custom_data"] = safe_serialize(data)
