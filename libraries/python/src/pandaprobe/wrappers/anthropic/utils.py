"""Anthropic-specific utilities for the wrap_anthropic instrumentation."""

from __future__ import annotations

import logging
from typing import Any

from pandaprobe.client import get_client
from pandaprobe.schemas import SpanKind
from pandaprobe.tracing.context import get_current_trace
from pandaprobe.tracing.session import get_current_session_id, get_current_user_id
from pandaprobe.validation import extract_last_user_message
from pandaprobe.wrappers._base import safe_serialize

logger = logging.getLogger("pandaprobe")

SAFE_ANTHROPIC_PARAMS: set[str] = {
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "stop_sequences",
    "thinking",
}


def strip_not_given(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove Anthropic ``NotGiven``/``Omit`` sentinel types from kwargs.

    The Anthropic SDK uses these sentinel types for unset parameters.
    They must be stripped before logging to avoid non-serialisable objects.
    """
    try:
        from anthropic._types import NotGiven, Omit

        return {k: v for k, v in kwargs.items() if not isinstance(v, (NotGiven, Omit))}
    except ImportError:
        return kwargs


def extract_anthropic_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pull safe invocation parameters from Anthropic call kwargs."""
    return {k: v for k, v in kwargs.items() if k in SAFE_ANTHROPIC_PARAMS}


def normalize_anthropic_input(cleaned_kwargs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convert Anthropic ``system`` + ``messages`` into the standard messages format.

    Anthropic passes ``system`` as a separate top-level kwarg rather than
    inside the ``messages`` list.  This function prepends it as a system
    message so the trace conforms to our universal schema.
    """
    messages: list[dict[str, Any]] = []

    system = cleaned_kwargs.get("system")
    if system is not None:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            messages.append({"role": "system", "content": safe_serialize(system)})

    raw_messages = cleaned_kwargs.get("messages", [])
    serialized = safe_serialize(raw_messages)
    if isinstance(serialized, list):
        messages.extend(serialized)

    return {"messages": messages}


def enter_anthropic_span(
    cleaned_kwargs: dict[str, Any],
    method_name: str = "anthropic-messages",
):
    """Open an LLM span for an Anthropic API call.

    Normalises the ``system`` / ``messages`` parameters into the standard
    messages schema before creating the span.
    """
    input_data = normalize_anthropic_input(cleaned_kwargs)
    model_params = extract_anthropic_params(cleaned_kwargs)
    trace_ctx = get_current_trace()

    if trace_ctx is not None:
        span_ctx = trace_ctx.span(
            method_name,
            kind=SpanKind.LLM,
            model=cleaned_kwargs.get("model"),
        )
        span_ctx.__enter__()
        span_ctx.set_input(input_data)
        if model_params:
            span_ctx.set_model_parameters(model_params)
        return span_ctx

    client = get_client()
    if client is None or not client.enabled:
        return None

    standalone = client.trace(
        method_name,
        input=extract_last_user_message(input_data),
        session_id=get_current_session_id(),
        user_id=get_current_user_id(),
    )
    standalone.__enter__()

    span_ctx = standalone.span(
        method_name,
        kind=SpanKind.LLM,
        model=cleaned_kwargs.get("model"),
    )
    span_ctx.__enter__()
    span_ctx.set_input(input_data)
    if model_params:
        span_ctx.set_model_parameters(model_params)
    span_ctx._standalone_trace = standalone  # type: ignore[attr-defined]
    return span_ctx
