"""OpenAI-specific utilities for the wrap_openai instrumentation."""

from __future__ import annotations

from typing import Any

from pandaprobe.client import get_client
from pandaprobe.schemas import SpanKind
from pandaprobe.tracing.context import get_current_trace
from pandaprobe.tracing.session import get_current_session_id, get_current_user_id
from pandaprobe.validation import extract_last_user_message
from pandaprobe.wrappers._base import safe_serialize


def strip_not_given(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove OpenAI ``NotGiven``/``Omit`` sentinel types from kwargs.

    OpenAI's SDK uses these sentinel types for unset parameters. They must
    be stripped before logging, otherwise captured inputs are polluted with
    non-serialisable objects.
    """
    try:
        from openai._types import NotGiven, Omit

        return {k: v for k, v in kwargs.items() if not isinstance(v, (NotGiven, Omit))}
    except ImportError:
        return kwargs


# ---------------------------------------------------------------------------
# Responses API helpers
# ---------------------------------------------------------------------------

SAFE_RESPONSES_PARAMS: set[str] = {
    "max_output_tokens",
    "temperature",
    "top_p",
    "reasoning",
    "truncation",
    "service_tier",
}


def extract_responses_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pull safe parameters from a Responses API call."""
    return {k: safe_serialize(v) for k, v in kwargs.items() if k in SAFE_RESPONSES_PARAMS}


def normalize_responses_input(cleaned_kwargs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convert Responses API ``input`` + ``instructions`` into standard messages format."""
    messages: list[dict[str, Any]] = []
    instructions = cleaned_kwargs.get("instructions")
    if instructions and isinstance(instructions, str):
        messages.append({"role": "system", "content": instructions})

    inp = cleaned_kwargs.get("input", [])
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict) and "role" in item:
                messages.append(safe_serialize(item))
            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})
            else:
                messages.append(safe_serialize(item))
    return {"messages": messages}


def enter_responses_span(cleaned_kwargs: dict[str, Any], method_name: str = "openai-response"):
    """Open an LLM span for an OpenAI Responses API call.

    Works like :func:`~pandaprobe.wrappers._base.enter_llm_span` but
    normalises the Responses API ``input`` / ``instructions`` parameters
    into the standard messages schema.
    """
    input_data = normalize_responses_input(cleaned_kwargs)
    model_params = extract_responses_params(cleaned_kwargs)
    trace_ctx = get_current_trace()

    if trace_ctx is not None:
        span_ctx = trace_ctx.span(method_name, kind=SpanKind.LLM, model=cleaned_kwargs.get("model"))
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

    span_ctx = standalone.span(method_name, kind=SpanKind.LLM, model=cleaned_kwargs.get("model"))
    span_ctx.__enter__()
    span_ctx.set_input(input_data)
    if model_params:
        span_ctx.set_model_parameters(model_params)
    span_ctx._standalone_trace = standalone  # type: ignore[attr-defined]
    return span_ctx
