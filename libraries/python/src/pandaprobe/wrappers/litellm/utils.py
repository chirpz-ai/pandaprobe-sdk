"""LiteLLM-specific utilities for the wrap_litellm instrumentation."""

from __future__ import annotations

import logging
from typing import Any

from pandaprobe.client import get_client
from pandaprobe.schemas import SpanKind
from pandaprobe.tracing.context import get_current_trace
from pandaprobe.tracing.session import get_current_session_id, get_current_user_id
from pandaprobe.validation import extract_last_user_message
from pandaprobe.wrappers._base import SAFE_INVOCATION_PARAMS, safe_serialize

logger = logging.getLogger("pandaprobe")

# LiteLLM accepts the OpenAI-superset of invocation parameters, so we reuse the
# shared whitelist and add the provider-agnostic reasoning control ``thinking``
# (e.g. Anthropic ``thinking={"type": "enabled", "budget_tokens": 1024}``).
SAFE_LITELLM_PARAMS: set[str] = SAFE_INVOCATION_PARAMS | {"thinking"}


def extract_litellm_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pull safe invocation parameters from LiteLLM call kwargs."""
    return {k: safe_serialize(v) for k, v in kwargs.items() if k in SAFE_LITELLM_PARAMS}


def normalize_litellm_input(cleaned_kwargs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convert LiteLLM ``messages`` into the standard universal-schema format.

    LiteLLM uses OpenAI-format ``[{role, content}]`` messages natively, so this
    only applies :func:`safe_serialize` to coerce any non-JSON-safe entries
    (e.g. Pydantic message models) before they enter a span payload.
    """
    raw_messages = cleaned_kwargs.get("messages", [])
    serialized = safe_serialize(raw_messages)
    if not isinstance(serialized, list):
        serialized = []
    return {"messages": serialized}


def enter_litellm_span(
    cleaned_kwargs: dict[str, Any],
    method_name: str = "litellm-chat",
):
    """Open an LLM span for a LiteLLM ``completion`` / ``acompletion`` call.

    Normalises the ``messages`` parameter into the standard messages schema
    before creating the span. If a parent trace context exists the span is
    nested; otherwise a standalone trace is created.
    """
    input_data = normalize_litellm_input(cleaned_kwargs)
    model_params = extract_litellm_params(cleaned_kwargs)
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
