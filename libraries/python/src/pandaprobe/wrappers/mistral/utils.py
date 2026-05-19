"""Mistral-specific utilities for the wrap_mistral instrumentation."""

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

SAFE_MISTRAL_PARAMS: set[str] = {
    "temperature",
    "top_p",
    "max_tokens",
    "random_seed",
    "safe_prompt",
    "response_format",
    "tool_choice",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "stop",
}


def strip_unset(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove Mistral ``UNSET`` sentinel values from kwargs.

    The ``mistralai`` SDK (v2.x) exposes an ``UNSET`` sentinel from
    ``mistralai.client.types``; older v1 layouts placed it at
    ``mistralai.utils``. We probe both so this helper works across SDK
    versions, then drop any kwargs whose value is the sentinel so they
    do not pollute span payloads.
    """
    sentinel_types: list[type] = []
    for module_path in ("mistralai.client.types", "mistralai.utils", "mistralai.types"):
        try:
            mod = __import__(module_path, fromlist=["UNSET"])
        except Exception:
            continue
        unset = getattr(mod, "UNSET", None)
        if unset is None:
            continue
        t = type(unset)
        if t not in sentinel_types:
            sentinel_types.append(t)

    if not sentinel_types:
        return kwargs
    sentinels = tuple(sentinel_types)
    return {k: v for k, v in kwargs.items() if not isinstance(v, sentinels)}


def extract_mistral_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pull safe invocation parameters from Mistral call kwargs."""
    return {k: safe_serialize(v) for k, v in kwargs.items() if k in SAFE_MISTRAL_PARAMS}


def normalize_mistral_input(cleaned_kwargs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convert Mistral ``messages`` into the standard universal-schema format.

    Mistral already uses ``[{role, content}]`` natively, so this function only
    applies :func:`safe_serialize` to handle Pydantic model instances that may
    arrive when callsites build messages programmatically (e.g.
    ``UserMessage(content="...")``).
    """
    raw_messages = cleaned_kwargs.get("messages", [])
    serialized = safe_serialize(raw_messages)
    if not isinstance(serialized, list):
        serialized = []
    return {"messages": serialized}


def enter_mistral_span(
    cleaned_kwargs: dict[str, Any],
    method_name: str = "mistral-chat",
):
    """Open an LLM span for a Mistral API call.

    Normalises the ``messages`` parameter into the standard messages schema
    before creating the span. If a parent trace context exists the span is
    nested; otherwise a standalone trace is created.
    """
    input_data = normalize_mistral_input(cleaned_kwargs)
    model_params = extract_mistral_params(cleaned_kwargs)
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
