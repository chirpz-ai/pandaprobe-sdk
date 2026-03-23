"""Gemini-specific utilities for the wrap_gemini instrumentation."""

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

SAFE_GEMINI_PARAMS: set[str] = {
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
    "stop_sequences",
    "candidate_count",
    "presence_penalty",
    "frequency_penalty",
}


def convert_config_to_dict(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert a ``GenerateContentConfig`` object to a plain dict.

    Also strips ``None`` values from the resulting config dict so that
    only explicitly-set parameters are captured.
    """
    config = kwargs.get("config")
    if config is None or isinstance(config, dict):
        return kwargs

    kwargs = dict(kwargs)
    try:
        if hasattr(config, "model_dump"):
            kwargs["config"] = config.model_dump(exclude_none=True)
        elif hasattr(config, "__dict__"):
            kwargs["config"] = {k: v for k, v in vars(config).items() if not k.startswith("_") and v is not None}
        else:
            kwargs["config"] = {}
    except Exception:
        kwargs["config"] = {}
    return kwargs


def extract_gemini_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pull safe generation parameters from kwargs and nested config."""
    params: dict[str, Any] = {}
    config = kwargs.get("config")
    if isinstance(config, dict):
        for k, v in config.items():
            if k in SAFE_GEMINI_PARAMS:
                params[k] = safe_serialize(v)
    for k, v in kwargs.items():
        if k in SAFE_GEMINI_PARAMS:
            params[k] = safe_serialize(v)
    return params


def _to_dict(obj: Any) -> Any:
    """Best-effort conversion of a SDK object to a dict."""
    if isinstance(obj, (dict, str)):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def _normalize_role(role: str) -> str:
    """Normalize Gemini's ``model`` role to ``assistant``."""
    if role == "model":
        return "assistant"
    return role


def normalize_gemini_input(cleaned_kwargs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convert Gemini ``contents`` + config system instruction into standard messages format."""
    messages: list[dict[str, Any]] = []

    config = cleaned_kwargs.get("config")
    if isinstance(config, dict):
        sys_inst = config.get("system_instruction")
        if sys_inst:
            sys_inst = _to_dict(sys_inst)
            if isinstance(sys_inst, str):
                messages.append({"role": "system", "content": sys_inst})
            elif isinstance(sys_inst, dict) and "parts" in sys_inst:
                text = _extract_text_from_parts(sys_inst["parts"])
                if text:
                    messages.append({"role": "system", "content": text})
            elif isinstance(sys_inst, list):
                text = _extract_text_from_parts(sys_inst)
                if text:
                    messages.append({"role": "system", "content": text})

    contents = cleaned_kwargs.get("contents")
    if contents is None:
        return {"messages": messages}

    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
        return {"messages": messages}

    if isinstance(contents, list):
        if all(isinstance(item, str) for item in contents):
            for item in contents:
                messages.append({"role": "user", "content": item})
            return {"messages": messages}

        for content in contents:
            content = _to_dict(content)
            if not isinstance(content, dict):
                continue
            role = _normalize_role(content.get("role", "user"))
            raw_parts = content.get("parts", [])
            text = _extract_text_from_parts(raw_parts)
            messages.append({"role": role, "content": text or ""})

    return {"messages": messages}


def _extract_text_from_parts(parts: Any) -> str:
    """Join text fragments from a list of Gemini ``Part`` objects or dicts."""
    if not isinstance(parts, list):
        return ""
    texts: list[str] = []
    for part in parts:
        part = _to_dict(part)
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict):
            text = part.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts) if texts else ""


def enter_gemini_span(
    cleaned_kwargs: dict[str, Any],
    method_name: str = "gemini-generate",
):
    """Open an LLM span for a Gemini API call.

    Normalises the ``contents`` / config parameters into the standard
    messages schema before creating the span.
    """
    input_data = normalize_gemini_input(cleaned_kwargs)
    model_params = extract_gemini_params(cleaned_kwargs)
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
