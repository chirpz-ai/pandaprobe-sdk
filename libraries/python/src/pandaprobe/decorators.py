"""@trace and @span decorators for automatic instrumentation."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any, Callable, TypeVar

from pandaprobe.client import get_client
from pandaprobe.schemas import SpanKind
from pandaprobe.tracing.context import get_current_trace
from pandaprobe.validation import (
    extract_last_assistant_message,
    extract_last_user_message,
    validate_span_input,
    validate_span_output,
    validate_trace_input,
    validate_trace_output,
)

logger = logging.getLogger("pandaprobe")

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# @trace
# ---------------------------------------------------------------------------


def trace(
    _fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Decorator that wraps a function in a :class:`TraceContext`.

    The function's arguments become the trace *input* and the return value
    becomes the trace *output*.  Works with both sync and async functions.

    Trace input should contain only the current turn's user message in
    ``{"messages": [{"role": "user", "content": "..."}]}`` format.
    Output should be ``{"messages": [{"role": "assistant", "content": "..."}]}``.

    Can be used with or without parentheses::

        @pandaprobe.trace
        def my_func(query): ...

        @pandaprobe.trace(name="custom-name")
        def my_func(query): ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        trace_name = name or fn.__name__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                client = get_client()
                if client is None or not client.enabled:
                    return await fn(*args, **kwargs)

                fn_input = _capture_input(fn, args, kwargs)
                validate_trace_input(fn_input)
                trace_input = extract_last_user_message(fn_input)
                async with client.trace(
                    trace_name,
                    input=trace_input,
                    session_id=session_id,
                    user_id=user_id,
                    tags=tags,
                    metadata=metadata,
                ) as ctx:
                    result = await fn(*args, **kwargs)
                    validate_trace_output(result)
                    ctx.set_output(extract_last_assistant_message(result))
                    return result

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            client = get_client()
            if client is None or not client.enabled:
                return fn(*args, **kwargs)

            fn_input = _capture_input(fn, args, kwargs)
            validate_trace_input(fn_input)
            trace_input = extract_last_user_message(fn_input)
            with client.trace(
                trace_name,
                input=trace_input,
                session_id=session_id,
                user_id=user_id,
                tags=tags,
                metadata=metadata,
            ) as ctx:
                result = fn(*args, **kwargs)
                validate_trace_output(result)
                ctx.set_output(extract_last_assistant_message(result))
                return result

        return sync_wrapper

    if _fn is not None:
        return decorator(_fn)
    return decorator


# ---------------------------------------------------------------------------
# @span
# ---------------------------------------------------------------------------


def span(
    _fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    kind: str | SpanKind = SpanKind.OTHER,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Decorator that wraps a function in a :class:`SpanContext`.

    For LLM spans (kind="LLM"), input/output must follow the messages
    schema: ``{"messages": [{"role": "...", "content": "..."}]}``.
    Other span kinds accept arbitrary input/output.

    Automatically parents to the current trace / enclosing span.
    Works with both sync and async functions.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or fn.__name__
        resolved_kind = SpanKind(kind) if isinstance(kind, str) else kind
        is_llm = resolved_kind == SpanKind.LLM

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                trace_ctx = get_current_trace()
                if trace_ctx is None:
                    return await fn(*args, **kwargs)

                fn_input = _capture_input(fn, args, kwargs)
                if is_llm:
                    validate_span_input(fn_input)
                async with trace_ctx.span(span_name, kind=kind, model=model, metadata=metadata) as s:
                    s.set_input(fn_input)
                    result = await fn(*args, **kwargs)
                    if is_llm:
                        validate_span_output(result)
                    s.set_output(result)
                    return result

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace_ctx = get_current_trace()
            if trace_ctx is None:
                return fn(*args, **kwargs)

            fn_input = _capture_input(fn, args, kwargs)
            if is_llm:
                validate_span_input(fn_input)
            with trace_ctx.span(span_name, kind=kind, model=model, metadata=metadata) as s:
                s.set_input(fn_input)
                result = fn(*args, **kwargs)
                if is_llm:
                    validate_span_output(result)
                s.set_output(result)
                return result

        return sync_wrapper

    if _fn is not None:
        return decorator(_fn)
    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_input(fn: Callable[..., Any], args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-friendly dict from the function's call arguments."""
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return {k: _safe(v) for k, v in bound.arguments.items()}
    except Exception:
        return {"args": [_safe(a) for a in args], "kwargs": {k: _safe(v) for k, v in kwargs.items()}}


def _safe(value: Any) -> Any:
    """Ensure a value is JSON-friendly."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return repr(value)
