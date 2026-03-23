"""Shared, provider-agnostic utilities for all LLM client wrappers.

Provider-specific logic belongs in its own sub-package under
``wrappers/<provider>/``.  For example, OpenAI helpers live in
``wrappers/openai/utils.py``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pandaprobe.client import get_client
from pandaprobe.schemas import SpanKind
from pandaprobe.tracing.context import get_current_trace
from pandaprobe.tracing.session import get_current_session_id, get_current_user_id
from pandaprobe.validation import extract_last_user_message

logger = logging.getLogger("pandaprobe")

# ---------------------------------------------------------------------------
# Safe parameter whitelists
# ---------------------------------------------------------------------------

SAFE_INVOCATION_PARAMS: set[str] = {
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "n",
    "response_format",
    "stop",
    "logprobs",
    "top_logprobs",
    "reasoning_effort",
    "stream_options",
    "service_tier",
}


def extract_model_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pull only safe invocation parameters from the call kwargs."""
    return {k: v for k, v in kwargs.items() if k in SAFE_INVOCATION_PARAMS}


def safe_serialize(obj: Any) -> Any:
    """Best-effort JSON-safe serialisation of an arbitrary object."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: safe_serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return repr(obj)


# ---------------------------------------------------------------------------
# Span lifecycle helpers
# ---------------------------------------------------------------------------


def enter_llm_span(cleaned_kwargs: dict[str, Any], method_name: str):
    """Open an LLM span, creating a standalone trace if none is active.

    This is the shared entry-point used by every wrapper provider.
    Returns a SpanContext (or None if the SDK is disabled).
    """
    input_data = safe_serialize({"messages": cleaned_kwargs.get("messages", [])})
    model_params = extract_model_params(cleaned_kwargs)
    trace_ctx = get_current_trace()

    if trace_ctx is not None:
        span_ctx = trace_ctx.span(method_name, kind=SpanKind.LLM, model=cleaned_kwargs.get("model"))
        span_ctx.__enter__()
        span_ctx.set_input(input_data)
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
    span_ctx.set_model_parameters(model_params)
    span_ctx._standalone_trace = standalone  # type: ignore[attr-defined]
    return span_ctx


def close_llm_span(span_ctx: Any) -> None:
    """Exit the span (and standalone trace if applicable)."""
    if span_ctx is None:
        return
    span_ctx.__exit__(None, None, None)
    standalone = getattr(span_ctx, "_standalone_trace", None)
    if standalone is not None:
        standalone.set_output(span_ctx._output)
        standalone.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Stream reducer base classes
# ---------------------------------------------------------------------------


class SyncStreamReducer:
    """Wraps a sync streaming response, collecting chunks for the span.

    Subclasses can override :meth:`extract_chunk_data` for provider-specific
    chunk formats.
    """

    def __init__(self, stream: Any, span_ctx: Any) -> None:
        self._stream = stream
        self._span_ctx = span_ctx
        self._chunks: list[Any] = []
        self._first_chunk = True

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
            if self._first_chunk and self._span_ctx:
                self._span_ctx.set_completion_start_time(datetime.now(timezone.utc))
                self._first_chunk = False
            self._chunks.append(chunk)
            return chunk
        except StopIteration:
            self._finalize()
            raise

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args):
        self._finalize()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*args)
        return None

    def _finalize(self) -> None:
        if self._span_ctx is None:
            return
        try:
            self.reduce_chunks(self._span_ctx, self._chunks)
        except Exception:
            pass
        self._span_ctx = None

    def reduce_chunks(self, span_ctx: Any, chunks: list[Any]) -> None:
        """Override in subclass to reduce provider-specific chunks."""
        close_llm_span(span_ctx)


class AsyncStreamReducer:
    """Wraps an async streaming response, collecting chunks for the span."""

    def __init__(self, stream: Any, span_ctx: Any) -> None:
        self._stream = stream
        self._span_ctx = span_ctx
        self._chunks: list[Any] = []
        self._first_chunk = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
            if self._first_chunk and self._span_ctx:
                self._span_ctx.set_completion_start_time(datetime.now(timezone.utc))
                self._first_chunk = False
            self._chunks.append(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize()
            raise

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        self._finalize()
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(*args)
        return None

    def _finalize(self) -> None:
        if self._span_ctx is None:
            return
        try:
            self.reduce_chunks(self._span_ctx, self._chunks)
        except Exception:
            pass
        self._span_ctx = None

    def reduce_chunks(self, span_ctx: Any, chunks: list[Any]) -> None:
        """Override in subclass to reduce provider-specific chunks."""
        close_llm_span(span_ctx)
