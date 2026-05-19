"""wrap_mistral — automatic LLM span instrumentation for the Mistral AI client."""

from __future__ import annotations

import functools
import logging
from typing import Any, TypeVar

from pandaprobe.wrappers._base import (
    AsyncStreamReducer,
    SyncStreamReducer,
    close_llm_span,
    error_llm_span,
    safe_serialize,
)
from pandaprobe.wrappers.mistral.utils import enter_mistral_span, strip_unset

logger = logging.getLogger("pandaprobe")

T = TypeVar("T")


def wrap_mistral(client: T) -> T:
    """Instrument a ``mistralai.Mistral`` client for automatic LLM span tracing.

    Monkey-patches ``chat.complete`` / ``chat.complete_async`` (blocking) and
    ``chat.stream`` / ``chat.stream_async`` (streaming) so every call
    automatically produces an LLM span.

    Returns the same client instance (mutated).
    """
    _patch_chat_complete(client)
    _patch_chat_complete_async(client)
    _patch_chat_stream(client)
    _patch_chat_stream_async(client)
    return client


# ---------------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------------


def _patch_chat_complete(client: Any) -> None:
    if not hasattr(client, "chat") or not hasattr(client.chat, "complete"):
        return
    original = client.chat.complete
    client.chat.complete = _sync_blocking_wrapper(original)


def _patch_chat_complete_async(client: Any) -> None:
    if not hasattr(client, "chat") or not hasattr(client.chat, "complete_async"):
        return
    original = client.chat.complete_async
    client.chat.complete_async = _async_blocking_wrapper(original)


def _patch_chat_stream(client: Any) -> None:
    if not hasattr(client, "chat") or not hasattr(client.chat, "stream"):
        return
    original = client.chat.stream
    client.chat.stream = _sync_stream_wrapper(original)


def _patch_chat_stream_async(client: Any) -> None:
    if not hasattr(client, "chat") or not hasattr(client.chat, "stream_async"):
        return
    original = client.chat.stream_async
    client.chat.stream_async = _async_stream_wrapper(original)


# ---------------------------------------------------------------------------
# Sync / async blocking wrappers
# ---------------------------------------------------------------------------


def _sync_blocking_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_unset(kwargs)
        span_ctx = enter_mistral_span(cleaned, "mistral-chat")
        try:
            response = original(*args, **kwargs)
            _finish_mistral_span(span_ctx, response)
            return response
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


def _async_blocking_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_unset(kwargs)
        span_ctx = enter_mistral_span(cleaned, "mistral-chat")
        try:
            response = await original(*args, **kwargs)
            _finish_mistral_span(span_ctx, response)
            return response
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


# ---------------------------------------------------------------------------
# Sync / async streaming wrappers
# ---------------------------------------------------------------------------


def _sync_stream_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_unset(kwargs)
        span_ctx = enter_mistral_span(cleaned, "mistral-chat")
        try:
            stream = original(*args, **kwargs)
            return _MistralSyncStream(stream, span_ctx)
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


def _async_stream_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_unset(kwargs)
        span_ctx = enter_mistral_span(cleaned, "mistral-chat")
        try:
            stream = await original(*args, **kwargs)
            return _MistralAsyncStream(stream, span_ctx)
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


# ---------------------------------------------------------------------------
# Mistral-specific stream reducers
# ---------------------------------------------------------------------------


class _MistralSyncStream(SyncStreamReducer):
    def reduce_chunks(self, span_ctx: Any, chunks: list[Any]) -> None:
        _reduce_mistral_stream(span_ctx, chunks)


class _MistralAsyncStream(AsyncStreamReducer):
    def reduce_chunks(self, span_ctx: Any, chunks: list[Any]) -> None:
        _reduce_mistral_stream(span_ctx, chunks)


def _reduce_mistral_stream(span_ctx: Any, chunks: list[Any]) -> None:
    """Reduce Mistral streaming events into final span data.

    Each event yielded by ``client.chat.stream(...)`` exposes a ``.data``
    attribute holding the actual chunk (a ``CompletionEvent``).  The chunk's
    ``choices[0].delta.content`` carries the text delta and the terminal
    chunk includes ``usage``.
    """
    content_parts: list[str] = []
    model: str | None = None
    usage: dict[str, int] = {}

    for event in chunks:
        chunk = getattr(event, "data", event)

        chunk_model = getattr(chunk, "model", None)
        if chunk_model:
            model = chunk_model

        choices = getattr(chunk, "choices", None) or []
        if choices:
            delta = getattr(choices[0], "delta", None)
            if delta is not None:
                text = getattr(delta, "content", None)
                if isinstance(text, str) and text:
                    content_parts.append(text)

        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = _extract_mistral_usage(chunk_usage) or usage

    if content_parts:
        span_ctx.set_output(
            {"messages": [{"role": "assistant", "content": "".join(content_parts)}]},
        )
    if model:
        span_ctx.set_model(model)
    if usage:
        span_ctx.set_token_usage(**usage)

    close_llm_span(span_ctx)


# ---------------------------------------------------------------------------
# Response extraction helpers
# ---------------------------------------------------------------------------


def _finish_mistral_span(span_ctx: Any, response: Any) -> None:
    """Extract output, model, and token usage from a Mistral chat response."""
    if span_ctx is None:
        return

    try:
        choices = getattr(response, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                serialized = safe_serialize(message)
                if isinstance(serialized, dict):
                    serialized.setdefault("role", "assistant")
                    span_ctx.set_output({"messages": [serialized]})
                else:
                    span_ctx.set_output({"messages": [{"role": "assistant", "content": serialized}]})
    except Exception as exc:
        logger.debug("Error extracting Mistral response output: %s", exc)

    try:
        model = getattr(response, "model", None)
        if model:
            span_ctx.set_model(model)
    except Exception as exc:
        logger.debug("Error extracting Mistral response model: %s", exc)

    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            token_kwargs = _extract_mistral_usage(usage)
            if token_kwargs:
                span_ctx.set_token_usage(**token_kwargs)
    except Exception as exc:
        logger.debug("Error extracting Mistral response usage: %s", exc)

    close_llm_span(span_ctx)


def _extract_mistral_usage(usage: Any) -> dict[str, int]:
    """Map Mistral usage fields onto canonical PandaProbe token usage names.

    Mistral already uses ``prompt_tokens`` / ``completion_tokens`` /
    ``total_tokens`` natively so this is mostly a defensive coercion to
    integers.
    """
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or 0

    out: dict[str, int] = {}
    if isinstance(prompt, int) and prompt > 0:
        out["prompt_tokens"] = prompt
    if isinstance(completion, int) and completion > 0:
        out["completion_tokens"] = completion
    if isinstance(total, int) and total > 0:
        out["total_tokens"] = total
    return out
