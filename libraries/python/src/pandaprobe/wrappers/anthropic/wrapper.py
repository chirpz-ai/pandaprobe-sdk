"""wrap_anthropic — automatic LLM span instrumentation for the Anthropic client."""

from __future__ import annotations

import functools
import logging
from datetime import datetime, timezone
from typing import Any, TypeVar

from pandaprobe.wrappers._base import (
    AsyncStreamReducer,
    SyncStreamReducer,
    close_llm_span,
    error_llm_span,
    safe_serialize,
)
from pandaprobe.wrappers.anthropic.utils import enter_anthropic_span, strip_not_given

logger = logging.getLogger("pandaprobe")

T = TypeVar("T")


def wrap_anthropic(client: T) -> T:
    """Instrument an ``anthropic.Anthropic`` or ``anthropic.AsyncAnthropic`` client.

    Monkey-patches ``messages.create`` and ``messages.stream`` so every call
    automatically produces an LLM span.

    Returns the same client instance (mutated).
    """
    _patch_messages_create(client)
    _patch_messages_stream(client)
    return client


# ---------------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------------


def _patch_messages_create(client: Any) -> None:
    if not hasattr(client, "messages") or not hasattr(client.messages, "create"):
        return

    import asyncio

    original = client.messages.create
    if asyncio.iscoroutinefunction(original):
        client.messages.create = _async_create_wrapper(original)
    else:
        client.messages.create = _sync_create_wrapper(original)


def _patch_messages_stream(client: Any) -> None:
    if not hasattr(client, "messages") or not hasattr(client.messages, "stream"):
        return

    import asyncio

    original = client.messages.stream
    if asyncio.iscoroutinefunction(original):
        client.messages.stream = _async_stream_manager_factory(original)
    else:
        client.messages.stream = _sync_stream_manager_factory(original)


# ---------------------------------------------------------------------------
# Sync wrappers — messages.create
# ---------------------------------------------------------------------------


def _sync_create_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_not_given(kwargs)
        is_stream = cleaned.get("stream", False)

        if is_stream:
            return _sync_streaming_create(original, args, kwargs, cleaned)
        return _sync_blocking_create(original, args, kwargs, cleaned)

    return wrapper


def _sync_blocking_create(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_anthropic_span(cleaned, "anthropic-messages")
    try:
        response = original(*args, **kwargs)
        _finish_anthropic_span(span_ctx, response)
        return response
    except Exception as exc:
        error_llm_span(span_ctx, exc)
        raise


def _sync_streaming_create(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_anthropic_span(cleaned, "anthropic-messages")
    try:
        stream = original(*args, **kwargs)
        return _AnthropicSyncStream(stream, span_ctx)
    except Exception as exc:
        error_llm_span(span_ctx, exc)
        raise


# ---------------------------------------------------------------------------
# Async wrappers — messages.create
# ---------------------------------------------------------------------------


def _async_create_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_not_given(kwargs)
        is_stream = cleaned.get("stream", False)

        if is_stream:
            return await _async_streaming_create(original, args, kwargs, cleaned)
        return await _async_blocking_create(original, args, kwargs, cleaned)

    return wrapper


async def _async_blocking_create(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_anthropic_span(cleaned, "anthropic-messages")
    try:
        response = await original(*args, **kwargs)
        _finish_anthropic_span(span_ctx, response)
        return response
    except Exception as exc:
        error_llm_span(span_ctx, exc)
        raise


async def _async_streaming_create(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_anthropic_span(cleaned, "anthropic-messages")
    try:
        stream = await original(*args, **kwargs)
        return _AnthropicAsyncStream(stream, span_ctx)
    except Exception as exc:
        error_llm_span(span_ctx, exc)
        raise


# ---------------------------------------------------------------------------
# Sync/Async stream context manager wrappers — messages.stream
# ---------------------------------------------------------------------------


def _sync_stream_manager_factory(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_not_given(kwargs)
        return _SyncStreamManager(original, args, kwargs, cleaned)

    return wrapper


class _SyncStreamManager:
    """Wraps Anthropic's ``messages.stream()`` sync context manager."""

    def __init__(self, original: Any, args: Any, kwargs: Any, cleaned: dict[str, Any]) -> None:
        self._original = original
        self._args = args
        self._kwargs = kwargs
        self._cleaned = cleaned
        self._span_ctx: Any = None
        self._inner_manager: Any = None
        self._inner_stream: Any = None

    def __enter__(self):
        self._span_ctx = enter_anthropic_span(self._cleaned, "anthropic-messages")
        try:
            self._inner_manager = self._original(*self._args, **self._kwargs)
            self._inner_stream = self._inner_manager.__enter__()
            return _SyncStreamWrapper(self._inner_stream, self._span_ctx, owner=self)
        except Exception as exc:
            error_llm_span(self._span_ctx, exc)
            self._span_ctx = None
            raise

    def __exit__(self, *exc_info):
        try:
            if self._inner_manager is not None:
                self._inner_manager.__exit__(*exc_info)
        finally:
            if self._span_ctx is not None:
                if exc_info[0] is not None:
                    error_llm_span(self._span_ctx, exc_info[1])
                else:
                    close_llm_span(self._span_ctx)
                self._span_ctx = None


class _SyncStreamWrapper:
    """Wraps the inner ``MessageStream`` to capture text and finalize the span."""

    def __init__(self, stream: Any, span_ctx: Any, owner: Any = None) -> None:
        self._stream = stream
        self._span_ctx = span_ctx
        self._owner = owner
        self._first_text = True

    @property
    def text_stream(self):
        return self._wrap_text_stream()

    def _wrap_text_stream(self):
        for text in self._stream.text_stream:
            if self._first_text and self._span_ctx:
                self._span_ctx.set_completion_start_time(datetime.now(timezone.utc))
                self._first_text = False
            yield text
        self._extract_final()

    def _extract_final(self) -> None:
        if self._span_ctx is None:
            return
        try:
            final = self._stream.get_final_message()
            _finish_anthropic_span(self._span_ctx, final)
            self._span_ctx = None
            if self._owner is not None:
                self._owner._span_ctx = None
        except Exception:
            pass

    def get_final_message(self):
        return self._stream.get_final_message()

    def get_final_text(self):
        return self._stream.get_final_text()

    def __iter__(self):
        return self._stream.__iter__()

    def __next__(self):
        return next(self._stream)


def _async_stream_manager_factory(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_not_given(kwargs)
        return _AsyncStreamManager(original, args, kwargs, cleaned)

    return wrapper


class _AsyncStreamManager:
    """Wraps Anthropic's ``messages.stream()`` async context manager."""

    def __init__(self, original: Any, args: Any, kwargs: Any, cleaned: dict[str, Any]) -> None:
        self._original = original
        self._args = args
        self._kwargs = kwargs
        self._cleaned = cleaned
        self._span_ctx: Any = None
        self._inner_manager: Any = None
        self._inner_stream: Any = None

    async def __aenter__(self):
        self._span_ctx = enter_anthropic_span(self._cleaned, "anthropic-messages")
        try:
            self._inner_manager = self._original(*self._args, **self._kwargs)
            self._inner_stream = await self._inner_manager.__aenter__()
            return _AsyncStreamWrapper(self._inner_stream, self._span_ctx, owner=self)
        except Exception as exc:
            error_llm_span(self._span_ctx, exc)
            self._span_ctx = None
            raise

    async def __aexit__(self, *exc_info):
        try:
            if self._inner_manager is not None:
                await self._inner_manager.__aexit__(*exc_info)
        finally:
            if self._span_ctx is not None:
                if exc_info[0] is not None:
                    error_llm_span(self._span_ctx, exc_info[1])
                else:
                    close_llm_span(self._span_ctx)
                self._span_ctx = None


class _AsyncStreamWrapper:
    """Wraps the inner ``AsyncMessageStream`` to capture text and finalize the span."""

    def __init__(self, stream: Any, span_ctx: Any, owner: Any = None) -> None:
        self._stream = stream
        self._span_ctx = span_ctx
        self._owner = owner
        self._first_text = True

    @property
    def text_stream(self):
        return self._wrap_text_stream()

    async def _wrap_text_stream(self):
        async for text in self._stream.text_stream:
            if self._first_text and self._span_ctx:
                self._span_ctx.set_completion_start_time(datetime.now(timezone.utc))
                self._first_text = False
            yield text
        await self._extract_final()

    async def _extract_final(self) -> None:
        if self._span_ctx is None:
            return
        try:
            final = await self._stream.get_final_message()
            _finish_anthropic_span(self._span_ctx, final)
            self._span_ctx = None
            if self._owner is not None:
                self._owner._span_ctx = None
        except Exception:
            pass

    async def get_final_message(self):
        return await self._stream.get_final_message()

    async def get_final_text(self):
        return await self._stream.get_final_text()

    def __aiter__(self):
        return self._stream.__aiter__()

    async def __anext__(self):
        return await self._stream.__anext__()


# ---------------------------------------------------------------------------
# Anthropic-specific stream reducers (for stream=True on messages.create)
# ---------------------------------------------------------------------------


class _AnthropicSyncStream(SyncStreamReducer):
    def reduce_chunks(self, span_ctx: Any, chunks: list[Any]) -> None:
        _reduce_anthropic_stream(span_ctx, chunks)


class _AnthropicAsyncStream(AsyncStreamReducer):
    def reduce_chunks(self, span_ctx: Any, chunks: list[Any]) -> None:
        _reduce_anthropic_stream(span_ctx, chunks)


def _reduce_anthropic_stream(span_ctx: Any, chunks: list[Any]) -> None:
    """Reduce Anthropic streaming events into final span data."""
    accumulated = _try_accumulate(chunks)
    if accumulated is not None:
        _finish_anthropic_span(span_ctx, accumulated)
        return

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    model: str | None = None
    usage: dict[str, int] | None = None

    for event in chunks:
        event_type = getattr(event, "type", "")

        if event_type == "message_start":
            msg = getattr(event, "message", None)
            if msg:
                model = getattr(msg, "model", None)
                u = getattr(msg, "usage", None)
                if u:
                    usage = {"prompt_tokens": getattr(u, "input_tokens", 0) or 0}

        elif event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta:
                delta_type = getattr(delta, "type", "")
                if delta_type == "thinking_delta":
                    thinking = getattr(delta, "thinking", None)
                    if thinking:
                        thinking_parts.append(thinking)
                elif delta_type == "text_delta":
                    text = getattr(delta, "text", None)
                    if text:
                        text_parts.append(text)
                else:
                    text = getattr(delta, "text", None)
                    if text:
                        text_parts.append(text)

        elif event_type == "message_delta":
            u = getattr(event, "usage", None)
            if u:
                output_tokens = getattr(u, "output_tokens", 0) or 0
                if usage is not None:
                    usage["completion_tokens"] = output_tokens
                else:
                    usage = {"prompt_tokens": 0, "completion_tokens": output_tokens}

    if text_parts:
        span_ctx.set_output(
            {"messages": [{"role": "assistant", "content": "".join(text_parts)}]},
        )
    if thinking_parts:
        span_ctx.set_metadata({"reasoning_summary": "\n\n".join(thinking_parts)})
    if model:
        span_ctx.set_model(model)
    if usage:
        span_ctx.set_token_usage(**usage)

    close_llm_span(span_ctx)


def _try_accumulate(chunks: list[Any]) -> Any:
    """Try to reconstruct a full Message using Anthropic's SDK accumulator."""
    try:
        from anthropic.lib.streaming._messages import accumulate_event
    except ImportError:
        return None

    message = None
    for event in chunks:
        try:
            message = accumulate_event(event=event, current_snapshot=message)
        except Exception:
            return None
    return message


# ---------------------------------------------------------------------------
# Response extraction helpers
# ---------------------------------------------------------------------------


def _finish_anthropic_span(span_ctx: Any, response: Any) -> None:
    """Extract output, model, thinking summary, and usage from an Anthropic Message response."""
    if span_ctx is None:
        return

    try:
        text_parts, thinking_parts = _split_content_blocks(response)
        if text_parts:
            span_ctx.set_output(
                {"messages": [{"role": "assistant", "content": "".join(text_parts)}]},
            )
        elif not thinking_parts:
            serialized = safe_serialize(response)
            if isinstance(serialized, dict):
                serialized.pop("type", None)
                span_ctx.set_output({"messages": [serialized]})
            else:
                span_ctx.set_output({"messages": [{"role": "assistant", "content": serialized}]})
        if thinking_parts:
            span_ctx.set_metadata({"reasoning_summary": "\n\n".join(thinking_parts)})
    except Exception as exc:
        logger.debug("Error extracting Anthropic response output: %s", exc)

    try:
        model = getattr(response, "model", None)
        if model:
            span_ctx.set_model(model)
    except Exception as exc:
        logger.debug("Error extracting Anthropic response model: %s", exc)

    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            token_kwargs = _extract_anthropic_usage(usage)
            if token_kwargs:
                span_ctx.set_token_usage(**token_kwargs)
    except Exception as exc:
        logger.debug("Error extracting Anthropic response usage: %s", exc)

    close_llm_span(span_ctx)


def _split_content_blocks(response: Any) -> tuple[list[str], list[str]]:
    """Separate text content from thinking summaries in an Anthropic response.

    Returns ``(text_parts, thinking_parts)``.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []

    content = getattr(response, "content", None)
    if not content:
        return text_parts, thinking_parts

    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "thinking":
            thinking = getattr(block, "thinking", None)
            if thinking:
                thinking_parts.append(thinking)
        elif block_type == "text":
            text = getattr(block, "text", None)
            if text:
                text_parts.append(text)

    return text_parts, thinking_parts


def _extract_anthropic_usage(usage: Any) -> dict[str, int] | None:
    """Map Anthropic usage fields to our standard token usage dict."""
    if usage is None:
        return None

    prompt = getattr(usage, "input_tokens", 0) or 0
    completion = getattr(usage, "output_tokens", 0) or 0
    extra: dict[str, int] = {}

    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    if cache_read > 0:
        extra["cache_read_tokens"] = cache_read

    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    if cache_creation > 0:
        extra["cache_creation_tokens"] = cache_creation

    if prompt == 0 and completion == 0 and not extra:
        return None

    return {"prompt_tokens": prompt, "completion_tokens": completion, **extra}
