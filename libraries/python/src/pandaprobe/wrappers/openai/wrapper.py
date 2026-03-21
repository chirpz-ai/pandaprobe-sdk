"""wrap_openai — automatic LLM span instrumentation for the OpenAI client."""

from __future__ import annotations

import functools
import logging
from typing import Any, TypeVar

from pandaprobe.wrappers._base import (
    AsyncStreamReducer,
    SyncStreamReducer,
    close_llm_span,
    enter_llm_span,
    safe_serialize,
)
from pandaprobe.wrappers.openai.utils import strip_not_given

logger = logging.getLogger("pandaprobe")

T = TypeVar("T")


def wrap_openai(client: T) -> T:
    """Instrument an ``openai.OpenAI`` or ``openai.AsyncOpenAI`` client.

    Monkey-patches ``chat.completions.create`` (and ``completions.create``
    when present) so every call automatically produces an LLM span.

    Returns the same client instance (mutated).
    """
    _patch_chat_completions(client)
    _patch_legacy_completions(client)
    return client


# ---------------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------------


def _patch_chat_completions(client: Any) -> None:
    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        return
    completions = client.chat.completions
    if not hasattr(completions, "create"):
        return

    import asyncio

    original = completions.create
    if asyncio.iscoroutinefunction(original):
        completions.create = _async_chat_wrapper(original)
    else:
        completions.create = _sync_chat_wrapper(original)


def _patch_legacy_completions(client: Any) -> None:
    if not hasattr(client, "completions") or not hasattr(client.completions, "create"):
        return
    if hasattr(client, "chat") and client.completions is getattr(client.chat, "completions", None):
        return

    import asyncio

    original = client.completions.create
    if asyncio.iscoroutinefunction(original):
        client.completions.create = _async_legacy_wrapper(original)
    else:
        client.completions.create = _sync_legacy_wrapper(original)


# ---------------------------------------------------------------------------
# Sync wrappers
# ---------------------------------------------------------------------------


def _sync_chat_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_not_given(kwargs)
        is_stream = cleaned.get("stream", False)

        if is_stream:
            return _sync_streaming_chat(original, args, kwargs, cleaned)
        return _sync_blocking_chat(original, args, kwargs, cleaned)

    return wrapper


def _sync_blocking_chat(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_llm_span(cleaned, "openai-chat")
    try:
        response = original(*args, **kwargs)
        actual_response = _maybe_parse_raw(response)
        _finish_span_from_chat_response(span_ctx, actual_response)
        return response
    except Exception as exc:
        if span_ctx:
            span_ctx.set_error(str(exc))
            span_ctx.__exit__(type(exc), exc, exc.__traceback__)
        raise


def _sync_streaming_chat(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_llm_span(cleaned, "openai-chat")
    try:
        stream = original(*args, **kwargs)
        return _OpenAISyncStream(stream, span_ctx)
    except Exception as exc:
        if span_ctx:
            span_ctx.set_error(str(exc))
            span_ctx.__exit__(type(exc), exc, exc.__traceback__)
        raise


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


def _async_chat_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_not_given(kwargs)
        is_stream = cleaned.get("stream", False)

        if is_stream:
            return await _async_streaming_chat(original, args, kwargs, cleaned)
        return await _async_blocking_chat(original, args, kwargs, cleaned)

    return wrapper


async def _async_blocking_chat(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_llm_span(cleaned, "openai-chat")
    try:
        response = await original(*args, **kwargs)
        actual_response = _maybe_parse_raw(response)
        _finish_span_from_chat_response(span_ctx, actual_response)
        return response
    except Exception as exc:
        if span_ctx:
            span_ctx.set_error(str(exc))
            span_ctx.__exit__(type(exc), exc, exc.__traceback__)
        raise


async def _async_streaming_chat(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_llm_span(cleaned, "openai-chat")
    try:
        stream = await original(*args, **kwargs)
        return _OpenAIAsyncStream(stream, span_ctx)
    except Exception as exc:
        if span_ctx:
            span_ctx.set_error(str(exc))
            span_ctx.__exit__(type(exc), exc, exc.__traceback__)
        raise


# ---------------------------------------------------------------------------
# Legacy completions
# ---------------------------------------------------------------------------


def _sync_legacy_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_not_given(kwargs)
        span_ctx = enter_llm_span(cleaned, "openai-completion", input_key="prompt")
        try:
            response = original(*args, **kwargs)
            actual_response = _maybe_parse_raw(response)
            _finish_span_legacy(span_ctx, actual_response)
            return response
        except Exception as exc:
            if span_ctx:
                span_ctx.set_error(str(exc))
                span_ctx.__exit__(type(exc), exc, exc.__traceback__)
            raise

    return wrapper


def _async_legacy_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_not_given(kwargs)
        span_ctx = enter_llm_span(cleaned, "openai-completion", input_key="prompt")
        try:
            response = await original(*args, **kwargs)
            actual_response = _maybe_parse_raw(response)
            _finish_span_legacy(span_ctx, actual_response)
            return response
        except Exception as exc:
            if span_ctx:
                span_ctx.set_error(str(exc))
                span_ctx.__exit__(type(exc), exc, exc.__traceback__)
            raise

    return wrapper


# ---------------------------------------------------------------------------
# OpenAI-specific stream reducers
# ---------------------------------------------------------------------------


class _OpenAISyncStream(SyncStreamReducer):
    def reduce_chunks(self, span_ctx: Any, chunks: list[Any]) -> None:
        _reduce_openai_stream(span_ctx, chunks)


class _OpenAIAsyncStream(AsyncStreamReducer):
    def reduce_chunks(self, span_ctx: Any, chunks: list[Any]) -> None:
        _reduce_openai_stream(span_ctx, chunks)


def _reduce_openai_stream(span_ctx: Any, chunks: list[Any]) -> None:
    """Reduce OpenAI streaming chunks into final span data."""
    content_parts: list[str] = []
    model: str | None = None
    usage: dict[str, int] | None = None

    for chunk in chunks:
        if hasattr(chunk, "model") and chunk.model:
            model = chunk.model
        if hasattr(chunk, "choices") and chunk.choices:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and hasattr(delta, "content") and delta.content:
                content_parts.append(delta.content)
        if hasattr(chunk, "usage") and chunk.usage:
            u = chunk.usage
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            }

    if content_parts:
        span_ctx.set_output({"messages": [{"role": "assistant", "content": "".join(content_parts)}]})
    if model:
        span_ctx.set_model(model)
    if usage:
        span_ctx.set_token_usage(**usage)

    close_llm_span(span_ctx)


# ---------------------------------------------------------------------------
# OpenAI response extraction helpers
# ---------------------------------------------------------------------------


def _finish_span_from_chat_response(span_ctx: Any, response: Any) -> None:
    if span_ctx is None:
        return
    try:
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                span_ctx.set_output({"messages": [safe_serialize(choice.message)]})
            elif hasattr(choice, "text"):
                span_ctx.set_output({"messages": [{"role": "assistant", "content": choice.text}]})
        if hasattr(response, "model"):
            span_ctx.set_model(response.model)
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            span_ctx.set_token_usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
    except Exception as exc:
        logger.debug("Error extracting OpenAI response data: %s", exc)

    close_llm_span(span_ctx)


def _finish_span_legacy(span_ctx: Any, response: Any) -> None:
    if span_ctx is None:
        return
    try:
        if hasattr(response, "choices") and response.choices:
            span_ctx.set_output({"messages": [{"role": "assistant", "content": safe_serialize(response.choices[0].text)}]})
        if hasattr(response, "model"):
            span_ctx.set_model(response.model)
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            span_ctx.set_token_usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
    except Exception as exc:
        logger.debug("Error extracting OpenAI legacy response data: %s", exc)

    close_llm_span(span_ctx)


def _maybe_parse_raw(response: Any) -> Any:
    """If the response is an APIResponse (with_raw_response), parse it."""
    if hasattr(response, "parse"):
        return response.parse()
    return response
