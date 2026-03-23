"""wrap_gemini — automatic LLM span instrumentation for the Google GenAI client."""

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
from pandaprobe.wrappers.gemini.utils import convert_config_to_dict, enter_gemini_span

logger = logging.getLogger("pandaprobe")

T = TypeVar("T")


def wrap_gemini(client: T) -> T:
    """Instrument a ``google.genai.Client`` for automatic LLM span tracing.

    Monkey-patches ``models.generate_content``, ``models.generate_content_stream``
    and their async counterparts so every call automatically produces an LLM span.

    Returns the same client instance (mutated).
    """
    _patch_sync_generate(client)
    _patch_sync_stream(client)
    _patch_async_generate(client)
    _patch_async_stream(client)
    return client


# ---------------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------------


def _patch_sync_generate(client: Any) -> None:
    if not hasattr(client, "models") or not hasattr(client.models, "generate_content"):
        return
    original = client.models.generate_content
    client.models.generate_content = _sync_generate_wrapper(original)


def _patch_sync_stream(client: Any) -> None:
    if not hasattr(client, "models") or not hasattr(client.models, "generate_content_stream"):
        return
    original = client.models.generate_content_stream
    client.models.generate_content_stream = _sync_stream_wrapper(original)


def _patch_async_generate(client: Any) -> None:
    if (
        not hasattr(client, "aio")
        or not hasattr(client.aio, "models")
        or not hasattr(client.aio.models, "generate_content")
    ):
        return
    original = client.aio.models.generate_content
    client.aio.models.generate_content = _async_generate_wrapper(original)


def _patch_async_stream(client: Any) -> None:
    if (
        not hasattr(client, "aio")
        or not hasattr(client.aio, "models")
        or not hasattr(client.aio.models, "generate_content_stream")
    ):
        return
    original = client.aio.models.generate_content_stream
    client.aio.models.generate_content_stream = _async_stream_wrapper(original)


# ---------------------------------------------------------------------------
# Sync wrappers
# ---------------------------------------------------------------------------


def _sync_generate_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = convert_config_to_dict(kwargs)
        return _sync_blocking_generate(original, args, kwargs, cleaned)

    return wrapper


def _sync_stream_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = convert_config_to_dict(kwargs)
        return _sync_streaming_generate(original, args, kwargs, cleaned)

    return wrapper


def _sync_blocking_generate(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_gemini_span(cleaned, "gemini-generate")
    try:
        response = original(*args, **kwargs)
        _finish_gemini_span(span_ctx, response)
        return response
    except Exception as exc:
        error_llm_span(span_ctx, exc)
        raise


def _sync_streaming_generate(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_gemini_span(cleaned, "gemini-generate")
    try:
        stream = original(*args, **kwargs)
        return _GeminiSyncStream(stream, span_ctx)
    except Exception as exc:
        error_llm_span(span_ctx, exc)
        raise


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


def _async_generate_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = convert_config_to_dict(kwargs)
        return await _async_blocking_generate(original, args, kwargs, cleaned)

    return wrapper


def _async_stream_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = convert_config_to_dict(kwargs)
        return await _async_streaming_generate(original, args, kwargs, cleaned)

    return wrapper


async def _async_blocking_generate(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_gemini_span(cleaned, "gemini-generate")
    try:
        response = await original(*args, **kwargs)
        _finish_gemini_span(span_ctx, response)
        return response
    except Exception as exc:
        error_llm_span(span_ctx, exc)
        raise


async def _async_streaming_generate(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_gemini_span(cleaned, "gemini-generate")
    try:
        stream = await original(*args, **kwargs)
        return _GeminiAsyncStream(stream, span_ctx)
    except Exception as exc:
        error_llm_span(span_ctx, exc)
        raise


# ---------------------------------------------------------------------------
# Gemini-specific stream reducers
# ---------------------------------------------------------------------------


class _GeminiSyncStream(SyncStreamReducer):
    def reduce_chunks(self, span_ctx: Any, chunks: list[Any]) -> None:
        _reduce_gemini_stream(span_ctx, chunks)


class _GeminiAsyncStream(AsyncStreamReducer):
    def reduce_chunks(self, span_ctx: Any, chunks: list[Any]) -> None:
        _reduce_gemini_stream(span_ctx, chunks)


def _reduce_gemini_stream(span_ctx: Any, chunks: list[Any]) -> None:
    """Reduce Gemini streaming chunks into final span data."""
    answer_parts: list[str] = []
    thought_parts: list[str] = []
    usage: dict[str, int] | None = None

    for chunk in chunks:
        try:
            candidates = getattr(chunk, "candidates", None)
            if candidates:
                content = getattr(candidates[0], "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    for part in parts:
                        text = getattr(part, "text", None)
                        if not text:
                            continue
                        if getattr(part, "thought", False):
                            thought_parts.append(text)
                        else:
                            answer_parts.append(text)
                    continue
            text = getattr(chunk, "text", None)
            if text:
                answer_parts.append(text)
        except Exception:
            pass

    last_chunk = chunks[-1] if chunks else None
    if last_chunk is not None:
        try:
            usage = _extract_gemini_usage(getattr(last_chunk, "usage_metadata", None))
        except Exception:
            pass

    if answer_parts:
        span_ctx.set_output(
            {"messages": [{"role": "assistant", "content": "".join(answer_parts)}]},
        )
    if thought_parts:
        span_ctx.set_metadata({"reasoning_summary": "\n\n".join(thought_parts)})
    if usage:
        span_ctx.set_token_usage(**usage)

    close_llm_span(span_ctx)


# ---------------------------------------------------------------------------
# Response extraction helpers
# ---------------------------------------------------------------------------


def _finish_gemini_span(span_ctx: Any, response: Any) -> None:
    """Extract output, model, reasoning summary, and usage from a Gemini response."""
    if span_ctx is None:
        return

    try:
        answer_texts, thought_texts = _split_parts(response)
        if answer_texts:
            span_ctx.set_output(
                {"messages": [{"role": "assistant", "content": "".join(answer_texts)}]},
            )
        elif not thought_texts:
            text = getattr(response, "text", None)
            if text is not None:
                span_ctx.set_output(
                    {"messages": [{"role": "assistant", "content": text}]},
                )
            else:
                candidates = getattr(response, "candidates", None)
                if candidates:
                    content = getattr(candidates[0], "content", None)
                    if content is not None:
                        serialized = safe_serialize(content)
                        if isinstance(serialized, dict):
                            serialized["role"] = "assistant"
                        span_ctx.set_output({"messages": [serialized]})
        if thought_texts:
            span_ctx.set_metadata({"reasoning_summary": "\n\n".join(thought_texts)})
    except Exception as exc:
        logger.debug("Error extracting Gemini response output: %s", exc)

    try:
        usage = _extract_gemini_usage(getattr(response, "usage_metadata", None))
        if usage:
            span_ctx.set_token_usage(**usage)
    except Exception as exc:
        logger.debug("Error extracting Gemini response usage: %s", exc)

    close_llm_span(span_ctx)


def _split_parts(response: Any) -> tuple[list[str], list[str]]:
    """Separate answer text from thought summaries in a Gemini response.

    Returns ``(answer_texts, thought_texts)``.
    """
    answer_texts: list[str] = []
    thought_texts: list[str] = []

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return answer_texts, thought_texts

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return answer_texts, thought_texts

    for part in parts:
        text = getattr(part, "text", None)
        if not text:
            continue
        if getattr(part, "thought", False):
            thought_texts.append(text)
        else:
            answer_texts.append(text)

    return answer_texts, thought_texts


def _extract_gemini_usage(usage_metadata: Any) -> dict[str, int] | None:
    """Map Gemini ``usage_metadata`` fields to our standard token usage dict."""
    if usage_metadata is None:
        return None

    prompt = getattr(usage_metadata, "prompt_token_count", 0) or 0
    completion = getattr(usage_metadata, "candidates_token_count", 0) or 0
    extra: dict[str, int] = {}

    total = getattr(usage_metadata, "total_token_count", 0) or 0
    if total > 0:
        extra["total_tokens"] = total

    thoughts = getattr(usage_metadata, "thoughts_token_count", 0) or 0
    if thoughts > 0:
        extra["reasoning_tokens"] = thoughts

    cached = getattr(usage_metadata, "cached_content_token_count", 0) or 0
    if cached > 0:
        extra["cache_read_tokens"] = cached

    if prompt == 0 and completion == 0 and not extra:
        return None

    return {"prompt_tokens": prompt, "completion_tokens": completion, **extra}
