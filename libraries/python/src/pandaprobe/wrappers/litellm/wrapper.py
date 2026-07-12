"""wrap_litellm — automatic LLM span instrumentation for the LiteLLM SDK.

LiteLLM exposes a unified, OpenAI-compatible interface via the module-level
functions ``litellm.completion`` (sync) and ``litellm.acompletion`` (async).
Because responses are OpenAI ChatCompletion-shaped (``.choices[0].message``,
``.model``, ``.usage``), the extraction logic mirrors the OpenAI wrapper.

The LiteLLM *proxy* exposes an OpenAI-compatible REST endpoint, so proxy-based
apps that point the ``openai`` client at it are already traced by
:func:`~pandaprobe.wrappers.openai.wrap_openai` — no LiteLLM-specific wrapping
is needed for that case.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from pandaprobe.wrappers._base import (
    AsyncStreamReducer,
    SyncStreamReducer,
    close_llm_span,
    error_llm_span,
    safe_serialize,
)
from pandaprobe.wrappers.litellm.utils import enter_litellm_span

logger = logging.getLogger("pandaprobe")

_WRAPPED_MARKER = "_pandaprobe_wrapped"


def wrap_litellm(litellm_module: Any | None = None) -> Any:
    """Instrument the LiteLLM SDK for automatic LLM span tracing.

    Monkey-patches the module-level ``completion`` (sync) and ``acompletion``
    (async) functions — covering both blocking and streaming calls — so every
    call automatically produces an LLM span.

    Pass the imported ``litellm`` module explicitly, or omit the argument to
    import and patch it globally::

        import litellm
        wrap_litellm(litellm)          # explicit
        litellm = wrap_litellm()       # import + patch

    Returns the (mutated) LiteLLM module. Idempotent — patching is global, so a
    second call is a no-op guarded by a marker attribute.
    """
    if litellm_module is None:
        import litellm as litellm_module  # type: ignore[no-redef]

    if getattr(litellm_module, _WRAPPED_MARKER, False):
        return litellm_module

    if hasattr(litellm_module, "completion"):
        litellm_module.completion = _sync_wrapper(litellm_module.completion)
    if hasattr(litellm_module, "acompletion"):
        litellm_module.acompletion = _async_wrapper(litellm_module.acompletion)

    setattr(litellm_module, _WRAPPED_MARKER, True)
    return litellm_module


# ---------------------------------------------------------------------------
# Sync / async wrappers
# ---------------------------------------------------------------------------


def _sync_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = dict(kwargs)
        is_stream = cleaned.get("stream", False)
        span_ctx = enter_litellm_span(cleaned, "litellm-chat")
        try:
            result = original(*args, **kwargs)
            if is_stream:
                return _LiteLLMSyncStream(result, span_ctx)
            _finish_litellm_span(span_ctx, result)
            return result
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


def _async_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = dict(kwargs)
        is_stream = cleaned.get("stream", False)
        span_ctx = enter_litellm_span(cleaned, "litellm-chat")
        try:
            result = await original(*args, **kwargs)
            if is_stream:
                return _LiteLLMAsyncStream(result, span_ctx)
            _finish_litellm_span(span_ctx, result)
            return result
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


# ---------------------------------------------------------------------------
# LiteLLM-specific stream reducers
# ---------------------------------------------------------------------------


class _LiteLLMSyncStream(SyncStreamReducer):
    def reduce_chunks(self, span_ctx: Any, chunks: list[Any]) -> None:
        _reduce_litellm_stream(span_ctx, chunks)


class _LiteLLMAsyncStream(AsyncStreamReducer):
    def reduce_chunks(self, span_ctx: Any, chunks: list[Any]) -> None:
        _reduce_litellm_stream(span_ctx, chunks)


def _reduce_litellm_stream(span_ctx: Any, chunks: list[Any]) -> None:
    """Reduce LiteLLM streaming chunks into final span data.

    Chunks are OpenAI-shaped ``ModelResponseStream`` objects
    (``chunk.choices[0].delta.content``); the terminal chunk carries ``usage``
    when ``stream_options={"include_usage": True}`` was passed. Reasoning is
    accumulated from ``delta.reasoning_content`` when present.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    model: str | None = None
    usage: dict[str, int] | None = None

    for chunk in chunks:
        if getattr(chunk, "model", None):
            model = chunk.model
        choices = getattr(chunk, "choices", None) or []
        if choices:
            delta = getattr(choices[0], "delta", None)
            if delta is not None:
                text = getattr(delta, "content", None)
                if isinstance(text, str) and text:
                    content_parts.append(text)
                reasoning = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning, str) and reasoning:
                    reasoning_parts.append(reasoning)
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage:
            usage = {
                "prompt_tokens": getattr(chunk_usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(chunk_usage, "completion_tokens", 0) or 0,
                **_extract_token_details(chunk_usage),
            }

    try:
        if content_parts:
            span_ctx.set_output({"messages": [{"role": "assistant", "content": "".join(content_parts)}]})
        if model:
            span_ctx.set_model(model)
        if usage:
            span_ctx.set_token_usage(**usage)
        if reasoning_parts:
            span_ctx.set_metadata({"reasoning_summary": "".join(reasoning_parts)})
    except Exception:
        logger.debug("Error populating LiteLLM stream span data", exc_info=True)

    try:
        close_llm_span(span_ctx)
    except Exception:
        logger.debug("close_llm_span failed during LiteLLM stream finalize", exc_info=True)


# ---------------------------------------------------------------------------
# Response extraction helpers
# ---------------------------------------------------------------------------


def _finish_litellm_span(span_ctx: Any, response: Any) -> None:
    """Extract output, model, usage, and reasoning from a LiteLLM ModelResponse."""
    if span_ctx is None:
        return

    try:
        choices = getattr(response, "choices", None) or []
        if choices:
            choice = choices[0]
            message = getattr(choice, "message", None)
            if message is not None:
                serialized = safe_serialize(message)
                if isinstance(serialized, dict):
                    serialized.setdefault("role", "assistant")
                    span_ctx.set_output({"messages": [serialized]})
                else:
                    span_ctx.set_output({"messages": [{"role": "assistant", "content": serialized}]})
                reasoning = getattr(message, "reasoning_content", None)
                if isinstance(reasoning, str) and reasoning:
                    span_ctx.set_metadata({"reasoning_summary": reasoning})
            elif hasattr(choice, "text"):
                span_ctx.set_output({"messages": [{"role": "assistant", "content": choice.text}]})
    except Exception as exc:
        logger.debug("Error extracting LiteLLM response output: %s", exc)

    try:
        model = getattr(response, "model", None)
        if model:
            span_ctx.set_model(model)
    except Exception as exc:
        logger.debug("Error extracting LiteLLM response model: %s", exc)

    try:
        usage = getattr(response, "usage", None)
        if usage:
            span_ctx.set_token_usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                **_extract_token_details(usage),
            )
    except Exception as exc:
        logger.debug("Error extracting LiteLLM response usage: %s", exc)

    try:
        close_llm_span(span_ctx)
    except Exception:
        logger.debug("close_llm_span failed during LiteLLM blocking finalize", exc_info=True)


# ---------------------------------------------------------------------------
# Token usage helpers (LiteLLM usage is OpenAI-shaped)
# ---------------------------------------------------------------------------


def _extract_token_details(usage: Any) -> dict[str, int]:
    """Extract total_tokens and all detail fields from an OpenAI-shaped usage object."""
    extra: dict[str, int] = {}

    total = getattr(usage, "total_tokens", None)
    if isinstance(total, int) and total > 0:
        extra["total_tokens"] = total

    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        _merge_detail_fields(extra, details)

    return extra


def _merge_detail_fields(target: dict[str, int], details: Any) -> None:
    """Merge all non-zero integer fields from a token-details object."""
    detail_dict: dict[str, Any] | None = None

    if hasattr(details, "model_dump"):
        try:
            detail_dict = details.model_dump(exclude_none=True)
        except Exception:
            pass

    if detail_dict is None and isinstance(details, dict):
        detail_dict = {k: v for k, v in details.items() if v is not None}

    if detail_dict is None and hasattr(details, "__dict__"):
        detail_dict = {k: v for k, v in details.__dict__.items() if not k.startswith("_") and v is not None}

    if detail_dict:
        for k, v in detail_dict.items():
            if isinstance(v, int) and v > 0:
                target[k] = v
