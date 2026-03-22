"""wrap_openai — automatic LLM span instrumentation for the OpenAI client."""

from __future__ import annotations

import functools
import logging
from typing import Any, TypeVar

from datetime import datetime, timezone

from pandaprobe.schemas import SpanKind
from pandaprobe.wrappers._base import (
    AsyncStreamReducer,
    SyncStreamReducer,
    close_llm_span,
    enter_llm_span,
    safe_serialize,
)
from pandaprobe.wrappers.openai.utils import enter_responses_span, strip_not_given

logger = logging.getLogger("pandaprobe")

T = TypeVar("T")


def wrap_openai(client: T) -> T:
    """Instrument an ``openai.OpenAI`` or ``openai.AsyncOpenAI`` client.

    Monkey-patches ``chat.completions.create``, ``completions.create``, and
    ``responses.create`` so every call automatically produces an LLM span.
    For the Responses API, built-in tool calls (web_search, function_call,
    etc.) are captured as child TOOL spans.

    Returns the same client instance (mutated).
    """
    _patch_chat_completions(client)
    _patch_legacy_completions(client)
    _patch_responses(client)
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


def _patch_responses(client: Any) -> None:
    if not hasattr(client, "responses") or not hasattr(client.responses, "create"):
        return

    import asyncio

    original = client.responses.create
    if asyncio.iscoroutinefunction(original):
        client.responses.create = _async_responses_wrapper(original)
    else:
        client.responses.create = _sync_responses_wrapper(original)


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
                **_extract_token_details(u),
            }

    if content_parts:
        span_ctx.set_output({"messages": [{"role": "assistant", "content": "".join(content_parts)}]})
    if model:
        span_ctx.set_model(model)
    if usage:
        span_ctx.set_token_usage(**usage)

    close_llm_span(span_ctx)


# ---------------------------------------------------------------------------
# Token usage helpers
# ---------------------------------------------------------------------------


def _extract_token_details(usage: Any) -> dict[str, int]:
    """Extract total_tokens and all detail fields from a usage object.

    Handles both Chat Completions (``completion_tokens_details``) and
    Responses API (``output_tokens_details``) usage objects, with robust
    extraction via Pydantic ``model_dump``, dict access, and attribute access.
    """
    extra: dict[str, int] = {}

    total = getattr(usage, "total_tokens", None)
    if isinstance(total, int) and total > 0:
        extra["total_tokens"] = total

    for attr_name in ("completion_tokens_details", "output_tokens_details"):
        details = getattr(usage, attr_name, None)
        if details is None:
            continue
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
            extra = _extract_token_details(usage)
            span_ctx.set_token_usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                **extra,
            )
    except Exception as exc:
        logger.debug("Error extracting OpenAI response data: %s", exc)

    close_llm_span(span_ctx)


def _finish_span_legacy(span_ctx: Any, response: Any) -> None:
    if span_ctx is None:
        return
    try:
        if hasattr(response, "choices") and response.choices:
            span_ctx.set_output(
                {"messages": [{"role": "assistant", "content": safe_serialize(response.choices[0].text)}]}
            )
        if hasattr(response, "model"):
            span_ctx.set_model(response.model)
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            extra = _extract_token_details(usage)
            span_ctx.set_token_usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                **extra,
            )
    except Exception as exc:
        logger.debug("Error extracting OpenAI legacy response data: %s", exc)

    close_llm_span(span_ctx)


def _maybe_parse_raw(response: Any) -> Any:
    """If the response is an APIResponse (with_raw_response), parse it."""
    if hasattr(response, "parse"):
        return response.parse()
    return response


# ---------------------------------------------------------------------------
# Responses API wrappers
# ---------------------------------------------------------------------------


def _sync_responses_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_not_given(kwargs)
        is_stream = cleaned.get("stream", False)

        if is_stream:
            return _sync_streaming_response(original, args, kwargs, cleaned)
        return _sync_blocking_response(original, args, kwargs, cleaned)

    return wrapper


def _async_responses_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleaned = strip_not_given(kwargs)
        is_stream = cleaned.get("stream", False)

        if is_stream:
            return await _async_streaming_response(original, args, kwargs, cleaned)
        return await _async_blocking_response(original, args, kwargs, cleaned)

    return wrapper


def _sync_blocking_response(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_responses_span(cleaned)
    try:
        response = original(*args, **kwargs)
        _finish_from_response(span_ctx, response)
        return response
    except Exception as exc:
        if span_ctx:
            span_ctx.set_error(str(exc))
            span_ctx.__exit__(type(exc), exc, exc.__traceback__)
        raise


def _sync_streaming_response(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_responses_span(cleaned)
    try:
        stream = original(*args, **kwargs)
        return _ResponsesSyncStream(stream, span_ctx)
    except Exception as exc:
        if span_ctx:
            span_ctx.set_error(str(exc))
            span_ctx.__exit__(type(exc), exc, exc.__traceback__)
        raise


async def _async_blocking_response(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_responses_span(cleaned)
    try:
        response = await original(*args, **kwargs)
        _finish_from_response(span_ctx, response)
        return response
    except Exception as exc:
        if span_ctx:
            span_ctx.set_error(str(exc))
            span_ctx.__exit__(type(exc), exc, exc.__traceback__)
        raise


async def _async_streaming_response(original, args, kwargs, cleaned):  # noqa: ANN001
    span_ctx = enter_responses_span(cleaned)
    try:
        stream = await original(*args, **kwargs)
        return _ResponsesAsyncStream(stream, span_ctx)
    except Exception as exc:
        if span_ctx:
            span_ctx.set_error(str(exc))
            span_ctx.__exit__(type(exc), exc, exc.__traceback__)
        raise


# ---------------------------------------------------------------------------
# Responses API stream wrappers
# ---------------------------------------------------------------------------


class _ResponsesSyncStream:
    """Wraps a Responses API sync stream, yielding events and finalising the span."""

    def __init__(self, stream: Any, span_ctx: Any) -> None:
        self._stream = stream
        self._span_ctx = span_ctx
        self._first_text_delta = True
        self._completed_response: Any = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self._stream)
            self._handle_event(event)
            return event
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

    def _handle_event(self, event: Any) -> None:
        event_type = getattr(event, "type", "")
        if event_type == "response.output_text.delta" and self._first_text_delta and self._span_ctx:
            self._span_ctx.set_completion_start_time(datetime.now(timezone.utc))
            self._first_text_delta = False
        if event_type == "response.completed":
            self._completed_response = getattr(event, "response", None)

    def _finalize(self) -> None:
        if self._span_ctx is None:
            return
        try:
            if self._completed_response is not None:
                _finish_from_response(self._span_ctx, self._completed_response)
            else:
                close_llm_span(self._span_ctx)
        except Exception:
            pass
        self._span_ctx = None


class _ResponsesAsyncStream:
    """Wraps a Responses API async stream, yielding events and finalising the span."""

    def __init__(self, stream: Any, span_ctx: Any) -> None:
        self._stream = stream
        self._span_ctx = span_ctx
        self._first_text_delta = True
        self._completed_response: Any = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self._stream.__anext__()
            self._handle_event(event)
            return event
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

    def _handle_event(self, event: Any) -> None:
        event_type = getattr(event, "type", "")
        if event_type == "response.output_text.delta" and self._first_text_delta and self._span_ctx:
            self._span_ctx.set_completion_start_time(datetime.now(timezone.utc))
            self._first_text_delta = False
        if event_type == "response.completed":
            self._completed_response = getattr(event, "response", None)

    def _finalize(self) -> None:
        if self._span_ctx is None:
            return
        try:
            if self._completed_response is not None:
                _finish_from_response(self._span_ctx, self._completed_response)
            else:
                close_llm_span(self._span_ctx)
        except Exception:
            pass
        self._span_ctx = None


# ---------------------------------------------------------------------------
# Responses API extraction helpers
# ---------------------------------------------------------------------------


def _finish_from_response(span_ctx: Any, response: Any) -> None:
    """Extract output, model, usage, reasoning, and tool calls from a Responses API response."""
    if span_ctx is None:
        return
    output_items = getattr(response, "output", None) or []

    try:
        message_items = [safe_serialize(item) for item in output_items if _get_field(item, "type") == "message"]
        if message_items:
            span_ctx.set_output({"messages": message_items})
    except Exception as exc:
        logger.debug("Error extracting Responses API output: %s", exc)

    try:
        if hasattr(response, "model") and response.model:
            span_ctx.set_model(response.model)
    except Exception as exc:
        logger.debug("Error extracting Responses API model: %s", exc)

    try:
        if hasattr(response, "usage") and response.usage:
            _set_responses_usage(span_ctx, response.usage)
    except Exception as exc:
        logger.debug("Error extracting Responses API usage: %s", exc)

    try:
        _extract_reasoning_summary(span_ctx, output_items)
    except Exception as exc:
        logger.debug("Error extracting reasoning summary: %s", exc)

    try:
        _create_tool_child_spans(span_ctx, output_items)
    except Exception as exc:
        logger.debug("Error creating tool child spans: %s", exc)

    close_llm_span(span_ctx)


def _set_responses_usage(span_ctx: Any, usage: Any) -> None:
    """Normalise Responses API usage (input_tokens/output_tokens) to standard format."""
    prompt = getattr(usage, "input_tokens", 0) or 0
    completion = getattr(usage, "output_tokens", 0) or 0
    extra = _extract_token_details(usage)
    span_ctx.set_token_usage(prompt_tokens=prompt, completion_tokens=completion, **extra)


def _extract_reasoning_summary(span_ctx: Any, output_items: list[Any]) -> None:
    """Pull reasoning summaries from output items and store in span metadata."""
    summaries: list[str] = []
    for item in output_items:
        item_type = _get_field(item, "type")
        if item_type != "reasoning":
            continue

        summary_list = _get_field(item, "summary")
        if not summary_list:
            continue

        for s in summary_list:
            text = _get_field(s, "text")
            if text:
                summaries.append(text)

    if summaries:
        span_ctx.set_metadata({"reasoning_summary": "\n\n".join(summaries)})


def _get_field(obj: Any, field: str) -> Any:
    """Get a field from a Pydantic model, dict, or namespace."""
    val = getattr(obj, field, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(field)
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump().get(field)
        except Exception:
            pass
    return None


def _create_tool_child_spans(span_ctx: Any, output_items: list[Any]) -> None:
    """Create child TOOL spans for built-in tool calls found in the response output."""
    trace_ctx = getattr(span_ctx, "_trace_ctx", None)
    if trace_ctx is None:
        return

    for item in output_items:
        item_type = _get_field(item, "type")

        if item_type == "function_call":
            _create_function_call_span(trace_ctx, item)
        elif item_type == "web_search_call":
            _create_simple_tool_span(trace_ctx, "web_search", item)
        elif item_type == "file_search_call":
            _create_simple_tool_span(trace_ctx, "file_search", item)
        elif item_type == "code_interpreter_call":
            _create_simple_tool_span(trace_ctx, "code_interpreter", item)


def _create_function_call_span(trace_ctx: Any, item: Any) -> None:
    """Create a TOOL span for a function_call output item."""
    name = _get_field(item, "name") or "function_call"
    span_name = f"function_call:{name}"

    arguments = _get_field(item, "arguments")

    tool_span = trace_ctx.span(span_name, kind=SpanKind.TOOL)
    tool_span.__enter__()
    if arguments:
        tool_span.set_input(safe_serialize(arguments))
    tool_span.__exit__(None, None, None)


def _create_simple_tool_span(trace_ctx: Any, tool_name: str, item: Any) -> None:
    """Create a TOOL span for a built-in tool call (web_search, file_search, etc.)."""
    status = _get_field(item, "status")

    tool_span = trace_ctx.span(tool_name, kind=SpanKind.TOOL)
    tool_span.__enter__()
    tool_span.set_input(safe_serialize(item))
    if status:
        tool_span.set_output({"status": status})
    tool_span.__exit__(None, None, None)
