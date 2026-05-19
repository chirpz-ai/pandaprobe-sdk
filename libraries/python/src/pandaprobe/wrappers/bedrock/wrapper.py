"""wrap_bedrock — automatic LLM span instrumentation for AWS Bedrock clients.

Supports both the **Converse** API (preferred — provider-agnostic, normalised
input/output across all foundation models on Bedrock) and the legacy
**InvokeModel** API (best-effort with provider-specific body parsing).

Both blocking and streaming variants are instrumented, on sync ``boto3``
clients and async ``aioboto3`` clients.  Async support is opportunistic:
``aioboto3`` is *not* required to be installed; the wrapper detects coroutine
methods and routes accordingly.
"""

from __future__ import annotations

import functools
import inspect
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, TypeVar

from pandaprobe.wrappers._base import (
    close_llm_span,
    error_llm_span,
    safe_serialize,
)
from pandaprobe.wrappers.bedrock.utils import (
    enter_bedrock_span,
    map_converse_usage,
    map_invoke_model_usage,
)

logger = logging.getLogger("pandaprobe")

T = TypeVar("T")


def wrap_bedrock(client: T) -> T:
    """Instrument a ``boto3`` / ``aioboto3`` ``bedrock-runtime`` client.

    Patches ``converse``, ``converse_stream``, ``invoke_model`` and
    ``invoke_model_with_response_stream`` so every call automatically produces
    an LLM span.

    Returns the same client instance (mutated).
    """
    _patch_method(
        client,
        "converse",
        sync_factory=_make_sync_converse_wrapper,
        async_factory=_make_async_converse_wrapper,
    )
    _patch_method(
        client,
        "converse_stream",
        sync_factory=_make_sync_converse_stream_wrapper,
        async_factory=_make_async_converse_stream_wrapper,
    )
    _patch_method(
        client,
        "invoke_model",
        sync_factory=_make_sync_invoke_model_wrapper,
        async_factory=_make_async_invoke_model_wrapper,
    )
    _patch_method(
        client,
        "invoke_model_with_response_stream",
        sync_factory=_make_sync_invoke_model_stream_wrapper,
        async_factory=_make_async_invoke_model_stream_wrapper,
    )
    return client


# ---------------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------------


def _is_async_method(method: Any, owner: Any) -> bool:
    """Detect whether a bound boto-style method is asynchronous.

    botocore builds methods dynamically so ``inspect.iscoroutinefunction``
    works for plain ``async def`` functions but the safer cross-package
    detection also looks at the owning client class' module to spot
    aioboto3-shaped clients.
    """
    if inspect.iscoroutinefunction(method):
        return True
    underlying = getattr(method, "__func__", None)
    if underlying is not None and inspect.iscoroutinefunction(underlying):
        return True
    cls_module = getattr(type(owner), "__module__", "") or ""
    return cls_module.startswith("aioboto3") or cls_module.startswith("aiobotocore")


def _patch_method(
    client: Any,
    name: str,
    *,
    sync_factory,  # noqa: ANN001
    async_factory,  # noqa: ANN001
) -> None:
    if not hasattr(client, name):
        return
    original = getattr(client, name)
    if _is_async_method(original, client):
        wrapped = async_factory(original)
    else:
        wrapped = sync_factory(original)
    try:
        setattr(client, name, wrapped)
    except Exception:
        # botocore clients sometimes restrict attribute assignment; fall back
        # to monkey-patching via __dict__.
        try:
            client.__dict__[name] = wrapped
        except Exception as exc:
            logger.debug("Unable to patch bedrock client attribute %s: %s", name, exc)


# ---------------------------------------------------------------------------
# Converse — blocking
# ---------------------------------------------------------------------------


def _make_sync_converse_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        span_ctx = enter_bedrock_span(kwargs, "bedrock-converse", api="converse")
        try:
            response = original(*args, **kwargs)
            _finish_converse_span(span_ctx, response, kwargs.get("modelId"))
            return response
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


def _make_async_converse_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        span_ctx = enter_bedrock_span(kwargs, "bedrock-converse", api="converse")
        try:
            response = await original(*args, **kwargs)
            _finish_converse_span(span_ctx, response, kwargs.get("modelId"))
            return response
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


def _finish_converse_span(span_ctx: Any, response: Any, model_id: str | None) -> None:
    """Extract output, model, usage and reasoning from a Converse blocking response."""
    if span_ctx is None:
        return

    try:
        output = response.get("output") if isinstance(response, dict) else None
        message = output.get("message") if isinstance(output, dict) else None
        if isinstance(message, dict):
            text_parts, reasoning_parts = _split_converse_content(message.get("content"))
            if text_parts:
                span_ctx.set_output(
                    {"messages": [{"role": "assistant", "content": "".join(text_parts)}]},
                )
            else:
                span_ctx.set_output({"messages": [safe_serialize(message)]})
            if reasoning_parts:
                span_ctx.set_metadata({"reasoning_summary": "\n\n".join(reasoning_parts)})
    except Exception as exc:
        logger.debug("Error extracting Bedrock Converse output: %s", exc)

    try:
        if model_id:
            span_ctx.set_model(model_id)
    except Exception as exc:
        logger.debug("Error setting Bedrock Converse model: %s", exc)

    try:
        if isinstance(response, dict):
            usage = response.get("usage")
            mapped = map_converse_usage(usage)
            if mapped:
                span_ctx.set_token_usage(**mapped)
    except Exception as exc:
        logger.debug("Error extracting Bedrock Converse usage: %s", exc)

    close_llm_span(span_ctx)


def _split_converse_content(content: Any) -> tuple[list[str], list[str]]:
    """Split Converse content blocks into (text, reasoning) lists."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []

    if not isinstance(content, list):
        return text_parts, reasoning_parts

    for block in content:
        if not isinstance(block, dict):
            continue
        if "text" in block and isinstance(block["text"], str):
            text_parts.append(block["text"])
        elif "reasoningContent" in block:
            rc = block["reasoningContent"]
            if isinstance(rc, dict):
                rt = rc.get("reasoningText")
                if isinstance(rt, dict):
                    text = rt.get("text")
                    if isinstance(text, str) and text:
                        reasoning_parts.append(text)

    return text_parts, reasoning_parts


# ---------------------------------------------------------------------------
# Converse — streaming
# ---------------------------------------------------------------------------


def _make_sync_converse_stream_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        span_ctx = enter_bedrock_span(kwargs, "bedrock-converse", api="converse")
        model_id = kwargs.get("modelId")
        try:
            response = original(*args, **kwargs)
            return _wrap_converse_stream_response(response, span_ctx, model_id, is_async=False)
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


def _make_async_converse_stream_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        span_ctx = enter_bedrock_span(kwargs, "bedrock-converse", api="converse")
        model_id = kwargs.get("modelId")
        try:
            response = await original(*args, **kwargs)
            return _wrap_converse_stream_response(response, span_ctx, model_id, is_async=True)
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


def _wrap_converse_stream_response(
    response: Any,
    span_ctx: Any,
    model_id: str | None,
    *,
    is_async: bool,
) -> Any:
    """Wrap the ``stream`` value inside a Converse-stream response dict.

    Bedrock returns ``{"stream": <iterator>, "ResponseMetadata": {...}}`` —
    user code accesses ``response["stream"]`` so we must return a dict with
    the same shape.  Only the inner iterator is replaced with our reducer.
    """
    if not isinstance(response, dict) or "stream" not in response:
        if span_ctx is not None:
            close_llm_span(span_ctx)
        return response

    inner = response["stream"]
    if is_async:
        wrapped = _ConverseAsyncStream(inner, span_ctx, model_id)
    else:
        wrapped = _ConverseSyncStream(inner, span_ctx, model_id)

    response = dict(response)
    response["stream"] = wrapped
    return response


class _ConverseStreamMixin:
    """Shared chunk-handling logic for sync + async Converse stream wrappers."""

    def __init__(self, stream: Any, span_ctx: Any, model_id: str | None) -> None:
        self._stream = stream
        self._span_ctx = span_ctx
        self._model_id = model_id
        self._first_text = True
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._usage: dict[str, int] = {}
        self._finalized = False

    def _handle_event(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta") if isinstance(event["contentBlockDelta"], dict) else None
            if isinstance(delta, dict):
                text = delta.get("text")
                if isinstance(text, str) and text:
                    if self._first_text and self._span_ctx is not None:
                        self._span_ctx.set_completion_start_time(datetime.now(timezone.utc))
                        self._first_text = False
                    self._text_parts.append(text)
                rc = delta.get("reasoningContent")
                if isinstance(rc, dict):
                    rtext = rc.get("text")
                    if isinstance(rtext, str) and rtext:
                        self._reasoning_parts.append(rtext)
        elif "metadata" in event:
            meta = event["metadata"]
            if isinstance(meta, dict):
                mapped = map_converse_usage(meta.get("usage"))
                if mapped:
                    self._usage = mapped

    def _finalize(self) -> None:
        if self._finalized or self._span_ctx is None:
            return
        self._finalized = True
        try:
            if self._text_parts:
                self._span_ctx.set_output(
                    {"messages": [{"role": "assistant", "content": "".join(self._text_parts)}]},
                )
            if self._reasoning_parts:
                self._span_ctx.set_metadata(
                    {"reasoning_summary": "\n\n".join(self._reasoning_parts)},
                )
            if self._model_id:
                self._span_ctx.set_model(self._model_id)
            if self._usage:
                self._span_ctx.set_token_usage(**self._usage)
        except Exception:
            pass
        close_llm_span(self._span_ctx)
        self._span_ctx = None


class _ConverseSyncStream(_ConverseStreamMixin):
    """Sync iterator wrapping a Converse-stream's ``stream`` value."""

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self._stream)
        except StopIteration:
            self._finalize()
            raise
        self._handle_event(event)
        return event


class _ConverseAsyncStream(_ConverseStreamMixin):
    """Async iterator wrapping an aioboto3 Converse-stream's ``stream`` value."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        self._handle_event(event)
        return event


# ---------------------------------------------------------------------------
# InvokeModel — blocking
# ---------------------------------------------------------------------------


def _make_sync_invoke_model_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        span_ctx = enter_bedrock_span(kwargs, "bedrock-invoke-model", api="invoke_model")
        try:
            response = original(*args, **kwargs)
            _finish_invoke_model_span(span_ctx, response, kwargs.get("modelId"))
            return response
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


def _make_async_invoke_model_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        span_ctx = enter_bedrock_span(kwargs, "bedrock-invoke-model", api="invoke_model")
        try:
            response = await original(*args, **kwargs)
            await _finish_invoke_model_span_async(span_ctx, response, kwargs.get("modelId"))
            return response
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


def _rewindable_streaming_body(raw: bytes) -> Any:
    """Return a botocore-``StreamingBody``-compatible wrapper over ``raw``.

    Falls back to a plain ``io.BytesIO`` if botocore is not importable — the
    ``.read()`` surface is sufficient for the documented user-code pattern
    (``response["body"].read()``).
    """
    buf = io.BytesIO(raw)
    try:
        from botocore.response import StreamingBody  # type: ignore[import-not-found]

        return StreamingBody(buf, len(raw))
    except Exception:
        return buf


def _decode_invoke_payload(raw: Any, response: Any, consumed_stream: bool) -> Any:
    """Shared post-read decoding + rewind for the InvokeModel response body."""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw_bytes: bytes | None = bytes(raw)
        try:
            text: str | None = raw_bytes.decode("utf-8")
        except Exception:
            text = None
    elif isinstance(raw, str):
        text = raw
        raw_bytes = raw.encode("utf-8") if consumed_stream else None
    else:
        # Unexpected type (e.g. unawaited coroutine slipped through) — bail out
        # rather than corrupting the body.
        return None

    if consumed_stream and raw_bytes is not None and isinstance(response, dict):
        try:
            response["body"] = _rewindable_streaming_body(raw_bytes)
        except Exception:
            logger.debug("Failed to rewind Bedrock InvokeModel body", exc_info=True)

    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def _read_invoke_body(response: Any) -> Any:
    """Synchronously decode an InvokeModel response body (boto3 path).

    Botocore's ``StreamingBody`` is a one-shot stream — naively calling
    ``body.read()`` here would consume the response and any subsequent
    ``response["body"].read()`` from user code would silently return ``b""``.
    To avoid that, we read the bytes once and replace ``response["body"]``
    with a fresh in-memory streaming body holding the same payload, so
    downstream consumers see an unread body.
    """
    if not isinstance(response, dict):
        return None
    body = response.get("body")
    if body is None:
        return None

    raw: Any = None
    consumed_stream = False
    if hasattr(body, "read"):
        # Async clients (aioboto3) expose `read` as a coroutine function — we
        # cannot await here and must defer to ``_read_invoke_body_async``.
        if inspect.iscoroutinefunction(getattr(body, "read", None)):
            return None
        try:
            raw = body.read()
            consumed_stream = True
        except Exception:
            return None
        if inspect.iscoroutine(raw):
            # Defensive: some hybrid clients return an awaitable from a
            # non-``async def`` ``read``. Close it so we don't leak an
            # unawaited coroutine, and bail out of the sync path.
            raw.close()
            return None
    elif isinstance(body, (bytes, bytearray, str)):
        raw = body

    return _decode_invoke_payload(raw, response, consumed_stream)


async def _read_invoke_body_async(response: Any) -> Any:
    """Asynchronously decode an InvokeModel response body (aioboto3 path).

    Mirrors :func:`_read_invoke_body` but awaits ``body.read()`` when it is a
    coroutine function (the aioboto3 ``StreamingBody`` shape).  Falls back to
    synchronous reads when the body is already a sync stream / bytes / str so
    the same wrapper covers mixed clients.
    """
    if not isinstance(response, dict):
        return None
    body = response.get("body")
    if body is None:
        return None

    raw: Any = None
    consumed_stream = False
    if hasattr(body, "read"):
        try:
            value = body.read()
            if inspect.iscoroutine(value):
                raw = await value
            else:
                raw = value
            consumed_stream = True
        except Exception:
            return None
    elif isinstance(body, (bytes, bytearray, str)):
        raw = body

    return _decode_invoke_payload(raw, response, consumed_stream)


def _extract_invoke_model_text(parsed: Any) -> str | None:
    """Pull a single assistant text out of the most common provider body shapes."""
    if not isinstance(parsed, dict):
        return None

    content = parsed.get("content")
    if isinstance(content, list):
        text_parts = [
            block.get("text") for block in content if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        if text_parts:
            return "".join(text_parts)
    if isinstance(parsed.get("completion"), str):
        return parsed["completion"]
    if isinstance(parsed.get("generation"), str):
        return parsed["generation"]
    if isinstance(parsed.get("output_text"), str):
        return parsed["output_text"]

    results = parsed.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        for key in ("outputText", "output_text", "text"):
            v = results[0].get(key)
            if isinstance(v, str):
                return v

    choices = parsed.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
        if isinstance(choices[0].get("text"), str):
            return choices[0]["text"]

    if isinstance(parsed.get("generations"), list) and parsed["generations"]:
        gen = parsed["generations"][0]
        if isinstance(gen, dict) and isinstance(gen.get("text"), str):
            return gen["text"]

    return None


def _finalize_invoke_model_span(span_ctx: Any, parsed: Any, model_id: str | None) -> None:
    """Span population shared by the sync and async InvokeModel wrappers."""
    try:
        text = _extract_invoke_model_text(parsed)
        if text is not None:
            span_ctx.set_output({"messages": [{"role": "assistant", "content": text}]})
        elif parsed is not None:
            span_ctx.set_output({"messages": [{"role": "assistant", "content": safe_serialize(parsed)}]})
    except Exception as exc:
        logger.debug("Error extracting Bedrock InvokeModel output: %s", exc)

    try:
        if model_id:
            span_ctx.set_model(model_id)
    except Exception as exc:
        logger.debug("Error setting Bedrock InvokeModel model: %s", exc)

    try:
        mapped = map_invoke_model_usage(parsed)
        if mapped:
            span_ctx.set_token_usage(**mapped)
    except Exception as exc:
        logger.debug("Error extracting Bedrock InvokeModel usage: %s", exc)

    close_llm_span(span_ctx)


def _finish_invoke_model_span(span_ctx: Any, response: Any, model_id: str | None) -> None:
    """Sync entry point — used by the boto3 InvokeModel wrapper."""
    if span_ctx is None:
        return
    parsed = _read_invoke_body(response)
    _finalize_invoke_model_span(span_ctx, parsed, model_id)


async def _finish_invoke_model_span_async(span_ctx: Any, response: Any, model_id: str | None) -> None:
    """Async entry point — used by the aioboto3 InvokeModel wrapper.

    aioboto3's ``StreamingBody.read()`` is a coroutine — calling it from the
    sync path leaks an unawaited coroutine *and* fails to capture span output.
    """
    if span_ctx is None:
        return
    parsed = await _read_invoke_body_async(response)
    _finalize_invoke_model_span(span_ctx, parsed, model_id)


# ---------------------------------------------------------------------------
# InvokeModel — streaming
# ---------------------------------------------------------------------------


def _make_sync_invoke_model_stream_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        span_ctx = enter_bedrock_span(kwargs, "bedrock-invoke-model-stream", api="invoke_model")
        model_id = kwargs.get("modelId")
        try:
            response = original(*args, **kwargs)
            return _wrap_invoke_stream_response(response, span_ctx, model_id, is_async=False)
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


def _make_async_invoke_model_stream_wrapper(original):  # noqa: ANN001
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        span_ctx = enter_bedrock_span(kwargs, "bedrock-invoke-model-stream", api="invoke_model")
        model_id = kwargs.get("modelId")
        try:
            response = await original(*args, **kwargs)
            return _wrap_invoke_stream_response(response, span_ctx, model_id, is_async=True)
        except Exception as exc:
            error_llm_span(span_ctx, exc)
            raise

    return wrapper


def _wrap_invoke_stream_response(
    response: Any,
    span_ctx: Any,
    model_id: str | None,
    *,
    is_async: bool,
) -> Any:
    """Wrap the ``body`` event-stream inside an InvokeModel-with-response-stream dict.

    Closes the span without attempting per-provider chunk reduction (bodies
    differ wildly by provider); the span still captures input, model and
    error state.  Use the Converse stream API when you need full output +
    usage capture.
    """
    if not isinstance(response, dict) or "body" not in response:
        if span_ctx is not None:
            close_llm_span(span_ctx)
        return response

    inner = response["body"]
    if is_async:
        wrapped = _InvokeModelAsyncStream(inner, span_ctx, model_id)
    else:
        wrapped = _InvokeModelSyncStream(inner, span_ctx, model_id)

    response = dict(response)
    response["body"] = wrapped
    return response


class _InvokeModelStreamMixin:
    def __init__(self, stream: Any, span_ctx: Any, model_id: str | None) -> None:
        self._stream = stream
        self._span_ctx = span_ctx
        self._model_id = model_id
        self._first_chunk = True
        self._finalized = False

    def _on_chunk(self) -> None:
        if self._first_chunk and self._span_ctx is not None:
            self._span_ctx.set_completion_start_time(datetime.now(timezone.utc))
            self._first_chunk = False

    def _finalize(self) -> None:
        if self._finalized or self._span_ctx is None:
            return
        self._finalized = True
        try:
            if self._model_id:
                self._span_ctx.set_model(self._model_id)
        except Exception:
            pass
        close_llm_span(self._span_ctx)
        self._span_ctx = None


class _InvokeModelSyncStream(_InvokeModelStreamMixin):
    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self._stream)
        except StopIteration:
            self._finalize()
            raise
        self._on_chunk()
        return event


class _InvokeModelAsyncStream(_InvokeModelStreamMixin):
    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        self._on_chunk()
        return event
