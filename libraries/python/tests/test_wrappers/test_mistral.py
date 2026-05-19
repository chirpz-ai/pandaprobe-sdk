"""Tests for pandaprobe.wrappers.mistral."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.wrappers.mistral import wrap_mistral
from pandaprobe.wrappers.mistral.utils import (
    extract_mistral_params,
    normalize_mistral_input,
    strip_unset,
)


@pytest.fixture(autouse=True)
def _setup_client():
    original = client_module._global_client
    pandaprobe.init(
        api_key="sk_pp_test",
        project_name="proj",
        endpoint="http://testserver",
        flush_interval=60.0,
    )
    yield
    if client_module._global_client is not None:
        client_module._global_client.shutdown()
    client_module._global_client = original


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestMistralUtilities:
    def test_strip_unset_without_mistral_passthrough(self):
        d = {"a": 1, "b": None, "c": "hello"}
        assert strip_unset(d) == d

    def test_extract_mistral_params(self):
        kwargs = {
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "random_seed": 42,
            "safe_prompt": True,
            "tool_choice": "auto",
            "api_key": "secret",
        }
        params = extract_mistral_params(kwargs)
        assert params == {
            "temperature": 0.7,
            "max_tokens": 100,
            "random_seed": 42,
            "safe_prompt": True,
            "tool_choice": "auto",
        }
        assert "model" not in params
        assert "messages" not in params
        assert "api_key" not in params

    def test_normalize_passthrough_messages(self):
        result = normalize_mistral_input(
            {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]}
        )
        assert result == {
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "hi"},
            ]
        }

    def test_normalize_with_pydantic_like_messages(self):
        msg = SimpleNamespace(role="user", content="Hi")
        result = normalize_mistral_input({"messages": [msg]})
        assert isinstance(result["messages"], list)
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hi"

    def test_normalize_no_messages(self):
        assert normalize_mistral_input({}) == {"messages": []}


# ---------------------------------------------------------------------------
# Mock client helpers
# ---------------------------------------------------------------------------


def _make_mock_mistral_response(text: str = "Bonjour!") -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=18, total_tokens=30)
    message = SimpleNamespace(role="assistant", content=text, tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        id="cmpl-mistral-1",
        object="chat.completion",
        model="mistral-small-latest",
        choices=[choice],
        usage=usage,
    )


def _make_mock_mistral_client(complete_return: object | None = None):
    """Build a mock that mirrors mistralai.Mistral()."""
    response = complete_return if complete_return is not None else _make_mock_mistral_response()
    complete_fn = MagicMock(return_value=response)
    complete_async_fn = AsyncMock(return_value=response)
    stream_fn = MagicMock()
    stream_async_fn = AsyncMock()
    chat = SimpleNamespace(
        complete=complete_fn,
        complete_async=complete_async_fn,
        stream=stream_fn,
        stream_async=stream_async_fn,
    )
    client = SimpleNamespace(chat=chat)
    return client, complete_fn, complete_async_fn, stream_fn, stream_async_fn, response


# ---------------------------------------------------------------------------
# Sync wrapper tests
# ---------------------------------------------------------------------------


class TestSyncWrapper:
    @respx.mock
    def test_sync_blocking_complete(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        client, complete_fn, *_ = _make_mock_mistral_client()
        wrapped = wrap_mistral(client)

        result = wrapped.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "Bonjour"}],
            temperature=0.5,
            max_tokens=100,
        )

        complete_fn.assert_called_once()
        assert result.choices[0].message.content == "Bonjour!"

    @respx.mock
    def test_returns_same_instance(self):
        client, *_ = _make_mock_mistral_client()
        assert wrap_mistral(client) is client

    @respx.mock
    def test_error_propagates(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        client, complete_fn, *_ = _make_mock_mistral_client()
        complete_fn.side_effect = RuntimeError("API down")
        wrap_mistral(client)

        with pytest.raises(RuntimeError, match="API down"):
            client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": "Hi"}],
            )

    @respx.mock
    def test_returns_original_response(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        client, _, _, _, _, expected = _make_mock_mistral_client()
        wrap_mistral(client)
        result = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result is expected

    @respx.mock
    def test_sync_streaming(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        def _make_chunk(text: str | None, *, model: str = "mistral-small-latest", usage=None):
            delta = (
                SimpleNamespace(role="assistant", content=text) if text is not None else SimpleNamespace(content=None)
            )
            choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
            data = SimpleNamespace(model=model, choices=[choice], usage=usage)
            return SimpleNamespace(data=data)

        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        events = [
            _make_chunk("Hel"),
            _make_chunk("lo!"),
            _make_chunk(None, usage=usage),
        ]

        client, _, _, stream_fn, _, _ = _make_mock_mistral_client()
        stream_fn.return_value = iter(events)
        wrap_mistral(client)

        stream = client.chat.stream(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "Hi"}],
        )
        collected = list(stream)
        assert len(collected) == 3


# ---------------------------------------------------------------------------
# Async wrapper tests
# ---------------------------------------------------------------------------


class TestAsyncWrapper:
    @respx.mock
    async def test_async_blocking(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        client, _, complete_async_fn, _, _, expected = _make_mock_mistral_client()
        wrap_mistral(client)

        result = await client.chat.complete_async(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "Bonjour"}],
            temperature=0.5,
        )

        complete_async_fn.assert_awaited_once()
        assert result is expected

    @respx.mock
    async def test_async_blocking_error(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        client, _, complete_async_fn, *_ = _make_mock_mistral_client()
        complete_async_fn.side_effect = RuntimeError("boom")
        wrap_mistral(client)

        with pytest.raises(RuntimeError, match="boom"):
            await client.chat.complete_async(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": "Hi"}],
            )

    @respx.mock
    async def test_async_streaming(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        async def _aiter(items):
            for it in items:
                yield it

        def _make_chunk(text: str | None, *, usage=None):
            delta = (
                SimpleNamespace(role="assistant", content=text) if text is not None else SimpleNamespace(content=None)
            )
            choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
            data = SimpleNamespace(model="mistral-small-latest", choices=[choice], usage=usage)
            return SimpleNamespace(data=data)

        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        events = [_make_chunk("Hel"), _make_chunk("lo"), _make_chunk(None, usage=usage)]

        client, _, _, _, stream_async_fn, _ = _make_mock_mistral_client()
        stream_async_fn.return_value = _aiter(events)
        wrap_mistral(client)

        stream = await client.chat.stream_async(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "Hi"}],
        )

        collected = []
        async for ev in stream:
            collected.append(ev)
        assert len(collected) == 3


# ---------------------------------------------------------------------------
# Span-finalize defensiveness — telemetry-side failures must not leak the
# span open, and the close path must be independently guarded so a setter
# failure can't skip ``close_llm_span``.
#
# This is provider-specific because the bug lives in ``_reduce_mistral_stream``
# (subclass-level), not in the base reducer.  Every provider's ``_reduce_*_stream``
# follows the same pattern; if you add another provider, mirror this test.
# ---------------------------------------------------------------------------


class TestStreamFinalizeDefensiveness:
    @respx.mock
    def test_setter_failure_in_reduce_stream_still_closes_span(self, monkeypatch):
        """Regression: ``_reduce_mistral_stream`` used to call ``set_output`` /
        ``set_model`` / ``set_token_usage`` followed by ``close_llm_span`` in
        an unguarded flow.  A setter raise would skip the close, the outer
        ``SyncStreamReducer._finalize`` would swallow the exception, and the
        span (and any standalone trace) would silently leak open.
        """
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        from pandaprobe.tracing.span import SpanContext
        from pandaprobe.wrappers.mistral import wrapper as mistral_wrapper

        close_calls: list[object] = []
        monkeypatch.setattr(mistral_wrapper, "close_llm_span", lambda ctx: close_calls.append(ctx))

        def _broken_set_token_usage(self, **kwargs):
            raise RuntimeError("set_token_usage exploded")

        monkeypatch.setattr(SpanContext, "set_token_usage", _broken_set_token_usage)

        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        events = [
            SimpleNamespace(
                data=SimpleNamespace(
                    model="mistral-small-latest",
                    choices=[SimpleNamespace(index=0, delta=SimpleNamespace(role="assistant", content="Hi"))],
                    usage=None,
                ),
            ),
            SimpleNamespace(
                data=SimpleNamespace(
                    model="mistral-small-latest",
                    choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content=None))],
                    usage=usage,
                ),
            ),
        ]

        client, _, _, stream_fn, _, _ = _make_mock_mistral_client()
        stream_fn.return_value = iter(events)
        wrap_mistral(client)

        stream = client.chat.stream(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "Hi"}],
        )

        collected = list(stream)
        assert len(collected) == 2
        assert len(close_calls) == 1, "span did not close — setter failure leaked the span"

    @respx.mock
    def test_close_failure_in_reduce_stream_does_not_propagate(self, monkeypatch):
        """Regression: ``close_llm_span`` itself raising during stream
        finalize must not surface as an iteration exception — the user's
        normal stream consumption must complete cleanly.
        """
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        from pandaprobe.wrappers.mistral import wrapper as mistral_wrapper

        def _broken_close(span_ctx):
            raise RuntimeError("close-blew-up")

        monkeypatch.setattr(mistral_wrapper, "close_llm_span", _broken_close)

        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        events = [
            SimpleNamespace(
                data=SimpleNamespace(
                    model="mistral-small-latest",
                    choices=[SimpleNamespace(index=0, delta=SimpleNamespace(role="assistant", content="Hi"))],
                    usage=None,
                ),
            ),
            SimpleNamespace(
                data=SimpleNamespace(
                    model="mistral-small-latest",
                    choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content=None))],
                    usage=usage,
                ),
            ),
        ]

        client, _, _, stream_fn, _, _ = _make_mock_mistral_client()
        stream_fn.return_value = iter(events)
        wrap_mistral(client)

        stream = client.chat.stream(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "Hi"}],
        )

        collected = list(stream)
        assert len(collected) == 2


# ---------------------------------------------------------------------------
# Schema compliance — gates the universal-trace contract
# ---------------------------------------------------------------------------


class TestSchemaCompliance:
    @respx.mock
    def test_universal_schema_input_and_output(self, monkeypatch):
        """span.set_input / set_output must use the universal {messages: [...]} shape."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        from pandaprobe.tracing.span import SpanContext

        calls: dict[str, object] = {}
        original_set_input = SpanContext.set_input
        original_set_output = SpanContext.set_output
        original_set_model = SpanContext.set_model
        original_set_token_usage = SpanContext.set_token_usage

        def _set_input(self, data):
            calls["input"] = data
            return original_set_input(self, data)

        def _set_output(self, data):
            calls["output"] = data
            return original_set_output(self, data)

        def _set_model(self, model):
            calls["model"] = model
            return original_set_model(self, model)

        def _set_token_usage(self, **kwargs):
            calls["usage"] = kwargs
            return original_set_token_usage(self, **kwargs)

        monkeypatch.setattr(SpanContext, "set_input", _set_input)
        monkeypatch.setattr(SpanContext, "set_output", _set_output)
        monkeypatch.setattr(SpanContext, "set_model", _set_model)
        monkeypatch.setattr(SpanContext, "set_token_usage", _set_token_usage)

        client, *_ = _make_mock_mistral_client()
        wrap_mistral(client)
        client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Bonjour"},
            ],
            temperature=0.5,
            max_tokens=50,
        )

        assert calls["input"] == {
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Bonjour"},
            ]
        }
        out = calls["output"]
        assert isinstance(out, dict) and "messages" in out
        assert isinstance(out["messages"], list) and len(out["messages"]) == 1
        msg = out["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Bonjour!"
        assert calls["model"] == "mistral-small-latest"
        assert calls["usage"] == {
            "prompt_tokens": 12,
            "completion_tokens": 18,
            "total_tokens": 30,
        }
