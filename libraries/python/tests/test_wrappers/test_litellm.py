"""Tests for pandaprobe.wrappers.litellm."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.wrappers.litellm import wrap_litellm
from pandaprobe.wrappers.litellm.utils import (
    extract_litellm_params,
    normalize_litellm_input,
)


@pytest.fixture(autouse=True)
def _setup_client():
    from pandaprobe.tracing import context as ctx_module

    original = client_module._global_client
    # Reset the ambient trace/span context so a trace leaked by an earlier test
    # (e.g. a defensiveness test that stubs out close_llm_span and never exits
    # the trace) can't route our standalone-trace path into the nested branch.
    trace_token = ctx_module._current_trace.set(None)
    stack_token = ctx_module._span_stack.set([])
    pandaprobe.init(
        api_key="sk_pp_test",
        project_name="proj",
        endpoint="http://testserver",
        flush_interval=60.0,
    )
    yield
    ctx_module._current_trace.reset(trace_token)
    ctx_module._span_stack.reset(stack_token)
    if client_module._global_client is not None:
        client_module._global_client.shutdown()
    client_module._global_client = original


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestBaseUtilities:
    def test_extract_litellm_params(self):
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "reasoning_effort": "low",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "api_key": "secret",
            "api_base": "https://example.com",
        }
        params = extract_litellm_params(kwargs)
        assert params == {
            "temperature": 0.7,
            "max_tokens": 100,
            "reasoning_effort": "low",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }
        assert "model" not in params
        assert "messages" not in params
        assert "api_key" not in params
        assert "api_base" not in params

    def test_normalize_passthrough_messages(self):
        result = normalize_litellm_input(
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
        result = normalize_litellm_input({"messages": [msg]})
        assert isinstance(result["messages"], list)
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hi"

    def test_normalize_no_messages(self):
        assert normalize_litellm_input({}) == {"messages": []}


# ---------------------------------------------------------------------------
# Mock module helpers
# ---------------------------------------------------------------------------


def _make_mock_response(text: str = "Hello!", *, model: str = "gpt-4o", reasoning: str | None = None):
    message = SimpleNamespace(role="assistant", content=text, tool_calls=None)
    if reasoning is not None:
        message.reasoning_content = reasoning
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=18, total_tokens=30)
    return SimpleNamespace(id="chatcmpl-1", model=model, choices=[choice], usage=usage)


def _make_mock_litellm(completion_return: object | None = None):
    """Build a fake ``litellm`` module exposing completion / acompletion."""
    response = completion_return if completion_return is not None else _make_mock_response()
    completion_fn = MagicMock(return_value=response)
    acompletion_fn = AsyncMock(return_value=response)
    module = SimpleNamespace(completion=completion_fn, acompletion=acompletion_fn)
    return module, completion_fn, acompletion_fn, response


def _make_chunk(text: str | None, *, model: str = "gpt-4o", usage=None, reasoning: str | None = None):
    delta = SimpleNamespace(content=text, reasoning_content=reasoning)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
    return SimpleNamespace(model=model, choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# Sync wrapper tests
# ---------------------------------------------------------------------------


class TestWrapLiteLLM:
    @respx.mock
    def test_sync_blocking_completion(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        module, completion_fn, *_ = _make_mock_litellm()
        wrapped = wrap_litellm(module)

        result = wrapped.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            max_tokens=100,
        )

        completion_fn.assert_called_once()
        assert result.choices[0].message.content == "Hello!"

    @respx.mock
    def test_returns_same_module(self):
        module, *_ = _make_mock_litellm()
        assert wrap_litellm(module) is module

    @respx.mock
    def test_idempotent_double_wrap(self):
        module, completion_fn, *_ = _make_mock_litellm()
        wrap_litellm(module)
        patched = module.completion
        wrap_litellm(module)
        # Second wrap is a no-op — the function is not wrapped again.
        assert module.completion is patched

    @respx.mock
    def test_error_propagates(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        module, completion_fn, *_ = _make_mock_litellm()
        completion_fn.side_effect = RuntimeError("API down")
        wrap_litellm(module)

        with pytest.raises(RuntimeError, match="API down"):
            module.completion(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    @respx.mock
    def test_returns_original_response(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        module, _, _, expected = _make_mock_litellm()
        wrap_litellm(module)
        result = module.completion(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])
        assert result is expected

    @respx.mock
    def test_standalone_trace_input_is_last_user_message(self):
        import json

        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        module, *_ = _make_mock_litellm()
        wrap_litellm(module)
        module.completion(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "What is 2+2?"},
            ],
        )
        pandaprobe.flush()

        assert route.called, "no trace was sent"
        body = json.loads(route.calls.last.request.content)
        # Standalone trace input is trimmed to the current turn's user message.
        assert body["input"] == {"messages": [{"role": "user", "content": "What is 2+2?"}]}


# ---------------------------------------------------------------------------
# Async wrapper tests
# ---------------------------------------------------------------------------


class TestAsyncWrapper:
    @respx.mock
    async def test_async_blocking(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        module, _, acompletion_fn, expected = _make_mock_litellm()
        wrap_litellm(module)

        result = await module.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
        )

        acompletion_fn.assert_awaited_once()
        assert result is expected

    @respx.mock
    async def test_async_blocking_error(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        module, _, acompletion_fn, _ = _make_mock_litellm()
        acompletion_fn.side_effect = RuntimeError("boom")
        wrap_litellm(module)

        with pytest.raises(RuntimeError, match="boom"):
            await module.acompletion(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestStreamingWrapper:
    @respx.mock
    def test_sync_streaming(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [_make_chunk("Hel"), _make_chunk("lo!"), _make_chunk(None, usage=usage)]

        module, completion_fn, *_ = _make_mock_litellm()
        completion_fn.return_value = iter(chunks)
        wrap_litellm(module)

        stream = module.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        collected = list(stream)
        assert len(collected) == 3

    @respx.mock
    async def test_async_streaming(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        async def _aiter(items):
            for it in items:
                yield it

        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [_make_chunk("Hel"), _make_chunk("lo"), _make_chunk(None, usage=usage)]

        module, _, acompletion_fn, _ = _make_mock_litellm()
        acompletion_fn.return_value = _aiter(chunks)
        wrap_litellm(module)

        stream = await module.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        collected = []
        async for chunk in stream:
            collected.append(chunk)
        assert len(collected) == 3


# ---------------------------------------------------------------------------
# Reasoning + token-detail extraction
# ---------------------------------------------------------------------------


class TestReasoningAndTokenDetails:
    @respx.mock
    def test_reasoning_tokens_surfaced(self, monkeypatch):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        from pandaprobe.tracing.span import SpanContext

        usage_calls: list[dict] = []
        original = SpanContext.set_token_usage

        def _capture(self, **kwargs):
            usage_calls.append(kwargs)
            return original(self, **kwargs)

        monkeypatch.setattr(SpanContext, "set_token_usage", _capture)

        details = SimpleNamespace(reasoning_tokens=7)
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            completion_tokens_details=details,
        )
        response = _make_mock_response()
        response.usage = usage

        module, completion_fn, *_ = _make_mock_litellm(completion_return=response)
        wrap_litellm(module)
        module.completion(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

        assert usage_calls
        assert usage_calls[-1] == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "reasoning_tokens": 7,
        }

    @respx.mock
    def test_reasoning_content_stored_as_metadata(self, monkeypatch):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        from pandaprobe.tracing.span import SpanContext

        metadata_calls: list[dict] = []
        original = SpanContext.set_metadata

        def _capture(self, data):
            metadata_calls.append(data)
            return original(self, data)

        monkeypatch.setattr(SpanContext, "set_metadata", _capture)

        response = _make_mock_response(reasoning="Let me think step by step.")
        module, *_ = _make_mock_litellm(completion_return=response)
        wrap_litellm(module)
        module.completion(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

        assert any(c.get("reasoning_summary") == "Let me think step by step." for c in metadata_calls)


# ---------------------------------------------------------------------------
# Span-finalize defensiveness — mirrors test_mistral. The reduce path must
# keep the setters and close_llm_span in separate try/except blocks so a
# setter failure can't leak the span open, and a close failure can't surface
# as an iteration error.
# ---------------------------------------------------------------------------


class TestStreamFinalizeDefensiveness:
    @respx.mock
    def test_setter_failure_in_reduce_stream_still_closes_span(self, monkeypatch):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        from pandaprobe.tracing.span import SpanContext
        from pandaprobe.wrappers.litellm import wrapper as litellm_wrapper

        close_calls: list[object] = []
        monkeypatch.setattr(litellm_wrapper, "close_llm_span", lambda ctx: close_calls.append(ctx))

        def _broken_set_token_usage(self, **kwargs):
            raise RuntimeError("set_token_usage exploded")

        monkeypatch.setattr(SpanContext, "set_token_usage", _broken_set_token_usage)

        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [_make_chunk("Hi"), _make_chunk(None, usage=usage)]

        module, completion_fn, *_ = _make_mock_litellm()
        completion_fn.return_value = iter(chunks)
        wrap_litellm(module)

        stream = module.completion(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}], stream=True)
        collected = list(stream)
        assert len(collected) == 2
        assert len(close_calls) == 1, "span did not close — setter failure leaked the span"

    @respx.mock
    def test_close_failure_in_reduce_stream_does_not_propagate(self, monkeypatch):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        from pandaprobe.wrappers.litellm import wrapper as litellm_wrapper

        def _broken_close(span_ctx):
            raise RuntimeError("close-blew-up")

        monkeypatch.setattr(litellm_wrapper, "close_llm_span", _broken_close)

        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [_make_chunk("Hi"), _make_chunk(None, usage=usage)]

        module, completion_fn, *_ = _make_mock_litellm()
        completion_fn.return_value = iter(chunks)
        wrap_litellm(module)

        stream = module.completion(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}], stream=True)
        collected = list(stream)
        assert len(collected) == 2


# ---------------------------------------------------------------------------
# Schema compliance — gates the universal-trace contract
# ---------------------------------------------------------------------------


class TestSchemaCompliance:
    @respx.mock
    def test_universal_schema_input_and_output(self, monkeypatch):
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

        module, *_ = _make_mock_litellm()
        wrap_litellm(module)
        module.completion(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hi"},
            ],
            temperature=0.5,
            max_tokens=50,
        )

        assert calls["input"] == {
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hi"},
            ]
        }
        out = calls["output"]
        assert isinstance(out, dict) and "messages" in out
        assert isinstance(out["messages"], list) and len(out["messages"]) == 1
        msg = out["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"
        assert calls["model"] == "gpt-4o"
        assert calls["usage"] == {
            "prompt_tokens": 12,
            "completion_tokens": 18,
            "total_tokens": 30,
        }
