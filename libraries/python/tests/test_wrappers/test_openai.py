"""Tests for pandaprobe.wrappers.openai."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.wrappers._base import extract_model_params
from pandaprobe.wrappers.openai import wrap_openai
from pandaprobe.wrappers.openai.utils import strip_not_given


@pytest.fixture(autouse=True)
def _setup_client():
    original = client_module._global_client
    pandaprobe.init(api_key="sk_pp_test", project_name="proj", endpoint="http://testserver", flush_interval=60.0)
    yield
    if client_module._global_client is not None:
        client_module._global_client.shutdown()
    client_module._global_client = original


class TestBaseUtilities:
    def test_strip_not_given_without_openai(self):
        d = {"a": 1, "b": None, "c": "hello"}
        assert strip_not_given(d) == d

    def test_extract_model_params(self):
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "api_key": "secret",
        }
        params = extract_model_params(kwargs)
        assert params == {"temperature": 0.7, "max_tokens": 100}
        assert "model" not in params
        assert "messages" not in params
        assert "api_key" not in params


def _make_mock_openai_client():
    """Build a mock that looks like openai.OpenAI()."""
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
    message = SimpleNamespace(role="assistant", content="Hello!")
    choice = SimpleNamespace(message=message, index=0, finish_reason="stop")
    response = SimpleNamespace(
        id="chatcmpl-123",
        model="gpt-4",
        choices=[choice],
        usage=usage,
    )

    create_fn = MagicMock(return_value=response)
    completions = SimpleNamespace(create=create_fn)
    chat = SimpleNamespace(completions=completions)
    client = SimpleNamespace(chat=chat)
    return client, create_fn


class TestWrapOpenAI:
    @respx.mock
    def test_sync_chat_completion(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        mock_client, create_fn = _make_mock_openai_client()
        wrapped = wrap_openai(mock_client)

        result = wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
        )

        create_fn.assert_called_once()
        assert result.choices[0].message.content == "Hello!"

    @respx.mock
    def test_returns_same_instance(self):
        mock_client, _ = _make_mock_openai_client()
        result = wrap_openai(mock_client)
        assert result is mock_client

    @respx.mock
    def test_error_propagates(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        mock_client, create_fn = _make_mock_openai_client()
        create_fn.side_effect = RuntimeError("API error")
        wrap_openai(mock_client)

        with pytest.raises(RuntimeError, match="API error"):
            mock_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "hi"}],
            )


class TestStreamingWrapper:
    @respx.mock
    def test_sync_streaming(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        chunk1 = SimpleNamespace(
            model="gpt-4",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"), index=0)],
            usage=None,
        )
        chunk2 = SimpleNamespace(
            model="gpt-4",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="lo!"), index=0)],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )

        create_fn = MagicMock(return_value=iter([chunk1, chunk2]))
        completions = SimpleNamespace(create=create_fn)
        chat = SimpleNamespace(completions=completions)
        mock_client = SimpleNamespace(chat=chat)

        wrap_openai(mock_client)
        stream = mock_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )

        chunks = list(stream)
        assert len(chunks) == 2
