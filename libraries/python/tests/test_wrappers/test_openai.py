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
from pandaprobe.wrappers.openai.utils import (
    extract_responses_params,
    normalize_responses_input,
    strip_not_given,
)


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

    @respx.mock
    def test_output_wrapped_in_messages(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        mock_client, create_fn = _make_mock_openai_client()
        wrap_openai(mock_client)

        mock_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
        )

    @respx.mock
    def test_standalone_trace_input_trimmed(self):
        """When wrap_openai creates a standalone trace, trace input should contain only the last user message."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        mock_client, create_fn = _make_mock_openai_client()
        wrap_openai(mock_client)

        mock_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "What about 3+3?"},
            ],
            temperature=0.7,
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


# ---------------------------------------------------------------------------
# Reasoning token extraction tests
# ---------------------------------------------------------------------------


class TestReasoningTokens:
    @respx.mock
    def test_chat_completion_captures_reasoning_tokens(self):
        """Reasoning tokens from completion_tokens_details should be included in token_usage."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        details = SimpleNamespace(reasoning_tokens=512)
        usage = SimpleNamespace(prompt_tokens=30, completion_tokens=66, completion_tokens_details=details)
        message = SimpleNamespace(role="assistant", content="Hello!")
        choice = SimpleNamespace(message=message, index=0, finish_reason="stop")
        response = SimpleNamespace(id="chatcmpl-r1", model="gpt-5.4-nano", choices=[choice], usage=usage)

        create_fn = MagicMock(return_value=response)
        completions = SimpleNamespace(create=create_fn)
        chat = SimpleNamespace(completions=completions)
        mock_client = SimpleNamespace(chat=chat)

        wrap_openai(mock_client)
        mock_client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="low",
        )

    @respx.mock
    def test_streaming_captures_reasoning_tokens(self):
        """Streaming usage with reasoning tokens should be captured."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        details = SimpleNamespace(reasoning_tokens=256)
        chunk1 = SimpleNamespace(
            model="gpt-5.4-nano",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"), index=0)],
            usage=None,
        )
        chunk2 = SimpleNamespace(
            model="gpt-5.4-nano",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="!"), index=0)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, completion_tokens_details=details),
        )

        create_fn = MagicMock(return_value=iter([chunk1, chunk2]))
        completions = SimpleNamespace(create=create_fn)
        chat = SimpleNamespace(completions=completions)
        mock_client = SimpleNamespace(chat=chat)

        wrap_openai(mock_client)
        stream = mock_client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        list(stream)


# ---------------------------------------------------------------------------
# Responses API base utility tests
# ---------------------------------------------------------------------------


class TestResponsesBaseUtilities:
    def test_normalize_responses_input_string(self):
        result = normalize_responses_input({"input": "Hello"})
        assert result == {"messages": [{"role": "user", "content": "Hello"}]}

    def test_normalize_responses_input_with_instructions(self):
        result = normalize_responses_input(
            {
                "instructions": "Be helpful.",
                "input": [{"role": "user", "content": "Hi"}],
            }
        )
        assert result == {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ]
        }

    def test_normalize_responses_input_list_of_messages(self):
        result = normalize_responses_input(
            {
                "input": [
                    {"role": "user", "content": "First"},
                    {"role": "assistant", "content": "Response"},
                    {"role": "user", "content": "Second"},
                ]
            }
        )
        assert len(result["messages"]) == 3

    def test_extract_responses_params(self):
        kwargs = {
            "model": "gpt-5.4-nano",
            "input": "Hello",
            "reasoning": {"effort": "low"},
            "max_output_tokens": 200,
            "api_key": "secret",
        }
        params = extract_responses_params(kwargs)
        assert "reasoning" in params
        assert "max_output_tokens" in params
        assert "model" not in params
        assert "input" not in params
        assert "api_key" not in params


# ---------------------------------------------------------------------------
# Responses API wrapper tests
# ---------------------------------------------------------------------------


def _make_mock_responses_client():
    """Build a mock that looks like openai.OpenAI() with responses.create."""
    output_tokens_details = SimpleNamespace(reasoning_tokens=128)
    usage = SimpleNamespace(
        input_tokens=30, output_tokens=50, output_tokens_details=output_tokens_details, total_tokens=80
    )

    reasoning_item = SimpleNamespace(
        type="reasoning",
        summary=[SimpleNamespace(type="summary_text", text="Thinking about the answer...")],
    )
    message_item = SimpleNamespace(
        type="message",
        role="assistant",
        content=[SimpleNamespace(type="output_text", text="A list is mutable, a tuple is not.")],
        status="completed",
    )

    response = SimpleNamespace(
        id="resp-123",
        model="gpt-5.4-nano",
        output=[reasoning_item, message_item],
        output_text="A list is mutable, a tuple is not.",
        usage=usage,
        status="completed",
    )

    create_fn = MagicMock(return_value=response)
    responses_ns = SimpleNamespace(create=create_fn)

    chat_completions = SimpleNamespace(create=MagicMock())
    chat = SimpleNamespace(completions=chat_completions)

    client = SimpleNamespace(chat=chat, responses=responses_ns)
    return client, create_fn, response


class TestResponsesAPI:
    @respx.mock
    def test_sync_responses_blocking(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        mock_client, create_fn, _ = _make_mock_responses_client()
        wrap_openai(mock_client)

        result = mock_client.responses.create(
            model="gpt-5.4-nano",
            input="What is a list vs tuple?",
            reasoning={"effort": "low"},
        )

        create_fn.assert_called_once()
        assert result.output_text == "A list is mutable, a tuple is not."

    @respx.mock
    def test_responses_returns_original_response(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        mock_client, _, expected = _make_mock_responses_client()
        wrap_openai(mock_client)

        result = mock_client.responses.create(
            model="gpt-5.4-nano",
            input="Hello",
        )
        assert result is expected

    @respx.mock
    def test_responses_error_propagates(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        mock_client, create_fn, _ = _make_mock_responses_client()
        create_fn.side_effect = RuntimeError("API error")
        wrap_openai(mock_client)

        with pytest.raises(RuntimeError, match="API error"):
            mock_client.responses.create(model="gpt-5.4-nano", input="Hello")


class TestResponsesToolSpans:
    @respx.mock
    def test_function_call_creates_tool_span(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        fn_call = SimpleNamespace(
            type="function_call",
            name="get_weather",
            arguments='{"city": "London"}',
            call_id="call_123",
        )
        message = SimpleNamespace(
            type="message",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text="The weather is nice.")],
            status="completed",
        )
        usage = SimpleNamespace(input_tokens=20, output_tokens=30, output_tokens_details=None, total_tokens=50)
        response = SimpleNamespace(
            id="resp-fn",
            model="gpt-5.4-nano",
            output=[fn_call, message],
            output_text="The weather is nice.",
            usage=usage,
            status="completed",
        )

        create_fn = MagicMock(return_value=response)
        responses_ns = SimpleNamespace(create=create_fn)
        chat_completions = SimpleNamespace(create=MagicMock())
        chat = SimpleNamespace(completions=chat_completions)
        mock_client = SimpleNamespace(chat=chat, responses=responses_ns)

        wrap_openai(mock_client)
        mock_client.responses.create(model="gpt-5.4-nano", input="Weather in London?")

    @respx.mock
    def test_web_search_creates_tool_span(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        search_call = SimpleNamespace(type="web_search_call", id="ws_123", status="completed")
        message = SimpleNamespace(
            type="message",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text="Found the answer.")],
            status="completed",
        )
        usage = SimpleNamespace(input_tokens=20, output_tokens=30, output_tokens_details=None, total_tokens=50)
        response = SimpleNamespace(
            id="resp-ws",
            model="gpt-5.4-nano",
            output=[search_call, message],
            output_text="Found the answer.",
            usage=usage,
            status="completed",
        )

        create_fn = MagicMock(return_value=response)
        responses_ns = SimpleNamespace(create=create_fn)
        chat_completions = SimpleNamespace(create=MagicMock())
        chat = SimpleNamespace(completions=chat_completions)
        mock_client = SimpleNamespace(chat=chat, responses=responses_ns)

        wrap_openai(mock_client)
        mock_client.responses.create(
            model="gpt-5.4-nano",
            input="Who is the president of France?",
            tools=[{"type": "web_search"}],
        )


class TestResponsesStreaming:
    @respx.mock
    def test_sync_streaming_responses(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        usage = SimpleNamespace(
            input_tokens=20,
            output_tokens=30,
            output_tokens_details=SimpleNamespace(reasoning_tokens=64),
            total_tokens=50,
        )
        completed_response = SimpleNamespace(
            id="resp-stream",
            model="gpt-5.4-nano",
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="Hello!")],
                    status="completed",
                ),
            ],
            output_text="Hello!",
            usage=usage,
            status="completed",
        )

        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta", delta="Hel"),
            SimpleNamespace(type="response.output_text.delta", delta="lo!"),
            SimpleNamespace(type="response.completed", response=completed_response),
        ]

        create_fn = MagicMock(return_value=iter(events))
        responses_ns = SimpleNamespace(create=create_fn)
        chat_completions = SimpleNamespace(create=MagicMock())
        chat = SimpleNamespace(completions=chat_completions)
        mock_client = SimpleNamespace(chat=chat, responses=responses_ns)

        wrap_openai(mock_client)
        stream = mock_client.responses.create(
            model="gpt-5.4-nano",
            input="Hello",
            stream=True,
        )

        collected = list(stream)
        assert len(collected) == 4
