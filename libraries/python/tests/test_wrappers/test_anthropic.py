"""Tests for pandaprobe.wrappers.anthropic."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.wrappers.anthropic import wrap_anthropic
from pandaprobe.wrappers.anthropic.utils import (
    extract_anthropic_params,
    normalize_anthropic_input,
    strip_not_given,
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


class TestAnthropicUtilities:
    def test_strip_not_given_without_anthropic(self):
        d = {"a": 1, "b": None, "c": "hello"}
        assert strip_not_given(d) == d

    def test_normalize_with_system_string(self):
        result = normalize_anthropic_input(
            {
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )
        assert result == {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
        }

    def test_normalize_without_system(self):
        result = normalize_anthropic_input({"messages": [{"role": "user", "content": "Hello"}]})
        assert result == {"messages": [{"role": "user", "content": "Hello"}]}

    def test_normalize_with_system_list(self):
        result = normalize_anthropic_input(
            {
                "system": [
                    {"type": "text", "text": "Be helpful."},
                    {"type": "text", "text": "Be concise."},
                ],
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"

    def test_normalize_multi_turn(self):
        result = normalize_anthropic_input(
            {
                "system": "Be helpful.",
                "messages": [
                    {"role": "user", "content": "First"},
                    {"role": "assistant", "content": "Response"},
                    {"role": "user", "content": "Second"},
                ],
            }
        )
        assert len(result["messages"]) == 4
        assert result["messages"][0]["role"] == "system"

    def test_extract_anthropic_params(self):
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "api_key": "secret",
            "system": "Be helpful.",
        }
        params = extract_anthropic_params(kwargs)
        assert params == {"temperature": 0.7, "max_tokens": 100}
        assert "model" not in params
        assert "messages" not in params
        assert "api_key" not in params
        assert "system" not in params


# ---------------------------------------------------------------------------
# Mock client helpers
# ---------------------------------------------------------------------------


def _make_mock_anthropic_client():
    """Build a mock that looks like anthropic.Anthropic()."""
    usage = SimpleNamespace(
        input_tokens=20,
        output_tokens=35,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    content_block = SimpleNamespace(type="text", text="Hello from Claude!")
    response = SimpleNamespace(
        id="msg-123",
        type="message",
        role="assistant",
        content=[content_block],
        model="claude-sonnet-4-20250514",
        stop_reason="end_turn",
        usage=usage,
    )

    create_fn = MagicMock(return_value=response)
    stream_fn = MagicMock()
    messages = SimpleNamespace(create=create_fn, stream=stream_fn)
    client = SimpleNamespace(messages=messages)
    return client, create_fn, stream_fn, response


# ---------------------------------------------------------------------------
# Wrapper tests
# ---------------------------------------------------------------------------


class TestWrapAnthropic:
    @respx.mock
    def test_sync_blocking_create(self):
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )
        mock_client, create_fn, _, _ = _make_mock_anthropic_client()
        wrapped = wrap_anthropic(mock_client)

        result = wrapped.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hello"}],
        )

        create_fn.assert_called_once()
        assert result.content[0].text == "Hello from Claude!"

    @respx.mock
    def test_returns_same_instance(self):
        mock_client, _, _, _ = _make_mock_anthropic_client()
        result = wrap_anthropic(mock_client)
        assert result is mock_client

    @respx.mock
    def test_error_propagates(self):
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )
        mock_client, create_fn, _, _ = _make_mock_anthropic_client()
        create_fn.side_effect = RuntimeError("API error")
        wrap_anthropic(mock_client)

        with pytest.raises(RuntimeError, match="API error"):
            mock_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=150,
                messages=[{"role": "user", "content": "Hello"}],
            )

    @respx.mock
    def test_returns_original_response(self):
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )
        mock_client, _, _, expected = _make_mock_anthropic_client()
        wrap_anthropic(mock_client)

        result = mock_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result is expected


# ---------------------------------------------------------------------------
# Streaming tests (stream=True on messages.create)
# ---------------------------------------------------------------------------


class TestAnthropicStreaming:
    @respx.mock
    def test_sync_streaming_create(self):
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )

        message_start_usage = SimpleNamespace(input_tokens=20)
        message_start_msg = SimpleNamespace(
            model="claude-sonnet-4-20250514",
            usage=message_start_usage,
        )
        events = [
            SimpleNamespace(type="message_start", message=message_start_msg),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="Hel"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="lo!"),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=10),
            ),
        ]

        create_fn = MagicMock(return_value=iter(events))
        stream_fn = MagicMock()
        messages = SimpleNamespace(create=create_fn, stream=stream_fn)
        mock_client = SimpleNamespace(messages=messages)

        wrap_anthropic(mock_client)
        stream = mock_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        collected = list(stream)
        assert len(collected) == 4


class TestAnthropicCacheTokens:
    @respx.mock
    def test_cache_tokens_in_usage(self):
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )

        usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=20,
        )
        content_block = SimpleNamespace(type="text", text="Cached response.")
        response = SimpleNamespace(
            id="msg-cache",
            type="message",
            role="assistant",
            content=[content_block],
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            usage=usage,
        )

        create_fn = MagicMock(return_value=response)
        stream_fn = MagicMock()
        messages = SimpleNamespace(create=create_fn, stream=stream_fn)
        mock_client = SimpleNamespace(messages=messages)

        wrap_anthropic(mock_client)
        mock_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{"role": "user", "content": "Hello"}],
        )


# ---------------------------------------------------------------------------
# Extended thinking tests
# ---------------------------------------------------------------------------


class TestAnthropicThinking:
    @respx.mock
    def test_thinking_blocks_extracted_to_metadata(self):
        """Thinking content blocks should be stored in metadata.reasoning_summary."""
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )

        usage = SimpleNamespace(
            input_tokens=30,
            output_tokens=200,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        thinking_block = SimpleNamespace(
            type="thinking",
            thinking="Let me reason step by step...",
            signature="abc123",
        )
        text_block = SimpleNamespace(type="text", text="The answer is 42.")
        response = SimpleNamespace(
            id="msg-think",
            type="message",
            role="assistant",
            content=[thinking_block, text_block],
            model="claude-sonnet-4-6",
            stop_reason="end_turn",
            usage=usage,
        )

        create_fn = MagicMock(return_value=response)
        stream_fn = MagicMock()
        messages = SimpleNamespace(create=create_fn, stream=stream_fn)
        mock_client = SimpleNamespace(messages=messages)

        wrap_anthropic(mock_client)
        result = mock_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            messages=[{"role": "user", "content": "What is the meaning of life?"}],
        )
        assert result is response

    @respx.mock
    def test_no_thinking_blocks_no_metadata(self):
        """When no thinking blocks exist, no reasoning_summary should be set."""
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )

        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        text_block = SimpleNamespace(type="text", text="Hello!")
        response = SimpleNamespace(
            id="msg-nothink",
            type="message",
            role="assistant",
            content=[text_block],
            model="claude-sonnet-4-6",
            stop_reason="end_turn",
            usage=usage,
        )

        create_fn = MagicMock(return_value=response)
        stream_fn = MagicMock()
        messages = SimpleNamespace(create=create_fn, stream=stream_fn)
        mock_client = SimpleNamespace(messages=messages)

        wrap_anthropic(mock_client)
        result = mock_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result is response

    @respx.mock
    def test_streaming_thinking_deltas(self):
        """Streaming with thinking_delta events should extract reasoning into metadata."""
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )

        message_start_usage = SimpleNamespace(input_tokens=20)
        message_start_msg = SimpleNamespace(
            model="claude-sonnet-4-6",
            usage=message_start_usage,
        )
        events = [
            SimpleNamespace(type="message_start", message=message_start_msg),
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="thinking", thinking="", signature=""),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="thinking_delta", thinking="Let me think..."),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="thinking_delta", thinking=" Step 2."),
            ),
            SimpleNamespace(type="content_block_stop", index=0),
            SimpleNamespace(
                type="content_block_start",
                index=1,
                content_block=SimpleNamespace(type="text", text=""),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(type="text_delta", text="The answer."),
            ),
            SimpleNamespace(type="content_block_stop", index=1),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=50),
            ),
        ]

        create_fn = MagicMock(return_value=iter(events))
        stream_fn = MagicMock()
        messages = SimpleNamespace(create=create_fn, stream=stream_fn)
        mock_client = SimpleNamespace(messages=messages)

        wrap_anthropic(mock_client)
        stream = mock_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            messages=[{"role": "user", "content": "Solve this problem"}],
            stream=True,
        )

        collected = list(stream)
        assert len(collected) == 9

    @respx.mock
    def test_multiple_thinking_blocks_joined(self):
        """Multiple thinking blocks should be joined with double newlines."""
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )

        usage = SimpleNamespace(
            input_tokens=30,
            output_tokens=200,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        thinking1 = SimpleNamespace(type="thinking", thinking="First thought.", signature="sig1")
        text1 = SimpleNamespace(type="text", text="Intermediate answer.")
        thinking2 = SimpleNamespace(type="thinking", thinking="Second thought.", signature="sig2")
        text2 = SimpleNamespace(type="text", text="Final answer.")
        response = SimpleNamespace(
            id="msg-interleaved",
            type="message",
            role="assistant",
            content=[thinking1, text1, thinking2, text2],
            model="claude-sonnet-4-6",
            stop_reason="end_turn",
            usage=usage,
        )

        create_fn = MagicMock(return_value=response)
        stream_fn = MagicMock()
        messages = SimpleNamespace(create=create_fn, stream=stream_fn)
        mock_client = SimpleNamespace(messages=messages)

        wrap_anthropic(mock_client)
        result = mock_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            messages=[{"role": "user", "content": "Complex problem"}],
        )
        assert result is response


class TestStreamManagerDoubleClose:
    """Verify that stream wrappers don't close the span twice."""

    @respx.mock
    def test_sync_stream_manager_no_double_close(self):
        """_SyncStreamWrapper._extract_final should prevent __exit__ from re-closing."""
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )

        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=50,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        final_msg = SimpleNamespace(
            id="msg-stream",
            type="message",
            role="assistant",
            content=[SimpleNamespace(type="text", text="Hello")],
            model="claude-sonnet-4-6",
            stop_reason="end_turn",
            usage=usage,
        )

        class FakeTextStream:
            def __init__(self):
                self._items = ["Hello"]

            def __iter__(self):
                return iter(self._items)

        class FakeMessageStream:
            def __init__(self):
                self.text_stream = FakeTextStream()

            def get_final_message(self):
                return final_msg

            def get_final_text(self):
                return "Hello"

        class FakeStreamCM:
            """Mimics the context manager returned by anthropic messages.stream()."""

            def __enter__(self):
                return FakeMessageStream()

            def __exit__(self, *exc_info):
                pass

        create_fn = MagicMock()
        stream_fn = MagicMock(return_value=FakeStreamCM())
        messages = SimpleNamespace(create=create_fn, stream=stream_fn)
        mock_client = SimpleNamespace(messages=messages)

        wrap_anthropic(mock_client)

        from pandaprobe.wrappers._base import close_llm_span
        from unittest.mock import patch

        close_calls: list[object] = []
        original_close = close_llm_span

        def tracking_close(span_ctx):
            close_calls.append(span_ctx)
            original_close(span_ctx)

        with patch("pandaprobe.wrappers.anthropic.wrapper.close_llm_span", tracking_close):
            with mock_client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hi"}],
            ) as stream:
                texts = list(stream.text_stream)
                assert texts == ["Hello"]

        assert len(close_calls) == 1, (
            f"close_llm_span should be called exactly once, got {len(close_calls)}"
        )
