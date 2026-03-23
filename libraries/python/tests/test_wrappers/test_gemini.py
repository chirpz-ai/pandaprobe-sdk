"""Tests for pandaprobe.wrappers.gemini."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.wrappers.gemini import wrap_gemini
from pandaprobe.wrappers.gemini.utils import (
    convert_config_to_dict,
    extract_gemini_params,
    normalize_gemini_input,
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


class TestGeminiUtilities:
    def test_normalize_string_contents(self):
        result = normalize_gemini_input({"contents": "Hello"})
        assert result == {"messages": [{"role": "user", "content": "Hello"}]}

    def test_normalize_list_of_strings(self):
        result = normalize_gemini_input({"contents": ["Hello", "World"]})
        assert result == {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "World"},
            ]
        }

    def test_normalize_content_dicts_with_parts(self):
        result = normalize_gemini_input(
            {
                "contents": [
                    {"role": "user", "parts": [{"text": "What is AI?"}]},
                    {"role": "model", "parts": [{"text": "AI is..."}]},
                ]
            }
        )
        assert len(result["messages"]) == 2
        assert result["messages"][0] == {"role": "user", "content": "What is AI?"}
        assert result["messages"][1] == {"role": "assistant", "content": "AI is..."}

    def test_normalize_with_system_instruction_in_config(self):
        result = normalize_gemini_input(
            {
                "contents": "Hello",
                "config": {"system_instruction": "Be helpful."},
            }
        )
        assert result == {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
            ]
        }

    def test_normalize_empty_contents(self):
        result = normalize_gemini_input({})
        assert result == {"messages": []}

    def test_extract_gemini_params_from_config(self):
        kwargs = {
            "model": "gemini-2.5-flash",
            "contents": "Hello",
            "config": {
                "temperature": 0.7,
                "max_output_tokens": 200,
                "system_instruction": "Be helpful.",
            },
        }
        params = extract_gemini_params(kwargs)
        assert params == {"temperature": 0.7, "max_output_tokens": 200}
        assert "model" not in params
        assert "system_instruction" not in params

    def test_extract_gemini_params_from_top_level(self):
        kwargs = {"temperature": 0.5, "top_k": 40, "model": "gemini-2.5-flash"}
        params = extract_gemini_params(kwargs)
        assert params == {"temperature": 0.5, "top_k": 40}

    def test_convert_config_to_dict_passthrough(self):
        kwargs = {"config": {"temperature": 0.7}}
        assert convert_config_to_dict(kwargs) is kwargs

    def test_convert_config_to_dict_object(self):
        config_obj = SimpleNamespace(temperature=0.7, max_output_tokens=200, _internal="x")
        kwargs = {"config": config_obj}
        result = convert_config_to_dict(kwargs)
        assert isinstance(result["config"], dict)
        assert result["config"]["temperature"] == 0.7
        assert "_internal" not in result["config"]


# ---------------------------------------------------------------------------
# Mock client helpers
# ---------------------------------------------------------------------------


def _make_mock_gemini_client():
    """Build a mock that looks like google.genai.Client()."""
    usage_metadata = SimpleNamespace(
        prompt_token_count=15,
        candidates_token_count=25,
        total_token_count=40,
        thoughts_token_count=0,
        cached_content_token_count=0,
    )
    part = SimpleNamespace(text="Hello from Gemini!")
    content = SimpleNamespace(role="model", parts=[part])
    candidate = SimpleNamespace(content=content, finish_reason="STOP")
    response = SimpleNamespace(
        text="Hello from Gemini!",
        candidates=[candidate],
        usage_metadata=usage_metadata,
    )

    generate_fn = MagicMock(return_value=response)
    stream_fn = MagicMock()
    models = SimpleNamespace(
        generate_content=generate_fn,
        generate_content_stream=stream_fn,
    )
    client = SimpleNamespace(models=models)
    return client, generate_fn, stream_fn, response


# ---------------------------------------------------------------------------
# Wrapper tests
# ---------------------------------------------------------------------------


class TestWrapGemini:
    @respx.mock
    def test_sync_generate_content(self):
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )
        mock_client, generate_fn, _, _ = _make_mock_gemini_client()
        wrapped = wrap_gemini(mock_client)

        result = wrapped.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello",
        )

        generate_fn.assert_called_once()
        assert result.text == "Hello from Gemini!"

    @respx.mock
    def test_returns_same_instance(self):
        mock_client, _, _, _ = _make_mock_gemini_client()
        result = wrap_gemini(mock_client)
        assert result is mock_client

    @respx.mock
    def test_error_propagates(self):
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )
        mock_client, generate_fn, _, _ = _make_mock_gemini_client()
        generate_fn.side_effect = RuntimeError("API error")
        wrap_gemini(mock_client)

        with pytest.raises(RuntimeError, match="API error"):
            mock_client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Hello",
            )

    @respx.mock
    def test_output_wrapped_in_messages(self):
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )
        mock_client, _, _, _ = _make_mock_gemini_client()
        wrap_gemini(mock_client)

        mock_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello",
        )

    @respx.mock
    def test_returns_original_response(self):
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )
        mock_client, _, _, expected = _make_mock_gemini_client()
        wrap_gemini(mock_client)

        result = mock_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello",
        )
        assert result is expected


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestGeminiStreaming:
    @respx.mock
    def test_sync_streaming(self):
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )

        usage_metadata = SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
            thoughts_token_count=0,
            cached_content_token_count=0,
        )
        chunk1 = SimpleNamespace(text="Hel", usage_metadata=None)
        chunk2 = SimpleNamespace(text="lo!", usage_metadata=usage_metadata)

        stream_fn = MagicMock(return_value=iter([chunk1, chunk2]))
        generate_fn = MagicMock()
        models = SimpleNamespace(
            generate_content=generate_fn,
            generate_content_stream=stream_fn,
        )
        mock_client = SimpleNamespace(models=models)

        wrap_gemini(mock_client)
        stream = mock_client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents="Hello",
        )

        chunks = list(stream)
        assert len(chunks) == 2
        assert chunks[0].text == "Hel"
        assert chunks[1].text == "lo!"


class TestGeminiReasoningTokens:
    @respx.mock
    def test_reasoning_tokens_in_usage(self):
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )

        usage_metadata = SimpleNamespace(
            prompt_token_count=30,
            candidates_token_count=50,
            total_token_count=80,
            thoughts_token_count=256,
            cached_content_token_count=0,
        )
        part = SimpleNamespace(text="Thinking complete.", thought=False)
        content = SimpleNamespace(role="model", parts=[part])
        candidate = SimpleNamespace(content=content, finish_reason="STOP")
        response = SimpleNamespace(
            text="Thinking complete.",
            candidates=[candidate],
            usage_metadata=usage_metadata,
        )

        generate_fn = MagicMock(return_value=response)
        stream_fn = MagicMock()
        models = SimpleNamespace(
            generate_content=generate_fn,
            generate_content_stream=stream_fn,
        )
        mock_client = SimpleNamespace(models=models)

        wrap_gemini(mock_client)
        mock_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Think about this.",
        )


# ---------------------------------------------------------------------------
# Thought summary extraction tests
# ---------------------------------------------------------------------------


class TestGeminiThoughtSummary:
    @respx.mock
    def test_thought_parts_extracted_to_metadata(self):
        """Thought summary parts (part.thought=True) should be stored in metadata."""
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )

        usage_metadata = SimpleNamespace(
            prompt_token_count=20,
            candidates_token_count=40,
            total_token_count=60,
            thoughts_token_count=100,
            cached_content_token_count=0,
        )
        thought_part = SimpleNamespace(text="Let me reason about this...", thought=True)
        answer_part = SimpleNamespace(text="The answer is 42.", thought=False)
        content = SimpleNamespace(role="model", parts=[thought_part, answer_part])
        candidate = SimpleNamespace(content=content, finish_reason="STOP")
        response = SimpleNamespace(
            text="The answer is 42.",
            candidates=[candidate],
            usage_metadata=usage_metadata,
        )

        generate_fn = MagicMock(return_value=response)
        stream_fn = MagicMock()
        models = SimpleNamespace(
            generate_content=generate_fn,
            generate_content_stream=stream_fn,
        )
        mock_client = SimpleNamespace(models=models)

        wrap_gemini(mock_client)
        result = mock_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="What is the meaning of life?",
        )
        assert result is response

    @respx.mock
    def test_no_thought_parts_no_metadata(self):
        """When no thought parts exist, no reasoning_summary metadata should be set."""
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )

        usage_metadata = SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=30,
            thoughts_token_count=0,
            cached_content_token_count=0,
        )
        answer_part = SimpleNamespace(text="Hello!", thought=False)
        content = SimpleNamespace(role="model", parts=[answer_part])
        candidate = SimpleNamespace(content=content, finish_reason="STOP")
        response = SimpleNamespace(
            text="Hello!",
            candidates=[candidate],
            usage_metadata=usage_metadata,
        )

        generate_fn = MagicMock(return_value=response)
        stream_fn = MagicMock()
        models = SimpleNamespace(
            generate_content=generate_fn,
            generate_content_stream=stream_fn,
        )
        mock_client = SimpleNamespace(models=models)

        wrap_gemini(mock_client)
        result = mock_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello",
        )
        assert result is response

    @respx.mock
    def test_streaming_thought_parts(self):
        """Streaming chunks with thought parts should be extracted into metadata."""
        respx.post("http://testserver/traces").mock(
            return_value=httpx.Response(202, json={}),
        )

        usage_metadata = SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=25,
            thoughts_token_count=10,
            cached_content_token_count=0,
        )
        chunk1 = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text="Thinking...", thought=True)],
                    ),
                )
            ],
            usage_metadata=None,
        )
        chunk2 = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text="Answer!", thought=False)],
                    ),
                )
            ],
            usage_metadata=usage_metadata,
        )

        stream_fn = MagicMock(return_value=iter([chunk1, chunk2]))
        generate_fn = MagicMock()
        models = SimpleNamespace(
            generate_content=generate_fn,
            generate_content_stream=stream_fn,
        )
        mock_client = SimpleNamespace(models=models)

        wrap_gemini(mock_client)
        stream = mock_client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents="Hello",
        )

        chunks = list(stream)
        assert len(chunks) == 2
