"""Tests for pandaprobe.wrappers.bedrock."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.wrappers.bedrock import wrap_bedrock
from pandaprobe.wrappers.bedrock.utils import (
    extract_bedrock_params,
    map_converse_usage,
    map_invoke_model_usage,
    normalize_converse_input,
    normalize_invoke_model_input,
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


class TestBedrockUtilities:
    def test_normalize_converse_with_system_list(self):
        result = normalize_converse_input(
            {
                "system": [{"text": "Be helpful."}, {"text": "Be concise."}],
                "messages": [
                    {"role": "user", "content": [{"text": "Hello"}]},
                    {"role": "assistant", "content": [{"text": "Hi!"}]},
                ],
            }
        )
        assert result == {
            "messages": [
                {"role": "system", "content": "Be helpful.\nBe concise."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }

    def test_normalize_converse_no_system(self):
        result = normalize_converse_input({"messages": [{"role": "user", "content": [{"text": "Hi"}]}]})
        assert result == {"messages": [{"role": "user", "content": "Hi"}]}

    def test_normalize_converse_role_mapping(self):
        """The Converse 'model' role (rare) must be normalized to 'assistant'."""
        result = normalize_converse_input(
            {
                "messages": [
                    {"role": "user", "content": [{"text": "Hi"}]},
                    {"role": "model", "content": [{"text": "Hello"}]},
                ]
            }
        )
        assert result["messages"][1]["role"] == "assistant"

    def test_normalize_converse_multi_block_content(self):
        """Non-text-only content blocks must round-trip through safe_serialize."""
        result = normalize_converse_input(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": "Look at this image:"},
                            {"image": {"format": "png", "source": {"bytes": b"..."}}},
                        ],
                    }
                ]
            }
        )
        assert isinstance(result["messages"][0]["content"], list)
        assert result["messages"][0]["content"][0] == {"text": "Look at this image:"}

    def test_extract_bedrock_params_flattens_inference_config(self):
        params = extract_bedrock_params(
            {
                "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
                "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
                "inferenceConfig": {
                    "temperature": 0.5,
                    "topP": 0.9,
                    "maxTokens": 200,
                    "stopSequences": ["\nHuman:"],
                },
                "toolConfig": {"tools": []},
                "secret_value": "should-not-leak",
            }
        )
        assert params == {
            "temperature": 0.5,
            "topP": 0.9,
            "maxTokens": 200,
            "stopSequences": ["\nHuman:"],
            "toolConfig": {"tools": []},
        }
        assert "modelId" not in params
        assert "messages" not in params
        assert "secret_value" not in params

    def test_map_converse_usage(self):
        assert map_converse_usage({"inputTokens": 10, "outputTokens": 20, "totalTokens": 30}) == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
        assert map_converse_usage(
            {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30,
                "cacheReadInputTokens": 5,
                "cacheWriteInputTokens": 7,
            }
        ) == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "cache_read_tokens": 5,
            "cache_creation_tokens": 7,
        }

    def test_normalize_invoke_model_anthropic_body(self):
        body = json.dumps(
            {
                "system": "Be helpful.",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
            }
        ).encode("utf-8")
        result = normalize_invoke_model_input({"body": body})
        assert result == {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ]
        }

    def test_normalize_invoke_model_titan_body(self):
        body = json.dumps({"inputText": "Hi", "textGenerationConfig": {"maxTokenCount": 100}}).encode()
        result = normalize_invoke_model_input({"body": body})
        assert result == {"messages": [{"role": "user", "content": "Hi"}]}

    def test_normalize_invoke_model_string_body(self):
        result = normalize_invoke_model_input({"body": "raw prompt"})
        assert result == {"messages": [{"role": "user", "content": "raw prompt"}]}

    def test_map_invoke_model_usage_anthropic(self):
        assert map_invoke_model_usage({"usage": {"input_tokens": 10, "output_tokens": 20}}) == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
        }

    def test_map_invoke_model_usage_titan(self):
        assert map_invoke_model_usage({"inputTextTokenCount": 5, "results": [{"tokenCount": 7}]}) == {
            "prompt_tokens": 5,
            "completion_tokens": 7,
        }


# ---------------------------------------------------------------------------
# Mock client helpers
# ---------------------------------------------------------------------------


def _make_converse_response(text: str = "Hello from Claude!") -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 12, "outputTokens": 18, "totalTokens": 30},
        "metrics": {"latencyMs": 123},
    }


def _make_mock_bedrock_client(*, converse_return: dict | None = None):
    converse_fn = MagicMock(return_value=converse_return or _make_converse_response())
    converse_stream_fn = MagicMock()
    invoke_model_fn = MagicMock()
    invoke_model_stream_fn = MagicMock()
    client = SimpleNamespace(
        converse=converse_fn,
        converse_stream=converse_stream_fn,
        invoke_model=invoke_model_fn,
        invoke_model_with_response_stream=invoke_model_stream_fn,
    )
    return client, converse_fn, converse_stream_fn, invoke_model_fn, invoke_model_stream_fn


# ---------------------------------------------------------------------------
# Sync Converse blocking
# ---------------------------------------------------------------------------


class TestConverseBlocking:
    @respx.mock
    def test_sync_converse(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        client, converse_fn, *_ = _make_mock_bedrock_client()
        wrap_bedrock(client)

        response = client.converse(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            system=[{"text": "Be helpful."}],
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            inferenceConfig={"temperature": 0.5, "maxTokens": 100},
        )

        converse_fn.assert_called_once()
        assert response["output"]["message"]["content"][0]["text"] == "Hello from Claude!"

    @respx.mock
    def test_returns_same_instance(self):
        client, *_ = _make_mock_bedrock_client()
        assert wrap_bedrock(client) is client

    @respx.mock
    def test_error_propagates(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        client, converse_fn, *_ = _make_mock_bedrock_client()
        converse_fn.side_effect = RuntimeError("AWS error")
        wrap_bedrock(client)

        with pytest.raises(RuntimeError, match="AWS error"):
            client.converse(
                modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
                messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            )


# ---------------------------------------------------------------------------
# Sync Converse streaming
# ---------------------------------------------------------------------------


class TestConverseStreaming:
    @respx.mock
    def test_sync_streaming_reduction_and_dict_passthrough(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Hel"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": "lo!"}, "contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}},
        ]

        stream_response = {
            "stream": iter(events),
            "ResponseMetadata": {"RequestId": "abc"},
        }

        client, _, converse_stream_fn, *_ = _make_mock_bedrock_client()
        converse_stream_fn.return_value = stream_response
        wrap_bedrock(client)

        response = client.converse_stream(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
        )

        assert isinstance(response, dict)
        assert "stream" in response
        assert "ResponseMetadata" in response

        collected = list(response["stream"])
        assert len(collected) == 5


# ---------------------------------------------------------------------------
# InvokeModel (Anthropic-on-Bedrock and Titan body shapes)
# ---------------------------------------------------------------------------


class _FakeStreamingBody:
    def __init__(self, payload: bytes) -> None:
        self._buf = io.BytesIO(payload)

    def read(self, *args, **kwargs) -> bytes:
        return self._buf.read(*args, **kwargs)


class TestInvokeModelBlocking:
    @respx.mock
    def test_anthropic_on_bedrock_body(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        body = json.dumps(
            {
                "id": "msg_abc",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
                "model": "claude-3-5-haiku-20241022",
                "usage": {"input_tokens": 12, "output_tokens": 18},
            }
        ).encode()

        response = {
            "body": _FakeStreamingBody(body),
            "contentType": "application/json",
        }

        client, _, _, invoke_fn, _ = _make_mock_bedrock_client()
        invoke_fn.return_value = response
        wrap_bedrock(client)

        result = client.invoke_model(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                }
            ).encode(),
        )
        assert result is response
        invoke_fn.assert_called_once()

    @respx.mock
    def test_titan_body(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        body = json.dumps(
            {
                "inputTextTokenCount": 7,
                "results": [
                    {
                        "tokenCount": 12,
                        "outputText": "Hello from Titan",
                        "completionReason": "FINISH",
                    }
                ],
            }
        ).encode()

        response = {
            "body": _FakeStreamingBody(body),
            "contentType": "application/json",
        }

        client, _, _, invoke_fn, _ = _make_mock_bedrock_client()
        invoke_fn.return_value = response
        wrap_bedrock(client)

        result = client.invoke_model(
            modelId="amazon.titan-text-express-v1",
            body=json.dumps({"inputText": "Hi", "textGenerationConfig": {"maxTokenCount": 100}}).encode(),
        )
        assert result is response


# ---------------------------------------------------------------------------
# Streaming reasoning + error paths
# ---------------------------------------------------------------------------


class TestConverseExtraEvents:
    @respx.mock
    def test_streaming_reasoning_extracted_to_metadata(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "Thinking..."}}}},
            {"contentBlockDelta": {"delta": {"text": "Answer"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7}}},
        ]
        client, _, converse_stream_fn, *_ = _make_mock_bedrock_client()
        converse_stream_fn.return_value = {"stream": iter(events)}
        wrap_bedrock(client)

        resp = client.converse_stream(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
        )
        list(resp["stream"])

    @respx.mock
    def test_converse_stream_error_propagates(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        client, _, converse_stream_fn, *_ = _make_mock_bedrock_client()
        converse_stream_fn.side_effect = RuntimeError("aws-stream-down")
        wrap_bedrock(client)

        with pytest.raises(RuntimeError, match="aws-stream-down"):
            client.converse_stream(
                modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
                messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            )

    @respx.mock
    def test_invoke_model_error_propagates(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        client, _, _, invoke_fn, _ = _make_mock_bedrock_client()
        invoke_fn.side_effect = RuntimeError("invoke-down")
        wrap_bedrock(client)

        with pytest.raises(RuntimeError, match="invoke-down"):
            client.invoke_model(
                modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
                body=b"{}",
            )


class TestInvokeModelStreaming:
    @respx.mock
    def test_invoke_model_stream_sync_passthrough(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        events = [{"chunk": {"bytes": b"Hel"}}, {"chunk": {"bytes": b"lo!"}}]
        response = {"body": iter(events), "contentType": "application/json"}

        client, _, _, _, invoke_stream_fn = _make_mock_bedrock_client()
        invoke_stream_fn.return_value = response
        wrap_bedrock(client)

        result = client.invoke_model_with_response_stream(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            body=b'{"messages":[{"role":"user","content":"Hi"}]}',
        )
        assert isinstance(result, dict)
        collected = list(result["body"])
        assert len(collected) == 2


# ---------------------------------------------------------------------------
# Async Converse / async streaming via aioboto3-shaped mock
# ---------------------------------------------------------------------------


class TestAsyncBedrock:
    @respx.mock
    async def test_async_converse(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        client = SimpleNamespace(
            converse=AsyncMock(return_value=_make_converse_response()),
            converse_stream=AsyncMock(),
            invoke_model=AsyncMock(),
            invoke_model_with_response_stream=AsyncMock(),
        )
        wrap_bedrock(client)

        response = await client.converse(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            inferenceConfig={"temperature": 0.5, "maxTokens": 100},
        )
        assert response["output"]["message"]["content"][0]["text"] == "Hello from Claude!"

    @respx.mock
    async def test_async_invoke_model(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        body = json.dumps({"content": [{"type": "text", "text": "Hi!"}]}).encode()
        invoke_resp = {"body": _FakeStreamingBody(body), "contentType": "application/json"}

        client = SimpleNamespace(
            converse=AsyncMock(),
            converse_stream=AsyncMock(),
            invoke_model=AsyncMock(return_value=invoke_resp),
            invoke_model_with_response_stream=AsyncMock(),
        )
        wrap_bedrock(client)

        result = await client.invoke_model(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            body=b'{"messages":[]}',
        )
        assert result is invoke_resp

    @respx.mock
    async def test_async_converse_stream(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))

        async def _aiter(items):
            for it in items:
                yield it

        events = [
            {"contentBlockDelta": {"delta": {"text": "Hel"}}},
            {"contentBlockDelta": {"delta": {"text": "lo"}}},
            {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}}},
        ]
        stream_resp = {"stream": _aiter(events), "ResponseMetadata": {}}

        client = SimpleNamespace(
            converse=AsyncMock(),
            converse_stream=AsyncMock(return_value=stream_resp),
            invoke_model=AsyncMock(),
            invoke_model_with_response_stream=AsyncMock(),
        )
        wrap_bedrock(client)

        result = await client.converse_stream(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
        )
        collected = []
        async for ev in result["stream"]:
            collected.append(ev)
        assert len(collected) == 3


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

        client, *_ = _make_mock_bedrock_client()
        wrap_bedrock(client)
        client.converse(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            system=[{"text": "Be terse."}],
            messages=[{"role": "user", "content": [{"text": "Bonjour"}]}],
            inferenceConfig={"temperature": 0.5, "maxTokens": 50},
        )

        assert calls["input"] == {
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Bonjour"},
            ]
        }
        assert calls["output"] == {"messages": [{"role": "assistant", "content": "Hello from Claude!"}]}
        assert calls["model"] == "anthropic.claude-3-5-haiku-20241022-v1:0"
        assert calls["usage"] == {
            "prompt_tokens": 12,
            "completion_tokens": 18,
            "total_tokens": 30,
        }
