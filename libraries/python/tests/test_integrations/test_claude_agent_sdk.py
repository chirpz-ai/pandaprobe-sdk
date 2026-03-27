"""Tests for pandaprobe.integrations.claude_agent_sdk."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.integrations.claude_agent_sdk.adapter import (
    ClaudeAgentSDKAdapter,
    _TraceState,
    _current_span_id,
    _current_trace_state,
    _get_adapter,
    _get_current_span,
    _get_trace_state,
    _handle_assistant_message,
    _handle_result_message,
    _handle_user_message,
    _make_tool_hooks,
    _set_current_span,
    _set_trace_state,
    _store_adapter,
    _wrap_client_query,
    _wrap_receive_response,
    _wrap_standalone_query,
)
from pandaprobe.integrations.claude_agent_sdk.utils import (
    extract_model_parameters,
    extract_thinking_from_content,
    extract_token_usage,
    flatten_content_blocks,
    normalize_content_to_text,
    safe_serialize,
    serialize_tool_response,
    strip_thinking_blocks,
)
from pandaprobe.schemas import SpanKind, SpanStatusCode


@pytest.fixture(autouse=True)
def _setup_client():
    original_client = client_module._global_client
    original_flag = client_module._auto_init_attempted
    pandaprobe.init(
        api_key="sk_pp_test",
        project_name="proj",
        endpoint="http://testserver",
        flush_interval=60.0,
    )
    yield
    if client_module._global_client is not None:
        client_module._global_client.shutdown()
    client_module._global_client = original_client
    client_module._auto_init_attempted = original_flag


# ---------------------------------------------------------------------------
# Mock Claude Agent SDK objects
# ---------------------------------------------------------------------------

# Named classes so type(obj).__name__ returns the expected SDK class name.


class _TextBlock:
    __name__ = "TextBlock"

    def __init__(self, text: str = "Hello"):
        self.text = text


_TextBlock.__name__ = "TextBlock"
_TextBlock.__qualname__ = "TextBlock"


class _ThinkingBlock:
    def __init__(self, thinking: str = "Let me think...", signature: str = "sig123"):
        self.thinking = thinking
        self.signature = signature


_ThinkingBlock.__name__ = "ThinkingBlock"
_ThinkingBlock.__qualname__ = "ThinkingBlock"


class _ToolUseBlock:
    def __init__(self, id: str = "tu_1", name: str = "search", input: dict | None = None):
        self.id = id
        self.name = name
        self.input = input or {}


_ToolUseBlock.__name__ = "ToolUseBlock"
_ToolUseBlock.__qualname__ = "ToolUseBlock"


class _ToolResultBlock:
    def __init__(self, tool_use_id: str = "tu_1", content: str = "result text", is_error: bool = False):
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


_ToolResultBlock.__name__ = "ToolResultBlock"
_ToolResultBlock.__qualname__ = "ToolResultBlock"


class _AssistantMessage:
    def __init__(self, content=None, model="claude-sonnet-4-20250514", parent_tool_use_id=None, error=None):
        self.content = content or [_TextBlock("Hello")]
        self.model = model
        self.parent_tool_use_id = parent_tool_use_id
        self.error = error


_AssistantMessage.__name__ = "AssistantMessage"
_AssistantMessage.__qualname__ = "AssistantMessage"


class _UserMessage:
    def __init__(self, content=None, parent_tool_use_id=None, tool_use_result=None):
        self.content = content
        self.parent_tool_use_id = parent_tool_use_id
        self.tool_use_result = tool_use_result


_UserMessage.__name__ = "UserMessage"
_UserMessage.__qualname__ = "UserMessage"


class _ResultMessage:
    def __init__(
        self,
        usage=None,
        num_turns=1,
        session_id="sess-1",
        is_error=False,
        duration_ms=1000,
        duration_api_ms=800,
        total_cost_usd=None,
        result=None,
    ):
        self.usage = usage
        self.num_turns = num_turns
        self.session_id = session_id
        self.is_error = is_error
        self.duration_ms = duration_ms
        self.duration_api_ms = duration_api_ms
        self.total_cost_usd = total_cost_usd
        self.result = result
        self.subtype = "success"


_ResultMessage.__name__ = "ResultMessage"
_ResultMessage.__qualname__ = "ResultMessage"


def _make_text_block(text: str = "Hello"):
    return _TextBlock(text)


def _make_thinking_block(thinking: str = "Let me think...", signature: str = "sig123"):
    return _ThinkingBlock(thinking, signature)


def _make_tool_use_block(tool_id: str = "tu_1", name: str = "search", tool_input: dict | None = None):
    return _ToolUseBlock(tool_id, name, tool_input)


def _make_tool_result_block(tool_use_id: str = "tu_1", content: str = "result text", is_error: bool = False):
    return _ToolResultBlock(tool_use_id, content, is_error)


def _make_assistant_message(
    content: list | None = None,
    model: str = "claude-sonnet-4-20250514",
    parent_tool_use_id: str | None = None,
    error: str | None = None,
):
    return _AssistantMessage(content, model, parent_tool_use_id, error)


def _make_user_message(
    content: str | list | None = None,
    parent_tool_use_id: str | None = None,
    tool_use_result: dict | None = None,
):
    return _UserMessage(content, parent_tool_use_id, tool_use_result)


def _make_result_message(
    usage: dict | None = None,
    num_turns: int = 1,
    session_id: str = "sess-1",
    is_error: bool = False,
    duration_ms: int = 1000,
    duration_api_ms: int = 800,
    total_cost_usd: float | None = None,
    result: str | None = None,
):
    return _ResultMessage(usage, num_turns, session_id, is_error, duration_ms, duration_api_ms, total_cost_usd, result)


def _make_options(
    system_prompt: str | dict | None = None,
    model: str | None = None,
    max_turns: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    thinking: dict | None = None,
    hooks: dict | None = None,
):
    return SimpleNamespace(
        system_prompt=system_prompt,
        model=model,
        max_turns=max_turns,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        hooks=hooks,
        permission_mode=None,
    )


# ===================================================================
# Utils tests
# ===================================================================


class TestSafeSerialize:
    def test_primitives(self):
        assert safe_serialize("hello") == "hello"
        assert safe_serialize(42) == 42
        assert safe_serialize(None) is None
        assert safe_serialize(True) is True

    def test_dict_and_list(self):
        assert safe_serialize({"a": 1}) == {"a": 1}
        assert safe_serialize([1, 2]) == [1, 2]

    def test_model_dump(self):
        obj = SimpleNamespace()
        obj.model_dump = lambda: {"key": "value"}
        assert safe_serialize(obj) == {"key": "value"}

    def test_fallback_repr(self):
        result = safe_serialize(object())
        assert isinstance(result, str)


class TestFlattenContentBlocks:
    def test_text_block(self):
        blocks = [_make_text_block("Hello")]
        result = flatten_content_blocks(blocks)
        assert result == [{"type": "text", "text": "Hello"}]

    def test_thinking_block(self):
        blocks = [_make_thinking_block("Let me think...", "sig")]
        result = flatten_content_blocks(blocks)
        assert result == [{"type": "thinking", "thinking": "Let me think...", "signature": "sig"}]

    def test_tool_use_block(self):
        blocks = [_make_tool_use_block("tu_1", "search", {"query": "test"})]
        result = flatten_content_blocks(blocks)
        assert result == [{"type": "tool_use", "id": "tu_1", "name": "search", "input": {"query": "test"}}]

    def test_tool_result_block(self):
        blocks = [_make_tool_result_block("tu_1", "found it", False)]
        result = flatten_content_blocks(blocks)
        assert result == [{"type": "tool_result", "tool_use_id": "tu_1", "content": "found it", "is_error": False}]

    def test_mixed_blocks(self):
        blocks = [
            _make_thinking_block("Think..."),
            _make_text_block("Answer"),
            _make_tool_use_block("tu_1", "calc", {"x": 1}),
        ]
        result = flatten_content_blocks(blocks)
        assert len(result) == 3
        assert result[0]["type"] == "thinking"
        assert result[1]["type"] == "text"
        assert result[2]["type"] == "tool_use"

    def test_string_content(self):
        result = flatten_content_blocks("plain text")
        assert result == [{"type": "text", "text": "plain text"}]

    def test_none_content(self):
        result = flatten_content_blocks(None)
        assert result == []

    def test_dict_passthrough(self):
        blocks = [{"type": "text", "text": "already a dict"}]
        result = flatten_content_blocks(blocks)
        assert result == [{"type": "text", "text": "already a dict"}]


class TestExtractThinkingFromContent:
    def test_extracts_from_dataclass_blocks(self):
        blocks = [_make_thinking_block("Step 1"), _make_text_block("Answer")]
        result = extract_thinking_from_content(blocks)
        assert result == "Step 1"

    def test_extracts_from_dicts(self):
        blocks = [
            {"type": "thinking", "thinking": "My reasoning"},
            {"type": "text", "text": "The answer"},
        ]
        result = extract_thinking_from_content(blocks)
        assert result == "My reasoning"

    def test_multiple_thinking_blocks(self):
        blocks = [
            _make_thinking_block("Part A"),
            _make_thinking_block("Part B"),
            _make_text_block("Final"),
        ]
        result = extract_thinking_from_content(blocks)
        assert result == "Part A\n\nPart B"

    def test_no_thinking(self):
        blocks = [_make_text_block("No thinking here")]
        result = extract_thinking_from_content(blocks)
        assert result is None

    def test_none_content(self):
        assert extract_thinking_from_content(None) is None

    def test_string_content(self):
        assert extract_thinking_from_content("not a list") is None


class TestStripThinkingBlocks:
    def test_strips_thinking_dicts(self):
        content = [
            {"type": "thinking", "thinking": "Hmm"},
            {"type": "text", "text": "The answer"},
        ]
        result = strip_thinking_blocks(content)
        assert result == "The answer"

    def test_multiple_text_blocks_joined(self):
        content = [
            {"type": "thinking", "thinking": "Hmm"},
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]
        result = strip_thinking_blocks(content)
        assert result == "Part 1 Part 2"

    def test_non_text_blocks_preserved(self):
        content = [
            {"type": "thinking", "thinking": "Hmm"},
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "name": "search"},
        ]
        result = strip_thinking_blocks(content)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Hello"}

    def test_string_passthrough(self):
        assert strip_thinking_blocks("plain text") == "plain text"

    def test_all_thinking_returns_original(self):
        content = [{"type": "thinking", "thinking": "Only thoughts"}]
        result = strip_thinking_blocks(content)
        assert result == content


class TestNormalizeContentToText:
    def test_string(self):
        assert normalize_content_to_text("hello") == "hello"

    def test_none(self):
        assert normalize_content_to_text(None) is None

    def test_text_blocks(self):
        blocks = [_make_text_block("Part 1"), _make_text_block("Part 2")]
        assert normalize_content_to_text(blocks) == "Part 1 Part 2"

    def test_skips_thinking_blocks(self):
        blocks = [_make_thinking_block("Think"), _make_text_block("Answer")]
        assert normalize_content_to_text(blocks) == "Answer"

    def test_dict_text_blocks(self):
        blocks = [{"type": "text", "text": "Hello"}, {"type": "thinking", "thinking": "..."}]
        assert normalize_content_to_text(blocks) == "Hello"

    def test_empty_list(self):
        assert normalize_content_to_text([]) is None


class TestExtractTokenUsage:
    def test_full_usage(self):
        usage = {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 20}
        result = extract_token_usage(usage)
        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cache_read_tokens": 20,
        }

    def test_basic_usage(self):
        usage = {"input_tokens": 100, "output_tokens": 50}
        result = extract_token_usage(usage)
        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_no_cache(self):
        usage = {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0}
        result = extract_token_usage(usage)
        assert "cache_read_tokens" not in result

    def test_none_usage(self):
        assert extract_token_usage(None) is None

    def test_empty_usage(self):
        assert extract_token_usage({}) is None

    def test_object_usage(self):
        usage = SimpleNamespace(input_tokens=200, output_tokens=80, cache_read_input_tokens=None)
        result = extract_token_usage(usage)
        assert result == {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}


class TestExtractModelParameters:
    def test_extracts_safe_params(self):
        options = _make_options(temperature=0.7, max_tokens=1024)
        result = extract_model_parameters(options)
        assert result == {"temperature": 0.7, "max_tokens": 1024}

    def test_extracts_thinking_config(self):
        options = _make_options(temperature=0.5, thinking={"type": "enabled", "budget_tokens": 10000})
        result = extract_model_parameters(options)
        assert result["temperature"] == 0.5
        assert result["thinking_config"] == {"type": "enabled", "budget_tokens": 10000}

    def test_no_params(self):
        options = _make_options()
        assert extract_model_parameters(options) is None

    def test_none_options(self):
        assert extract_model_parameters(None) is None

    def test_filters_unsafe_keys(self):
        options = SimpleNamespace(
            temperature=0.7,
            model="claude-sonnet-4-20250514",
            system_prompt="Be helpful",
            permission_mode="acceptEdits",
        )
        result = extract_model_parameters(options)
        assert result == {"temperature": 0.7}


class TestSerializeToolResponse:
    def test_dict_response(self):
        assert serialize_tool_response({"key": "val"}) == '{"key": "val"}'

    def test_list_response(self):
        assert serialize_tool_response([1, 2, 3]) == "[1, 2, 3]"

    def test_string_response(self):
        assert serialize_tool_response("hello") == "hello"

    def test_none_response(self):
        assert serialize_tool_response(None) == ""

    def test_zero_preserved(self):
        assert serialize_tool_response(0) == "0"

    def test_false_preserved(self):
        assert serialize_tool_response(False) == "False"


# ===================================================================
# Adapter tests
# ===================================================================


class TestClaudeAgentSDKAdapter:
    def test_extends_base(self):
        adapter = ClaudeAgentSDKAdapter(
            session_id="s1",
            user_id="u1",
            tags=["t"],
            metadata={"k": "v"},
        )
        assert adapter._session_id == "s1"
        assert adapter._user_id == "u1"
        assert adapter._tags == ["t"]
        assert adapter._metadata == {"k": "v"}

    def test_instrument_without_sdk_returns_false(self):
        import pandaprobe.integrations.claude_agent_sdk.adapter as mod

        original = mod._patched
        mod._patched = False
        try:
            adapter = ClaudeAgentSDKAdapter()
            with patch.dict("sys.modules", {"claude_agent_sdk": None}):
                result = adapter.instrument()
            assert result is False
        finally:
            mod._patched = original

    def test_default_values(self):
        adapter = ClaudeAgentSDKAdapter()
        assert adapter._session_id is None
        assert adapter._user_id is None
        assert adapter._tags == []
        assert adapter._metadata == {}

    def test_custom_values(self):
        adapter = ClaudeAgentSDKAdapter(
            session_id="s",
            user_id="u",
            tags=["a", "b"],
            metadata={"key": "val"},
        )
        assert adapter._session_id == "s"
        assert adapter._user_id == "u"
        assert adapter._tags == ["a", "b"]
        assert adapter._metadata == {"key": "val"}


class TestAdapterStorage:
    def test_store_and_get(self):
        adapter = ClaudeAgentSDKAdapter()
        _store_adapter(adapter)
        assert _get_adapter() is adapter

    def test_store_replaces(self):
        a1 = ClaudeAgentSDKAdapter()
        a2 = ClaudeAgentSDKAdapter()
        _store_adapter(a1)
        assert _get_adapter() is a1
        _store_adapter(a2)
        assert _get_adapter() is a2


# ===================================================================
# Context var tests
# ===================================================================


class TestContextVars:
    def test_trace_state(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)
        token = _set_trace_state(state)
        assert _get_trace_state() is state
        _current_trace_state.reset(token)
        assert _get_trace_state() is None

    def test_current_span(self):
        sid = "span-123"
        token = _set_current_span(sid)
        assert _get_current_span() == sid
        _current_span_id.reset(token)
        assert _get_current_span() is None

    def test_instance_state_fallback(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)
        client = SimpleNamespace(_pandaprobe_trace_state=state)
        resolve = lambda: _current_trace_state.get(None) or getattr(client, "_pandaprobe_trace_state", None)  # noqa: E731
        assert resolve() is state
        client._pandaprobe_trace_state = None
        assert resolve() is None


# ===================================================================
# Wrapper: _wrap_client_query
# ===================================================================


class TestWrapClientQuery:
    @pytest.mark.asyncio
    async def test_captures_prompt(self):
        instance = SimpleNamespace()
        called = False

        async def mock_query(*args, **kwargs):
            nonlocal called
            called = True

        await _wrap_client_query(mock_query, instance, ("Hello Claude",), {})
        assert called
        assert instance._pandaprobe_prompt == "Hello Claude"
        assert hasattr(instance, "_pandaprobe_start_time")

    @pytest.mark.asyncio
    async def test_non_string_prompt(self):
        instance = SimpleNamespace()

        async def mock_query(*args, **kwargs):
            pass

        await _wrap_client_query(mock_query, instance, (["stream"],), {})
        assert instance._pandaprobe_prompt is None


# ===================================================================
# Wrapper: _wrap_receive_response
# ===================================================================


class TestWrapReceiveResponse:
    @respx.mock
    @pytest.mark.asyncio
    async def test_creates_trace_with_chain_and_agent_spans(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = ClaudeAgentSDKAdapter(session_id="sess-1", user_id="user-1", tags=["test"])
        _store_adapter(adapter)

        messages = [
            _make_assistant_message([_make_text_block("Hello!")], model="claude-sonnet-4-20250514"),
            _make_result_message(usage={"input_tokens": 10, "output_tokens": 20}),
        ]

        async def mock_receive(*args, **kwargs):
            for m in messages:
                yield m

        instance = SimpleNamespace(
            options=_make_options(system_prompt="Be helpful", model="claude-sonnet-4-20250514"),
            _pandaprobe_prompt="Hi there",
            _pandaprobe_start_time=None,
        )

        collected = []
        async for msg in _wrap_receive_response(mock_receive, instance, (), {}):
            collected.append(msg)

        assert len(collected) == 2
        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_span_hierarchy(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = ClaudeAgentSDKAdapter()
        _store_adapter(adapter)

        messages = [
            _make_assistant_message([_make_text_block("Response")], model="claude-sonnet-4-20250514"),
            _make_result_message(usage={"input_tokens": 5, "output_tokens": 10}),
        ]

        async def mock_receive(*args, **kwargs):
            for m in messages:
                yield m

        instance = SimpleNamespace(
            options=_make_options(model="claude-sonnet-4-20250514"),
            _pandaprobe_prompt="Hello",
            _pandaprobe_start_time=None,
        )

        collected = []
        async for msg in _wrap_receive_response(mock_receive, instance, (), {}):
            collected.append(msg)

        # Verify we can still access collected messages
        assert len(collected) == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_error_produces_error_trace(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = ClaudeAgentSDKAdapter()
        _store_adapter(adapter)

        async def mock_receive(*args, **kwargs):
            yield _make_assistant_message([_make_text_block("partial")])
            raise RuntimeError("connection lost")

        instance = SimpleNamespace(
            options=None,
            _pandaprobe_prompt="Hi",
            _pandaprobe_start_time=None,
        )

        with pytest.raises(RuntimeError, match="connection lost"):
            async for _ in _wrap_receive_response(mock_receive, instance, (), {}):
                pass

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_adapter_passthrough(self):
        _store_adapter(None)

        messages = [_make_assistant_message([_make_text_block("hi")])]

        async def mock_receive(*args, **kwargs):
            for m in messages:
                yield m

        collected = []
        async for msg in _wrap_receive_response(mock_receive, SimpleNamespace(), (), {}):
            collected.append(msg)
        assert len(collected) == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_thinking_extraction(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = ClaudeAgentSDKAdapter()
        _store_adapter(adapter)

        content = [_make_thinking_block("My reasoning"), _make_text_block("The answer")]
        messages = [
            _make_assistant_message(content, model="claude-sonnet-4-20250514"),
            _make_result_message(),
        ]

        async def mock_receive(*args, **kwargs):
            for m in messages:
                yield m

        instance = SimpleNamespace(
            options=_make_options(model="claude-sonnet-4-20250514"),
            _pandaprobe_prompt="Question",
            _pandaprobe_start_time=None,
        )

        async for _ in _wrap_receive_response(mock_receive, instance, (), {}):
            pass

    @respx.mock
    @pytest.mark.asyncio
    async def test_system_prompt_preset(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = ClaudeAgentSDKAdapter()
        _store_adapter(adapter)

        messages = [
            _make_assistant_message([_make_text_block("Hello")]),
            _make_result_message(),
        ]

        async def mock_receive(*args, **kwargs):
            for m in messages:
                yield m

        instance = SimpleNamespace(
            options=_make_options(
                system_prompt={"type": "preset", "preset": "claude_code", "append": "Extra instructions"},
            ),
            _pandaprobe_prompt="Hi",
            _pandaprobe_start_time=None,
        )

        async for _ in _wrap_receive_response(mock_receive, instance, (), {}):
            pass


# ===================================================================
# Message handler tests
# ===================================================================


class TestHandleAssistantMessage:
    def test_creates_llm_span(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)
        agent_span_id = str(uuid4())
        state.agent_span_id = agent_span_id

        msg = _make_assistant_message(
            [_make_text_block("The answer is 42")],
            model="claude-sonnet-4-20250514",
        )

        from datetime import datetime, timezone

        _handle_assistant_message(msg, state, "claude-sonnet-4-20250514", None, datetime.now(timezone.utc))

        assert len(state.spans) == 1
        span = list(state.spans.values())[0]
        assert span.kind == SpanKind.LLM
        assert span.model == "claude-sonnet-4-20250514"
        assert span.name == "claude-sonnet-4-20250514"
        assert str(span.parent_span_id) == agent_span_id
        assert span.status == SpanStatusCode.OK
        assert span.output == {"messages": [{"role": "assistant", "content": "The answer is 42"}]}

    def test_thinking_in_metadata(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)
        state.agent_span_id = str(uuid4())

        content = [_make_thinking_block("Deep thought"), _make_text_block("42")]
        msg = _make_assistant_message(content, model="claude-sonnet-4-20250514")

        from datetime import datetime, timezone

        _handle_assistant_message(msg, state, "claude-sonnet-4-20250514", None, datetime.now(timezone.utc))

        span = list(state.spans.values())[0]
        assert span.metadata["reasoning_summary"] == "Deep thought"
        assert span.output["messages"][0]["content"] == "42"

    def test_tool_calls_in_output(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)
        state.agent_span_id = str(uuid4())

        content = [
            _make_text_block("Let me search"),
            _make_tool_use_block("tu_1", "search", {"query": "test"}),
        ]
        msg = _make_assistant_message(content, model="claude-sonnet-4-20250514")

        from datetime import datetime, timezone

        _handle_assistant_message(msg, state, "claude-sonnet-4-20250514", None, datetime.now(timezone.utc))

        span = list(state.spans.values())[0]
        output_msg = span.output["messages"][0]
        assert output_msg["tool_calls"] == [{"name": "search", "arguments": {"query": "test"}}]

    def test_error_on_assistant_message(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)
        state.agent_span_id = str(uuid4())

        msg = _make_assistant_message(
            [_make_text_block("")],
            model="claude-sonnet-4-20250514",
            error="rate_limit",
        )

        from datetime import datetime, timezone

        _handle_assistant_message(msg, state, "claude-sonnet-4-20250514", None, datetime.now(timezone.utc))

        span = list(state.spans.values())[0]
        assert span.status == SpanStatusCode.ERROR
        assert span.error == "rate_limit"


class TestThinkingOnlyBuffering:
    """Thinking-only AssistantMessages should NOT create LLM spans."""

    def test_thinking_only_message_is_buffered(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)
        state.agent_span_id = str(uuid4())

        thinking_msg = _make_assistant_message(
            [_make_thinking_block("Deep thought")],
            model="claude-sonnet-4-20250514",
        )

        from datetime import datetime, timezone

        _handle_assistant_message(thinking_msg, state, "claude-sonnet-4-20250514", None, datetime.now(timezone.utc))

        assert len(state.spans) == 0
        assert state.pending_thinking == "Deep thought"

    def test_buffered_thinking_applied_to_next_span(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)
        state.agent_span_id = str(uuid4())

        from datetime import datetime, timezone

        thinking_msg = _make_assistant_message(
            [_make_thinking_block("My reasoning")],
            model="claude-sonnet-4-20250514",
        )
        _handle_assistant_message(thinking_msg, state, "claude-sonnet-4-20250514", None, datetime.now(timezone.utc))

        text_msg = _make_assistant_message(
            [_make_text_block("The answer is 42")],
            model="claude-sonnet-4-20250514",
        )
        _handle_assistant_message(text_msg, state, "claude-sonnet-4-20250514", None, datetime.now(timezone.utc))

        assert len(state.spans) == 1
        span = list(state.spans.values())[0]
        assert span.metadata["reasoning_summary"] == "My reasoning"
        assert span.output["messages"][0]["content"] == "The answer is 42"
        assert state.pending_thinking is None


class TestHandleUserMessage:
    def test_tool_result(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)
        state.agent_span_id = str(uuid4())

        msg = _make_user_message(
            content="tool output text",
            parent_tool_use_id="tu_1",
            tool_use_result={"result": "found"},
        )

        _handle_user_message(msg, state)

        assert len(state.collected_messages) == 1
        m = state.collected_messages[0]
        assert m["role"] == "tool"
        assert m["tool_call_id"] == "tu_1"
        assert "found" in m["content"]

    def test_regular_user_message(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)
        state.agent_span_id = str(uuid4())

        msg = _make_user_message(content="Follow-up question")

        _handle_user_message(msg, state)

        assert len(state.collected_messages) == 1
        assert state.collected_messages[0] == {"role": "user", "content": "Follow-up question"}


class TestHandleResultMessage:
    def test_assigns_token_usage(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)

        from datetime import datetime, timezone

        from pandaprobe.schemas import SpanData

        llm_span_id = str(uuid4())
        span = SpanData(
            span_id=llm_span_id,
            name="claude",
            kind=SpanKind.LLM,
            started_at=datetime.now(timezone.utc),
        )
        state.spans[llm_span_id] = span
        state.current_llm_span_id = llm_span_id
        state.collected_messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        msg = _make_result_message(usage={"input_tokens": 100, "output_tokens": 50})

        _handle_result_message(msg, state)

        assert span.token_usage == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_no_usage(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)

        msg = _make_result_message(usage=None)
        _handle_result_message(msg, state)


# ===================================================================
# Hook tests
# ===================================================================


class TestToolHooks:
    @pytest.mark.asyncio
    async def test_pre_tool_use_creates_span(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)
        agent_span_id = str(uuid4())
        state.agent_span_id = agent_span_id
        span_token = _set_current_span(agent_span_id)

        pre_hook, _, _ = _make_tool_hooks(lambda: state)

        try:
            input_data = {"tool_name": "search", "tool_input": {"query": "test"}, "session_id": "s1"}
            await pre_hook(input_data, "tu_1", {})

            assert len(state.spans) == 1
            span = list(state.spans.values())[0]
            assert span.name == "search"
            assert span.kind == SpanKind.TOOL
            assert str(span.parent_span_id) == agent_span_id
            assert "tu_1" in state.tool_spans
        finally:
            _current_span_id.reset(span_token)

    @pytest.mark.asyncio
    async def test_post_tool_use_ends_span(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)
        state.agent_span_id = str(uuid4())

        _, post_hook, _ = _make_tool_hooks(lambda: state)

        from datetime import datetime, timezone

        from pandaprobe.schemas import SpanData

        span_id = str(uuid4())
        span = SpanData(
            span_id=span_id,
            parent_span_id=state.agent_span_id,
            name="search",
            kind=SpanKind.TOOL,
            started_at=datetime.now(timezone.utc),
        )
        state.spans[span_id] = span
        state.tool_spans["tu_1"] = span_id

        input_data = {
            "tool_name": "search",
            "tool_input": {"query": "test"},
            "tool_response": {"results": ["a", "b"]},
        }
        await post_hook(input_data, "tu_1", {})

        assert span.status == SpanStatusCode.OK
        assert span.ended_at is not None
        assert "tu_1" not in state.tool_spans

    @pytest.mark.asyncio
    async def test_post_tool_use_failure_sets_error(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)
        state.agent_span_id = str(uuid4())

        _, _, failure_hook = _make_tool_hooks(lambda: state)

        from datetime import datetime, timezone

        from pandaprobe.schemas import SpanData

        span_id = str(uuid4())
        span = SpanData(
            span_id=span_id,
            parent_span_id=state.agent_span_id,
            name="dangerous_tool",
            kind=SpanKind.TOOL,
            started_at=datetime.now(timezone.utc),
        )
        state.spans[span_id] = span
        state.tool_spans["tu_2"] = span_id

        input_data = {"tool_name": "dangerous_tool", "tool_input": {}, "error": "Permission denied"}
        await failure_hook(input_data, "tu_2", {})

        assert span.status == SpanStatusCode.ERROR
        assert span.error == "Permission denied"
        assert span.ended_at is not None

    @pytest.mark.asyncio
    async def test_hooks_noop_without_state(self):
        pre_hook, post_hook, failure_hook = _make_tool_hooks(lambda: None)
        result = await pre_hook({"tool_name": "x"}, "tu_1", {})
        assert result == {}
        result = await post_hook({"tool_name": "x"}, "tu_1", {})
        assert result == {}
        result = await failure_hook({"tool_name": "x", "error": "e"}, "tu_1", {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_hooks_noop_without_tool_use_id(self):
        adapter = ClaudeAgentSDKAdapter()
        state = _TraceState(adapter=adapter)

        pre_hook, _, _ = _make_tool_hooks(lambda: state)
        result = await pre_hook({"tool_name": "x"}, None, {})
        assert result == {}
        assert len(state.spans) == 0


# ===================================================================
# Full lifecycle test
# ===================================================================


class TestFullLifecycle:
    @respx.mock
    @pytest.mark.asyncio
    async def test_agent_llm_tool_lifecycle(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = ClaudeAgentSDKAdapter(
            session_id="sess-lifecycle",
            user_id="user-lifecycle",
            tags=["lifecycle-test"],
            metadata={"env": "test"},
        )
        _store_adapter(adapter)

        # Simulate: assistant calls tool, then responds after tool result
        messages = [
            _make_assistant_message(
                [
                    _make_text_block("Let me search for that."),
                    _make_tool_use_block("tu_1", "search", {"query": "weather"}),
                ],
                model="claude-sonnet-4-20250514",
            ),
            _make_user_message(
                content=[_make_tool_result_block("tu_1", "Sunny, 22C")],
                parent_tool_use_id="tu_1",
                tool_use_result={"weather": "sunny"},
            ),
            _make_assistant_message(
                [_make_text_block("The weather is sunny and 22C.")],
                model="claude-sonnet-4-20250514",
            ),
            _make_result_message(usage={"input_tokens": 150, "output_tokens": 60}),
        ]

        async def mock_receive(*args, **kwargs):
            for m in messages:
                yield m

        instance = SimpleNamespace(
            options=_make_options(system_prompt="Be helpful", model="claude-sonnet-4-20250514"),
            _pandaprobe_prompt="What's the weather?",
            _pandaprobe_start_time=None,
        )

        collected = []
        async for msg in _wrap_receive_response(mock_receive, instance, (), {}):
            collected.append(msg)

        assert len(collected) == 4
        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = ClaudeAgentSDKAdapter(session_id="sess-multi")
        _store_adapter(adapter)

        messages = [
            _make_assistant_message([_make_text_block("Hi! How can I help?")]),
            _make_result_message(usage={"input_tokens": 20, "output_tokens": 10}),
        ]

        async def mock_receive(*args, **kwargs):
            for m in messages:
                yield m

        instance = SimpleNamespace(
            options=_make_options(model="claude-sonnet-4-20250514"),
            _pandaprobe_prompt="Hello",
            _pandaprobe_start_time=None,
            _pandaprobe_history=[],
        )

        collected = []
        async for msg in _wrap_receive_response(mock_receive, instance, (), {}):
            collected.append(msg)

        assert len(collected) == 2
        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_thinking_only_message_not_creates_extra_span(self):
        """SDK with thinking sends ThinkingBlock-only AssistantMessage first, then text.

        Only ONE LLM span should be created (not two), with reasoning on it.
        """
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = ClaudeAgentSDKAdapter()
        _store_adapter(adapter)

        messages = [
            _make_assistant_message(
                [_make_thinking_block("Let me reason about this...")],
                model="claude-sonnet-4-20250514",
            ),
            _make_assistant_message(
                [_make_text_block("The answer is 42.")],
                model="claude-sonnet-4-20250514",
            ),
            _make_result_message(usage={"input_tokens": 50, "output_tokens": 30}),
        ]

        async def mock_receive(*args, **kwargs):
            for m in messages:
                yield m

        instance = SimpleNamespace(
            options=_make_options(model="claude-sonnet-4-20250514"),
            _pandaprobe_prompt="What is the meaning of life?",
            _pandaprobe_start_time=None,
            _pandaprobe_history=[],
        )

        async for _ in _wrap_receive_response(mock_receive, instance, (), {}):
            pass

        pandaprobe.get_client().flush(timeout=5.0)

    @respx.mock
    @pytest.mark.asyncio
    async def test_multi_turn_history_persists(self):
        """In multi-turn, the second call should have history from the first call."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = ClaudeAgentSDKAdapter()
        _store_adapter(adapter)

        instance = SimpleNamespace(
            options=_make_options(system_prompt="Be helpful", model="claude-sonnet-4-20250514"),
            _pandaprobe_prompt="Hello",
            _pandaprobe_start_time=None,
            _pandaprobe_history=[],
        )

        # Turn 1
        turn1_msgs = [
            _make_assistant_message([_make_text_block("Hi there!")]),
            _make_result_message(usage={"input_tokens": 10, "output_tokens": 5}),
        ]

        async def mock_receive_1(*args, **kwargs):
            for m in turn1_msgs:
                yield m

        async for _ in _wrap_receive_response(mock_receive_1, instance, (), {}):
            pass

        # Verify history was persisted
        assert len(instance._pandaprobe_history) == 2
        assert instance._pandaprobe_history[0] == {"role": "user", "content": "Hello"}
        assert instance._pandaprobe_history[1] == {"role": "assistant", "content": "Hi there!"}

        # Turn 2
        instance._pandaprobe_prompt = "How are you?"
        instance._pandaprobe_start_time = None

        turn2_msgs = [
            _make_assistant_message([_make_text_block("I'm doing well!")]),
            _make_result_message(usage={"input_tokens": 20, "output_tokens": 10}),
        ]

        async def mock_receive_2(*args, **kwargs):
            for m in turn2_msgs:
                yield m

        async for _ in _wrap_receive_response(mock_receive_2, instance, (), {}):
            pass

        # History should now have all 4 non-system messages
        assert len(instance._pandaprobe_history) == 4

        pandaprobe.get_client().flush(timeout=5.0)

    @respx.mock
    @pytest.mark.asyncio
    async def test_thinking_stripped_from_chain_and_agent_output(self):
        """Thinking blocks must be stripped from CHAIN and AGENT output messages."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = ClaudeAgentSDKAdapter()
        _store_adapter(adapter)

        messages = [
            _make_assistant_message(
                [_make_thinking_block("Deep thought"), _make_text_block("The answer")],
                model="claude-sonnet-4-20250514",
            ),
            _make_result_message(usage={"input_tokens": 10, "output_tokens": 5}),
        ]

        async def mock_receive(*args, **kwargs):
            for m in messages:
                yield m

        instance = SimpleNamespace(
            options=_make_options(model="claude-sonnet-4-20250514"),
            _pandaprobe_prompt="Question",
            _pandaprobe_start_time=None,
            _pandaprobe_history=[],
        )

        async for _ in _wrap_receive_response(mock_receive, instance, (), {}):
            pass

        pandaprobe.get_client().flush(timeout=5.0)


# ===================================================================
# Standalone query wrapper test
# ===================================================================


class TestWrapStandaloneQuery:
    @respx.mock
    @pytest.mark.asyncio
    async def test_traces_standalone_query(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = ClaudeAgentSDKAdapter()
        _store_adapter(adapter)

        messages = [
            _make_assistant_message([_make_text_block("I can help")]),
            _make_result_message(usage={"input_tokens": 10, "output_tokens": 5}),
        ]

        async def mock_query(*args, **kwargs):
            for m in messages:
                yield m

        collected = []
        async for msg in _wrap_standalone_query(
            mock_query, None, (), {"prompt": "Help me", "options": _make_options(model="claude-sonnet-4-20250514")}
        ):
            collected.append(msg)

        assert len(collected) == 2
        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_adapter_passthrough(self):
        _store_adapter(None)

        messages = [_make_assistant_message([_make_text_block("hi")])]

        async def mock_query(*args, **kwargs):
            for m in messages:
                yield m

        collected = []
        async for msg in _wrap_standalone_query(mock_query, None, (), {"prompt": "Hi"}):
            collected.append(msg)
        assert len(collected) == 1


# ===================================================================
# Configuration overrides test
# ===================================================================


class TestConfigurationOverrides:
    @respx.mock
    @pytest.mark.asyncio
    async def test_session_and_user_from_adapter(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = ClaudeAgentSDKAdapter(
            session_id="custom-session",
            user_id="custom-user",
            tags=["tag1", "tag2"],
            metadata={"source": "test"},
        )
        _store_adapter(adapter)

        messages = [
            _make_assistant_message([_make_text_block("OK")]),
            _make_result_message(),
        ]

        async def mock_receive(*args, **kwargs):
            for m in messages:
                yield m

        instance = SimpleNamespace(
            options=None,
            _pandaprobe_prompt="Test",
            _pandaprobe_start_time=None,
        )

        async for _ in _wrap_receive_response(mock_receive, instance, (), {}):
            pass

        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1
