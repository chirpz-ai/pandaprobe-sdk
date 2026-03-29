"""Tests for pandaprobe.integrations.openai_agents."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.integrations.openai_agents.adapter import (
    OpenAIAgentsAdapter,
    _fill_agent_span,
    _fill_custom_span,
    _fill_guardrail_span,
    _fill_handoff_span,
    _get_adapter,
    _on_span_end,
    _on_span_start,
    _on_trace_end,
    _on_trace_start,
    _resolve_span_kind,
    _resolve_span_name,
    _store_adapter,
    _trace_states,
)
from pandaprobe.integrations.openai_agents.utils import (
    collapse_content,
    extract_generation_model_parameters,
    extract_reasoning,
    extract_response_model_parameters,
    extract_token_usage,
    normalize_generation_input,
    normalize_generation_output,
    normalize_response_input,
    normalize_response_output,
    serialize_tool_io,
)
from pandaprobe.schemas import SpanData, SpanKind, SpanStatusCode


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


@pytest.fixture(autouse=True)
def _cleanup_trace_states():
    """Ensure _trace_states is clean between tests."""
    _trace_states.clear()
    yield
    _trace_states.clear()


# ---------------------------------------------------------------------------
# Mock OpenAI Agents SDK objects
# ---------------------------------------------------------------------------


def _make_trace(trace_id="trace-1", name="Agent workflow"):
    return SimpleNamespace(trace_id=trace_id, name=name, export=lambda: {"workflow_name": name})


def _make_span(
    span_id="span-1",
    trace_id="trace-1",
    parent_id=None,
    span_data=None,
    started_at="2025-01-01T00:00:00+00:00",
    ended_at="2025-01-01T00:00:01+00:00",
    error=None,
    name=None,
):
    return SimpleNamespace(
        span_id=span_id,
        trace_id=trace_id,
        parent_id=parent_id if parent_id is not None else trace_id,
        span_data=span_data,
        started_at=started_at,
        ended_at=ended_at,
        error=error,
        name=name,
    )


def _make_agent_span_data(name="my_agent", tools=None, handoffs=None, output_type=None):
    return SimpleNamespace(type="agent", name=name, tools=tools, handoffs=handoffs, output_type=output_type)


def _make_response_span_data(response=None, input_data=None):
    return SimpleNamespace(type="response", name="Response", response=response, input=input_data)


def _make_generation_span_data(
    input_data=None, output_data=None, model="gpt-4.1-mini", model_config=None, usage=None
):
    return SimpleNamespace(
        type="generation",
        name="Generation",
        input=input_data,
        output=output_data,
        model=model,
        model_config=model_config,
        usage=usage,
    )


def _make_function_span_data(name="my_tool", input_data=None, output_data=None, mcp_data=None):
    return SimpleNamespace(type="function", name=name, input=input_data, output=output_data, mcp_data=mcp_data)


def _make_handoff_span_data(from_agent="agent_a", to_agent="agent_b"):
    return SimpleNamespace(type="handoff", name="Handoff", from_agent=from_agent, to_agent=to_agent)


def _make_guardrail_span_data(name="content_filter", triggered=True):
    return SimpleNamespace(type="guardrail", name=name, triggered=triggered)


def _make_custom_span_data(name="custom_op", data=None):
    return SimpleNamespace(type="custom", name=name, data=data)


def _make_response(
    model="gpt-4.1-mini",
    output=None,
    usage=None,
    instructions=None,
    temperature=None,
    top_p=None,
    max_output_tokens=None,
    reasoning=None,
):
    return SimpleNamespace(
        id="resp-1",
        model=model,
        output=output or [],
        usage=usage,
        instructions=instructions,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
    )


def _make_usage(input_tokens=100, output_tokens=50, total_tokens=150, input_details=None, output_details=None):
    ns = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_tokens_details=input_details,
        output_tokens_details=output_details,
    )
    return ns


# ===================================================================
# Utils tests
# ===================================================================


class TestCollapseContent:
    def test_single_output_text(self):
        assert collapse_content([{"type": "output_text", "text": "Hello"}]) == "Hello"

    def test_multiple_output_text(self):
        items = [{"type": "output_text", "text": "Hello"}, {"type": "output_text", "text": "world"}]
        assert collapse_content(items) == "Hello world"

    def test_single_text_type(self):
        assert collapse_content([{"type": "text", "text": "Hi"}]) == "Hi"

    def test_mixed_types_not_collapsed(self):
        items = [{"type": "output_text", "text": "Hi"}, {"type": "function_call", "name": "f"}]
        assert isinstance(collapse_content(items), list)

    def test_string_passthrough(self):
        assert collapse_content("hello") == "hello"

    def test_empty_list(self):
        assert collapse_content([]) == []

    def test_non_list(self):
        assert collapse_content(42) == 42


class TestNormalizeResponseInput:
    def test_string_input(self):
        span_data = _make_response_span_data(response=_make_response(instructions="Be helpful"), input_data="Hello")
        result = normalize_response_input(span_data)
        assert result == {
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ]
        }

    def test_list_input_with_message(self):
        span_data = _make_response_span_data(
            response=_make_response(),
            input_data=[{"type": "message", "role": "user", "content": [{"type": "output_text", "text": "Hi"}]}],
        )
        result = normalize_response_input(span_data)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hi"

    def test_function_call_output_in_input(self):
        span_data = _make_response_span_data(
            response=_make_response(),
            input_data=[{"type": "function_call_output", "call_id": "call-1", "output": "sunny"}],
        )
        result = normalize_response_input(span_data)
        assert result["messages"][0]["role"] == "tool"
        assert result["messages"][0]["content"] == "sunny"

    def test_no_input(self):
        span_data = _make_response_span_data(response=_make_response())
        result = normalize_response_input(span_data)
        assert result == {"messages": []}

    def test_no_instructions(self):
        span_data = _make_response_span_data(response=_make_response(), input_data="Hi")
        result = normalize_response_input(span_data)
        assert len(result["messages"]) == 1
        assert result["messages"][0] == {"role": "user", "content": "Hi"}

    def test_item_reference_skipped(self):
        span_data = _make_response_span_data(
            response=_make_response(),
            input_data=[{"type": "item_reference", "id": "ref-1"}, "Hello"],
        )
        result = normalize_response_input(span_data)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello"


class TestNormalizeResponseOutput:
    def test_message_output(self):
        response = _make_response(
            output=[
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello!"}]}
            ]
        )
        span_data = _make_response_span_data(response=response)
        result = normalize_response_output(span_data)
        assert result == {"messages": [{"role": "assistant", "content": "Hello!"}]}

    def test_function_call_output(self):
        response = _make_response(
            output=[{"type": "function_call", "id": "fc-1", "name": "search", "arguments": '{"q": "test"}'}]
        )
        span_data = _make_response_span_data(response=response)
        result = normalize_response_output(span_data)
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["tool_calls"][0]["name"] == "search"

    def test_reasoning_stripped(self):
        response = _make_response(
            output=[
                {"type": "reasoning", "id": "r-1", "summary": [{"type": "summary_text", "text": "thinking..."}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
            ]
        )
        span_data = _make_response_span_data(response=response)
        result = normalize_response_output(span_data)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "answer"

    def test_no_response(self):
        span_data = _make_response_span_data(response=None)
        result = normalize_response_output(span_data)
        assert result == {"messages": []}

    def test_empty_output(self):
        span_data = _make_response_span_data(response=_make_response(output=[]))
        result = normalize_response_output(span_data)
        assert result == {"messages": []}


class TestNormalizeGenerationInput:
    def test_list_messages(self):
        span_data = _make_generation_span_data(
            input_data=[{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hi"}]
        )
        result = normalize_generation_input(span_data)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"

    def test_string_input(self):
        span_data = _make_generation_span_data(input_data="Hello")
        result = normalize_generation_input(span_data)
        assert result == {"messages": [{"role": "user", "content": "Hello"}]}

    def test_empty(self):
        span_data = _make_generation_span_data(input_data=None)
        result = normalize_generation_input(span_data)
        assert result == {"messages": []}


class TestNormalizeGenerationOutput:
    def test_list_messages(self):
        span_data = _make_generation_span_data(
            output_data=[{"role": "assistant", "content": "Response"}]
        )
        result = normalize_generation_output(span_data)
        assert result == {"messages": [{"role": "assistant", "content": "Response"}]}

    def test_string_output(self):
        span_data = _make_generation_span_data(output_data="Response text")
        result = normalize_generation_output(span_data)
        assert result == {"messages": [{"role": "assistant", "content": "Response text"}]}


class TestExtractReasoning:
    def test_extracts_summary_text(self):
        response = _make_response(
            output=[
                {"type": "reasoning", "id": "r-1", "summary": [{"type": "summary_text", "text": "I thought about it"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
            ]
        )
        assert extract_reasoning(response) == "I thought about it"

    def test_multiple_reasoning_parts(self):
        response = _make_response(
            output=[
                {
                    "type": "reasoning",
                    "id": "r-1",
                    "summary": [
                        {"type": "summary_text", "text": "First thought"},
                        {"type": "summary_text", "text": "Second thought"},
                    ],
                },
            ]
        )
        assert extract_reasoning(response) == "First thought\n\nSecond thought"

    def test_no_reasoning(self):
        response = _make_response(
            output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}]
        )
        assert extract_reasoning(response) is None

    def test_none_response(self):
        assert extract_reasoning(None) is None

    def test_empty_output(self):
        response = _make_response(output=[])
        assert extract_reasoning(response) is None


class TestExtractTokenUsage:
    def test_responses_api_format(self):
        usage = _make_usage(input_tokens=100, output_tokens=50, total_tokens=150)
        result = extract_token_usage(usage)
        assert result == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    def test_with_nested_details(self):
        usage = _make_usage(
            input_tokens=432,
            output_tokens=63,
            total_tokens=495,
            input_details=SimpleNamespace(cached_tokens=128),
            output_details=SimpleNamespace(reasoning_tokens=10),
        )
        result = extract_token_usage(usage)
        assert result == {
            "prompt_tokens": 432,
            "completion_tokens": 63,
            "total_tokens": 495,
            "cache_read_tokens": 128,
            "reasoning_tokens": 10,
        }

    def test_dict_format(self):
        usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        result = extract_token_usage(usage)
        assert result == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def test_already_mapped_keys(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        result = extract_token_usage(usage)
        assert result == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def test_none_usage(self):
        assert extract_token_usage(None) is None

    def test_empty_usage(self):
        assert extract_token_usage({}) is None

    def test_computed_total(self):
        usage = {"input_tokens": 10, "output_tokens": 20}
        result = extract_token_usage(usage)
        assert result["total_tokens"] == 30

    def test_zero_cached_and_reasoning_skipped(self):
        usage = _make_usage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            input_details=SimpleNamespace(cached_tokens=0),
            output_details=SimpleNamespace(reasoning_tokens=0),
        )
        result = extract_token_usage(usage)
        assert "cache_read_tokens" not in result
        assert "reasoning_tokens" not in result


class TestExtractResponseModelParameters:
    def test_extracts_params(self):
        response = _make_response(temperature=0.7, top_p=0.9, max_output_tokens=1024)
        result = extract_response_model_parameters(response)
        assert result == {"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 1024}

    def test_none_response(self):
        assert extract_response_model_parameters(None) is None

    def test_no_params(self):
        response = _make_response()
        result = extract_response_model_parameters(response)
        assert result is None

    def test_reasoning_param(self):
        response = _make_response(reasoning={"effort": "high"})
        result = extract_response_model_parameters(response)
        assert result == {"reasoning": {"effort": "high"}}


class TestExtractGenerationModelParameters:
    def test_extracts_from_config(self):
        span_data = _make_generation_span_data(model_config={"temperature": 0.5, "top_p": 0.8})
        result = extract_generation_model_parameters(span_data)
        assert result == {"temperature": 0.5, "top_p": 0.8}

    def test_none_config(self):
        span_data = _make_generation_span_data(model_config=None)
        assert extract_generation_model_parameters(span_data) is None


class TestSerializeToolIo:
    def test_string(self):
        assert serialize_tool_io("hello") == "hello"

    def test_json_string(self):
        result = serialize_tool_io('{"key": "value"}')
        assert result == {"key": "value"}

    def test_none(self):
        assert serialize_tool_io(None) is None

    def test_dict(self):
        result = serialize_tool_io({"a": 1})
        assert result == {"a": 1}

    def test_preserves_falsy(self):
        assert serialize_tool_io(0) == 0
        assert serialize_tool_io(False) is False


# ===================================================================
# Adapter tests
# ===================================================================


class TestOpenAIAgentsAdapter:
    def test_extends_base(self):
        adapter = OpenAIAgentsAdapter(
            session_id="s1",
            user_id="u1",
            tags=["t"],
            metadata={"k": "v"},
        )
        assert adapter._session_id == "s1"
        assert adapter._user_id == "u1"
        assert adapter._tags == ["t"]
        assert adapter._metadata == {"k": "v"}

    def test_instrument_without_agents_returns_false(self):
        import pandaprobe.integrations.openai_agents.adapter as mod

        original = mod._patched
        mod._patched = False
        try:
            adapter = OpenAIAgentsAdapter()
            with patch.dict("sys.modules", {"agents": None, "agents.tracing": None}):
                result = adapter.instrument()
            assert result is False
        finally:
            mod._patched = original

    def test_default_values(self):
        adapter = OpenAIAgentsAdapter()
        assert adapter._session_id is None
        assert adapter._user_id is None
        assert adapter._tags == []
        assert adapter._metadata == {}


class TestAdapterStorage:
    def test_store_and_get(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)
        assert _get_adapter() is adapter

    def test_last_adapter_wins(self):
        a1 = OpenAIAgentsAdapter()
        a2 = OpenAIAgentsAdapter()
        _store_adapter(a1)
        _store_adapter(a2)
        assert _get_adapter() is a2


class TestInstrumentIdempotency:
    def test_store_adapter_updates_on_repeated_calls(self):
        import pandaprobe.integrations.openai_agents.adapter as mod

        original = mod._patched
        mod._patched = True
        try:
            a1 = OpenAIAgentsAdapter(session_id="first")
            a1.instrument()

            a2 = OpenAIAgentsAdapter(session_id="second")
            a2.instrument()

            assert _get_adapter() is a2
        finally:
            mod._patched = original


# ===================================================================
# Span kind / name resolution tests
# ===================================================================


class TestSpanKindResolution:
    def test_agent(self):
        assert _resolve_span_kind(SimpleNamespace(type="agent")) == SpanKind.AGENT

    def test_handoff(self):
        assert _resolve_span_kind(SimpleNamespace(type="handoff")) == SpanKind.AGENT

    def test_response(self):
        assert _resolve_span_kind(SimpleNamespace(type="response")) == SpanKind.LLM

    def test_generation(self):
        assert _resolve_span_kind(SimpleNamespace(type="generation")) == SpanKind.LLM

    def test_function(self):
        assert _resolve_span_kind(SimpleNamespace(type="function")) == SpanKind.TOOL

    def test_guardrail(self):
        assert _resolve_span_kind(SimpleNamespace(type="guardrail")) == SpanKind.OTHER

    def test_custom(self):
        assert _resolve_span_kind(SimpleNamespace(type="custom")) == SpanKind.OTHER

    def test_unknown(self):
        assert _resolve_span_kind(SimpleNamespace(type="unknown")) == SpanKind.OTHER

    def test_none_data(self):
        assert _resolve_span_kind(None) == SpanKind.OTHER


class TestSpanNameResolution:
    def test_from_span_data_name(self):
        span = SimpleNamespace(span_data=SimpleNamespace(name="my_agent", type="agent"), name=None)
        assert _resolve_span_name(span) == "my_agent"

    def test_from_span_name(self):
        span = SimpleNamespace(span_data=SimpleNamespace(name=None, type="agent"), name="fallback")
        assert _resolve_span_name(span) == "fallback"

    def test_from_type(self):
        span = SimpleNamespace(span_data=SimpleNamespace(name=None, type="generation"), name=None)
        assert _resolve_span_name(span) == "Generation"

    def test_default(self):
        span = SimpleNamespace(span_data=None, name=None)
        assert _resolve_span_name(span) == "Span"


# ===================================================================
# Event handler tests
# ===================================================================


class TestOnTraceStart:
    def test_creates_state_and_root_span(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)
        trace = _make_trace(trace_id="t-1", name="My Workflow")

        _on_trace_start(trace)

        assert "t-1" in _trace_states
        state = _trace_states["t-1"]
        assert state.root_span_id is not None
        assert len(state.spans) == 1
        root = list(state.spans.values())[0]
        assert root.kind == SpanKind.CHAIN
        assert root.name == "My Workflow"
        assert root.parent_span_id is None

    def test_no_adapter_noop(self):
        _store_adapter(None)
        _on_trace_start(_make_trace())
        assert len(_trace_states) == 0

    def test_no_trace_id_noop(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)
        _on_trace_start(SimpleNamespace(trace_id=None, name="x"))
        assert len(_trace_states) == 0


class TestOnSpanStart:
    def test_creates_span_with_parent(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)
        trace = _make_trace(trace_id="t-2")
        _on_trace_start(trace)

        state = _trace_states["t-2"]
        root_pp_id = state.root_span_id

        span = _make_span(
            span_id="sdk-span-1",
            trace_id="t-2",
            parent_id="t-2",
            span_data=_make_agent_span_data(name="agent_1"),
        )
        _on_span_start(span)

        assert "sdk-span-1" in state.sdk_to_pandaprobe
        pp_id = state.sdk_to_pandaprobe["sdk-span-1"]
        pp_span = state.spans[pp_id]
        assert pp_span.kind == SpanKind.AGENT
        assert pp_span.name == "agent_1"
        assert str(pp_span.parent_span_id) == root_pp_id

    def test_nested_parent_resolution(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)
        trace = _make_trace(trace_id="t-3")
        _on_trace_start(trace)

        agent_span = _make_span(span_id="agent-1", trace_id="t-3", span_data=_make_agent_span_data())
        _on_span_start(agent_span)

        child_span = _make_span(
            span_id="llm-1",
            trace_id="t-3",
            parent_id="agent-1",
            span_data=_make_response_span_data(),
        )
        _on_span_start(child_span)

        state = _trace_states["t-3"]
        agent_pp_id = state.sdk_to_pandaprobe["agent-1"]
        llm_pp_id = state.sdk_to_pandaprobe["llm-1"]
        llm_span = state.spans[llm_pp_id]
        assert str(llm_span.parent_span_id) == agent_pp_id

    def test_no_state_noop(self):
        span = _make_span(trace_id="nonexistent")
        _on_span_start(span)


class TestOnSpanEnd:
    def test_fills_response_span(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)
        trace = _make_trace(trace_id="t-4")
        _on_trace_start(trace)

        response = _make_response(
            model="gpt-4.1-mini",
            output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello!"}]}],
            usage=_make_usage(10, 20, 30),
            instructions="Be helpful",
            temperature=0.7,
        )
        span_data = _make_response_span_data(response=response, input_data="What is 2+2?")

        span = _make_span(span_id="resp-1", trace_id="t-4", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        state = _trace_states["t-4"]
        pp_id = state.sdk_to_pandaprobe["resp-1"]
        pp_span = state.spans[pp_id]

        assert pp_span.model == "gpt-4.1-mini"
        assert pp_span.token_usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        assert pp_span.status == SpanStatusCode.OK
        assert pp_span.ended_at is not None
        assert pp_span.input["messages"][0] == {"role": "system", "content": "Be helpful"}
        assert pp_span.output["messages"][0]["content"] == "Hello!"
        assert pp_span.model_parameters == {"temperature": 0.7}

    def test_fills_generation_span(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)
        trace = _make_trace(trace_id="t-5")
        _on_trace_start(trace)

        span_data = _make_generation_span_data(
            input_data=[{"role": "user", "content": "Hi"}],
            output_data=[{"role": "assistant", "content": "Hello!"}],
            model="gpt-4o",
            usage={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
            model_config={"temperature": 0.5},
        )

        span = _make_span(span_id="gen-1", trace_id="t-5", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        state = _trace_states["t-5"]
        pp_id = state.sdk_to_pandaprobe["gen-1"]
        pp_span = state.spans[pp_id]

        assert pp_span.model == "gpt-4o"
        assert pp_span.token_usage == {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        assert pp_span.model_parameters == {"temperature": 0.5}

    def test_fills_function_span(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)
        trace = _make_trace(trace_id="t-6")
        _on_trace_start(trace)

        span_data = _make_function_span_data(name="get_weather", input_data='{"city": "London"}', output_data="Sunny")

        span = _make_span(span_id="func-1", trace_id="t-6", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        state = _trace_states["t-6"]
        pp_id = state.sdk_to_pandaprobe["func-1"]
        pp_span = state.spans[pp_id]

        assert pp_span.name == "get_weather"
        assert pp_span.input == {"city": "London"}
        assert pp_span.output == "Sunny"

    def test_fills_handoff_span(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)
        trace = _make_trace(trace_id="t-7")
        _on_trace_start(trace)

        span_data = _make_handoff_span_data(from_agent="agent_a", to_agent="agent_b")
        span = _make_span(span_id="ho-1", trace_id="t-7", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        state = _trace_states["t-7"]
        pp_id = state.sdk_to_pandaprobe["ho-1"]
        pp_span = state.spans[pp_id]

        assert pp_span.metadata["from_agent"] == "agent_a"
        assert pp_span.metadata["to_agent"] == "agent_b"

    def test_fills_guardrail_span(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)
        trace = _make_trace(trace_id="t-8")
        _on_trace_start(trace)

        span_data = _make_guardrail_span_data(name="filter", triggered=True)
        span = _make_span(span_id="gr-1", trace_id="t-8", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        state = _trace_states["t-8"]
        pp_id = state.sdk_to_pandaprobe["gr-1"]
        pp_span = state.spans[pp_id]

        assert pp_span.metadata["triggered"] is True

    def test_fills_custom_span(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)
        trace = _make_trace(trace_id="t-9")
        _on_trace_start(trace)

        span_data = _make_custom_span_data(name="custom", data={"key": "value"})
        span = _make_span(span_id="cust-1", trace_id="t-9", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        state = _trace_states["t-9"]
        pp_id = state.sdk_to_pandaprobe["cust-1"]
        pp_span = state.spans[pp_id]

        assert pp_span.metadata["custom_data"] == {"key": "value"}

    def test_error_propagation(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)
        trace = _make_trace(trace_id="t-10")
        _on_trace_start(trace)

        span_data = _make_agent_span_data()
        span = _make_span(span_id="err-1", trace_id="t-10", span_data=span_data, error="Something failed")
        _on_span_start(span)
        _on_span_end(span)

        state = _trace_states["t-10"]
        pp_id = state.sdk_to_pandaprobe["err-1"]
        pp_span = state.spans[pp_id]

        assert pp_span.status == SpanStatusCode.ERROR
        assert pp_span.error == "Something failed"

    def test_reasoning_extraction(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)
        trace = _make_trace(trace_id="t-11")
        _on_trace_start(trace)

        response = _make_response(
            output=[
                {"type": "reasoning", "id": "r-1", "summary": [{"type": "summary_text", "text": "I think..."}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Answer"}]},
            ]
        )
        span_data = _make_response_span_data(response=response, input_data="Question?")
        span = _make_span(span_id="reason-1", trace_id="t-11", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        state = _trace_states["t-11"]
        pp_id = state.sdk_to_pandaprobe["reason-1"]
        pp_span = state.spans[pp_id]

        assert pp_span.metadata["reasoning_summary"] == "I think..."
        assert len(pp_span.output["messages"]) == 1
        assert pp_span.output["messages"][0]["content"] == "Answer"


class TestOnTraceEnd:
    @respx.mock
    def test_submits_trace(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = OpenAIAgentsAdapter(session_id="sess-1", user_id="user-1", tags=["test"])
        _store_adapter(adapter)

        trace = _make_trace(trace_id="t-end-1")
        _on_trace_start(trace)

        response = _make_response(
            output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done"}]}],
            instructions="Be helpful",
        )
        span_data = _make_response_span_data(response=response, input_data="Hello")
        span = _make_span(span_id="s-end-1", trace_id="t-end-1", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        _on_trace_end(trace)

        assert "t-end-1" not in _trace_states

        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    @respx.mock
    def test_error_trace(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)

        trace = _make_trace(trace_id="t-err")
        _on_trace_start(trace)

        span = _make_span(
            span_id="s-err",
            trace_id="t-err",
            span_data=_make_agent_span_data(),
            error="Agent crashed",
        )
        _on_span_start(span)
        _on_span_end(span)

        _on_trace_end(trace)

        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    def test_nonexistent_trace_noop(self):
        _on_trace_end(_make_trace(trace_id="nonexistent"))


# ===================================================================
# Fill span functions (unit tests)
# ===================================================================


class TestFillAgentSpan:
    def test_metadata(self):
        from datetime import datetime, timezone
        from uuid import uuid4

        span = SpanData(span_id=str(uuid4()), name="agent", kind=SpanKind.AGENT, started_at=datetime.now(timezone.utc))
        data = _make_agent_span_data(name="my_agent", tools=["search", "calc"], handoffs=["sub_agent"], output_type="str")
        _fill_agent_span(span, data)
        assert span.metadata["tools"] == ["search", "calc"]
        assert span.metadata["handoffs"] == ["sub_agent"]
        assert span.metadata["output_type"] == "str"


class TestFillHandoffSpan:
    def test_from_to(self):
        from datetime import datetime, timezone
        from uuid import uuid4

        span = SpanData(span_id=str(uuid4()), name="handoff", kind=SpanKind.AGENT, started_at=datetime.now(timezone.utc))
        data = _make_handoff_span_data("a", "b")
        _fill_handoff_span(span, data)
        assert span.metadata["from_agent"] == "a"
        assert span.metadata["to_agent"] == "b"


class TestFillGuardrailSpan:
    def test_triggered(self):
        from datetime import datetime, timezone
        from uuid import uuid4

        span = SpanData(span_id=str(uuid4()), name="guardrail", kind=SpanKind.OTHER, started_at=datetime.now(timezone.utc))
        data = _make_guardrail_span_data(triggered=False)
        _fill_guardrail_span(span, data)
        assert span.metadata["triggered"] is False


class TestFillCustomSpan:
    def test_data(self):
        from datetime import datetime, timezone
        from uuid import uuid4

        span = SpanData(span_id=str(uuid4()), name="custom", kind=SpanKind.OTHER, started_at=datetime.now(timezone.utc))
        data = _make_custom_span_data(data={"hello": "world"})
        _fill_custom_span(span, data)
        assert span.metadata["custom_data"] == {"hello": "world"}


# ===================================================================
# Full lifecycle tests
# ===================================================================


class TestFullLifecycle:
    @respx.mock
    def test_agent_response_tool_lifecycle(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = OpenAIAgentsAdapter(
            session_id="sess-lc",
            user_id="user-lc",
            tags=["lifecycle"],
            metadata={"env": "test"},
        )
        _store_adapter(adapter)

        trace = _make_trace(trace_id="lc-1", name="Agent workflow")
        _on_trace_start(trace)

        agent_data = _make_agent_span_data(name="weather_agent", tools=["get_weather"])
        agent_span = _make_span(span_id="agent-lc", trace_id="lc-1", span_data=agent_data)
        _on_span_start(agent_span)

        response1 = _make_response(
            model="gpt-4.1-mini",
            output=[{"type": "function_call", "id": "fc-1", "name": "get_weather", "arguments": '{"city":"London"}'}],
            usage=_make_usage(50, 10, 60),
            instructions="Help the user",
        )
        resp_span1 = _make_span(
            span_id="resp-lc-1",
            trace_id="lc-1",
            parent_id="agent-lc",
            span_data=_make_response_span_data(response=response1, input_data="What's the weather in London?"),
        )
        _on_span_start(resp_span1)
        _on_span_end(resp_span1)

        func_data = _make_function_span_data(name="get_weather", input_data='{"city":"London"}', output_data="Sunny")
        func_span = _make_span(span_id="func-lc", trace_id="lc-1", parent_id="agent-lc", span_data=func_data)
        _on_span_start(func_span)
        _on_span_end(func_span)

        response2 = _make_response(
            model="gpt-4.1-mini",
            output=[
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "It's sunny!"}]}
            ],
            usage=_make_usage(80, 20, 100),
        )
        resp_span2 = _make_span(
            span_id="resp-lc-2",
            trace_id="lc-1",
            parent_id="agent-lc",
            span_data=_make_response_span_data(response=response2, input_data=[
                {"type": "function_call_output", "call_id": "fc-1", "output": "Sunny"},
            ]),
        )
        _on_span_start(resp_span2)
        _on_span_end(resp_span2)

        _on_span_end(agent_span)
        _on_trace_end(trace)

        assert "lc-1" not in _trace_states
        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    @respx.mock
    def test_nested_agent_hierarchy(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)

        trace = _make_trace(trace_id="nested-1")
        _on_trace_start(trace)

        agent1 = _make_span(span_id="a1", trace_id="nested-1", span_data=_make_agent_span_data(name="root_agent"))
        _on_span_start(agent1)

        agent2 = _make_span(
            span_id="a2", trace_id="nested-1", parent_id="a1", span_data=_make_agent_span_data(name="sub_agent")
        )
        _on_span_start(agent2)

        resp = _make_span(
            span_id="r1",
            trace_id="nested-1",
            parent_id="a2",
            span_data=_make_response_span_data(
                response=_make_response(
                    output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi"}]}]
                ),
                input_data="Hello",
            ),
        )
        _on_span_start(resp)
        _on_span_end(resp)

        _on_span_end(agent2)
        _on_span_end(agent1)
        _on_trace_end(trace)

        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

        assert "nested-1" not in _trace_states


class TestConfigurationOverrides:
    def test_custom_values(self):
        adapter = OpenAIAgentsAdapter(
            session_id="s",
            user_id="u",
            tags=["a", "b"],
            metadata={"key": "val"},
        )
        assert adapter._session_id == "s"
        assert adapter._user_id == "u"
        assert adapter._tags == ["a", "b"]
        assert adapter._metadata == {"key": "val"}

    @respx.mock
    def test_trace_uses_adapter_config(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = OpenAIAgentsAdapter(session_id="my-session", user_id="my-user", tags=["tag1"])
        _store_adapter(adapter)

        trace = _make_trace(trace_id="cfg-1")
        _on_trace_start(trace)
        _on_trace_end(trace)

        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1


class TestTraceInputOutput:
    def test_trace_input_from_first_response(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)

        trace = _make_trace(trace_id="io-1")
        _on_trace_start(trace)

        response = _make_response(
            instructions="System",
            output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi"}]}],
        )
        span_data = _make_response_span_data(response=response, input_data="User message")
        span = _make_span(span_id="io-s1", trace_id="io-1", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        state = _trace_states["io-1"]
        assert state.trace_input is not None
        assert state.trace_input["messages"][0]["role"] == "user"
        assert state.trace_input["messages"][0]["content"] == "User message"

    def test_trace_output_from_last_response(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)

        trace = _make_trace(trace_id="io-2")
        _on_trace_start(trace)

        response1 = _make_response(
            output=[{"type": "function_call", "id": "fc", "name": "tool", "arguments": "{}"}],
        )
        s1 = _make_span(span_id="io-r1", trace_id="io-2", span_data=_make_response_span_data(response=response1, input_data="Q"))
        _on_span_start(s1)
        _on_span_end(s1)

        response2 = _make_response(
            output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Final"}]}],
        )
        s2 = _make_span(span_id="io-r2", trace_id="io-2", span_data=_make_response_span_data(response=response2, input_data="Context"))
        _on_span_start(s2)
        _on_span_end(s2)

        state = _trace_states["io-2"]
        assert state.trace_output["messages"][0]["content"] == "Final"


# ===================================================================
# Bug fix regression tests
# ===================================================================


class TestInputNormalizationMessageDicts:
    """Regression: dicts with role+content but no type must not be double-wrapped."""

    def test_plain_message_dict_not_double_wrapped(self):
        from pandaprobe.integrations.openai_agents.utils import _normalize_input_item

        item = {"role": "user", "content": "Hello"}
        result = _normalize_input_item(item)
        assert result == {"role": "user", "content": "Hello"}

    def test_assistant_message_dict(self):
        from pandaprobe.integrations.openai_agents.utils import _normalize_input_item

        item = {"role": "assistant", "content": "Hi there"}
        result = _normalize_input_item(item)
        assert result == {"role": "assistant", "content": "Hi there"}

    def test_message_dict_with_list_content_collapsed(self):
        from pandaprobe.integrations.openai_agents.utils import _normalize_input_item

        item = {"role": "user", "content": [{"type": "output_text", "text": "Hello"}]}
        result = _normalize_input_item(item)
        assert result == {"role": "user", "content": "Hello"}

    def test_message_dict_preserves_tool_calls(self):
        from pandaprobe.integrations.openai_agents.utils import _normalize_input_item

        item = {"role": "assistant", "content": None, "tool_calls": [{"name": "search"}]}
        result = _normalize_input_item(item)
        assert result["tool_calls"] == [{"name": "search"}]

    def test_full_response_input_normalization(self):
        """End-to-end: ResponseSpanData with EasyInputMessage-like input."""
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)

        trace = _make_trace(trace_id="norm-1")
        _on_trace_start(trace)

        response = _make_response(
            instructions="Be helpful",
            output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi!"}]}],
        )
        span_data = _make_response_span_data(
            response=response,
            input_data=[{"role": "user", "content": "Hello"}],
        )
        span = _make_span(span_id="norm-s1", trace_id="norm-1", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        state = _trace_states["norm-1"]
        pp_id = state.sdk_to_pandaprobe["norm-s1"]
        pp_span = state.spans[pp_id]

        assert pp_span.input["messages"][0] == {"role": "system", "content": "Be helpful"}
        assert pp_span.input["messages"][1] == {"role": "user", "content": "Hello"}

    def test_function_call_in_input(self):
        from pandaprobe.integrations.openai_agents.utils import _normalize_input_item

        item = {"type": "function_call", "id": "fc-1", "name": "search", "arguments": '{"q":"test"}'}
        result = _normalize_input_item(item)
        assert result["role"] == "assistant"
        assert result["tool_calls"][0]["name"] == "search"


class TestChainSpanInputOutput:
    """CHAIN span must contain full conversation, not just trimmed last message."""

    def test_chain_input_has_full_conversation(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)

        trace = _make_trace(trace_id="chain-io-1")
        _on_trace_start(trace)

        response = _make_response(
            instructions="You are a travel advisor.",
            output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Visit Paris!"}]}],
        )
        span_data = _make_response_span_data(response=response, input_data="Where should I go?")
        span = _make_span(span_id="chain-r1", trace_id="chain-io-1", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        state = _trace_states["chain-io-1"]

        assert state.chain_input == {
            "messages": [
                {"role": "system", "content": "You are a travel advisor."},
                {"role": "user", "content": "Where should I go?"},
            ]
        }

    def test_chain_output_combines_input_and_output(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)

        trace = _make_trace(trace_id="chain-io-2")
        _on_trace_start(trace)

        response = _make_response(
            instructions="Be helpful.",
            output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "The answer"}]}],
        )
        span_data = _make_response_span_data(response=response, input_data="Question?")
        span = _make_span(span_id="chain-r2", trace_id="chain-io-2", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        state = _trace_states["chain-io-2"]

        assert state.chain_output == {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "The answer"},
            ]
        }

    @respx.mock
    def test_chain_span_gets_full_io_on_trace_end(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)

        trace = _make_trace(trace_id="chain-io-3")
        _on_trace_start(trace)

        response = _make_response(
            instructions="System prompt",
            output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Reply"}]}],
        )
        span_data = _make_response_span_data(response=response, input_data="User msg")
        span = _make_span(span_id="chain-r3", trace_id="chain-io-3", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        _on_trace_end(trace)

        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    def test_trace_input_still_trimmed(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)

        trace = _make_trace(trace_id="chain-io-4")
        _on_trace_start(trace)

        response = _make_response(
            instructions="System prompt",
            output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Reply"}]}],
        )
        span_data = _make_response_span_data(response=response, input_data="User msg")
        span = _make_span(span_id="chain-r4", trace_id="chain-io-4", span_data=span_data)
        _on_span_start(span)
        _on_span_end(span)

        state = _trace_states["chain-io-4"]

        assert state.trace_input == {"messages": [{"role": "user", "content": "User msg"}]}
        assert state.trace_output == {"messages": [{"role": "assistant", "content": "Reply"}]}


class TestAgentSpanPropagation:
    """AGENT span must get input/output from child LLM spans."""

    def test_agent_gets_input_from_first_response(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)

        trace = _make_trace(trace_id="agent-prop-1")
        _on_trace_start(trace)

        agent_span = _make_span(
            span_id="ag-1", trace_id="agent-prop-1", span_data=_make_agent_span_data(name="travel_advisor")
        )
        _on_span_start(agent_span)

        response = _make_response(
            instructions="You are a helpful travel advisor.",
            output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Visit Tokyo!"}]}],
        )
        resp_span = _make_span(
            span_id="resp-ag-1",
            trace_id="agent-prop-1",
            parent_id="ag-1",
            span_data=_make_response_span_data(response=response, input_data="Where to go?"),
        )
        _on_span_start(resp_span)
        _on_span_end(resp_span)

        _on_span_end(agent_span)

        state = _trace_states["agent-prop-1"]
        agent_pp_id = state.sdk_to_pandaprobe["ag-1"]
        agent_pp_span = state.spans[agent_pp_id]

        assert agent_pp_span.input == {
            "messages": [
                {"role": "system", "content": "You are a helpful travel advisor."},
                {"role": "user", "content": "Where to go?"},
            ]
        }
        assert agent_pp_span.output == {
            "messages": [{"role": "assistant", "content": "Visit Tokyo!"}]
        }

    def test_agent_output_updates_with_last_response(self):
        adapter = OpenAIAgentsAdapter()
        _store_adapter(adapter)

        trace = _make_trace(trace_id="agent-prop-2")
        _on_trace_start(trace)

        agent_span = _make_span(
            span_id="ag-2", trace_id="agent-prop-2", span_data=_make_agent_span_data(name="my_agent")
        )
        _on_span_start(agent_span)

        resp1 = _make_response(
            output=[{"type": "function_call", "id": "fc-1", "name": "search", "arguments": "{}"}],
            instructions="System",
        )
        s1 = _make_span(
            span_id="r-ag-1", trace_id="agent-prop-2", parent_id="ag-2",
            span_data=_make_response_span_data(response=resp1, input_data="Query"),
        )
        _on_span_start(s1)
        _on_span_end(s1)

        resp2 = _make_response(
            output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Final answer"}]}],
        )
        s2 = _make_span(
            span_id="r-ag-2", trace_id="agent-prop-2", parent_id="ag-2",
            span_data=_make_response_span_data(response=resp2, input_data=[
                {"type": "function_call_output", "call_id": "fc-1", "output": "result"},
            ]),
        )
        _on_span_start(s2)
        _on_span_end(s2)

        _on_span_end(agent_span)

        state = _trace_states["agent-prop-2"]
        agent_pp_id = state.sdk_to_pandaprobe["ag-2"]
        agent_pp_span = state.spans[agent_pp_id]

        assert agent_pp_span.input["messages"][0] == {"role": "system", "content": "System"}
        assert agent_pp_span.output == {
            "messages": [{"role": "assistant", "content": "Final answer"}]
        }
