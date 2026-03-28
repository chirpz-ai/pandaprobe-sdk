"""Tests for pandaprobe.integrations.crewai."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.integrations.crewai.adapter import (
    CrewAIAdapter,
    _AgentRecord,
    _TraceState,
    _build_chain_output,
    _current_span_id,
    _current_trace_state,
    _get_adapter,
    _get_current_span,
    _get_trace_state,
    _set_current_span,
    _set_trace_state,
    _store_adapter,
    _wrap_agent_execute_task,
    _wrap_kickoff,
    _wrap_llm_call,
    _wrap_tool_execute,
)
from pandaprobe.integrations.crewai.utils import (
    _normalize_content,
    _normalize_role,
    build_agent_system_message,
    build_crew_system_message,
    extract_model_name,
    extract_model_parameters,
    extract_reasoning_from_messages,
    extract_token_usage,
    normalize_messages,
    safe_serialize,
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
# Mock CrewAI objects
# ---------------------------------------------------------------------------


def _make_crew(tasks=None, agents=None, process="sequential"):
    return SimpleNamespace(
        tasks=tasks or [],
        agents=agents or [],
        process=process,
        verbose=False,
        memory=False,
    )


def _make_agent(role="Researcher", goal="Find info", backstory="Expert researcher", model="gpt-4o"):
    llm = SimpleNamespace(
        model=model,
        temperature=0.7,
        top_p=None,
        n=None,
        stop=None,
        max_completion_tokens=None,
        max_tokens=1024,
        presence_penalty=None,
        frequency_penalty=None,
        seed=None,
        _token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "successful_requests": 0,
            "cached_prompt_tokens": 0,
        },
    )
    return SimpleNamespace(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        tools=[],
        allow_delegation=False,
        max_iter=15,
    )


def _make_task(description="Research the topic", expected_output="A summary", agent=None):
    return SimpleNamespace(
        description=description,
        expected_output=expected_output,
        agent=agent,
        tools=[],
        async_execution=False,
        human_input=False,
    )


def _make_crew_output(raw="The final result", token_usage=None, tasks_output=None):
    return SimpleNamespace(
        raw=raw,
        token_usage=token_usage or {},
        tasks_output=tasks_output or [],
        pydantic=None,
        json_dict=None,
    )


def _make_llm(model="gpt-4o", temperature=0.7, max_tokens=1024):
    return SimpleNamespace(
        model=model,
        temperature=temperature,
        top_p=None,
        n=None,
        stop=None,
        max_completion_tokens=None,
        max_tokens=max_tokens,
        presence_penalty=None,
        frequency_penalty=None,
        seed=None,
        _token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "successful_requests": 0,
            "cached_prompt_tokens": 0,
        },
    )


def _make_agent_action(tool="get_weather", tool_input='{"city": "London"}', thought="I should check", text=""):
    return SimpleNamespace(tool=tool, tool_input=tool_input, thought=thought, text=text, result=None)


def _make_tool_result(result="sunny, 22C", result_as_answer=False):
    return SimpleNamespace(result=result, result_as_answer=result_as_answer)


# ===================================================================
# Utils tests
# ===================================================================


class TestNormalizeRole:
    def test_standard_roles(self):
        assert _normalize_role("user") == "user"
        assert _normalize_role("assistant") == "assistant"
        assert _normalize_role("system") == "system"
        assert _normalize_role("tool") == "tool"

    def test_mapped_roles(self):
        assert _normalize_role("human") == "user"
        assert _normalize_role("ai") == "assistant"
        assert _normalize_role("model") == "assistant"
        assert _normalize_role("function") == "tool"

    def test_unknown_passthrough(self):
        assert _normalize_role("custom_role") == "custom_role"


class TestNormalizeContent:
    def test_string_passthrough(self):
        assert _normalize_content("hello") == "hello"

    def test_none_passthrough(self):
        assert _normalize_content(None) is None

    def test_strips_thinking_blocks(self):
        content = [
            {"type": "thinking", "thinking": "Let me think..."},
            {"type": "text", "text": "The answer is 42."},
        ]
        assert _normalize_content(content) == "The answer is 42."

    def test_collapses_text_only_list(self):
        content = [
            {"type": "text", "text": "Part one."},
            {"type": "text", "text": "Part two."},
        ]
        assert _normalize_content(content) == "Part one. Part two."

    def test_single_text_block(self):
        content = [{"type": "text", "text": "Just one."}]
        assert _normalize_content(content) == "Just one."

    def test_preserves_mixed_content(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "url": "http://example.com/img.png"},
        ]
        result = _normalize_content(content)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_all_thinking_preserves_original(self):
        content = [{"type": "thinking", "thinking": "hmm"}]
        assert _normalize_content(content) == content

    def test_string_items_in_list(self):
        content = ["hello", "world"]
        assert _normalize_content(content) == "hello world"


class TestNormalizeMessages:
    def test_simple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = normalize_messages(messages)
        assert result == {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

    def test_role_normalization(self):
        messages = [
            {"role": "human", "content": "hello"},
            {"role": "ai", "content": "hi"},
        ]
        result = normalize_messages(messages)
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_string_messages(self):
        messages = ["What is 2+2?"]
        result = normalize_messages(messages)
        assert result == {"messages": [{"role": "user", "content": "What is 2+2?"}]}

    def test_empty_list(self):
        assert normalize_messages([]) == {"messages": []}

    def test_non_list_input(self):
        assert normalize_messages("not a list") == {"messages": []}
        assert normalize_messages(None) == {"messages": []}

    def test_thinking_stripped_from_content(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "reasoning..."},
                    {"type": "text", "text": "The answer."},
                ],
            }
        ]
        result = normalize_messages(messages)
        assert result["messages"][0]["content"] == "The answer."


class TestExtractReasoningFromMessages:
    def test_extracts_thinking(self):
        messages = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me reason..."},
                    {"type": "text", "text": "The answer."},
                ],
            },
        ]
        assert extract_reasoning_from_messages(messages) == "Let me reason..."

    def test_multiple_thinking_blocks(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Step 1"},
                    {"type": "thinking", "thinking": "Step 2"},
                    {"type": "text", "text": "Done."},
                ],
            }
        ]
        assert extract_reasoning_from_messages(messages) == "Step 1\n\nStep 2"

    def test_no_thinking(self):
        messages = [{"role": "assistant", "content": "plain text"}]
        assert extract_reasoning_from_messages(messages) is None

    def test_empty_messages(self):
        assert extract_reasoning_from_messages([]) is None

    def test_non_list(self):
        assert extract_reasoning_from_messages("string") is None
        assert extract_reasoning_from_messages(None) is None

    def test_thinking_text_key(self):
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "text": "via text key"}],
            }
        ]
        assert extract_reasoning_from_messages(messages) == "via text key"


class TestExtractTokenUsage:
    def test_basic_usage(self):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        result = extract_token_usage(usage)
        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_with_cached_and_reasoning(self):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cached_prompt_tokens": 30,
            "reasoning_tokens": 20,
        }
        result = extract_token_usage(usage)
        assert result["cache_read_tokens"] == 30
        assert result["reasoning_tokens"] == 20

    def test_computes_total(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50}
        result = extract_token_usage(usage)
        assert result["total_tokens"] == 150

    def test_zero_cached_omitted(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cached_prompt_tokens": 0}
        result = extract_token_usage(usage)
        assert "cache_read_tokens" not in result

    def test_empty_usage(self):
        assert extract_token_usage({}) is None
        assert extract_token_usage(None) is None

    def test_object_usage(self):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        result = extract_token_usage(usage)
        assert result == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_non_int_values(self):
        usage = {"prompt_tokens": "100", "completion_tokens": "50", "total_tokens": "150"}
        result = extract_token_usage(usage)
        assert result == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    def test_from_llm_instance(self):
        llm = _make_llm()
        llm._token_usage = {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}
        result = extract_token_usage(llm)
        assert result == {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}

    def test_from_llm_instance_no_token_usage(self):
        llm = SimpleNamespace(model="gpt-4o")
        assert extract_token_usage(llm) is None

    def test_pydantic_like_instance_with_getitem_does_not_raise(self):
        """Pydantic models have __getitem__; must not crash with KeyError."""

        class FakePydanticLLM:
            _token_usage = None

            def __getitem__(self, key: str) -> object:
                raise KeyError(key)

        assert extract_token_usage(FakePydanticLLM()) is None

    def test_llm_instance_with_empty_token_usage(self):
        """Empty _token_usage dict falls through to source; must not crash."""

        class FakeLLM:
            _token_usage: dict = {}

            def __getitem__(self, key: str) -> object:
                raise KeyError(key)

        assert extract_token_usage(FakeLLM()) is None


class TestExtractModelParameters:
    def test_extracts_safe_params(self):
        llm = _make_llm(temperature=0.7, max_tokens=1024)
        result = extract_model_parameters(llm)
        assert result == {"temperature": 0.7, "max_tokens": 1024}

    def test_filters_unsafe_params(self):
        llm = SimpleNamespace(
            temperature=0.5,
            api_key="secret",
            model="gpt-4o",
        )
        result = extract_model_parameters(llm)
        assert result == {"temperature": 0.5}
        assert "api_key" not in result
        assert "model" not in result

    def test_none_values_dropped(self):
        llm = _make_llm()
        result = extract_model_parameters(llm)
        assert "top_p" not in (result or {})
        assert "stop" not in (result or {})

    def test_none_llm(self):
        assert extract_model_parameters(None) is None

    def test_thinking_config(self):
        llm = SimpleNamespace(temperature=0.5, thinking_config={"thinking_budget": 2048})
        result = extract_model_parameters(llm)
        assert result == {"temperature": 0.5, "thinking_config": {"thinking_budget": 2048}}


class TestExtractModelName:
    def test_from_model_attr(self):
        llm = _make_llm(model="gpt-4o")
        assert extract_model_name(llm) == "gpt-4o"

    def test_none_llm(self):
        assert extract_model_name(None) is None

    def test_no_model_attr(self):
        assert extract_model_name(SimpleNamespace()) is None


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


class TestBuildAgentSystemMessage:
    def test_full_agent(self):
        agent = _make_agent(role="Researcher", goal="Find info", backstory="Expert")
        result = build_agent_system_message(agent)
        assert "Role: Researcher" in result
        assert "Goal: Find info" in result
        assert "Backstory: Expert" in result

    def test_partial_agent(self):
        agent = SimpleNamespace(role="Writer", goal=None, backstory=None)
        result = build_agent_system_message(agent)
        assert result == "Role: Writer"

    def test_empty_agent(self):
        agent = SimpleNamespace()
        assert build_agent_system_message(agent) is None


class TestBuildCrewSystemMessage:
    def test_crew_with_agents(self):
        agents = [
            _make_agent(role="Researcher", goal="Research"),
            _make_agent(role="Writer", goal="Write"),
        ]
        crew = _make_crew(agents=agents)
        result = build_crew_system_message(crew)
        assert "Researcher" in result
        assert "Writer" in result
        assert "Research" in result
        assert "Write" in result

    def test_empty_crew(self):
        crew = _make_crew(agents=[])
        assert build_crew_system_message(crew) is None


# ===================================================================
# Adapter tests
# ===================================================================


class TestCrewAIAdapter:
    def test_extends_base(self):
        adapter = CrewAIAdapter(
            session_id="s1",
            user_id="u1",
            tags=["t"],
            metadata={"k": "v"},
        )
        assert adapter._session_id == "s1"
        assert adapter._user_id == "u1"
        assert adapter._tags == ["t"]
        assert adapter._metadata == {"k": "v"}

    def test_default_values(self):
        adapter = CrewAIAdapter()
        assert adapter._session_id is None
        assert adapter._user_id is None
        assert adapter._tags == []
        assert adapter._metadata == {}

    def test_instrument_without_crewai_returns_false(self):
        import pandaprobe.integrations.crewai.adapter as mod

        original = mod._patched
        mod._patched = False
        try:
            adapter = CrewAIAdapter()
            with patch.dict("sys.modules", {"crewai": None}):
                result = adapter.instrument()
            assert result is False
        finally:
            mod._patched = original


class TestAdapterStorage:
    def test_store_and_get(self):
        adapter = CrewAIAdapter()
        _store_adapter(adapter)
        assert _get_adapter() is adapter

    def test_none_adapter(self):
        _store_adapter(None)
        assert _get_adapter() is None


# ===================================================================
# Context var tests
# ===================================================================


class TestContextVars:
    def test_trace_state(self):
        adapter = CrewAIAdapter()
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


# ===================================================================
# Wrapper tests (using mock CrewAI objects)
# ===================================================================


class TestWrapKickoff:
    @respx.mock
    def test_creates_trace_and_root_span(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter(session_id="sess-1", user_id="user-1", tags=["test"])
        _store_adapter(adapter)

        crew_output = _make_crew_output(raw="The final answer")

        def mock_kickoff(*args, **kwargs):
            return crew_output

        task = _make_task("Research AI trends")
        crew = _make_crew(tasks=[task])
        result = _wrap_kickoff(mock_kickoff, crew, (), {"inputs": {"topic": "AI"}})

        assert result is crew_output
        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    @respx.mock
    def test_error_produces_error_trace(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        _store_adapter(adapter)

        def mock_kickoff(*args, **kwargs):
            raise RuntimeError("crew failed")

        crew = _make_crew()
        with pytest.raises(RuntimeError, match="crew failed"):
            _wrap_kickoff(mock_kickoff, crew, (), {})

    @respx.mock
    def test_no_adapter_passthrough(self):
        _store_adapter(None)

        def mock_kickoff(*args, **kwargs):
            return _make_crew_output("result")

        result = _wrap_kickoff(mock_kickoff, _make_crew(), (), {})
        assert result.raw == "result"

    @respx.mock
    def test_trace_input_from_inputs_kwarg(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        _store_adapter(adapter)

        state_holder = {}

        def mock_kickoff(*args, **kwargs):
            state_holder["state"] = _get_trace_state()
            return _make_crew_output("done")

        crew = _make_crew()
        _wrap_kickoff(mock_kickoff, crew, (), {"inputs": {"query": "What is AI?"}})

        state = state_holder["state"]
        assert state is not None
        assert state.trace_input == {"messages": [{"role": "user", "content": "query: What is AI?"}]}

    @respx.mock
    def test_trace_input_from_positional_arg(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        _store_adapter(adapter)

        state_holder = {}

        def mock_kickoff(*args, **kwargs):
            state_holder["state"] = _get_trace_state()
            return _make_crew_output("done")

        crew = _make_crew()
        _wrap_kickoff(mock_kickoff, crew, ({"query": "What is AI?"},), {})

        state = state_holder["state"]
        assert state is not None
        assert state.trace_input == {"messages": [{"role": "user", "content": "query: What is AI?"}]}

    @respx.mock
    def test_trace_input_from_task_descriptions(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        _store_adapter(adapter)

        state_holder = {}

        def mock_kickoff(*args, **kwargs):
            state_holder["state"] = _get_trace_state()
            return _make_crew_output("done")

        task1 = _make_task("Research topic A")
        task2 = _make_task("Write report on A")
        crew = _make_crew(tasks=[task1, task2])
        _wrap_kickoff(mock_kickoff, crew, (), {})

        state = state_holder["state"]
        assert state is not None
        assert state.trace_input is not None

    @respx.mock
    def test_chain_span_has_no_token_usage(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        _store_adapter(adapter)

        state_holder = {}

        def mock_kickoff(*args, **kwargs):
            state_holder["state"] = _get_trace_state()
            return _make_crew_output(
                "done",
                token_usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
            )

        crew = _make_crew()
        _wrap_kickoff(mock_kickoff, crew, (), {})

        state = state_holder["state"]
        root_span = state.spans[state.root_span_id]
        assert root_span.token_usage is None

    @respx.mock
    def test_chain_input_has_system_message_from_crew(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        _store_adapter(adapter)

        state_holder = {}

        def mock_kickoff(*args, **kwargs):
            state_holder["state"] = _get_trace_state()
            return _make_crew_output("done")

        agents = [_make_agent(role="Researcher", goal="Research")]
        task = _make_task("Do research")
        crew = _make_crew(tasks=[task], agents=agents)
        _wrap_kickoff(mock_kickoff, crew, (), {})

        state = state_holder["state"]
        root_span = state.spans[state.root_span_id]
        messages = root_span.input["messages"]
        assert messages[0]["role"] == "system"
        assert "Researcher" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "Do research" in messages[1]["content"]

    @respx.mock
    def test_chain_output_from_agent_records(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        _store_adapter(adapter)

        state_holder = {}

        def mock_kickoff(*args, **kwargs):
            state = _get_trace_state()
            state_holder["state"] = state
            state.agent_records.append(_AgentRecord(agent="R", task="task1", output="out1"))
            state.agent_records.append(_AgentRecord(agent="W", task="task2", output="out2"))
            return _make_crew_output("done")

        agents = [_make_agent(role="R"), _make_agent(role="W")]
        crew = _make_crew(tasks=[_make_task("task1"), _make_task("task2")], agents=agents)
        _wrap_kickoff(mock_kickoff, crew, (), {})

        state = state_holder["state"]
        root_span = state.spans[state.root_span_id]
        output_messages = root_span.output["messages"]
        assert output_messages[0]["role"] == "system"
        assert output_messages[1] == {"role": "user", "content": "task1"}
        assert output_messages[2] == {"role": "assistant", "content": "out1"}
        assert output_messages[3] == {"role": "user", "content": "task2"}
        assert output_messages[4] == {"role": "assistant", "content": "out2"}


class TestBuildChainOutput:
    def test_with_records_and_system(self):
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state.crew_system_message = "Crew agents:\n- Agent1"
        state.agent_records = [
            _AgentRecord(agent="A1", task="Task 1", output="Output 1"),
        ]
        result = _build_chain_output(state)
        assert result == {
            "messages": [
                {"role": "system", "content": "Crew agents:\n- Agent1"},
                {"role": "user", "content": "Task 1"},
                {"role": "assistant", "content": "Output 1"},
            ]
        }

    def test_empty_records(self):
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        assert _build_chain_output(state) is None


class TestWrapAgentExecuteTask:
    @respx.mock
    def test_creates_agent_span_with_system_message(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        root_span_id = str(uuid4())
        parent_token = _set_current_span(root_span_id)

        try:

            def mock_execute_task(*args, **kwargs):
                return SimpleNamespace(raw="Task completed successfully")

            agent = _make_agent(role="Researcher", goal="Find info", backstory="Expert", model="gpt-4o")
            task = _make_task("Research the topic")
            result = _wrap_agent_execute_task(mock_execute_task, agent, (task,), {})

            assert result.raw == "Task completed successfully"
            assert len(state.spans) == 1
            span = list(state.spans.values())[0]
            assert span.name == "Researcher"
            assert span.kind == SpanKind.AGENT
            assert str(span.parent_span_id) == root_span_id
            assert span.status == SpanStatusCode.OK
            assert span.model == "gpt-4o"
            assert span.output == {"messages": [{"role": "assistant", "content": "Task completed successfully"}]}

            messages = span.input["messages"]
            assert messages[0]["role"] == "system"
            assert "Researcher" in messages[0]["content"]
            assert "Expert" in messages[0]["content"]
            assert messages[-1]["role"] == "user"
            assert "Research the topic" in messages[-1]["content"]
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_expected_output_in_metadata(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:

            def mock_execute_task(*args, **kwargs):
                return SimpleNamespace(raw="Done")

            agent = _make_agent(role="Writer")
            task = _make_task("Write a blog post", expected_output="A 500-word article")
            _wrap_agent_execute_task(mock_execute_task, agent, (task,), {})

            span = list(state.spans.values())[0]
            assert span.metadata["expected_output"] == "A 500-word article"
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_no_model_parameters_on_agent_span(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:

            def mock_execute_task(*args, **kwargs):
                return SimpleNamespace(raw="Done")

            agent = _make_agent(role="Agent")
            task = _make_task("Task")
            _wrap_agent_execute_task(mock_execute_task, agent, (task,), {})

            span = list(state.spans.values())[0]
            assert span.model_parameters is None
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_prior_agent_context_in_input(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state.agent_records.append(_AgentRecord(agent="Researcher", task="Research AI", output="AI findings"))
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:

            def mock_execute_task(*args, **kwargs):
                return SimpleNamespace(raw="Article written")

            agent = _make_agent(role="Writer")
            task = _make_task("Write article")
            _wrap_agent_execute_task(mock_execute_task, agent, (task,), {})

            span = list(state.spans.values())[0]
            messages = span.input["messages"]
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "assistant"
            assert "[Researcher]" in messages[1]["content"]
            assert "AI findings" in messages[1]["content"]
            assert messages[2]["role"] == "user"
            assert "Write article" in messages[2]["content"]
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_agent_records_agent_output(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:

            def mock_execute_task(*args, **kwargs):
                return SimpleNamespace(raw="Research results")

            agent = _make_agent(role="Researcher")
            task = _make_task("Research AI")
            _wrap_agent_execute_task(mock_execute_task, agent, (task,), {})

            assert len(state.agent_records) == 1
            assert state.agent_records[0].agent == "Researcher"
            assert state.agent_records[0].task == "Research AI"
            assert state.agent_records[0].output == "Research results"
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_agent_error(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:

            def mock_execute_task(*args, **kwargs):
                raise ValueError("agent failed")

            agent = _make_agent(role="Broken Agent")
            task = _make_task("Do something")

            with pytest.raises(ValueError, match="agent failed"):
                _wrap_agent_execute_task(mock_execute_task, agent, (task,), {})

            span = list(state.spans.values())[0]
            assert span.status == SpanStatusCode.ERROR
            assert span.error == "agent failed"
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_no_state_passthrough(self):
        state_token = _set_trace_state(None)
        try:

            def mock_execute_task(*args, **kwargs):
                return "direct result"

            result = _wrap_agent_execute_task(mock_execute_task, _make_agent(), (_make_task(),), {})
            assert result == "direct result"
        finally:
            _current_trace_state.reset(state_token)


class TestWrapLlmCall:
    @respx.mock
    def test_creates_llm_span(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        agent_span_id = str(uuid4())
        parent_token = _set_current_span(agent_span_id)

        try:

            def mock_llm_call(*args, **kwargs):
                return "The capital of France is Paris."

            llm = _make_llm(model="gpt-4o", temperature=0.7, max_tokens=1024)
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the capital of France?"},
            ]
            result = _wrap_llm_call(mock_llm_call, llm, (messages,), {})

            assert result == "The capital of France is Paris."
            assert len(state.spans) == 1
            span = list(state.spans.values())[0]
            assert span.name == "gpt-4o"
            assert span.kind == SpanKind.LLM
            assert span.model == "gpt-4o"
            assert str(span.parent_span_id) == agent_span_id
            assert span.status == SpanStatusCode.OK
            assert span.model_parameters == {"temperature": 0.7, "max_tokens": 1024}
            assert span.input == {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "What is the capital of France?"},
                ]
            }
            assert span.output == {"messages": [{"role": "assistant", "content": "The capital of France is Paris."}]}
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_token_usage_is_per_call_delta(self):
        """Token usage must reflect only the current call, not cumulative totals."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:
            llm = _make_llm(model="gpt-4o")
            # Simulate a prior call having already accumulated tokens
            llm._token_usage = {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }

            def mock_llm_call(*args, **kwargs):
                # Simulate the LLM call incrementing the cumulative counter
                llm._token_usage = {
                    "prompt_tokens": 220,
                    "completion_tokens": 80,
                    "total_tokens": 300,
                }
                return "response"

            messages = [{"role": "user", "content": "hi"}]
            _wrap_llm_call(mock_llm_call, llm, (messages,), {})

            span = list(state.spans.values())[0]
            assert span.token_usage == {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
            }
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_token_usage_not_compounded_across_calls(self):
        """Two consecutive LLM calls must each record only their own tokens."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:
            llm = _make_llm(model="gpt-4o")
            llm._token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

            call_count = 0

            def mock_llm_call(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    llm._token_usage = {
                        "prompt_tokens": 100,
                        "completion_tokens": 40,
                        "total_tokens": 140,
                    }
                else:
                    llm._token_usage = {
                        "prompt_tokens": 350,
                        "completion_tokens": 110,
                        "total_tokens": 460,
                    }
                return f"response {call_count}"

            messages = [{"role": "user", "content": "hi"}]
            _wrap_llm_call(mock_llm_call, llm, (messages,), {})
            _wrap_llm_call(mock_llm_call, llm, (messages,), {})

            spans = list(state.spans.values())
            assert spans[0].token_usage == {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
            }
            assert spans[1].token_usage == {
                "prompt_tokens": 250,
                "completion_tokens": 70,
                "total_tokens": 320,
            }
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_llm_error(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:

            def mock_llm_call(*args, **kwargs):
                raise ValueError("model error")

            llm = _make_llm(model="gpt-4o")
            with pytest.raises(ValueError, match="model error"):
                _wrap_llm_call(mock_llm_call, llm, ([{"role": "user", "content": "hi"}],), {})

            span = list(state.spans.values())[0]
            assert span.status == SpanStatusCode.ERROR
            assert span.error == "model error"
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_no_state_passthrough(self):
        state_token = _set_trace_state(None)
        try:

            def mock_llm_call(*args, **kwargs):
                return "direct result"

            result = _wrap_llm_call(mock_llm_call, _make_llm(), ([],), {})
            assert result == "direct result"
        finally:
            _current_trace_state.reset(state_token)

    @respx.mock
    def test_string_messages_input(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:

            def mock_llm_call(*args, **kwargs):
                return "response"

            llm = _make_llm(model="gpt-4o")
            _wrap_llm_call(mock_llm_call, llm, ("What is 2+2?",), {})

            span = list(state.spans.values())[0]
            assert span.input == {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_reasoning_extraction(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:

            def mock_llm_call(*args, **kwargs):
                return "The answer is 42."

            llm = _make_llm(model="gpt-4o")
            messages = [
                {"role": "user", "content": "What is the meaning of life?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Deep philosophical question..."},
                        {"type": "text", "text": "42."},
                    ],
                },
            ]
            _wrap_llm_call(mock_llm_call, llm, (messages,), {})

            span = list(state.spans.values())[0]
            assert span.metadata.get("reasoning_summary") == "Deep philosophical question..."
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_no_model_name_fallback(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:

            def mock_llm_call(*args, **kwargs):
                return "response"

            llm = SimpleNamespace(_token_usage={})
            _wrap_llm_call(mock_llm_call, llm, ([],), {})

            span = list(state.spans.values())[0]
            assert span.name == "crewai_llm"
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)


class TestWrapToolExecute:
    @respx.mock
    def test_creates_tool_span_from_agent_action(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        agent_span_id = str(uuid4())
        parent_token = _set_current_span(agent_span_id)

        try:

            def mock_tool_execute(*args, **kwargs):
                return _make_tool_result("sunny, 22C")

            action = _make_agent_action(tool="get_weather", tool_input='{"city": "London"}')
            result = _wrap_tool_execute(mock_tool_execute, None, (action,), {})

            assert result.result == "sunny, 22C"
            assert len(state.spans) == 1
            span = list(state.spans.values())[0]
            assert span.name == "get_weather"
            assert span.kind == SpanKind.TOOL
            assert str(span.parent_span_id) == agent_span_id
            assert span.input == '{"city": "London"}'
            assert span.output == "sunny, 22C"
            assert span.status == SpanStatusCode.OK
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_tool_error(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:

            def mock_tool_execute(*args, **kwargs):
                raise RuntimeError("tool broke")

            action = _make_agent_action(tool="bad_tool")
            with pytest.raises(RuntimeError, match="tool broke"):
                _wrap_tool_execute(mock_tool_execute, None, (action,), {})

            span = list(state.spans.values())[0]
            assert span.status == SpanStatusCode.ERROR
            assert span.error == "tool broke"
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    def test_no_state_passthrough(self):
        state_token = _set_trace_state(None)
        try:

            def mock_tool_execute(*args, **kwargs):
                return _make_tool_result("result")

            action = _make_agent_action(tool="tool")
            result = _wrap_tool_execute(mock_tool_execute, None, (action,), {})
            assert result.result == "result"
        finally:
            _current_trace_state.reset(state_token)

    @respx.mock
    def test_tool_result_dataclass(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:

            def mock_tool_execute(*args, **kwargs):
                return _make_tool_result("tool output string", result_as_answer=True)

            action = _make_agent_action(tool="my_tool", tool_input="args")
            _wrap_tool_execute(mock_tool_execute, None, (action,), {})

            span = list(state.spans.values())[0]
            assert span.output == "tool output string"
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)


# ===================================================================
# Full lifecycle test
# ===================================================================


class TestFullLifecycle:
    @respx.mock
    def test_crew_agent_llm_tool_lifecycle(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter(
            session_id="sess-lifecycle",
            user_id="user-lifecycle",
            tags=["lifecycle-test"],
            metadata={"env": "test"},
        )
        _store_adapter(adapter)

        def mock_tool_execute(*args, **kwargs):
            return _make_tool_result("sunny, 22C")

        def mock_llm_call(*args, **kwargs):
            return "Based on the weather data, London is sunny at 22C."

        def mock_execute_task(*args, **kwargs):
            state = _get_trace_state()
            assert state is not None

            action = _make_agent_action(tool="get_weather", tool_input='{"city": "London"}')
            _wrap_tool_execute(mock_tool_execute, None, (action,), {})

            llm = _make_llm(model="gpt-4o")
            messages = [
                {"role": "system", "content": "You are a weather expert."},
                {"role": "user", "content": "What's the weather in London?"},
            ]
            _wrap_llm_call(lambda *a, **k: "London is sunny at 22C.", llm, (messages,), {})

            return SimpleNamespace(raw="London is sunny at 22C.")

        def mock_kickoff(*args, **kwargs):
            agent = _make_agent(role="Weather Agent", model="gpt-4o")
            task = _make_task("Get the weather in London")
            _wrap_agent_execute_task(mock_execute_task, agent, (task,), {})
            return _make_crew_output("London is sunny at 22C.")

        task = _make_task("Get the weather in London")
        crew = _make_crew(tasks=[task])
        result = _wrap_kickoff(mock_kickoff, crew, (), {"inputs": {"city": "London"}})

        assert result.raw == "London is sunny at 22C."
        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    @respx.mock
    def test_multi_agent_lifecycle(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter(session_id="multi-agent")
        _store_adapter(adapter)

        def mock_execute_researcher(*args, **kwargs):
            return SimpleNamespace(raw="Research findings about AI")

        def mock_execute_writer(*args, **kwargs):
            return SimpleNamespace(raw="Final article about AI")

        def mock_kickoff(*args, **kwargs):
            researcher = _make_agent(role="Researcher", model="gpt-4o")
            writer = _make_agent(role="Writer", model="gpt-4o")

            task1 = _make_task("Research AI trends")
            _wrap_agent_execute_task(mock_execute_researcher, researcher, (task1,), {})

            task2 = _make_task("Write article about AI")
            _wrap_agent_execute_task(mock_execute_writer, writer, (task2,), {})

            return _make_crew_output("Final article about AI")

        task1 = _make_task("Research AI trends")
        task2 = _make_task("Write article about AI")
        researcher = _make_agent(role="Researcher")
        writer = _make_agent(role="Writer")
        crew = _make_crew(tasks=[task1, task2], agents=[researcher, writer])
        result = _wrap_kickoff(mock_kickoff, crew, (), {})

        assert result.raw == "Final article about AI"
        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    @respx.mock
    def test_multi_agent_context_propagation(self):
        """Second agent should see first agent's output in its input."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter()
        _store_adapter(adapter)

        span_holder = {}

        def mock_execute_researcher(*args, **kwargs):
            return SimpleNamespace(raw="AI is trending")

        def mock_execute_writer(*args, **kwargs):
            state = _get_trace_state()
            writer_span_id = list(state.spans.keys())[-1]
            span_holder["writer_span"] = state.spans[writer_span_id]
            return SimpleNamespace(raw="Article about AI trends")

        def mock_kickoff(*args, **kwargs):
            researcher = _make_agent(role="Researcher")
            writer = _make_agent(role="Writer")

            _wrap_agent_execute_task(mock_execute_researcher, researcher, (_make_task("Research AI"),), {})
            _wrap_agent_execute_task(mock_execute_writer, writer, (_make_task("Write article"),), {})

            return _make_crew_output("Article about AI trends")

        crew = _make_crew(
            tasks=[_make_task("Research AI"), _make_task("Write article")],
            agents=[_make_agent(role="Researcher"), _make_agent(role="Writer")],
        )
        _wrap_kickoff(mock_kickoff, crew, (), {})

        writer_span = span_holder["writer_span"]
        messages = writer_span.input["messages"]
        context_msg = next(m for m in messages if m["role"] == "assistant")
        assert "[Researcher]" in context_msg["content"]
        assert "AI is trending" in context_msg["content"]


# ===================================================================
# Configuration overrides test
# ===================================================================


class TestConfigurationOverrides:
    @respx.mock
    def test_session_and_user_from_adapter(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = CrewAIAdapter(
            session_id="custom-session",
            user_id="custom-user",
            tags=["tag1", "tag2"],
            metadata={"source": "test"},
        )
        _store_adapter(adapter)

        def mock_kickoff(*args, **kwargs):
            return _make_crew_output("done")

        crew = _make_crew()
        _wrap_kickoff(mock_kickoff, crew, (), {})

        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    def test_custom_values_propagated(self):
        adapter = CrewAIAdapter(
            session_id="s",
            user_id="u",
            tags=["a", "b"],
            metadata={"key": "val"},
        )
        assert adapter._session_id == "s"
        assert adapter._user_id == "u"
        assert adapter._tags == ["a", "b"]
        assert adapter._metadata == {"key": "val"}


# ===================================================================
# Instrument idempotency test
# ===================================================================


class TestInstrumentIdempotency:
    def test_reregisters_adapter(self):
        adapter1 = CrewAIAdapter(session_id="first")
        adapter2 = CrewAIAdapter(session_id="second")

        _store_adapter(adapter1)
        assert _get_adapter()._session_id == "first"

        _store_adapter(adapter2)
        assert _get_adapter()._session_id == "second"
