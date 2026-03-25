"""Tests for pandaprobe.integrations.google_adk."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.integrations.google_adk.adapter import (
    GoogleADKAdapter,
    _TraceState,
    _current_span_id,
    _current_trace_state,
    _get_adapter,
    _get_current_span,
    _get_trace_state,
    _set_current_span,
    _set_trace_state,
    _store_adapter,
    _wrap_agent_run_async,
    _wrap_llm_call_async,
    _wrap_runner_run_async,
    _wrap_tool_run_async,
)
from pandaprobe.integrations.google_adk.utils import (
    extract_model_name,
    extract_model_parameters,
    extract_text_from_content,
    extract_token_usage,
    normalize_contents_to_messages,
    normalize_llm_response_to_messages,
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
# Mock ADK objects
# ---------------------------------------------------------------------------


def _make_content(role: str = "user", text: str = "hello"):
    part = SimpleNamespace(
        text=text,
        function_call=None,
        function_response=None,
        executable_code=None,
        code_execution_result=None,
        thought=None,
    )
    return SimpleNamespace(role=role, parts=[part])


def _make_function_call_part(name: str, args: dict):
    fc = SimpleNamespace(name=name, args=args)
    return SimpleNamespace(
        text=None,
        function_call=fc,
        function_response=None,
        executable_code=None,
        code_execution_result=None,
        thought=None,
    )


def _make_function_response_part(name: str, response: dict):
    fr = SimpleNamespace(name=name, response=response)
    return SimpleNamespace(
        text=None,
        function_call=None,
        function_response=fr,
        executable_code=None,
        code_execution_result=None,
        thought=None,
    )


def _make_llm_request(model="gemini-2.5-flash", contents=None, system_instruction=None, **config_overrides):
    defaults = dict(
        model=model,
        system_instruction=system_instruction,
        temperature=None,
        top_p=None,
        top_k=None,
        max_output_tokens=None,
        stop_sequences=None,
        seed=None,
        candidate_count=None,
        presence_penalty=None,
        frequency_penalty=None,
        response_modalities=None,
        response_mime_type=None,
        thinking_config=None,
    )
    defaults.update(config_overrides)
    config = SimpleNamespace(**defaults)
    return SimpleNamespace(config=config, contents=contents or [])


def _make_event(content=None, partial=False, usage_metadata=None):
    return SimpleNamespace(content=content, partial=partial, usage_metadata=usage_metadata)


def _make_usage_metadata(prompt=10, candidates=20, total=30):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=total,
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


class TestExtractTextFromContent:
    def test_text_content(self):
        content = _make_content("user", "Hello world")
        assert extract_text_from_content(content) == "Hello world"

    def test_multiple_text_parts(self):
        p1 = SimpleNamespace(text="Hello")
        p2 = SimpleNamespace(text="world")
        content = SimpleNamespace(parts=[p1, p2])
        assert extract_text_from_content(content) == "Hello world"

    def test_none_content(self):
        assert extract_text_from_content(None) is None

    def test_no_text_parts(self):
        part = SimpleNamespace(text=None)
        content = SimpleNamespace(parts=[part])
        assert extract_text_from_content(content) is None

    def test_empty_parts(self):
        content = SimpleNamespace(parts=[])
        assert extract_text_from_content(content) is None


class TestNormalizeContentsToMessages:
    def test_simple_user_message(self):
        contents = [_make_content("user", "Hello")]
        result = normalize_contents_to_messages(contents)
        assert result == {"messages": [{"role": "user", "content": "Hello"}]}

    def test_model_role_mapped_to_assistant(self):
        contents = [_make_content("model", "Hi there")]
        result = normalize_contents_to_messages(contents)
        assert result["messages"][0]["role"] == "assistant"

    def test_system_instruction(self):
        sys = _make_content("system", "Be helpful")
        contents = [_make_content("user", "Hi")]
        result = normalize_contents_to_messages(contents, system_instruction=sys)
        assert len(result["messages"]) == 2
        assert result["messages"][0] == {"role": "system", "content": "Be helpful"}

    def test_system_instruction_string(self):
        contents = [_make_content("user", "Hi")]
        result = normalize_contents_to_messages(contents, system_instruction="Be helpful")
        assert result["messages"][0] == {"role": "system", "content": "Be helpful"}

    def test_function_call_in_content(self):
        fc_part = _make_function_call_part("search", {"query": "test"})
        text_part = SimpleNamespace(
            text="Let me search",
            function_call=None,
            function_response=None,
            executable_code=None,
            code_execution_result=None,
            thought=None,
        )
        content = SimpleNamespace(role="model", parts=[text_part, fc_part])
        result = normalize_contents_to_messages([content])
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me search"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["name"] == "search"

    def test_function_response(self):
        fr_part = _make_function_response_part("search", {"result": "found"})
        content = SimpleNamespace(role="user", parts=[fr_part])
        result = normalize_contents_to_messages([content])
        msg = result["messages"][0]
        assert msg["role"] == "tool"
        assert msg["name"] == "search"

    def test_inline_data_part(self):
        inline = SimpleNamespace(data=b"image-bytes", mime_type="image/png")
        part = SimpleNamespace(
            text=None,
            function_call=None,
            function_response=None,
            executable_code=None,
            code_execution_result=None,
            thought=None,
            inline_data=inline,
            file_data=None,
        )
        content = SimpleNamespace(role="user", parts=[part])
        result = normalize_contents_to_messages([content])
        msg = result["messages"][0]
        assert msg["role"] == "user"
        import base64

        assert msg["content"] == [
            {"type": "image", "data": base64.b64encode(b"image-bytes").decode("utf-8"), "mime_type": "image/png"}
        ]

    def test_file_data_part(self):
        file_d = SimpleNamespace(file_uri="gs://bucket/file.pdf", mime_type="application/pdf")
        part = SimpleNamespace(
            text=None,
            function_call=None,
            function_response=None,
            executable_code=None,
            code_execution_result=None,
            thought=None,
            inline_data=None,
            file_data=file_d,
        )
        content = SimpleNamespace(role="user", parts=[part])
        result = normalize_contents_to_messages([content])
        msg = result["messages"][0]
        assert msg["content"] == [{"type": "file", "file_uri": "gs://bucket/file.pdf", "mime_type": "application/pdf"}]

    def test_empty_contents(self):
        result = normalize_contents_to_messages([])
        assert result == {"messages": []}

    def test_none_contents(self):
        result = normalize_contents_to_messages(None)
        assert result == {"messages": []}


class TestNormalizeLlmResponseToMessages:
    def test_text_response(self):
        content = _make_content("model", "Hello!")
        result = normalize_llm_response_to_messages(content)
        assert result == {"messages": [{"role": "assistant", "content": "Hello!"}]}

    def test_none_response(self):
        result = normalize_llm_response_to_messages(None)
        assert result == {"messages": [{"role": "assistant", "content": None}]}


class TestExtractTokenUsage:
    def test_valid_usage(self):
        event = _make_event(usage_metadata=_make_usage_metadata(10, 20, 30))
        result = extract_token_usage(event)
        assert result == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def test_no_usage(self):
        event = _make_event()
        assert extract_token_usage(event) is None

    def test_partial_usage(self):
        meta = SimpleNamespace(prompt_token_count=5, candidates_token_count=None, total_token_count=None)
        event = _make_event(usage_metadata=meta)
        result = extract_token_usage(event)
        assert result == {"prompt_tokens": 5}

    def test_cached_content_tokens(self):
        meta = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
            cached_content_token_count=30,
            thoughts_token_count=None,
        )
        event = _make_event(usage_metadata=meta)
        result = extract_token_usage(event)
        assert result["prompt_tokens"] == 100
        assert result["cache_read_tokens"] == 30

    def test_thoughts_tokens(self):
        meta = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
            cached_content_token_count=None,
            thoughts_token_count=200,
        )
        event = _make_event(usage_metadata=meta)
        result = extract_token_usage(event)
        assert result["completion_tokens"] == 50
        assert result["reasoning_tokens"] == 200

    def test_all_extended_usage(self):
        meta = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=350,
            cached_content_token_count=20,
            thoughts_token_count=180,
        )
        event = _make_event(usage_metadata=meta)
        result = extract_token_usage(event)
        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 350,
            "cache_read_tokens": 20,
            "reasoning_tokens": 180,
        }


class TestExtractModelName:
    def test_from_config(self):
        req = _make_llm_request(model="gemini-2.5-flash")
        assert extract_model_name(req) == "gemini-2.5-flash"

    def test_no_model(self):
        req = SimpleNamespace(config=SimpleNamespace(model=None), model=None)
        assert extract_model_name(req) is None

    def test_from_top_level(self):
        req = SimpleNamespace(config=SimpleNamespace(model=None), model="fallback-model")
        assert extract_model_name(req) == "fallback-model"


class TestExtractModelParameters:
    def test_extracts_safe_params(self):
        req = _make_llm_request(temperature=0.7)
        result = extract_model_parameters(req)
        assert result == {"temperature": 0.7}

    def test_extracts_multiple_params(self):
        req = _make_llm_request(temperature=0.5, top_p=0.9, max_output_tokens=1024, presence_penalty=0.3)
        result = extract_model_parameters(req)
        assert result == {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_output_tokens": 1024,
            "presence_penalty": 0.3,
        }

    def test_extracts_thinking_config(self):
        req = _make_llm_request(temperature=0.7, thinking_config={"thinking_budget": 2048})
        result = extract_model_parameters(req)
        assert result == {"temperature": 0.7, "thinking_config": {"thinking_budget": 2048}}

    def test_extracts_response_modalities(self):
        req = _make_llm_request(response_modalities=["TEXT"])
        result = extract_model_parameters(req)
        assert result == {"response_modalities": ["TEXT"]}

    def test_no_config(self):
        req = SimpleNamespace(config=None)
        assert extract_model_parameters(req) is None

    def test_no_params(self):
        req = _make_llm_request()
        assert extract_model_parameters(req) is None

    def test_pydantic_model_dump(self):
        """Config with model_dump() should be used preferentially."""

        class PydanticLikeConfig:
            def model_dump(self, exclude_none=False):
                d = {"temperature": 0.8, "top_p": 0.95, "model": "gemini-2.5-flash", "system_instruction": "hi"}
                if exclude_none:
                    return d
                return {**d, "top_k": None, "seed": None}

        req = SimpleNamespace(config=PydanticLikeConfig(), contents=[])
        result = extract_model_parameters(req)
        assert result == {"temperature": 0.8, "top_p": 0.95}


# ===================================================================
# Adapter tests
# ===================================================================


class TestGoogleADKAdapter:
    def test_extends_base(self):
        adapter = GoogleADKAdapter(
            session_id="s1",
            user_id="u1",
            tags=["t"],
            metadata={"k": "v"},
        )
        assert adapter._session_id == "s1"
        assert adapter._user_id == "u1"
        assert adapter._tags == ["t"]
        assert adapter._metadata == {"k": "v"}

    def test_instrument_without_adk_returns_false(self):
        import pandaprobe.integrations.google_adk.adapter as mod

        original = mod._patched
        mod._patched = False
        try:
            adapter = GoogleADKAdapter()
            with patch.dict("sys.modules", {"google.adk": None}):
                result = adapter.instrument()
            assert result is False
        finally:
            mod._patched = original


class TestAdapterStorage:
    def test_store_and_get(self):
        adapter = GoogleADKAdapter()
        _store_adapter(adapter)
        assert _get_adapter() is adapter


# ===================================================================
# Context var tests
# ===================================================================


class TestContextVars:
    def test_trace_state(self):
        adapter = GoogleADKAdapter()
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
# Wrapper tests (using mock ADK objects)
# ===================================================================


class TestWrapRunnerRunAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_creates_trace_and_root_span(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = GoogleADKAdapter(session_id="sess-1", user_id="user-1", tags=["test"])
        _store_adapter(adapter)

        final_content = _make_content("model", "The answer is 42")
        events = [
            _make_event(content=final_content),
        ]

        async def mock_run_async(*args, **kwargs):
            for e in events:
                yield e

        instance = SimpleNamespace(app_name="test_app")
        collected = []
        async for event in _wrap_runner_run_async(
            mock_run_async, instance, (), {"new_message": _make_content("user", "What is 42?")}
        ):
            collected.append(event)

        assert len(collected) == 1
        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_error_produces_error_trace(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = GoogleADKAdapter()
        _store_adapter(adapter)

        async def mock_run_async(*args, **kwargs):
            yield _make_event()
            raise RuntimeError("boom")

        instance = SimpleNamespace(app_name="test_app")
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in _wrap_runner_run_async(
                mock_run_async, instance, (), {"new_message": _make_content("user", "Hi")}
            ):
                pass

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_adapter_passthrough(self):
        _store_adapter(None)

        events_in = [_make_event(content=_make_content("model", "hi"))]

        async def mock_run(*args, **kwargs):
            for e in events_in:
                yield e

        collected = []
        async for event in _wrap_runner_run_async(mock_run, SimpleNamespace(), (), {}):
            collected.append(event)
        assert len(collected) == 1


class TestWrapAgentRunAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_creates_agent_span(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = GoogleADKAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        root_span_id = str(uuid4())
        parent_token = _set_current_span(root_span_id)

        try:
            agent_content = _make_content("model", "I'll help")
            events = [_make_event(content=agent_content)]

            async def mock_agent_run(*args, **kwargs):
                for e in events:
                    yield e

            instance = SimpleNamespace(name="my_agent")
            collected = []
            async for event in _wrap_agent_run_async(mock_agent_run, instance, (), {}):
                collected.append(event)

            assert len(collected) == 1
            assert len(state.spans) == 1
            span = list(state.spans.values())[0]
            assert span.name == "my_agent"
            assert span.kind == SpanKind.AGENT
            assert str(span.parent_span_id) == root_span_id
            assert span.status == SpanStatusCode.OK
            assert span.output == {"messages": [{"role": "assistant", "content": "I'll help"}]}
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    @pytest.mark.asyncio
    async def test_full_messages_from_session(self):
        """Agent span should contain full conversation history from session."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = GoogleADKAdapter()
        state = _TraceState(adapter=adapter)
        root_span_id = str(uuid4())
        from datetime import datetime, timezone
        from pandaprobe.schemas import SpanData

        root_span = SpanData(
            span_id=root_span_id,
            name="test",
            kind=SpanKind.CHAIN,
            started_at=datetime.now(timezone.utc),
        )
        state.spans[root_span_id] = root_span
        state.root_span_id = root_span_id

        state_token = _set_trace_state(state)
        parent_token = _set_current_span(root_span_id)

        try:
            session_events = [
                SimpleNamespace(content=_make_content("user", "Hello")),
                SimpleNamespace(content=_make_content("model", "Hi there")),
                SimpleNamespace(content=_make_content("user", "How are you?")),
            ]
            ctx = SimpleNamespace(session=SimpleNamespace(events=session_events))

            agent_content = _make_content("model", "I'm fine!")
            post_events = list(session_events) + [SimpleNamespace(content=agent_content)]

            async def mock_agent_run(*args, **kwargs):
                yield _make_event(content=agent_content)
                ctx.session.events = post_events

            instance = SimpleNamespace(name="my_agent", instruction="Be helpful.")
            collected = []
            async for event in _wrap_agent_run_async(mock_agent_run, instance, (ctx,), {}):
                collected.append(event)

            agent_span = [s for s in state.spans.values() if s.kind == SpanKind.AGENT][0]
            assert agent_span.input == {
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"},
                ]
            }
            assert agent_span.output == {"messages": [{"role": "assistant", "content": "I'm fine!"}]}

            assert root_span.input == {
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"},
                ]
            }
            assert state.chain_output == {
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm fine!"},
                ]
            }
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_state_passthrough(self):
        state_token = _set_trace_state(None)
        try:
            events_in = [_make_event()]

            async def mock_agent_run(*args, **kwargs):
                for e in events_in:
                    yield e

            collected = []
            async for event in _wrap_agent_run_async(mock_agent_run, SimpleNamespace(name="a"), (), {}):
                collected.append(event)
            assert len(collected) == 1
        finally:
            _current_trace_state.reset(state_token)


class TestWrapLlmCallAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_creates_llm_span_with_usage(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = GoogleADKAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        agent_span_id = str(uuid4())
        parent_token = _set_current_span(agent_span_id)

        try:
            response_content = _make_content("model", "Hello!")
            usage = _make_usage_metadata(10, 20, 30)
            final_event = _make_event(content=response_content, usage_metadata=usage)

            async def mock_llm_call(*args, **kwargs):
                yield _make_event(partial=True)
                yield final_event

            llm_request = _make_llm_request(
                model="gemini-2.5-flash",
                contents=[_make_content("user", "Hi")],
                system_instruction=_make_content("system", "Be helpful"),
                temperature=0.7,
            )

            collected = []
            async for event in _wrap_llm_call_async(
                mock_llm_call, SimpleNamespace(), (SimpleNamespace(), llm_request), {}
            ):
                collected.append(event)

            assert len(collected) == 2
            assert len(state.spans) == 1
            span = list(state.spans.values())[0]
            assert span.name == "gemini-2.5-flash"
            assert span.kind == SpanKind.LLM
            assert span.model == "gemini-2.5-flash"
            assert str(span.parent_span_id) == agent_span_id
            assert span.token_usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            assert span.status == SpanStatusCode.OK
            assert span.completion_start_time is not None
            assert span.model_parameters == {"temperature": 0.7}
            assert span.input["messages"][0] == {"role": "system", "content": "Be helpful"}
            assert span.output == {"messages": [{"role": "assistant", "content": "Hello!"}]}
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    @pytest.mark.asyncio
    async def test_llm_error(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = GoogleADKAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:

            async def mock_llm_call(*args, **kwargs):
                raise ValueError("model error")
                yield  # make it an async generator

            llm_request = _make_llm_request(model="gemini-2.5-flash")

            with pytest.raises(ValueError, match="model error"):
                async for _ in _wrap_llm_call_async(
                    mock_llm_call, SimpleNamespace(), (SimpleNamespace(), llm_request), {}
                ):
                    pass

            span = list(state.spans.values())[0]
            assert span.status == SpanStatusCode.ERROR
            assert span.error == "model error"
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)


class TestWrapToolRunAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_creates_tool_span(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = GoogleADKAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        agent_span_id = str(uuid4())
        parent_token = _set_current_span(agent_span_id)

        try:

            async def mock_tool_run(*args, **kwargs):
                return {"result": "weather is sunny"}

            instance = SimpleNamespace(name="get_weather")
            result = await _wrap_tool_run_async(mock_tool_run, instance, (), {"args": {"city": "London"}})

            assert result == {"result": "weather is sunny"}
            assert len(state.spans) == 1
            span = list(state.spans.values())[0]
            assert span.name == "get_weather"
            assert span.kind == SpanKind.TOOL
            assert str(span.parent_span_id) == agent_span_id
            assert span.input == {"city": "London"}
            assert span.output == {"result": "weather is sunny"}
            assert span.status == SpanStatusCode.OK
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_error(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = GoogleADKAdapter()
        state = _TraceState(adapter=adapter)
        state_token = _set_trace_state(state)
        parent_token = _set_current_span(str(uuid4()))

        try:

            async def mock_tool_run(*args, **kwargs):
                raise RuntimeError("tool failed")

            with pytest.raises(RuntimeError, match="tool failed"):
                await _wrap_tool_run_async(mock_tool_run, SimpleNamespace(name="bad_tool"), (), {})

            span = list(state.spans.values())[0]
            assert span.status == SpanStatusCode.ERROR
            assert span.error == "tool failed"
        finally:
            _current_trace_state.reset(state_token)
            _current_span_id.reset(parent_token)

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_state_passthrough(self):
        state_token = _set_trace_state(None)
        try:

            async def mock_tool_run(*args, **kwargs):
                return "direct result"

            result = await _wrap_tool_run_async(mock_tool_run, SimpleNamespace(name="t"), (), {})
            assert result == "direct result"
        finally:
            _current_trace_state.reset(state_token)


# ===================================================================
# Full lifecycle test
# ===================================================================


class TestFullLifecycle:
    @respx.mock
    @pytest.mark.asyncio
    async def test_agent_llm_tool_lifecycle(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = GoogleADKAdapter(
            session_id="sess-lifecycle",
            user_id="user-lifecycle",
            tags=["lifecycle-test"],
            metadata={"env": "test"},
        )
        _store_adapter(adapter)

        llm_response_content = _make_content("model", "The weather in London is sunny.")
        usage = _make_usage_metadata(15, 25, 40)
        llm_request = _make_llm_request(
            model="gemini-2.5-flash",
            contents=[_make_content("user", "What's the weather?")],
        )

        async def mock_tool_run(*a, **kw):
            return {"weather": "sunny", "temp": "22C"}

        async def mock_llm_call(*a, **kw):
            yield _make_event(content=llm_response_content, usage_metadata=usage)

        async def mock_agent_run(*a, **kw):
            state = _get_trace_state()
            assert state is not None

            tool_instance = SimpleNamespace(name="get_weather")
            await _wrap_tool_run_async(mock_tool_run, tool_instance, (), {"args": {"city": "London"}})

            async for event in _wrap_llm_call_async(
                mock_llm_call, SimpleNamespace(), (SimpleNamespace(), llm_request), {}
            ):
                yield event

        async def mock_runner_run(*a, **kw):
            agent_instance = SimpleNamespace(name="weather_agent")
            async for event in _wrap_agent_run_async(mock_agent_run, agent_instance, (), {}):
                yield event

        runner_instance = SimpleNamespace(app_name="weather_app")
        collected = []
        async for event in _wrap_runner_run_async(
            mock_runner_run,
            runner_instance,
            (),
            {"new_message": _make_content("user", "What's the weather in London?")},
        ):
            collected.append(event)

        assert len(collected) == 1

        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_nested_agent_hierarchy(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        adapter = GoogleADKAdapter()
        _store_adapter(adapter)

        async def mock_sub_agent_run(*a, **kw):
            yield _make_event(content=_make_content("model", "sub-agent result"))

        async def mock_agent_run(*a, **kw):
            sub_instance = SimpleNamespace(name="sub_agent")
            async for event in _wrap_agent_run_async(mock_sub_agent_run, sub_instance, (), {}):
                yield event

        async def mock_runner_run(*a, **kw):
            agent_instance = SimpleNamespace(name="root_agent")
            async for event in _wrap_agent_run_async(mock_agent_run, agent_instance, (), {}):
                yield event

        runner_instance = SimpleNamespace(app_name="nested_app")
        collected = []
        async for event in _wrap_runner_run_async(
            mock_runner_run,
            runner_instance,
            (),
            {"new_message": _make_content("user", "Go")},
        ):
            collected.append(event)

        assert len(collected) == 1


class TestConfigurationOverrides:
    def test_default_values(self):
        adapter = GoogleADKAdapter()
        assert adapter._session_id is None
        assert adapter._user_id is None
        assert adapter._tags == []
        assert adapter._metadata == {}

    def test_custom_values(self):
        adapter = GoogleADKAdapter(
            session_id="s",
            user_id="u",
            tags=["a", "b"],
            metadata={"key": "val"},
        )
        assert adapter._session_id == "s"
        assert adapter._user_id == "u"
        assert adapter._tags == ["a", "b"]
        assert adapter._metadata == {"key": "val"}
