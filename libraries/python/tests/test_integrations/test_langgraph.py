"""Tests for pandaprobe.integrations.langgraph.callback."""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.integrations.langgraph.callback import LangGraphCallbackHandler
from pandaprobe.integrations.langgraph.utils import (
    _normalize_content_blocks,
    extract_model_parameters,
    extract_name,
    extract_reasoning_from_generation,
    extract_token_usage,
    normalize_langchain_input,
    normalize_langchain_output,
    normalize_llm_generation_output,
    normalize_type_to_role,
    safe_output,
)


@pytest.fixture(autouse=True)
def _setup_client():
    original_client = client_module._global_client
    original_flag = client_module._auto_init_attempted
    pandaprobe.init(api_key="sk_pp_test", project_name="proj", endpoint="http://testserver", flush_interval=60.0)
    yield
    if client_module._global_client is not None:
        client_module._global_client.shutdown()
    client_module._global_client = original_client
    client_module._auto_init_attempted = original_flag


class TestUtils:
    def test_extract_name_from_name_field(self):
        assert extract_name({"name": "MyChain"}) == "MyChain"

    def test_extract_name_from_id_list(self):
        assert extract_name({"id": ["langchain", "chains", "MyChain"]}) == "MyChain"

    def test_extract_name_fallback(self):
        assert extract_name({}) == "unknown"

    def test_safe_output_primitives(self):
        assert safe_output("hello") == "hello"
        assert safe_output(42) == 42
        assert safe_output(None) is None

    def test_safe_output_dict(self):
        assert safe_output({"key": "val"}) == {"key": "val"}

    def test_safe_output_list(self):
        assert safe_output([1, 2, 3]) == [1, 2, 3]


class TestNormalization:
    def test_normalize_langchain_input_list_of_lists(self):
        data = {"messages": [["human", "hello"], ["ai", "hi"]]}
        result = normalize_langchain_input(data)
        assert result == {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        }

    def test_normalize_langchain_input_dicts_with_type(self):
        data = {
            "messages": [
                {"type": "human", "content": "hello", "id": "1"},
                {"type": "ai", "content": "hi", "id": "2"},
            ]
        }
        result = normalize_langchain_input(data)
        assert result["messages"][0] == {"role": "user", "content": "hello", "id": "1"}
        assert result["messages"][1] == {"role": "assistant", "content": "hi", "id": "2"}

    def test_normalize_langchain_input_passthrough(self):
        assert normalize_langchain_input("string") == "string"
        assert normalize_langchain_input({"query": "hi"}) == {"query": "hi"}
        data_no_list = {"messages": "not a list"}
        assert normalize_langchain_input(data_no_list) is data_no_list

    def test_normalize_langchain_output_last_message(self):
        data = {
            "messages": [
                {"type": "human", "content": "hi"},
                {"type": "ai", "content": "hello"},
            ]
        }
        result = normalize_langchain_output(data)
        assert result == {"messages": [{"role": "assistant", "content": "hello"}]}

    def test_normalize_langchain_output_passthrough(self):
        assert normalize_langchain_output("string") == "string"
        assert normalize_langchain_output({"key": "val"}) == {"key": "val"}
        empty = {"messages": []}
        assert normalize_langchain_output(empty) is empty

    def test_normalize_type_to_role_recursive(self):
        data = {
            "messages": [
                {"type": "human", "content": "hi"},
                {"type": "ai", "content": "hello"},
            ]
        }
        result = normalize_type_to_role(data)
        assert result == {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }

    def test_normalize_type_to_role_existing_role(self):
        data = {
            "messages": [
                {"role": "human", "content": "hi"},
                {"role": "ai", "content": "hello"},
            ]
        }
        result = normalize_type_to_role(data)
        assert result == {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }

    def test_normalize_type_to_role_no_content_key(self):
        data = {"type": "chain", "name": "MyChain"}
        result = normalize_type_to_role(data)
        assert result == {"type": "chain", "name": "MyChain"}

    def test_normalize_llm_generation_output_chat(self):
        from types import SimpleNamespace

        msg = SimpleNamespace(type="ai", content="hello")
        msg.model_dump = lambda: {"type": "ai", "content": "hello"}
        gen = SimpleNamespace(message=msg, text="hello")
        response = SimpleNamespace(generations=[[gen]])
        result = normalize_llm_generation_output(response)
        assert result == {"messages": [{"role": "assistant", "content": "hello"}]}

    def test_normalize_llm_generation_output_plain(self):
        from types import SimpleNamespace

        gen = SimpleNamespace(text="hello")
        response = SimpleNamespace(generations=[[gen]])
        result = normalize_llm_generation_output(response)
        assert result == {"messages": [{"role": "assistant", "content": "hello"}]}

    def test_normalize_llm_generation_output_empty(self):
        from types import SimpleNamespace

        response = SimpleNamespace(generations=[])
        assert normalize_llm_generation_output(response) is None
        assert normalize_llm_generation_output(SimpleNamespace()) is None


class TestCallbackHandler:
    @respx.mock
    def test_simple_chain_trace(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangGraphCallbackHandler()

        root_id = uuid4()
        handler.on_chain_start({"name": "MyGraph"}, {"query": "hello"}, run_id=root_id)
        handler.on_chain_end({"answer": "world"}, run_id=root_id)

        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    @respx.mock
    def test_nested_chain_with_llm(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangGraphCallbackHandler()

        root_id = uuid4()
        node_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Graph"}, {"input": "hi"}, run_id=root_id)
        handler.on_chain_start({"name": "AgentNode"}, {"input": "hi"}, run_id=node_id, parent_run_id=root_id)
        handler.on_llm_start(
            {"name": "ChatOpenAI"},
            ["What is 2+2?"],
            run_id=llm_id,
            parent_run_id=node_id,
            invocation_params={"model": "gpt-4"},
        )
        handler.on_llm_end(
            _mock_llm_response("4"),
            run_id=llm_id,
        )
        handler.on_chain_end({"output": "4"}, run_id=node_id)
        handler.on_chain_end({"answer": "4"}, run_id=root_id)

        assert len(handler._spans) == 0  # cleared after finalization

    @respx.mock
    def test_tool_callback(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangGraphCallbackHandler()

        root_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start({"name": "Graph"}, {}, run_id=root_id)
        handler.on_tool_start({"name": "search"}, "query text", run_id=tool_id, parent_run_id=root_id)
        handler.on_tool_end("search results", run_id=tool_id)
        handler.on_chain_end({}, run_id=root_id)

    @respx.mock
    def test_error_handling(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangGraphCallbackHandler()

        root_id = uuid4()
        handler.on_chain_start({"name": "Graph"}, {}, run_id=root_id)
        handler.on_chain_error(RuntimeError("boom"), run_id=root_id)

    @respx.mock
    def test_retriever_callback(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangGraphCallbackHandler()

        root_id = uuid4()
        ret_id = uuid4()

        handler.on_chain_start({"name": "Graph"}, {}, run_id=root_id)
        handler.on_retriever_start({"name": "FAISS"}, "search query", run_id=ret_id, parent_run_id=root_id)
        handler.on_retriever_end(["doc1", "doc2"], run_id=ret_id)
        handler.on_chain_end({}, run_id=root_id)

    @respx.mock
    def test_overrides(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangGraphCallbackHandler(
            session_id="sess-1",
            user_id="user-1",
            tags=["test"],
            metadata={"version": "1.0"},
        )
        assert handler._session_id == "sess-1"
        assert handler._user_id == "user-1"
        assert handler._tags == ["test"]
        assert handler._metadata == {"version": "1.0"}

    @respx.mock
    def test_on_chat_model_start(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangGraphCallbackHandler()

        from types import SimpleNamespace

        root_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Graph"}, {"input": "hi"}, run_id=root_id)

        msg1 = SimpleNamespace(type="system", content="You are helpful.")
        msg1.model_dump = lambda: {"type": "system", "content": "You are helpful."}
        msg2 = SimpleNamespace(type="human", content="hello")
        msg2.model_dump = lambda: {"type": "human", "content": "hello"}

        handler.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[msg1, msg2]],
            run_id=llm_id,
            parent_run_id=root_id,
            invocation_params={"model": "gpt-4"},
        )

        span = handler._spans[str(llm_id)]
        assert span.input == {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ]
        }

    @respx.mock
    def test_trace_input_trimmed_to_last_user_message(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangGraphCallbackHandler()

        root_id = uuid4()
        handler.on_chain_start(
            {"name": "Graph"},
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "Paris."},
                    {"role": "user", "content": "What about Germany?"},
                ]
            },
            run_id=root_id,
        )

        assert handler._trace_input == {"messages": [{"role": "user", "content": "What about Germany?"}]}

    @respx.mock
    def test_on_llm_end_normalized_output(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangGraphCallbackHandler()

        root_id = uuid4()
        llm_id = uuid4()
        handler.on_chain_start({"name": "Graph"}, {"input": "hi"}, run_id=root_id)
        handler.on_llm_start({"name": "ChatOpenAI"}, ["hi"], run_id=llm_id, parent_run_id=root_id)
        handler.on_llm_end(_mock_llm_response("world"), run_id=llm_id)

        span = handler._spans[str(llm_id)]
        assert span.output == {"messages": [{"role": "assistant", "content": "world"}]}

    @respx.mock
    def test_on_chat_model_start_captures_model_parameters(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangGraphCallbackHandler()

        root_id = uuid4()
        llm_id = uuid4()
        handler.on_chain_start({"name": "Graph"}, {"input": "hi"}, run_id=root_id)

        from types import SimpleNamespace

        msg = SimpleNamespace(type="human", content="hi")
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[msg]],
            run_id=llm_id,
            parent_run_id=root_id,
            name="ChatOpenAI",
            invocation_params={
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": 200,
                "api_key": "secret",
            },
        )

        span = handler._spans[str(llm_id)]
        assert span.model == "gpt-4o-mini"
        assert span.model_parameters == {"temperature": 0.3, "max_tokens": 200}
        assert "api_key" not in (span.model_parameters or {})

    @respx.mock
    def test_on_llm_start_captures_model_parameters(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangGraphCallbackHandler()

        root_id = uuid4()
        llm_id = uuid4()
        handler.on_chain_start({"name": "Graph"}, {"input": "hi"}, run_id=root_id)
        handler.on_llm_start(
            {"name": "OpenAI"},
            ["hello"],
            run_id=llm_id,
            parent_run_id=root_id,
            invocation_params={
                "model": "gpt-4o",
                "temperature": 0.7,
                "seed": 42,
                "api_key": "secret",
            },
        )

        span = handler._spans[str(llm_id)]
        assert span.model == "gpt-4o"
        assert span.model_parameters == {"temperature": 0.7, "seed": 42}


def _mock_llm_response(text: str, *, prompt_tokens: int = 10, completion_tokens: int = 5):
    """Create a mock LLM response with usage_metadata on the message."""
    from types import SimpleNamespace

    msg = SimpleNamespace(
        type="ai",
        content=text,
        usage_metadata={
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )
    msg.model_dump = lambda: {"type": "ai", "content": text}
    gen = SimpleNamespace(message=msg, text=text)
    return SimpleNamespace(generations=[[gen]], llm_output=None)


def _mock_gemini_llm_response(text: str):
    """Create a mock Gemini response with reasoning tokens in usage_metadata."""
    from types import SimpleNamespace

    msg = SimpleNamespace(
        type="ai",
        content=text,
        usage_metadata={
            "input_tokens": 431,
            "output_tokens": 196,
            "total_tokens": 627,
            "input_token_details": {"cache_read": 0},
            "output_token_details": {"reasoning": 161},
        },
    )
    msg.model_dump = lambda: {"type": "ai", "content": text}
    gen = SimpleNamespace(message=msg, text=text)
    return SimpleNamespace(generations=[[gen]], llm_output=None)


# ---------------------------------------------------------------------------
# Token usage extraction tests
# ---------------------------------------------------------------------------


class TestExtractTokenUsage:
    def test_basic_usage_metadata(self):
        resp = _mock_llm_response("hello", prompt_tokens=50, completion_tokens=20)
        usage = extract_token_usage(resp)
        assert usage == {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}

    def test_gemini_with_details(self):
        resp = _mock_gemini_llm_response("hello")
        usage = extract_token_usage(resp)
        assert usage["input_tokens"] == 431
        assert usage["output_tokens"] == 196
        assert usage["total_tokens"] == 627
        assert usage["input_token_details"] == {"cache_read": 0}
        assert usage["output_token_details"] == {"reasoning": 161}

    def test_no_message(self):
        from types import SimpleNamespace

        gen = SimpleNamespace(text="hi")
        resp = SimpleNamespace(generations=[[gen]])
        assert extract_token_usage(resp) is None

    def test_no_usage_metadata(self):
        from types import SimpleNamespace

        msg = SimpleNamespace(type="ai", content="hi", usage_metadata=None)
        gen = SimpleNamespace(message=msg, text="hi")
        resp = SimpleNamespace(generations=[[gen]])
        assert extract_token_usage(resp) is None

    def test_empty_generations(self):
        from types import SimpleNamespace

        resp = SimpleNamespace(generations=[])
        assert extract_token_usage(resp) is None

    def test_callback_captures_token_usage(self):
        import respx as respx_mod

        with respx_mod.mock:
            respx_mod.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
            handler = LangGraphCallbackHandler()
            root_id = uuid4()
            llm_id = uuid4()
            handler.on_chain_start({"name": "Graph"}, {"input": "hi"}, run_id=root_id)
            handler.on_llm_start(
                {"name": "Gemini"},
                ["hi"],
                run_id=llm_id,
                parent_run_id=root_id,
            )
            handler.on_llm_end(_mock_gemini_llm_response("world"), run_id=llm_id)

            span = handler._spans[str(llm_id)]
            assert span.token_usage is not None
            assert span.token_usage["input_tokens"] == 431
            assert span.token_usage["output_tokens"] == 196
            assert span.token_usage["output_token_details"] == {"reasoning": 161}


# ---------------------------------------------------------------------------
# Model parameters extraction tests
# ---------------------------------------------------------------------------


class TestExtractModelParameters:
    def test_filters_safe_params(self):
        params = {
            "model": "gpt-4o",
            "temperature": 0.7,
            "seed": 42,
            "api_key": "secret",
        }
        result = extract_model_parameters(params)
        assert result == {"temperature": 0.7, "seed": 42}
        assert "api_key" not in result
        assert "model" not in result

    def test_filters_none_values(self):
        params = {
            "temperature": 1,
            "top_p": None,
            "stop": None,
            "n": None,
        }
        result = extract_model_parameters(params)
        assert result == {"temperature": 1}

    def test_gemini_params(self):
        params = {
            "model": "gemini-3.1-flash-lite-preview",
            "temperature": 1,
            "max_output_tokens": 200,
            "thinking_level": "low",
            "api_key": "secret",
        }
        result = extract_model_parameters(params)
        assert result == {
            "temperature": 1,
            "max_output_tokens": 200,
            "thinking_level": "low",
        }

    def test_empty_returns_none(self):
        assert extract_model_parameters(None) is None
        assert extract_model_parameters({}) is None

    def test_all_none_returns_none(self):
        assert extract_model_parameters({"temperature": None, "stop": None}) is None

    def test_config_object(self):
        from types import SimpleNamespace

        config = SimpleNamespace(
            temperature=0.5,
            max_output_tokens=100,
            thinking_config={"thinking_level": "low"},
            api_key="secret",
        )
        result = extract_model_parameters(config)
        assert result == {
            "temperature": 0.5,
            "max_output_tokens": 100,
            "thinking_config": {"thinking_level": "low"},
        }
        assert "api_key" not in result


# ---------------------------------------------------------------------------
# Thinking block stripping tests
# ---------------------------------------------------------------------------

_GEMINI_CONTENT = [
    {"type": "thinking", "thinking": "Let me reason..."},
    {"type": "text", "text": "The answer is 42.", "extras": {"signature": "abc"}},
]

_ANTHROPIC_CONTENT = [
    {"type": "thinking", "thinking": "Step by step..."},
    {"type": "text", "text": "The result is 7."},
]

_GEMINI_TEXT_ONLY = [
    {"type": "text", "text": "Hello world.", "extras": {"signature": "xyz"}},
]


class TestNormalizeContentBlocks:
    """Test universal content normalization across Gemini, Anthropic, OpenAI."""

    def test_gemini_thinking_plus_text_collapses_to_string(self):
        assert _normalize_content_blocks(_GEMINI_CONTENT) == "The answer is 42."

    def test_anthropic_thinking_plus_text_collapses_to_string(self):
        assert _normalize_content_blocks(_ANTHROPIC_CONTENT) == "The result is 7."

    def test_gemini_text_only_collapses_to_string(self):
        assert _normalize_content_blocks(_GEMINI_TEXT_ONLY) == "Hello world."

    def test_openai_string_passthrough(self):
        assert _normalize_content_blocks("plain text") == "plain text"

    def test_preserves_non_list(self):
        assert _normalize_content_blocks(42) == 42
        assert _normalize_content_blocks(None) is None

    def test_keeps_all_if_only_thinking(self):
        only_thinking = [{"type": "thinking", "thinking": "hmm"}]
        assert _normalize_content_blocks(only_thinking) == only_thinking

    def test_multiple_text_blocks_joined(self):
        blocks = [
            {"type": "text", "text": "Part one."},
            {"type": "text", "text": "Part two."},
        ]
        assert _normalize_content_blocks(blocks) == "Part one. Part two."

    def test_mixed_non_text_blocks_preserved_as_list(self):
        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ]
        result = _normalize_content_blocks(blocks)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_normalize_langchain_input_collapses_gemini_content(self):
        data = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": list(_GEMINI_CONTENT)},
                {"role": "user", "content": "follow up"},
            ]
        }
        result = normalize_langchain_input(data)
        assert result["messages"][1]["content"] == "The answer is 42."

    def test_normalize_langchain_input_collapses_with_type_key(self):
        data = {
            "messages": [
                {"type": "ai", "content": list(_GEMINI_CONTENT)},
            ]
        }
        result = normalize_langchain_input(data)
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "The answer is 42."

    def test_normalize_langchain_output_collapses(self):
        data = {
            "messages": [
                {"type": "ai", "content": list(_GEMINI_CONTENT)},
            ]
        }
        result = normalize_langchain_output(data)
        assert result["messages"][0]["content"] == "The answer is 42."

    def test_normalize_type_to_role_collapses(self):
        data = {
            "messages": [
                {"type": "ai", "content": list(_GEMINI_CONTENT)},
            ]
        }
        result = normalize_type_to_role(data)
        assert result["messages"][0]["content"] == "The answer is 42."

    def test_normalize_type_to_role_collapses_with_role_key(self):
        data = {
            "messages": [
                {"role": "assistant", "content": list(_GEMINI_TEXT_ONLY)},
            ]
        }
        result = normalize_type_to_role(data)
        assert result["messages"][0]["content"] == "Hello world."

    def test_normalize_llm_generation_output_collapses(self):
        from types import SimpleNamespace

        msg = SimpleNamespace(type="ai", content=list(_GEMINI_CONTENT))
        msg.model_dump = lambda: {"type": "ai", "content": list(_GEMINI_CONTENT)}
        gen = SimpleNamespace(message=msg, text="The answer is 42.")
        response = SimpleNamespace(generations=[[gen]])
        result = normalize_llm_generation_output(response)
        assert result["messages"][0]["content"] == "The answer is 42."


# ---------------------------------------------------------------------------
# Reasoning extraction tests
# ---------------------------------------------------------------------------


class TestExtractReasoningFromGeneration:
    def test_anthropic_thinking_blocks(self):
        """Anthropic-style thinking blocks in content list."""
        from types import SimpleNamespace

        msg = SimpleNamespace(
            content=[
                {"type": "thinking", "thinking": "Let me reason about this..."},
                {"type": "text", "text": "The answer is 42."},
            ],
            additional_kwargs={},
        )
        gen = SimpleNamespace(message=msg)
        response = SimpleNamespace(generations=[[gen]])
        assert extract_reasoning_from_generation(response) == "Let me reason about this..."

    def test_anthropic_multiple_thinking_blocks(self):
        from types import SimpleNamespace

        msg = SimpleNamespace(
            content=[
                {"type": "thinking", "thinking": "Step 1"},
                {"type": "thinking", "thinking": "Step 2"},
                {"type": "text", "text": "Done."},
            ],
            additional_kwargs={},
        )
        gen = SimpleNamespace(message=msg)
        response = SimpleNamespace(generations=[[gen]])
        assert extract_reasoning_from_generation(response) == "Step 1\n\nStep 2"

    def test_openai_reasoning_content(self):
        """OpenAI-style reasoning_content in additional_kwargs."""
        from types import SimpleNamespace

        msg = SimpleNamespace(
            content="The answer.",
            additional_kwargs={"reasoning_content": "I need to think about this carefully."},
        )
        gen = SimpleNamespace(message=msg)
        response = SimpleNamespace(generations=[[gen]])
        assert extract_reasoning_from_generation(response) == "I need to think about this carefully."

    def test_openai_reasoning_field(self):
        """OpenAI-compatible reasoning field."""
        from types import SimpleNamespace

        msg = SimpleNamespace(
            content="The answer.",
            additional_kwargs={"reasoning": "My reasoning process."},
        )
        gen = SimpleNamespace(message=msg)
        response = SimpleNamespace(generations=[[gen]])
        assert extract_reasoning_from_generation(response) == "My reasoning process."

    def test_no_reasoning(self):
        from types import SimpleNamespace

        msg = SimpleNamespace(content="Just a plain answer.", additional_kwargs={})
        gen = SimpleNamespace(message=msg)
        response = SimpleNamespace(generations=[[gen]])
        assert extract_reasoning_from_generation(response) is None

    def test_no_generations(self):
        from types import SimpleNamespace

        response = SimpleNamespace(generations=[])
        assert extract_reasoning_from_generation(response) is None

    def test_no_message(self):
        """Generation without message attribute (plain LLM, not chat)."""
        from types import SimpleNamespace

        gen = SimpleNamespace(text="plain text")
        response = SimpleNamespace(generations=[[gen]])
        assert extract_reasoning_from_generation(response) is None
