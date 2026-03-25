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
    extract_name,
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


def _mock_llm_response(text: str):
    """Create a mock LLM response object."""
    from types import SimpleNamespace

    generation = SimpleNamespace(text=text)
    return SimpleNamespace(
        generations=[[generation]],
        llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )
