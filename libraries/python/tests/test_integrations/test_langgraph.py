"""Tests for pandaprobe.integrations.langgraph.callback."""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.integrations.langgraph.callback import LangGraphCallbackHandler
from pandaprobe.integrations.langgraph.utils import extract_name, safe_output


@pytest.fixture(autouse=True)
def _setup_client():
    original = client_module._global_client
    pandaprobe.init(api_key="sk_pp_test", project_name="proj", endpoint="http://testserver", flush_interval=60.0)
    yield
    if client_module._global_client is not None:
        client_module._global_client.shutdown()
    client_module._global_client = original


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


def _mock_llm_response(text: str):
    """Create a mock LLM response object."""
    from types import SimpleNamespace

    generation = SimpleNamespace(text=text)
    return SimpleNamespace(
        generations=[[generation]],
        llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )
