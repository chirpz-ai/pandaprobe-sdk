"""Tests for pandaprobe.integrations.langchain.callback.

The heavy event-mapping logic is shared with LangGraph through
``pandaprobe.integrations._langchain_core``, so these tests focus on subclass-
specific behavior (default trace name, public class identity) plus a few
smoke tests through the callback API to guard against regressions in the
LangChain entry point.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import respx

import pandaprobe
import pandaprobe.client as client_module
from pandaprobe.integrations._langchain_core.callback import BaseLangChainCallbackHandler
from pandaprobe.integrations.langchain import LangChainCallbackHandler
from pandaprobe.integrations.langchain.callback import (
    LangChainCallbackHandler as LangChainCallbackHandlerDirect,
)
from pandaprobe.schemas import SpanKind


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


def _mock_llm_response(text: str, *, prompt_tokens: int = 10, completion_tokens: int = 5):
    """Create a mock LangChain LLM response with usage_metadata on the message."""
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


class TestSubclassIdentity:
    """LangChainCallbackHandler must be a thin subclass of the shared base."""

    def test_is_subclass_of_base(self):
        assert issubclass(LangChainCallbackHandler, BaseLangChainCallbackHandler)

    def test_default_trace_name_is_langchain(self):
        assert LangChainCallbackHandler.DEFAULT_TRACE_NAME == "LangChain"

    def test_default_trace_name_on_instance(self):
        handler = LangChainCallbackHandler()
        assert handler._trace_name == "LangChain"

    def test_module_re_export_matches(self):
        assert LangChainCallbackHandler is LangChainCallbackHandlerDirect

    def test_finalize_trace_error_log_uses_langchain_label(self, caplog):
        """Diagnostic regression: trace-submission failures must say 'LangChain', not the wrong framework."""
        import logging

        # Point at an unreachable client so log_trace raises inside _finalize_trace.
        original = client_module._global_client
        client_module._global_client = None
        try:
            handler = LangChainCallbackHandler()
            handler._client = None  # force _resolve_client to raise (no global, none injected)
            with caplog.at_level(logging.ERROR, logger="pandaprobe"):
                handler._finalize_trace()
            assert any(
                "PandaProbe LangChain callback failed to submit trace" in rec.getMessage() for rec in caplog.records
            )
            assert not any("PandaProbe LangGraph callback" in rec.getMessage() for rec in caplog.records)
        finally:
            client_module._global_client = original


class TestCallbackHandler:
    @respx.mock
    def test_simple_chain_trace(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangChainCallbackHandler()

        root_id = uuid4()
        handler.on_chain_start({"name": "MyAgent"}, {"query": "hello"}, run_id=root_id)
        handler.on_chain_end({"answer": "world"}, run_id=root_id)

        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    @respx.mock
    def test_nested_chain_with_chat_model(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangChainCallbackHandler()

        root_id = uuid4()
        node_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Agent"}, {"input": "hi"}, run_id=root_id)
        handler.on_chain_start({"name": "ModelNode"}, {"input": "hi"}, run_id=node_id, parent_run_id=root_id)

        msg = SimpleNamespace(type="human", content="hi")
        msg.model_dump = lambda: {"type": "human", "content": "hi"}
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[msg]],
            run_id=llm_id,
            parent_run_id=node_id,
            invocation_params={"model": "gpt-4o-mini", "temperature": 0.5, "api_key": "secret"},
        )

        llm_span = handler._spans[str(llm_id)]
        assert llm_span.kind == SpanKind.LLM
        assert llm_span.input == {"messages": [{"role": "user", "content": "hi"}]}
        assert llm_span.model == "gpt-4o-mini"
        assert llm_span.model_parameters == {"temperature": 0.5}

        handler.on_llm_end(_mock_llm_response("hello"), run_id=llm_id)
        handler.on_chain_end({"output": "hello"}, run_id=node_id)
        handler.on_chain_end({"answer": "hello"}, run_id=root_id)

        assert len(handler._spans) == 0  # cleared after finalization

    @respx.mock
    def test_tool_and_retriever(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangChainCallbackHandler()

        root_id = uuid4()
        tool_id = uuid4()
        ret_id = uuid4()

        handler.on_chain_start({"name": "Agent"}, {}, run_id=root_id)
        handler.on_tool_start({"name": "search"}, "what is python", run_id=tool_id, parent_run_id=root_id)
        assert handler._spans[str(tool_id)].kind == SpanKind.TOOL
        handler.on_tool_end("python is a programming language", run_id=tool_id)

        handler.on_retriever_start({"name": "FAISS"}, "python docs", run_id=ret_id, parent_run_id=root_id)
        assert handler._spans[str(ret_id)].kind == SpanKind.RETRIEVER
        handler.on_retriever_end(["doc1", "doc2"], run_id=ret_id)

        handler.on_chain_end({}, run_id=root_id)
        assert len(handler._spans) == 0

    @respx.mock
    def test_chain_error_finalizes_with_error_status(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangChainCallbackHandler()

        root_id = uuid4()
        handler.on_chain_start({"name": "Agent"}, {}, run_id=root_id)
        handler.on_chain_error(RuntimeError("boom"), run_id=root_id)

        assert len(handler._spans) == 0

    def test_kind_classification_matches_langgraph(self):
        """Root chain -> CHAIN; nested chains -> AGENT; consistent with LangGraph."""
        handler = LangChainCallbackHandler()

        root_id = uuid4()
        child_id = uuid4()
        handler.on_chain_start({"name": "Outer"}, {}, run_id=root_id)
        handler.on_chain_start({"name": "Inner"}, {}, run_id=child_id, parent_run_id=root_id)

        assert handler._spans[str(root_id)].kind == SpanKind.CHAIN
        assert handler._spans[str(child_id)].kind == SpanKind.AGENT

    def test_constructor_propagates_session_user_tags_metadata(self):
        handler = LangChainCallbackHandler(
            session_id="sess-1",
            user_id="user-1",
            tags=["lc", "test"],
            metadata={"version": "1.0"},
        )
        assert handler._session_id == "sess-1"
        assert handler._user_id == "user-1"
        assert handler._tags == ["lc", "test"]
        assert handler._metadata == {"version": "1.0"}

    @respx.mock
    def test_trace_name_overridden_by_root_chain_name(self):
        """The default 'LangChain' trace name is replaced by a custom root chain's name on start."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangChainCallbackHandler()
        assert handler._trace_name == "LangChain"

        root_id = uuid4()
        handler.on_chain_start({"name": "MyCustomChain"}, {}, run_id=root_id)
        assert handler._trace_name == "MyCustomChain"

    @respx.mock
    def test_internal_root_name_langgraph_is_remapped(self):
        """`create_agent` exposes its compiled graph as 'LangGraph'; we rewrite to 'LangChain'."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangChainCallbackHandler()

        root_id = uuid4()
        handler.on_chain_start({"name": "LangGraph"}, {"messages": []}, run_id=root_id)

        assert handler._trace_name == "LangChain"
        assert handler._spans[str(root_id)].name == "LangChain"

    @respx.mock
    def test_internal_root_name_runnable_sequence_is_remapped(self):
        """LCEL pipelines surface as 'RunnableSequence'; we rewrite to 'LangChain'."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangChainCallbackHandler()

        root_id = uuid4()
        handler.on_chain_start({"name": "RunnableSequence"}, {}, run_id=root_id)

        assert handler._trace_name == "LangChain"
        assert handler._spans[str(root_id)].name == "LangChain"

    @respx.mock
    def test_model_node_command_output_is_serialized_as_dict(self):
        """Regression: ``create_agent``'s model node returns ``[Command(update=...)]``.

        ``Command`` is a first-class LangGraph API expressing both state
        updates and routing decisions, so the SDK records it faithfully as a
        structured dict — not a ``repr()`` string, and without stripping the
        routing fields. Inner messages still get ``type``→``role``
        normalization. Uses a local stand-in with the same field layout as
        ``langgraph.types.Command`` so the test runs without ``langgraph``
        installed.
        """
        import dataclasses
        from typing import Any

        @dataclasses.dataclass
        class FakeCommand:
            graph: Any = None
            update: Any = None
            resume: Any = None
            goto: tuple = ()

        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangChainCallbackHandler()

        root_id = uuid4()
        model_id = uuid4()
        handler.on_chain_start({"name": "LangGraph"}, {"messages": []}, run_id=root_id)
        handler.on_chain_start({"name": "model"}, {"messages": []}, run_id=model_id, parent_run_id=root_id)

        msg = SimpleNamespace(type="ai", content="hello", tool_calls=[])
        msg.model_dump = lambda: {"type": "ai", "content": "hello", "tool_calls": []}
        handler.on_chain_end([FakeCommand(update={"messages": [msg]})], run_id=model_id)

        model_span = handler._spans[str(model_id)]
        assert isinstance(model_span.output, list)
        assert len(model_span.output) == 1
        assert isinstance(model_span.output[0], dict)
        assert set(model_span.output[0].keys()) == {"graph", "update", "resume", "goto"}
        assert model_span.output[0]["update"] == {
            "messages": [{"role": "assistant", "content": "hello", "tool_calls": []}]
        }
        assert not any(isinstance(v, str) and v.startswith("FakeCommand(") for v in model_span.output)

    @respx.mock
    def test_nested_internal_name_is_not_remapped(self):
        """Only the root chain name is filtered; nested chains keep their LangChain-given names."""
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangChainCallbackHandler()

        root_id = uuid4()
        nested_id = uuid4()
        handler.on_chain_start({"name": "MyAgent"}, {}, run_id=root_id)
        handler.on_chain_start({"name": "RunnableSequence"}, {}, run_id=nested_id, parent_run_id=root_id)

        assert handler._trace_name == "MyAgent"
        assert handler._spans[str(nested_id)].name == "RunnableSequence"

    @respx.mock
    def test_trace_input_trimmed_to_last_user_message(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = LangChainCallbackHandler()

        root_id = uuid4()
        handler.on_chain_start(
            {"name": "Agent"},
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "First question?"},
                    {"role": "assistant", "content": "First answer."},
                    {"role": "user", "content": "Follow-up?"},
                ]
            },
            run_id=root_id,
        )

        assert handler._trace_input == {"messages": [{"role": "user", "content": "Follow-up?"}]}
