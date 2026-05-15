"""Tests for pandaprobe.integrations.deepagents.callback.

The heavy event-mapping logic is shared with LangGraph and LangChain through
``pandaprobe.integrations._langchain_core``, so these tests focus on subclass-
specific behavior (default trace name, internal-name remap, public class
identity) plus an end-to-end smoke test that walks the exact callback event
sequence a real ``deepagents`` run produces — including a sub-agent dispatch
nested inside the built-in ``task`` tool.
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
from pandaprobe.integrations.deepagents import DeepAgentsCallbackHandler
from pandaprobe.integrations.deepagents.callback import (
    DeepAgentsCallbackHandler as DeepAgentsCallbackHandlerDirect,
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
    """DeepAgentsCallbackHandler must be a thin subclass of the shared base."""

    def test_is_subclass_of_base(self):
        assert issubclass(DeepAgentsCallbackHandler, BaseLangChainCallbackHandler)

    def test_default_trace_name_is_deepagents(self):
        assert DeepAgentsCallbackHandler.DEFAULT_TRACE_NAME == "DeepAgents"

    def test_default_trace_name_on_instance(self):
        handler = DeepAgentsCallbackHandler()
        assert handler._trace_name == "DeepAgents"

    def test_module_re_export_matches(self):
        assert DeepAgentsCallbackHandler is DeepAgentsCallbackHandlerDirect

    def test_finalize_trace_error_log_uses_deepagents_label(self, caplog):
        """Diagnostic regression: trace-submission failures must say 'DeepAgents'."""
        import logging

        class _RaisingClient:
            def log_trace(self, trace):
                raise RuntimeError("boom")

        handler = DeepAgentsCallbackHandler(client=_RaisingClient())
        with caplog.at_level(logging.ERROR, logger="pandaprobe"):
            handler._finalize_trace()
        assert any(
            "PandaProbe DeepAgents callback failed to submit trace" in rec.getMessage() for rec in caplog.records
        )
        assert not any(
            "PandaProbe LangChain callback" in rec.getMessage() or "PandaProbe LangGraph callback" in rec.getMessage()
            for rec in caplog.records
        )


class TestRootChainNameRemap:
    """``deepagents`` exposes its compiled graph as 'LangGraph'; we rewrite to 'DeepAgents'."""

    @respx.mock
    def test_internal_root_name_langgraph_is_remapped(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = DeepAgentsCallbackHandler()

        root_id = uuid4()
        handler.on_chain_start({"name": "LangGraph"}, {"messages": []}, run_id=root_id)

        assert handler._trace_name == "DeepAgents"
        assert handler._spans[str(root_id)].name == "DeepAgents"

    @respx.mock
    def test_custom_root_name_passes_through(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = DeepAgentsCallbackHandler()

        root_id = uuid4()
        handler.on_chain_start({"name": "MyResearchAgent"}, {}, run_id=root_id)

        assert handler._trace_name == "MyResearchAgent"
        assert handler._spans[str(root_id)].name == "MyResearchAgent"

    @respx.mock
    def test_nested_langgraph_name_is_not_remapped(self):
        """Only root names are filtered. The shared base preserves nested names verbatim,
        which is important for sub-agent runs that may also be named 'LangGraph' in odd cases.
        """
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = DeepAgentsCallbackHandler()

        root_id = uuid4()
        nested_id = uuid4()
        handler.on_chain_start({"name": "DeepAgents"}, {}, run_id=root_id)
        handler.on_chain_start({"name": "LangGraph"}, {}, run_id=nested_id, parent_run_id=root_id)

        assert handler._spans[str(nested_id)].name == "LangGraph"


class TestCallbackHandler:
    @respx.mock
    def test_simple_chain_trace(self):
        route = respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = DeepAgentsCallbackHandler()

        root_id = uuid4()
        handler.on_chain_start({"name": "LangGraph"}, {"messages": []}, run_id=root_id)
        handler.on_chain_end({"messages": []}, run_id=root_id)

        pandaprobe.get_client().flush(timeout=5.0)
        assert route.call_count >= 1

    def test_kind_classification_matches_langgraph(self):
        """Root chain -> CHAIN; nested chains -> AGENT; consistent with LangGraph."""
        handler = DeepAgentsCallbackHandler()

        root_id = uuid4()
        child_id = uuid4()
        handler.on_chain_start({"name": "LangGraph"}, {}, run_id=root_id)
        handler.on_chain_start({"name": "tools"}, {}, run_id=child_id, parent_run_id=root_id)

        assert handler._spans[str(root_id)].kind == SpanKind.CHAIN
        assert handler._spans[str(child_id)].kind == SpanKind.AGENT

    def test_constructor_propagates_session_user_tags_metadata(self):
        handler = DeepAgentsCallbackHandler(
            session_id="sess-1",
            user_id="user-1",
            tags=["deepagents", "test"],
            metadata={"version": "1.0"},
        )
        assert handler._session_id == "sess-1"
        assert handler._user_id == "user-1"
        assert handler._tags == ["deepagents", "test"]
        assert handler._metadata == {"version": "1.0"}

    @respx.mock
    def test_chain_error_finalizes_with_error_status(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = DeepAgentsCallbackHandler()

        root_id = uuid4()
        handler.on_chain_start({"name": "LangGraph"}, {}, run_id=root_id)
        handler.on_chain_error(RuntimeError("boom"), run_id=root_id)

        assert len(handler._spans) == 0

    @respx.mock
    def test_trace_input_trimmed_to_last_user_message(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = DeepAgentsCallbackHandler()

        root_id = uuid4()
        handler.on_chain_start(
            {"name": "LangGraph"},
            {
                "messages": [
                    {"role": "system", "content": "You are a deep research agent."},
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Sunny."},
                    {"role": "user", "content": "And tomorrow?"},
                ]
            },
            run_id=root_id,
        )

        assert handler._trace_input == {"messages": [{"role": "user", "content": "And tomorrow?"}]}


class TestEndToEndDeepAgentSequence:
    """Walk the exact callback event sequence a real ``deepagents`` run emits.

    Verifies (a) the full trace tree shape, (b) span kinds for every node type,
    (c) that the ``task`` tool stays kind=TOOL with the sub-agent's root chain
    nested inside as an AGENT (kind heuristic for nested chains).
    """

    @respx.mock
    def test_full_run_with_write_todos_and_subagent_via_task(self):
        respx.post("http://testserver/traces").mock(return_value=httpx.Response(202, json={}))
        handler = DeepAgentsCallbackHandler()

        # IDs for the parent agent's run tree.
        root = uuid4()
        tools_node_1 = uuid4()
        write_todos_tool = uuid4()
        agent_llm_round_2 = uuid4()
        tools_node_2 = uuid4()
        task_tool = uuid4()
        # IDs for the sub-agent dispatched by ``task``.
        subagent_root = uuid4()
        subagent_tools_node = uuid4()
        subagent_grep_tool = uuid4()
        # Final agent LLM round.
        agent_llm_final = uuid4()

        # 1. Root: deepagents wraps a CompiledStateGraph; LangGraph reports name="LangGraph".
        handler.on_chain_start(
            {"name": "LangGraph"},
            {"messages": [{"role": "user", "content": "research X and write me a summary"}]},
            run_id=root,
        )

        # 2. First LLM round inside agent node (deepagents disables tracing on the model
        #    node itself via RunnableCallable(trace=False), so we only see the chat-model events).
        round1_id = uuid4()
        msg = SimpleNamespace(type="human", content="research X and write me a summary")
        msg.model_dump = lambda: {"type": "human", "content": "research X and write me a summary"}
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[msg]],
            run_id=round1_id,
            parent_run_id=root,
            invocation_params={"model": "gpt-5.4-nano"},
        )
        handler.on_llm_end(_mock_llm_response("calling write_todos"), run_id=round1_id)

        # 3. tools node fires, then the write_todos tool itself.
        handler.on_chain_start({"name": "tools"}, {}, run_id=tools_node_1, parent_run_id=root)
        handler.on_tool_start(
            {"name": "write_todos"},
            '[{"content": "investigate X", "status": "pending"}]',
            run_id=write_todos_tool,
            parent_run_id=tools_node_1,
        )
        handler.on_tool_end("Updated todo list to: ...", run_id=write_todos_tool)
        handler.on_chain_end({}, run_id=tools_node_1)

        # 4. Second LLM round; model decides to dispatch a sub-agent via ``task``.
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[msg]],
            run_id=agent_llm_round_2,
            parent_run_id=root,
            invocation_params={"model": "gpt-5.4-nano"},
        )
        handler.on_llm_end(_mock_llm_response("dispatching task"), run_id=agent_llm_round_2)

        # 5. tools node + ``task`` tool that synchronously runs a sub-agent.
        handler.on_chain_start({"name": "tools"}, {}, run_id=tools_node_2, parent_run_id=root)
        handler.on_tool_start(
            {"name": "task"},
            '{"description": "find recent papers on X", "subagent_type": "general-purpose"}',
            run_id=task_tool,
            parent_run_id=tools_node_2,
        )

        # 5a. The sub-agent's compiled graph runs *inside* the task tool body —
        #     callbacks/tags/configurable forwarded, so its events nest under ``task``.
        handler.on_chain_start(
            {"name": "general-purpose"},
            {"messages": [{"role": "user", "content": "find recent papers on X"}]},
            run_id=subagent_root,
            parent_run_id=task_tool,
        )
        handler.on_chain_start(
            {"name": "tools"},
            {},
            run_id=subagent_tools_node,
            parent_run_id=subagent_root,
        )
        handler.on_tool_start(
            {"name": "grep"},
            '{"pattern": "X", "path": "."}',
            run_id=subagent_grep_tool,
            parent_run_id=subagent_tools_node,
        )
        handler.on_tool_end("3 matches", run_id=subagent_grep_tool)
        handler.on_chain_end({}, run_id=subagent_tools_node)
        handler.on_chain_end({"messages": [{"role": "assistant", "content": "found 3 papers"}]}, run_id=subagent_root)

        # 5b. ``task`` tool returns the sub-agent's final state.
        handler.on_tool_end("found 3 papers", run_id=task_tool)
        handler.on_chain_end({}, run_id=tools_node_2)

        # 6. Final agent LLM round + close root.
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[msg]],
            run_id=agent_llm_final,
            parent_run_id=root,
            invocation_params={"model": "gpt-5.4-nano"},
        )
        handler.on_llm_end(_mock_llm_response("here is the summary"), run_id=agent_llm_final)
        handler.on_chain_end(
            {"messages": [{"role": "assistant", "content": "here is the summary"}]},
            run_id=root,
        )

        # All spans flushed on root close.
        assert handler._spans == {}
        assert handler._trace_name == "DeepAgents"

    def test_task_tool_is_classified_as_tool_with_agent_child(self):
        """Faithful-to-runtime mapping: ``task`` is kind=TOOL even though it dispatches a sub-agent.

        The nested sub-agent root chain is naturally classified as AGENT by the
        default ``parent_run_id is not None`` heuristic, which produces a clear
        TOOL→AGENT parent/child shape in the trace tree.
        """
        handler = DeepAgentsCallbackHandler()

        root = uuid4()
        tools_node = uuid4()
        task_tool = uuid4()
        subagent_root = uuid4()

        handler.on_chain_start({"name": "LangGraph"}, {}, run_id=root)
        handler.on_chain_start({"name": "tools"}, {}, run_id=tools_node, parent_run_id=root)
        handler.on_tool_start(
            {"name": "task"},
            "{}",
            run_id=task_tool,
            parent_run_id=tools_node,
        )
        handler.on_chain_start(
            {"name": "general-purpose"},
            {"messages": []},
            run_id=subagent_root,
            parent_run_id=task_tool,
        )

        assert handler._spans[str(task_tool)].kind == SpanKind.TOOL
        assert handler._spans[str(task_tool)].parent_span_id == tools_node
        assert handler._spans[str(subagent_root)].kind == SpanKind.AGENT
        assert handler._spans[str(subagent_root)].parent_span_id == task_tool
        assert handler._spans[str(subagent_root)].name == "general-purpose"
