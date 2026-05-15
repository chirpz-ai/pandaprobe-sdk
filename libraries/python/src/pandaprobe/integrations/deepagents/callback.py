"""DeepAgents CallbackHandler for PandaProbe tracing.

Thin subclass of :class:`BaseLangChainCallbackHandler`. ``deepagents`` is an
opinionated harness on top of ``langchain.agents.create_agent``;
``create_deep_agent(...)`` returns a vanilla LangGraph ``CompiledStateGraph``
that fully respects ``config={"callbacks": [...]}``. The full event-mapping
logic is therefore shared with the LangGraph and LangChain integrations via
:mod:`pandaprobe.integrations._langchain_core.callback`.
"""

from __future__ import annotations

from typing import Any

from pandaprobe.integrations._langchain_core.callback import BaseLangChainCallbackHandler


class DeepAgentsCallbackHandler(BaseLangChainCallbackHandler):
    """LangChain ``BaseCallbackHandler`` for ``deepagents``-built agents.

    The deep agent is a LangGraph ``CompiledStateGraph`` under the hood, so the
    standard LangChain callback interface captures the full trace tree —
    including sub-agent invocations dispatched via the built-in ``task`` tool,
    which forwards ``callbacks`` / ``tags`` / ``configurable`` to each
    sub-agent's ``.invoke(...)``. A single handler instance therefore captures
    the parent agent **and** every sub-agent invocation as one trace.

    Span shape recorded for a typical deep-agent run::

        DeepAgents (CHAIN, root)
        ├── tools (AGENT)
        │   └── write_todos (TOOL)
        ├── tools (AGENT)
        │   └── task (TOOL)            ← sub-agent dispatcher; faithful to runtime
        │       └── general-purpose (AGENT)   ← nested sub-agent root
        │           └── …
        └── …

    Usage::

        from pandaprobe.integrations.deepagents import DeepAgentsCallbackHandler
        from deepagents import create_deep_agent

        handler = DeepAgentsCallbackHandler()
        agent = create_deep_agent(tools=[...], instructions="...")
        agent.invoke(
            {"messages": [{"role": "user", "content": "..."}]},
            config={"callbacks": [handler]},
        )
    """

    DEFAULT_TRACE_NAME = "DeepAgents"

    # LangGraph reports the literal string "LangGraph" as the root chain name
    # for every compiled graph. Custom user-declared graph names pass through.
    _INTERNAL_ROOT_NAMES: frozenset[str] = frozenset({"LangGraph"})

    def _filter_root_chain_name(self, name: str, serialized: dict[str, Any] | None) -> str:
        if name in self._INTERNAL_ROOT_NAMES:
            return self.DEFAULT_TRACE_NAME
        return name
