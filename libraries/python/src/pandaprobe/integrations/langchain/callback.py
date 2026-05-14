"""LangChain CallbackHandler for PandaProbe tracing.

Thin subclass of :class:`BaseLangChainCallbackHandler`. The full event-mapping
logic is shared with the LangGraph integration via
:mod:`pandaprobe.integrations._langchain_core.callback`.
"""

from __future__ import annotations

from typing import Any

from pandaprobe.integrations._langchain_core.callback import BaseLangChainCallbackHandler


class LangChainCallbackHandler(BaseLangChainCallbackHandler):
    """LangChain ``BaseCallbackHandler`` for LangChain agents and LCEL chains.

    Works with anything built on top of ``langchain-core``: the
    ``create_agent`` factory, plain LCEL pipelines (``prompt | model | parser``),
    custom ``Runnable`` compositions, etc. Spans are keyed by LangChain
    ``run_id`` and submitted as a single :class:`TraceData` when the outermost
    chain finishes.

    Usage with ``create_agent``::

        from pandaprobe.integrations.langchain import LangChainCallbackHandler
        from langchain.agents import create_agent

        handler = LangChainCallbackHandler()
        agent = create_agent("openai:gpt-4o-mini", tools=[...])
        agent.invoke(
            {"messages": [{"role": "user", "content": "..."}]},
            config={"callbacks": [handler]},
        )

    Usage with an LCEL chain::

        from pandaprobe.integrations.langchain import LangChainCallbackHandler

        handler = LangChainCallbackHandler()
        chain = prompt | model | parser
        chain.invoke({"question": "..."}, config={"callbacks": [handler]})
    """

    DEFAULT_TRACE_NAME = "LangChain"

    _INTERNAL_ROOT_NAMES: frozenset[str] = frozenset({"LangGraph", "RunnableSequence"})

    def _filter_root_chain_name(self, name: str, serialized: dict[str, Any] | None) -> str:
        if name in self._INTERNAL_ROOT_NAMES:
            return self.DEFAULT_TRACE_NAME
        return name
