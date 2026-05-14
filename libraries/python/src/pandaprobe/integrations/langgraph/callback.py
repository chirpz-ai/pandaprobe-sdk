"""LangGraph CallbackHandler for PandaProbe tracing.

A thin subclass of :class:`BaseLangChainCallbackHandler` that only changes
the default trace name. The full event-mapping logic lives in
:mod:`pandaprobe.integrations._langchain_core.callback`.
"""

from __future__ import annotations

from pandaprobe.integrations._langchain_core.callback import BaseLangChainCallbackHandler


class LangGraphCallbackHandler(BaseLangChainCallbackHandler):
    """LangChain ``BaseCallbackHandler`` that maps graph events to PandaProbe traces.

    Usage::

        from pandaprobe.integrations.langgraph import LangGraphCallbackHandler

        handler = LangGraphCallbackHandler()
        result = graph.invoke(input, config={"callbacks": [handler]})
    """

    DEFAULT_TRACE_NAME = "LangGraph"
