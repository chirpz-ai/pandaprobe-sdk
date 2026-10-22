"""LangGraph: an 18-turn live support queue with checkpointed conversations and Reflect.

Install: uv sync --extra dev --extra langgraph --locked
Run: uv run python examples/langgraph/04_reflect.py
See examples/reflect/live_support.md for credentials, workload, and reports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))
from _support.runner import AgentBinding, main
from _support.scenario import INSTRUCTIONS


def build(runtime, desk, model, workdir, *, validation=False):
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import create_react_agent

    from pandaprobe.integrations.langgraph import LangGraphCallbackHandler

    # Native LangGraph factory; LangChain's create_agent is an alternative if installed.
    agent = runtime.agent(
        create_react_agent,
        model=ChatOpenAI(model=model, reasoning_effort="none", max_retries=0, timeout=60),
        tools=[StructuredTool.from_function(fn) for fn in desk.functions],
        prompt=INSTRUCTIONS,
        checkpointer=InMemorySaver(),
    )
    return AgentBinding(
        agent, {"config": {"callbacks": [LangGraphCallbackHandler(client=runtime.client)], "recursion_limit": 32}}
    )


if __name__ == "__main__":
    main("langgraph", build, lambda sdk: None, "gpt-6-luna", "OPENAI_API_KEY")
