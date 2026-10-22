"""LangChain: an 18-turn live customer-support queue with Reflect.

Install: uv sync --extra dev --extra langchain --locked
Run: uv run python examples/langchain/04_reflect.py
See examples/reflect/live_support.md for credentials, workload, and reports.
"""

import sys
from pathlib import Path

# Shared example application, deliberately outside the distributed SDK.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))
from _support.runner import AgentBinding, main
from _support.scenario import INSTRUCTIONS


def build(runtime, desk, model, workdir, *, validation=False):
    from langchain.agents import create_agent
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver

    from pandaprobe.integrations.langchain import LangChainCallbackHandler

    agent = runtime.agent(
        create_agent,
        model=ChatOpenAI(model=model, reasoning_effort="none", max_retries=0, timeout=60),
        tools=[StructuredTool.from_function(fn) for fn in desk.functions],
        system_prompt=INSTRUCTIONS,
        checkpointer=InMemorySaver(),
    )
    return AgentBinding(
        agent, {"config": {"callbacks": [LangChainCallbackHandler(client=runtime.client)], "recursion_limit": 32}}
    )


if __name__ == "__main__":
    main("langchain", build, lambda sdk: None, "gpt-6-luna", "OPENAI_API_KEY")
