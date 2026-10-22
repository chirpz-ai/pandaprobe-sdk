"""Deep Agents: an 18-turn live support queue with planning, memory, and Reflect.

Requires Python 3.11+. Install: uv sync --extra dev --extra deepagents --locked
Run: uv run python examples/deepagents/04_reflect.py
See examples/reflect/live_support.md for credentials, workload, and reports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))
from _support.runner import AgentBinding, main
from _support.scenario import INSTRUCTIONS


def build(runtime, desk, model, workdir, *, validation=False):
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver

    from pandaprobe.integrations.deepagents import DeepAgentsCallbackHandler

    agent = runtime.agent(
        create_deep_agent,
        model=ChatOpenAI(model=model, reasoning_effort="none", max_retries=0, timeout=60),
        tools=[StructuredTool.from_function(fn) for fn in desk.functions],
        system_prompt=INSTRUCTIONS
        + "\nUse reflect_bash for /rules and /context.md; other files are task scratch space.",
        checkpointer=InMemorySaver(),
        backend=StateBackend(),
    )
    return AgentBinding(
        agent, {"config": {"callbacks": [DeepAgentsCallbackHandler(client=runtime.client)], "recursion_limit": 40}}
    )


if __name__ == "__main__":
    main("deepagents", build, lambda sdk: None, "gpt-6-luna", "OPENAI_API_KEY")
