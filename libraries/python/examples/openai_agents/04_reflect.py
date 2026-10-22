"""OpenAI Agents: an 18-turn live support queue with native sessions and Reflect.

Install: uv sync --extra dev --extra openai-agents --locked
Run: uv run python examples/openai_agents/04_reflect.py
See examples/reflect/live_support.md for credentials, workload, and reports.
"""

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))
from _support.runner import AgentBinding, main
from _support.scenario import INSTRUCTIONS


def instrument(sdk):
    from pandaprobe.integrations.openai_agents import OpenAIAgentsAdapter

    adapter = OpenAIAgentsAdapter(client=sdk, tags=["reflect-support", "example"])
    adapter.instrument()
    return adapter


def build(runtime, desk, model, workdir, *, validation=False):
    from agents import Agent, SQLiteSession, function_tool

    agent = runtime.agent(
        Agent,
        name="Northstar Support",
        model=model,
        instructions=INSTRUCTIONS,
        tools=[function_tool(fn) for fn in desk.functions],
    )
    kwargs = {"max_turns": 16}
    if not validation:
        kwargs["session"] = SQLiteSession(uuid4().hex)  # Includes the reflection continuation.
    return AgentBinding(agent, kwargs)


if __name__ == "__main__":
    main("openai_agents", build, instrument, "gpt-6-luna", "OPENAI_API_KEY")
