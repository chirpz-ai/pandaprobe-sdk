"""CrewAI: an 18-turn live support queue with a native Agent and Reflect.

Install: uv sync --extra dev --extra crewai --locked
Run: uv run python examples/crewai/04_reflect.py
See examples/reflect/live_support.md for credentials, workload, and reports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))
from _support.runner import AgentBinding, main
from _support.scenario import INSTRUCTIONS


def instrument(sdk):
    from pandaprobe.integrations.crewai import CrewAIAdapter

    adapter = CrewAIAdapter(client=sdk, tags=["reflect-support", "example"])
    adapter.instrument()
    return adapter


def build(runtime, desk, model, workdir, *, validation=False):
    from crewai import Agent, LLM
    from crewai.tools import tool

    # GPT-6 Luna requires "none" for Chat Completions tool calls. Use extra_body
    # because the locked CrewAI version forwards reasoning_effort only for o1.
    model_options = (
        {"extra_body": {"reasoning_effort": "none"}} if model in ("openai/gpt-6-luna", "gpt-6-luna") else {}
    )
    agent = runtime.agent(
        Agent,
        role="Northstar Support",
        goal="Resolve the customer's authorized support ticket correctly.",
        backstory=INSTRUCTIONS,
        llm=LLM(model=model, **model_options),
        tools=[tool(fn.__name__)(fn) for fn in desk.functions],
        max_iter=16,
        allow_delegation=False,
        allow_code_execution=False,
        cache=False,
        verbose=False,
    )
    return AgentBinding(agent, history_mode="messages")


if __name__ == "__main__":
    main("crewai", build, instrument, "openai/gpt-6-luna", "OPENAI_API_KEY")
