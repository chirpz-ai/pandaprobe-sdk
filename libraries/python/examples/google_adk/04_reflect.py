"""Google ADK: an 18-turn live support queue with a persistent Runner and Reflect.

Install: uv sync --extra dev --extra google-adk --locked
Run: uv run python examples/google_adk/04_reflect.py
See examples/reflect/live_support.md for credentials, workload, and reports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))
from _support.runner import AgentBinding, main
from _support.scenario import INSTRUCTIONS


def instrument(sdk):
    from pandaprobe.integrations.google_adk import GoogleADKAdapter

    adapter = GoogleADKAdapter(client=sdk, tags=["reflect-support", "example"])
    adapter.instrument()
    return adapter


def build(runtime, desk, model, workdir, *, validation=False):
    from google.adk.agents import LlmAgent
    from google.adk.agents.run_config import RunConfig
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    agent = runtime.agent(
        LlmAgent, name="northstar_support", model=model, instruction=INSTRUCTIONS, tools=desk.functions
    )
    runner = Runner(agent=agent, app_name="northstar_support", session_service=InMemorySessionService())
    return AgentBinding(runner, {"run_config": RunConfig(max_llm_calls=16)})


if __name__ == "__main__":
    main("google_adk", build, instrument, "gemini-3.8-flash", "GOOGLE_API_KEY")
