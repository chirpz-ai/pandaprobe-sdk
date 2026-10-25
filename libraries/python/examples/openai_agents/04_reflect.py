"""OpenAI Agents: an 18-turn live support queue with native sessions and Reflect.

Install: uv sync --extra dev --extra openai-agents --locked
Run: uv run python examples/openai_agents/04_reflect.py
See examples/reflect/live_support.md for credentials, workload, and reports.
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))

from _support.agents import build_openai_agents, instrument_openai_agents
from _support.runner import main
from _support.scenario import INSTRUCTIONS

build = partial(build_openai_agents, instructions=INSTRUCTIONS)

instrument = partial(instrument_openai_agents, tag="reflect-support")

if __name__ == "__main__":
    main("openai_agents", build, instrument, "gpt-6-luna", "OPENAI_API_KEY")
