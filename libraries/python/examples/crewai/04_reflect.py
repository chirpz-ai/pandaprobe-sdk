"""CrewAI: an 18-turn live support queue with a native Agent and Reflect.

Install: uv sync --extra dev --extra crewai --locked
Run: uv run python examples/crewai/04_reflect.py
See examples/reflect/live_support.md for credentials, workload, and reports.
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))

from _support.agents import build_crewai, instrument_crewai
from _support.runner import main
from _support.scenario import INSTRUCTIONS

build = partial(build_crewai, instructions=INSTRUCTIONS)

instrument = partial(instrument_crewai, tag="reflect-support")

if __name__ == "__main__":
    main("crewai", build, instrument, "openai/gpt-6-luna", "OPENAI_API_KEY")
