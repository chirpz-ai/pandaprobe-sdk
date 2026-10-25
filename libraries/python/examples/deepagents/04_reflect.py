"""Deep Agents: an 18-turn live support queue with planning, memory, and Reflect.

Requires Python 3.11+. Install: uv sync --extra dev --extra deepagents --locked
Run: uv run python examples/deepagents/04_reflect.py
See examples/reflect/live_support.md for credentials, workload, and reports.
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))

from _support.agents import build_deepagents
from _support.runner import main
from _support.scenario import INSTRUCTIONS

build = partial(build_deepagents, instructions=INSTRUCTIONS)

if __name__ == "__main__":
    main("deepagents", build, lambda sdk: None, "gpt-6-luna", "OPENAI_API_KEY")
