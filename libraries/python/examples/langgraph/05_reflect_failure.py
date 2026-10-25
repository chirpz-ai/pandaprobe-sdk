"""langgraph: a live controlled refund failure with Reflect.

Install: uv sync --extra dev --extra langgraph --locked
Run: uv run python examples/langgraph/05_reflect_failure.py
See examples/reflect/live_failure.md for bounds, independent checks, and reports.
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))

from _support.agents import build_langgraph
from _support.failure_runner import main
from _support.refund_incident import INSTRUCTIONS

build = partial(build_langgraph, instructions=INSTRUCTIONS)

if __name__ == "__main__":
    main("langgraph", build, lambda sdk: None, "gpt-6-luna", "OPENAI_API_KEY")
