"""LangChain: an 18-turn live customer-support queue with Reflect.

Install: uv sync --extra dev --extra langchain --locked
Run: uv run python examples/langchain/04_reflect.py
See examples/reflect/live_support.md for credentials, workload, and reports.
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))

from _support.agents import build_langchain
from _support.runner import main
from _support.scenario import INSTRUCTIONS

build = partial(build_langchain, instructions=INSTRUCTIONS)

if __name__ == "__main__":
    main("langchain", build, lambda sdk: None, "gpt-6-luna", "OPENAI_API_KEY")
