"""Google ADK: an 18-turn live support queue with a persistent Runner and Reflect.

Install: uv sync --extra dev --extra google-adk --locked
Run: uv run python examples/google_adk/04_reflect.py
See examples/reflect/live_support.md for credentials, workload, and reports.
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))

from _support.agents import build_google_adk, instrument_google_adk
from _support.runner import main
from _support.scenario import INSTRUCTIONS

build = partial(build_google_adk, instructions=INSTRUCTIONS)

instrument = partial(instrument_google_adk, tag="reflect-support")

if __name__ == "__main__":
    main("google_adk", build, instrument, "gemini-3.8-flash", "GOOGLE_API_KEY")
