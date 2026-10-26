"""Claude Agent SDK: an 18-turn live support queue with native MCP tools and Reflect.

Install: uv sync --extra dev --extra claude-agent-sdk --locked
Run: uv run python examples/claude_agent_sdk/04_reflect.py
See examples/reflect/live_support.md for credentials, workload, and reports.
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))

from _support.agents import build_claude_agent_sdk, instrument_claude_agent_sdk
from _support.runner import main
from _support.scenario import INSTRUCTIONS

build = partial(build_claude_agent_sdk, instructions=INSTRUCTIONS)

instrument = partial(instrument_claude_agent_sdk, tag="reflect-support")

if __name__ == "__main__":
    main("claude_agent_sdk", build, instrument, "claude-haiku-5-5", "ANTHROPIC_API_KEY")
