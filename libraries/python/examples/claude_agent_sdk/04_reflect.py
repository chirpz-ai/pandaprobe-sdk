"""Claude Agent SDK: an 18-turn live support queue with native MCP tools and Reflect.

Install: uv sync --extra dev --extra claude-agent-sdk --locked
Run: uv run python examples/claude_agent_sdk/04_reflect.py
See examples/reflect/live_support.md for credentials, workload, and reports.
"""

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reflect"))
from _support.runner import AgentBinding, main
from _support.scenario import INSTRUCTIONS


def instrument(sdk):
    from pandaprobe.integrations.claude_agent_sdk import ClaudeAgentSDKAdapter

    adapter = ClaudeAgentSDKAdapter(client=sdk, tags=["reflect-support", "example"])
    adapter.instrument()
    return adapter


def build(runtime, desk, model, workdir, *, validation=False):
    import inspect
    from typing import get_type_hints

    from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool

    def mcp_tool(fn):
        hints = get_type_hints(fn)
        schema = {name: hints[name] for name in inspect.signature(fn).parameters}

        async def call(arguments):
            return {"content": [{"type": "text", "text": fn(**arguments)}]}

        return tool(fn.__name__, fn.__doc__, schema)(call)

    tools = [mcp_tool(fn) for fn in desk.functions]
    server = create_sdk_mcp_server(name="support", tools=tools)
    cwd = workdir / "claude" / uuid4().hex
    cwd.mkdir(parents=True)
    # No host Bash/filesystem tools. Reflect supplies its isolated Bash through MCP.
    # Empty settings prevent local project plugins/instructions changing the exercise.
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=INSTRUCTIONS,
        mcp_servers={"support": server},
        allowed_tools=[f"mcp__support__{fn.__name__}" for fn in desk.functions],
        tools=[],
        setting_sources=[],
        cwd=str(cwd),
        max_turns=16,
        extra_args={"strict-mcp-config": None},
    )
    return AgentBinding(options, history_mode="claude")


if __name__ == "__main__":
    main("claude_agent_sdk", build, instrument, "claude-haiku-5-5", "ANTHROPIC_API_KEY")
