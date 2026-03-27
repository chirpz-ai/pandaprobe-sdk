"""Claude Agent SDK — agent with custom tool calls and automatic tracing.

Creates a Claude Agent SDK client with custom tools and traces the full
tool-calling loop via ClaudeAgentSDKAdapter.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export ANTHROPIC_API_KEY="sk-ant-..."

Run:
    uv run python examples/claude_agent_sdk/02_tool_agent.py
"""

import asyncio
import uuid
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    tool,
)

import pandaprobe
from pandaprobe.integrations.claude_agent_sdk import ClaudeAgentSDKAdapter

SESSION_ID = str(uuid.uuid4())
USER_ID = "user_1"


@tool("get_weather", "Get the current weather for a city", {"city": str})
async def get_weather(args: dict[str, Any]) -> dict[str, Any]:
    """Return mock weather data."""
    weather_data = {
        "london": {"condition": "Cloudy", "temp": "15°C", "humidity": "70%"},
        "tokyo": {"condition": "Sunny", "temp": "28°C", "humidity": "45%"},
        "paris": {"condition": "Rainy", "temp": "12°C", "humidity": "85%"},
    }
    city = args.get("city", "").lower()
    data = weather_data.get(city, {"error": f"No data for {city}"})
    return {"content": [{"type": "text", "text": str(data)}]}


@tool("get_population", "Get the population of a city", {"city": str})
async def get_population(args: dict[str, Any]) -> dict[str, Any]:
    """Return mock population data."""
    populations = {
        "london": "8.8 million",
        "tokyo": "13.9 million",
        "paris": "2.2 million",
    }
    city = args.get("city", "").lower()
    pop = populations.get(city, f"Unknown for {city}")
    return {"content": [{"type": "text", "text": pop}]}


async def main():
    adapter = ClaudeAgentSDKAdapter(
        session_id=SESSION_ID,
        user_id=USER_ID,
        tags=["tool-agent", "example"],
    )
    adapter.instrument()

    server = create_sdk_mcp_server(
        name="city_info",
        tools=[get_weather, get_population],
    )

    options = ClaudeAgentOptions(
        system_prompt="You are a helpful assistant with access to weather and population tools.",
        model="claude-sonnet-4-20250514",
        mcp_servers={"city_info": server},
        allowed_tools=["mcp__city_info__get_weather", "mcp__city_info__get_population"],
        permission_mode="bypassPermissions",
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("What's the weather in London and its population?")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Agent: {block.text}")
            elif isinstance(message, ResultMessage):
                if message.usage:
                    print(f"\nTokens: {message.usage}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"\nTrace sent to PandaProbe backend (session={SESSION_ID}).")


if __name__ == "__main__":
    asyncio.run(main())
