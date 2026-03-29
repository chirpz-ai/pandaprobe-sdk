"""OpenAI Agents SDK — agent with function tools and automatic tracing.

Creates an OpenAI agent with weather and population tools, and traces the
full tool-calling loop via OpenAIAgentsAdapter.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/openai_agents/02_tool_agent.py
"""

import asyncio
import uuid

from agents import Agent, Runner, function_tool

import pandaprobe
from pandaprobe.integrations.openai_agents import OpenAIAgentsAdapter

SESSION_ID = str(uuid.uuid4())
USER_ID = "user_1"


@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "london": "Cloudy, 15°C, 70% humidity",
        "tokyo": "Sunny, 28°C, 45% humidity",
        "new york": "Partly cloudy, 22°C, 55% humidity",
        "paris": "Rainy, 12°C, 85% humidity",
    }
    return weather_data.get(city.lower(), f"No weather data for {city}")


@function_tool
def get_population(city: str) -> str:
    """Get the approximate population of a city."""
    populations = {
        "london": "8.8 million",
        "tokyo": "13.9 million",
        "new york": "8.3 million",
        "paris": "2.2 million",
    }
    return populations.get(city.lower(), f"Unknown population for {city}")


async def main():
    adapter = OpenAIAgentsAdapter(
        session_id=SESSION_ID,
        user_id=USER_ID,
        tags=["tool-agent", "example"],
    )
    adapter.instrument()

    agent = Agent(
        name="City Info Agent",
        instructions=(
            "You are a helpful assistant with access to weather and population tools. "
            "Use the tools to answer questions about cities."
        ),
        model="gpt-5.4-nano",
        tools=[get_weather, get_population],
    )

    result = await Runner.run(agent, "What's the weather in London and what's its population?")
    print(f"Agent: {result.final_output}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"\nTrace sent to PandaProbe backend (session={SESSION_ID}).")


if __name__ == "__main__":
    asyncio.run(main())
