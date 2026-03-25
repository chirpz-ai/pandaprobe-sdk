"""Google ADK — agent with tool calls and automatic tracing.

Creates an ADK agent with weather and population tools, and traces the
full tool-calling loop via GoogleADKAdapter.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export GOOGLE_API_KEY="..."

Run:
    uv run python examples/google_adk/02_tool_agent.py
"""

import asyncio
import uuid

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

import pandaprobe
from pandaprobe.integrations.google_adk import GoogleADKAdapter


def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    weather_data = {
        "london": {"condition": "Cloudy", "temp": "15°C", "humidity": "70%"},
        "tokyo": {"condition": "Sunny", "temp": "28°C", "humidity": "45%"},
        "new york": {"condition": "Partly cloudy", "temp": "22°C", "humidity": "55%"},
        "paris": {"condition": "Rainy", "temp": "12°C", "humidity": "85%"},
    }
    return weather_data.get(city.lower(), {"error": f"No data for {city}"})


def get_population(city: str) -> dict:
    """Get the approximate population of a city."""
    populations = {
        "london": {"population": "8.8 million"},
        "tokyo": {"population": "13.9 million"},
        "new york": {"population": "8.3 million"},
        "paris": {"population": "2.2 million"},
    }
    return populations.get(city.lower(), {"error": f"No data for {city}"})


agent = LlmAgent(
    name="city_info_agent",
    model="gemini-3.1-flash-lite-preview",
    instruction=(
        "You are a helpful assistant with access to weather and population tools. "
        "Use the tools to answer questions about cities."
    ),
    tools=[get_weather, get_population],
)

APP_NAME = "tool_agent"
USER_ID = "user_1"
SESSION_ID = str(uuid.uuid4())


async def main():
    adapter = GoogleADKAdapter(
        session_id=SESSION_ID,
        user_id=USER_ID,
        tags=["tool-agent", "example"],
    )
    adapter.instrument()

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    runner = Runner(
        agent=agent, app_name=APP_NAME, session_service=session_service
    )

    user_message = Content(
        role="user",
        parts=[Part(text="What's the weather like in London and what's its population?")],
    )

    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=user_message
    ):
        if event.is_final_response():
            text = " ".join(p.text for p in event.content.parts if p.text)
            print(f"Agent: {text}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")


if __name__ == "__main__":
    asyncio.run(main())
