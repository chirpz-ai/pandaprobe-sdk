"""CrewAI — crew with tool-calling agents and automatic tracing.

Creates a CrewAI crew with an agent that uses custom tools, traced via
CrewAIAdapter.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export GOOGLE_API_KEY="..."

Run:
    uv run python examples/crewai/02_tool_crew.py
"""

import uuid

from crewai import Agent, Crew, LLM, Task
from crewai.tools import tool

import pandaprobe
from pandaprobe.integrations.crewai import CrewAIAdapter

SESSION_ID = str(uuid.uuid4())
USER_ID = "user_1"


@tool("get_weather")
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "london": "Cloudy, 15°C, 70% humidity",
        "tokyo": "Sunny, 28°C, 45% humidity",
        "new york": "Partly cloudy, 22°C, 55% humidity",
        "paris": "Rainy, 12°C, 85% humidity",
    }
    return weather_data.get(city.lower(), f"No weather data for {city}")


@tool("get_population")
def get_population(city: str) -> str:
    """Get the approximate population of a city."""
    populations = {
        "london": "8.8 million",
        "tokyo": "13.9 million",
        "new york": "8.3 million",
        "paris": "2.2 million",
    }
    return populations.get(city.lower(), f"Unknown population for {city}")


def main():
    adapter = CrewAIAdapter(
        session_id=SESSION_ID,
        user_id=USER_ID,
        tags=["tool-agent", "example"],
    )
    adapter.instrument()

    llm = LLM(model="gemini/gemini-3.1-flash-lite-preview")

    city_expert = Agent(
        role="City Information Expert",
        goal="Provide comprehensive city information using available tools",
        backstory="You are an expert at gathering city data using weather and population tools.",
        llm=llm,
        tools=[get_weather, get_population],
    )

    info_task = Task(
        description="Get the weather and population for London and summarize the findings.",
        expected_output="A summary including current weather conditions and population of London.",
        agent=city_expert,
    )

    crew = Crew(
        agents=[city_expert],
        tasks=[info_task],
        verbose=True,
    )

    result = crew.kickoff()
    print(f"\nResult: {result.raw}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"\nTrace sent to PandaProbe backend (session={SESSION_ID}).")


if __name__ == "__main__":
    main()
