"""CrewAI — simple crew with automatic tracing.

Creates a basic CrewAI crew with a single agent and task, traced via
CrewAIAdapter.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export GOOGLE_API_KEY="..."

Run:
    uv run python examples/crewai/01_simple_crew.py
"""

import uuid

from crewai import Agent, Crew, LLM, Task

import pandaprobe
from pandaprobe.integrations.crewai import CrewAIAdapter

SESSION_ID = str(uuid.uuid4())
USER_ID = "user_1"


def main():
    adapter = CrewAIAdapter(
        session_id=SESSION_ID,
        user_id=USER_ID,
        tags=["chatbot", "example"],
    )
    adapter.instrument()

    llm = LLM(model="gemini/gemini-3.1-flash-lite-preview")

    researcher = Agent(
        role="Researcher",
        goal="Find accurate information about the given topic",
        backstory="You are an expert researcher with years of experience.",
        llm=llm,
    )

    research_task = Task(
        description="Research the history of artificial intelligence and summarize the key milestones.",
        expected_output="A concise summary of AI history milestones in 3-5 bullet points.",
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True,
    )

    result = crew.kickoff()
    print(f"\nResult: {result.raw}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"\nTrace sent to PandaProbe backend (session={SESSION_ID}).")


if __name__ == "__main__":
    main()
