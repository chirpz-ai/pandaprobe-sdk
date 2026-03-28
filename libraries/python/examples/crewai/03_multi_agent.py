"""CrewAI — multi-agent crew with sequential task execution.

Creates a CrewAI crew with multiple agents executing tasks sequentially,
traced via CrewAIAdapter with session grouping.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export GOOGLE_API_KEY="..."

Run:
    uv run python examples/crewai/03_multi_agent.py
"""

import uuid

from crewai import Agent, Crew, LLM, Task

import pandaprobe
from pandaprobe.integrations.crewai import CrewAIAdapter

SESSION_ID = str(uuid.uuid4())
USER_ID = "user_1"


def main():
    adapter = CrewAIAdapter(
        user_id=USER_ID,
        tags=["multi-agent", "example"],
    )
    adapter.instrument()

    llm = LLM(model="gemini/gemini-3.1-flash-lite-preview", reasoning_effort="low")

    researcher = Agent(
        role="Senior Researcher",
        goal="Research and identify the most important trends in the given topic",
        backstory="You are an expert researcher who excels at finding key insights.",
        llm=llm,
    )

    analyst = Agent(
        role="Data Analyst",
        goal="Analyze research findings and identify actionable insights",
        backstory="You are a skilled analyst who turns raw research into clear conclusions.",
        llm=llm,
    )

    writer = Agent(
        role="Content Writer",
        goal="Write a compelling article based on research and analysis",
        backstory="You are a talented writer who creates engaging content from technical material.",
        llm=llm,
    )

    research_task = Task(
        description="Research the latest trends in renewable energy for 2026.",
        expected_output="A list of 5 key trends with brief descriptions.",
        agent=researcher,
    )

    analysis_task = Task(
        description="Analyze the research findings and identify the top 3 most impactful trends.",
        expected_output="A ranked list of the top 3 trends with impact analysis.",
        agent=analyst,
    )

    writing_task = Task(
        description="Write a short article summarizing the top renewable energy trends for 2026.",
        expected_output="A 200-word article suitable for a tech blog.",
        agent=writer,
    )

    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        verbose=True,
    )

    print(f"Session: {SESSION_ID}\n")

    with pandaprobe.session(SESSION_ID):
        result = crew.kickoff()
        print(f"\nResult:\n{result.raw}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"\nTrace sent to PandaProbe backend (session={SESSION_ID}).")


if __name__ == "__main__":
    main()
