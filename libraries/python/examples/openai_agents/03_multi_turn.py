"""OpenAI Agents SDK — multi-turn conversation with session grouping.

Runs multiple conversation turns with an OpenAI agent under a single
session_id using pandaprobe.session().  Each Runner.run() call produces
a separate trace, and all traces share the same session.

Conversation history is accumulated via ``result.to_input_list()`` so
each turn sees the full prior context — matching the pattern used by
Google ADK sessions and Claude Agent SDK history.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/openai_agents/03_multi_turn.py
"""

import asyncio
import uuid

from agents import Agent, Runner

import pandaprobe
from pandaprobe.integrations.openai_agents import OpenAIAgentsAdapter

SESSION_ID = str(uuid.uuid4())
USER_ID = "user_1"


async def main():
    adapter = OpenAIAgentsAdapter(
        user_id=USER_ID,
        tags=["multi-turn", "example"],
    )
    adapter.instrument()

    agent = Agent(
        name="Travel Advisor",
        instructions="You are a helpful travel advisor. Keep answers to 2-3 sentences.",
        model="gpt-5.4-nano",
    )

    questions = [
        "I'm planning a trip to Japan. What's the best time to visit?",
        "What cities should I prioritize for a 10-day trip?",
        "Any tips for getting around between cities?",
    ]

    print(f"Session: {SESSION_ID}\n")

    with pandaprobe.session(SESSION_ID):
        input_items: list = []
        for question in questions:
            print(f"User:    {question}")
            input_items.append({"role": "user", "content": question})
            result = await Runner.run(agent, input=input_items)
            print(f"Advisor: {result.final_output}\n")
            input_items = result.to_input_list()

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"All {len(questions)} turns traced under session_id={SESSION_ID}")


if __name__ == "__main__":
    asyncio.run(main())
