"""Google ADK — multi-turn conversation with session grouping.

Runs multiple conversation turns with an ADK agent under a single
session_id using pandaprobe.session().  Each runner.run_async() call
produces a separate trace, and all traces share the same session.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export GOOGLE_API_KEY="..."

Run:
    uv run python examples/google_adk/03_multi_turn.py
"""

import asyncio
import uuid

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

import pandaprobe
from pandaprobe.integrations.google_adk import GoogleADKAdapter

agent = LlmAgent(
    name="travel_advisor",
    model="gemini-3.1-flash-lite-preview",
    instruction=(
        "You are a helpful travel advisor. Keep answers to 2-3 sentences."
    ),
)

APP_NAME = "travel_advisor"
USER_ID = "user_1"
SESSION_ID = str(uuid.uuid4())


async def chat_turn(runner: Runner, message: str) -> str:
    """Send one user message and return the agent's text reply."""
    user_message = Content(role="user", parts=[Part(text=message)])
    response_text = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=user_message
    ):
        if event.is_final_response():
            response_text = " ".join(
                p.text for p in event.content.parts if p.text
            )
    return response_text


async def main():
    adapter = GoogleADKAdapter(user_id=USER_ID, tags=["multi-turn", "example"])
    adapter.instrument()

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    runner = Runner(
        agent=agent, app_name=APP_NAME, session_service=session_service
    )

    questions = [
        "I'm planning a trip to Japan. What's the best time to visit?",
        "What cities should I prioritize for a 10-day trip?",
        "Any tips for getting around between cities?",
    ]

    print(f"Session: {SESSION_ID}\n")

    with pandaprobe.session(SESSION_ID):
        for question in questions:
            print(f"User:    {question}")
            reply = await chat_turn(runner, question)
            print(f"Advisor: {reply}\n")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"All {len(questions)} turns traced under session_id={SESSION_ID}")


if __name__ == "__main__":
    asyncio.run(main())
