"""Google ADK — simple agent with automatic tracing.

Creates a basic ADK agent and traces the invocation via GoogleADKAdapter.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export GOOGLE_API_KEY="..."

Run:
    uv run python examples/google_adk/01_simple_agent.py
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
    name="chatbot",
    model="gemini-3.1-flash-lite-preview",
    instruction="You are a helpful assistant. Keep answers concise.",
)

APP_NAME = "simple_chatbot"
USER_ID = "user_1"
SESSION_ID = str(uuid.uuid4())


async def main():
    adapter = GoogleADKAdapter(
        session_id=SESSION_ID,
        user_id=USER_ID,
        tags=["chatbot", "example"],
    )
    adapter.instrument()

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    user_message = Content(
        role="user",
        parts=[Part(text="What is the capital of France?")],
    )

    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=user_message):
        if event.is_final_response():
            text = " ".join(p.text for p in event.content.parts if p.text)
            print(f"Bot: {text}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")


if __name__ == "__main__":
    asyncio.run(main())
