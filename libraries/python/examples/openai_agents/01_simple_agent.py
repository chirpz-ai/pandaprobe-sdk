"""OpenAI Agents SDK — simple agent with automatic tracing.

Creates a basic OpenAI agent and traces the invocation via
OpenAIAgentsAdapter.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/openai_agents/01_simple_agent.py
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
        session_id=SESSION_ID,
        user_id=USER_ID,
        tags=["chatbot", "example"],
    )
    adapter.instrument()

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant. Keep answers concise.",
        model="gpt-5.4-nano",
    )

    result = await Runner.run(agent, "What is the capital of France?")
    print(f"Agent: {result.final_output}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"\nTrace sent to PandaProbe backend (session={SESSION_ID}).")


if __name__ == "__main__":
    asyncio.run(main())
