"""Claude Agent SDK — multi-turn conversation with session grouping.

Runs multiple conversation turns with a Claude Agent SDK client under a
single session_id using pandaprobe.session().  Each query/receive_response
cycle produces a separate trace, and all traces share the same session.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export ANTHROPIC_API_KEY="sk-ant-..."

Run:
    uv run python examples/claude_agent_sdk/03_multi_turn.py
"""

import asyncio
import uuid

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
)

import pandaprobe
from pandaprobe.integrations.claude_agent_sdk import ClaudeAgentSDKAdapter

SESSION_ID = str(uuid.uuid4())
USER_ID = "user_1"


async def chat_turn(client: ClaudeSDKClient, message: str) -> str:
    """Send one user message and return Claude's text reply."""
    await client.query(message)
    response_text = ""
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
    return response_text


async def main():
    adapter = ClaudeAgentSDKAdapter(
        user_id=USER_ID,
        tags=["multi-turn", "example"],
    )
    adapter.instrument()

    options = ClaudeAgentOptions(
        system_prompt="You are a helpful travel advisor. Keep answers to 2-3 sentences.",
        model="claude-haiku-4-5-20251001",
        permission_mode="bypassPermissions",
        thinking={"type": "enabled", "budget_tokens": 5000},
    )

    questions = [
        "I'm planning a trip to Japan. What's the best time to visit?",
        "What cities should I prioritize for a 10-day trip?",
        "Any tips for getting around between cities?",
    ]

    print(f"Session: {SESSION_ID}\n")

    async with ClaudeSDKClient(options=options) as client:
        with pandaprobe.session(SESSION_ID):
            for question in questions:
                print(f"User:    {question}")
                reply = await chat_turn(client, question)
                print(f"Advisor: {reply}\n")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"All {len(questions)} turns traced under session_id={SESSION_ID}")


if __name__ == "__main__":
    asyncio.run(main())
