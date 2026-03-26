"""Claude Agent SDK — simple agent with automatic tracing.

Creates a basic Claude Agent SDK client and traces the interaction via
ClaudeAgentSDKAdapter.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export ANTHROPIC_API_KEY="sk-ant-..."

Run:
    uv run python examples/claude_agent_sdk/01_simple_agent.py
"""

import asyncio
import uuid

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, TextBlock

import pandaprobe
from pandaprobe.integrations.claude_agent_sdk import ClaudeAgentSDKAdapter

SESSION_ID = str(uuid.uuid4())
USER_ID = "user_1"


async def main():
    adapter = ClaudeAgentSDKAdapter(
        session_id=SESSION_ID,
        user_id=USER_ID,
        tags=["chatbot", "example"],
    )
    adapter.instrument()

    options = ClaudeAgentOptions(
        system_prompt="You are a helpful assistant. Keep answers concise.",
        model="claude-sonnet-4-20250514",
        permission_mode="bypassPermissions",
        max_turns=1,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("What is the capital of France?")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"\nTrace sent to PandaProbe backend (session={SESSION_ID}).")


if __name__ == "__main__":
    asyncio.run(main())
