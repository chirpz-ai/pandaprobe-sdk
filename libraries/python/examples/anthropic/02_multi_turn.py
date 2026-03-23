"""Anthropic wrapper — multi-turn conversation with sessions.

Demonstrates pandaprobe.session() + wrap_anthropic for a multi-turn conversation
with extended thinking enabled. The wrapper automatically creates traces and LLM
spans for each call, and the session context groups them together.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export ANTHROPIC_API_KEY="sk-ant-..."

Run:
    uv run python examples/anthropic/02_multi_turn.py
"""

import uuid

import anthropic

import pandaprobe
from pandaprobe.wrappers import wrap_anthropic

client = wrap_anthropic(anthropic.Anthropic())

if __name__ == "__main__":
    session_id = f"geography-{uuid.uuid4().hex[:8]}"
    print(f"Session: {session_id}\n")

    conversation: list[dict] = []

    questions = [
        "What is the capital of France?",
        "What about Germany?",
        "Which of those two cities has more people?",
    ]

    with pandaprobe.session(session_id):
        for question in questions:
            print(f"User: {question}")
            conversation.append({"role": "user", "content": question})

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                system="You are a concise geography assistant.",
                messages=conversation,
            )

            reply_text = ""
            for block in response.content:
                if block.type == "text":
                    reply_text += block.text

            conversation.append({"role": "assistant", "content": reply_text})
            print(f"Bot:  {reply_text}\n")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"All traces grouped under session_id={session_id}")
