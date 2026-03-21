"""OpenAI wrapper — multi-turn conversation with sessions.

Demonstrates pandaprobe.session() + wrap_openai for a multi-turn conversation.
The wrapper automatically creates traces and LLM spans for each call, and the
session context groups them together.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/openai/03_multi_turn.py
"""

import uuid

import openai

import pandaprobe
from pandaprobe.wrappers import wrap_openai

client = wrap_openai(openai.OpenAI())

if __name__ == "__main__":
    session_id = f"geography-{uuid.uuid4().hex[:8]}"
    print(f"Session: {session_id}\n")

    conversation = [{"role": "system", "content": "You are a concise geography assistant."}]

    questions = [
        "What is the capital of France?",
        "What about Germany?",
        "Which of those two cities has more people?",
    ]

    with pandaprobe.session(session_id):
        for question in questions:
            print(f"User: {question}")
            conversation.append({"role": "user", "content": question})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,
                max_tokens=80,
            )

            reply = response.choices[0].message.content
            conversation.append({"role": "assistant", "content": reply})
            print(f"Bot:  {reply}\n")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"All traces grouped under session_id={session_id}")
