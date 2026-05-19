"""Mistral wrapper — multi-turn conversation with sessions.

Demonstrates pandaprobe.session() + wrap_mistral for a multi-turn
conversation. The wrapper automatically creates traces and LLM spans for
each call, and the session context groups them together.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export MISTRAL_API_KEY="..."

Run:
    uv run python examples/mistral/03_multi_turn.py
"""

import os
import uuid

from mistralai.client import Mistral

import pandaprobe
from pandaprobe.wrappers import wrap_mistral

client = wrap_mistral(Mistral(api_key=os.environ["MISTRAL_API_KEY"]))

if __name__ == "__main__":
    session_id = str(uuid.uuid4())
    print(f"Session: {session_id}\n")

    conversation: list[dict] = [
        {"role": "system", "content": "You are a concise geography assistant."},
    ]

    questions = [
        "What is the capital of France?",
        "What about Germany?",
        "Which of those two cities has more people?",
    ]

    with pandaprobe.session(session_id):
        for question in questions:
            print(f"User: {question}")
            conversation.append({"role": "user", "content": question})

            response = client.chat.complete(
                model="mistral-small-latest",
                messages=conversation,
                temperature=0.3,
                max_tokens=200,
            )

            reply = response.choices[0].message.content or ""
            conversation.append({"role": "assistant", "content": reply})
            print(f"Bot:  {reply}\n")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"All traces grouped under session_id={session_id}")
