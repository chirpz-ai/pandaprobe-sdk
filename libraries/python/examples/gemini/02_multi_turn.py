"""Gemini wrapper — multi-turn conversation with sessions.

Demonstrates pandaprobe.session() + wrap_gemini for a multi-turn conversation.
The wrapper automatically creates traces and LLM spans for each call, and the
session context groups them together.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export GOOGLE_API_KEY="..."

Run:
    uv run python examples/gemini/02_multi_turn.py
"""

import uuid

from google import genai
from google.genai import types

import pandaprobe
from pandaprobe.wrappers import wrap_gemini

client = wrap_gemini(genai.Client())

if __name__ == "__main__":
    session_id = str(uuid.uuid4())
    print(f"Session: {session_id}\n")

    conversation = []

    questions = [
        "What is the capital of France?",
        "What about Germany?",
        "Which of those two cities has more people?",
    ]

    with pandaprobe.session(session_id):
        for question in questions:
            print(f"User: {question}")
            conversation.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=conversation,
                config=types.GenerateContentConfig(
                    system_instruction="You are a concise geography assistant.",
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                    temperature=1,
                    max_output_tokens=80,
                ),
            )

            reply = response.text
            conversation.append(types.Content(role="model", parts=[types.Part.from_text(text=reply)]))
            print(f"Bot:  {reply}\n")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"All traces grouped under session_id={session_id}")
