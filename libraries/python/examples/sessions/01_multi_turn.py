"""Session-based tracing — multi-turn conversation with scoring.

Demonstrates client.session() to group multiple traces under one session,
with real OpenAI calls for each turn and client.score() after each response.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="sessions-example"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/sessions/01_multi_turn.py
"""

import time
import uuid

import openai

import pandaprobe

pp_client = pandaprobe.Client(debug=True)
oai_client = openai.OpenAI()

SYSTEM_PROMPT = "You are a helpful tutor. Keep answers short (2-3 sentences). Build on previous context."


def chat_turn(session, turn_name: str, conversation: list[dict], user_message: str) -> tuple[str, str]:
    """Run one conversation turn: send to OpenAI, trace it, score it."""
    conversation.append({"role": "user", "content": user_message})

    with session.trace(turn_name, input={"user_message": user_message}) as trace:
        with trace.span("openai-chat", kind="LLM", model="gpt-4o-mini") as llm:
            llm.set_input({"messages": conversation})

            response = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,
                temperature=0.5,
                max_tokens=200,
            )

            assistant_message = response.choices[0].message.content
            llm.set_output(assistant_message)
            llm.set_token_usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        trace.set_output({"assistant_message": assistant_message})

    conversation.append({"role": "assistant", "content": assistant_message})
    return assistant_message, trace.trace_id


if __name__ == "__main__":
    session_id = f"tutoring-{uuid.uuid4().hex[:8]}"
    session = pp_client.session(session_id)
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    turns = [
        ("turn-1", "What is recursion in programming?"),
        ("turn-2", "Can you give me a simple example in Python?"),
        ("turn-3", "What's the difference between recursion and iteration?"),
    ]

    print(f"Session: {session_id}\n")

    trace_ids = []
    for turn_name, user_msg in turns:
        print(f"User: {user_msg}")
        reply, trace_id = chat_turn(session, turn_name, conversation, user_msg)
        trace_ids.append(trace_id)
        print(f"Tutor: {reply}\n")

    pp_client.flush()
    time.sleep(2)

    for trace_id in trace_ids:
        pp_client.score(
            trace_id=trace_id,
            name="helpfulness",
            value="true",
            data_type="BOOLEAN",
            reason="Response is on-topic and builds on the conversation context.",
        )
    pp_client.flush()
    pp_client.shutdown()
    print("All traces + scores sent to PandaProbe backend.")
