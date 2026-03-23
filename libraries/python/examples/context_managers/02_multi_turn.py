"""Context manager tracing — multi-turn tutoring agent with sessions.

Demonstrates pandaprobe.session() to group multiple traces under one session,
with pandaprobe.start_trace() / trace.span() for each conversation turn.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/context_managers/02_multi_turn.py
"""

import uuid

import openai

import pandaprobe

oai_client = openai.OpenAI()

SYSTEM_PROMPT = "You are a helpful tutor. Keep answers short (2-3 sentences). Build on previous context."


def chat_turn(conversation: list[dict], user_message: str) -> tuple[str, str]:
    """Run one conversation turn: send to OpenAI, trace it."""
    conversation.append({"role": "user", "content": user_message})

    with pandaprobe.start_trace(
        "tutor-agent",
        input={"messages": [{"role": "user", "content": user_message}]},
    ) as trace:
        with trace.span("openai-chat", kind="LLM", model="gpt-5.4-nano") as llm:
            llm.set_input({"messages": conversation})

            response = oai_client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=conversation,
                reasoning_effort="low",
                max_completion_tokens=200,
            )

            assistant_message = response.choices[0].message.content
            llm.set_output({"messages": [{"role": "assistant", "content": assistant_message}]})
            llm.set_token_usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        trace.set_output({"messages": [{"role": "assistant", "content": assistant_message}]})

    conversation.append({"role": "assistant", "content": assistant_message})
    return assistant_message, trace.trace_id


if __name__ == "__main__":
    session_id = f"tutoring-{uuid.uuid4().hex[:8]}"
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    questions = [
        "What is recursion in programming?",
        "Can you give me a simple example in Python?",
        "What's the difference between recursion and iteration?",
    ]

    print(f"Session: {session_id}\n")

    with pandaprobe.session(session_id):
        for question in questions:
            print(f"User:  {question}")
            reply, trace_id = chat_turn(conversation, question)
            print(f"Tutor: {reply}\n")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("All traces sent to PandaProbe backend.")
