"""Decorator tracing — multi-turn topic assistant with sessions.

Demonstrates pandaprobe.session() + @pandaprobe.trace working together.
The session ID is dynamically changed between conversations, and decorated
functions automatically inherit the current session via ContextVar.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/decorators/02_multi_turn.py
"""

import uuid

import openai

import pandaprobe

oai_client = openai.OpenAI()


@pandaprobe.trace(name="topic-assistant")
def chat(messages: list) -> dict:
    """Single chat turn traced via decorator — session comes from ContextVar."""
    response = oai_client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=messages,
        reasoning_effort="low",
        max_completion_tokens=150,
    )

    reply = response.choices[0].message.content
    return {"messages": [{"role": "assistant", "content": reply}]}


def run_conversation(topic: str, questions: list[str]) -> None:
    """Run a multi-turn conversation under a unique session ID."""
    session_id = f"{topic}-{uuid.uuid4().hex[:8]}"
    print(f"\n--- Session: {session_id} (topic: {topic}) ---")

    system = f"You are a helpful assistant specializing in {topic}. Keep answers to 1-2 sentences."
    conversation: list[dict] = [{"role": "system", "content": system}]

    with pandaprobe.session(session_id):
        for q in questions:
            print(f"  User: {q}")
            conversation.append({"role": "user", "content": q})
            result = chat(conversation)
            reply = result["messages"][0]["content"]
            conversation.append({"role": "assistant", "content": reply})
            print(f"  Bot:  {reply}")


if __name__ == "__main__":
    run_conversation(
        "python",
        [
            "What's a list comprehension?",
            "Show me a simple example.",
        ],
    )

    run_conversation(
        "javascript",
        [
            "What's a Promise?",
            "How does async/await relate to Promises?",
        ],
    )

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nAll session traces sent to PandaProbe backend.")
