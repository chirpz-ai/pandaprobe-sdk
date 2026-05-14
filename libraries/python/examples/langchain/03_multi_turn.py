"""LangChain — multi-turn agent with sessions.

Builds a LangChain agent via ``create_agent`` and runs multiple conversation
turns under a single session_id using ``pandaprobe.session()``. The callback
handler automatically picks up the session from the ContextVar.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/langchain/03_multi_turn.py
"""

import uuid

from langchain.agents import create_agent
from langchain.tools import tool

import pandaprobe
from pandaprobe.integrations.langchain import LangChainCallbackHandler


@tool
def get_attraction(city: str) -> str:
    """Return a popular tourist attraction for a given city."""
    table = {
        "tokyo": "Shibuya Crossing",
        "kyoto": "Fushimi Inari Shrine",
        "osaka": "Dotonbori",
    }
    return table.get(city.lower(), f"No data for {city}.")


agent = create_agent(
    model="openai:gpt-5.4-nano",
    tools=[get_attraction],
    system_prompt=(
        "You are a helpful travel advisor. Keep answers to 2-3 sentences. "
        "Use the get_attraction tool when the user asks about specific cities."
    ),
)


def chat_turn(user_message: str, history: list, handler: LangChainCallbackHandler) -> str:
    """Invoke the agent for one user turn, appending to history."""
    history.append({"role": "user", "content": user_message})

    result = agent.invoke(
        {"messages": list(history)},
        config={"callbacks": [handler]},
    )

    reply = result["messages"][-1].content
    history.append({"role": "assistant", "content": reply})
    return reply


if __name__ == "__main__":
    session_id = str(uuid.uuid4())
    print(f"Session: {session_id}\n")

    history: list = []

    questions = [
        "I'm planning a trip to Japan. What cities should I prioritize for a 10-day trip?",
        "What's a must-see attraction in Tokyo?",
        "And in Kyoto?",
    ]

    with pandaprobe.session(session_id):
        for question in questions:
            handler = LangChainCallbackHandler()
            print(f"User:    {question}")
            reply = chat_turn(question, history, handler)
            print(f"Advisor: {reply}\n")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"All {len(questions)} turns traced under session_id={session_id}")
