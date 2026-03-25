"""LangGraph — multi-turn travel advisor with sessions.

Builds a LangGraph chatbot and runs multiple conversation turns under a
single session_id using pandaprobe.session(). The callback handler
automatically picks up the session from the ContextVar.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/langgraph/03_multi_turn.py
"""

import uuid
from typing import Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import pandaprobe
from pandaprobe.integrations.langgraph import LangGraphCallbackHandler

llm = ChatOpenAI(model="gpt-5.4-nano", model_kwargs={"reasoning_effort": "low"})


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


graph = StateGraph(State)
graph.add_node("chatbot", chatbot_node)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
app = graph.compile()


def chat_turn(user_message: str, history: list, handler: LangGraphCallbackHandler) -> str:
    """Invoke the graph for one user turn, appending to history."""
    history.append(("user", user_message))

    result = app.invoke(
        {"messages": list(history)},
        config={"callbacks": [handler]},
    )

    reply = result["messages"][-1].content
    history.append(("assistant", reply))
    return reply


if __name__ == "__main__":
    session_id = str(uuid.uuid4())
    print(f"Session: {session_id}\n")

    history: list = [("system", "You are a helpful travel advisor. Keep answers to 2-3 sentences.")]

    questions = [
        "I'm planning a trip to Japan. What's the best time to visit?",
        "What cities should I prioritize for a 10-day trip?",
        "Any tips for getting around between cities?",
    ]

    with pandaprobe.session(session_id):
        for question in questions:
            handler = LangGraphCallbackHandler()
            print(f"User:    {question}")
            reply = chat_turn(question, history, handler)
            print(f"Advisor: {reply}\n")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print(f"All {len(questions)} turns traced under session_id={session_id}")
