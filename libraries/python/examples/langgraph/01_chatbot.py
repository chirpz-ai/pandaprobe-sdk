"""LangGraph — simple chatbot with automatic tracing.

Builds a minimal StateGraph chatbot (single LLM node) and traces the
invocation via LangGraphCallbackHandler.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/langgraph/01_chatbot.py
"""

from typing import Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import pandaprobe
from pandaprobe.integrations.langgraph import LangGraphCallbackHandler

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


graph = StateGraph(State)
graph.add_node("chatbot", chatbot_node)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
app = graph.compile()


if __name__ == "__main__":
    handler = LangGraphCallbackHandler(tags=["chatbot", "example"])

    result = app.invoke(
        {"messages": [("user", "What is the capital of France?")]},
        config={"callbacks": [handler]},
    )

    final_message = result["messages"][-1]
    print(f"Bot: {final_message.content}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
