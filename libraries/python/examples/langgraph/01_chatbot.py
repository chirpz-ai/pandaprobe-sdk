"""LangGraph integration — simple chatbot with PandaProbe tracing.

Builds a minimal LangGraph StateGraph chatbot (single LLM node) and traces
the full invocation using PandaProbe's LangGraphCallbackHandler.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="langgraph-chatbot-example"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Extra deps:
    uv pip install langchain-openai

Run:
    uv run python examples/langgraph/01_chatbot.py
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph

import pandaprobe
from pandaprobe.integrations.langgraph import LangGraphCallbackHandler

pandaprobe.init(debug=True)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


def chatbot_node(state: MessagesState) -> MessagesState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot_node)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)
app = graph.compile()

if __name__ == "__main__":
    handler = LangGraphCallbackHandler()

    result = app.invoke(
        {"messages": [{"role": "user", "content": "What are the three laws of robotics?"}]},
        config={"callbacks": [handler]},
    )

    ai_message = result["messages"][-1]
    print(f"Chatbot response:\n{ai_message.content}")

    pandaprobe.get_client().flush()
    pandaprobe.get_client().shutdown()
    print("\nTrace sent to PandaProbe backend.")
