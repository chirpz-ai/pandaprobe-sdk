"""LangGraph — ReAct agent with tool calls.

Builds a ReAct-style agent with weather and population tools, and traces
the full tool-calling loop via LangGraphCallbackHandler.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/langgraph/02_tool_agent.py
"""

from typing import Annotated

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

import pandaprobe
from pandaprobe.integrations.langgraph import LangGraphCallbackHandler


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "london": "Cloudy, 15°C, 70% humidity",
        "tokyo": "Sunny, 28°C, 45% humidity",
        "new york": "Partly cloudy, 22°C, 55% humidity",
        "paris": "Rainy, 12°C, 85% humidity",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def get_population(city: str) -> str:
    """Get the approximate population of a city."""
    populations = {
        "london": "8.8 million",
        "tokyo": "13.9 million",
        "new york": "8.3 million",
        "paris": "2.2 million",
    }
    return populations.get(city.lower(), f"Population data not available for {city}")


tools = [get_weather, get_population]
llm = ChatOpenAI(model="gpt-5.4-nano").bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState) -> dict:
    system = SystemMessage(content="You are a helpful assistant with access to weather and population tools.")
    messages = [system, *state["messages"]]
    return {"messages": [llm.invoke(messages)]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile()


if __name__ == "__main__":
    handler = LangGraphCallbackHandler(tags=["tool-agent", "example"])

    result = app.invoke(
        {"messages": [("user", "What's the weather like in London and what's its population?")]},
        config={"callbacks": [handler]},
    )

    final_message = result["messages"][-1]
    print(f"Agent: {final_message.content}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
