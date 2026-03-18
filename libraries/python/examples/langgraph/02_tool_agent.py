"""LangGraph integration — tool-calling agent with PandaProbe tracing.

Builds a LangGraph ReAct agent with two tools (weather lookup and calculator).
The agent decides which tools to call based on the user's query.
PandaProbe's LangGraphCallbackHandler captures LLM, tool, and chain spans.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="langgraph-tool-agent-example"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Extra deps:
    uv pip install langchain-openai

Run:
    uv run python examples/langgraph/02_tool_agent.py
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import pandaprobe
from pandaprobe.integrations.langgraph import LangGraphCallbackHandler

pandaprobe.init(debug=True)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "san francisco": "62°F, foggy",
        "new york": "75°F, sunny",
        "london": "58°F, overcast",
        "tokyo": "80°F, humid",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Only supports basic arithmetic."""
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: only basic arithmetic is supported"
    try:
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(llm, tools=[get_weather, calculator])

if __name__ == "__main__":
    handler = LangGraphCallbackHandler()

    queries = [
        "What's the weather in San Francisco and New York?",
        "What is (42 * 3) + 17?",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("=" * 60)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"callbacks": [handler]},
        )

        ai_message = result["messages"][-1]
        print(f"\nAgent response:\n{ai_message.content}")

    pandaprobe.get_client().flush()
    pandaprobe.get_client().shutdown()
    print(f"\n{'=' * 60}")
    print("All traces sent to PandaProbe backend.")
