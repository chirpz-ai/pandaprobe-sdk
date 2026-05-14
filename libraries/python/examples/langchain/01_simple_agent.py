"""LangChain — simple agent with automatic tracing.

Builds a minimal LangChain agent via ``create_agent`` with one tool and
traces the invocation via ``LangChainCallbackHandler``.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/langchain/01_simple_agent.py
"""

from langchain.agents import create_agent
from langchain.tools import tool

import pandaprobe
from pandaprobe.integrations.langchain import LangChainCallbackHandler


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="openai:gpt-5.4-nano",
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant. Use the get_weather tool when asked about weather.",
)


if __name__ == "__main__":
    handler = LangChainCallbackHandler(tags=["weather-agent", "example"])

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
        config={"callbacks": [handler]},
    )

    final_message = result["messages"][-1]
    print(f"Agent: {final_message.content}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
