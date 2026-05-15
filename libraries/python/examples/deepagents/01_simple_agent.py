"""DeepAgents — simple agent with automatic tracing.

Builds a minimal deep agent via ``create_deep_agent`` with one custom tool and
traces the invocation via ``DeepAgentsCallbackHandler``. The deep agent ships
with a built-in todo-planning tool (``write_todos``) and a virtual-filesystem
tool family, all of which appear automatically in the recorded trace tree.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/deepagents/01_simple_agent.py
"""

from deepagents import create_deep_agent
from langchain.tools import tool

import pandaprobe
from pandaprobe.integrations.deepagents import DeepAgentsCallbackHandler


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model="openai:gpt-5.4-nano",
    tools=[get_weather],
    system_prompt=(
        "You are a helpful weather assistant. Plan your work using write_todos "
        "before answering, and call get_weather when asked about weather."
    ),
)


if __name__ == "__main__":
    handler = DeepAgentsCallbackHandler(tags=["weather-agent", "example"])

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
        config={"callbacks": [handler]},
    )

    final_message = result["messages"][-1]
    print(f"Agent: {final_message.content}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
