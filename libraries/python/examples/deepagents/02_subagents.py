"""DeepAgents — sub-agent dispatch via the built-in ``task`` tool.

Declares a custom sub-agent (``researcher``) alongside the main deep agent.
When the main agent decides it needs research, it calls the built-in ``task``
tool, which synchronously invokes the sub-agent's compiled graph with the same
callbacks forwarded — so the entire sub-agent run is captured under the parent
trace as a nested subtree:

    DeepAgents (CHAIN, root)
    └── tools (AGENT)
        └── task (TOOL)             ← sub-agent dispatcher
            └── researcher (AGENT)  ← sub-agent root
                └── …

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/deepagents/02_subagents.py
"""

from deepagents import create_deep_agent
from langchain.tools import tool

import pandaprobe
from pandaprobe.integrations.deepagents import DeepAgentsCallbackHandler


@tool
def search_papers(topic: str) -> str:
    """Return a short list of recent papers on a topic."""
    return (
        f"Top 3 recent papers on {topic}:\n"
        "1. Smith et al., 2024 — A survey of advances\n"
        "2. Lee & Kumar, 2024 — Empirical comparisons\n"
        "3. Garcia, 2023 — Foundations and theory"
    )


researcher_subagent = {
    "name": "researcher",
    "description": (
        "Searches for and summarizes recent academic papers on a given topic. "
        "Use this when the user asks for a literature overview."
    ),
    "system_prompt": (
        "You are a research assistant. When given a topic, call search_papers, "
        "then return a concise 2-sentence summary of what you found."
    ),
    "tools": [search_papers],
    "model": "openai:gpt-5.4-nano",
}


agent = create_deep_agent(
    model="openai:gpt-5.4-nano",
    tools=[],
    system_prompt=(
        "You are a senior research lead. For any literature/research request, "
        "delegate to the 'researcher' sub-agent via the task tool, then "
        "synthesize the result into a final answer."
    ),
    subagents=[researcher_subagent],
)


if __name__ == "__main__":
    handler = DeepAgentsCallbackHandler(tags=["research-agent", "subagents"])

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Give me a brief summary of recent research on retrieval-augmented generation.",
                }
            ]
        },
        config={"callbacks": [handler]},
    )

    final_message = result["messages"][-1]
    print(f"Agent: {final_message.content}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
