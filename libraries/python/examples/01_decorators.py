"""Example: Decorator-based tracing for custom agents.

Run with:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="example-project"
    python examples/01_decorators.py

Optionally override the backend URL:
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
"""

import time

import pandaprobe

pandaprobe.init(debug=True)


@pandaprobe.trace(name="customer-support-agent")
def run_agent(query: str) -> str:
    """Simulates an agent that searches a knowledge base and generates a response."""
    context = search_kb(query)
    answer = call_llm(f"Context: {context}\nQuery: {query}")
    return answer


@pandaprobe.span(name="knowledge-base-search", kind="RETRIEVER")
def search_kb(query: str) -> str:
    """Simulates a retriever call."""
    time.sleep(0.1)
    return f"[KB result for '{query}': Password reset instructions are in the FAQ.]"


@pandaprobe.span(name="generate-response", kind="LLM")
def call_llm(prompt: str) -> str:
    """Simulates an LLM call."""
    time.sleep(0.2)
    return "You can reset your password by clicking 'Forgot password' on the login page."


if __name__ == "__main__":
    result = run_agent("How do I reset my password?")
    print(f"\nAgent response: {result}")

    # Flush to ensure the trace is sent before the script exits.
    pandaprobe.get_client().flush()
    pandaprobe.get_client().shutdown()
    print("Trace sent to PandaProbe backend.")
