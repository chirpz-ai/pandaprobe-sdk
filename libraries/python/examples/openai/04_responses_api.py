"""OpenAI wrapper — Responses API with reasoning summaries.

Demonstrates wrap_openai automatically tracing a Responses API call.
The wrapper captures reasoning summaries, token usage (including reasoning
tokens), and creates child TOOL spans for any built-in tool calls.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/openai/04_responses_api.py
"""

import openai

import pandaprobe
from pandaprobe.wrappers import wrap_openai

client = wrap_openai(openai.OpenAI())

if __name__ == "__main__":
    response = client.responses.create(
        model="gpt-5.4-nano",
        instructions="You are a concise assistant. Reason and answer the question.",
        input="with the rapid development of AI, what are the potential risks for humanity in the next 10 years?",
        reasoning={"effort": "medium", "summary": "auto"},
        max_output_tokens=500,
    )

    print(f"Model: {response.model}")
    print(f"\nResponse:\n{response.output_text}")

    for item in response.output:
        if getattr(item, "type", None) == "reasoning" and item.summary:
            print(f"\nReasoning summary:\n{item.summary[0].text}")

    if response.usage:
        print(f"\nTokens: {response.usage.input_tokens} input, {response.usage.output_tokens} output")
        if response.usage.output_tokens_details:
            print(f"Reasoning tokens: {response.usage.output_tokens_details.reasoning_tokens}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
