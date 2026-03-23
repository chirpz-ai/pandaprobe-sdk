"""Anthropic wrapper — messages.create with extended thinking.

Demonstrates wrap_anthropic instrumenting a messages.create call with
extended thinking enabled. The wrapper automatically creates an LLM span
capturing model, tokens, I/O, and reasoning summary.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export ANTHROPIC_API_KEY="sk-ant-..."

Run:
    uv run python examples/anthropic/01_chat_completion.py
"""

import anthropic

import pandaprobe
from pandaprobe.wrappers import wrap_anthropic

client = wrap_anthropic(anthropic.Anthropic())

if __name__ == "__main__":
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        thinking={"type": "enabled", "budget_tokens": 1024},
        system="You are a concise assistant. Answer in one or two sentences.",
        messages=[
            {"role": "user", "content": "Explain what a Python decorator is."},
        ],
    )

    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.input_tokens} input, {response.usage.output_tokens} output")

    for block in response.content:
        if block.type == "thinking":
            print(f"\nThinking:\n{block.thinking}")
        elif block.type == "text":
            print(f"\nResponse:\n{block.text}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
