"""LiteLLM wrapper — routing to an Anthropic Claude model with automatic tracing.

Demonstrates wrap_litellm instrumenting litellm.completion against Claude. The
same wrapper works across every LiteLLM-supported provider — only the ``model``
string changes ("anthropic/..." selects Claude via the Anthropic API).

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export ANTHROPIC_API_KEY="sk-ant-..."

Run:
    uv run python examples/litellm/02_claude.py
"""

import litellm

import pandaprobe
from pandaprobe.wrappers import wrap_litellm

wrap_litellm(litellm)

if __name__ == "__main__":
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": "You are a concise assistant. Answer in one or two sentences."},
            {"role": "user", "content": "What is the difference between a list and a tuple in Python?"},
        ],
        max_tokens=150,
    )

    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion")
    print(f"\nResponse:\n{response.choices[0].message.content}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
