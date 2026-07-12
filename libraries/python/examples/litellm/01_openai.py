"""LiteLLM wrapper — routing to an OpenAI model with automatic tracing.

Demonstrates wrap_litellm instrumenting litellm.completion. LiteLLM exposes a
unified, OpenAI-compatible interface; the wrapper produces an LLM span capturing
model, tokens, and I/O — exactly like the provider-specific wrappers.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/litellm/01_openai.py
"""

import litellm

import pandaprobe
from pandaprobe.wrappers import wrap_litellm

wrap_litellm(litellm)

if __name__ == "__main__":
    response = litellm.completion(
        model="gpt-5.4-nano",
        messages=[
            {"role": "system", "content": "You are a concise assistant. Answer in one or two sentences."},
            {"role": "user", "content": "Explain what a Python decorator is."},
        ],
        max_tokens=150,
    )

    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion")
    print(f"\nResponse:\n{response.choices[0].message.content}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
