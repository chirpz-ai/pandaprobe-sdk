"""Mistral wrapper — single-turn chat completion with automatic tracing.

Demonstrates wrap_mistral instrumenting a chat.complete call. The wrapper
automatically creates an LLM span capturing model, tokens, I/O, and model
parameters.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export MISTRAL_API_KEY="..."

Run:
    uv run python examples/mistral/01_chat_completion.py
"""

import os

from mistralai.client import Mistral

import pandaprobe
from pandaprobe.wrappers import wrap_mistral

client = wrap_mistral(Mistral(api_key=os.environ["MISTRAL_API_KEY"]))

if __name__ == "__main__":
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain what a Python decorator is in one sentence."},
        ],
        temperature=0.5,
        max_tokens=200,
    )

    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion")
    print(f"\nResponse:\n{response.choices[0].message.content}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
