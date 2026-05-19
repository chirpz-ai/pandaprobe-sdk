"""Mistral wrapper — streaming chat completion with automatic tracing.

Demonstrates wrap_mistral handling a streamed response via chat.stream(...).
The wrapper collects chunks and reduces them into a single LLM span with the
full output text and final token usage.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export MISTRAL_API_KEY="..."

Run:
    uv run python examples/mistral/02_streaming.py
"""

import os

from mistralai.client import Mistral

import pandaprobe
from pandaprobe.wrappers import wrap_mistral

client = wrap_mistral(Mistral(api_key=os.environ["MISTRAL_API_KEY"]))

if __name__ == "__main__":
    print("Streaming response:\n")

    res = client.chat.stream(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short poem about debugging code."},
        ],
        temperature=0.7,
        max_tokens=200,
    )

    with res as event_stream:
        for event in event_stream:
            delta = event.data.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                print(text, end="", flush=True)

    print("\n")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("Trace sent to PandaProbe backend.")
