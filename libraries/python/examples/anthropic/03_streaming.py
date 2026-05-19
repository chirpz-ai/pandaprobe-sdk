"""Anthropic wrapper — streaming messages.stream with automatic tracing.

Demonstrates wrap_anthropic handling a streamed response via the
``messages.stream(...)`` context manager.  The wrapper collects events and
reduces them into a single LLM span with the full output text,
time-to-first-token, reasoning summary (if extended thinking is enabled),
and final token usage.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export ANTHROPIC_API_KEY="sk-ant-..."

Run:
    uv run python examples/anthropic/03_streaming.py
"""

import anthropic

import pandaprobe
from pandaprobe.wrappers import wrap_anthropic

client = wrap_anthropic(anthropic.Anthropic())

if __name__ == "__main__":
    print("Streaming response:\n")

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system="You are a helpful assistant.",
        messages=[
            {"role": "user", "content": "Write a short poem about debugging code."},
        ],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print("\n")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("Trace sent to PandaProbe backend.")
