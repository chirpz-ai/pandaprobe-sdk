"""Gemini wrapper — streaming generate_content with automatic tracing.

Demonstrates wrap_gemini handling a streamed response via
``models.generate_content_stream(...)``.  The wrapper collects chunks and
reduces them into a single LLM span with the full output text, time-to-first-
token, and final token usage.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export GOOGLE_API_KEY="..."

Run:
    uv run python examples/gemini/03_streaming.py
"""

from google import genai
from google.genai import types

import pandaprobe
from pandaprobe.wrappers import wrap_gemini

client = wrap_gemini(genai.Client())

if __name__ == "__main__":
    print("Streaming response:\n")

    stream = client.models.generate_content_stream(
        model="gemini-3.1-flash-lite-preview",
        contents="Write a short poem about debugging code.",
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant.",
            temperature=0.7,
            max_output_tokens=200,
        ),
    )

    for chunk in stream:
        if chunk.text:
            print(chunk.text, end="", flush=True)

    print("\n")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("Trace sent to PandaProbe backend.")
