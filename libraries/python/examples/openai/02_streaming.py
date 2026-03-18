"""OpenAI wrapper — streaming chat completion with automatic tracing.

Demonstrates wrap_openai handling a streamed response. The wrapper collects
chunks and reduces them into a single LLM span with the full output.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="openai-streaming-example"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/openai/02_streaming.py
"""

import openai

import pandaprobe
from pandaprobe.wrappers import wrap_openai

pandaprobe.init(debug=True)

client = wrap_openai(openai.OpenAI())

if __name__ == "__main__":
    print("Streaming response:\n")

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short poem about debugging code."},
        ],
        temperature=0.7,
        max_tokens=200,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")

    pandaprobe.get_client().flush()
    pandaprobe.get_client().shutdown()
    print("Trace sent to PandaProbe backend.")
