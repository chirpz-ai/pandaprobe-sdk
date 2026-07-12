"""LiteLLM wrapper — routing to a Google Gemini model with automatic tracing.

Demonstrates wrap_litellm instrumenting litellm.completion against Gemini. The
"gemini/" prefix routes to Google AI Studio (API-key auth); LiteLLM reads the
GEMINI_API_KEY env var for this provider.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export GEMINI_API_KEY="..."

Run:
    uv run python examples/litellm/03_gemini.py
"""

import litellm

import pandaprobe
from pandaprobe.wrappers import wrap_litellm

wrap_litellm(litellm)

if __name__ == "__main__":
    response = litellm.completion(
        model="gemini/gemini-3.1-flash-lite",
        messages=[
            {"role": "system", "content": "You are a concise assistant. Answer in one or two sentences."},
            {"role": "user", "content": "What is a generator in Python and when should I use one?"},
        ],
        max_tokens=150,
    )

    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion")
    print(f"\nResponse:\n{response.choices[0].message.content}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
