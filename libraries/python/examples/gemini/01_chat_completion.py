"""Gemini wrapper — basic generate_content with automatic tracing.

Demonstrates wrap_gemini instrumenting a standard models.generate_content call.
The wrapper automatically creates an LLM span capturing model, tokens, and I/O.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export GOOGLE_API_KEY="..."

Run:
    uv run python examples/gemini/01_chat_completion.py
"""

from google import genai
from google.genai import types

import pandaprobe
from pandaprobe.wrappers import wrap_gemini

client = wrap_gemini(genai.Client())

if __name__ == "__main__":
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents="with the rapid development of AI, what are the potential risks for humanity in the next 10 years?",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low", include_thoughts=True),
            temperature=1,
            max_output_tokens=200,
        ),
    )

    print(f"Response:\n{response.text}")

    if response.usage_metadata:
        print(
            f"Tokens: {response.usage_metadata.prompt_token_count} prompt, "
            f"{response.usage_metadata.candidates_token_count} completion"
        )

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
