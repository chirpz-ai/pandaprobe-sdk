"""OpenAI wrapper — basic chat completion with automatic tracing.

Demonstrates wrap_openai instrumenting a standard chat.completions.create call.
The wrapper automatically creates an LLM span capturing model, tokens, and I/O.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="openai-chat-example"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/openai/01_chat_completion.py
"""

import openai

import pandaprobe
from pandaprobe.wrappers import wrap_openai

pandaprobe.init(debug=True)

client = wrap_openai(openai.OpenAI())

if __name__ == "__main__":
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise assistant. Answer in one or two sentences."},
            {"role": "user", "content": "Explain what a Python decorator is."},
        ],
        temperature=0.5,
        max_tokens=150,
    )

    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion")
    print(f"\nResponse:\n{response.choices[0].message.content}")

    pandaprobe.get_client().flush()
    pandaprobe.get_client().shutdown()
    print("\nTrace sent to PandaProbe backend.")
