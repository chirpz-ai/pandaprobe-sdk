"""Example: OpenAI wrapper with automatic LLM tracing.

Prerequisites:
    uv pip install openai

Run with:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="openai-wrapper-example"
    export OPENAI_API_KEY="sk-..."
    python examples/02_openai_wrapper.py

Optionally override the backend URL:
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
"""

import openai

import pandaprobe
from pandaprobe.wrappers import wrap_openai

pandaprobe.init(debug=True)

client = wrap_openai(openai.OpenAI())

if __name__ == "__main__":
    # Every call is now automatically traced as an LLM span.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        temperature=0.3,
        max_tokens=100,
    )

    print(f"\nResponse: {response.choices[0].message.content}")

    pandaprobe.get_client().flush()
    pandaprobe.get_client().shutdown()
    print("Trace sent to PandaProbe backend.")
