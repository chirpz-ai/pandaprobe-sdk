"""Decorator-based tracing with real OpenAI calls.

Demonstrates @pandaprobe.trace and @pandaprobe.span decorators wrapping
real functions — including a live OpenAI LLM call.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="decorator-example"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/decorators/01_trace_and_span.py
"""

import openai

import pandaprobe

pandaprobe.init(debug=True)

KNOWLEDGE_BASE = {
    "password": "To reset your password, click 'Forgot password' on the login page and follow the email link.",
    "billing": "You can update billing info under Settings > Billing. We accept Visa, Mastercard, and PayPal.",
    "refund": "Refunds are processed within 5-7 business days. Contact support@example.com for help.",
}

client = openai.OpenAI()


@pandaprobe.trace(name="customer-support-agent", tags=["support", "example"])
def run_agent(query: str) -> str:
    """A simple support agent: retrieve context, then generate an answer via OpenAI."""
    context = search_knowledge_base(query)
    answer = generate_response(query, context)
    return answer


@pandaprobe.span(name="knowledge-base-search", kind="RETRIEVER")
def search_knowledge_base(query: str) -> str:
    """Keyword search over a local knowledge base."""
    query_lower = query.lower()
    matches = [v for k, v in KNOWLEDGE_BASE.items() if k in query_lower]
    if matches:
        return "\n".join(matches)
    return "No relevant articles found."


@pandaprobe.span(name="generate-response", kind="LLM", model="gpt-4o-mini")
def generate_response(query: str, context: str) -> str:
    """Call OpenAI to generate an answer grounded in the retrieved context."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support assistant. "
                    "Answer the user's question using only the provided context. "
                    "If the context doesn't help, say you don't know."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    result = run_agent("How do I reset my password?")
    print(f"\nAgent response:\n{result}")

    pandaprobe.get_client().flush()
    pandaprobe.get_client().shutdown()
    print("\nTrace sent to PandaProbe backend.")
