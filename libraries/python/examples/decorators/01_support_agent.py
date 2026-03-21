"""Decorator tracing — customer support agent with retrieval + LLM.

Demonstrates @pandaprobe.trace and @pandaprobe.span decorators wrapping
a support agent flow: keyword retrieval followed by a live OpenAI call.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/decorators/01_support_agent.py
"""

import openai

import pandaprobe

KNOWLEDGE_BASE = {
    "password": "To reset your password, click 'Forgot password' on the login page and follow the email link.",
    "billing": "You can update billing info under Settings > Billing. We accept Visa, Mastercard, and PayPal.",
    "refund": "Refunds are processed within 5-7 business days. Contact support@example.com for help.",
}

client = openai.OpenAI()


@pandaprobe.trace(name="customer-support-agent", tags=["support", "example"])
def run_agent(messages: dict) -> dict:
    """A simple support agent: retrieve context, then generate an answer via OpenAI."""
    query = messages["messages"][-1]["content"]
    context = search_knowledge_base(query)
    answer = generate_response(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful customer support assistant. "
                        "Answer the user's question using only the provided context. "
                        "If the context doesn't help, say you don't know."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
        }
    )
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
def generate_response(messages: dict) -> dict:
    """Call OpenAI to generate an answer grounded in the retrieved context."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages["messages"],
        temperature=0.3,
        max_tokens=200,
    )
    content = response.choices[0].message.content
    return {"messages": [{"role": "assistant", "content": content}]}


if __name__ == "__main__":
    query = "How do I reset my password?"
    result = run_agent({"messages": [{"role": "user", "content": query}]})
    answer = result["messages"][0]["content"]
    print(f"\nAgent response:\n{answer}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
