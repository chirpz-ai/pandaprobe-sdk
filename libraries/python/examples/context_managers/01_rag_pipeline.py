"""Context manager tracing — RAG pipeline with retrieval + LLM + scoring.

Demonstrates pandaprobe.start_trace() and trace.span() for manual
instrumentation, plus pandaprobe.score() to evaluate the trace after completion.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/context_managers/01_rag_pipeline.py
"""

import time

import openai

import pandaprobe

oai_client = openai.OpenAI()

DOCUMENTS = [
    {
        "id": 1,
        "topic": "python",
        "text": "Python is a high-level, interpreted programming language known for its readability.",
    },
    {
        "id": 2,
        "topic": "python",
        "text": "Python supports multiple paradigms including procedural, OOP, and functional programming.",
    },
    {
        "id": 3,
        "topic": "rust",
        "text": "Rust is a systems programming language focused on safety, speed, and concurrency.",
    },
    {
        "id": 4,
        "topic": "javascript",
        "text": "JavaScript is the language of the web, running in browsers and on servers via Node.js.",
    },
    {
        "id": 5,
        "topic": "python",
        "text": "Popular Python frameworks include Django, Flask, and FastAPI for web development.",
    },
]

if __name__ == "__main__":
    query = "What are some popular Python web frameworks?"

    with pandaprobe.start_trace(
        "rag-pipeline", input={"messages": [{"role": "user", "content": query}]}, tags=["rag", "example"]
    ) as trace:
        with trace.span("document-retrieval", kind="RETRIEVER") as retriever:
            retriever.set_input({"query": query, "top_k": 3})

            query_lower = query.lower()
            results = [doc for doc in DOCUMENTS if any(word in doc["text"].lower() for word in query_lower.split())][
                :3
            ]

            retriever.set_output(results)
            retriever.set_metadata({"num_results": len(results)})

        context = "\n".join(doc["text"] for doc in results)

        with trace.span("llm-generation", kind="LLM", model="gpt-5.4-nano") as llm:
            messages = [
                {
                    "role": "system",
                    "content": "Answer the question using only the provided context. Be concise.",
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
            llm.set_input({"messages": messages})

            response = oai_client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=messages,
                reasoning_effort="low",
                max_completion_tokens=200,
            )

            answer = response.choices[0].message.content
            llm.set_output({"messages": [{"role": "assistant", "content": answer}]})
            llm.set_token_usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
            llm.set_model_parameters({"reasoning_effort": "low", "max_completion_tokens": 200})

        trace.set_output({"messages": [{"role": "assistant", "content": answer}]})

    print(f"Query: {query}")
    print(f"Retrieved {len(results)} documents")
    print(f"\nAnswer:\n{answer}")
    print(f"\nTrace ID: {trace.trace_id}")

    pandaprobe.flush()
    time.sleep(2)

    pandaprobe.score(
        trace_id=trace.trace_id,
        name="relevance",
        value="true",
        data_type="BOOLEAN",
        reason="Answer directly addresses the question using retrieved context about Python frameworks.",
    )
    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace + score sent to PandaProbe backend.")
