"""Example: Manual trace/span context managers with scoring.

Run with:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="context-manager-example"
    python examples/03_context_manager.py

Optionally override the backend URL:
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
"""

import time

import pandaprobe

client = pandaprobe.Client(debug=True)

if __name__ == "__main__":
    with client.trace("rag-pipeline", input={"query": "Tell me about pandas"}) as trace:
        # Retriever span
        with trace.span("vector-search", kind="RETRIEVER") as retriever:
            retriever.set_input({"query": "Tell me about pandas", "top_k": 5})
            time.sleep(0.1)
            docs = ["Pandas are bears native to China.", "Giant pandas eat bamboo."]
            retriever.set_output(docs)

        # LLM span
        with trace.span("llm-generation", kind="LLM", model="gpt-4o") as llm:
            llm.set_input({"messages": [{"role": "user", "content": "Tell me about pandas"}]})
            time.sleep(0.2)
            answer = "Pandas are bears native to China that primarily eat bamboo."
            llm.set_output(answer)
            llm.set_token_usage(prompt_tokens=45, completion_tokens=15)
            llm.set_model_parameters({"temperature": 0.7, "max_tokens": 200})

        trace.set_output({"answer": answer})

    # Submit a programmatic score for the trace
    client.score(
        trace_id=trace.trace_id,
        name="relevance",
        value="0.95",
        data_type="NUMERIC",
        reason="Answer directly addresses the query using retrieved context.",
    )

    print(f"Trace ID: {trace.trace_id}")
    client.flush()
    client.shutdown()
    print("Trace + score sent to PandaProbe backend.")
