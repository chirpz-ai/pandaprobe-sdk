"""LangChain — LCEL pipeline with automatic tracing.

Wires a plain LangChain Expression Language (LCEL) pipeline
``prompt | model | parser`` and traces the invocation via
``LangChainCallbackHandler``. Demonstrates that the same handler works for
non-agent chain runs in addition to ``create_agent`` agents.

Required env vars:
    export PANDAPROBE_API_KEY="sk_pp_..."
    export PANDAPROBE_PROJECT_NAME="my-project"
    export PANDAPROBE_ENDPOINT="http://localhost:8000"
    export OPENAI_API_KEY="sk-..."

Run:
    uv run python examples/langchain/02_lcel_chain.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import pandaprobe
from pandaprobe.integrations.langchain import LangChainCallbackHandler

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer concisely (one sentence)."),
        ("user", "Question: {question}\nContext: {context}"),
    ]
)
model = ChatOpenAI(model="gpt-5.4-nano", reasoning_effort="low")
parser = StrOutputParser()
chain = prompt | model | parser


if __name__ == "__main__":
    handler = LangChainCallbackHandler(tags=["lcel", "example"])

    answer = chain.invoke(
        {
            "question": "Can you summarize this morning's meetings?",
            "context": "During this morning's meeting, we solved all world conflict.",
        },
        config={"callbacks": [handler]},
    )

    print(f"Answer: {answer}")

    pandaprobe.flush()
    pandaprobe.shutdown()
    print("\nTrace sent to PandaProbe backend.")
