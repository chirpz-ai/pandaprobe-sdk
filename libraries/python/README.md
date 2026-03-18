# PandaProbe Python SDK

Python SDK for [PandaProbe](https://github.com/chirpz-ai/pandaprobe-sdk) — open-source agent tracing and evaluation.

## Installation

```bash
pip install pandaprobe
```

With optional integrations:

```bash
pip install pandaprobe[openai]       # OpenAI wrapper
pip install pandaprobe[langgraph]    # LangGraph integration
pip install pandaprobe[all]          # Everything
```

## Quick Start

### 1. Initialize the SDK

```python
import pandaprobe

pandaprobe.init(
    api_key="sk_pp_...",
    project_name="my-project",
    endpoint="http://localhost:8000",
)
```

Or set environment variables:

```bash
export PANDAPROBE_API_KEY="sk_pp_..."
export PANDAPROBE_PROJECT_NAME="my-project"
```

### 2. Decorator-based tracing (custom agents)

```python
import pandaprobe

@pandaprobe.trace(name="my-agent")
def run_agent(query: str):
    @pandaprobe.span(name="llm-call", kind="LLM")
    def call_llm(prompt):
        return openai_client.chat.completions.create(...)

    @pandaprobe.span(name="search", kind="TOOL")
    def search(q):
        return search_engine.search(q)

    context = search(query)
    return call_llm(f"Context: {context}\nQuery: {query}")
```

### 3. OpenAI wrapper (automatic LLM tracing)

```python
from pandaprobe.wrappers import wrap_openai
import openai

client = wrap_openai(openai.OpenAI())

# Every call is now automatically traced:
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### 4. LangGraph integration

```python
from pandaprobe.integrations.langgraph import LangGraphCallbackHandler

handler = LangGraphCallbackHandler()
result = graph.invoke(
    {"messages": [HumanMessage(content="hello")]},
    config={"callbacks": [handler]},
)
```

### 5. Programmatic scoring

```python
client.score(
    trace_id="...",
    name="user_satisfaction",
    value="0.9",
    data_type="NUMERIC",
    reason="User clicked thumbs up",
)
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `PANDAPROBE_API_KEY` | *(required)* | API key |
| `PANDAPROBE_PROJECT_NAME` | *(required)* | Project name |
| `PANDAPROBE_ENDPOINT` | `http://localhost:8000` | Backend URL |
| `PANDAPROBE_ENABLED` | `true` | Disable SDK silently |
| `PANDAPROBE_BATCH_SIZE` | `10` | Traces per flush batch |
| `PANDAPROBE_FLUSH_INTERVAL` | `5.0` | Seconds between flushes |
| `PANDAPROBE_DEBUG` | `false` | Verbose logging |

## Development

```bash
make py-install-dev   # Install with dev deps
make py-lint          # Run linter
make py-format        # Auto-format
make py-test          # Run tests
make py-test-cov      # Tests with coverage
```

## License

MIT
