# PandaProbe SDK Examples

Real end-to-end examples that call live services and send traces to the PandaProbe backend. Use these to validate SDK integrations during development.

## Setup

### 1. Install the SDK with example dependencies

From the repo root:

```bash
make py-sync
# or install example extras specifically:
cd libraries/python && uv pip install -e ".[examples]"
```

### 2. Export environment variables

All examples require:

```bash
export PANDAPROBE_API_KEY="sk_pp_..."
export PANDAPROBE_PROJECT_NAME="my-project"
export PANDAPROBE_ENDPOINT="http://localhost:8000"

# Provider keys (set whichever you need):
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

The SDK auto-initializes from these environment variables — no `pandaprobe.init()` call is needed.


To enable debug logging:

```bash
export PANDAPROBE_DEBUG=true
```

To disable tracing:

```bash
export PANDAPROBE_ENABLED=false
```

### 3. Run any example

```bash
cd libraries/python
uv run python examples/openai/01_chat_completion.py
```

## Examples

### Context Managers

| Example | Description |
|---|---|
| `context_managers/01_rag_pipeline.py` | RAG pipeline with retrieval + LLM generation + scoring via `pandaprobe.start_trace()` and `trace.span()` |
| `context_managers/02_multi_turn.py` | Multi-turn tutoring agent with session grouping via `pandaprobe.session()` |

### Decorators

| Example | Description |
|---|---|
| `decorators/01_support_agent.py` | Customer support agent with retrieval + LLM via `@pandaprobe.trace` and `@pandaprobe.span` |
| `decorators/02_multi_turn.py` | Multi-turn topic assistant with session grouping via `pandaprobe.session()` + `@pandaprobe.trace` |

### OpenAI Wrapper

| Example | Description |
|---|---|
| `openai/01_chat_completion.py` | Chat Completions API with automatic tracing via `wrap_openai` |
| `openai/02_streaming.py` | Streaming chat completion with automatic tracing via `wrap_openai` |
| `openai/03_multi_turn.py` | Multi-turn conversation with session grouping via `pandaprobe.session()` + `wrap_openai` |
| `openai/04_responses_api.py` | Responses API with reasoning summaries and automatic tracing via `wrap_openai` |

### Gemini Wrapper

| Example | Description |
|---|---|
| `gemini/01_chat_completion.py` | generate_content with automatic tracing via `wrap_gemini` |
| `gemini/02_multi_turn.py` | Multi-turn conversation with session grouping via `pandaprobe.session()` + `wrap_gemini` |

### Anthropic Wrapper

| Example | Description |
|---|---|
| `anthropic/01_chat_completion.py` | messages.create with automatic tracing via `wrap_anthropic` |

### LangGraph

| Example | Description |
|---|---|
| `langgraph/01_chatbot.py` | Simple chatbot with `LangGraphCallbackHandler` |
| `langgraph/02_tool_agent.py` | ReAct agent with weather + population tools via `LangGraphCallbackHandler` |
| `langgraph/03_multi_turn.py` | Multi-turn travel advisor with session grouping via `pandaprobe.session()` + `LangGraphCallbackHandler` |

## Session API

The SDK provides a universal `session_id` propagation mechanism that works across all layers:

```python
import pandaprobe

# Option 1: Context manager (recommended for scoped usage)
with pandaprobe.session("conversation-123"):
    # All traces created here inherit session_id="conversation-123"
    run_agent(query)

# Option 2: Imperative setter (for dynamic switching)
pandaprobe.set_session("conversation-456")
run_agent(query)
```

Session precedence (highest to lowest):
1. Explicit parameter (`session_id="..."` passed directly)
2. Context var (`pandaprobe.session()` / `pandaprobe.set_session()`)
3. None

## What to Look For

After running an example, check your PandaProbe backend to verify:

- **Traces** appear with the correct name and project
- **Spans** show proper parent-child hierarchy
- **LLM spans** capture model name, token usage, and input/output
- **Tool spans** (LangGraph examples) capture tool name and results
- **Scores** are attached to the correct traces
- **Sessions** group related traces together (multi-turn examples)
