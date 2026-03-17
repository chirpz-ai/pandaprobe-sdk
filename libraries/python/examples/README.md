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
```

OpenAI-based examples additionally require:

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run any example

```bash
cd libraries/python
uv run python libraries/python/examples/openai/01_chat_completion.py
```

## Examples

### Decorators

| Example | Description | Env Vars |
|---|---|---|
| `decorators/01_trace_and_span.py` | `@trace` + `@span` wrapping a real OpenAI call inside a support agent flow | `OPENAI_API_KEY` |

### OpenAI Wrapper

| Example | Description | Env Vars |
|---|---|---|
| `openai/01_chat_completion.py` | `wrap_openai` with a standard chat completion | `OPENAI_API_KEY` |
| `openai/02_streaming.py` | `wrap_openai` with a streaming chat completion | `OPENAI_API_KEY` |

### Context Managers

| Example | Description | Env Vars |
|---|---|---|
| `context_managers/01_rag_pipeline.py` | Manual `client.trace()` / `trace.span()` building a RAG pipeline with real OpenAI + scoring | `OPENAI_API_KEY` |

### LangGraph

| Example | Description | Env Vars |
|---|---|---|
| `langgraph/01_chatbot.py` | Simple `StateGraph` chatbot with `CallbackHandler` | `OPENAI_API_KEY` |
| `langgraph/02_tool_agent.py` | ReAct agent with tool calls (weather + calculator) traced via `CallbackHandler` | `OPENAI_API_KEY` |

### Sessions

| Example | Description | Env Vars |
|---|---|---|
| `sessions/01_multi_turn.py` | Multi-turn conversation grouped under a session, with scoring per turn | `OPENAI_API_KEY` |

## What to Look For

After running an example, check your PandaProbe backend to verify:

- **Traces** appear with the correct name and project
- **Spans** show proper parent-child hierarchy
- **LLM spans** capture model name, token usage, and input/output
- **Tool spans** (LangGraph examples) capture tool name and results
- **Scores** are attached to the correct traces
- **Sessions** group related traces together
