# PandaProbe SDK Examples

Real end-to-end examples that call live services and send traces to the PandaProbe backend. Use these to validate SDK integrations during development.

## Setup

### 1. Install the SDK with all dependencies

From the repo root:

```bash
make py-install
```

### 2. Export environment variables

All examples require:

```bash
export PANDAPROBE_API_KEY="sk_pp_..."
export PANDAPROBE_PROJECT_NAME="my-project"
export PANDAPROBE_ENDPOINT="https://api.pandaprobe.com/"

# Provider keys (set whichever you need):
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

The SDK auto-initializes from these environment variables — no `pandaprobe.init()` call is needed.

Examples use [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
[Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash), and
[Claude Haiku 5.5](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide).
Provider prefixes are retained where required; Bedrock uses
[`us.anthropic.claude-haiku-5-5`](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-5-5.html).
The OpenAI examples that use Chat Completions with tools explicitly disable
reasoning, as required by GPT-6 Luna on that endpoint. OpenAI Agents and the
Responses API example use Responses, which supports reasoning with tools.
Claude examples use adaptive thinking, and Gemini examples use supported thinking levels.


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

### Reflect: live agent pressure tests

Each of the seven Python agent-framework folders includes `04_reflect.py`: a real
model-backed support agent handling six three-turn customer conversations with
policy changes, duplicate requests, rule retrieval, reflection, and independent
paired replay. See [the live support guide](reflect/live_support.md) for installs,
credentials, longer runs, costs, and how to interpret the generated reports.
The business tools use fictional local data; model calls and platform evaluations
are real. The [offline demo](reflect/README.md) remains available without credentials.

Each framework also includes `05_reflect_failure.py`: a controlled stale-runbook incident with
independent ledger checks, held-out paired replay, and accepted-guidance retrieval
on a fresh execution. See [the live failure guide](reflect/live_failure.md) for
commands and framework evidence limits, and the [workspace guide](reflect/workspace.md)
for the generated files.

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
| `anthropic/01_chat_completion.py` | messages.create with adaptive thinking and automatic tracing via `wrap_anthropic` |
| `anthropic/02_multi_turn.py` | Multi-turn conversation with adaptive thinking and session grouping via `pandaprobe.session()` + `wrap_anthropic` |

### LiteLLM Wrapper

| Example | Description |
|---|---|
| `litellm/01_openai.py` | `litellm.completion` routed to an OpenAI model with automatic tracing via `wrap_litellm` |
| `litellm/02_claude.py` | `litellm.completion` routed to an Anthropic Claude model via `wrap_litellm` |
| `litellm/03_gemini.py` | `litellm.completion` routed to a Google Gemini model via `wrap_litellm` |

> Running LiteLLM as a **proxy** instead of the embedded SDK? The proxy exposes an OpenAI-compatible endpoint, so point the `openai` client at the proxy base URL and trace it with [`wrap_openai`](#openai-wrapper) — no LiteLLM-specific wrapping needed.

### LangGraph

| Example | Description |
|---|---|
| `langgraph/01_chatbot.py` | Simple chatbot with `LangGraphCallbackHandler` |
| `langgraph/02_tool_agent.py` | ReAct agent with weather + population tools via `LangGraphCallbackHandler` |
| `langgraph/03_multi_turn.py` | Multi-turn travel advisor with session grouping via `pandaprobe.session()` + `LangGraphCallbackHandler` |

### Google ADK

| Example | Description |
|---|---|
| `google_adk/01_simple_agent.py` | Simple chatbot agent with automatic tracing via `GoogleADKAdapter` |
| `google_adk/02_tool_agent.py` | Agent with weather + population tools via `GoogleADKAdapter` |
| `google_adk/03_multi_turn.py` | Multi-turn travel advisor with session grouping via `pandaprobe.session()` + `GoogleADKAdapter` |

### Claude Agent SDK

| Example | Description |
|---|---|
| `claude_agent_sdk/01_simple_agent.py` | Simple chatbot with automatic tracing via `ClaudeAgentSDKAdapter` |
| `claude_agent_sdk/02_tool_agent.py` | Agent with custom tools (weather + population) via `ClaudeAgentSDKAdapter` |
| `claude_agent_sdk/03_multi_turn.py` | Multi-turn travel advisor with session grouping via `pandaprobe.session()` + `ClaudeAgentSDKAdapter` |

### CrewAI

| Example | Description |
|---|---|
| `crewai/01_simple_crew.py` | Simple crew with a single agent and task via `CrewAIAdapter` |
| `crewai/02_tool_crew.py` | Crew with tool-calling agent (weather + population) via `CrewAIAdapter` |
| `crewai/03_multi_agent.py` | Multi-agent crew with sequential task execution via `CrewAIAdapter` |

### OpenAI Agents SDK

| Example | Description |
|---|---|
| `openai_agents/01_simple_agent.py` | Simple agent with automatic tracing via `OpenAIAgentsAdapter` |
| `openai_agents/02_tool_agent.py` | Agent with function tools (weather + population) via `OpenAIAgentsAdapter` |
| `openai_agents/03_multi_turn.py` | Multi-turn conversation with session grouping via `pandaprobe.session()` + `OpenAIAgentsAdapter` |


## Session API

The SDK provides a universal `session_id` propagation mechanism that works across all layers:

```python
import uuid
import pandaprobe

session_id = str(uuid.uuid4())

# Option 1: Context manager (recommended for scoped usage)
with pandaprobe.session(session_id):
    # All traces created here inherit the session_id
    run_agent(query)

# Option 2: Imperative setter (for dynamic switching)
pandaprobe.set_session(str(uuid.uuid4()))
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
