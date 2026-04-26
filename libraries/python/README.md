# PandaProbe Python SDK

Python SDK for [PandaProbe](https://www.pandaprobe.com/) — open source agent engineering platform.

## Installation

```bash
pip install pandaprobe
```

With optional LLM provider wrappers:

```bash
pip install "pandaprobe[openai]"       # OpenAI wrapper
pip install "pandaprobe[gemini]"       # Google Gemini wrapper
pip install "pandaprobe[anthropic]"    # Anthropic wrapper
```

With optional agent framework integrations:

```bash
pip install "pandaprobe[langgraph]"         # LangGraph / LangChain
pip install "pandaprobe[google-adk]"        # Google Agent Development Kit
pip install "pandaprobe[claude-agent-sdk]"  # Anthropic Claude Agent SDK
pip install "pandaprobe[crewai]"            # CrewAI
pip install "pandaprobe[openai-agents]"     # OpenAI Agents SDK
```

## Quick Start

### 1. Set environment variables

```bash
export PANDAPROBE_API_KEY="sk_pp_..."
export PANDAPROBE_PROJECT_NAME="my-project"
export PANDAPROBE_ENDPOINT="https://api.pandaprobe.com"   # optional — this is the default
export PANDAPROBE_ENVIRONMENT="production"   # optional
export PANDAPROBE_RELEASE="v1.2.0"           # optional
```

The SDK auto-initializes from these environment variables on first use — no explicit `init()` call is needed. To disable tracing, set `PANDAPROBE_ENABLED=false`.

You can still use `pandaprobe.init(...)` for programmatic configuration if preferred.

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

# Chat Completions API — automatically traced:
response = client.chat.completions.create(
    model="gpt-5.4-nano",
    messages=[{"role": "user", "content": "Hello"}],
)

# Responses API — also automatically traced, including reasoning summaries
# and built-in tool calls (web_search, function_call, etc.) as child spans:
response = client.responses.create(
    model="gpt-5.4-nano",
    input="Explain recursion in one sentence.",
    reasoning={"effort": "low", "summary": "auto"},
)
```

### 4. Gemini wrapper (automatic LLM tracing)

```python
from pandaprobe.wrappers import wrap_gemini
from google import genai

client = wrap_gemini(genai.Client())

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain recursion in one sentence.",
)
```

### 5. Anthropic wrapper (automatic LLM tracing)

```python
from pandaprobe.wrappers import wrap_anthropic
import anthropic

client = wrap_anthropic(anthropic.Anthropic())

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=150,
    system="You are a concise assistant.",
    messages=[{"role": "user", "content": "Explain recursion in one sentence."}],
)
```

### 6. Agent framework integrations

All integrations below auto-trace agent execution — LLM calls, tool use, handoffs, and more — with no manual span creation.

#### LangGraph

```python
from pandaprobe.integrations.langgraph import LangGraphCallbackHandler

handler = LangGraphCallbackHandler()
result = graph.invoke(
    {"messages": [HumanMessage(content="hello")]},
    config={"callbacks": [handler]},
)
```

#### Google ADK

```python
from pandaprobe.integrations.google_adk import GoogleADKAdapter

adapter = GoogleADKAdapter()
adapter.instrument()

# All Runner.run_async() calls are now traced automatically
result = await runner.run_async(user_id="user-1", session_id="s-1", new_message=msg)
```

#### Claude Agent SDK

```python
from pandaprobe.integrations.claude_agent_sdk import ClaudeAgentSDKAdapter

adapter = ClaudeAgentSDKAdapter()
adapter.instrument()

# All client.query() / client.receive_response() calls are now traced automatically
result = client.query(prompt="Explain recursion.")
```

#### CrewAI

```python
from pandaprobe.integrations.crewai import CrewAIAdapter

adapter = CrewAIAdapter()
adapter.instrument()

# All crew.kickoff() calls are now traced automatically
result = crew.kickoff()
```

#### OpenAI Agents SDK

```python
from pandaprobe.integrations.openai_agents import OpenAIAgentsAdapter

adapter = OpenAIAgentsAdapter()
adapter.instrument()

# All Runner.run() calls are now traced automatically
result = await Runner.run(agent, input="Explain recursion.")
```

### 7. Session and user tracking

Group related traces under a session and/or user using the universal context API:

```python
import pandaprobe

# Context managers — scoped to the block
with pandaprobe.session("conversation-123"):
    with pandaprobe.user("user-abc"):
        run_agent("What is recursion?")
        run_agent("Can you give me an example?")

# Imperative — useful for dynamic switching
pandaprobe.set_session("conversation-456")
pandaprobe.set_user("user-xyz")
run_agent("New topic")
```

Both propagate across all SDK layers (decorators, wrappers, integrations, context managers). Explicit parameters (`session_id=`, `user_id=`) take precedence over the context.

### 8. Programmatic scoring

```python
pandaprobe.score(
    trace_id="...",
    name="user_satisfaction",
    value="0.9",
    data_type="NUMERIC",
    reason="User clicked thumbs up",
)
```

### 9. Flushing

For short-lived scripts, call `pandaprobe.flush()` before exiting to ensure all traces are sent. For long-running processes, the SDK flushes automatically via a background thread and an `atexit` handler.

```python
pandaprobe.flush()
pandaprobe.shutdown()
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `PANDAPROBE_API_KEY` | *(required)* | API key |
| `PANDAPROBE_PROJECT_NAME` | *(required)* | Project name |
| `PANDAPROBE_ENDPOINT` | `https://api.pandaprobe.com` | Backend URL |
| `PANDAPROBE_ENVIRONMENT` | `None` | Environment tag (e.g. `production`, `staging`) |
| `PANDAPROBE_RELEASE` | `None` | Release/version tag (e.g. `v1.2.0`) |
| `PANDAPROBE_ENABLED` | `true` | Enable/disable SDK |
| `PANDAPROBE_BATCH_SIZE` | `10` | Traces per flush batch |
| `PANDAPROBE_FLUSH_INTERVAL` | `5.0` | Seconds between flushes |
| `PANDAPROBE_DEBUG` | `false` | Verbose logging |

## Development

```bash
make py-install       # Install all deps (providers, examples, dev tools)
make py-lint          # Run linter
make py-format        # Auto-format
make py-test          # Run tests
make py-test-cov      # Tests with coverage
```

## License

MIT
