# PandaProbe SDK

Multi-language SDK for [PandaProbe](https://github.com/chirpz-ai/pandaprobe-sdk) -- open-source agent engineering platform.

## Languages

| Language | Status | Path |
|----------|--------|------|
| **Python** | Beta | [`libraries/python/`](libraries/python/) |
| TypeScript | Planned | [`libraries/typescript/`](libraries/typescript/) |

## Python SDK

```bash
pip install pandaprobe
```

### Quick Start

Set environment variables:

```bash
export PANDAPROBE_API_KEY="sk_pp_..."
export PANDAPROBE_PROJECT_NAME="my-project"
```

Then use the SDK — auto-initialization happens on first use:

```python
import pandaprobe

# Decorator-based tracing
@pandaprobe.trace(name="my-agent")
def run_agent(query: str):
    @pandaprobe.span(name="llm-call", kind="LLM")
    def call_llm(prompt):
        return openai_client.chat.completions.create(...)
    return call_llm(query)

# OpenAI wrapper (automatic LLM tracing)
from pandaprobe.wrappers import wrap_openai
client = wrap_openai(openai.OpenAI())

# LangGraph integration
from pandaprobe.integrations.langgraph import LangGraphCallbackHandler
result = graph.invoke(input, config={"callbacks": [LangGraphCallbackHandler()]})

# Session management (works across all layers)
with pandaprobe.session("conversation-123"):
    run_agent("Hello!")
```

See [`libraries/python/README.md`](libraries/python/README.md) for full documentation.

## Development

```bash
# Python SDK
make py-install-dev   # Install with dev dependencies
make py-lint          # Run linter (ruff)
make py-format        # Auto-format
make py-test          # Run tests
make py-test-cov      # Tests with coverage
make py-build         # Build wheel
make py-clean         # Clean build artifacts
```

## License

MIT -- see [LICENSE](LICENSE).
