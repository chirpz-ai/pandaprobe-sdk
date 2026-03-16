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

```python
import pandaprobe

# Initialize the SDK
pandaprobe.init(api_key="sk_pp_...", project_name="my-project")

# Option 1: Decorator-based tracing
@pandaprobe.trace(name="my-agent")
def run_agent(query: str):
    @pandaprobe.span(name="llm-call", kind="LLM")
    def call_llm(prompt):
        return openai_client.chat.completions.create(...)
    return call_llm(query)

# Option 2: OpenAI wrapper (automatic LLM tracing)
from pandaprobe.wrappers import wrap_openai
client = wrap_openai(openai.OpenAI())

# Option 3: LangGraph integration
from pandaprobe.integrations.langgraph import CallbackHandler
result = graph.invoke(input, config={"callbacks": [CallbackHandler()]})
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
