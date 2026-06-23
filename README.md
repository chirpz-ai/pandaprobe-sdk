# PandaProbe SDK

[![Release Notes](https://img.shields.io/github/release/chirpz-ai/pandaprobe-sdk)](https://github.com/chirpz-ai/pandaprobe-sdk/releases)
[![PyPI - Version](https://img.shields.io/pypi/v/pandaprobe?logo=pypi&logoColor=white)](https://pypi.org/project/pandaprobe/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pandaprobe?logo=python&logoColor=white)](https://pypi.org/project/pandaprobe/)
[![npm - Version](https://img.shields.io/npm/v/pandaprobe?logo=npm&logoColor=white)](https://www.npmjs.com/package/pandaprobe)

Multi-language SDK for [PandaProbe](https://github.com/chirpz-ai/pandaprobe-sdk) - open source agent engineering platform.

Both SDKs mirror the same architecture, trace/span schema, and `CHAIN → AGENT → LLM → TOOL` normalization, so traces are cross-compatible across languages.

## Languages

| Language | Status | Path |
|----------|--------|------|
| **Python** | Beta | [`libraries/python/`](libraries/python/) |
| **TypeScript** | Beta | [`libraries/typescript/`](libraries/typescript/) |

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

## TypeScript SDK

Requires Node.js ≥ 18.

```bash
npm install pandaprobe
```

### Quick Start

Set environment variables:

```bash
export PANDAPROBE_API_KEY="sk_pp_..."
export PANDAPROBE_PROJECT_NAME="my-project"
```

Then use the SDK — auto-initialization happens on first use:

```ts
import { trace, span, withTrace, withSpan, SpanKind, session, flush } from "pandaprobe";

// Callback wrappers (the analog of Python's `with` context managers)
await withTrace("my-agent", { input: { messages: [{ role: "user", content: "Hello!" }] } }, async (t) => {
  const answer = await withSpan("llm-call", { kind: SpanKind.LLM, model: "gpt-4o" }, async (s) => {
    const out = { messages: [{ role: "assistant", content: "Hi!" }] };
    s.setOutput(out);
    return out;
  });
  t.setOutput(answer);
});

// ...or TS class-method decorators (@trace / @span)

// OpenAI wrapper (automatic LLM tracing)
import { wrapOpenAI } from "pandaprobe/wrappers/openai";
const client = wrapOpenAI(new OpenAI());

// LangGraph integration
import { LangGraphCallbackHandler } from "pandaprobe/integrations/langgraph";
await graph.invoke(input, { callbacks: [new LangGraphCallbackHandler()] });

// Session management (works across all layers)
await session("conversation-123", async () => {
  await runAgent("Hello!");
});

await flush();
```

Wrappers: `pandaprobe/wrappers/{openai,anthropic,gemini,bedrock,mistral}`.
Integrations: `pandaprobe/integrations/{langchain,langgraph,deepagents,claude-agent-sdk,openai-agents,vercel-ai}`.

See [`libraries/typescript/README.md`](libraries/typescript/README.md) for full documentation.

## Development

```bash
# Python SDK
make py-install       # Install all deps (providers, examples, dev tools)
make py-lint          # Run linter (ruff)
make py-format        # Auto-format
make py-test          # Run tests
make py-test-cov      # Tests with coverage
make py-build         # Build wheel
make py-clean         # Clean build artifacts

# TypeScript SDK
make ts-install       # Install dev tooling + lightweight LLM provider SDKs
make ts-lint          # Run linter (Biome)
make ts-format        # Auto-format
make ts-typecheck     # Type-check (tsc --noEmit)
make ts-test          # Run tests (vitest)
make ts-test-cov      # Tests with coverage
make ts-build         # Build (tsup → ESM + CJS + .d.ts)
make ts-clean         # Clean build artifacts

# Agent frameworks install one at a time (heavyweight) — e.g.:
make ts-install-langgraph
make ts-install-vercel-ai
```

## License

MIT -- see [LICENSE](LICENSE).
