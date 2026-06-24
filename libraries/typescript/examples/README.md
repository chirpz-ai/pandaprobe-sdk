# PandaProbe TypeScript SDK Examples

Real end-to-end examples that call live services and send traces to the PandaProbe backend. Use these to validate SDK integrations during development.

## Setup

### 1. Install and build the SDK

From `libraries/typescript`:

```bash
make ts-install   # or: pnpm install
pnpm run build    # examples import the built package via self-reference
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
export MISTRAL_API_KEY="..."
# Bedrock uses the standard AWS credential chain.
```

The SDK auto-initializes from these environment variables — no `init()` call is needed.

Optional:

```bash
export PANDAPROBE_DEBUG=true     # debug logging
export PANDAPROBE_ENABLED=false  # disable tracing
```

### 3. Run any example with tsx

```bash
cd libraries/typescript
pnpm exec tsx examples/openai/chat.ts
```

Each example also installs the relevant provider/framework SDK as needed — install it
on demand, e.g. `pnpm add openai` or `pnpm add @langchain/langgraph @langchain/openai`.

## Layout

```
examples/
  decorators/          @trace / @span method decorators
  context-managers/    withTrace / withSpan callback wrappers
  openai/  anthropic/  gemini/  bedrock/  mistral/      LLM provider wrappers
  langchain/  langgraph/  deepagents/                   LangChain-family integrations
  claude-agent-sdk/  openai-agents/  vercel-ai/         agent-framework integrations
```

Always call `await flush()` (or `await shutdown()`) before the process exits so buffered
traces are sent.
