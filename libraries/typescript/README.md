# PandaProbe TypeScript SDK

TypeScript/JavaScript SDK for [PandaProbe](https://github.com/chirpz-ai/pandaprobe-sdk) — an open-source agent engineering platform. Mirrors the Python SDK's architecture, trace/span schema, and `CHAIN → AGENT → LLM → TOOL` normalization, so traces are cross-compatible across languages.

Requires Node.js ≥ 18 (native `fetch` + `AsyncLocalStorage`).

## Install

```bash
npm install pandaprobe
# or: pnpm add pandaprobe
```

Provider/framework SDKs are **optional peer dependencies** — install only what you use
(e.g. `npm install openai`, `npm install @langchain/langgraph`).

## Quick start

The SDK auto-initializes from environment variables:

```bash
export PANDAPROBE_API_KEY="sk_pp_..."
export PANDAPROBE_PROJECT_NAME="my-project"
export PANDAPROBE_ENDPOINT="http://localhost:8000"   # defaults to https://api.pandaprobe.com
```

…or call `init()` explicitly:

```ts
import { init } from "pandaprobe";
init({ apiKey: "sk_pp_...", projectName: "my-project" });
```

## Three instrumentation layers

### 1. Manual instrumentation

Callback wrappers (the analog of Python's `with` context managers):

```ts
import { withTrace, withSpan, SpanKind, flush } from "pandaprobe";

await withTrace("agent", { input: { messages: [{ role: "user", content: "hi" }] } }, async (t) => {
  const answer = await withSpan("llm", { kind: SpanKind.LLM, model: "gpt-4o" }, async (s) => {
    s.setInput({ messages: [{ role: "user", content: "hi" }] });
    const out = { messages: [{ role: "assistant", content: "hello" }] };
    s.setOutput(out);
    return out;
  });
  t.setOutput(answer);
});
await flush();
```

Or TS class-method decorators (`tsconfig` needs `experimentalDecorators`):

```ts
import { trace, span, SpanKind } from "pandaprobe";

class Agent {
  @span({ kind: SpanKind.LLM })
  async generate(input: { messages: unknown[] }) { /* ... */ }

  @trace({ name: "agent" })
  async run(input: { messages: unknown[] }) { return this.generate(input); }
}
```

### 2. Provider wrappers

Monkey-patch an LLM client to emit `LLM` spans automatically:

```ts
import OpenAI from "openai";
import { wrapOpenAI } from "pandaprobe/wrappers/openai";

const client = wrapOpenAI(new OpenAI());
await client.chat.completions.create({ model: "gpt-4o-mini", messages: [...] });
```

Available: `pandaprobe/wrappers/{openai,anthropic,gemini,bedrock,mistral}`.

### 3. Framework integrations

LangChain family (callback-based):

```ts
import { LangGraphCallbackHandler } from "pandaprobe/integrations/langgraph";
await graph.invoke(input, { callbacks: [new LangGraphCallbackHandler()] });
```

Available: `pandaprobe/integrations/{langchain,langgraph,deepagents,claude-agent-sdk,openai-agents,vercel-ai}`.

## Session / user grouping

```ts
import { session, user, setSession } from "pandaprobe";

await session("conv-123", async () => {
  await runAgent(query); // traces inherit session_id
});
```

## Development

```bash
make ts-install        # pnpm install — dev tooling only (Biome, tsup, tsx, TypeScript, Vitest)
make ts-typecheck      # tsc --noEmit
make ts-lint           # biome check
make ts-format-check   # biome format (check only)
make ts-test           # vitest run
make ts-test-cov       # vitest run --coverage
make ts-build          # tsup → dist (ESM + CJS + .d.ts)
```

The base install is deliberately minimal: the build/typecheck/test/lint toolchain is all that's
needed to develop the core, and every dev dependency supports **Node ≥ 18**, so the CI matrix
(Node 18/20/22) installs cleanly. Provider/framework SDKs are optional `peerDependencies` and are
installed on demand (below) — several (e.g. `@aws-sdk/client-bedrock-runtime`, `@langchain/*`)
require Node ≥ 20, so they're kept out of the base install.

### Installing provider SDKs and agent frameworks (on demand)

Like the Python SDK's `uv sync --extra <name>`, the SDKs are installed on demand rather than as part
of the base install (also gated by `auto-install-peers=false` in `.npmrc`):

```bash
make ts-install-base                # LLM provider SDKs + LangChain glue (to run the examples)
make ts-install-langgraph           # @langchain/langgraph + glue
make ts-install-langchain           # langchain + glue
make ts-install-deepagents          # deepagents
make ts-install-claude-agent-sdk    # @anthropic-ai/claude-agent-sdk
make ts-install-openai-agents       # @openai/agents
make ts-install-vercel-ai           # ai + @ai-sdk/openai
```

Each target adds its packages to `devDependencies`. Install only what you're working with —
unlike Python's shared environment, JS frameworks coexist in nested `node_modules` without
conflicting. To reset to a clean base: `git checkout -- package.json pnpm-lock.yaml && make ts-install`.

Tests mock HTTP via a `fetch` stub (the analog of Python's `respx`) and never hit a real backend.

## See also

- [`examples/`](./examples) — runnable end-to-end examples per provider, framework, and manual method.
- The Python SDK in [`../python`](../python) — the reference implementation this mirrors.
