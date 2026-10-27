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

## Reflect: learn guidance from independently evaluated work

PandaProbe Reflect adds **Detect → Notify → Reflect → Validate** to your existing
agent. The same agent uses one `reflect_bash` tool in an isolated filesystem;
independent paired outcomes decide whether its proposals become accepted guidance.
Core has no delivery checker and no default 100-evaluation quota.

```sh
npm install pandaprobe @everruns/bashkit@0.19.0 fs-native-extensions@1.5.1
```

```ts
import { Client } from "pandaprobe";
import { Reflect as ReflectRuntime } from "pandaprobe/reflect";
import { PlatformClient } from "pandaprobe/api";
const client = new Client();
const reflect = new ReflectRuntime(client);
const api = new PlatformClient(client); // separately reusable native platform API
// Attach your native agent with reflect.agent(...), then await reflect.run(...).
await api.close();
await reflect.close();
await client.shutdown();
```

Tracing imports retain zero required runtime dependencies. Reflect's shell/lock
packages and framework SDKs are optional; native framework examples use Node ≥22.
All configuration timeouts use **seconds**. Reading a rule, proposing a change,
accepting it, and succeeding on a new task remain separate recorded facts.

See the [Reflect guide](examples/reflect/README.md),
[native quickstarts](examples/reflect/native.md),
[workspace and limits](examples/reflect/workspace.md),
[credential-free complete loop](examples/reflect/offline-demo.ts),
[multi-turn support](examples/reflect/live_support.md), and
[controlled failure example](examples/reflect/live_failure.md).

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

Vercel AI SDK 6/7 agents use `wrapVercelAgent` for `CHAIN → AGENT → LLM → TOOL`
traces. The separate model middleware records exact provider-call timing and also
supports standalone model calls:

```ts
import { ToolLoopAgent, wrapLanguageModel } from "ai";
import { pandaProbeMiddleware, wrapVercelAgent } from "pandaprobe/integrations/vercel-ai";

const agent = wrapVercelAgent(new ToolLoopAgent({
  model: wrapLanguageModel({ model, middleware: pandaProbeMiddleware() }),
  tools,
}));
await agent.generate({ prompt: "Help with this task." });
```

See the [Vercel Reflect example](examples/reflect/native.md#vercel-ai-sdk) for
constructor wrapping and the [agent example](examples/vercel-ai/agent.ts) for a
standalone native tool loop. Model middleware alone remains an LLM-level integration.

## Session / user grouping

```ts
import { session, user, setSession } from "pandaprobe";

await session("conv-123", async () => {
  await runAgent(query); // traces inherit session_id
});
```

## Development

```bash
make ts-install        # pnpm install — dev tooling and Reflect shell/lock test runtimes
make ts-install-reflect # explicitly install the pinned Reflect shell/lock runtimes
make ts-typecheck      # tsc --noEmit
make ts-lint           # biome check
make ts-format-check   # biome format (check only)
make ts-test           # vitest run
make ts-test-cov       # vitest run --coverage
make ts-build          # tsup → dist (ESM + CJS + .d.ts)
```

The base install includes the build/typecheck/test/lint toolchain and Reflect's shell/lock
runtimes for core tests. Every dev dependency supports **Node ≥ 18**, so the CI matrix
(Node 18/20/22) installs cleanly. Provider/framework SDKs are optional `peerDependencies` and are
installed on demand (below) — several (e.g. `@aws-sdk/client-bedrock-runtime`, `@langchain/*`)
require Node ≥ 20, so they're kept out of the base install.

From `libraries/typescript`, use `make install-reflect`; from the repository root,
use `make ts-install-reflect`. Like Python's development extra, the default development
install already includes Reflect. The separate `install-reflect` recipe explicitly installs
`@everruns/bashkit@0.19.0` and `fs-native-extensions@1.5.1` with `pnpm add -D --save-exact`,
matching the versions in `package.json`. Add your chosen framework/provider separately for live examples.

### Installing provider SDKs and agent frameworks (on demand)

Like the Python SDK's `uv sync --extra <name>`, the SDKs are installed on demand rather than as part
of the base install (also gated by `auto-install-peers=false` in `.npmrc`):

```bash
make ts-install-base                # LLM provider SDKs + LangChain glue (to run the examples)
make ts-install-langgraph           # @langchain/langgraph + glue
make ts-install-langchain           # langchain + glue
make ts-install-deepagents          # deepagents + OpenAI/LangChain runtime peers
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
