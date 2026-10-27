# Native integrations

Install `pandaprobe @everruns/bashkit@0.19.0 fs-native-extensions@1.5.1`, then your
framework and model provider. The tested native fixture uses Node ≥22 and these
versions; base tracing retains its broader optional-peer ranges for older users.
New framework peer declarations are deliberately unrestricted so adding Reflect
does not force tracing-only consumers to upgrade their frameworks. Those install
ranges do not establish Reflect compatibility: use the tested versions below.
The packed-package checks also exercise existing tracing with AI SDK 5.0.100.

| Framework | Tested dependencies | Factory and options |
| --- | --- | --- |
| LangChain | langchain 1.5.19, @langchain/core 1.3.0, @langchain/langgraph 1.5.2 | createAgent, `{model, systemPrompt, tools, checkpointer}` |
| LangGraph | @langchain/langgraph 1.5.2, @langchain/core 1.3.0 | createReactAgent from prebuilt, `{llm, prompt, tools, checkpointer}` |
| DeepAgents | deepagents 1.15.1 and the LangChain versions above | createDeepAgent, `{model, systemPrompt, tools, checkpointer}` |
| OpenAI Agents | @openai/agents 0.21.0 | Agent class, `{name, model, instructions, tools}` |
| Claude Agent SDK | @anthropic-ai/claude-agent-sdk 0.3.310, zod 4.3.6 | wrapped query function, native query options |
| Vercel AI | ai 6.0.116 (pinned fixture), 7.0.154 (local verification) | ToolLoopAgent class, `{model, instructions, tools}` |

The committed `test/native/package-lock.json` pins the complete tested combination.
The Vercel tracing follow-up also verifies the locally installed AI SDK 7.0.154
with mocked providers, including the complete Reflect loops. Native dependencies
are installed separately from the Node 18-compatible base dev dependencies:

```sh
npm ci --prefix test/native --ignore-scripts
pnpm test:native
```

## LangChain / LangGraph / DeepAgents

```ts
import { Client } from "pandaprobe";
import { Reflect as ReflectRuntime } from "pandaprobe/reflect";
import { LangChainCallbackHandler } from "pandaprobe/integrations/langchain";
import { createAgent } from "langchain";
import { MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";

const client = new Client();
const reflect = new ReflectRuntime(client);
try {
  const agent = await reflect.agent(createAgent, {
    framework: "langchain",
    options: {
      model: new ChatOpenAI({ model: process.env.MODEL!, reasoning: { effort: "none" } }),
      systemPrompt: "Answer the user's question using evidence.",
      tools: [], checkpointer: new MemorySaver(),
    },
    invokeOptions: { callbacks: [new LangChainCallbackHandler({ client })] },
  });
  console.log(await reflect.run(agent, "What is 2 + 2?", { sessionId: "demo" }));
} finally { await reflect.close(); await client.shutdown(); }
```

Install `@langchain/openai@1.7.1` for this example (also pinned in the native test fixture). Substitute `createReactAgent` from
`@langchain/langgraph/prebuilt`, `framework: "langgraph"`, and `llm`/`prompt` for
LangGraph. For DeepAgents use `createDeepAgent` from `deepagents`,
`framework: "deepagents"`, with its native default subagents. Business tools stay
native LangChain tools. Do not also add `reflect_bash`; it is supplied automatically.
Existing saved checkpoints receive only new feedback. Absent/cleared history is
reconstructed from the native result; tool replies are not appended twice. Custom
precompiled graphs need explicit `Invocation` integration; arbitrary graph state
and tool nodes cannot be invisibly retrofitted.

## OpenAI Agents

```ts
import { Agent, addTraceProcessor } from "@openai/agents";
import { OpenAIAgentsAdapter } from "pandaprobe/integrations/openai-agents";
// client and reflect are constructed/closed as above.
addTraceProcessor(new OpenAIAgentsAdapter({ client }).createTraceProcessor());
const agent = await reflect.agent(Agent, {
  framework: "openai-agents",
  options: { name: "Host", model: process.env.MODEL!, instructions: "Help the user.", tools: [] },
});
const result = await reflect.run(agent, "What is 2 + 2?", { sessionId: "demo" });
console.log(result.finalOutput);
```

The adapter clones invocation options, preserving dynamic instructions and native
results. Reflection offers only the shell, without handoffs/MCP domain servers.
With `invokeOptions.session`, only feedback is appended to the session; otherwise
the completed result's history is used. Validation rejects persistent session,
conversation, previous-response and automatic-previous-response options. Build a
fresh replay agent without these options.

## Claude Agent SDK

```ts
import { query } from "@anthropic-ai/claude-agent-sdk";
import { wrapClaudeAgentQuery } from "pandaprobe/integrations/claude-agent-sdk";
const agent = await reflect.agent(wrapClaudeAgentQuery(query, { client }), {
  framework: "claude-agent-sdk",
  options: { model: process.env.MODEL!, systemPrompt: "Help the user.", tools: [], maxTurns: 12 },
});
const messages = await reflect.run(agent, "What is 2 + 2?", { sessionId: "demo" });
```

The supplied tool is `mcp__pandaprobe_reflect__reflect_bash`. Native MCP dispatch is
bound directly to the execution. Reflection resumes the completed session with
only that server/tool; validation clears resume/continue/session options. Native
`is_error` and non-success result subtypes throw with provider details. The SDK
closes its query and Reflect server, not host-owned task servers. Configure model
and thinking settings for your account; there is no hardcoded repair model.

Like Python, the reflection query uses `tools: []`, only Reflect's MCP server,
`strictMcpConfig: true`, and its qualified tool in `allowedTools`. It does not
add a redundant `canUseTool` callback: native allow rules run before that callback,
which caused `CLAUDE_SDK_CAN_USE_TOOL_SHADOWED` in SDK 0.3.311. Host-supplied
callbacks and hooks are preserved. See the native
[permission evaluation order](https://code.claude.com/docs/en/agent-sdk/permissions).

## Vercel AI SDK

```ts
import { ToolLoopAgent, wrapLanguageModel } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { pandaProbeMiddleware, wrapVercelAgent } from "pandaprobe/integrations/vercel-ai";
const model = wrapLanguageModel({
  model: createOpenAI()(process.env.MODEL!), middleware: pandaProbeMiddleware(),
});
const agent = await reflect.agent(wrapVercelAgent(ToolLoopAgent, { client }), {
  framework: "vercel-ai", options: { model, instructions: "Help the user.", tools: {} },
});
const result = await reflect.run(agent, "What is 2 + 2?", { sessionId: "demo" });
console.log(result.text);
```

`wrapVercelAgent` accepts a native ToolLoopAgent constructor or an existing instance
and supports AI SDK 6 and 7. It emits one `CHAIN → AGENT → LLM → TOOL` tree per
invocation using native lifecycle callbacks. Tool spans describe actual execution,
including results and errors, rather than merely the model's proposed calls.
An existing ambient trace is reused; otherwise the adapter owns the trace.

`pandaProbeMiddleware` remains the low-level model integration (including AI SDK 5).
Inside an instrumented agent it enriches the same LLM spans with exact provider
timing and individual retry attempts, without duplicating them. Without middleware,
LLM timings are bounded by native step callbacks and identified as such in metadata.
Standalone model middleware intentionally emits only LLM spans.

The live examples use both layers. `prepareCall`/`prepareStep` and native callbacks are preserved,
with memory and reflection restrictions applied at their supported boundaries.
Reflection continues completed response messages; fresh validation has no prior
conversation. Stream errors/aborts are failures, not successful final text.
Consume the native stream or pass an AbortSignal to cancel it. As with Vercel's
native stream API, abandoning every consumer without cancellation does not signal
completion. Cancelling one exposed stream closes its trace with a cancellation
error; Vercel's other tee'd consumers can still own the underlying source.

All adapters retain the host model and existing tracing integration. Framework
permissions cannot stop arbitrary side effects in developer-supplied hooks; those
hooks are trusted host code. For host conversation management across ordinary
turns, keep native sessions/checkpoints or pass full native history yourself. The
shared live examples demonstrate each framework's supported approach.

## Python-matching live example configuration

The shared example factories default to the same model settings and lifecycle as Python.
The configurable SDK integration itself does not impose these example settings.

| Framework | Example configuration and necessary native translation |
| --- | --- |
| LangChain / LangGraph | ChatOpenAI with `reasoning: { effort: "none" }`, no retries, 60-second provider timeout; native checkpointer; recursion limit 32 |
| DeepAgents | The same ChatOpenAI settings, native StateBackend and default subagents; recursion limit 40; the same scratch-space instruction appended to the prompt |
| OpenAI Agents | Agent name `Northstar Support`, maxTurns 16; native MemorySession for task conversations and no session for validation. This replaces Python's SQLiteSession used as an in-memory conversation store. |
| Claude Agent SDK | A fresh working directory, thinking disabled, effort low, no built-in tools or settings sources, maxTurns 16, strict MCP configuration and the native business-tool server |
| Vercel AI | Native ToolLoopAgent using the same business workload; no Python-native framework counterpart |

`examples/langgraph/05_reflect_failure.ts` explicitly uses a **120-second model
request timeout**. This is a local exception to Python's 60-second setting: a live
candidate replay exhausted that deadline on its first LLM request, before any
business tool ran. The extra time accommodates slower responses; it cannot
guarantee provider availability. Other examples retain their existing defaults.
Automatic retries remain disabled, and the 360-second task and 600-second
validation limits and all acceptance checks are unchanged. A request that still
times out leaves validation pending and the example incomplete; it does not reject
the proposal or count missing evidence as a successful verification.

In the JS ChatOpenAI constructor, use `reasoning: { effort: "none" }` to match
Python's `reasoning_effort="none"`. The deprecated `reasoningEffort` call option
is ignored at construction by the tested provider client, leaving the HTTP
request without `reasoning_effort`. The native HTTP regression tests exercise
the real provider client with mocked responses, including task, reflection and
validation tool calls.

The request regression was also verified with DeepAgents 1.16.0, LangChain 1.5.20,
LangGraph 1.5.3 and @langchain/openai 1.7.1, as installed by the framework Make target.

Python timeouts expressed in seconds become milliseconds only at JS APIs requiring
milliseconds. Python async context managers become Promise/callback scopes with
AsyncLocalStorage and AbortSignals. JS framework middleware, schemas, session IDs
and MCP objects follow their native APIs; they do not add Reflect policy. Python
and JS Claude Agent SDKs manage their own native subprocess runtime; Reflect does
not introduce a CLI or invoke Python. `examples/reflect/_support/cli.ts` is only
example argument parsing and startup.
