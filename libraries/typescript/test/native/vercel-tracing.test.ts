import { expect, it, vi } from "vitest";
import { Client } from "../../src/client.js";
import { pandaProbeMiddleware, wrapVercelAgent } from "../../src/integrations/vercel-ai/index.js";
import { optionalModule } from "../../src/reflect/integrations.js";
import type { TraceData } from "../../src/schemas.js";
import { fixture } from "../reflect/helpers.js";

async function sdk(version: string) {
  if (version === "6") return import("./node_modules/ai/dist/index.mjs");
  return optionalModule("ai");
}
const usage = { inputTokens: { total: 10 }, outputTokens: { total: 5 } };
const finishReason = { unified: "stop", raw: "stop" };
function capture() {
  const f = fixture({ evaluationSampleRate: 0 });
  const traces: TraceData[] = [];
  const original = f.client.logTrace.bind(f.client);
  vi.spyOn(f.client, "logTrace").mockImplementation((trace) => {
    traces.push(trace);
    original(trace);
  });
  return { ...f, traces };
}
function assertTree(trace: TraceData, models: number, tools: number) {
  const chain = trace.spans.filter((s) => s.kind === "CHAIN");
  const agent = trace.spans.filter((s) => s.kind === "AGENT");
  const llms = trace.spans.filter((s) => s.kind === "LLM");
  const actions = trace.spans.filter((s) => s.kind === "TOOL");
  expect(chain).toHaveLength(1);
  expect(agent).toHaveLength(1);
  expect(agent[0]!.parentSpanId).toBe(chain[0]!.spanId);
  expect(llms).toHaveLength(models);
  expect(actions).toHaveLength(tools);
  for (const llm of llms) expect(llm.parentSpanId).toBe(agent[0]!.spanId);
  for (const tool of actions) expect(llms.map((s) => s.spanId)).toContain(tool.parentSpanId);
  for (const span of trace.spans) expect(span.endedAt).not.toBeNull();
}

it.each(["6", "installed"])(
  "AI SDK %s records real multi-step tools with exact model spans and host callbacks",
  async (version) => {
    const ai = await sdk(version);
    const { MockLanguageModelV3 } = await optionalModule("ai/test");
    const f = capture();
    const settingsHook = vi.fn();
    const methodHook = vi.fn();
    const outputs: number[] = [];
    let turn = 0;
    const model = ai.wrapLanguageModel({
      model: new MockLanguageModelV3({
        doGenerate: async () => ({
          content:
            turn++ === 0
              ? [1, 2].map((n) => ({
                  type: "tool-call",
                  toolCallId: `call-${n}`,
                  toolName: "double",
                  input: JSON.stringify({ n }),
                }))
              : [{ type: "text", text: "All done." }],
          finishReason: turn === 1 ? { unified: "tool-calls", raw: "tool_calls" } : finishReason,
          usage,
          warnings: [],
        }),
      }),
      middleware: pandaProbeMiddleware(),
    });
    const tools = {
      double: ai.tool({
        description: "Double an integer",
        inputSchema: ai.jsonSchema({ type: "object", properties: { n: { type: "integer" } }, required: ["n"] }),
        execute: async ({ n }: { n: number }) => {
          await new Promise((resolve) => setTimeout(resolve, n === 1 ? 10 : 1));
          outputs.push(n * 2);
          return n * 2;
        },
      }),
    };
    const agent = wrapVercelAgent(new ai.ToolLoopAgent({ id: "worker", model, tools, onStepFinish: settingsHook }), {
      client: f.client,
    });
    expect(wrapVercelAgent(agent)).toBe(agent);
    expect(agent.tools).toBe(tools);
    const result = await agent.generate({
      messages: [
        { role: "user", content: "old question" },
        { role: "assistant", content: "old answer" },
        { role: "user", content: "Double 1 and 2." },
      ],
      onStepFinish: methodHook,
    });
    expect(result.text).toBe("All done.");
    expect(outputs.sort()).toEqual([2, 4]);
    expect(settingsHook).toHaveBeenCalledTimes(2);
    expect(methodHook).toHaveBeenCalledTimes(2);
    expect(f.traces).toHaveLength(1);
    const trace = f.traces[0]!;
    assertTree(trace, 2, 2);
    expect(trace.input).toEqual({ messages: [{ role: "user", content: "Double 1 and 2." }] });
    expect(trace.output).toEqual({ messages: [{ role: "assistant", content: "All done." }] });
    for (const span of trace.spans.filter((s) => s.kind === "LLM")) {
      expect(span.metadata.timing_source).toBe("model_middleware");
      expect(span.tokenUsage).toMatchObject({ prompt_tokens: 10, completion_tokens: 5 });
    }
    for (const span of trace.spans.filter((s) => s.kind === "TOOL")) {
      expect(span.output).toBe((span.input as any).n * 2);
      expect(span.status).toBe("OK");
    }
    expect((trace.spans.find((s) => s.kind === "LLM")!.output as any).messages[0].tool_calls).toHaveLength(2);
  },
);

it.each(["success", "error", "cancel"])("Vercel native streaming closes the complete tree on %s", async (mode) => {
  const ai = await sdk("installed");
  const { MockLanguageModelV3 } = await optionalModule("ai/test");
  const f = capture();
  const model = ai.wrapLanguageModel({
    model: new MockLanguageModelV3({
      doStream: async () => ({
        stream: new ReadableStream({
          start(c) {
            c.enqueue({ type: "stream-start", warnings: [] });
            c.enqueue({ type: "text-start", id: "text" });
            c.enqueue({ type: "text-delta", id: "text", delta: "Hello" });
            if (mode === "cancel") return;
            if (mode === "error") c.enqueue({ type: "error", error: new Error("stream failed") });
            else {
              c.enqueue({ type: "text-end", id: "text" });
              c.enqueue({ type: "finish", finishReason, usage });
            }
            c.close();
          },
        }),
      }),
    }),
    middleware: pandaProbeMiddleware(),
  });
  const agent = wrapVercelAgent(new ai.ToolLoopAgent({ model }), { client: f.client });
  const controller = new AbortController();
  const result = await agent.stream({ prompt: "Hello", abortSignal: controller.signal });
  if (mode === "cancel") {
    const reader = result.textStream.getReader();
    await reader.read();
    // Cancel the consumer; then stop the native upstream explicitly as AI SDK
    // stream getters tee the underlying source for other possible consumers.
    const cancellation = reader.cancel();
    controller.abort();
    await cancellation;
  } else {
    for await (const part of result.fullStream) {
      /* consume native events, including error values */
    }
  }
  expect(f.traces).toHaveLength(1);
  assertTree(f.traces[0]!, 1, 0);
  expect(f.traces[0]!.status).toBe(mode === "success" ? "COMPLETED" : "ERROR");
});

it("Vercel tool errors and concurrent invocations keep distinct parents and original results", async () => {
  const ai = await sdk("installed");
  const { MockLanguageModelV3 } = await optionalModule("ai/test");
  const f = capture();
  const agent = wrapVercelAgent(
    new ai.ToolLoopAgent({
      model: new MockLanguageModelV3({
        doGenerate: async () => ({
          content: [{ type: "tool-call", toolCallId: "same-id", toolName: "fail", input: "{}" }],
          finishReason: { unified: "tool-calls", raw: "tool_calls" },
          usage,
          warnings: [],
        }),
      }),
      stopWhen: ai.stepCountIs(1),
      tools: {
        fail: ai.tool({
          inputSchema: ai.jsonSchema({ type: "object", properties: {} }),
          execute: () => {
            throw new Error("business failed");
          },
        }),
      },
    }),
    { client: f.client },
  );
  const results = await Promise.all([agent.generate({ prompt: "one" }), agent.generate({ prompt: "two" })]);
  expect(results).toHaveLength(2);
  expect(f.traces).toHaveLength(2);
  for (const trace of f.traces) {
    assertTree(trace, 1, 1);
    expect(trace.spans.find((s) => s.kind === "TOOL")).toMatchObject({ status: "ERROR", error: "business failed" });
  }
  expect(new Set(f.traces.flatMap((t) => t.spans.map((s) => s.spanId))).size).toBe(8);
});

it.each(["6", "installed"])("AI SDK %s traces streaming tool execution through completion", async (version) => {
  const ai = await sdk(version);
  const { MockLanguageModelV3 } = await optionalModule("ai/test");
  const f = capture();
  let calls = 0;
  const finish = vi.fn();
  const model = ai.wrapLanguageModel({
    model: new MockLanguageModelV3({
      doStream: async () => ({
        stream: new ReadableStream({
          start(c) {
            const first = calls++ === 0;
            c.enqueue({ type: "stream-start", warnings: [] });
            if (first) c.enqueue({ type: "tool-call", toolCallId: "tool", toolName: "lookup", input: "{}" });
            else {
              c.enqueue({ type: "text-start", id: "text" });
              c.enqueue({ type: "text-delta", id: "text", delta: "Found." });
              c.enqueue({ type: "text-end", id: "text" });
            }
            c.enqueue({
              type: "finish",
              finishReason: first ? { unified: "tool-calls", raw: "tool_calls" } : finishReason,
              usage,
            });
            c.close();
          },
        }),
      }),
    }),
    middleware: pandaProbeMiddleware(),
  });
  const Traced = wrapVercelAgent(ai.ToolLoopAgent, { client: f.client });
  const agent = new Traced({
    model,
    tools: {
      lookup: ai.tool({ inputSchema: ai.jsonSchema({ type: "object", properties: {} }), execute: () => "record" }),
    },
    onFinish: finish,
  });
  const result = await agent.stream({ prompt: "Find it" });
  let output = "";
  for await (const delta of result.textStream) output += delta;
  expect(output).toBe("Found.");
  expect(finish).toHaveBeenCalledTimes(1);
  expect(f.traces).toHaveLength(1);
  assertTree(f.traces[0]!, 2, 1);
  expect(f.traces[0]!.spans.find((s) => s.kind === "TOOL")!.output).toBe("record");
});

it.each(["provider", "host callback"])("Vercel preserves native %s error handling and closes runs", async (mode) => {
  const ai = await sdk("installed");
  const { MockLanguageModelV3 } = await optionalModule("ai/test");
  const f = capture();
  const error = new Error(`${mode} failed`);
  const model = ai.wrapLanguageModel({
    model: new MockLanguageModelV3({
      doGenerate: async () => {
        if (mode === "provider") throw error;
        return { content: [{ type: "text", text: "done" }], usage, finishReason, warnings: [] };
      },
    }),
    middleware: pandaProbeMiddleware(),
  });
  const agent = wrapVercelAgent(
    new ai.ToolLoopAgent({
      model,
      maxRetries: 0,
      onStepFinish: () => {
        if (mode === "host callback") throw error;
      },
    }),
    { client: f.client },
  );
  if (mode === "provider") await expect(agent.generate({ prompt: "x" })).rejects.toBe(error);
  else expect((await agent.generate({ prompt: "x" })).text).toBe("done"); // SDK 7 reports callback errors but continues.
  expect(f.traces).toHaveLength(1);
  assertTree(f.traces[0]!, 1, 0);
  expect(f.traces[0]!.status).toBe(mode === "provider" ? "ERROR" : "COMPLETED");
  expect(f.traces[0]!.spans.find((s) => s.kind === "AGENT")!.error).toBe(mode === "provider" ? error.message : null);
});

it("Vercel records a proposed tool call without inventing a tool execution", async () => {
  const ai = await sdk("installed");
  const { MockLanguageModelV3 } = await optionalModule("ai/test");
  const f = capture();
  const agent = wrapVercelAgent(
    new ai.ToolLoopAgent({
      model: new MockLanguageModelV3({
        doGenerate: async () => ({
          content: [{ type: "tool-call", toolCallId: "client-call", toolName: "client_action", input: "{}" }],
          finishReason: { unified: "tool-calls", raw: "tool_calls" },
          usage,
          warnings: [],
        }),
      }),
      tools: { client_action: ai.tool({ inputSchema: ai.jsonSchema({ type: "object", properties: {} }) }) },
    }),
    { client: f.client },
  );
  await agent.generate({ prompt: "Use the client tool." });
  expect(f.traces).toHaveLength(1);
  assertTree(f.traces[0]!, 1, 0);
  expect((f.traces[0]!.spans.find((s) => s.kind === "LLM")!.output as any).messages[0].tool_calls).toHaveLength(1);
});

it("Vercel binds cancellation capture to its execution and ignores late provider results", async () => {
  const ai = await sdk("installed");
  const { MockLanguageModelV3 } = await optionalModule("ai/test");
  const f = capture();
  let release!: (value: any) => void;
  let started!: () => void;
  const ready = new Promise<void>((resolve) => {
    started = resolve;
  });
  const model = ai.wrapLanguageModel({
    model: new MockLanguageModelV3({
      doGenerate: () => {
        started();
        return new Promise((resolve) => {
          release = resolve;
        });
      },
    }),
    middleware: pandaProbeMiddleware(),
  });
  const agent = wrapVercelAgent(new ai.ToolLoopAgent({ model, maxRetries: 0 }), { client: f.client });
  const controller = new AbortController();
  const pending = f.runtime
    .execution({ sessionId: "cancel", executionId: "cancel" }, () =>
      agent.generate({ prompt: "x", abortSignal: controller.signal }),
    )
    .catch(() => undefined);
  await ready;
  controller.abort(new Error("Stop the request")); // outside the Reflect async scope
  expect(f.traces).toHaveLength(1);
  const before = JSON.stringify(f.traces[0]!.toApiDict());
  release({ content: [{ type: "text", text: "Late response" }], usage, finishReason, warnings: [] });
  await pending;
  expect(f.traces).toHaveLength(1);
  expect(JSON.stringify(f.traces[0]!.toApiDict())).toBe(before);
  expect(f.runtime.report("cancel").trace_ids).toEqual([f.traces[0]!.traceId]);
  expect(f.traces[0]!.status).toBe("ERROR");
});

it("Vercel joins an explicit trace, preserves prepareCall, and is transparent when disabled", async () => {
  const ai = await sdk("installed");
  const { MockLanguageModelV3 } = await optionalModule("ai/test");
  const f = capture();
  const prepareCall = vi.fn((value: any) => value);
  const settings = {
    model: new MockLanguageModelV3({
      doGenerate: async () => ({ content: [{ type: "text", text: "done" }], usage, finishReason, warnings: [] }),
    }),
    prepareCall,
  };
  const native = new ai.ToolLoopAgent(settings);
  const agent = wrapVercelAgent(native, { client: f.client });
  await f.client.trace("host").run((trace) => trace.span("outer").run(() => agent.generate({ prompt: "x" })));
  expect(f.traces).toHaveLength(1);
  const spans = f.traces[0]!.spans;
  expect(spans.find((s) => s.kind === "CHAIN")!.parentSpanId).toBe(spans.find((s) => s.name === "outer")!.spanId);
  expect(prepareCall).toHaveBeenCalledTimes(1);
  const disabled = new Client({ enabled: false });
  const log = vi.spyOn(disabled, "logTrace");
  try {
    const silent = wrapVercelAgent(native, { client: disabled });
    expect((await silent.generate({ prompt: "x" })).text).toBe("done");
    expect(log).not.toHaveBeenCalled();
  } finally {
    await disabled.shutdown();
  }
});

it.each(["text", "consumeStream"])("Vercel closes stream errors when consumed through %s", async (access) => {
  const ai = await sdk("installed");
  const { MockLanguageModelV3 } = await optionalModule("ai/test");
  const f = capture();
  const model = ai.wrapLanguageModel({
    model: new MockLanguageModelV3({
      doStream: async () => ({
        stream: new ReadableStream({
          start(c) {
            c.enqueue({ type: "error", error: new Error("stream failed") });
            c.close();
          },
        }),
      }),
    }),
    middleware: pandaProbeMiddleware(),
  });
  const agent = wrapVercelAgent(new ai.ToolLoopAgent({ model }), { client: f.client });
  const result = await agent.stream({ prompt: "x" });
  try {
    if (access === "text") await result.text;
    else await result.consumeStream();
  } catch {
    /* preserve native semantics */
  }
  expect(f.traces).toHaveLength(1);
  expect(f.traces[0]!.status).toBe("ERROR");
  assertTree(f.traces[0]!, 1, 0);
});
