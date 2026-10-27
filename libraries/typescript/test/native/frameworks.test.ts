import { randomUUID } from "node:crypto";
import { expect, it, vi } from "vitest";
import { wrapClaudeAgentQuery } from "../../src/integrations/claude-agent-sdk/index.js";
import { LangChainCallbackHandler } from "../../src/integrations/langchain/index.js";
import { OpenAIAgentsAdapter } from "../../src/integrations/openai-agents/adapter.js";
import { pandaProbeMiddleware, wrapVercelAgent } from "../../src/integrations/vercel-ai/index.js";
import { type Invocation, Outcome, ReplayCase } from "../../src/reflect/index.js";
import { optionalModule } from "../../src/reflect/integrations.js";
import { WRITE_RULE, fixture } from "../reflect/helpers.js";
function persist(f: ReturnType<typeof fixture>) {
  const original = f.client.logTrace.bind(f.client);
  vi.spyOn(f.client, "logTrace").mockImplementation((trace) => {
    f.server.traces.set(trace.traceId, {
      ...trace.toApiDict(),
      project_id: randomUUID(),
      spans: trace.spans.map((span) => ({ ...span.toApiDict(), trace_id: trace.traceId, status: span.status })),
    });
    original(trace);
  });
}
it.each(
  (["langchain", "langgraph", "deepagents"] as const).flatMap((framework) =>
    ["saved", "absent", "cleared", "hostile"].map((checkpoint) => ({ framework, checkpoint })),
  ),
)("$framework uses native loops with $checkpoint history", async ({ framework, checkpoint }) => {
  const f = fixture({ evaluationTimeout: 2, traceVisibilityTimeout: 2 });
  persist(f);
  const { RunnableBinding } = await optionalModule("@langchain/core/runnables");
  const { BaseChatModel } = await optionalModule("@langchain/core/language_models/chat_models");
  const { AIMessage } = await optionalModule("@langchain/core/messages");
  const { tool } = await optionalModule("@langchain/core/tools");
  const { MemorySaver } = await optionalModule("@langchain/langgraph");
  const turns = new Map<string, number>();
  const requests: any[] = [];
  let domainCalls = 0;
  class Model extends BaseChatModel {
    constructor() {
      super({});
    }
    _llmType() {
      return "reflect-fixture";
    }
    bindTools(tools: any[]) {
      return new RunnableBinding({ bound: this, kwargs: { tools }, config: {} });
    }
    async _generate(messages: any[], options: any) {
      const execution = f.runtime.currentExecution!;
      const key = `${execution.id}:${execution.role}`;
      const turn = turns.get(key) ?? 0;
      turns.set(key, turn + 1);
      requests.push({ phase: execution.role, messages, tools: options.tools });
      expect(messages.filter((m) => m.id === "pandaprobe-reflect-active-memory")).toHaveLength(1);
      expect(JSON.stringify(messages)).toContain("domain prompt");
      let name: string | undefined;
      let args: any = {};
      if (execution.role === "reflection") {
        expect(options.tools.map((t: any) => t.name)).toEqual(["reflect_bash"]);
        if (turn === 0 && checkpoint === "hostile") {
          name = "domain";
        } else if (turn === (checkpoint === "hostile" ? 1 : 0)) {
          name = "reflect_bash";
          args = { command: `cat /inbox/evidence.txt; ${WRITE_RULE}` };
        }
      } else if (turn === 0) {
        name = "reflect_bash";
        args = { command: "cat /context.md; grep -r shipping /rules || true" };
      } else if (turn === 1) {
        name = "domain";
        args = {};
      }
      const message = name
        ? new AIMessage({ content: "", tool_calls: [{ id: randomUUID(), name, args, type: "tool_call" }] })
        : new AIMessage("finished");
      return { generations: [{ text: "", message }] };
    }
  }
  const model = new Model();
  const domain = tool(
    async () => {
      domainCalls++;
      return "domain done";
    },
    {
      name: "domain",
      description: "domain action",
      schema: { type: "object", properties: {}, additionalProperties: false },
    },
  );
  const factory =
    framework === "langgraph"
      ? (await optionalModule("@langchain/langgraph/prebuilt")).createReactAgent
      : framework === "deepagents"
        ? (await optionalModule("deepagents")).createDeepAgent
        : (await optionalModule("langchain")).createAgent;
  const checkpointer = new MemorySaver();
  const agent = await f.runtime.agent(factory, {
    framework,
    options: {
      ...(framework === "langgraph"
        ? { llm: model, prompt: "domain prompt" }
        : { model, systemPrompt: "domain prompt" }),
      tools: [domain],
      ...(checkpoint === "absent" ? {} : { checkpointer }),
      ...(framework === "deepagents" ? { subagents: [] } : {}),
    },
    invokeOptions: {
      callbacks: [new LangChainCallbackHandler({ client: f.client })],
      stream: checkpoint === "absent",
    },
  });
  if (checkpoint === "cleared") {
    const invoke = agent.invokeReflect.bind(agent);
    agent.invokeReflect = async (request) => {
      const result = await invoke(request);
      if (request.phase === "task") await checkpointer.deleteThread(request.execution.sessionId);
      return result;
    };
  }
  const result = await f.runtime.run(agent, "refund", { sessionId: "s", executionId: "native" });
  expect(result.messages.at(-1).content).toBe("finished");
  expect(domainCalls).toBe(1);
  expect(f.runtime.report("native").reflection?.status, JSON.stringify(f.runtime.report("native"))).toBe("proposed");
  expect(requests.some((r) => r.phase === "reflection")).toBe(true);
  for (const request of requests) {
    const ids = new Set<string>();
    for (const message of request.messages) {
      for (const call of message.tool_calls ?? []) ids.add(call.id);
      if (message._getType?.() === "tool") expect(ids.has(message.tool_call_id)).toBe(true);
    }
  }
  expect(f.runtime.report("native").trace_ids.length).toBeGreaterThan(0);
});
it("OpenAI Agents runs native multi-turn tools and keeps agent options invocation-local", async () => {
  const f = fixture({ evaluationTimeout: 2, traceVisibilityTimeout: 2 });
  persist(f);
  const sdk = await optionalModule("@openai/agents");
  sdk.setTracingDisabled(false);
  sdk.setTraceProcessors([new OpenAIAgentsAdapter({ client: f.client }).createTraceProcessor()]);
  const turns = new Map<string, number>();
  let domainCalls = 0;
  const model = {
    async getResponse(request: any) {
      const execution = f.runtime.currentExecution!;
      const key = `${execution.id}:${execution.role}`;
      const turn = turns.get(key) ?? 0;
      turns.set(key, turn + 1);
      expect(request.systemInstructions).toContain("domain prompt");
      expect(request.systemInstructions).toContain("Reflect memory");
      let name: string | undefined;
      let args: any = {};
      if (execution.role === "reflection") {
        expect(request.tools.map((t: any) => t.name)).toEqual(["reflect_bash"]);
        if (turn === 0) {
          name = "reflect_bash";
          args = { command: WRITE_RULE };
        }
      } else if (turn === 0) {
        name = "reflect_bash";
        args = { command: "ls /rules" };
      } else if (turn === 1) name = "domain";
      return {
        usage: new sdk.Usage(),
        output: name
          ? [
              {
                type: "function_call",
                callId: randomUUID(),
                name,
                arguments: JSON.stringify(args),
                status: "completed",
              },
            ]
          : [
              {
                type: "message",
                role: "assistant",
                status: "completed",
                content: [{ type: "output_text", text: "finished" }],
              },
            ],
        responseId: randomUUID(),
      };
    },
    getStreamedResponse() {
      throw Error("stream fixture unavailable");
    },
  };
  const domain = sdk.tool({
    name: "domain",
    description: "domain",
    parameters: { type: "object", properties: {}, required: [], additionalProperties: false },
    execute: async () => {
      domainCalls++;
      return "done";
    },
  });
  const agent = await f.runtime.agent(sdk.Agent, {
    framework: "openai-agents",
    options: { name: "host", model, instructions: async () => "domain prompt", tools: [domain] },
  });
  const result = await f.runtime.run(agent, "refund", { sessionId: "s", executionId: "native" });
  expect(result.finalOutput).toBe("finished");
  expect(domainCalls).toBe(1);
  expect(agent.native.tools).toHaveLength(2);
  expect(f.runtime.report("native").reflection?.status, JSON.stringify(f.runtime.report("native"))).toBe("proposed");
  sdk.setTracingDisabled(false);
  sdk.setTraceProcessors([]);
});
it("Vercel ToolLoopAgent preserves tracing middleware across multi-step task and reflection", async () => {
  const f = fixture({ evaluationTimeout: 2, traceVisibilityTimeout: 2 });
  persist(f);
  const ai = await optionalModule("ai");
  const { MockLanguageModelV3 } = await optionalModule("ai/test");
  const turns = new Map<string, number>();
  let domainCalls = 0;
  const requests: any[] = [];
  const base = new MockLanguageModelV3({
    doGenerate: async (request: any) => {
      const e = f.runtime.currentExecution!;
      const key = `${e.id}:${e.role}`;
      const turn = turns.get(key) ?? 0;
      turns.set(key, turn + 1);
      requests.push(request);
      expect(JSON.stringify(request.prompt)).toContain("domain prompt");
      expect(JSON.stringify(request.prompt)).toContain("Reflect memory");
      let name: string | undefined;
      let input: any = {};
      if (e.role === "reflection") {
        expect(request.tools.map((t: any) => t.name)).toEqual(["reflect_bash"]);
        if (turn === 0) {
          name = "reflect_bash";
          input = { command: WRITE_RULE };
        }
      } else if (turn === 0) {
        name = "reflect_bash";
        input = { command: "cat /context.md" };
      } else if (turn === 1) name = "domain";
      return {
        content: name
          ? [{ type: "tool-call", toolCallId: randomUUID(), toolName: name, input: JSON.stringify(input) }]
          : [{ type: "text", text: "finished" }],
        finishReason: { unified: name ? "tool-calls" : "stop", raw: name ? "tool_calls" : "stop" },
        usage: {
          inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 5, text: 5, reasoning: 0 },
        },
        warnings: [],
      };
    },
  });
  const model = ai.wrapLanguageModel({ model: base, middleware: pandaProbeMiddleware() });
  const agent = await f.runtime.agent(wrapVercelAgent(ai.ToolLoopAgent, { client: f.client }), {
    framework: "vercel-ai",
    options: {
      model,
      instructions: "domain prompt",
      tools: {
        domain: ai.tool({
          description: "domain",
          inputSchema: ai.jsonSchema({ type: "object", properties: {}, additionalProperties: false }),
          execute: async () => {
            domainCalls++;
            return "done";
          },
        }),
      },
    },
  });
  const result = await f.runtime.run(agent, "refund", { sessionId: "s", executionId: "native" });
  expect(result.text).toBe("finished");
  expect(domainCalls).toBe(1);
  expect(f.runtime.report("native").reflection?.status, JSON.stringify(f.runtime.report("native"))).toBe("proposed");
  expect(requests.length).toBe(5);
  const traces = [...f.server.traces.values()];
  expect(traces).toHaveLength(2);
  for (const trace of traces) {
    expect(trace.spans.filter((s: any) => s.kind === "CHAIN")).toHaveLength(1);
    expect(trace.spans.filter((s: any) => s.kind === "AGENT")).toHaveLength(1);
    expect(trace.spans.some((s: any) => s.kind === "TOOL")).toBe(true);
  }
});
it("Claude uses native MCP dispatch, completed-session continuation and native error values", async () => {
  const f = fixture({ evaluationTimeout: 2, traceVisibilityTimeout: 2 });
  persist(f);
  const { Client } = await import("./node_modules/@modelcontextprotocol/sdk/dist/esm/client/index.js");
  const { InMemoryTransport } = await import("./node_modules/@modelcontextprotocol/sdk/dist/esm/inMemory.js");
  let calls = 0;
  const hostServer = { type: "http", url: "http://business.invalid/mcp" };
  const hostOptions = {
    model: "host-model",
    systemPrompt: "domain prompt",
    tools: ["Read"],
    mcpServers: { business: hostServer },
    allowedTools: ["mcp__business__submit"],
  };
  const query = async function* (args: any) {
    calls++;
    const execution = f.runtime.currentExecution!;
    expect(args.options.systemPrompt).toContain("domain prompt");
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "fixture", version: "1" });
    await args.options.mcpServers.pandaprobe_reflect.instance.connect(serverTransport);
    await client.connect(clientTransport);
    try {
      if (execution.role === "reflection") {
        expect(args.options.resume).toBe("native-session");
        expect(args.options.tools).toEqual([]);
        expect(args.options.strictMcpConfig).toBe(true);
        expect(args.options.allowedTools).toEqual(["mcp__pandaprobe_reflect__reflect_bash"]);
        expect(Object.keys(args.options.mcpServers)).toEqual(["pandaprobe_reflect"]);
        // Do not install a callback that allowedTools would bypass and warn about.
        expect(args.options.canUseTool).toBeUndefined();
        await client.callTool({ name: "reflect_bash", arguments: { command: WRITE_RULE } });
      } else {
        expect(args.options.mcpServers.business).toBe(hostServer);
        expect(args.options.tools).toEqual(["Read"]);
        expect(args.options.allowedTools).toContain("mcp__business__submit");
        await client.callTool({ name: "reflect_bash", arguments: { command: "ls /rules" } });
      }
      yield { type: "assistant", message: { content: [{ type: "text", text: "finished" }] } };
      yield { type: "result", subtype: "success", is_error: false, result: "finished", session_id: "native-session" };
    } finally {
      await client.close();
    }
  };
  const agent = await f.runtime.agent(wrapClaudeAgentQuery(query, { client: f.client }), {
    framework: "claude-agent-sdk",
    options: hostOptions,
  });
  const result = await f.runtime.run(agent, "refund", { sessionId: "s", executionId: "native" });
  expect(result.at(-1).result).toBe("finished");
  expect(calls).toBe(2);
  expect(hostOptions.allowedTools).toEqual(["mcp__business__submit"]);
  expect(hostOptions.tools).toEqual(["Read"]);
  expect(hostOptions.mcpServers).toEqual({ business: hostServer });
  expect(f.runtime.report("native").reflection?.status, JSON.stringify(f.runtime.report("native"))).toBe("proposed");
  const failed = await f.runtime.agent(
    async function* () {
      yield {
        type: "result",
        subtype: "error_max_turns",
        is_error: true,
        errors: ["provider detail"],
        session_id: "failed",
      };
    },
    { framework: "claude-agent-sdk" },
  );
  await expect(f.runtime.run(failed, "x", { sessionId: "s", executionId: "failure" })).rejects.toThrow(
    "provider detail",
  );
  expect(f.runtime.report("failure").task_status).toBe("failed");
});
