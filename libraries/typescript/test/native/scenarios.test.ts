import { randomUUID } from "node:crypto";
import { beforeEach, expect, it, vi } from "vitest";
import { failureLoop, supportLoop } from "../../examples/reflect/_support/loops.js";
import { nativeFactory } from "../../examples/reflect/_support/native.js";
import type { Framework } from "../../src/reflect/index.js";
import { optionalModule } from "../../src/reflect/integrations.js";
import { WRITE_RULE, fixture } from "../reflect/helpers.js";
const hostTraceEnd = vi.fn();
beforeEach(async () => {
  const sdk = await optionalModule("@openai/agents");
  sdk.setTracingDisabled(false);
  sdk.setTraceProcessors([
    {
      onTraceStart() {},
      onSpanStart() {},
      onSpanEnd() {},
      onTraceEnd: hostTraceEnd,
      forceFlush() {},
      shutdown() {},
    },
  ]);
  hostTraceEnd.mockClear();
});
function strings(value: any): string[] {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.flatMap(strings);
  if (value && typeof value === "object") return Object.values(value).flatMap(strings);
  return [];
}
function objects(value: any): any[] {
  const result: any[] = [];
  for (const s of strings(value)) {
    try {
      const parsed = JSON.parse(s);
      if (parsed && typeof parsed === "object") result.push(parsed, ...objects(parsed));
    } catch {}
  }
  return result;
}
async function scripted(f: ReturnType<typeof fixture>, framework: Exclude<Framework, "generic">, noProposal = false) {
  const turns = new Map<string, number>();
  const next = (messages: any) => {
    const execution = f.runtime.currentExecution!;
    const key = `${execution.id}:${execution.role}`;
    const turn = turns.get(key) ?? 0;
    turns.set(key, turn + 1);
    const all = strings(messages).join("\n");
    const records = objects(messages);
    const order = [...records]
      .reverse()
      .find((o) => o.merchandise_paid_cents !== undefined || o.paid_cents !== undefined);
    if (execution.role === "reflection")
      return turn === 0 && !noProposal ? { name: "reflect_bash", args: { command: WRITE_RULE } } : null;
    const isRefund = all.includes("settlement batches");
    const id =
      all.match(/order (INCIDENT-101|FRESH-501|HELD-\d+)/)?.[1] ?? all.match(/Ticket for order ([\w-]+)/)?.[1];
    if (turn === 0) return { name: "reflect_bash", args: { command: "grep -rn shipping /rules || true" } };
    if (turn === 1) return { name: isRefund ? "lookup_refund_order" : "lookup_order", args: { order_id: id } };
    if (isRefund) {
      if (turn === 2) {
        expect(order).toBeDefined();
        const learned = all.includes("Exclude outbound shipping; refund merchandise only.");
        return {
          name: "submit_refund",
          args: {
            order_id: id,
            amount_cents: learned ? order.merchandise_paid_cents : order.invoice_total_cents,
            idempotency_key: id,
          },
        };
      }
      return null;
    }
    if (turn === 2) return { name: "search_policy", args: { query: "current returns" } };
    const inspect =
      all.includes("Do not take action yet.") && !all.includes("You are now authorized to resolve this ticket.");
    const latestHuman =
      strings(messages)
        .filter((s) => s.startsWith("Ticket "))
        .at(-1) ?? "";
    if (turn === 3 && !latestHuman.includes("Do not take action yet.") && !inspect) {
      const allowed =
        order.damaged || (!order.final_sale && order.days_since_delivery <= (order.policy === "returns-v1" ? 30 : 14));
      return {
        name: "resolve_ticket",
        args: {
          order_id: id,
          action: allowed ? "refund" : "escalate",
          amount_cents: allowed ? order.paid_cents : 0,
          policy_version: order.policy,
          idempotency_key: id,
        },
      };
    }
    return null;
  };
  if (["langchain", "langgraph", "deepagents"].includes(framework)) {
    const { BaseChatModel } = await optionalModule("@langchain/core/language_models/chat_models");
    const { RunnableBinding } = await optionalModule("@langchain/core/runnables");
    const { AIMessage } = await optionalModule("@langchain/core/messages");
    class Model extends BaseChatModel {
      constructor() {
        super({});
      }
      _llmType() {
        return "scripted-business";
      }
      bindTools(tools: any[]) {
        return new RunnableBinding({ bound: this, kwargs: { tools }, config: {} });
      }
      async _generate(messages: any[]) {
        const action = next(messages);
        const message = action
          ? new AIMessage({
              content: "",
              tool_calls: [{ id: randomUUID(), name: action.name, args: action.args, type: "tool_call" }],
            })
          : new AIMessage("The service returned the recorded result.");
        return { generations: [{ message, text: "" }] };
      }
    }
    return { model: new Model() };
  }
  if (framework === "openai-agents") {
    const sdk = await optionalModule("@openai/agents");
    return {
      model: {
        async getResponse(request: any) {
          const action = next([request.systemInstructions, request.input]);
          const response = {
            usage: new sdk.Usage(),
            responseId: randomUUID(),
            output: action
              ? [
                  {
                    type: "function_call",
                    callId: randomUUID(),
                    name: action.name,
                    arguments: JSON.stringify(action.args),
                    status: "completed",
                  },
                ]
              : [
                  {
                    type: "message",
                    role: "assistant",
                    status: "completed",
                    content: [{ type: "output_text", text: "The service returned the recorded result." }],
                  },
                ],
          };
          return sdk.withGenerationSpan(async () => response, {
            data: {
              model: "scripted-business",
              input: [
                { role: "system", content: request.systemInstructions },
                ...request.input.map((item: any) =>
                  item.type === "function_call_result"
                    ? { role: "tool", content: strings(item.output).join("\n") }
                    : item,
                ),
              ],
              output: response.output,
            },
          });
        },
        getStreamedResponse() {
          throw Error("Use nonstream fixture");
        },
      },
    };
  }
  if (framework === "vercel-ai") {
    const { MockLanguageModelV3 } = await optionalModule("ai/test");
    return {
      model: new MockLanguageModelV3({
        doGenerate: async (request: any) => {
          const action = next(request.prompt);
          return {
            content: action
              ? [
                  {
                    type: "tool-call",
                    toolCallId: randomUUID(),
                    toolName: action.name,
                    input: JSON.stringify(action.args),
                  },
                ]
              : [{ type: "text", text: "The service returned the recorded result." }],
            finishReason: { unified: action ? "tool-calls" : "stop", raw: action ? "tool_calls" : "stop" },
            usage: {
              inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
              outputTokens: { total: 5, text: 5, reasoning: 0 },
            },
            warnings: [],
          };
        },
      }),
    };
  }
  const { Client } = await import("./node_modules/@modelcontextprotocol/sdk/dist/esm/client/index.js");
  const { InMemoryTransport } = await import("./node_modules/@modelcontextprotocol/sdk/dist/esm/inMemory.js");
  return {
    model: "scripted-claude",
    query: async function* (args: any) {
      const clients: Record<string, any> = {};
      const history: any[] = [args.options.systemPrompt, args.prompt];
      try {
        for (const [name, server] of Object.entries(args.options.mcpServers) as any) {
          const [a, b] = InMemoryTransport.createLinkedPair();
          const client = new Client({ name: "business-fixture", version: "1" });
          await server.instance.connect(b);
          await client.connect(a);
          clients[name] = client;
        }
        for (let turn = 0; turn < 8; turn++) {
          const action = next(history);
          if (!action) break;
          const server = action.name === "reflect_bash" ? "pandaprobe_reflect" : "support";
          const id = randomUUID();
          yield {
            type: "assistant",
            message: {
              content: [{ type: "tool_use", id, name: `mcp__${server}__${action.name}`, input: action.args }],
            },
          };
          const output = await clients[server].callTool({ name: action.name, arguments: action.args });
          history.push(output);
          yield {
            type: "user",
            message: { content: [{ type: "tool_result", tool_use_id: id, content: output.content }] },
          };
        }
        yield {
          type: "result",
          subtype: "success",
          is_error: false,
          result: "The service returned the recorded result.",
          session_id: f.runtime.currentExecution!.sessionId,
        };
      } finally {
        for (const client of Object.values(clients)) await client.close();
      }
    },
  };
}
const frameworks = ["langchain", "langgraph", "deepagents", "openai-agents", "claude-agent-sdk", "vercel-ai"] as const;
function persisted(f: ReturnType<typeof fixture>) {
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
it.each(frameworks)(
  "%s completes controlled failure, six independent arms, and a fresh verified refund",
  async (framework) => {
    const f = fixture({
      evaluationTimeout: 2,
      traceVisibilityTimeout: 2,
      reflectionTimeout: 10,
      validationTimeout: 20,
    });
    persisted(f);
    const factory = await nativeFactory(f.runtime, framework, await scripted(f, framework));
    factory.forRuntime = async (runtime) => {
      f.server.score = 1;
      return nativeFactory(runtime, framework, await scripted({ ...f, runtime }, framework));
    };
    const partial: any[] = [];
    const report = await failureLoop(f.runtime, factory, { onReport: (r) => partial.push(r) });
    expect(report.status, JSON.stringify(report)).toBe("completed");
    expect(report.turns[0].verification.success).toBe(false);
    expect(report.validations[0].cases).toHaveLength(3);
    expect(report.turns[1].verification.success).toBe(true);
    expect(report.turns[1].verification.evidence.actual_ledger[0].amount_cents).toBe(9250);
    expect(report.milestones.independently_accepted).toBe(true);
    expect(report.turns[1].initial_active_memory_bytes).toBe(0);
    expect(partial.length).toBeGreaterThanOrEqual(4);
    expect(report.milestones.fresh_guidance_retrieved).toBe(true);
    if (framework === "openai-agents") {
      // Reopening the runtime must not register a second global exporter.
      for (const turn of report.turns) expect(turn.reflect.trace_ids).toHaveLength(1);
      expect(f.client.logTrace).toHaveBeenCalledTimes(9); // initial + reflection + six arms + fresh
      expect(hostTraceEnd).toHaveBeenCalledTimes(9);
    }
  },
);
it.each(frameworks)(
  "%s runs the shared 18-turn support scenario using independent business verification",
  async (framework) => {
    const f = fixture({ evaluationTimeout: 2, traceVisibilityTimeout: 2 });
    f.server.score = 1;
    persisted(f);
    const factory = await nativeFactory(f.runtime, framework, await scripted(f, framework));
    const report = await supportLoop(f.runtime, factory);
    expect(report.status, JSON.stringify(report)).toBe("completed");
    expect(report.turns).toHaveLength(18);
    expect(
      report.summary.verified_successes,
      JSON.stringify(report.turns.filter((t: any) => !t.verification.success)),
    ).toBe(18);
  },
);
it.each(frameworks)("%s does not call an incomplete failure loop a success", async (framework) => {
  const f = fixture({ evaluationTimeout: 2, traceVisibilityTimeout: 2 });
  persisted(f);
  const factory = await nativeFactory(f.runtime, framework, await scripted(f, framework, true));
  const report = await failureLoop(f.runtime, factory);
  expect(report.status, JSON.stringify(report)).toBe("incomplete");
  expect(report.stopped_at).toBe("reflection");
  expect(report.validations).toEqual([]);
  expect(report.turns).toHaveLength(1);
});

it.each(frameworks)("%s keeps simultaneous runtime/tool instances isolated", async (framework) => {
  const left = fixture({ evaluationTimeout: 2, traceVisibilityTimeout: 2 });
  const right = fixture({ evaluationTimeout: 2, traceVisibilityTimeout: 2 });
  left.server.score = 1;
  right.server.score = 1;
  persisted(left);
  persisted(right);
  const first = await nativeFactory(left.runtime, framework, await scripted(left, framework));
  const second = await nativeFactory(right.runtime, framework, await scripted(right, framework));
  const { RefundBatch, DISCOVERY, REFUND_INSTRUCTIONS, refundPrompt } = await import(
    "../../examples/reflect/_support/refund.js"
  );
  const { refundTools } = await import("../../examples/reflect/_support/loops.js");
  const a = new RefundBatch({ ...DISCOVERY, outbound_shipping_cents: 0 });
  const b = new RefundBatch({ ...DISCOVERY, merchandise_paid_cents: 1500, outbound_shipping_cents: 0 });
  const aa = await first(refundTools(a), REFUND_INSTRUCTIONS);
  const bb = await second(refundTools(b), REFUND_INSTRUCTIONS);
  await Promise.all([
    left.runtime.run(aa, refundPrompt(a.order), { sessionId: "same", executionId: "same" }),
    right.runtime.run(bb, refundPrompt(b.order), { sessionId: "same", executionId: "same" }),
  ]);
  expect(a.ledger[0]?.amount_cents).toBe(6500);
  expect(b.ledger[0]?.amount_cents).toBe(1500);
  expect(left.runtime.workspace.path).not.toBe(right.runtime.workspace.path);
  if (framework === "openai-agents") {
    expect(left.client.logTrace).toHaveBeenCalledTimes(1);
    expect(right.client.logTrace).toHaveBeenCalledTimes(1);
    expect(left.runtime.report("same").trace_ids).toHaveLength(1);
    expect(right.runtime.report("same").trace_ids).toHaveLength(1);
    expect(hostTraceEnd).toHaveBeenCalledTimes(2);
    const sdk = await optionalModule("@openai/agents");
    await sdk.withTrace("unrelated host trace", async () => {});
    expect(hostTraceEnd).toHaveBeenCalledTimes(3);
    expect(left.client.logTrace).toHaveBeenCalledTimes(1);
    expect(right.client.logTrace).toHaveBeenCalledTimes(1);
  }
});

it.each(frameworks)("%s propagates native task failures without claiming reflection success", async (framework) => {
  const f = fixture();
  persisted(f);
  const options: import("../../examples/reflect/_support/native.js").NativeOptions = await scripted(f, framework);
  if (["langchain", "langgraph", "deepagents"].includes(framework))
    options.model._generate = async () => {
      throw Error("native provider failure");
    };
  else if (framework === "openai-agents")
    options.model.getResponse = async () => {
      throw Error("native provider failure");
    };
  else if (framework === "vercel-ai")
    options.model.doGenerate = async () => {
      throw Error("native provider failure");
    };
  else
    options.query = async function* () {
      yield {
        type: "result",
        subtype: "error_during_execution",
        is_error: true,
        errors: ["native provider failure"],
        session_id: "failed",
      };
    };
  const factory = await nativeFactory(f.runtime, framework, options);
  const agent = await factory([], "host");
  await expect(f.runtime.run(agent, "task", { sessionId: "s", executionId: "failed" })).rejects.toThrow(
    "native provider failure",
  );
  expect(f.runtime.report("failed").task_status).toBe("failed");
  expect(f.runtime.report("failed").reflection).toBeNull();
});

it("support honors Python's conversation/validation limits and preserves an interrupted report", async () => {
  const f = fixture({ thresholds: { task_completion: 0.8 } });
  persisted(f);
  const factory = await nativeFactory(f.runtime, "langchain", await scripted(f, "langchain"));
  const validate = vi.spyOn(f.runtime, "validate");
  const report = await supportLoop(f.runtime, factory, { rounds: 2, maxTickets: 1, maxValidations: 0 });
  expect(report.status).toBe("completed");
  expect(report.turns.map((row: any) => row.stage)).toEqual(["inspect", "resolve", "repeat"]);
  expect(report.turns.every((row: any) => row.reflect.evaluation.status === "evaluated")).toBe(true);
  expect(validate).not.toHaveBeenCalled();
  expect(report.validations).toEqual([]);

  const original = f.runtime.report.bind(f.runtime);
  vi.spyOn(f.runtime, "report").mockImplementation((id) => ({
    ...original(id),
    evaluation: { execution_id: id, status: "pending", run_ids: [], scores: {}, notice: null, reason: "pending" },
  }));
  const saved: any[] = [];
  await expect(
    supportLoop(f.runtime, factory, { maxTickets: 1, onReport: (r) => saved.push(structuredClone(r)) }),
  ).rejects.toThrow("Platform evidence unavailable");
  expect(saved.at(-1)).toMatchObject({ status: "interrupted", turns: [{ stage: "inspect" }] });
});

it.each(["detection", "validation", "fresh_evaluation"])("failure report stops honestly at %s", async (stage) => {
  const f = fixture({ thresholds: { task_completion: 0.8 } });
  persisted(f);
  if (stage === "detection") f.server.score = 1;
  if (stage === "validation")
    vi.spyOn(f.runtime, "validate").mockResolvedValue({
      proposal_id: "pending",
      outcome: "pending",
      reason: "Missing independent evidence",
      mode: "replay",
      cases: [],
      created_at: "fixed",
    });
  const factory = await nativeFactory(f.runtime, "langchain", await scripted(f, "langchain"));
  factory.forRuntime = async (runtime) =>
    nativeFactory(runtime, "langchain", await scripted({ ...f, runtime }, "langchain"));
  const report = await failureLoop(f.runtime, factory);
  expect(report.status).toBe("incomplete");
  expect(report.stopped_at).toBe(stage);
  if (stage === "fresh_evaluation") {
    expect(report.milestones.fresh_guidance_retrieved).toBe(true);
    expect(report.milestones.fresh_task_verified).toBe(true);
  }
});

it("Claude reports an evaluated score exactly at the threshold without forcing reflection", async () => {
  const f = fixture({ thresholds: { task_completion: 0.8 }, evaluationTimeout: 2, traceVisibilityTimeout: 2 });
  f.server.score = 0.8;
  persisted(f);
  const factory = await nativeFactory(f.runtime, "claude-agent-sdk", await scripted(f, "claude-agent-sdk"));
  const report = await failureLoop(f.runtime, factory);
  expect(report.status).toBe("incomplete");
  expect(report.stopped_at).toBe("detection");
  expect(report.milestones).toMatchObject({ initial_failure_verified: true, platform_failure_detected: false });
  expect(report.turns).toHaveLength(1);
  expect(report.turns[0].reflect.evaluation).toMatchObject({
    status: "evaluated",
    scores: { task_completion: 0.8 },
    notice: null,
  });
  expect(report.reason).toContain('Scores: {"task_completion":0.8}');
  expect(report.reason).toContain('thresholds: {"task_completion":0.8}');
  expect(report.reason).toContain("strictly below");
  expect(f.runtime.proposals()).toEqual([]);
});
