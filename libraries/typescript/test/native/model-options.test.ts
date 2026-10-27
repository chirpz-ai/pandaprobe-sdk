import { randomUUID } from "node:crypto";
import { afterEach, expect, it, vi } from "vitest";
import { nativeFactory } from "../../examples/reflect/_support/native.js";
import { Outcome, ReplayCase } from "../../src/reflect/index.js";
import { WRITE_RULE, fixture } from "../reflect/helpers.js";

interface ChatRequest {
  model: string;
  reasoning_effort?: string;
  tools: Array<{ function: { name: string } }>;
}

afterEach(() => vi.unstubAllEnvs());

it.each(["langchain", "langgraph", "deepagents"] as const)(
  "%s sends Python's reasoning setting through the real ChatOpenAI client in every Reflect phase",
  async (framework) => {
    vi.stubEnv("OPENAI_API_KEY", "offline-fixture");
    vi.stubEnv("OPENAI_BASE_URL", "http://openai.invalid/v1");
    vi.stubEnv("OPENAI_API_BASE", "http://openai.invalid/v1");
    vi.stubEnv("LANGSMITH_TRACING", "false");
    vi.stubEnv("LANGCHAIN_TRACING_V2", "false");
    const f = fixture({ evaluationTimeout: 2, traceVisibilityTimeout: 2 });
    const exportTrace = f.client.logTrace.bind(f.client);
    vi.spyOn(f.client, "logTrace").mockImplementation((trace) => {
      f.server.traces.set(trace.traceId, {
        ...trace.toApiDict(),
        project_id: randomUUID(),
        spans: trace.spans.map((span) => ({ ...span.toApiDict(), trace_id: trace.traceId, status: span.status })),
      });
      exportTrace(trace);
    });
    const requests: Array<{ phase: string; body: ChatRequest }> = [];
    const turns = new Map<string, number>();
    const receipts = new Set<string>();
    // Only HTTP is faked: construct ChatOpenAI from the example's model string,
    // then run real framework tools, reflection and fresh paired replay agents.
    globalThis.fetch = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const request = new Request(input, init);
      const url = new URL(request.url);
      if (url.host === "fixture") return new Response(null, { status: 202 });
      if (url.href !== "http://openai.invalid/v1/chat/completions")
        throw new Error(`Unexpected test request: ${url.origin}${url.pathname}`);
      const body = (await request.json()) as ChatRequest;
      const execution = f.runtime.currentExecution!;
      requests.push({ phase: execution.role, body });
      if (body.reasoning_effort !== "none")
        return Response.json(
          {
            error: {
              message: "Function tools require reasoning_effort='none' for this model",
              type: "invalid_request_error",
            },
          },
          { status: 400 },
        );
      const key = `${execution.id}:${execution.role}`;
      const turn = turns.get(key) ?? 0;
      turns.set(key, turn + 1);
      const tool = execution.role === "reflection" ? "reflect_bash" : "record_receipt";
      const message =
        turn === 0
          ? {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: randomUUID(),
                  type: "function",
                  function: {
                    name: tool,
                    arguments: JSON.stringify(tool === "reflect_bash" ? { command: WRITE_RULE } : {}),
                  },
                },
              ],
            }
          : { role: "assistant", content: "Receipt recorded." };
      return Response.json({
        id: randomUUID(),
        object: "chat.completion",
        created: 1,
        model: body.model,
        choices: [{ index: 0, message, finish_reason: turn === 0 ? "tool_calls" : "stop" }],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      });
    }) as typeof fetch;
    const factory = await nativeFactory(f.runtime, framework, { model: "gpt-6-luna" });
    const tools = [
      {
        name: "record_receipt",
        description: "Record independent proof that this task ran.",
        fields: {},
        execute: () => {
          receipts.add(f.runtime.currentExecution!.id);
          return "Receipt recorded.";
        },
      },
    ];
    const agent = await factory(tools, "Use record_receipt, then report its result.");
    await f.runtime.run(agent, "Record a receipt.", { sessionId: "wire", executionId: "wire" });
    const report = f.runtime.report("wire");
    expect(report.reflection?.status).toBe("proposed");
    const verdict = await f.runtime.validate(report.reflection!.proposal!.id, {
      cases: [1, 2, 3].map((id) => new ReplayCase({ id: String(id), input: "Record a receipt." })),
      replay: async (request) => {
        const fresh = await factory(tools, "Use record_receipt, then report its result.", { validation: true });
        await f.runtime.run(fresh, request.case.input, { sessionId: request.execution.sessionId });
        const success = receipts.has(request.execution.id);
        return new Outcome({ scores: { task_completion: success ? 1 : 0 }, success });
      },
    });
    expect(verdict.cases).toHaveLength(3);
    expect(receipts.size).toBe(7);
    expect(requests).toHaveLength(16);
    expect(new Set(requests.map((r) => r.phase))).toEqual(new Set(["task", "reflection", "validation"]));
    for (const { phase, body } of requests) {
      expect(body.model).toBe("gpt-6-luna");
      expect(body.reasoning_effort).toBe("none");
      const names = body.tools.map((tool) => tool.function.name);
      if (phase === "reflection") expect(names).toEqual(["reflect_bash"]);
      else expect(names).toContain("record_receipt");
    }
  },
);
