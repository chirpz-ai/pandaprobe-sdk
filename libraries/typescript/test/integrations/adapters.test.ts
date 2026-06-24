import { beforeEach, describe, expect, it } from "vitest";
import { flush, init } from "../../src/index.js";
import { wrapClaudeAgentQuery } from "../../src/integrations/claude-agent-sdk/index.js";
import { OpenAIAgentsAdapter } from "../../src/integrations/openai-agents/index.js";
import { requestsTo } from "../helpers.js";

type Any = any;

beforeEach(() => {
  init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
});

function lastTrace(): Record<string, unknown> {
  return requestsTo("/traces").at(-1)?.body as Record<string, unknown>;
}

describe("Claude Agent SDK — wrapClaudeAgentQuery", () => {
  it("builds a CHAIN → AGENT → LLM/TOOL trace from streamed messages", async () => {
    async function* fakeQuery() {
      yield {
        type: "assistant",
        message: {
          model: "claude-sonnet-4",
          content: [
            { type: "text", text: "Let me check." },
            { type: "tool_use", id: "t1", name: "get_weather", input: { city: "SF" } },
          ],
        },
      };
      yield { type: "user", message: { content: [{ type: "tool_result", tool_use_id: "t1", content: "sunny" }] } };
      yield {
        type: "assistant",
        message: { model: "claude-sonnet-4", content: [{ type: "text", text: "It's sunny in SF." }] },
      };
      yield {
        type: "result",
        subtype: "success",
        result: "It's sunny in SF.",
        usage: { input_tokens: 10, output_tokens: 8 },
      };
    }

    const query = wrapClaudeAgentQuery(fakeQuery, { sessionId: "s1" });
    const seen: string[] = [];
    for await (const msg of query({ prompt: "weather in SF?" })) {
      seen.push((msg as { type: string }).type);
    }
    expect(seen).toEqual(["assistant", "user", "assistant", "result"]);
    await flush();

    const body = lastTrace();
    expect(body.name).toBe("ClaudeAgentSDK");
    expect(body.session_id).toBe("s1");
    expect(body.input).toEqual({ messages: [{ role: "user", content: "weather in SF?" }] });
    expect(body.output).toEqual({ messages: [{ role: "assistant", content: "It's sunny in SF." }] });

    const spans = body.spans as Array<Record<string, unknown>>;
    const chain = spans.find((s) => s.kind === "CHAIN")!;
    const agent = spans.find((s) => s.kind === "AGENT")!;
    const llms = spans.filter((s) => s.kind === "LLM");
    const tool = spans.find((s) => s.kind === "TOOL")!;

    expect(chain.name).toBe("ClaudeAgentSDK");
    expect(chain.parent_span_id).toBeUndefined();
    expect(agent.name).toBe("ClaudeAgentSDK");
    expect(agent.parent_span_id).toBe(chain.span_id);
    expect(llms).toHaveLength(2);
    expect(llms.every((l) => l.parent_span_id === agent.span_id)).toBe(true);

    // First LLM output carries the tool_call; the tool span is parented to the agent.
    const firstLlm = llms[0]!;
    expect((firstLlm.output as Any).messages[0].tool_calls[0]).toMatchObject({ name: "get_weather" });
    expect(tool.parent_span_id).toBe(agent.span_id);
    expect(tool.name).toBe("get_weather");
    expect(tool.input).toEqual({ city: "SF" });
    expect(tool.output).toBe("sunny");

    // Token usage from the result message lands on the last LLM span.
    expect(llms[1]!.token_usage).toEqual({ prompt_tokens: 10, completion_tokens: 8, total_tokens: 18 });

    // CHAIN holds the full conversation; AGENT output is the final assistant message.
    expect((chain.output as Any).messages.map((m: Any) => m.role)).toEqual(["user", "assistant", "tool", "assistant"]);
    expect(agent.output).toEqual({ messages: [{ role: "assistant", content: "It's sunny in SF." }] });
  });

  it("buffers a thinking-only turn into the next LLM span's reasoning_summary", async () => {
    async function* fakeQuery() {
      yield {
        type: "assistant",
        message: { model: "claude-sonnet-4", content: [{ type: "thinking", thinking: "reasoning..." }] },
      };
      yield { type: "assistant", message: { model: "claude-sonnet-4", content: [{ type: "text", text: "Answer." }] } };
      yield { type: "result", subtype: "success", usage: { input_tokens: 3, output_tokens: 2 } };
    }
    const query = wrapClaudeAgentQuery(fakeQuery);
    for await (const _ of query({ prompt: "q" })) {
      // consume
    }
    await flush();

    const spans = lastTrace().spans as Array<Record<string, unknown>>;
    const llms = spans.filter((s) => s.kind === "LLM");
    // The thinking-only turn does NOT create its own span.
    expect(llms).toHaveLength(1);
    expect((llms[0]!.metadata as Record<string, unknown>).reasoning_summary).toBe("reasoning...");
    expect(llms[0]!.output).toEqual({ messages: [{ role: "assistant", content: "Answer." }] });
  });

  it("marks the trace ERROR when the stream throws", async () => {
    async function* fakeQuery() {
      yield { type: "assistant", message: { model: "claude-sonnet-4", content: [{ type: "text", text: "partial" }] } };
      throw new Error("stream died");
    }
    const query = wrapClaudeAgentQuery(fakeQuery);
    await expect(
      (async () => {
        for await (const _ of query({ prompt: "q" })) {
          // consume
        }
      })(),
    ).rejects.toThrow("stream died");
    await flush();
    expect(lastTrace().status).toBe("ERROR");
  });
});

describe("OpenAI Agents — trace processor", () => {
  it("builds a root CHAIN span and maps the SDK span tree with normalized I/O", async () => {
    const adapter = new OpenAIAgentsAdapter({ tags: ["agents"] });
    const proc = adapter.createTraceProcessor();

    const gen = {
      type: "generation",
      model: "gpt-4o",
      input: [{ role: "user", content: "hi" }],
      output: [{ role: "assistant", content: "hello" }],
      usage: { input_tokens: 7, output_tokens: 3 },
    };
    proc.onTraceStart({ traceId: "tr1", name: "MyAgent" });
    proc.onSpanStart({ traceId: "tr1", spanId: "a1", parentId: null, spanData: { type: "agent", name: "planner" } });
    proc.onSpanStart({ traceId: "tr1", spanId: "g1", parentId: "a1", spanData: gen });
    proc.onSpanEnd({ traceId: "tr1", spanId: "g1", parentId: "a1", spanData: gen });
    proc.onSpanEnd({ traceId: "tr1", spanId: "a1", parentId: null, spanData: { type: "agent", name: "planner" } });
    proc.onTraceEnd({ traceId: "tr1", name: "MyAgent" });
    await flush();

    const body = lastTrace();
    expect(body.name).toBe("OpenAIAgentsSDK"); // trace name is fixed
    expect(body.tags).toEqual(["agents"]);

    const spans = body.spans as Array<Record<string, unknown>>;
    const root = spans.find((s) => s.kind === "CHAIN")!;
    const agent = spans.find((s) => s.kind === "AGENT")!;
    const llm = spans.find((s) => s.kind === "LLM")!;

    // Hierarchy: root CHAIN ← AGENT ← LLM, all remapped to UUID span ids.
    expect(root.name).toBe("MyAgent");
    expect(root.parent_span_id).toBeUndefined();
    expect(agent.name).toBe("planner");
    expect(agent.parent_span_id).toBe(root.span_id);
    expect(llm.parent_span_id).toBe(agent.span_id);

    // LLM span: model becomes the name, usage + normalized message I/O.
    expect(llm.name).toBe("gpt-4o");
    expect(llm.model).toBe("gpt-4o");
    expect(llm.token_usage).toEqual({ prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 });
    expect(llm.input).toEqual({ messages: [{ role: "user", content: "hi" }] });
    expect(llm.output).toEqual({ messages: [{ role: "assistant", content: "hello" }] });

    // LLM I/O propagates to the parent AGENT span.
    expect(agent.input).toEqual({ messages: [{ role: "user", content: "hi" }] });
    expect(agent.output).toEqual({ messages: [{ role: "assistant", content: "hello" }] });

    // Trace I/O: trimmed input (last user message) + latest LLM output.
    expect(body.input).toEqual({ messages: [{ role: "user", content: "hi" }] });
    expect(body.output).toEqual({ messages: [{ role: "assistant", content: "hello" }] });

    // Root CHAIN holds the full conversation (input messages + output messages).
    expect(root.output).toEqual({
      messages: [
        { role: "user", content: "hi" },
        { role: "assistant", content: "hello" },
      ],
    });
    expect(body.status).toBe("COMPLETED");
  });

  it("maps span kinds: handoff→AGENT, guardrail→OTHER, function→TOOL", async () => {
    const adapter = new OpenAIAgentsAdapter();
    const proc = adapter.createTraceProcessor();
    proc.onTraceStart({ traceId: "tk", name: "T" });
    for (const [id, type] of [
      ["h1", "handoff"],
      ["gr1", "guardrail"],
      ["f1", "function"],
    ] as const) {
      proc.onSpanStart({ traceId: "tk", spanId: id, parentId: null, spanData: { type } });
      proc.onSpanEnd({ traceId: "tk", spanId: id, parentId: null, spanData: { type } });
    }
    proc.onTraceEnd({ traceId: "tk", name: "T" });
    await flush();

    const spans = (lastTrace().spans as Array<Record<string, unknown>>).filter((s) => s.kind !== "CHAIN");
    const byName = (n: string) => spans.find((s) => s.name === n)!;
    expect(byName("Handoff").kind).toBe("AGENT");
    expect(byName("Guardrail").kind).toBe("OTHER");
    expect(byName("Function").kind).toBe("TOOL");
  });

  it("marks the trace and root CHAIN span ERROR when any span errors", async () => {
    const adapter = new OpenAIAgentsAdapter();
    const proc = adapter.createTraceProcessor();
    proc.onTraceStart({ traceId: "tr3", name: "MyAgent" });
    proc.onSpanStart({ traceId: "tr3", spanId: "f1", parentId: null, spanData: { type: "function", name: "lookup" } });
    proc.onSpanEnd({
      traceId: "tr3",
      spanId: "f1",
      parentId: null,
      spanData: { type: "function", name: "lookup" },
      error: { message: "tool blew up" },
    });
    proc.onTraceEnd({ traceId: "tr3", name: "MyAgent" });
    await flush();

    const body = lastTrace();
    expect(body.status).toBe("ERROR");
    const spans = body.spans as Array<Record<string, unknown>>;
    expect(spans.find((s) => s.kind === "CHAIN")?.status).toBe("ERROR");
    const fn = spans.find((s) => s.name === "lookup")!;
    expect(fn.kind).toBe("TOOL");
    expect(fn.status).toBe("ERROR");
    expect(fn.error).toContain("tool blew up");
  });
});
