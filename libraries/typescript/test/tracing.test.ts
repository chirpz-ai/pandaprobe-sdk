import { beforeEach, describe, expect, it } from "vitest";
import { SpanKind, flush, getCurrentTrace, init, startTrace, withSpan, withTrace } from "../src/index.js";
import { requestsTo } from "./helpers.js";

beforeEach(() => {
  init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
});

function lastTrace(): Record<string, unknown> {
  return requestsTo("/traces").at(-1)?.body as Record<string, unknown>;
}

describe("SpanContext setters", () => {
  it("serializes cost, completionStartTime, modelParameters and token extras", async () => {
    const ts = new Date("2025-06-21T10:30:02Z");
    await withTrace("t", async (t) => {
      const s = t.span("llm", { kind: SpanKind.LLM });
      await s.run(async () => {
        s.setModel("gpt-4o");
        s.setModelParameters({ temperature: 0.7, top_p: 0.9 });
        s.setCompletionStartTime(ts);
        s.setCost({ total: 0.0021, input: 0.001, output: 0.0011 });
        s.setTokenUsage({ promptTokens: 10, completionTokens: 5, total_tokens: 15, reasoning_tokens: 3 });
        s.setOutput({ messages: [{ role: "assistant", content: "hi" }] });
      });
    });
    await flush();

    const span = (lastTrace().spans as Array<Record<string, unknown>>)[0]!;
    expect(span.model).toBe("gpt-4o");
    expect(span.model_parameters).toEqual({ temperature: 0.7, top_p: 0.9 });
    expect(span.completion_start_time).toBe("2025-06-21T10:30:02.000Z");
    expect(span.cost).toEqual({ total: 0.0021, input: 0.001, output: 0.0011 });
    expect(span.token_usage).toEqual({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
      reasoning_tokens: 3,
    });
  });

  it("setError marks the span ERROR even on a successful return", async () => {
    await withTrace("t", async (t) => {
      const s = t.span("step", { kind: SpanKind.TOOL });
      await s.run(async () => {
        s.setError("downstream failure");
      });
    });
    await flush();
    const span = (lastTrace().spans as Array<Record<string, unknown>>)[0]!;
    expect(span.status).toBe("ERROR");
    expect(span.error).toBe("downstream failure");
  });
});

describe("nested spans", () => {
  it("links a three-level span hierarchy via the span stack", async () => {
    await withTrace("t", async () => {
      await withSpan("a", { kind: SpanKind.CHAIN }, async () => {
        await withSpan("b", { kind: SpanKind.AGENT }, async () => {
          await withSpan("c", { kind: SpanKind.LLM }, async () => {});
        });
      });
    });
    await flush();
    const spans = lastTrace().spans as Array<Record<string, unknown>>;
    const a = spans.find((s) => s.name === "a")!;
    const b = spans.find((s) => s.name === "b")!;
    const c = spans.find((s) => s.name === "c")!;
    expect(a.parent_span_id).toBeUndefined();
    expect(b.parent_span_id).toBe(a.span_id);
    expect(c.parent_span_id).toBe(b.span_id);
  });
});

describe("imperative startTrace / end", () => {
  it("captures a trace and exposes getCurrentTrace within the active context", async () => {
    const t = startTrace("manual", { input: { messages: [{ role: "user", content: "hi" }] } });
    expect(getCurrentTrace()?.traceId).toBe(t.traceId);
    t.setOutput({ messages: [{ role: "assistant", content: "bye" }] });
    t.end();
    await flush();
    const body = lastTrace();
    expect(body.name).toBe("manual");
    expect(body.output).toEqual({ messages: [{ role: "assistant", content: "bye" }] });
  });

  it("marks the trace ERROR when ended with an error", async () => {
    const t = startTrace("manual-err");
    t.end(new Error("boom"));
    await flush();
    expect(lastTrace().status).toBe("ERROR");
  });

  it("restores the parent trace on LIFO end()", () => {
    const t1 = startTrace("A");
    const t2 = startTrace("B");
    expect(getCurrentTrace()).toBe(t2);
    t2.end();
    expect(getCurrentTrace()).toBe(t1);
    t1.end();
    expect(getCurrentTrace()).toBeNull();
  });

  it("out-of-order end() does not wipe a still-active trace or resurrect an ended one", () => {
    const t1 = startTrace("A");
    const t2 = startTrace("B");
    // End the outer trace first (out of order) — the active inner trace must survive.
    t1.end();
    expect(getCurrentTrace()).toBe(t2);
    // Ending the inner trace must not restore the already-ended outer trace.
    t2.end();
    expect(getCurrentTrace()).toBeNull();
  });
});

describe("concurrent traces", () => {
  it("keeps two concurrent traces isolated (no span bleed)", async () => {
    await Promise.all([
      withTrace("trace-a", async () => {
        await withSpan("span-a", { kind: SpanKind.LLM }, async () => {
          await new Promise((r) => setTimeout(r, 10));
        });
      }),
      withTrace("trace-b", async () => {
        await withSpan("span-b", { kind: SpanKind.TOOL }, async () => {
          await new Promise((r) => setTimeout(r, 5));
        });
      }),
    ]);
    await flush();

    const traces = requestsTo("/traces").map((r) => r.body as Record<string, unknown>);
    const a = traces.find((t) => t.name === "trace-a")!;
    const b = traces.find((t) => t.name === "trace-b")!;
    const aSpans = a.spans as Array<Record<string, unknown>>;
    const bSpans = b.spans as Array<Record<string, unknown>>;
    expect(aSpans).toHaveLength(1);
    expect(bSpans).toHaveLength(1);
    expect(aSpans[0]?.name).toBe("span-a");
    expect(bSpans[0]?.name).toBe("span-b");
  });
});
