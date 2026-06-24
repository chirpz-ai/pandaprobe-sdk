import { describe, expect, it } from "vitest";
import {
  ScoreData,
  ScoreDataType,
  SpanData,
  SpanKind,
  SpanStatusCode,
  TraceData,
  TraceStatus,
  safeJson,
} from "../src/schemas.js";

describe("SpanData.toApiDict", () => {
  it("omits null/empty fields and serializes enums + dates", () => {
    const span = new SpanData({
      name: "llm-call",
      kind: SpanKind.LLM,
      status: SpanStatusCode.OK,
      input: { messages: [{ role: "user", content: "hi" }] },
      output: { messages: [{ role: "assistant", content: "yo" }] },
      model: "gpt-4",
      tokenUsage: { prompt_tokens: 10, completion_tokens: 5 },
      startedAt: new Date("2025-06-21T10:30:01Z"),
      endedAt: new Date("2025-06-21T10:30:04Z"),
    });
    const d = span.toApiDict();
    expect(d.kind).toBe("LLM");
    expect(d.status).toBe("OK");
    expect(d.model).toBe("gpt-4");
    expect(d.token_usage).toEqual({ prompt_tokens: 10, completion_tokens: 5 });
    expect(d.started_at).toBe("2025-06-21T10:30:01.000Z");
    expect(d.ended_at).toBe("2025-06-21T10:30:04.000Z");
    // Unset optional fields must be omitted.
    expect(d).not.toHaveProperty("parent_span_id");
    expect(d).not.toHaveProperty("error");
    expect(d).not.toHaveProperty("cost");
    expect(d).not.toHaveProperty("metadata");
  });

  it("includes parent_span_id and metadata when present", () => {
    const span = new SpanData({
      name: "tool",
      kind: SpanKind.TOOL,
      parentSpanId: "abc",
      metadata: { foo: "bar" },
      startedAt: new Date("2025-06-21T10:30:01Z"),
    });
    const d = span.toApiDict();
    expect(d.parent_span_id).toBe("abc");
    expect(d.metadata).toEqual({ foo: "bar" });
  });
});

describe("TraceData.toApiDict", () => {
  it("serializes spans recursively and omits empties", () => {
    const trace = new TraceData({
      name: "t",
      status: TraceStatus.COMPLETED,
      startedAt: new Date("2025-06-21T10:30:00Z"),
      tags: ["prod"],
      spans: [new SpanData({ name: "s", startedAt: new Date("2025-06-21T10:30:01Z") })],
    });
    const d = trace.toApiDict();
    expect(d.status).toBe("COMPLETED");
    expect(d.tags).toEqual(["prod"]);
    expect(Array.isArray(d.spans)).toBe(true);
    expect((d.spans as unknown[]).length).toBe(1);
    expect(d).not.toHaveProperty("session_id");
    expect(d).not.toHaveProperty("metadata");
  });
});

describe("ScoreData.toApiDict", () => {
  it("defaults data_type and source", () => {
    const score = new ScoreData({ traceId: "t1", name: "rel", value: "8.5" });
    const d = score.toApiDict();
    expect(d).toEqual({
      trace_id: "t1",
      name: "rel",
      value: "8.5",
      data_type: ScoreDataType.NUMERIC,
      source: "PROGRAMMATIC",
    });
  });
});

describe("safeJson", () => {
  it("passes primitives and recurses containers", () => {
    expect(safeJson("x")).toBe("x");
    expect(safeJson([1, "a", true])).toEqual([1, "a", true]);
    expect(safeJson({ a: { b: 1 } })).toEqual({ a: { b: 1 } });
  });

  it("handles non-serializable values without throwing", () => {
    expect(() => safeJson({ fn: () => 1 })).not.toThrow();
    expect(safeJson(BigInt(5))).toBe("5");
  });
});
