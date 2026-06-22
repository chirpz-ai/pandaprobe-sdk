import { beforeEach, describe, expect, it } from "vitest";
import { flush, init, withTrace } from "../../src/index.js";
import { SpanKind } from "../../src/schemas.js";
import { closeLlmSpan, safeSerialize, wrapAsyncStream } from "../../src/wrappers/base.js";
import { requestsTo } from "../helpers.js";

beforeEach(() => {
  init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
});

describe("safeSerialize", () => {
  it("uses toJSON when available and walks plain objects", () => {
    const withToJson = { toJSON: () => ({ a: 1 }) };
    expect(safeSerialize(withToJson)).toEqual({ a: 1 });
    expect(safeSerialize({ x: [1, { y: 2 }] })).toEqual({ x: [1, { y: 2 }] });
    expect(safeSerialize(7n)).toBe("7");
  });
});

describe("wrapAsyncStream leak prevention", () => {
  it("finalizes the span as ERROR when the stream throws mid-iteration", async () => {
    async function* throwingStream() {
      yield { choices: [{ delta: { content: "a" } }] };
      throw new Error("stream blew up");
    }

    await withTrace("agent", async (t) => {
      const span = t.span("llm", { kind: SpanKind.LLM });
      span.start();
      const wrapped = wrapAsyncStream(throwingStream(), span, (s) => {
        s.setOutput({ messages: [{ role: "assistant", content: "never" }] });
        closeLlmSpan(s);
      });
      await expect(
        (async () => {
          for await (const _ of wrapped) {
            // consume
          }
        })(),
      ).rejects.toThrow("stream blew up");
    });
    await flush();

    const body = requestsTo("/traces").at(-1)?.body as Record<string, unknown>;
    const spans = body.spans as Array<Record<string, unknown>>;
    const llm = spans.find((s) => s.kind === "LLM")!;
    expect(llm.status).toBe("ERROR");
    expect(llm.error).toContain("stream blew up");
  });

  it("finalizes successfully when the stream completes", async () => {
    async function* okStream() {
      yield { choices: [{ delta: { content: "x" } }] };
    }
    await withTrace("agent", async (t) => {
      const span = t.span("llm", { kind: SpanKind.LLM });
      span.start();
      const wrapped = wrapAsyncStream(okStream(), span, (s) => {
        s.setOutput({ messages: [{ role: "assistant", content: "x" }] });
        closeLlmSpan(s);
      });
      for await (const _ of wrapped) {
        // consume
      }
    });
    await flush();

    const body = requestsTo("/traces").at(-1)?.body as Record<string, unknown>;
    const spans = body.spans as Array<Record<string, unknown>>;
    const llm = spans.find((s) => s.kind === "LLM")!;
    expect(llm.status).toBe("OK");
    expect(llm.output).toEqual({ messages: [{ role: "assistant", content: "x" }] });
  });
});
