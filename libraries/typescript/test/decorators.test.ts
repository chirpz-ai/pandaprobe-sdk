import { beforeEach, describe, expect, it } from "vitest";
import { flush, init, span, trace, withSpan, withTrace } from "../src/index.js";
import { SpanKind } from "../src/schemas.js";
import { requestsTo } from "./helpers.js";

beforeEach(() => {
  init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
});

function lastTrace(): Record<string, unknown> {
  const reqs = requestsTo("/traces");
  return reqs[reqs.length - 1]?.body as Record<string, unknown>;
}

describe("withTrace", () => {
  it("captures a trace with input, output and name", async () => {
    const result = await withTrace("agent", { input: { messages: [{ role: "user", content: "hi" }] } }, async (t) => {
      t.setOutput({ messages: [{ role: "assistant", content: "yo" }] });
      return 42;
    });
    expect(result).toBe(42);
    await flush();
    const body = lastTrace();
    expect(body.name).toBe("agent");
    expect(body.status).toBe("COMPLETED");
    expect(body.input).toEqual({ messages: [{ role: "user", content: "hi" }] });
    expect(body.output).toEqual({ messages: [{ role: "assistant", content: "yo" }] });
  });

  it("records ERROR status when the callback throws", async () => {
    await expect(
      withTrace("boom", async () => {
        throw new Error("kaboom");
      }),
    ).rejects.toThrow("kaboom");
    await flush();
    expect(lastTrace().status).toBe("ERROR");
  });
});

describe("withSpan", () => {
  it("nests spans under the active trace with parent links", async () => {
    await withTrace("agent", async () => {
      await withSpan("outer", { kind: SpanKind.CHAIN }, async () => {
        await withSpan("inner", { kind: SpanKind.LLM }, async (s) => {
          s.setInput({ messages: [{ role: "user", content: "q" }] });
          s.setOutput({ messages: [{ role: "assistant", content: "a" }] });
          s.setModel("gpt-4");
        });
      });
    });
    await flush();
    const spans = lastTrace().spans as Array<Record<string, unknown>>;
    expect(spans.length).toBe(2);
    const outer = spans.find((s) => s.name === "outer")!;
    const inner = spans.find((s) => s.name === "inner")!;
    expect(outer.kind).toBe("CHAIN");
    expect(inner.kind).toBe("LLM");
    expect(inner.parent_span_id).toBe(outer.span_id);
    expect(inner.model).toBe("gpt-4");
  });

  it("runs the callback even without an active trace (detached)", async () => {
    let ran = false;
    await withSpan("orphan", async () => {
      ran = true;
    });
    expect(ran).toBe(true);
  });
});

describe("@trace / @span method decorators", () => {
  it("instruments async methods and captures output", async () => {
    class Agent {
      @trace({ name: "run-agent" })
      async run(_input: { messages: { role: string; content: string }[] }) {
        return { messages: [{ role: "assistant", content: "done" }] };
      }
    }
    const out = await new Agent().run({ messages: [{ role: "user", content: "go" }] });
    expect(out).toEqual({ messages: [{ role: "assistant", content: "done" }] });
    await flush();
    const body = lastTrace();
    expect(body.name).toBe("run-agent");
    expect(body.input).toEqual({ messages: [{ role: "user", content: "go" }] });
    expect(body.output).toEqual({ messages: [{ role: "assistant", content: "done" }] });
  });

  it("instruments sync methods (returns synchronously)", async () => {
    class Calc {
      @trace()
      add(a: number, b: number) {
        return a + b;
      }
    }
    const value = new Calc().add(2, 3);
    expect(value).toBe(5); // not a Promise
    await flush();
    expect(lastTrace().name).toBe("add");
  });

  it("@span nests under an active @trace", async () => {
    class Pipeline {
      @span({ kind: SpanKind.TOOL })
      async tool() {
        return "ok";
      }

      @trace({ name: "pipe" })
      async run() {
        return await this.tool();
      }
    }
    await new Pipeline().run();
    await flush();
    const spans = lastTrace().spans as Array<Record<string, unknown>>;
    expect(spans.length).toBe(1);
    expect(spans[0]?.name).toBe("tool");
    expect(spans[0]?.kind).toBe("TOOL");
  });
});
