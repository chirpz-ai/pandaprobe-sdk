import { beforeEach, describe, expect, it } from "vitest";
import { flush, init } from "../../src/index.js";
import { pandaProbeMiddleware } from "../../src/integrations/vercel-ai/index.js";
import { requestsTo } from "../helpers.js";

type Any = any;

beforeEach(() => {
  init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
});

function llmSpan(): Record<string, unknown> {
  const body = requestsTo("/traces").at(-1)?.body as Record<string, unknown>;
  const spans = body.spans as Array<Record<string, unknown>>;
  return spans.find((s) => s.kind === "LLM") as Record<string, unknown>;
}

/** A ReadableStream that yields `chunks`, then closes or (if `error`) errors. */
function sourceStream(chunks: Any[], error?: Error): ReadableStream<Any> {
  let i = 0;
  return new ReadableStream<Any>({
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(chunks[i++]);
        return;
      }
      if (error) {
        controller.error(error);
        return;
      }
      controller.close();
    },
  });
}

async function drain(stream: ReadableStream<Any>): Promise<void> {
  const reader = stream.getReader();
  while (true) {
    const { done } = await reader.read();
    if (done) {
      break;
    }
  }
}

describe("pandaProbeMiddleware (Vercel AI SDK)", () => {
  it("wraps doGenerate and emits an LLM span", async () => {
    const mw = pandaProbeMiddleware();
    const params = {
      prompt: [
        { role: "system", content: "Be terse." },
        { role: "user", content: [{ type: "text", text: "hi" }] },
      ],
    };
    const model = { modelId: "gpt-4o" };
    const result = await mw.wrapGenerate({
      doGenerate: async () => ({
        content: [{ type: "text", text: "yo" }],
        usage: { inputTokens: 4, outputTokens: 1, totalTokens: 5 },
      }),
      params,
      model,
    });
    expect(result.content[0].text).toBe("yo");
    await flush();

    const span = llmSpan();
    expect(span.model).toBe("gpt-4o");
    expect(span.input).toEqual({
      messages: [
        { role: "system", content: "Be terse." },
        { role: "user", content: "hi" },
      ],
    });
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "yo" }] });
    expect(span.token_usage).toEqual({ prompt_tokens: 4, completion_tokens: 1, total_tokens: 5 });
  });

  it("finalizes the span as ERROR when doGenerate throws", async () => {
    const mw = pandaProbeMiddleware();
    await expect(
      mw.wrapGenerate({
        doGenerate: async () => {
          throw new Error("model error");
        },
        params: { prompt: "hi" },
        model: { modelId: "m" },
      }),
    ).rejects.toThrow("model error");
    await flush();
    expect(llmSpan().status).toBe("ERROR");
  });

  it("wrapStream finalizes the span on normal completion", async () => {
    const mw = pandaProbeMiddleware();
    const chunks = [
      { type: "text-delta", delta: "Hel" },
      { type: "text-delta", delta: "lo" },
      { type: "finish", usage: { inputTokens: 4, outputTokens: 2 } },
    ];
    const { stream } = await mw.wrapStream({
      doStream: async () => ({ stream: sourceStream(chunks) }),
      params: { prompt: "hi" },
      model: { modelId: "gpt-4o" },
    });
    await drain(stream);
    await flush();

    const span = llmSpan();
    expect(span.model).toBe("gpt-4o");
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "Hello" }] });
    expect(span.token_usage).toEqual({ prompt_tokens: 4, completion_tokens: 2 });
  });

  it("wrapStream finalizes the span as ERROR when the stream errors mid-read", async () => {
    const mw = pandaProbeMiddleware();
    const { stream } = await mw.wrapStream({
      doStream: async () => ({
        stream: sourceStream([{ type: "text-delta", delta: "x" }], new Error("stream blew up")),
      }),
      params: { prompt: "hi" },
      model: { modelId: "m" },
    });
    await expect(drain(stream)).rejects.toThrow("stream blew up");
    await flush();
    // Without the fix the span would leak and no trace would be submitted at all.
    expect(llmSpan().status).toBe("ERROR");
  });

  it("wrapStream finalizes the span when the consumer cancels early", async () => {
    const mw = pandaProbeMiddleware();
    const { stream } = await mw.wrapStream({
      doStream: async () => ({
        stream: sourceStream([
          { type: "text-delta", delta: "Hel" },
          { type: "text-delta", delta: "lo" },
        ]),
      }),
      params: { prompt: "hi" },
      model: { modelId: "m" },
    });
    const reader = stream.getReader();
    await reader.read(); // consume one chunk
    await reader.cancel(); // abandon the rest
    await flush();

    const span = llmSpan();
    expect(span.status).toBe("OK");
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "Hel" }] }); // partial
  });
});
