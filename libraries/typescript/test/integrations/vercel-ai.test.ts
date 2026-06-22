import { beforeEach, describe, expect, it } from "vitest";
import { flush, init } from "../../src/index.js";
import { pandaProbeMiddleware } from "../../src/integrations/vercel-ai/index.js";
import { requestsTo } from "../helpers.js";

beforeEach(() => {
  init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
});

function llmSpan(): Record<string, unknown> {
  const body = requestsTo("/traces").at(-1)?.body as Record<string, unknown>;
  const spans = body.spans as Array<Record<string, unknown>>;
  return spans.find((s) => s.kind === "LLM") as Record<string, unknown>;
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
});
