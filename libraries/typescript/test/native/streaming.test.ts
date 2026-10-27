import { expect, it } from "vitest";
import { optionalModule } from "../../src/reflect/integrations.js";
import { fixture } from "../reflect/helpers.js";
it.each(["success", "error"])("Vercel consumes native streams and preserves %s", async (mode) => {
  const f = fixture();
  const ai = await optionalModule("ai");
  const { MockLanguageModelV3 } = await optionalModule("ai/test");
  let finished = false;
  const model = new MockLanguageModelV3({
    doStream: async () => ({
      stream: new ReadableStream({
        start(controller) {
          controller.enqueue({ type: "stream-start", warnings: [] });
          if (mode === "error") controller.enqueue({ type: "error", error: new Error("provider stream detail") });
          else {
            controller.enqueue({ type: "text-start", id: "text" });
            controller.enqueue({ type: "text-delta", id: "text", delta: "native final" });
            controller.enqueue({ type: "text-end", id: "text" });
            controller.enqueue({
              type: "finish",
              finishReason: { unified: "stop", raw: "stop" },
              usage: { inputTokens: { total: 1 }, outputTokens: { total: 1 } },
            });
          }
          finished = true;
          controller.close();
        },
      }),
    }),
  });
  const agent = await f.runtime.agent(ai.ToolLoopAgent, {
    framework: "vercel-ai",
    options: { model, instructions: "host" },
    invokeOptions: { stream: true },
  });
  const run = f.runtime.run(agent, "stream", { sessionId: "s", executionId: "stream" });
  if (mode === "error") {
    await expect(run).rejects.toThrow("provider stream detail");
    expect(f.runtime.report("stream").task_status).toBe("failed");
  } else {
    const result = await run;
    expect(await result.text).toBe("native final");
  }
  expect(finished).toBe(true);
});
it("Vercel preserves dynamic host hooks while composing one memory block", async () => {
  const f = fixture();
  const ai = await optionalModule("ai");
  const { MockLanguageModelV3 } = await optionalModule("ai/test");
  const requests: any[] = [];
  const model = new MockLanguageModelV3({
    doGenerate: async (request: any) => {
      requests.push(request);
      return {
        content: [{ type: "text", text: "done" }],
        finishReason: { unified: "stop", raw: "stop" },
        usage: { inputTokens: { total: 1 }, outputTokens: { total: 1 } },
        warnings: [],
      };
    },
  });
  const agent = await f.runtime.agent(ai.ToolLoopAgent, {
    framework: "vercel-ai",
    options: {
      model,
      instructions: "static",
      prepareCall: (options: any) => ({ ...options, instructions: "dynamic host", activeTools: [] }),
      prepareStep: () => ({ providerOptions: { fixture: { custom: true } } }),
    },
  });
  await f.runtime.run(agent, "x", { sessionId: "s" });
  const text = JSON.stringify(requests[0].prompt);
  expect(text).toContain("dynamic host");
  expect(text.split("Reflect memory")).toHaveLength(2);
  expect(requests[0].tools.map((tool: any) => tool.name)).toContain("reflect_bash");
});
it.each(["success", "error"])("OpenAI native streamed results preserve %s", async (mode) => {
  const f = fixture();
  const sdk = await optionalModule("@openai/agents");
  sdk.setTraceProcessors([]);
  let finished = false;
  const model = {
    async *getStreamedResponse() {
      yield { type: "response_started" };
      if (mode === "error") throw Error("native stream failed");
      yield {
        type: "response_done",
        response: {
          id: "response",
          usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
          output: [
            {
              type: "message",
              role: "assistant",
              status: "completed",
              content: [{ type: "output_text", text: "native final" }],
            },
          ],
        },
      };
      finished = true;
    },
    getResponse() {
      throw Error("stream required");
    },
  };
  const agent = await f.runtime.agent(sdk.Agent, {
    framework: "openai-agents",
    options: { name: "host", model },
    invokeOptions: { stream: true },
  });
  const run = f.runtime.run(agent, "x", { sessionId: "s", executionId: "stream" });
  if (mode === "error") {
    await expect(run).rejects.toThrow("native stream failed");
    expect(f.runtime.report("stream").task_status).toBe("failed");
  } else {
    expect((await run).finalOutput).toBe("native final");
    expect(finished).toBe(true);
  }
});
