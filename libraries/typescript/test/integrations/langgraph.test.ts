import { beforeEach, describe, expect, it } from "vitest";
import { flush, init } from "../../src/index.js";
import { LangGraphCallbackHandler } from "../../src/integrations/langgraph/index.js";
import { requestsTo } from "../helpers.js";

type Any = any;

beforeEach(() => {
  init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
});

function lastTrace(): Record<string, unknown> {
  return requestsTo("/traces").at(-1)?.body as Record<string, unknown>;
}

describe("LangGraphCallbackHandler", () => {
  it("maps a chain→LLM run into a single trace", async () => {
    const h = new LangGraphCallbackHandler();

    h.handleChainStart({ name: "LangGraph" }, { messages: [{ role: "user", content: "hi" }] }, "root");
    h.handleChatModelStart({ name: "ChatOpenAI" }, [[{ _getType: () => "human", content: "hi" }]], "llm1", "root", {
      invocation_params: { model: "gpt-4o", temperature: 0.3 },
    });
    h.handleLLMEnd(
      {
        generations: [
          [
            {
              message: {
                _getType: () => "ai",
                content: "hello",
                usage_metadata: { input_tokens: 5, output_tokens: 2, total_tokens: 7 },
              },
            },
          ],
        ],
      },
      "llm1",
    );
    h.handleChainEnd({ messages: [{ role: "assistant", content: "hello" }] }, "root");
    await flush();

    const body = lastTrace();
    expect(body.name).toBe("LangGraph");
    expect(body.status).toBe("COMPLETED");
    expect(body.input).toEqual({ messages: [{ role: "user", content: "hi" }] });
    expect(body.output).toEqual({ messages: [{ role: "assistant", content: "hello" }] });

    const spans = body.spans as Array<Record<string, unknown>>;
    const root = spans.find((s) => s.span_id === "root")!;
    const llm = spans.find((s) => s.span_id === "llm1")!;
    expect(root.kind).toBe("CHAIN");
    expect(llm.kind).toBe("LLM");
    expect(llm.parent_span_id).toBe("root");
    expect(llm.model).toBe("gpt-4o");
    expect(llm.model_parameters).toEqual({ temperature: 0.3 });
    expect(llm.output).toEqual({ messages: [{ role: "assistant", content: "hello" }] });
    expect(llm.token_usage).toEqual({ prompt_tokens: 5, total_tokens: 7, completion_tokens: 2 });
  });

  it("captures tool spans and marks chain errors", async () => {
    const h = new LangGraphCallbackHandler();
    h.handleChainStart({ name: "LangGraph" }, { messages: [{ role: "user", content: "go" }] }, "root");
    h.handleToolStart({ name: "search" }, "query text", "tool1", "root");
    h.handleToolEnd("results", "tool1");
    h.handleChainError(new Error("boom"), "root");
    await flush();

    const body = lastTrace();
    expect(body.status).toBe("ERROR");
    const spans = body.spans as Array<Record<string, unknown>>;
    const tool = spans.find((s) => s.span_id === "tool1")!;
    expect(tool.kind).toBe("TOOL");
    expect(tool.output).toBe("results");
  });

  it("keeps concurrent interleaved runs on one handler in separate traces", async () => {
    // Regression: a single handler instance shared across concurrent invoke()
    // calls must not merge their spans. Interleave two runs (roots A and B).
    const h = new LangGraphCallbackHandler();
    const llmEnd = (content: string) => ({
      generations: [[{ message: { _getType: () => "ai", content } }]],
    });

    h.handleChainStart({ name: "LangGraph" }, { messages: [{ role: "user", content: "A" }] }, "rootA");
    h.handleChainStart({ name: "LangGraph" }, { messages: [{ role: "user", content: "B" }] }, "rootB");
    h.handleChatModelStart({ name: "m" }, [[{ _getType: () => "human", content: "A" }]], "llmA", "rootA");
    h.handleChatModelStart({ name: "m" }, [[{ _getType: () => "human", content: "B" }]], "llmB", "rootB");
    h.handleLLMEnd(llmEnd("answer A"), "llmA");
    h.handleLLMEnd(llmEnd("answer B"), "llmB");
    h.handleChainEnd({ messages: [{ role: "assistant", content: "answer B" }] }, "rootB");
    h.handleChainEnd({ messages: [{ role: "assistant", content: "answer A" }] }, "rootA");
    await flush();

    const traces = requestsTo("/traces").map((r) => r.body as Record<string, unknown>);
    const byInput = (c: string) =>
      traces.find((t) => (t.input as Any)?.messages?.[0]?.content === c) as Record<string, unknown>;

    const a = byInput("A");
    const b = byInput("B");
    const aSpans = a.spans as Array<Record<string, unknown>>;
    const bSpans = b.spans as Array<Record<string, unknown>>;

    // Each trace holds exactly its own root + LLM span — no cross-contamination.
    expect(aSpans.map((s) => s.span_id).sort()).toEqual(["llmA", "rootA"]);
    expect(bSpans.map((s) => s.span_id).sort()).toEqual(["llmB", "rootB"]);
    expect(aSpans.find((s) => s.span_id === "llmA")?.parent_span_id).toBe("rootA");
    expect(bSpans.find((s) => s.span_id === "llmB")?.parent_span_id).toBe("rootB");
    expect(a.output).toEqual({ messages: [{ role: "assistant", content: "answer A" }] });
    expect(b.output).toEqual({ messages: [{ role: "assistant", content: "answer B" }] });
  });
});
