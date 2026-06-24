import { beforeEach, describe, expect, it } from "vitest";
import { flush, init } from "../../src/index.js";
import { LangChainCallbackHandler } from "../../src/integrations/langchain/index.js";
import { requestsTo } from "../helpers.js";

beforeEach(() => {
  init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
});

function lastTrace(): Record<string, unknown> {
  return requestsTo("/traces").at(-1)?.body as Record<string, unknown>;
}

describe("LangChainCallbackHandler", () => {
  it("remaps the internal root chain name to LangChain", async () => {
    const h = new LangChainCallbackHandler();
    h.handleChainStart({ name: "RunnableSequence" }, { messages: [{ role: "user", content: "hi" }] }, "root");
    h.handleChainEnd({ messages: [{ role: "assistant", content: "ok" }] }, "root");
    await flush();
    expect(lastTrace().name).toBe("LangChain");
  });

  it("classifies nested chains as AGENT and the root as CHAIN", async () => {
    const h = new LangChainCallbackHandler();
    h.handleChainStart({ name: "agent" }, { input: "x" }, "root");
    h.handleChainStart({ name: "inner" }, { input: "y" }, "child", "root");
    h.handleChainEnd({ output: "y2" }, "child");
    h.handleChainEnd({ output: "x2" }, "root");
    await flush();
    const spans = lastTrace().spans as Array<Record<string, unknown>>;
    expect(spans.find((s) => s.span_id === "root")?.kind).toBe("CHAIN");
    expect(spans.find((s) => s.span_id === "child")?.kind).toBe("AGENT");
  });

  it("captures retriever spans and propagates LLM errors", async () => {
    const h = new LangChainCallbackHandler();
    h.handleChainStart({ name: "rag" }, { messages: [{ role: "user", content: "q" }] }, "root");
    h.handleRetrieverStart({ name: "vectorstore" }, "search query", "ret1", "root");
    h.handleRetrieverEnd([{ pageContent: "doc text", metadata: { id: 1 } }], "ret1");
    h.handleLLMStart({ name: "llm" }, ["prompt"], "llm1", "root", { invocation_params: { model: "gpt-4o" } });
    h.handleLLMError(new Error("rate limited"), "llm1");
    h.handleChainEnd({ messages: [{ role: "assistant", content: "" }] }, "root");
    await flush();

    const spans = lastTrace().spans as Array<Record<string, unknown>>;
    const ret = spans.find((s) => s.span_id === "ret1")!;
    expect(ret.kind).toBe("RETRIEVER");
    expect(ret.output).toEqual([{ page_content: "doc text", metadata: { id: 1 } }]);
    const llm = spans.find((s) => s.span_id === "llm1")!;
    expect(llm.status).toBe("ERROR");
    expect(llm.error).toContain("rate limited");
  });

  it("normalizes chat-model messages and extracts reasoning + usage", async () => {
    const h = new LangChainCallbackHandler();
    h.handleChainStart({ name: "agent" }, { messages: [{ role: "user", content: "hi" }] }, "root");
    h.handleChatModelStart({ name: "ChatAnthropic" }, [[{ _getType: () => "human", content: "hi" }]], "llm1", "root", {
      invocation_params: { model: "claude-sonnet-4", temperature: 0.4 },
    });
    h.handleLLMEnd(
      {
        generations: [
          [
            {
              message: {
                _getType: () => "ai",
                content: [
                  { type: "thinking", thinking: "let me reason" },
                  { type: "text", text: "answer" },
                ],
                usage_metadata: { input_tokens: 6, output_tokens: 4, total_tokens: 10 },
              },
            },
          ],
        ],
      },
      "llm1",
    );
    h.handleChainEnd({ messages: [{ role: "assistant", content: "answer" }] }, "root");
    await flush();

    const llm = (lastTrace().spans as Array<Record<string, unknown>>).find((s) => s.span_id === "llm1")!;
    expect(llm.model).toBe("claude-sonnet-4");
    expect(llm.model_parameters).toEqual({ temperature: 0.4 });
    expect(llm.output).toEqual({ messages: [{ role: "assistant", content: "answer" }] });
    expect(llm.token_usage).toEqual({ prompt_tokens: 6, completion_tokens: 4, total_tokens: 10 });
    expect((llm.metadata as Record<string, unknown>).reasoning_summary).toBe("let me reason");
  });
});
