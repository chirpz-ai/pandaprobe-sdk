import { beforeEach, describe, expect, it } from "vitest";
import { flush, init } from "../../src/index.js";
import { DeepAgentsCallbackHandler } from "../../src/integrations/deepagents/index.js";
import { requestsTo } from "../helpers.js";

beforeEach(() => {
  init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
});

function lastTrace(): Record<string, unknown> {
  return requestsTo("/traces").at(-1)?.body as Record<string, unknown>;
}

describe("DeepAgentsCallbackHandler", () => {
  it("remaps the LangGraph root name to DeepAgents", async () => {
    const h = new DeepAgentsCallbackHandler();
    h.handleChainStart({ name: "LangGraph" }, { messages: [{ role: "user", content: "hi" }] }, "root");
    h.handleChainEnd({ messages: [{ role: "assistant", content: "ok" }] }, "root");
    await flush();
    expect(lastTrace().name).toBe("DeepAgents");
  });

  it("passes through a user-declared graph name", async () => {
    const h = new DeepAgentsCallbackHandler();
    h.handleChainStart({ name: "research_agent" }, { messages: [{ role: "user", content: "hi" }] }, "root");
    h.handleChainEnd({ messages: [{ role: "assistant", content: "ok" }] }, "root");
    await flush();
    expect(lastTrace().name).toBe("research_agent");
  });

  it("captures a nested sub-agent tool tree as one trace", async () => {
    const h = new DeepAgentsCallbackHandler({ tags: ["deepagents"] });
    h.handleChainStart({ name: "LangGraph" }, { messages: [{ role: "user", content: "research X" }] }, "root");
    h.handleChainStart({ name: "tools" }, {}, "tools1", "root");
    h.handleToolStart({ name: "task" }, "delegate to sub-agent", "task1", "tools1");
    h.handleToolEnd("sub-agent result", "task1");
    h.handleChainEnd({}, "tools1");
    h.handleChainEnd({ messages: [{ role: "assistant", content: "done" }] }, "root");
    await flush();

    const body = lastTrace();
    expect(body.name).toBe("DeepAgents");
    expect(body.tags).toEqual(["deepagents"]);
    const spans = body.spans as Array<Record<string, unknown>>;
    expect(spans.find((s) => s.span_id === "root")?.kind).toBe("CHAIN");
    expect(spans.find((s) => s.span_id === "tools1")?.kind).toBe("AGENT");
    const tool = spans.find((s) => s.span_id === "task1")!;
    expect(tool.kind).toBe("TOOL");
    expect(tool.parent_span_id).toBe("tools1");
    expect(tool.output).toBe("sub-agent result");
  });
});
