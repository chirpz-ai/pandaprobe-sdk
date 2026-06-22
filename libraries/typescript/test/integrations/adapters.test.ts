import { beforeEach, describe, expect, it } from "vitest";
import { flush, init } from "../../src/index.js";
import { wrapClaudeAgentQuery } from "../../src/integrations/claude-agent-sdk/index.js";
import { OpenAIAgentsAdapter } from "../../src/integrations/openai-agents/index.js";
import { requestsTo } from "../helpers.js";

beforeEach(() => {
  init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
});

function lastTrace(): Record<string, unknown> {
  return requestsTo("/traces").at(-1)?.body as Record<string, unknown>;
}

describe("Claude Agent SDK — wrapClaudeAgentQuery", () => {
  it("builds an AGENT → LLM/TOOL trace from streamed messages", async () => {
    async function* fakeQuery() {
      yield {
        type: "assistant",
        message: {
          model: "claude-sonnet-4",
          content: [
            { type: "text", text: "Let me check." },
            { type: "tool_use", id: "t1", name: "get_weather", input: { city: "SF" } },
          ],
          usage: { input_tokens: 10, output_tokens: 4 },
        },
      };
      yield {
        type: "user",
        message: { content: [{ type: "tool_result", tool_use_id: "t1", content: "sunny" }] },
      };
      yield {
        type: "result",
        subtype: "success",
        result: "It's sunny in SF.",
        usage: { input_tokens: 10, output_tokens: 8 },
      };
    }

    const query = wrapClaudeAgentQuery(fakeQuery, { sessionId: "s1" });
    const seen: string[] = [];
    for await (const msg of query({ prompt: "weather in SF?" })) {
      seen.push((msg as { type: string }).type);
    }
    expect(seen).toEqual(["assistant", "user", "result"]);
    await flush();

    const body = lastTrace();
    expect(body.name).toBe("ClaudeAgent");
    expect(body.session_id).toBe("s1");
    expect(body.output).toEqual({ messages: [{ role: "assistant", content: "It's sunny in SF." }] });

    const spans = body.spans as Array<Record<string, unknown>>;
    const agent = spans.find((s) => s.kind === "AGENT")!;
    const llm = spans.find((s) => s.kind === "LLM")!;
    const tool = spans.find((s) => s.kind === "TOOL")!;
    expect(llm.parent_span_id).toBe(agent.span_id);
    expect(tool.name).toBe("get_weather");
    expect(tool.input).toEqual({ city: "SF" });
    expect(tool.output).toBe("sunny");
  });
});

describe("OpenAI Agents — trace processor", () => {
  it("converts SDK trace/span events into a PandaProbe trace", async () => {
    const adapter = new OpenAIAgentsAdapter({ tags: ["agents"] });
    const proc = adapter.createTraceProcessor();

    proc.onTraceStart({ traceId: "tr1", name: "MyAgent" });
    proc.onSpanEnd({
      traceId: "tr1",
      spanId: "a1",
      parentId: null,
      spanData: { type: "agent", name: "planner" },
    });
    proc.onSpanEnd({
      traceId: "tr1",
      spanId: "g1",
      parentId: "a1",
      spanData: { type: "generation", model: "gpt-4o", usage: { inputTokens: 7, outputTokens: 3 } },
    });
    proc.onTraceEnd({ traceId: "tr1", name: "MyAgent" });
    await flush();

    const body = lastTrace();
    expect(body.name).toBe("MyAgent");
    expect(body.tags).toEqual(["agents"]);
    const spans = body.spans as Array<Record<string, unknown>>;
    expect(spans.find((s) => s.span_id === "a1")?.kind).toBe("AGENT");
    const gen = spans.find((s) => s.span_id === "g1")!;
    expect(gen.kind).toBe("LLM");
    expect(gen.model).toBe("gpt-4o");
    expect(gen.token_usage).toEqual({ prompt_tokens: 7, completion_tokens: 3 });
    expect(gen.parent_span_id).toBe("a1");
  });
});
