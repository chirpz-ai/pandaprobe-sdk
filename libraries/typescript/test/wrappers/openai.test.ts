import { beforeEach, describe, expect, it } from "vitest";
import { flush, init } from "../../src/index.js";
import { wrapOpenAI } from "../../src/wrappers/openai/index.js";
import { requestsTo } from "../helpers.js";

beforeEach(() => {
  init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
});

function lastTrace(): Record<string, unknown> {
  const reqs = requestsTo("/traces");
  return reqs[reqs.length - 1]?.body as Record<string, unknown>;
}

function llmSpan(): Record<string, unknown> {
  const spans = lastTrace().spans as Array<Record<string, unknown>>;
  return spans.find((s) => s.kind === "LLM") as Record<string, unknown>;
}

describe("wrapOpenAI — chat.completions", () => {
  it("emits an LLM span for a blocking call", async () => {
    const client: any = {
      chat: {
        completions: {
          create: async () => ({
            model: "gpt-4o-mini",
            choices: [{ message: { role: "assistant", content: "hi there" } }],
            usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
          }),
        },
      },
    };
    wrapOpenAI(client);
    const res = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hello" }],
      temperature: 0.5,
    });
    expect(res.choices[0].message.content).toBe("hi there");
    await flush();

    const span = llmSpan();
    expect(span.kind).toBe("LLM");
    expect(span.model).toBe("gpt-4o-mini");
    expect(span.input).toEqual({ messages: [{ role: "user", content: "hello" }] });
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "hi there" }] });
    expect(span.token_usage).toEqual({ prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 });
    expect(span.model_parameters).toEqual({ temperature: 0.5 });
  });

  it("reduces a streaming call into a single output", async () => {
    async function* gen() {
      yield { model: "gpt-4o", choices: [{ delta: { content: "Hel" } }] };
      yield { model: "gpt-4o", choices: [{ delta: { content: "lo" } }] };
      yield { model: "gpt-4o", choices: [{ delta: {} }], usage: { prompt_tokens: 3, completion_tokens: 1 } };
    }
    const client: any = {
      chat: { completions: { create: async () => gen() } },
    };
    wrapOpenAI(client);
    const stream = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });
    const collected: string[] = [];
    for await (const chunk of stream) {
      const c = chunk.choices[0].delta.content;
      if (c) collected.push(c);
    }
    expect(collected.join("")).toBe("Hello");
    await flush();

    const span = llmSpan();
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "Hello" }] });
    expect(span.token_usage).toEqual({ prompt_tokens: 3, completion_tokens: 1 });
    expect(span.completion_start_time).toBeTruthy();
  });

  it("finalizes the span as ERROR when a blocking call throws", async () => {
    const client: any = {
      chat: {
        completions: {
          create: async () => {
            throw new Error("api down");
          },
        },
      },
    };
    wrapOpenAI(client);
    await expect(
      client.chat.completions.create({ model: "gpt-4o", messages: [{ role: "user", content: "x" }] }),
    ).rejects.toThrow("api down");
    await flush();
    const span = llmSpan();
    expect(span.status).toBe("ERROR");
    expect(span.error).toContain("api down");
  });
});

describe("wrapOpenAI — responses API", () => {
  it("extracts message output and creates tool child spans", async () => {
    const client: any = {
      responses: {
        create: async () => ({
          model: "gpt-4.1",
          output: [
            { type: "message", role: "assistant", content: [{ type: "output_text", text: "done" }] },
            { type: "web_search_call", status: "completed" },
            { type: "function_call", name: "get_weather", arguments: '{"city":"SF"}' },
          ],
          usage: { input_tokens: 10, output_tokens: 4 },
        }),
      },
    };
    wrapOpenAI(client);
    await client.responses.create({ model: "gpt-4.1", input: "weather?" });
    await flush();

    const spans = lastTrace().spans as Array<Record<string, unknown>>;
    const llm = spans.find((s) => s.kind === "LLM")!;
    expect(llm.token_usage).toEqual({ prompt_tokens: 10, completion_tokens: 4 });
    const tools = spans.filter((s) => s.kind === "TOOL");
    expect(tools.map((t) => t.name).sort()).toEqual(["function_call:get_weather", "web_search"]);
  });
});
