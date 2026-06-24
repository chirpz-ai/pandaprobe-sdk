import { beforeEach, describe, expect, it } from "vitest";
import { flush, init } from "../../src/index.js";
import { wrapAnthropic } from "../../src/wrappers/anthropic/index.js";
import { wrapBedrock } from "../../src/wrappers/bedrock/index.js";
import { wrapGemini } from "../../src/wrappers/gemini/index.js";
import { wrapMistral } from "../../src/wrappers/mistral/index.js";
import { requestsTo } from "../helpers.js";

beforeEach(() => {
  init({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
});

function llmSpan(): Record<string, unknown> {
  const body = requestsTo("/traces").at(-1)?.body as Record<string, unknown>;
  const spans = body.spans as Array<Record<string, unknown>>;
  return spans.find((s) => s.kind === "LLM") as Record<string, unknown>;
}

describe("wrapAnthropic", () => {
  it("normalizes system + messages and extracts text/usage", async () => {
    const client: any = {
      messages: {
        create: async () => ({
          model: "claude-sonnet-4",
          content: [{ type: "text", text: "Hello!" }],
          usage: { input_tokens: 8, output_tokens: 3 },
        }),
      },
    };
    wrapAnthropic(client);
    await client.messages.create({
      model: "claude-sonnet-4",
      system: "Be brief.",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 100,
    });
    await flush();

    const span = llmSpan();
    expect(span.model).toBe("claude-sonnet-4");
    expect(span.input).toEqual({
      messages: [
        { role: "system", content: "Be brief." },
        { role: "user", content: "hi" },
      ],
    });
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "Hello!" }] });
    expect(span.token_usage).toEqual({ prompt_tokens: 8, completion_tokens: 3 });
    expect(span.model_parameters).toEqual({ max_tokens: 100 });
  });
});

describe("wrapMistral", () => {
  it("extracts message and usage from chat.complete", async () => {
    const client: any = {
      chat: {
        complete: async () => ({
          model: "mistral-large",
          choices: [{ message: { role: "assistant", content: "bonjour" } }],
          usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 },
        }),
      },
    };
    wrapMistral(client);
    await client.chat.complete({ model: "mistral-large", messages: [{ role: "user", content: "salut" }] });
    await flush();

    const span = llmSpan();
    expect(span.model).toBe("mistral-large");
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "bonjour" }] });
    expect(span.token_usage).toEqual({ prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 });
  });
});

describe("wrapGemini", () => {
  it("normalizes contents and splits answer vs thought", async () => {
    const client: any = {
      models: {
        generateContent: async () => ({
          candidates: [
            {
              content: {
                role: "model",
                parts: [{ text: "thinking...", thought: true }, { text: "42" }],
              },
            },
          ],
          usageMetadata: { promptTokenCount: 6, candidatesTokenCount: 1, totalTokenCount: 7, thoughtsTokenCount: 2 },
        }),
      },
    };
    wrapGemini(client);
    await client.models.generateContent({
      model: "gemini-2.5-flash",
      contents: "what is 6*7?",
      config: { temperature: 0.2 },
    });
    await flush();

    const span = llmSpan();
    expect(span.model).toBe("gemini-2.5-flash");
    expect(span.input).toEqual({ messages: [{ role: "user", content: "what is 6*7?" }] });
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "42" }] });
    expect(span.token_usage).toMatchObject({ prompt_tokens: 6, completion_tokens: 1, reasoning_tokens: 2 });
    expect((span.metadata as Record<string, unknown>).reasoning_summary).toBe("thinking...");
    expect(span.model_parameters).toEqual({ temperature: 0.2 });
  });
});

describe("wrapBedrock", () => {
  it("instruments a ConverseCommand via client.send", async () => {
    class ConverseCommand {
      input: unknown;
      constructor(input: unknown) {
        this.input = input;
      }
    }
    const client: any = {
      send: async () => ({
        output: { message: { role: "assistant", content: [{ text: "hi from bedrock" }] } },
        usage: { inputTokens: 12, outputTokens: 5, totalTokens: 17 },
      }),
    };
    wrapBedrock(client);
    await client.send(
      new ConverseCommand({
        modelId: "anthropic.claude-3-5-sonnet",
        system: [{ text: "Be concise." }],
        messages: [{ role: "user", content: [{ text: "hello" }] }],
        inferenceConfig: { temperature: 0.7, maxTokens: 256 },
      }),
    );
    await flush();

    const span = llmSpan();
    expect(span.model).toBe("anthropic.claude-3-5-sonnet");
    expect(span.input).toEqual({
      messages: [
        { role: "system", content: "Be concise." },
        { role: "user", content: "hello" },
      ],
    });
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "hi from bedrock" }] });
    expect(span.token_usage).toEqual({ prompt_tokens: 12, completion_tokens: 5, total_tokens: 17 });
    expect(span.model_parameters).toEqual({ temperature: 0.7, maxTokens: 256 });
  });

  it("reduces a ConverseStreamCommand", async () => {
    class ConverseStreamCommand {
      input: unknown;
      constructor(input: unknown) {
        this.input = input;
      }
    }
    async function* stream() {
      yield { contentBlockDelta: { delta: { text: "Hel" } } };
      yield { contentBlockDelta: { delta: { text: "lo" } } };
      yield { metadata: { usage: { inputTokens: 2, outputTokens: 1, totalTokens: 3 } } };
    }
    const client: any = {
      send: async () => ({ stream: stream() }),
    };
    wrapBedrock(client);
    const res = await client.send(
      new ConverseStreamCommand({
        modelId: "amazon.nova-pro",
        messages: [{ role: "user", content: [{ text: "hi" }] }],
      }),
    );
    for await (const _ of res.stream) {
      // consume
    }
    await flush();

    const span = llmSpan();
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "Hello" }] });
    expect(span.token_usage).toEqual({ prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 });
  });
});
