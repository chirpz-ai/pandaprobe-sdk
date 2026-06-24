import { beforeEach, describe, expect, it } from "vitest";
import { flush, init } from "../../src/index.js";
import { wrapAnthropic } from "../../src/wrappers/anthropic/index.js";
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

// Minimal event-emitter mock for Anthropic's MessageStream.
function makeEmitter() {
  const handlers: Record<string, Array<(arg: unknown) => void>> = {};
  return {
    on(ev: string, cb: (arg: unknown) => void) {
      if (!handlers[ev]) {
        handlers[ev] = [];
      }
      handlers[ev].push(cb);
      return this;
    },
    emit(ev: string, arg?: unknown) {
      for (const cb of handlers[ev] ?? []) {
        cb(arg);
      }
    },
  };
}

describe("wrapAnthropic — messages.stream (event API)", () => {
  it("finalizes the span from the finalMessage event", async () => {
    const streamObj = makeEmitter();
    const client: any = { messages: { stream: () => streamObj } };
    wrapAnthropic(client);

    const s: any = client.messages.stream({ model: "claude", messages: [{ role: "user", content: "hi" }] });
    s.emit("streamEvent", {});
    s.emit("finalMessage", {
      model: "claude-sonnet-4",
      content: [{ type: "text", text: "hello there" }],
      usage: { input_tokens: 7, output_tokens: 3 },
    });
    await flush();

    const span = llmSpan();
    expect(span.model).toBe("claude-sonnet-4");
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "hello there" }] });
    expect(span.token_usage).toEqual({ prompt_tokens: 7, completion_tokens: 3 });
    expect(span.completion_start_time).toBeTruthy();
  });

  it("finalizes the span as ERROR on the error event", async () => {
    const streamObj = makeEmitter();
    const client: any = { messages: { stream: () => streamObj } };
    wrapAnthropic(client);
    const s: any = client.messages.stream({ model: "claude", messages: [{ role: "user", content: "hi" }] });
    s.emit("error", new Error("stream failed"));
    await flush();
    expect(llmSpan().status).toBe("ERROR");
  });
});

describe("wrapAnthropic — messages.create stream:true", () => {
  it("reduces raw stream events incl. thinking + cache tokens", async () => {
    async function* gen() {
      yield {
        type: "message_start",
        message: { model: "claude-sonnet-4", usage: { input_tokens: 8, cache_read_input_tokens: 2 } },
      };
      yield { type: "content_block_delta", delta: { type: "thinking_delta", thinking: "hmm" } };
      yield { type: "content_block_delta", delta: { type: "text_delta", text: "Hel" } };
      yield { type: "content_block_delta", delta: { type: "text_delta", text: "lo" } };
      yield { type: "message_delta", usage: { output_tokens: 5 } };
    }
    const client: any = { messages: { create: async () => gen() } };
    wrapAnthropic(client);
    const stream = await client.messages.create({
      model: "claude-sonnet-4",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });
    for await (const _ of stream) {
      // consume
    }
    await flush();

    const span = llmSpan();
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "Hello" }] });
    expect(span.token_usage).toEqual({ prompt_tokens: 8, completion_tokens: 5, cache_read_tokens: 2 });
    expect((span.metadata as Record<string, unknown>).reasoning_summary).toBe("hmm");
  });
});

describe("wrapGemini — generateContentStream", () => {
  it("reduces chunks and splits thought vs answer", async () => {
    async function* gen() {
      yield { candidates: [{ content: { parts: [{ text: "4" }] } }] };
      yield {
        candidates: [{ content: { parts: [{ text: "2", thought: false }] } }],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 2, totalTokenCount: 7 },
      };
    }
    const client: any = { models: { generateContentStream: async () => gen() } };
    wrapGemini(client);
    const stream = await client.models.generateContentStream({ model: "gemini-2.5-flash", contents: "6*7?" });
    for await (const _ of stream) {
      // consume
    }
    await flush();

    const span = llmSpan();
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "42" }] });
    expect(span.token_usage).toEqual({ prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 });
  });
});

describe("wrapMistral — chat.stream", () => {
  it("reduces events that expose the chunk on .data", async () => {
    async function* gen() {
      yield { data: { model: "mistral-large", choices: [{ delta: { content: "Bon" } }] } };
      yield { data: { model: "mistral-large", choices: [{ delta: { content: "jour" } }] } };
      yield { data: { choices: [{ delta: {} }], usage: { promptTokens: 3, completionTokens: 2, totalTokens: 5 } } };
    }
    const client: any = { chat: { stream: async () => gen() } };
    wrapMistral(client);
    const stream = await client.chat.stream({
      model: "mistral-large",
      messages: [{ role: "user", content: "salut" }],
    });
    for await (const _ of stream) {
      // consume
    }
    await flush();

    const span = llmSpan();
    expect(span.output).toEqual({ messages: [{ role: "assistant", content: "Bonjour" }] });
    expect(span.token_usage).toEqual({ prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 });
  });
});

describe("blocking error finalization", () => {
  it("marks the span ERROR for anthropic/gemini/mistral when the call throws", async () => {
    const anthropic: any = {
      messages: {
        create: async () => {
          throw new Error("anthropic down");
        },
      },
    };
    wrapAnthropic(anthropic);
    await expect(
      anthropic.messages.create({ model: "claude", messages: [{ role: "user", content: "x" }] }),
    ).rejects.toThrow("anthropic down");
    await flush();
    expect(llmSpan().status).toBe("ERROR");

    const gemini: any = {
      models: {
        generateContent: async () => {
          throw new Error("gemini down");
        },
      },
    };
    wrapGemini(gemini);
    await expect(gemini.models.generateContent({ model: "gemini-2.5-flash", contents: "x" })).rejects.toThrow(
      "gemini down",
    );
    await flush();
    expect(llmSpan().status).toBe("ERROR");

    const mistral: any = {
      chat: {
        complete: async () => {
          throw new Error("mistral down");
        },
      },
    };
    wrapMistral(mistral);
    await expect(
      mistral.chat.complete({ model: "mistral-large", messages: [{ role: "user", content: "x" }] }),
    ).rejects.toThrow("mistral down");
    await flush();
    expect(llmSpan().status).toBe("ERROR");
  });
});
