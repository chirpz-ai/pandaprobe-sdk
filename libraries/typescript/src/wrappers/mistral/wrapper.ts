/** wrapMistral — automatic LLM span instrumentation for the Mistral AI JS client. */

import type { SpanContext } from "../../tracing/span.js";
import { closeLlmSpan, errorLlmSpan, safeSerialize, tokenUsage, wrapAsyncStream } from "../base.js";
import { enterMistralSpan } from "./utils.js";

type Any = any;

/**
 * Instrument a `Mistral` client.
 *
 * Patches `chat.complete` (blocking) and `chat.stream` (streaming — events
 * expose the chunk on `.data`). Returns the same client instance (mutated).
 */
export function wrapMistral<T>(client: T): T {
  patchComplete(client as Any);
  patchStream(client as Any);
  return client;
}

function patchComplete(client: Any): void {
  const chat = client?.chat;
  if (!chat || typeof chat.complete !== "function") {
    return;
  }
  const original = chat.complete.bind(chat);
  chat.complete = async (params: Any, ...rest: Any[]): Promise<Any> => {
    const span = enterMistralSpan(params ?? {}, "mistral-chat");
    try {
      const response = await original(params, ...rest);
      finishMistralSpan(span, response);
      return response;
    } catch (exc) {
      errorLlmSpan(span, exc);
      throw exc;
    }
  };
}

function patchStream(client: Any): void {
  const chat = client?.chat;
  if (!chat || typeof chat.stream !== "function") {
    return;
  }
  const original = chat.stream.bind(chat);
  chat.stream = async (params: Any, ...rest: Any[]): Promise<Any> => {
    const span = enterMistralSpan(params ?? {}, "mistral-chat");
    try {
      const stream = await original(params, ...rest);
      return wrapAsyncStream(stream as AsyncIterable<Any>, span, reduceMistralStream);
    } catch (exc) {
      errorLlmSpan(span, exc);
      throw exc;
    }
  };
}

function extractMistralUsage(usage: Any): Record<string, number> | null {
  if (!usage) {
    return null;
  }
  return tokenUsage({
    prompt_tokens: usage.promptTokens ?? usage.prompt_tokens,
    completion_tokens: usage.completionTokens ?? usage.completion_tokens,
    total_tokens: usage.totalTokens ?? usage.total_tokens,
  });
}

function setUsage(span: SpanContext, usage: Record<string, number> | null): void {
  if (!usage) {
    return;
  }
  const { prompt_tokens, completion_tokens, ...extra } = usage;
  span.setTokenUsage({ promptTokens: prompt_tokens ?? 0, completionTokens: completion_tokens ?? 0, ...extra });
}

function reduceMistralStream(span: SpanContext, events: Any[]): void {
  const contentParts: string[] = [];
  let model: string | undefined;
  let usage: Record<string, number> | null = null;

  for (const event of events) {
    const chunk = event?.data ?? event;
    if (chunk?.model) {
      model = chunk.model;
    }
    const delta = chunk?.choices?.[0]?.delta;
    if (delta && typeof delta.content === "string" && delta.content) {
      contentParts.push(delta.content);
    }
    if (chunk?.usage) {
      usage = extractMistralUsage(chunk.usage) ?? usage;
    }
  }

  try {
    if (contentParts.length > 0) {
      span.setOutput({ messages: [{ role: "assistant", content: contentParts.join("") }] });
    }
    if (model) {
      span.setModel(model);
    }
    setUsage(span, usage);
  } catch {
    // best-effort
  }
  closeLlmSpan(span);
}

function finishMistralSpan(span: SpanContext | null, response: Any): void {
  if (span === null) {
    return;
  }
  try {
    const message = response?.choices?.[0]?.message;
    if (message != null) {
      const serialized = safeSerialize(message);
      if (serialized && typeof serialized === "object" && !Array.isArray(serialized)) {
        const copy = { role: "assistant", ...(serialized as Record<string, unknown>) };
        span.setOutput({ messages: [copy] });
      } else {
        span.setOutput({ messages: [{ role: "assistant", content: serialized }] });
      }
    }
  } catch {
    // ignore
  }

  try {
    if (response?.model) {
      span.setModel(response.model);
    }
  } catch {
    // ignore
  }

  try {
    setUsage(span, extractMistralUsage(response?.usage));
  } catch {
    // ignore
  }

  closeLlmSpan(span);
}
