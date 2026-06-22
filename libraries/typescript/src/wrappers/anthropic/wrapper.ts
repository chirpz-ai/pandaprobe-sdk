/** wrapAnthropic — automatic LLM span instrumentation for the Anthropic JS client. */

import type { SpanContext } from "../../tracing/span.js";
import { closeLlmSpan, errorLlmSpan, safeSerialize, tokenUsage, wrapAsyncStream } from "../base.js";
import { enterAnthropicSpan } from "./utils.js";

type Any = any;

/**
 * Instrument an `Anthropic` client.
 *
 * Patches `messages.create` (blocking + `stream: true`) and `messages.stream`
 * (the rich MessageStream API). Returns the same client instance (mutated).
 */
export function wrapAnthropic<T>(client: T): T {
  patchMessagesCreate(client as Any);
  patchMessagesStream(client as Any);
  return client;
}

function patchMessagesCreate(client: Any): void {
  const messages = client?.messages;
  if (!messages || typeof messages.create !== "function") {
    return;
  }
  const original = messages.create.bind(messages);
  messages.create = async (params: Any, ...rest: Any[]): Promise<Any> => {
    const cleaned = { ...(params ?? {}) };
    const span = enterAnthropicSpan(cleaned, "anthropic-messages");
    try {
      const response = await original(params, ...rest);
      if (cleaned.stream) {
        return wrapAsyncStream(response as AsyncIterable<Any>, span, reduceAnthropicStream);
      }
      finishAnthropicSpan(span, response);
      return response;
    } catch (exc) {
      errorLlmSpan(span, exc);
      throw exc;
    }
  };
}

function patchMessagesStream(client: Any): void {
  const messages = client?.messages;
  if (!messages || typeof messages.stream !== "function") {
    return;
  }
  const original = messages.stream.bind(messages);
  messages.stream = (params: Any, ...rest: Any[]): Any => {
    const cleaned = { ...(params ?? {}) };
    const span = enterAnthropicSpan(cleaned, "anthropic-messages");
    let stream: Any;
    try {
      stream = original(params, ...rest);
    } catch (exc) {
      errorLlmSpan(span, exc);
      throw exc;
    }
    attachStreamListeners(stream, span);
    return stream;
  };
}

/**
 * Attach event listeners to a MessageStream so the span finalizes when the
 * stream completes or errors, without consuming the stream from the user.
 */
function attachStreamListeners(stream: Any, span: SpanContext | null): void {
  if (span === null || typeof stream?.on !== "function") {
    return;
  }
  let finalized = false;
  let firstToken = true;

  const finalizeWith = (message: Any): void => {
    if (finalized) {
      return;
    }
    finalized = true;
    finishAnthropicSpan(span, message);
  };

  stream.on("streamEvent", () => {
    if (firstToken) {
      span.setCompletionStartTime(new Date());
      firstToken = false;
    }
  });
  stream.on("finalMessage", (message: Any) => finalizeWith(message));
  stream.on("error", (err: Any) => {
    if (!finalized) {
      finalized = true;
      errorLlmSpan(span, err);
    }
  });
  stream.on("end", () => {
    if (!finalized) {
      finalized = true;
      closeLlmSpan(span);
    }
  });
}

// ---------------------------------------------------------------------------
// Stream reduction (messages.create with stream: true)
// ---------------------------------------------------------------------------

function reduceAnthropicStream(span: SpanContext, events: Any[]): void {
  const textParts: string[] = [];
  const thinkingParts: string[] = [];
  let model: string | undefined;
  let promptTokens: number | undefined;
  let completionTokens: number | undefined;
  let cacheRead: number | undefined;
  let cacheCreation: number | undefined;

  for (const event of events) {
    const type = event?.type;
    if (type === "message_start") {
      const msg = event.message;
      if (msg) {
        model = msg.model ?? model;
        const u = msg.usage;
        if (u) {
          promptTokens = u.input_tokens ?? 0;
          if (u.cache_read_input_tokens) {
            cacheRead = u.cache_read_input_tokens;
          }
          if (u.cache_creation_input_tokens) {
            cacheCreation = u.cache_creation_input_tokens;
          }
        }
      }
    } else if (type === "content_block_delta") {
      const delta = event.delta;
      if (delta) {
        if (delta.type === "thinking_delta" && delta.thinking) {
          thinkingParts.push(delta.thinking);
        } else if (delta.text) {
          textParts.push(delta.text);
        }
      }
    } else if (type === "message_delta") {
      if (event.usage) {
        completionTokens = event.usage.output_tokens ?? 0;
      }
    }
  }

  try {
    if (textParts.length > 0) {
      span.setOutput({ messages: [{ role: "assistant", content: textParts.join("") }] });
    }
    if (thinkingParts.length > 0) {
      span.setMetadata({ reasoning_summary: thinkingParts.join("\n\n") });
    }
    if (model) {
      span.setModel(model);
    }
    const usage = tokenUsage({
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      cache_read_tokens: cacheRead,
      cache_creation_tokens: cacheCreation,
    });
    if (usage) {
      const { prompt_tokens, completion_tokens, ...extra } = usage;
      span.setTokenUsage({ promptTokens: prompt_tokens ?? 0, completionTokens: completion_tokens ?? 0, ...extra });
    }
  } catch {
    // best-effort
  }
  closeLlmSpan(span);
}

// ---------------------------------------------------------------------------
// Blocking response extraction
// ---------------------------------------------------------------------------

function splitContentBlocks(response: Any): { textParts: string[]; thinkingParts: string[] } {
  const textParts: string[] = [];
  const thinkingParts: string[] = [];
  const content = response?.content;
  if (!Array.isArray(content)) {
    return { textParts, thinkingParts };
  }
  for (const block of content) {
    if (block?.type === "thinking" && block.thinking) {
      thinkingParts.push(block.thinking);
    } else if (block?.type === "text" && block.text) {
      textParts.push(block.text);
    }
  }
  return { textParts, thinkingParts };
}

function extractAnthropicUsage(usage: Any): Record<string, number> | null {
  if (!usage) {
    return null;
  }
  const prompt = usage.input_tokens ?? 0;
  const completion = usage.output_tokens ?? 0;
  const extra: Record<string, number> = {};
  if (usage.cache_read_input_tokens > 0) {
    extra.cache_read_tokens = usage.cache_read_input_tokens;
  }
  if (usage.cache_creation_input_tokens > 0) {
    extra.cache_creation_tokens = usage.cache_creation_input_tokens;
  }
  if (prompt === 0 && completion === 0 && Object.keys(extra).length === 0) {
    return null;
  }
  return { prompt_tokens: prompt, completion_tokens: completion, ...extra };
}

function finishAnthropicSpan(span: SpanContext | null, response: Any): void {
  if (span === null) {
    return;
  }
  try {
    const { textParts, thinkingParts } = splitContentBlocks(response);
    if (textParts.length > 0) {
      span.setOutput({ messages: [{ role: "assistant", content: textParts.join("") }] });
    } else if (thinkingParts.length === 0) {
      const serialized = safeSerialize(response);
      if (serialized && typeof serialized === "object" && !Array.isArray(serialized)) {
        const { type: _type, ...copy } = serialized as Record<string, unknown>;
        span.setOutput({ messages: [copy] });
      } else {
        span.setOutput({ messages: [{ role: "assistant", content: serialized }] });
      }
    }
    if (thinkingParts.length > 0) {
      span.setMetadata({ reasoning_summary: thinkingParts.join("\n\n") });
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
    const usage = extractAnthropicUsage(response?.usage);
    if (usage) {
      const { prompt_tokens, completion_tokens, ...extra } = usage;
      span.setTokenUsage({ promptTokens: prompt_tokens ?? 0, completionTokens: completion_tokens ?? 0, ...extra });
    }
  } catch {
    // ignore
  }

  closeLlmSpan(span);
}
