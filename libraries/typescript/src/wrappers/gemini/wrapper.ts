/** wrapGemini — automatic LLM span instrumentation for the Google GenAI JS client. */

import type { SpanContext } from "../../tracing/span.js";
import { closeLlmSpan, errorLlmSpan, safeSerialize, tokenUsage, wrapAsyncStream } from "../base.js";
import { enterGeminiSpan } from "./utils.js";

type Any = any;

/**
 * Instrument a `GoogleGenAI` client.
 *
 * Patches `models.generateContent` (blocking) and `models.generateContentStream`
 * (streaming). Returns the same client instance (mutated).
 */
export function wrapGemini<T>(client: T): T {
  patchGenerate(client as Any);
  patchStream(client as Any);
  return client;
}

function patchGenerate(client: Any): void {
  const models = client?.models;
  if (!models || typeof models.generateContent !== "function") {
    return;
  }
  const original = models.generateContent.bind(models);
  models.generateContent = async (params: Any, ...rest: Any[]): Promise<Any> => {
    const span = enterGeminiSpan(params ?? {}, "gemini-generate");
    try {
      const response = await original(params, ...rest);
      finishGeminiSpan(span, response);
      return response;
    } catch (exc) {
      errorLlmSpan(span, exc);
      throw exc;
    }
  };
}

function patchStream(client: Any): void {
  const models = client?.models;
  if (!models || typeof models.generateContentStream !== "function") {
    return;
  }
  const original = models.generateContentStream.bind(models);
  models.generateContentStream = async (params: Any, ...rest: Any[]): Promise<Any> => {
    const span = enterGeminiSpan(params ?? {}, "gemini-generate");
    try {
      const stream = await original(params, ...rest);
      return wrapAsyncStream(stream as AsyncIterable<Any>, span, reduceGeminiStream);
    } catch (exc) {
      errorLlmSpan(span, exc);
      throw exc;
    }
  };
}

function extractGeminiUsage(usageMetadata: Any): Record<string, number> | null {
  if (!usageMetadata) {
    return null;
  }
  const u = usageMetadata;
  return tokenUsage({
    prompt_tokens: u.promptTokenCount ?? u.prompt_token_count,
    completion_tokens: u.candidatesTokenCount ?? u.candidates_token_count,
    total_tokens: u.totalTokenCount ?? u.total_token_count,
    reasoning_tokens: u.thoughtsTokenCount ?? u.thoughts_token_count,
    cache_read_tokens: u.cachedContentTokenCount ?? u.cached_content_token_count,
  });
}

function setUsage(span: SpanContext, usage: Record<string, number> | null): void {
  if (!usage) {
    return;
  }
  const { prompt_tokens, completion_tokens, ...extra } = usage;
  span.setTokenUsage({ promptTokens: prompt_tokens ?? 0, completionTokens: completion_tokens ?? 0, ...extra });
}

function splitParts(response: Any): { answerTexts: string[]; thoughtTexts: string[] } {
  const answerTexts: string[] = [];
  const thoughtTexts: string[] = [];
  const parts = response?.candidates?.[0]?.content?.parts;
  if (!Array.isArray(parts)) {
    return { answerTexts, thoughtTexts };
  }
  for (const part of parts) {
    const text = part?.text;
    if (!text) {
      continue;
    }
    if (part.thought) {
      thoughtTexts.push(text);
    } else {
      answerTexts.push(text);
    }
  }
  return { answerTexts, thoughtTexts };
}

function responseText(response: Any): string | undefined {
  // `.text` is a getter in the JS SDK; guard against throwing accessors.
  try {
    const t = response?.text;
    return typeof t === "string" ? t : undefined;
  } catch {
    return undefined;
  }
}

function reduceGeminiStream(span: SpanContext, chunks: Any[]): void {
  const answerParts: string[] = [];
  const thoughtParts: string[] = [];

  for (const chunk of chunks) {
    const parts = chunk?.candidates?.[0]?.content?.parts;
    if (Array.isArray(parts)) {
      for (const part of parts) {
        const text = part?.text;
        if (!text) {
          continue;
        }
        if (part.thought) {
          thoughtParts.push(text);
        } else {
          answerParts.push(text);
        }
      }
      continue;
    }
    const text = responseText(chunk);
    if (text) {
      answerParts.push(text);
    }
  }

  const last = chunks.length > 0 ? chunks[chunks.length - 1] : undefined;
  const usage = extractGeminiUsage(last?.usageMetadata ?? last?.usage_metadata);

  try {
    if (answerParts.length > 0) {
      span.setOutput({ messages: [{ role: "assistant", content: answerParts.join("") }] });
    }
    if (thoughtParts.length > 0) {
      span.setMetadata({ reasoning_summary: thoughtParts.join("\n\n") });
    }
    setUsage(span, usage);
  } catch {
    // best-effort
  }
  closeLlmSpan(span);
}

function finishGeminiSpan(span: SpanContext | null, response: Any): void {
  if (span === null) {
    return;
  }
  try {
    const { answerTexts, thoughtTexts } = splitParts(response);
    if (answerTexts.length > 0) {
      span.setOutput({ messages: [{ role: "assistant", content: answerTexts.join("") }] });
    } else if (thoughtTexts.length === 0) {
      const text = responseText(response);
      if (text !== undefined) {
        span.setOutput({ messages: [{ role: "assistant", content: text }] });
      } else {
        const content = response?.candidates?.[0]?.content;
        if (content != null) {
          const serialized = safeSerialize(content);
          if (serialized && typeof serialized === "object" && !Array.isArray(serialized)) {
            (serialized as Record<string, unknown>).role = "assistant";
          }
          span.setOutput({ messages: [serialized] });
        }
      }
    }
    if (thoughtTexts.length > 0) {
      span.setMetadata({ reasoning_summary: thoughtTexts.join("\n\n") });
    }
  } catch {
    // ignore
  }

  try {
    setUsage(span, extractGeminiUsage(response?.usageMetadata ?? response?.usage_metadata));
  } catch {
    // ignore
  }

  closeLlmSpan(span);
}
