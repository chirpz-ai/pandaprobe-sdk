/** Mistral-specific utilities for the wrapMistral instrumentation. */

import type { SpanContext } from "../../tracing/span.js";
import { openLlmSpan, safeSerialize } from "../base.js";

export const SAFE_MISTRAL_PARAMS = new Set([
  "temperature",
  "top_p",
  "max_tokens",
  "random_seed",
  "safe_prompt",
  "response_format",
  "tool_choice",
  "presence_penalty",
  "frequency_penalty",
  "n",
  "stop",
]);

export function extractMistralParams(kwargs: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(kwargs)) {
    if (SAFE_MISTRAL_PARAMS.has(k) && v !== undefined) {
      out[k] = safeSerialize(v);
    }
  }
  return out;
}

/** Mistral already uses `[{role, content}]`; just serialize for safety. */
export function normalizeMistralInput(kwargs: Record<string, unknown>): { messages: unknown[] } {
  const serialized = safeSerialize(kwargs.messages ?? []);
  return { messages: Array.isArray(serialized) ? serialized : [] };
}

/** Open an LLM span for a Mistral API call. */
export function enterMistralSpan(kwargs: Record<string, unknown>, methodName = "mistral-chat"): SpanContext | null {
  return openLlmSpan({
    methodName,
    inputData: normalizeMistralInput(kwargs),
    modelParams: extractMistralParams(kwargs),
    model: (kwargs.model as string | undefined) ?? null,
  });
}
