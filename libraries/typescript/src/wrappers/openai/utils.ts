/** OpenAI-specific utilities for the wrapOpenAI instrumentation. */

import type { SpanContext } from "../../tracing/span.js";
import { openLlmSpan, safeSerialize } from "../base.js";

/** Drop undefined values (JS SDK uses `undefined`, not NotGiven sentinels). */
export function stripUndefined(kwargs: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(kwargs)) {
    if (v !== undefined) {
      out[k] = v;
    }
  }
  return out;
}

export const SAFE_RESPONSES_PARAMS = new Set([
  "max_output_tokens",
  "temperature",
  "top_p",
  "reasoning",
  "truncation",
  "service_tier",
]);

export function extractResponsesParams(kwargs: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(kwargs)) {
    if (SAFE_RESPONSES_PARAMS.has(k) && v !== undefined) {
      out[k] = safeSerialize(v);
    }
  }
  return out;
}

/** Convert Responses API `input` + `instructions` into the standard messages format. */
export function normalizeResponsesInput(kwargs: Record<string, unknown>): { messages: unknown[] } {
  const messages: unknown[] = [];
  const instructions = kwargs.instructions;
  if (typeof instructions === "string" && instructions) {
    messages.push({ role: "system", content: instructions });
  }
  const inp = kwargs.input ?? [];
  if (typeof inp === "string") {
    messages.push({ role: "user", content: inp });
  } else if (Array.isArray(inp)) {
    for (const item of inp) {
      if (item && typeof item === "object" && "role" in item) {
        messages.push(safeSerialize(item));
      } else if (typeof item === "string") {
        messages.push({ role: "user", content: item });
      } else {
        messages.push(safeSerialize(item));
      }
    }
  }
  return { messages };
}

/**
 * Open an LLM span for an OpenAI Responses API call, normalizing the
 * `input` / `instructions` params into the standard messages schema.
 */
export function enterResponsesSpan(
  kwargs: Record<string, unknown>,
  methodName = "openai-response",
): SpanContext | null {
  return openLlmSpan({
    methodName,
    inputData: normalizeResponsesInput(kwargs),
    modelParams: extractResponsesParams(kwargs),
    model: (kwargs.model as string | undefined) ?? null,
  });
}
