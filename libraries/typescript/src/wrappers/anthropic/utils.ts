/** Anthropic-specific utilities for the wrapAnthropic instrumentation. */

import type { SpanContext } from "../../tracing/span.js";
import { openLlmSpan, safeSerialize } from "../base.js";

export const SAFE_ANTHROPIC_PARAMS = new Set([
  "temperature",
  "top_p",
  "top_k",
  "max_tokens",
  "stop_sequences",
  "thinking",
]);

export function extractAnthropicParams(kwargs: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(kwargs)) {
    if (SAFE_ANTHROPIC_PARAMS.has(k) && v !== undefined) {
      out[k] = v;
    }
  }
  return out;
}

/**
 * Convert Anthropic `system` + `messages` into the standard messages format.
 * Anthropic passes `system` as a separate top-level kwarg, so it is prepended
 * as a system message.
 */
export function normalizeAnthropicInput(kwargs: Record<string, unknown>): { messages: unknown[] } {
  const messages: unknown[] = [];
  const system = kwargs.system;
  if (typeof system === "string") {
    messages.push({ role: "system", content: system });
  } else if (Array.isArray(system)) {
    messages.push({ role: "system", content: safeSerialize(system) });
  }
  const serialized = safeSerialize(kwargs.messages ?? []);
  if (Array.isArray(serialized)) {
    messages.push(...serialized);
  }
  return { messages };
}

/** Open an LLM span for an Anthropic API call, normalizing system + messages. */
export function enterAnthropicSpan(
  kwargs: Record<string, unknown>,
  methodName = "anthropic-messages",
): SpanContext | null {
  return openLlmSpan({
    methodName,
    inputData: normalizeAnthropicInput(kwargs),
    modelParams: extractAnthropicParams(kwargs),
    model: (kwargs.model as string | undefined) ?? null,
  });
}
