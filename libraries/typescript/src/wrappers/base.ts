/**
 * Shared, provider-agnostic utilities for all LLM client wrappers.
 *
 * Provider-specific logic belongs in its own sub-package under
 * `wrappers/<provider>/`. Mirrors the Python `wrappers/_base.py`.
 */

import { getClient } from "../client.js";
import { SpanKind } from "../schemas.js";
import { getCurrentTrace } from "../tracing/context.js";
import type { TraceContext } from "../tracing/context.js";
import { getCurrentSessionId, getCurrentUserId } from "../tracing/session.js";
import type { SpanContext } from "../tracing/span.js";
import { safeSerialize } from "../util.js";
import { extractLastUserMessage } from "../validation.js";

// Re-export shared serialization so provider sub-packages keep importing it from `../base.js`.
export { safeSerialize };

// ---------------------------------------------------------------------------
// Safe parameter whitelists
// ---------------------------------------------------------------------------

export const SAFE_INVOCATION_PARAMS: Set<string> = new Set([
  "temperature",
  "top_p",
  "max_tokens",
  "max_completion_tokens",
  "frequency_penalty",
  "presence_penalty",
  "seed",
  "n",
  "response_format",
  "stop",
  "logprobs",
  "top_logprobs",
  "reasoning_effort",
  "stream_options",
  "service_tier",
]);

/** Pull only safe invocation parameters from the call kwargs. */
export function extractModelParams(
  kwargs: Record<string, unknown>,
  allowed: Set<string> = SAFE_INVOCATION_PARAMS,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(kwargs)) {
    if (allowed.has(k) && v !== undefined) {
      out[k] = v;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Span lifecycle helpers
// ---------------------------------------------------------------------------

const standaloneTraces = new WeakMap<SpanContext, TraceContext>();

export interface OpenLlmSpanOptions {
  methodName: string;
  /** Already-normalized input in the standard messages schema. */
  inputData: unknown;
  modelParams: Record<string, unknown>;
  model: string | null;
}

/**
 * Open an LLM span, creating a standalone trace if none is active.
 *
 * Shared entry-point used by every wrapper provider. Returns a SpanContext, or
 * null if the SDK is disabled and no trace is active. Standalone traces are
 * tracked so {@link closeLlmSpan} / {@link errorLlmSpan} can finalize them.
 */
export function openLlmSpan(options: OpenLlmSpanOptions): SpanContext | null {
  const { methodName, inputData, modelParams, model } = options;
  const traceCtx = getCurrentTrace();

  if (traceCtx !== null) {
    const span = traceCtx.span(methodName, { kind: SpanKind.LLM, model });
    span.start();
    span.setInput(inputData);
    if (Object.keys(modelParams).length > 0) {
      span.setModelParameters(modelParams);
    }
    return span;
  }

  const client = getClient();
  if (client === null || !client.enabled) {
    return null;
  }

  const standalone = client.trace(methodName, {
    input: extractLastUserMessage(inputData),
    sessionId: getCurrentSessionId(),
    userId: getCurrentUserId(),
  });
  // start(false): do NOT enter the global AsyncLocalStorage store — the span
  // nests via the trace's own spanStack. Entering would leak this standalone
  // trace into the caller's async frame and break subsequent wrapper calls.
  standalone.start(false);

  const span = standalone.span(methodName, { kind: SpanKind.LLM, model });
  span.start();
  span.setInput(inputData);
  if (Object.keys(modelParams).length > 0) {
    span.setModelParameters(modelParams);
  }
  standaloneTraces.set(span, standalone);
  return span;
}

/**
 * Open an LLM span from raw call kwargs (input is `{messages}` from kwargs).
 * The default entry-point for chat-style providers.
 */
export function enterLlmSpan(
  cleanedKwargs: Record<string, unknown>,
  methodName: string,
  allowedParams?: Set<string>,
): SpanContext | null {
  return openLlmSpan({
    methodName,
    inputData: safeSerialize({ messages: cleanedKwargs.messages ?? [] }),
    modelParams: extractModelParams(cleanedKwargs, allowedParams),
    model: (cleanedKwargs.model as string | undefined) ?? null,
  });
}

/** Exit the span (and standalone trace if applicable). */
export function closeLlmSpan(span: SpanContext | null): void {
  if (span === null) {
    return;
  }
  span.end();
  const standalone = standaloneTraces.get(span);
  if (standalone !== undefined) {
    standalone.setOutput(span.getOutput());
    standalone.end();
    standaloneTraces.delete(span);
  }
}

/** Record an error and close the span (and standalone trace if applicable). */
export function errorLlmSpan(span: SpanContext | null, exc: unknown): void {
  if (span === null) {
    return;
  }
  span.setError(String(exc instanceof Error ? exc.message : exc));
  span.end(exc);
  const standalone = standaloneTraces.get(span);
  if (standalone !== undefined) {
    standalone.setOutput(span.getOutput());
    standalone.end(exc);
    standaloneTraces.delete(span);
  }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

export type ReduceChunks<C> = (span: SpanContext, chunks: C[]) => void;

/**
 * Wrap an async-iterable streaming response, collecting chunks for the span.
 *
 * Failure-mode contract (mirrors Python's SyncStreamReducer/AsyncStreamReducer):
 * - normal completion → `reduceChunks` runs, span closed successfully;
 * - the stream throws → `errorLlmSpan` records the error before re-raising;
 * - early break/return → the span is still finalized with chunks seen so far.
 * The completion start time is set on the first chunk.
 */
export async function* wrapAsyncStream<C>(
  stream: AsyncIterable<C>,
  span: SpanContext | null,
  reduceChunks: ReduceChunks<C>,
): AsyncGenerator<C> {
  const chunks: C[] = [];
  let first = true;
  let finalized = false;

  const finalize = (): void => {
    if (finalized || span === null) {
      return;
    }
    finalized = true;
    try {
      reduceChunks(span, chunks);
    } catch {
      closeLlmSpan(span);
    }
  };

  try {
    for await (const chunk of stream) {
      if (first && span !== null) {
        span.setCompletionStartTime(new Date());
        first = false;
      }
      chunks.push(chunk);
      yield chunk;
    }
    finalize();
  } catch (exc) {
    if (span !== null && !finalized) {
      finalized = true;
      errorLlmSpan(span, exc);
    }
    throw exc;
  } finally {
    // Early break/return without exhausting the stream.
    finalize();
  }
}

// ---------------------------------------------------------------------------
// Shared usage-mapping helpers
// ---------------------------------------------------------------------------

/** Coerce a value to a finite integer, or undefined. */
export function asInt(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.trunc(value);
  }
  return undefined;
}

/** Build a token_usage record, dropping undefined entries. */
export function tokenUsage(entries: Record<string, number | undefined>): Record<string, number> | null {
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(entries)) {
    if (v !== undefined) {
      out[k] = v;
    }
  }
  return Object.keys(out).length > 0 ? out : null;
}

/** Apply a `{prompt_tokens, completion_tokens, ...}` record to a span. */
export function setSpanUsage(span: SpanContext, usage: Record<string, number> | null): void {
  if (!usage) {
    return;
  }
  const { prompt_tokens, completion_tokens, ...extra } = usage;
  span.setTokenUsage({ promptTokens: prompt_tokens ?? 0, completionTokens: completion_tokens ?? 0, ...extra });
}
