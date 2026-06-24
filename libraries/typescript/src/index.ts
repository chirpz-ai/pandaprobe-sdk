/**
 * PandaProbe — TypeScript SDK for open-source agent tracing and evaluation.
 */

import { getClient } from "./client.js";
import type { ScoreDataType } from "./schemas.js";
import { runWithSession, runWithUser, setCurrentSessionId, setCurrentUserId } from "./tracing/session.js";

export { VERSION } from "./version.js";
export { Client, init, getClient } from "./client.js";
export type { ConfigOptions, SdkConfig } from "./config.js";
export { SpanKind, SpanStatusCode, TraceStatus, ScoreDataType, SpanData, TraceData, ScoreData } from "./schemas.js";
export {
  trace,
  span,
  withTrace,
  withSpan,
  startTrace,
  type WithTraceOptions,
  type WithSpanOptions,
  type TraceDecoratorOptions,
  type SpanDecoratorOptions,
} from "./decorators.js";
export { TraceContext, type TraceContextOptions, getCurrentTrace, getSpanStack } from "./tracing/context.js";
export { SpanContext, type SpanContextOptions } from "./tracing/span.js";

// ---------------------------------------------------------------------------
// Module-level convenience functions
// ---------------------------------------------------------------------------

/** Block until all queued items are sent (delegates to the global client). */
export async function flush(timeout = 30.0): Promise<void> {
  const client = getClient();
  if (client !== null) {
    await client.flush(timeout);
  }
}

/** Flush remaining items and release resources (delegates to the global client). */
export async function shutdown(): Promise<void> {
  const client = getClient();
  if (client !== null) {
    await client.shutdown();
  }
}

/** Submit a programmatic score for a trace (delegates to the global client). */
export function score(
  traceId: string,
  name: string,
  value: string,
  options: { dataType?: ScoreDataType; reason?: string | null; metadata?: Record<string, unknown> } = {},
): void {
  const client = getClient();
  if (client !== null) {
    client.score(traceId, name, value, options);
  }
}

// ---------------------------------------------------------------------------
// Session propagation
// ---------------------------------------------------------------------------

/**
 * Set the session ID for the current context.
 *
 * All traces created after this call (via decorators, wrappers, or
 * integrations) inherit this session ID unless overridden explicitly.
 */
export function setSession(sessionId: string): void {
  setCurrentSessionId(sessionId);
}

/**
 * Run *fn* with a session ID set for its scope (the callback analog of Python's
 * `with pandaprobe.session(id):`).
 *
 * Uses `AsyncLocalStorage.run()`, so concurrent `session(...)` scopes on one
 * process stay fully isolated — no cross-conversation leakage.
 */
export async function session<T>(sessionId: string, fn: () => T | Promise<T>): Promise<T> {
  return runWithSession(sessionId, async () => fn());
}

// ---------------------------------------------------------------------------
// User ID propagation
// ---------------------------------------------------------------------------

/** Set the user ID for the current context. */
export function setUser(userId: string): void {
  setCurrentUserId(userId);
}

/** Run *fn* with a user ID set for its scope (concurrency-safe, run-based). */
export async function user<T>(userId: string, fn: () => T | Promise<T>): Promise<T> {
  return runWithUser(userId, async () => fn());
}
