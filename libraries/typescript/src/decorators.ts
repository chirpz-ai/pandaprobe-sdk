/**
 * Manual instrumentation API.
 *
 * Two forms (JS has no standalone function decorators like Python):
 *   1. Callback wrappers — `withTrace` / `withSpan` and `startTrace`, the
 *      idiomatic analog of Python's `with client.trace(...)` context managers.
 *   2. TS class-method decorators — `@trace` / `@span`, the closest visual match
 *      to Python's `@trace` / `@span` (require `experimentalDecorators`).
 */

import { getClient } from "./client.js";
import { SpanKind } from "./schemas.js";
import { type TraceClient, TraceContext } from "./tracing/context.js";
import { getCurrentTrace } from "./tracing/context.js";
import type { SpanContext } from "./tracing/span.js";
import { extractLastUserMessage } from "./validation.js";

/** No-op client used when tracing is disabled / no client is configured. */
const NOOP_TRACE_CLIENT: TraceClient = { logTrace() {} };

function traceClient(): TraceClient {
  return getClient() ?? NOOP_TRACE_CLIENT;
}

function detachedTrace(): TraceContext {
  return new TraceContext(NOOP_TRACE_CLIENT, "__detached__");
}

// ---------------------------------------------------------------------------
// Callback wrappers
// ---------------------------------------------------------------------------

export interface WithTraceOptions {
  input?: unknown;
  sessionId?: string | null;
  userId?: string | null;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface WithSpanOptions {
  kind?: SpanKind;
  model?: string | null;
  metadata?: Record<string, unknown>;
}

/**
 * Start a new trace and return its {@link TraceContext}, already entered.
 *
 * Imperative form — pair with `ctx.end()` (use try/finally). Prefer
 * {@link withTrace} for the safer callback form.
 */
export function startTrace(name: string, options: WithTraceOptions = {}): TraceContext {
  return new TraceContext(traceClient(), name, options).start();
}

/** Run *fn* inside a new trace (callback form). */
export function withTrace<T>(name: string, fn: (ctx: TraceContext) => T | Promise<T>): Promise<T>;
export function withTrace<T>(
  name: string,
  options: WithTraceOptions,
  fn: (ctx: TraceContext) => T | Promise<T>,
): Promise<T>;
export function withTrace<T>(
  name: string,
  optionsOrFn: WithTraceOptions | ((ctx: TraceContext) => T | Promise<T>),
  maybeFn?: (ctx: TraceContext) => T | Promise<T>,
): Promise<T> {
  const options = typeof optionsOrFn === "function" ? {} : optionsOrFn;
  const fn = (typeof optionsOrFn === "function" ? optionsOrFn : maybeFn) as (ctx: TraceContext) => T | Promise<T>;
  const ctx = new TraceContext(traceClient(), name, options);
  return ctx.run(fn);
}

/**
 * Run *fn* inside a span on the active trace (callback form).
 *
 * If there is no active trace the span is detached (never emitted), mirroring
 * Python's `@span` no-op behavior, but *fn* still runs and receives a span.
 */
export function withSpan<T>(name: string, fn: (span: SpanContext) => T | Promise<T>): Promise<T>;
export function withSpan<T>(
  name: string,
  options: WithSpanOptions,
  fn: (span: SpanContext) => T | Promise<T>,
): Promise<T>;
export function withSpan<T>(
  name: string,
  optionsOrFn: WithSpanOptions | ((span: SpanContext) => T | Promise<T>),
  maybeFn?: (span: SpanContext) => T | Promise<T>,
): Promise<T> {
  const options = typeof optionsOrFn === "function" ? {} : optionsOrFn;
  const fn = (typeof optionsOrFn === "function" ? optionsOrFn : maybeFn) as (span: SpanContext) => T | Promise<T>;
  const traceCtx = getCurrentTrace() ?? detachedTrace();
  const span = traceCtx.span(name, options);
  return span.run(fn);
}

// ---------------------------------------------------------------------------
// Method decorators
// ---------------------------------------------------------------------------

export interface TraceDecoratorOptions {
  name?: string;
  sessionId?: string | null;
  userId?: string | null;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface SpanDecoratorOptions {
  name?: string;
  kind?: SpanKind;
  model?: string | null;
  metadata?: Record<string, unknown>;
}

// Legacy decorator target is a prototype object (or constructor for statics).
type LegacyTarget = object;

function captureInput(args: unknown[]): unknown {
  // Single object arg: pass through (so `{messages: [...]}` is preserved for
  // message extraction). Otherwise wrap positionally, mirroring Python's
  // `_capture_input` fallback.
  if (args.length === 1 && typeof args[0] === "object" && args[0] !== null) {
    return args[0];
  }
  return { args };
}

function makeTraceDecorator(options: TraceDecoratorOptions) {
  return (_target: LegacyTarget, propertyKey: string | symbol, descriptor: PropertyDescriptor): PropertyDescriptor => {
    const original = descriptor.value as (...args: unknown[]) => unknown;
    const traceName = options.name ?? String(propertyKey);

    descriptor.value = function instrumented(this: unknown, ...args: unknown[]): unknown {
      const client = getClient();
      if (client === null || !client.enabled) {
        return original.apply(this, args);
      }
      const fnInput = captureInput(args);
      const ctx = client.trace(traceName, {
        input: extractLastUserMessage(fnInput),
        sessionId: options.sessionId,
        userId: options.userId,
        tags: options.tags,
        metadata: options.metadata,
      });
      return ctx.instrument(() => original.apply(this, args), true);
    };
    return descriptor;
  };
}

function makeSpanDecorator(options: SpanDecoratorOptions) {
  return (_target: LegacyTarget, propertyKey: string | symbol, descriptor: PropertyDescriptor): PropertyDescriptor => {
    const original = descriptor.value as (...args: unknown[]) => unknown;
    const spanName = options.name ?? String(propertyKey);

    descriptor.value = function instrumented(this: unknown, ...args: unknown[]): unknown {
      const traceCtx = getCurrentTrace();
      if (traceCtx === null) {
        return original.apply(this, args);
      }
      const span = traceCtx.span(spanName, {
        kind: options.kind ?? SpanKind.OTHER,
        model: options.model,
        metadata: options.metadata,
      });
      return span.instrument(() => original.apply(this, args), captureInput(args));
    };
    return descriptor;
  };
}

function isBareUsage(args: unknown[]): args is [LegacyTarget, string | symbol, PropertyDescriptor] {
  return args.length === 3 && (typeof args[1] === "string" || typeof args[1] === "symbol");
}

/**
 * Method decorator that wraps a class method in a {@link TraceContext}.
 *
 * Usable bare (`@trace`) or with options (`@trace({ name: "x" })`). Works with
 * sync and async methods.
 */
export function trace(options?: TraceDecoratorOptions): MethodDecorator;
export function trace(target: LegacyTarget, propertyKey: string | symbol, descriptor: PropertyDescriptor): void;
export function trace(...args: unknown[]): MethodDecorator | void {
  if (isBareUsage(args)) {
    makeTraceDecorator({})(args[0], args[1], args[2]);
    return;
  }
  return makeTraceDecorator((args[0] as TraceDecoratorOptions) ?? {}) as MethodDecorator;
}

/**
 * Method decorator that wraps a class method in a {@link SpanContext}.
 *
 * No-ops (runs the method directly) when there is no active trace, mirroring
 * Python's `@span`. Usable bare (`@span`) or with options.
 */
export function span(options?: SpanDecoratorOptions): MethodDecorator;
export function span(target: LegacyTarget, propertyKey: string | symbol, descriptor: PropertyDescriptor): void;
export function span(...args: unknown[]): MethodDecorator | void {
  if (isBareUsage(args)) {
    makeSpanDecorator({})(args[0], args[1], args[2]);
    return;
  }
  return makeSpanDecorator((args[0] as SpanDecoratorOptions) ?? {}) as MethodDecorator;
}
