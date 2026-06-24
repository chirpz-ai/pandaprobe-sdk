/**
 * TraceContext — manages the lifecycle of a single trace.
 *
 * Input must be `{messages: [{role: "user", content: "..."}]}` containing only
 * the current turn's user message. Output must be
 * `{messages: [{role: "assistant", content: "..."}]}`.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import { logger } from "../logger.js";
import { type SpanData, SpanKind, TraceStatus as TS, TraceData, type TraceStatus, newUuid } from "../schemas.js";
import { isThenable } from "../util.js";
import { extractLastAssistantMessage, warnIfInvalidMessages } from "../validation.js";
import { getCurrentSessionId, getCurrentUserId } from "./session.js";
import { SpanContext, type SpanContextOptions } from "./span.js";

/** What a TraceContext needs from the client (avoids a circular import). */
export interface TraceClient {
  logTrace(trace: TraceData): void;
}

interface TraceStore {
  trace: TraceContext | null;
}

const traceStorage = new AsyncLocalStorage<TraceStore>();

export function getCurrentTrace(): TraceContext | null {
  return traceStorage.getStore()?.trace ?? null;
}

/** The span stack of the active trace (each trace owns its own stack). */
export function getSpanStack(): string[] {
  return getCurrentTrace()?.spanStack ?? [];
}

export interface TraceContextOptions {
  input?: unknown;
  sessionId?: string | null;
  userId?: string | null;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export class TraceContext {
  private readonly client: TraceClient;
  private readonly _traceId: string;
  private readonly name: string;
  private input: unknown;
  private output: unknown = null;
  private readonly sessionId: string | null;
  private readonly userId: string | null;
  private readonly tags: string[];
  private readonly metadata: Record<string, unknown>;
  private status: TraceStatus = TS.COMPLETED;
  private error: string | null = null;
  private startedAt: Date | null = null;
  private endedAt: Date | null = null;
  private readonly spans: SpanData[] = [];
  readonly spanStack: string[] = [];
  private prevStore: TraceStore | undefined;
  private entered = false;
  private isEnded = false;

  constructor(client: TraceClient, name: string, options: TraceContextOptions = {}) {
    this.client = client;
    this._traceId = newUuid();
    this.name = name;
    this.input = options.input ?? null;
    warnIfInvalidMessages(this.input, "trace input");
    this.sessionId = options.sessionId ?? getCurrentSessionId();
    this.userId = options.userId ?? getCurrentUserId();
    this.tags = options.tags ?? [];
    this.metadata = options.metadata ?? {};
  }

  get traceId(): string {
    return this._traceId;
  }

  // ------------------------------------------------------------------
  // Lifecycle — imperative (start/end) and callback (run)
  // ------------------------------------------------------------------

  /**
   * Enter the trace context imperatively. Pair with `end()`.
   *
   * When *enterStore* is true (the default, for the imperative `startTrace`
   * API) the trace is pushed onto the AsyncLocalStorage store so ambient
   * lookups (`getCurrentTrace`) find it. Wrappers that create a standalone
   * trace pass `false` — their spans nest via this trace's own `spanStack`, so
   * they must NOT mutate the global store (`enterWith` would leak the trace
   * into the caller's async frame and contaminate later, unrelated calls).
   */
  start(enterStore = true): this {
    this.startedAt = new Date();
    if (enterStore) {
      this.prevStore = traceStorage.getStore();
      this.entered = true;
      traceStorage.enterWith({ trace: this });
    }
    return this;
  }

  /**
   * End the trace, finalize, and restore the previous ambient context.
   *
   * Imperative `start()`/`end()` is intended for LIFO (properly nested) usage,
   * the analog of Python's `with`. For overlapping or concurrent traces prefer
   * the scoped {@link run} / `withTrace` form. To stay safe under out-of-order
   * end() the restore is guarded: it only rewrites the ambient store when THIS
   * trace is still the active one (so ending an inner trace out of order can't
   * wipe a newer active trace), and it never restores to an already-ended
   * parent (which would resurrect a finished trace as "current").
   */
  end(error?: unknown): void {
    this.endedAt = new Date();
    if (error !== undefined && error !== null) {
      this.status = TS.ERROR;
      this.error = String(error instanceof Error ? error.message : error);
    }
    this.isEnded = true;
    this.finalize();
    if (this.entered && getCurrentTrace() === this) {
      const parent = this.prevStore?.trace ?? null;
      traceStorage.enterWith({ trace: parent?.isEnded ? null : parent });
    }
  }

  /** Run *fn* within this trace's context (callback form). */
  async run<T>(fn: (ctx: TraceContext) => T | Promise<T>): Promise<T> {
    return traceStorage.run({ trace: this }, async () => {
      this.startedAt = new Date();
      try {
        return await fn(this);
      } catch (exc) {
        this.status = TS.ERROR;
        this.error = String(exc instanceof Error ? exc.message : exc);
        throw exc;
      } finally {
        this.endedAt = new Date();
        this.finalize();
      }
    });
  }

  /**
   * Wrap a sync-or-async invocation in this trace's context (decorator form).
   *
   * Uses `AsyncLocalStorage.run` so async continuations inherit the trace store
   * without leaking it to the caller. Preserves the original return type:
   * a sync invocation returns synchronously, an async one returns a Promise.
   */
  instrument<T>(invoke: () => T, captureOutput: boolean): T {
    return traceStorage.run({ trace: this }, (): T => {
      this.startedAt = new Date();
      let result: T;
      try {
        result = invoke();
      } catch (exc) {
        this.failAndFinalize(exc);
        throw exc;
      }
      if (isThenable(result)) {
        const wrapped = (result as unknown as Promise<unknown>).then(
          (value) => {
            if (captureOutput) {
              this.setOutput(extractLastAssistantMessage(value));
            }
            this.endedAt = new Date();
            this.finalize();
            return value;
          },
          (exc) => {
            this.failAndFinalize(exc);
            throw exc;
          },
        );
        return wrapped as T;
      }
      if (captureOutput) {
        this.setOutput(extractLastAssistantMessage(result));
      }
      this.endedAt = new Date();
      this.finalize();
      return result;
    });
  }

  private failAndFinalize(exc: unknown): void {
    this.status = TS.ERROR;
    this.error = String(exc instanceof Error ? exc.message : exc);
    this.endedAt = new Date();
    this.finalize();
  }

  // ------------------------------------------------------------------
  // Span factory
  // ------------------------------------------------------------------

  span(name: string, options: SpanContextOptions = {}): SpanContext {
    return new SpanContext(this, name, {
      kind: options.kind ?? SpanKind.OTHER,
      model: options.model,
      metadata: options.metadata,
    });
  }

  // ------------------------------------------------------------------
  // Setters
  // ------------------------------------------------------------------

  setOutput(output: unknown): void {
    warnIfInvalidMessages(output, "trace output");
    this.output = output;
  }

  setInput(input: unknown): void {
    warnIfInvalidMessages(input, "trace input");
    this.input = input;
  }

  setMetadata(metadata: Record<string, unknown>): void {
    Object.assign(this.metadata, metadata);
  }

  setStatus(status: TraceStatus): void {
    this.status = status;
  }

  addSpanData(spanData: SpanData): void {
    this.spans.push(spanData);
  }

  // ------------------------------------------------------------------
  // Finalization
  // ------------------------------------------------------------------

  private finalize(): void {
    try {
      const trace = new TraceData({
        traceId: this._traceId,
        name: this.name,
        status: this.status,
        input: this.input,
        output: this.output,
        metadata: this.metadata,
        startedAt: this.startedAt ?? new Date(),
        endedAt: this.endedAt,
        sessionId: this.sessionId,
        userId: this.userId,
        tags: this.tags,
        spans: this.spans,
      });
      this.client.logTrace(trace);
    } catch (exc) {
      logger.error(`failed to finalize trace: ${String(exc)}`);
    }
  }
}
