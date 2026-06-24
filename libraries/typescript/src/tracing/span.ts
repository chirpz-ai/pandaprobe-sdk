/**
 * SpanContext — manages the lifecycle of a single span within a trace.
 *
 * For LLM spans (kind="LLM"), input/output must follow the messages schema.
 * Other span kinds accept arbitrary input/output. Parent-child relationships
 * are managed via the owning trace's own span stack (`traceCtx.spanStack`), so
 * spans always nest within their own trace regardless of the global context.
 */

import { logger } from "../logger.js";
import { SpanData, SpanKind, SpanStatusCode, newUuid } from "../schemas.js";
import { isThenable } from "../util.js";
import { warnIfInvalidMessages } from "../validation.js";
import type { TraceContext } from "./context.js";

export interface SpanContextOptions {
  kind?: SpanKind;
  model?: string | null;
  metadata?: Record<string, unknown>;
}

export class SpanContext {
  private readonly traceCtx: TraceContext;
  private readonly _spanId: string;
  private readonly name: string;
  private readonly kind: SpanKind;
  private model: string | null;
  private readonly metadata: Record<string, unknown>;
  private input: unknown = null;
  private output: unknown = null;
  private status: SpanStatusCode = SpanStatusCode.UNSET;
  private error: string | null = null;
  private startedAt: Date | null = null;
  private endedAt: Date | null = null;
  private tokenUsage: Record<string, number> | null = null;
  private modelParameters: Record<string, unknown> | null = null;
  private cost: Record<string, number> | null = null;
  private completionStartTime: Date | null = null;
  private parentSpanId: string | null = null;

  constructor(traceCtx: TraceContext, name: string, options: SpanContextOptions = {}) {
    this.traceCtx = traceCtx;
    this._spanId = newUuid();
    this.name = name;
    this.kind = options.kind ?? SpanKind.OTHER;
    this.model = options.model ?? null;
    this.metadata = options.metadata ?? {};
  }

  get spanId(): string {
    return this._spanId;
  }

  /** The output recorded on this span (used to mirror it onto standalone traces). */
  getOutput(): unknown {
    return this.output;
  }

  /** The trace this span belongs to (used to attach sibling/child tool spans). */
  getTraceContext(): TraceContext {
    return this.traceCtx;
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /** Enter the span: record start time, set parent, push onto the trace's stack. */
  start(): this {
    this.startedAt = new Date();
    const stack = this.traceCtx.spanStack;
    if (stack.length > 0) {
      this.parentSpanId = stack[stack.length - 1] ?? null;
    }
    stack.push(this._spanId);
    return this;
  }

  /** End the span: record end time/status, finalize, pop from the trace's stack. */
  end(error?: unknown): void {
    this.endedAt = new Date();
    if (error !== undefined && error !== null) {
      this.status = SpanStatusCode.ERROR;
      this.error = String(error instanceof Error ? error.message : error);
    } else if (this.status === SpanStatusCode.UNSET) {
      this.status = SpanStatusCode.OK;
    }
    this.finalize();
    const stack = this.traceCtx.spanStack;
    if (stack.length > 0 && stack[stack.length - 1] === this._spanId) {
      stack.pop();
    }
  }

  /** Run *fn* within this span (callback form). */
  async run<T>(fn: (span: SpanContext) => T | Promise<T>): Promise<T> {
    this.start();
    let errored: unknown;
    try {
      return await fn(this);
    } catch (exc) {
      errored = exc;
      throw exc;
    } finally {
      this.end(errored);
    }
  }

  /**
   * Wrap a sync-or-async invocation in this span (decorator form). Sets the
   * given *input*, captures the return value as output, and preserves the
   * original return type (sync stays sync, async stays async).
   */
  instrument<T>(invoke: () => T, input: unknown): T {
    this.start();
    this.setInput(input);
    let result: T;
    try {
      result = invoke();
    } catch (exc) {
      this.end(exc);
      throw exc;
    }
    if (isThenable(result)) {
      const wrapped = (result as unknown as Promise<unknown>).then(
        (value) => {
          this.setOutput(value);
          this.end();
          return value;
        },
        (exc) => {
          this.end(exc);
          throw exc;
        },
      );
      return wrapped as T;
    }
    this.setOutput(result);
    this.end();
    return result;
  }

  // ------------------------------------------------------------------
  // Setters
  // ------------------------------------------------------------------

  setInput(input: unknown): void {
    if (this.kind === SpanKind.LLM) {
      warnIfInvalidMessages(input, "LLM span input");
    }
    this.input = input;
  }

  setOutput(output: unknown): void {
    if (this.kind === SpanKind.LLM) {
      warnIfInvalidMessages(output, "LLM span output");
    }
    this.output = output;
  }

  setTokenUsage(usage: { promptTokens?: number; completionTokens?: number; [key: string]: number | undefined }): void {
    const { promptTokens = 0, completionTokens = 0, ...extra } = usage;
    const out: Record<string, number> = { prompt_tokens: promptTokens, completion_tokens: completionTokens };
    for (const [k, v] of Object.entries(extra)) {
      if (typeof v === "number") {
        out[k] = v;
      }
    }
    this.tokenUsage = out;
  }

  setModel(model: string): void {
    this.model = model;
  }

  setModelParameters(params: Record<string, unknown>): void {
    this.modelParameters = params;
  }

  setCost(cost: { total: number; [key: string]: number }): void {
    this.cost = { ...cost };
  }

  setCompletionStartTime(ts: Date): void {
    this.completionStartTime = ts;
  }

  setError(error: string): void {
    this.error = error;
    this.status = SpanStatusCode.ERROR;
  }

  setMetadata(metadata: Record<string, unknown>): void {
    Object.assign(this.metadata, metadata);
  }

  // ------------------------------------------------------------------
  // Finalization
  // ------------------------------------------------------------------

  private finalize(): void {
    try {
      const span = new SpanData({
        spanId: this._spanId,
        parentSpanId: this.parentSpanId,
        name: this.name,
        kind: this.kind,
        status: this.status,
        input: this.input,
        output: this.output,
        model: this.model,
        tokenUsage: this.tokenUsage,
        metadata: this.metadata,
        startedAt: this.startedAt ?? new Date(),
        endedAt: this.endedAt,
        error: this.error,
        completionStartTime: this.completionStartTime,
        modelParameters: this.modelParameters,
        cost: this.cost,
      });
      this.traceCtx.addSpanData(span);
    } catch (exc) {
      logger.error(`failed to finalize span: ${String(exc)}`);
    }
  }
}
