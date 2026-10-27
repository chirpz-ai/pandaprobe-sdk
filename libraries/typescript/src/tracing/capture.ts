import type { AsyncLocalStorage } from "node:async_hooks";
import { logger } from "../logger.js";

export interface TraceCaptureOptions {
  metadata?: Record<string, unknown>;
  sessionId?: string;
  maxTraces?: number;
  onSubmit?: (payload: Record<string, unknown>) => unknown;
}
/** Observes submissions, without creating traces or guaranteeing persistence. */
export class TraceCapture {
  readonly traceIds: string[] = [];
  overflow = false;
  closed = false;
  private entered = false;
  private readonly metadata: Record<string, unknown>;
  private readonly maximum: number;
  constructor(
    private readonly storage: AsyncLocalStorage<TraceCapture>,
    private readonly options: TraceCaptureOptions = {},
  ) {
    this.maximum = options.maxTraces ?? 1000;
    if (!Number.isSafeInteger(this.maximum) || this.maximum < 1)
      throw new TypeError("maxTraces must be a positive integer");
    this.metadata = structuredClone(options.metadata ?? {});
  }
  async run<T>(callback: (capture: TraceCapture) => T | Promise<T>): Promise<T> {
    if (this.entered) throw new Error("Trace capture scopes cannot be reused");
    this.entered = true;
    return this.storage.run(this, async () => {
      try {
        return await callback(this);
      } finally {
        this.closed = true;
      }
    });
  }
  /** @internal Decorate a copied export; late descendants retain phase attribution. */
  apply(payload: Record<string, unknown>): Record<string, unknown> {
    const result: Record<string, unknown> = {
      ...payload,
      metadata: { ...((payload.metadata as Record<string, unknown>) ?? {}), ...structuredClone(this.metadata) },
    };
    if (this.options.sessionId !== undefined) result.session_id = this.options.sessionId;
    if (this.closed) return result;
    const id = String(payload.trace_id);
    if (!this.traceIds.includes(id)) {
      if (this.traceIds.length < this.maximum) this.traceIds.push(id);
      else this.overflow = true;
    }
    if (this.options.onSubmit) {
      try {
        const value = this.options.onSubmit(structuredClone(result));
        if (value && typeof (value as Promise<unknown>).then === "function")
          void Promise.resolve(value).catch(() => logger.debug("Trace submission observer failed"));
      } catch {
        logger.debug("Trace submission observer failed");
      }
    }
    return result;
  }
}
