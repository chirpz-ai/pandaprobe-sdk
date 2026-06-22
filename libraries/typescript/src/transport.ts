/**
 * Background transport layer for the PandaProbe SDK.
 *
 * The Python SDK uses a background thread + thread-safe queue. Node is
 * single-threaded with async I/O, so this uses an in-memory buffer drained by
 * a `setInterval` timer (periodic flush) and on reaching `batchSize`. Sends use
 * the native `fetch`. Process-exit hooks flush remaining items (atexit analog).
 */

import type { SdkConfig } from "./config.js";
import { logger } from "./logger.js";
import { VERSION } from "./version.js";

const NO_RETRY_STATUSES = new Set([401, 403, 422]);
const RETRIABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

const MAX_RETRIES = 3;
const INITIAL_BACKOFF = 0.5; // seconds

export type QueueItemKind = "trace" | "spans" | "update_trace" | "update_span" | "score";

interface QueueItem {
  kind: QueueItemKind;
  payload: Record<string, unknown> | unknown[];
  traceId?: string;
  spanId?: string;
}

function sleep(seconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, Math.round(seconds * 1000)));
}

function parseRetryAfter(resp: Response): number | null {
  const val = resp.headers.get("Retry-After");
  if (val === null) {
    return null;
  }
  const parsed = Number.parseFloat(val);
  return Number.isNaN(parsed) ? null : parsed;
}

export class Transport {
  private readonly config: SdkConfig;
  private readonly onError?: (exc: unknown) => void;
  private readonly baseHeaders: Record<string, string>;
  private buffer: QueueItem[] = [];
  private timer: ReturnType<typeof setInterval> | null = null;
  private draining: Promise<void> | null = null;
  private shuttingDown = false;
  private readonly exitHandler: () => void;
  private readonly signalHandler: () => void;

  constructor(config: SdkConfig, onError?: (exc: unknown) => void) {
    this.config = config;
    this.onError = onError;
    this.baseHeaders = {
      "X-API-Key": config.apiKey ?? "",
      "X-Project-Name": config.projectName ?? "",
      "Content-Type": "application/json",
      "User-Agent": `pandaprobe-typescript/${VERSION}`,
    };

    this.exitHandler = () => {
      void this.flush(10.0);
    };
    this.signalHandler = () => {
      void this.shutdown();
    };

    if (config.enabled) {
      this.startTimer();
      process.once("beforeExit", this.exitHandler);
      process.once("SIGTERM", this.signalHandler);
      process.once("SIGINT", this.signalHandler);
    }
  }

  // ------------------------------------------------------------------
  // Public enqueue helpers
  // ------------------------------------------------------------------

  enqueueTrace(payload: Record<string, unknown>): void {
    this.put({ kind: "trace", payload });
  }

  enqueueSpans(traceId: string, payload: Record<string, unknown>[]): void {
    this.put({ kind: "spans", payload, traceId });
  }

  enqueueUpdateTrace(traceId: string, payload: Record<string, unknown>): void {
    this.put({ kind: "update_trace", payload, traceId });
  }

  enqueueUpdateSpan(traceId: string, spanId: string, payload: Record<string, unknown>): void {
    this.put({ kind: "update_span", payload, traceId, spanId });
  }

  enqueueScore(payload: Record<string, unknown>): void {
    this.put({ kind: "score", payload });
  }

  // ------------------------------------------------------------------
  // Flush / shutdown
  // ------------------------------------------------------------------

  /** Resolve once all buffered items have been sent (up to *timeout* seconds). */
  async flush(timeout = 30.0): Promise<void> {
    if (!this.config.enabled) {
      return;
    }
    const drain = this.drain();
    if (timeout <= 0) {
      await drain;
      return;
    }
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeoutPromise = new Promise<void>((resolve) => {
      timer = setTimeout(resolve, Math.round(timeout * 1000));
    });
    try {
      await Promise.race([drain, timeoutPromise]);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  /** Flush remaining items and stop the background timer. */
  async shutdown(): Promise<void> {
    if (this.shuttingDown) {
      return;
    }
    this.shuttingDown = true;
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    process.removeListener("beforeExit", this.exitHandler);
    process.removeListener("SIGTERM", this.signalHandler);
    process.removeListener("SIGINT", this.signalHandler);
    await this.flush(10.0);
  }

  // ------------------------------------------------------------------
  // Internals
  // ------------------------------------------------------------------

  private put(item: QueueItem): void {
    if (!this.config.enabled || this.shuttingDown) {
      return;
    }
    if (this.buffer.length >= this.config.maxQueueSize) {
      logger.warning("queue full — dropping oldest item");
      this.buffer.shift();
    }
    this.buffer.push(item);
    if (this.buffer.length >= this.config.batchSize) {
      void this.drain();
    }
  }

  private startTimer(): void {
    this.timer = setInterval(
      () => {
        void this.drain();
      },
      Math.round(this.config.flushInterval * 1000),
    );
    // Don't keep the event loop alive solely for the flush timer.
    if (typeof this.timer.unref === "function") {
      this.timer.unref();
    }
  }

  /** Drain the buffer, sending each item. Concurrent calls share one drain. */
  private drain(): Promise<void> {
    if (this.draining !== null) {
      return this.draining;
    }
    this.draining = this.doDrain().finally(() => {
      this.draining = null;
    });
    return this.draining;
  }

  private async doDrain(): Promise<void> {
    while (this.buffer.length > 0) {
      const batch = this.buffer.splice(0, this.buffer.length);
      for (const item of batch) {
        try {
          await this.send(item);
        } catch (exc) {
          logger.error(`transport error: ${String(exc)}`);
          if (this.onError) {
            try {
              this.onError(exc);
            } catch {
              // swallow callback errors
            }
          }
        }
      }
    }
  }

  private async send(item: QueueItem): Promise<void> {
    const { url, method, body } = this.buildRequest(item);
    const headers = { ...this.baseHeaders, "X-Request-ID": crypto.randomUUID() };

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        const resp = await fetch(url, { method, headers, body: JSON.stringify(body) });

        if (resp.status < 300) {
          logger.debug(`${method} ${url} → ${resp.status}`);
          return;
        }

        if (NO_RETRY_STATUSES.has(resp.status)) {
          const text = (await resp.text()).slice(0, 500);
          logger.error(`${method} ${url} → ${resp.status} (not retrying): ${text}`);
          return;
        }

        if (RETRIABLE_STATUSES.has(resp.status)) {
          const retryAfter = parseRetryAfter(resp);
          const backoff = retryAfter ?? INITIAL_BACKOFF * 2 ** attempt;
          logger.warning(
            `${method} ${url} → ${resp.status}, retrying in ${backoff.toFixed(1)}s (attempt ${
              attempt + 1
            }/${MAX_RETRIES})`,
          );
          await sleep(backoff);
          continue;
        }

        const text = (await resp.text()).slice(0, 500);
        logger.error(`${method} ${url} → ${resp.status}: ${text}`);
        return;
      } catch (exc) {
        if (attempt < MAX_RETRIES) {
          const backoff = INITIAL_BACKOFF * 2 ** attempt;
          logger.warning(
            `connection error: ${String(exc)}, retrying in ${backoff.toFixed(1)}s (attempt ${
              attempt + 1
            }/${MAX_RETRIES})`,
          );
          await sleep(backoff);
        } else {
          logger.error(`connection error after ${MAX_RETRIES} retries: ${String(exc)}`);
          throw exc;
        }
      }
    }
  }

  private buildRequest(item: QueueItem): { url: string; method: "POST" | "PATCH"; body: unknown } {
    const base = this.config.endpoint.replace(/\/+$/, "");
    switch (item.kind) {
      case "trace":
        return { url: `${base}/traces`, method: "POST", body: item.payload };
      case "spans":
        return { url: `${base}/traces/${item.traceId}/spans`, method: "POST", body: item.payload };
      case "update_trace":
        return { url: `${base}/traces/${item.traceId}`, method: "PATCH", body: item.payload };
      case "update_span":
        return {
          url: `${base}/traces/${item.traceId}/spans/${item.spanId}`,
          method: "PATCH",
          body: item.payload,
        };
      case "score":
        return { url: `${base}/evaluations/trace-scores`, method: "POST", body: item.payload };
      default: {
        const _exhaustive: never = item.kind;
        throw new Error(`Unknown item kind: ${String(_exhaustive)}`);
      }
    }
  }
}
