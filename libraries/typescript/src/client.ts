/**
 * PandaProbe Client — main entry point for the SDK.
 */

import { type ConfigOptions, type SdkConfig, envEnabled, resolveConfig } from "./config.js";
import { logger } from "./logger.js";
import { ScoreData, type ScoreDataType, type TraceData } from "./schemas.js";
import { TraceContext, type TraceContextOptions } from "./tracing/context.js";
import { Transport } from "./transport.js";

// ---------------------------------------------------------------------------
// Module-level singleton
// ---------------------------------------------------------------------------

let globalClient: Client | null = null;
let autoInitAttempted = false;

/** Configure and set the global PandaProbe client singleton. */
export function init(options: ConfigOptions = {}): Client {
  const previous = globalClient;
  globalClient = new Client(options);
  if (previous !== null) {
    // Drain the previous client's buffered traces in the background, then release its resources.
    void previous.shutdown().catch(() => {});
  }
  return globalClient;
}

/**
 * Return the current global client.
 *
 * On first call, if no client has been created via {@link init}, attempts
 * auto-initialization from environment variables. Auto-init is gated by
 * `PANDAPROBE_ENABLED` (defaults to true).
 */
export function getClient(): Client | null {
  if (globalClient !== null) {
    return globalClient;
  }
  if (!autoInitAttempted) {
    autoInitAttempted = true;
    globalClient = tryAutoInit();
  }
  return globalClient;
}

function tryAutoInit(): Client | null {
  if (!envEnabled()) {
    return null;
  }
  try {
    return new Client();
  } catch (exc) {
    logger.debug(`auto-init skipped: ${String(exc)}`);
    return null;
  }
}

/** Test-only: reset the module singleton (mirrors conftest reset). */
export function _resetClientForTests(): void {
  globalClient = null;
  autoInitAttempted = false;
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export class Client {
  private readonly _config: SdkConfig;
  private readonly errorCallbacks: Array<(exc: unknown) => void> = [];
  private readonly transport: Transport;

  constructor(options: ConfigOptions = {}) {
    this._config = resolveConfig(options);
    this.transport = new Transport(this._config, (exc) => this.dispatchError(exc));
    if (this._config.enabled) {
      logger.debug(`client initialised (endpoint=${this._config.endpoint})`);
    }
  }

  get enabled(): boolean {
    return this._config.enabled;
  }

  get config(): SdkConfig {
    return this._config;
  }

  // ------------------------------------------------------------------
  // Trace operations
  // ------------------------------------------------------------------

  /** Create a new {@link TraceContext}. */
  trace(name: string, options: TraceContextOptions = {}): TraceContext {
    return new TraceContext(this, name, options);
  }

  /** Enqueue a fully-formed trace for background submission. */
  logTrace(traceData: TraceData): void {
    if (!this._config.enabled) {
      return;
    }
    if (this._config.environment && !traceData.environment) {
      traceData.environment = this._config.environment;
    }
    if (this._config.release && !traceData.release) {
      traceData.release = this._config.release;
    }
    this.transport.enqueueTrace(traceData.toApiDict());
  }

  // ------------------------------------------------------------------
  // Score operations
  // ------------------------------------------------------------------

  /** Submit a programmatic score for a trace. */
  score(
    traceId: string,
    name: string,
    value: string,
    options: { dataType?: ScoreDataType; reason?: string | null; metadata?: Record<string, unknown> } = {},
  ): void {
    if (!this._config.enabled) {
      return;
    }
    const score = new ScoreData({
      traceId,
      name,
      value,
      dataType: options.dataType,
      reason: options.reason ?? null,
      metadata: options.metadata ?? {},
    });
    this.transport.enqueueScore(score.toApiDict());
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /** Block until all queued items are sent. */
  async flush(timeout = 30.0): Promise<void> {
    await this.transport.flush(timeout);
  }

  /** Flush remaining items and release resources. */
  async shutdown(): Promise<void> {
    await this.transport.shutdown();
  }

  // ------------------------------------------------------------------
  // Error handling
  // ------------------------------------------------------------------

  /** Register an error callback invoked when a transport error occurs. */
  onError(callback: (exc: unknown) => void): void {
    this.errorCallbacks.push(callback);
  }

  private dispatchError(exc: unknown): void {
    for (const cb of this.errorCallbacks) {
      try {
        cb(exc);
      } catch {
        // swallow callback errors
      }
    }
  }
}
