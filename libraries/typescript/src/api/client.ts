import { type Client, getClient } from "../client.js";
import { type SdkConfig, resolveConfig } from "../config.js";
import {
  ApiDeadlineExceeded,
  ApiError,
  AuthenticationError,
  CreationOutcomeUnknown,
  InvalidResponseError,
  NotFoundError,
  RateLimitError,
  TransportError,
} from "./errors.js";
import { Evaluations, Traces } from "./resources.js";

export interface RequestOptions {
  timeout?: number;
  signal?: AbortSignal;
  query?: Record<string, string | number | undefined>;
  body?: unknown;
}
export interface PlatformOptions {
  config?: SdkConfig;
  timeout?: number;
  readAttempts?: number;
  fetch?: typeof fetch;
}
export function positive(value: number, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0)
    throw new TypeError(`${name} must be finite and positive`);
}
export function abortError(): Error {
  const error = new Error("Operation cancelled");
  error.name = "AbortError";
  return error;
}
export function pause(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason ?? abortError());
    const abort = () => {
      clearTimeout(timer);
      reject(signal?.reason ?? abortError());
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", abort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", abort, { once: true });
  });
}
/** Native, independently owned platform requests. All durations are seconds. */
export class PlatformClient {
  readonly config: SdkConfig;
  readonly timeout: number;
  readonly readAttempts: number;
  readonly traces: Traces;
  readonly evaluations: Evaluations;
  private readonly transport: typeof fetch;
  private readonly requests = new Set<AbortController>();
  private closed = false;
  constructor(client?: Client, options: PlatformOptions = {}) {
    if (client && options.config) throw new TypeError("Pass client or config, not both");
    this.config = options.config ?? (client ?? getClient())?.config ?? resolveConfig();
    this.timeout = options.timeout ?? 30;
    this.readAttempts = options.readAttempts ?? 3;
    positive(this.timeout, "timeout");
    if (!Number.isSafeInteger(this.readAttempts) || this.readAttempts < 1)
      throw new TypeError("readAttempts must be positive");
    if (!this.config.apiKey || !this.config.projectName)
      throw new TypeError("Platform access requires the SDK API key and project name");
    const url = new URL(this.config.endpoint);
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash)
      throw new TypeError("The platform endpoint must be HTTP(S) without credentials, query, or fragment");
    this.transport = options.fetch ?? globalThis.fetch;
    this.traces = new Traces(this);
    this.evaluations = new Evaluations(this);
  }
  async close(): Promise<void> {
    this.closed = true;
    for (const request of this.requests) request.abort(abortError());
  }
  async request(method: "GET" | "POST", path: string, options: RequestOptions = {}): Promise<unknown> {
    if (this.closed) throw new Error("PlatformClient is closed");
    const budget = options.timeout ?? this.timeout;
    if (!Number.isFinite(budget) || budget <= 0) throw new ApiDeadlineExceeded("Platform operation deadline exceeded");
    // Resource paths only. Never allow an absolute URL to redirect identity headers.
    if (!/^[a-zA-Z0-9/-]+$/.test(path) || path.startsWith("/")) throw new TypeError("Invalid platform resource path");
    const url = new URL(`${this.config.endpoint.replace(/\/+$/, "")}/${path}`);
    for (const [key, value] of Object.entries(options.query ?? {}))
      if (value !== undefined) url.searchParams.set(key, String(value));
    const controller = new AbortController();
    this.requests.add(controller);
    const abort = () => controller.abort(options.signal?.reason ?? abortError());
    if (options.signal?.aborted) abort();
    else options.signal?.addEventListener("abort", abort, { once: true });
    let expired = false;
    const deadline = performance.now() + budget * 1000;
    const timer = setTimeout(() => {
      expired = true;
      controller.abort(new ApiDeadlineExceeded("Platform operation deadline exceeded"));
    }, budget * 1000);
    const attempts = method === "GET" ? this.readAttempts : 1;
    try {
      for (let attempt = 0; attempt < attempts; attempt++) {
        controller.signal.throwIfAborted();
        let response: Response;
        try {
          response = await this.transport(url, {
            method,
            redirect: "manual",
            signal: controller.signal,
            headers: {
              "X-API-Key": this.config.apiKey!,
              "X-Project-Name": this.config.projectName!,
              ...(options.body === undefined ? {} : { "Content-Type": "application/json" }),
            },
            body: options.body === undefined ? undefined : JSON.stringify(options.body),
          });
          if (response.ok) {
            try {
              return await response.json();
            } catch {
              if (controller.signal.aborted) throw controller.signal.reason;
              throw method === "GET"
                ? new InvalidResponseError("The platform returned invalid JSON")
                : new CreationOutcomeUnknown("The platform returned invalid JSON; creation outcome is unknown");
            }
          }
        } catch (error) {
          if (options.signal?.aborted || this.closed) throw controller.signal.reason;
          if (error instanceof InvalidResponseError || error instanceof CreationOutcomeUnknown) throw error;
          if (method !== "GET")
            throw new CreationOutcomeUnknown("Evaluation creation outcome is unknown; do not retry");
          if (expired) throw new ApiDeadlineExceeded("Platform operation deadline exceeded");
          if (attempt + 1 === attempts) throw new TransportError("Platform transport failed");
          await this.delay(attempt, deadline, controller.signal);
          continue;
        }
        const status = response.status;
        await response.body?.cancel();
        if (method !== "GET" && (status >= 500 || status === 408))
          throw new CreationOutcomeUnknown(`Creation returned HTTP ${status}; the run may exist`);
        if ([408, 429, 500, 502, 503, 504].includes(status) && attempt + 1 < attempts) {
          await this.delay(attempt, deadline, controller.signal, response.headers.get("Retry-After"));
          continue;
        }
        const ErrorType =
          (
            { 401: AuthenticationError, 403: AuthenticationError, 404: NotFoundError, 429: RateLimitError } as Record<
              number,
              typeof ApiError
            >
          )[status] ?? ApiError;
        throw new ErrorType(
          `Platform request failed with HTTP ${status}`,
          status,
          response.headers.get("X-Request-ID"),
        );
      }
      throw new Error("Unreachable");
    } finally {
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", abort);
      this.requests.delete(controller);
    }
  }
  private async delay(
    attempt: number,
    deadline: number,
    signal: AbortSignal,
    retryAfter?: string | null,
  ): Promise<void> {
    let delay = 0.1 + Math.random() * (Math.min(2 ** attempt, 5) - 0.1);
    if (retryAfter) {
      const numeric = Number(retryAfter);
      const date = Date.parse(retryAfter);
      if (Number.isFinite(numeric)) delay = numeric;
      else if (Number.isFinite(date)) delay = (date - Date.now()) / 1000;
    }
    delay = Math.max(0, Math.min(Number.isFinite(delay) ? delay : 1, 30));
    if (delay * 1000 >= deadline - performance.now())
      throw new ApiDeadlineExceeded("Retry exceeds platform operation deadline");
    await pause(delay * 1000, signal);
  }
}
