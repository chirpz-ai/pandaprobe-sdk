/**
 * Shared test helpers: a fetch mock that records outgoing requests (the analog
 * of Python's respx HTTP mocking) and assertion utilities.
 */

import { vi } from "vitest";

export interface CapturedRequest {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
}

export const captured: CapturedRequest[] = [];

let nextStatus = 202;
const statusQueue: number[] = [];

/** Queue specific HTTP status codes for the next N requests (for retry tests). */
export function queueStatuses(...statuses: number[]): void {
  statusQueue.push(...statuses);
}

/** Set a constant status for subsequent requests. */
export function setStatus(status: number): void {
  nextStatus = status;
}

export function installFetchMock(): void {
  captured.length = 0;
  statusQueue.length = 0;
  nextStatus = 202;
  const mock = vi.fn(async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const url = typeof input === "string" ? input : input.toString();
    const headers: Record<string, string> = {};
    if (init?.headers) {
      for (const [k, v] of Object.entries(init.headers as Record<string, string>)) {
        headers[k] = v;
      }
    }
    let body: unknown = null;
    if (typeof init?.body === "string") {
      try {
        body = JSON.parse(init.body);
      } catch {
        body = init.body;
      }
    }
    captured.push({ url, method: init?.method ?? "GET", headers, body });
    const status = statusQueue.length > 0 ? (statusQueue.shift() as number) : nextStatus;
    return new Response(null, { status });
  });
  globalThis.fetch = mock as unknown as typeof fetch;
}

/** Requests whose URL ends with *suffix*. */
export function requestsTo(suffix: string): CapturedRequest[] {
  return captured.filter((r) => r.url.endsWith(suffix));
}
