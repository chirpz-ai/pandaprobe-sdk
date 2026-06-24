import { describe, expect, it, vi } from "vitest";
import { resolveConfig } from "../src/config.js";
import { Transport } from "../src/transport.js";
import { captured, queueStatuses, requestsTo, setStatus } from "./helpers.js";

function makeTransport(overrides = {}) {
  const config = resolveConfig({
    apiKey: "sk_test",
    projectName: "proj",
    flushInterval: 60,
    batchSize: 10,
    ...overrides,
  });
  return new Transport(config);
}

describe("Transport", () => {
  it("POSTs a trace with the correct URL, method, headers and body", async () => {
    const t = makeTransport();
    t.enqueueTrace({ trace_id: "abc", name: "t" });
    await t.flush();

    const reqs = requestsTo("/traces");
    expect(reqs.length).toBe(1);
    const req = reqs[0]!;
    expect(req.method).toBe("POST");
    expect(req.headers["X-API-Key"]).toBe("sk_test");
    expect(req.headers["X-Project-Name"]).toBe("proj");
    expect(req.headers["Content-Type"]).toBe("application/json");
    expect(req.headers["User-Agent"]).toMatch(/^pandaprobe-typescript\//);
    expect(req.headers["X-Request-ID"]).toBeTruthy();
    expect(req.body).toEqual({ trace_id: "abc", name: "t" });
    await t.shutdown();
  });

  it("builds correct routes for spans, updates and scores", async () => {
    const t = makeTransport();
    t.enqueueSpans("tid", [{ span_id: "s1", name: "s" }]);
    t.enqueueUpdateTrace("tid", { status: "COMPLETED" });
    t.enqueueUpdateSpan("tid", "sid", { status: "OK" });
    t.enqueueScore({ trace_id: "tid", name: "n", value: "1" });
    await t.flush();

    expect(requestsTo("/traces/tid/spans")[0]?.method).toBe("POST");
    expect(requestsTo("/traces/tid/spans")[0]?.body).toEqual([{ span_id: "s1", name: "s" }]);
    expect(requestsTo("/traces/tid")[0]?.method).toBe("PATCH");
    expect(requestsTo("/traces/tid/spans/sid")[0]?.method).toBe("PATCH");
    expect(requestsTo("/evaluations/trace-scores")[0]?.method).toBe("POST");
    await t.shutdown();
  });

  it("retries on 503 then succeeds", async () => {
    const t = makeTransport();
    queueStatuses(503, 202);
    t.enqueueTrace({ trace_id: "r", name: "t" });
    await t.flush();
    expect(requestsTo("/traces").length).toBe(2);
    await t.shutdown();
  });

  it("does not retry on 401", async () => {
    const t = makeTransport();
    queueStatuses(401);
    t.enqueueTrace({ trace_id: "x", name: "t" });
    await t.flush();
    expect(requestsTo("/traces").length).toBe(1);
    await t.shutdown();
  });

  it("drops oldest items when the queue overflows", async () => {
    const t = makeTransport({ maxQueueSize: 2 });
    t.enqueueTrace({ trace_id: "1", name: "t" });
    t.enqueueTrace({ trace_id: "2", name: "t" });
    t.enqueueTrace({ trace_id: "3", name: "t" });
    await t.flush();
    const reqs = requestsTo("/traces");
    expect(reqs.length).toBe(2);
    const ids = reqs.map((r) => (r.body as { trace_id: string }).trace_id);
    expect(ids).toEqual(["2", "3"]);
    await t.shutdown();
  });

  it("auto-drains when the batch size is reached", async () => {
    const t = makeTransport({ batchSize: 2 });
    t.enqueueTrace({ trace_id: "a", name: "t" });
    t.enqueueTrace({ trace_id: "b", name: "t" });
    // Reaching batchSize triggers a drain without an explicit flush.
    await new Promise((r) => setTimeout(r, 50));
    expect(requestsTo("/traces").length).toBe(2);
    await t.shutdown();
  });

  it("no-ops entirely when disabled", async () => {
    const config = resolveConfig({ enabled: false });
    const t = new Transport(config);
    t.enqueueTrace({ trace_id: "z", name: "t" });
    await t.flush();
    expect(captured.length).toBe(0);
    await t.shutdown();
  });

  it("invokes the onError callback after retries are exhausted on network errors", async () => {
    const errors: unknown[] = [];
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      throw new Error("network down");
    }) as unknown as typeof fetch;

    const config = resolveConfig({ apiKey: "sk_test", projectName: "proj", flushInterval: 60 });
    const t = new Transport(config, (e) => errors.push(e));
    t.enqueueTrace({ trace_id: "n", name: "t" });
    await t.flush();

    globalThis.fetch = originalFetch;
    expect(errors.length).toBe(1);
    expect(String(errors[0])).toContain("network down");
    await t.shutdown();
  }, 15000);

  it("shutdown is idempotent", async () => {
    const t = makeTransport();
    await t.shutdown();
    await expect(t.shutdown()).resolves.toBeUndefined();
  });

  it("flushes buffered traces then re-raises the signal on SIGTERM", async () => {
    const t = makeTransport();
    t.enqueueTrace({ trace_id: "sig", name: "t" });

    // Grab the handler the constructor registered (re-raising is mocked so the
    // test process is not actually terminated).
    const listeners = process.listeners("SIGTERM");
    const handler = listeners[listeners.length - 1] as (s: NodeJS.Signals) => void;
    const killSpy = vi.spyOn(process, "kill").mockImplementation(() => true);

    try {
      handler("SIGTERM");
      await new Promise((r) => setTimeout(r, 50)); // let shutdown drain + finally run

      expect(requestsTo("/traces").length).toBe(1); // buffered trace was flushed
      expect(killSpy).toHaveBeenCalledWith(process.pid, "SIGTERM"); // signal re-raised
    } finally {
      killSpy.mockRestore();
    }
    await t.shutdown();
  });

  it("logs and stops (does not silently drop) after exhausting retries on a persistent 503", async () => {
    const t = makeTransport();
    setStatus(503); // every request returns 503
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      t.enqueueTrace({ trace_id: "fail", name: "t" });
      await t.flush();

      // 1 initial attempt + MAX_RETRIES(3) = 4 fetches, then it gives up.
      expect(requestsTo("/traces").length).toBe(4);
      // The give-up is logged (not a silent drop / no bogus "attempt 4/3").
      const loggedGiveUp = errSpy.mock.calls.some((c) => String(c[0]).includes("giving up"));
      expect(loggedGiveUp).toBe(true);
    } finally {
      errSpy.mockRestore();
    }
    await t.shutdown();
  }, 15000);
});
