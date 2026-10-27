import { expect, it } from "vitest";
import { Client } from "../src/client.js";
import { TraceData } from "../src/schemas.js";
import { captured } from "./helpers.js";
const client = () => new Client({ apiKey: "key", projectName: "project" });
const trace = () => new TraceData({ name: "task", startedAt: new Date() });
it("captures existing submissions without creating or mutating traces", async () => {
  const c = client();
  const t = trace();
  const capture = c.captureTraces({ metadata: { phase: "reflection" }, sessionId: "scope" });
  await capture.run(() => {
    c.logTrace(t);
    c.logTrace(t);
  });
  expect(capture.traceIds).toEqual([t.traceId]);
  expect(t.metadata).toEqual({});
  expect(t.sessionId).not.toBe("scope");
  await c.shutdown();
  expect(captured.some((r) => JSON.stringify(r.body).includes('"phase":"reflection"'))).toBe(true);
});
it("isolates concurrent clients, nested scopes, overflow and restoration", async () => {
  const a = client();
  const b = client();
  const outer = a.captureTraces({ maxTraces: 1 });
  const inner = a.captureTraces();
  const other = b.captureTraces();
  await Promise.all([
    outer.run(async () => {
      a.logTrace(trace());
      await inner.run(() => a.logTrace(trace()));
      a.logTrace(trace());
      b.logTrace(trace());
    }),
    other.run(async () => {
      await Promise.resolve();
      b.logTrace(trace());
    }),
  ]);
  expect([outer.traceIds.length, inner.traceIds.length, other.traceIds.length]).toEqual([1, 1, 1]);
  expect(outer.overflow).toBe(true);
  a.logTrace(trace());
  expect(outer.traceIds).toHaveLength(1);
  await expect(outer.run(() => {})).rejects.toThrow("reused");
  await Promise.all([a.shutdown(), b.shutdown()]);
});
it("preserves original errors and isolates throwing or asynchronous observers", async () => {
  const c = client();
  const original = new Error("task");
  const capture = c.captureTraces({
    onSubmit: () => {
      throw new Error("observer");
    },
  });
  await expect(
    capture.run(() => {
      c.logTrace(trace());
      throw original;
    }),
  ).rejects.toBe(original);
  const second = c.captureTraces({
    onSubmit: async (payload) => {
      payload.session_id = "evil";
      throw Error("async");
    },
  });
  await second.run(() => c.logTrace(trace()));
  await c.shutdown();
  expect(capture.closed).toBe(true);
  expect(captured.some((r) => JSON.stringify(r.body).includes("evil"))).toBe(false);
});
it("late descendants retain attribution but cannot extend closed boundaries or observers", async () => {
  const c = client();
  let release!: () => void;
  const gate = new Promise<void>((r) => {
    release = r;
  });
  let late!: Promise<void>;
  let calls = 0;
  const capture = c.captureTraces({
    metadata: { phase: "validation" },
    sessionId: "internal",
    onSubmit: () => calls++,
  });
  await capture.run(() => {
    late = (async () => {
      await gate;
      c.logTrace(trace());
    })();
  });
  release();
  await late;
  expect(capture.traceIds).toEqual([]);
  expect(calls).toBe(0);
  await c.shutdown();
  expect(captured.some((r) => JSON.stringify(r.body).includes('"phase":"validation"'))).toBe(true);
});
