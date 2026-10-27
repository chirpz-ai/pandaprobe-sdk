import { randomUUID } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import {
  ApiDeadlineExceeded,
  ApiError,
  AuthenticationError,
  CreationOutcomeUnknown,
  EvaluationFailed,
  InsufficientEvidence,
  InvalidResponseError,
  PlatformClient,
  RateLimitError,
  TransportError,
  numericValue,
} from "../src/api/index.js";
import { decode, decoders } from "../src/api/models.js";
import { resolveConfig } from "../src/config.js";
import contract from "./fixtures/platform_contract.json";
const tid = randomUUID();
const rid = randomUUID();
const pid = randomUUID();
const trace = (status = "COMPLETED") => ({
  trace_id: tid,
  project_id: pid,
  name: "task",
  status,
  started_at: "2026-10-19T00:00:00Z",
  session_id: "session",
  spans: [],
});
const run = (status = "COMPLETED") => ({
  id: rid,
  status,
  metric_names: ["task_completion"],
  total_targets: 1,
  evaluated_count: 1,
  target_type: "TRACE",
});
const score = (changes = {}) => ({
  id: randomUUID(),
  trace_id: tid,
  name: "task_completion",
  value: "0.8",
  status: "SUCCESS",
  data_type: "NUMERIC",
  eval_run_id: rid,
  ...changes,
});
const response = (v: unknown, status = 200, headers = {}) => new Response(JSON.stringify(v), { status, headers });
function api(handler: typeof fetch, options = {}) {
  return new PlatformClient(undefined, {
    config: resolveConfig({
      apiKey: "explicit-key",
      projectName: "explicit-project",
      endpoint: "https://api.example/root",
    }),
    fetch: handler,
    ...options,
  });
}

describe("platform API", () => {
  it("reuses explicit identity, preserves additive fields and owns its lifecycle", async () => {
    const fetch = vi.fn(async (url, options) => {
      expect(String(url)).toBe(`https://api.example/root/traces/${tid}`);
      expect(options.headers).toMatchObject({ "X-API-Key": "explicit-key", "X-Project-Name": "explicit-project" });
      expect(options.redirect).toBe("manual");
      return response({ ...trace(), future_field: 42 });
    });
    const client = api(fetch);
    expect((await client.traces.get(tid)).future_field).toBe(42);
    await client.close();
    await expect(client.traces.get(tid)).rejects.toThrow("closed");
  });
  it("rejects endpoints and absolute resource paths that could leak credentials", async () => {
    for (const endpoint of ["ftp://host", "https://user:password@host", "https://host?q=x", "https://host#x"])
      expect(
        () => new PlatformClient(undefined, { config: resolveConfig({ apiKey: "k", projectName: "p", endpoint }) }),
      ).toThrow();
    await expect(api(vi.fn()).request("GET", "https://elsewhere")).rejects.toThrow("resource path");
  });
  it("rejects mismatched response identities", async () => {
    await expect(api(async () => response(trace())).traces.get(randomUUID())).rejects.toBeInstanceOf(
      InvalidResponseError,
    );
    await expect(api(async () => response(run())).evaluations.get(randomUUID())).rejects.toBeInstanceOf(
      InvalidResponseError,
    );
  });
  it("waits for persisted terminal ERROR traces", async () => {
    let n = 0;
    const client = api(async () => (++n === 1 ? response({}, 404) : response(trace(n === 2 ? "RUNNING" : "ERROR"))));
    expect((await client.traces.waitVisible(tid, { timeout: 1, pollInterval: 0.001 })).status).toBe("ERROR");
    expect(n).toBe(3);
  });
  it("creates once and polls the same run", async () => {
    let creates = 0;
    let polls = 0;
    const client = api(async (url, options) => {
      if (options?.method === "POST") {
        creates++;
        expect(JSON.parse(String(options.body))).toEqual({ trace_ids: [tid], metrics: ["task_completion"] });
        return response(run("PENDING"), 202);
      }
      if (String(url).endsWith("scores")) return response([score()]);
      return response(run(++polls < 3 ? "RUNNING" : "COMPLETED"));
    });
    const created = await client.evaluations.create([tid, tid], ["task_completion", "task_completion"]);
    const result = await client.evaluations.wait(created.id, {
      traceIds: [tid],
      metrics: ["task_completion"],
      timeout: 1,
      pollInterval: 0.001,
    });
    expect(numericValue(result.scores[0]!)).toBe(0.8);
    expect([creates, polls]).toEqual([1, 3]);
  });
  it.each(
    [
      [],
      [score({ value: null })],
      [score({ status: "FAILED" })],
      [score({ value: "NaN" })],
      [score({ value: "Infinity" })],
      [score({ value: "" })],
      [score({ value: "0x1" })],
      [score({ eval_run_id: null })],
      [score({ data_type: "BOOLEAN" })],
      [score(), score()],
    ].map((values) => ({ values })),
  )("requires every usable unique score ($values)", async ({ values }) => {
    const client = api(async (url) => response(String(url).endsWith("scores") ? values : run()));
    await expect(
      client.evaluations.wait(rid, { traceIds: [tid], metrics: ["task_completion"] }),
    ).rejects.toBeInstanceOf(InsufficientEvidence);
  });
  it.each(["FAILED", "CANCELLED"])("preserves terminal %s", async (status) => {
    await expect(
      api(async () => response(run(status))).evaluations.wait(rid, { traceIds: [tid], metrics: ["task_completion"] }),
    ).rejects.toBeInstanceOf(EvaluationFailed);
  });
  it.each(["transport", "server", "json", "shape", "timeout"])("never retries ambiguous create: %s", async (fault) => {
    const fetch = vi.fn(async (_url, options) => {
      if (fault === "transport") throw new Error("lost");
      if (fault === "server") return response({}, 503);
      if (fault === "json") return new Response("no");
      if (fault === "timeout")
        return new Promise<Response>((_, reject) =>
          options.signal.addEventListener("abort", () => reject(options.signal.reason), { once: true }),
        );
      return response({});
    });
    await expect(api(fetch, { timeout: 0.01 }).evaluations.create([tid], ["task_completion"])).rejects.toBeInstanceOf(
      CreationOutcomeUnknown,
    );
    expect(fetch).toHaveBeenCalledTimes(1);
  });
  it("retries bounded safe reads, honors Retry-After and does not follow redirects", async () => {
    let n = 0;
    const client = api(async () => (++n === 1 ? response({}, 429, { "Retry-After": "0" }) : response(trace())));
    await client.traces.get(tid);
    expect(n).toBe(2);
    await expect(
      api(async () => response({}, 302, { Location: "https://elsewhere" })).traces.get(tid),
    ).rejects.toBeInstanceOf(ApiError);
  });
  it.each([
    [401, AuthenticationError],
    [403, AuthenticationError],
    [429, RateLimitError],
  ] as const)("distinguishes HTTP %s", async (status, ErrorType) => {
    await expect(api(async () => response({}, status), { readAttempts: 1 }).traces.get(tid)).rejects.toBeInstanceOf(
      ErrorType,
    );
  });
  it("bounds polling and read retry deadlines", async () => {
    await expect(
      api(async () => response(run("RUNNING"))).evaluations.wait(rid, {
        traceIds: [tid],
        metrics: ["task_completion"],
        timeout: 0.01,
        pollInterval: 0.001,
      }),
    ).rejects.toBeInstanceOf(ApiDeadlineExceeded);
    await expect(
      api(async () => response({}, 429, { "Retry-After": "1000" }), { timeout: 0.01 }).traces.get(tid),
    ).rejects.toBeInstanceOf(ApiDeadlineExceeded);
    await expect(
      api(
        async () => {
          throw Error("offline");
        },
        { readAttempts: 1 },
      ).traces.get(tid),
    ).rejects.toBeInstanceOf(TransportError);
  });
  it("propagates caller cancellation and close into the network operation", async () => {
    const fetch = vi.fn(
      (_url, options) =>
        new Promise<Response>((_, reject) =>
          options.signal.addEventListener("abort", () => reject(options.signal.reason), { once: true }),
        ),
    );
    const controller = new AbortController();
    const client = api(fetch);
    const pending = client.traces.get(tid, { signal: controller.signal });
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(fetch).toHaveBeenCalledTimes(1);
    const second = client.traces.get(tid);
    await client.close();
    await expect(second).rejects.toMatchObject({ name: "AbortError" });
  });
  it("validates required wire fields and keeps the OpenAPI fixture available", () => {
    const schemas = contract.schemas as Record<string, any>;
    expect(schemas.CreateBatchEvalRunRequest.required).toEqual(["trace_ids", "metrics"]);
    expect(contract.creation_headers).toEqual([]); // No server idempotency-key contract.
    const contracts: Array<[string, (value: any) => any, any, string[]]> = [
      ["TraceResponse", decoders.trace, trace(), ["trace_id", "project_id", "name", "status", "started_at"]],
      [
        "EvalRunResponse",
        decoders.run,
        run(),
        ["id", "status", "metric_names", "total_targets", "evaluated_count", "target_type"],
      ],
      ["TraceScoreResponse", decoders.score, score(), ["id", "trace_id", "name", "value", "status", "data_type"]],
    ];
    for (const [name, parser, example, fields] of contracts) {
      for (const field of fields) {
        expect(schemas[name].properties).toHaveProperty(field);
        const missing = { ...example };
        delete missing[field];
        expect(() => decode(parser, missing), `${name}.${field}`).toThrow(InvalidResponseError);
      }
    }
    expect(() => decode(decoders.trace, { ...trace(), status: "DONE" })).toThrow(InvalidResponseError);
    expect(decode(decoders.run, { ...run(), total_targets: true }).total_targets).toBe(1);
  });
  it("paginates with explicit bounds and filters", async () => {
    const offsets: number[] = [];
    const client = api(async (url) => {
      const u = new URL(String(url));
      expect(u.searchParams.get("session_id")).toBe("s");
      const offset = Number(u.searchParams.get("offset"));
      offsets.push(offset);
      return response({ items: [trace()], total: 2, limit: 1, offset });
    });
    const rows = [];
    for await (const row of client.traces.iterate({ sessionId: "s", maxItems: 2 })) rows.push(row);
    expect(rows).toHaveLength(2);
    expect(offsets).toEqual([0, 1]);
  });
});
