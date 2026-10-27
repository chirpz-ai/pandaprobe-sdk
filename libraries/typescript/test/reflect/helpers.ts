import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach } from "vitest";
import { PlatformClient } from "../../src/api/index.js";
import { Client } from "../../src/client.js";
import { ReflectConfig, type ReflectOptions, Reflect as ReflectRuntime } from "../../src/reflect/index.js";
import { TraceData } from "../../src/schemas.js";
export const RULE = "# Refund\nApplies to: return refunds\nExclude outbound shipping; refund merchandise only.\n";
export const WRITE_RULE = `cat > /rules/refund.md <<'RULE'\n${RULE}RULE`;
const cleanups: Array<() => Promise<void>> = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
export class PlatformFixture {
  traces = new Map<string, Record<string, any>>();
  runs = new Map<string, Record<string, any>>();
  creates = 0;
  polls = 0;
  fault: "none" | "lost" | "running" | "missing" | "auth" = "none";
  score = 0.1;
  reason = "The refund included shipping.";
  metadata: Record<string, unknown> = {};
  fetch: typeof fetch = async (input, options) => {
    const url = new URL(String(input));
    const parts = url.pathname.split("/");
    if (options?.signal?.aborted) throw options.signal.reason;
    if (options?.method === "POST") {
      this.creates++;
      if (this.fault === "lost") throw new Error("response lost");
      if (this.fault === "auth") return new Response(null, { status: 401 });
      const body = JSON.parse(String(options.body));
      const id = randomUUID();
      this.runs.set(id, {
        id,
        status: "PENDING",
        metric_names: body.metrics,
        total_targets: body.trace_ids.length,
        evaluated_count: body.trace_ids.length,
        target_type: "TRACE",
        trace_ids: body.trace_ids,
      });
      return Response.json(this.runs.get(id), { status: 202 });
    }
    if (parts.includes("traces")) {
      const id = parts[parts.length - 1]!;
      const value = this.traces.get(id);
      return value
        ? Response.json({ ...value, metadata: { ...value.metadata, ...this.metadata } })
        : new Response(null, { status: 404 });
    }
    if (url.pathname.endsWith("/scores")) {
      const id = parts[parts.length - 2]!;
      const run = this.runs.get(id)!;
      return Response.json(
        this.fault === "missing"
          ? []
          : run.trace_ids.flatMap((trace_id: string) =>
              run.metric_names.map((name: string) => ({
                id: randomUUID(),
                trace_id,
                name,
                value: String(this.score),
                status: "SUCCESS",
                data_type: "NUMERIC",
                eval_run_id: id,
                reason: this.reason,
              })),
            ),
      );
    }
    this.polls++;
    const run = this.runs.get(parts[parts.length - 1]!)!;
    return Response.json({ ...run, status: this.fault === "running" ? "RUNNING" : "COMPLETED" });
  };
  addTrace(client?: Client): string {
    const trace = new TraceData({ name: "task", startedAt: new Date() });
    const payload = {
      ...trace.toApiDict(),
      trace_id: trace.traceId,
      project_id: randomUUID(),
      status: "COMPLETED",
      started_at: new Date().toISOString(),
      metadata: {},
    };
    this.traces.set(trace.traceId, payload);
    client?.logTrace(trace);
    return trace.traceId;
  }
}
export function fixture(options: ReflectOptions = {}, server = new PlatformFixture(), folder?: string) {
  const workspace = folder ?? mkdtempSync(join(tmpdir(), "reflect-runtime-"));
  const client = new Client({
    apiKey: "test-key",
    projectName: "test-project",
    endpoint: "http://fixture",
    flushInterval: 100,
  });
  const api = new PlatformClient(client, { fetch: server.fetch });
  const config = new ReflectConfig({
    workspace,
    thresholds: { task_completion: 0.5 },
    diagnosticMetrics: [],
    cooldownSeconds: 0,
    pollInterval: 0.001,
    traceVisibilityTimeout: 0.05,
    evaluationTimeout: 0.05,
    ...options,
  });
  const runtime = new ReflectRuntime(client, { api, config });
  cleanups.push(async () => {
    await runtime.close();
    await api.close();
    await client.shutdown();
    if (!folder) rmSync(workspace, { recursive: true, force: true });
  });
  return { runtime, client, api, server, workspace };
}
