import { randomUUID } from "node:crypto";
/** Loopback-only example fixture. Never shipped as SDK runtime. */
import { createServer } from "node:http";
export async function localPlatform(judge: (trace: any, metric: string) => { score: number; reason: string }) {
  const state = {
    traces: new Map<string, any>(),
    runs: new Map<string, any>(),
    reads: new Map<string, number>(),
    polls: new Map<string, number>(),
    creates: 0,
    requests: 0,
  };
  const projectId = randomUUID();
  const server = createServer(async (request, response) => {
    state.requests++;
    const reply = (status: number, value: unknown) => {
      response.writeHead(status, { "Content-Type": "application/json" });
      response.end(JSON.stringify(value));
    };
    if (request.headers["x-api-key"] !== "offline-key" || request.headers["x-project-name"] !== "offline-demo")
      return reply(401, { detail: "Incorrect explicit SDK identity" });
    const url = new URL(request.url!, "http://127.0.0.1");
    const parts = url.pathname.split("/");
    const run = (id: string, status: string) => {
      const value = state.runs.get(id);
      return {
        id,
        status,
        metric_names: value.metrics,
        target_type: "TRACE",
        total_targets: value.trace_ids.length,
        evaluated_count: status === "COMPLETED" ? value.trace_ids.length : 0,
        failed_count: 0,
      };
    };
    try {
      if (request.method === "POST") {
        let body = "";
        for await (const chunk of request) body += chunk;
        const data = JSON.parse(body);
        if (url.pathname === "/traces") {
          state.traces.set(data.trace_id, {
            project_id: projectId,
            ...data,
            spans: (data.spans ?? []).map((span: any) => ({ trace_id: data.trace_id, ...span })),
          });
          return reply(202, { trace_id: data.trace_id });
        }
        if (url.pathname === "/evaluations/trace-runs/batch") {
          const id = randomUUID();
          state.creates++;
          state.runs.set(id, data);
          return reply(202, run(id, "PENDING"));
        }
      }
      if (request.method === "GET" && parts[1] === "traces") {
        const id = parts[2]!;
        const count = state.reads.get(id) ?? 0;
        state.reads.set(id, count + 1);
        return count && state.traces.has(id)
          ? reply(200, state.traces.get(id))
          : reply(404, { detail: "Not persisted yet" });
      }
      if (request.method === "GET" && parts[2] === "trace-runs") {
        const id = parts[3]!;
        const value = state.runs.get(id);
        if (!value) return reply(404, {});
        if (parts[4] === "scores")
          return reply(
            200,
            value.trace_ids.flatMap((traceId: string) =>
              value.metrics.map((metric: string) => {
                const result = judge(state.traces.get(traceId), metric);
                return {
                  id: randomUUID(),
                  trace_id: traceId,
                  eval_run_id: id,
                  name: metric,
                  value: String(result.score),
                  status: "SUCCESS",
                  data_type: "NUMERIC",
                  reason: result.reason,
                };
              }),
            ),
          );
        const count = (state.polls.get(id) ?? 0) + 1;
        state.polls.set(id, count);
        return reply(200, run(id, count >= 2 ? "COMPLETED" : "RUNNING"));
      }
      reply(404, {});
    } catch (error) {
      reply(500, { error: String(error) });
    }
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") throw Error("Missing loopback listener");
  return {
    endpoint: `http://127.0.0.1:${address.port}`,
    state,
    close: async () => {
      server.closeAllConnections();
      await new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
    },
  };
}
