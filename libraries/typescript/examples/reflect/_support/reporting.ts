/** Example equivalents of runner.py's artifact, deadline and trace-usage helpers. */
import { randomUUID } from "node:crypto";
import { readFileSync, renameSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { basename, dirname, join } from "node:path";
import { DeadlineExceeded, Semaphore, waitFor } from "../../../src/reflect/async.js";
import type { Reflect as ReflectRuntime } from "../../../src/reflect/index.js";
import { json } from "../../../src/reflect/serialization.js";
import { VERSION } from "../../../src/version.js";

export const identity = () => randomUUID().replaceAll("-", "");
/** Preserve sub-millisecond ordering of model input and business submission. */
export function timestamp(): string {
  const micros = BigInt(Math.floor((performance.timeOrigin + performance.now()) * 1000));
  return `${new Date(Number(micros / 1000n)).toISOString().slice(0, 19)}.${String(micros % 1000000n).padStart(6, "0")}+00:00`;
}
export function saveReport(path: string, report: unknown): void {
  const temporary = `${path.replace(/\.[^/.]+$/, "")}.tmp`;
  // Python's runner writes sorted, indented JSON through an atomic replacement.
  writeFileSync(temporary, `${JSON.stringify(JSON.parse(json(JSON.parse(JSON.stringify(report)))), null, 2)}\n`);
  renameSync(temporary, path);
}
export const seconds = (started: number) => Math.round(performance.now() - started) / 1000;
export function versions(framework: string): Record<string, string> {
  const result: Record<string, string> = { pandaprobe: VERSION };
  const require = createRequire(import.meta.url);
  for (const name of [
    "@everruns/bashkit",
    {
      langgraph: "@langchain/langgraph",
      "openai-agents": "@openai/agents",
      "claude-agent-sdk": "@anthropic-ai/claude-agent-sdk",
      "vercel-ai": "ai",
    }[framework] ?? framework,
  ]) {
    try {
      let folder = dirname(require.resolve(name));
      for (;;) {
        try {
          const value = JSON.parse(readFileSync(join(folder, "package.json"), "utf8"));
          if (value.name === name) {
            result[name] = value.version;
            break;
          }
        } catch {}
        const parent = dirname(folder);
        if (parent === folder) throw Error(`No package metadata for ${name}`);
        folder = parent;
      }
    } catch {
      result[name] = "unavailable";
    }
  }
  return result;
}
export interface RunOptions {
  model?: string;
  framework?: string;
  output?: string;
  taskTimeout?: number;
  rounds?: number;
  maxTickets?: number;
  maxValidations?: number;
  signal?: AbortSignal;
  onReport?: (report: any) => void;
}
export const save = (options: RunOptions, report: any) => {
  if (options.output) saveReport(join(options.output, "report.json"), report);
  options.onReport?.(structuredClone(report));
};
export const runId = (options: RunOptions) => (options.output ? basename(options.output) : identity().slice(0, 12));
export async function taskDeadline<T>(options: RunOptions, callback: (signal: AbortSignal) => Promise<T>): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort(options.signal?.reason);
  options.signal?.addEventListener("abort", abort, { once: true });
  if (options.signal?.aborted) abort();
  const timer = setTimeout(
    () => controller.abort(new DeadlineExceeded("Task deadline exceeded")),
    (options.taskTimeout ?? 360) * 1000,
  );
  try {
    return await waitFor(callback(controller.signal), undefined, controller.signal);
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", abort);
  }
}
export async function traceUsage(runtime: ReflectRuntime, report: any) {
  const ids = new Map<string, string>();
  for (const row of report.turns)
    for (const [role, traces] of Object.entries(row.phase_traces ?? {}))
      for (const id of traces as string[]) ids.set(id, role);
  for (const row of report.replay_arms) {
    const snapshot = runtime.workspace.loadExecution(row.execution_id);
    for (const traces of Object.values(snapshot?.phase_traces ?? {}))
      for (const id of traces as string[]) ids.set(id, "validation");
  }
  const slots = new Semaphore(4);
  return Promise.all(
    [...ids].map(([trace_id, role]) =>
      slots.run(async () => {
        try {
          const trace = await runtime.api.traces.get(trace_id, { timeout: 5 });
          return {
            trace_id,
            role,
            llm_spans: trace.spans
              .filter((span) => span.kind === "LLM")
              .map((span) => ({
                span_id: span.span_id,
                model: span.model ?? null,
                token_usage: span.token_usage ?? null,
                latency_ms: span.latency_ms ?? null,
                cost: span.cost ?? null,
              })),
          };
        } catch (error) {
          return { trace_id, role, unavailable: error instanceof Error ? error.name : "Error" };
        }
      }),
    ),
  );
}
export async function appendUsage(runtime: ReflectRuntime, report: any) {
  try {
    await runtime.waitExport({ timeout: 10 });
    report.trace_usage = await waitFor(traceUsage(runtime, report), 30);
  } catch (error) {
    report.trace_usage_error = error instanceof Error ? error.name : "Error";
  }
}
