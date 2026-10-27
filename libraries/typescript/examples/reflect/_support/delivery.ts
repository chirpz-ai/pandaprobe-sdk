/** Port of the Python example-only task-model-input audit. Never a core acceptance policy. */
import { waitFor } from "../../../src/reflect/async.js";
import type { Reflect as ReflectRuntime } from "../../../src/reflect/index.js";
import { digest } from "../../../src/reflect/models.js";
import { timestamp } from "./reporting.js";
export function inputText(value: any): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(inputText).join("\n");
  if (value?.role === "tool" && typeof value.content === "string") {
    // Claude tracing serializes MCP text blocks as JSON inside the LLM's tool
    // message. Decode that known representation, not arbitrary escaped text.
    // Keep the tracing payload and the exact rule/timestamp checks unchanged.
    try {
      const blocks = JSON.parse(value.content);
      if (Array.isArray(blocks) && blocks.every((block) => block?.type === "text" && typeof block.text === "string"))
        return blocks.map((block) => block.text).join("\n");
    } catch {}
  }
  if (value && typeof value === "object") return Object.values(value).map(inputText).join("\n");
  return "";
}
export class TaskInputAudit {
  executionId: string | null = null;
  records: any[] = [];
  record(value: unknown, runId: string | null = null): void {
    if (this.executionId !== null)
      this.records.push({
        span_id: runId,
        kind: "LLM",
        started_at: timestamp(),
        input: inputText(value),
        execution_id: this.executionId,
      });
  }
  callback() {
    return {
      name: "ReflectExampleInputAudit",
      handleChatModelStart: (_serialized: unknown, messages: unknown, runId: string) => this.record(messages, runId),
      handleLLMStart: (_serialized: unknown, prompts: unknown, runId: string) => this.record(prompts, runId),
    };
  }
}
export interface RuleCoverage {
  version: string;
  representation: string;
  matched_lines: number[];
  total_nonblank_lines: number;
  complete: boolean;
}
function numberedLines(text: string, path: string, source: string[], required: Set<number>): Set<number> {
  const prefix = `/rules/${path}:`;
  let best = new Set<number>();
  let block = new Set<number>();
  let previous = 0;
  for (const rendered of text.split("\n")) {
    let valid = false;
    const offset = rendered.indexOf(prefix);
    if (offset === 0 || (offset > 0 && /\s/.test(rendered[offset - 1]!))) {
      const tail = rendered.slice(offset + prefix.length);
      const separator = tail.indexOf(":");
      const number = tail.slice(0, separator);
      const value = tail.slice(separator + 1);
      if (separator >= 0 && /^[0-9]{1,9}$/.test(number)) {
        const index = Number(number);
        valid = index >= 1 && index <= source.length && source[index - 1] === value;
        if (valid) {
          if (index <= previous) block = new Set();
          previous = index;
          if (required.has(index)) block.add(index);
          if (block.size > best.size) best = new Set(block);
        }
      }
    }
    if (!valid) {
      block = new Set();
      previous = 0;
    }
  }
  return best;
}
export function matchRules(text: string, files: Record<string, string>): Record<string, RuleCoverage> {
  const coverage: Record<string, RuleCoverage> = {};
  for (const [path, content] of Object.entries(files)) {
    const source = content.replaceAll("\r\n", "\n");
    const required = new Set(source.split("\n").flatMap((line, index) => (line.trim() ? [index + 1] : [])));
    if (!required.size) continue;
    const verbatim = text.includes(source.trim());
    const matched = verbatim ? required : numberedLines(text, path, source.split("\n"), required);
    if (matched.size)
      coverage[path] = {
        version: digest(`${source.trim()}\n`),
        representation: verbatim ? "verbatim" : "numbered_grep",
        matched_lines: [...matched].sort((a, b) => a - b),
        total_nonblank_lines: required.size,
        complete: matched.size === required.size && [...matched].every((i) => required.has(i)),
      };
  }
  return coverage;
}
function instant(value: unknown): { micros: bigint; zoned: boolean } | null {
  if (typeof value !== "string") return null;
  const ms = Date.parse(value);
  if (!Number.isFinite(ms)) return null;
  const fraction = value.match(/[T ]\d\d:\d\d:\d\d[.,](\d+)/)?.[1] ?? "";
  return {
    micros: BigInt(Math.trunc(ms)) * 1000n + BigInt(fraction.padEnd(6, "0").slice(3, 6)),
    zoned: /(?:Z|[+-]\d\d:?\d\d)$/.test(value),
  };
}
export function inspectDelivery(traces: any[], files: Record<string, string>, firstSubmit: string | null) {
  const receipts: any[] = [];
  for (const trace of traces)
    for (const span of trace.spans ?? []) {
      if (span.kind !== "LLM" || span.input == null) continue;
      const coverage = matchRules(inputText(span.input), files);
      const started_at = span.started_at ?? null;
      const start = instant(started_at);
      const submit = instant(firstSubmit);
      receipts.push({
        trace_id: trace.trace_id,
        source: trace.source ?? "recorded task trace",
        execution_id: span.execution_id ?? null,
        span_id: span.span_id,
        started_at,
        before_submission: Boolean(start && submit && start.zoned === submit.zoned && start.micros < submit.micros),
        rule_versions: Object.fromEntries(
          Object.entries(coverage)
            .filter(([, row]) => row.complete)
            .map(([p, r]) => [p, r.version]),
        ),
        rule_coverage: coverage,
      });
    }
  const full = new Set<string>(receipts.flatMap((row) => Object.values(row.rule_versions) as string[]));
  const partial = new Set<string>(
    receipts.flatMap((row) =>
      (Object.values(row.rule_coverage) as RuleCoverage[]).filter((r) => !r.complete).map((r) => r.version),
    ),
  );
  const before = receipts.filter((row) => row.before_submission);
  return {
    source: "example audit of task LLM inputs",
    receipts,
    model_input_versions: [...full].sort(),
    partial_model_input_versions: [...partial].filter((v) => !full.has(v)).sort(),
    before_submission_versions: [
      ...new Set<string>(before.flatMap((row) => Object.values(row.rule_versions) as string[])),
    ].sort(),
    before_submission_excerpt_versions: [
      ...new Set<string>(
        before.flatMap((row) => (Object.values(row.rule_coverage) as RuleCoverage[]).map((r) => r.version)),
      ),
    ].sort(),
  };
}
export async function collectDelivery(
  runtime: ReflectRuntime,
  traceIds: string[],
  files: Record<string, string>,
  events: any[],
  observedInputs: any[] = [],
) {
  const firstSubmit = events.find((event) => event.operation === "submit")?.attempted_at ?? null;
  const observed = [{ trace_id: null, source: "example native observation", spans: observedInputs }];
  const empty = inspectDelivery(observed, files, firstSubmit);
  if (!Object.keys(files).length || !traceIds.length) return empty;
  try {
    await runtime.waitExport({ timeout: 10 });
    const traces = await waitFor(
      Promise.all([...new Set(traceIds)].map((id) => runtime.api.traces.waitVisible(id, { timeout: 30 }))),
      30,
    );
    return inspectDelivery([...observed, ...traces], files, firstSubmit);
  } catch (error) {
    return { ...empty, error: error instanceof Error ? error.name : "Error" };
  }
}
