import { InvalidResponseError } from "./errors.js";

/** Wire models preserve additive server fields and the platform's snake_case names. */
export interface ApiModel {
  [key: string]: unknown;
}
export interface Span extends ApiModel {
  span_id: string;
  trace_id: string;
  parent_span_id: string | null;
  name: string;
  kind: string;
  status: string;
  input: unknown;
  output: unknown;
  metadata: Record<string, unknown>;
  error: string | null;
}
export interface TraceSummary extends ApiModel {
  trace_id: string;
  name: string;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "ERROR";
  session_id: string | null;
  tags: string[];
  started_at: string;
  ended_at: string | null;
}
export interface Trace extends TraceSummary {
  project_id: string;
  input: unknown;
  output: unknown;
  metadata: Record<string, unknown>;
  spans: Span[];
}
export interface TracePage extends ApiModel {
  items: TraceSummary[];
  total: number;
  limit: number;
  offset: number;
}
export interface Metric extends ApiModel {
  name: string;
  description: string;
  category: string;
}
export interface EvaluationRun extends ApiModel {
  id: string;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";
  metric_names: string[];
  total_targets: number;
  evaluated_count: number;
  failed_count: number;
  target_type: string;
  error_message: string | null;
}
export interface Score extends ApiModel {
  id: string;
  trace_id: string;
  name: string;
  value: string | null;
  status: "PENDING" | "SUCCESS" | "FAILED";
  data_type: string;
  eval_run_id: string | null;
  reason: string | null;
  metadata: Record<string, unknown>;
}
export interface EvaluationRequest {
  trace_ids: string[];
  metrics: string[];
  name?: string;
  model?: string;
}
export interface EvaluationResult {
  run: EvaluationRun;
  scores: Score[];
}

export function uuid(value: string): string {
  if (typeof value !== "string") throw new TypeError("Expected a UUID string");
  const hex = value
    .replace(/^urn:uuid:/i, "")
    .replace(/^\{(.*)\}$/, "$1")
    .replaceAll("-", "");
  if (!/^[a-f0-9]{32}$/i.test(hex)) throw new TypeError("Expected a UUID string");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`.toLowerCase();
}
export function numericValue(score: Score): number | null {
  if (
    score.status !== "SUCCESS" ||
    score.data_type !== "NUMERIC" ||
    typeof score.value !== "string" ||
    !score.value.trim()
  )
    return null;
  // float(str) accepts decimal Unicode digits and underscores between digits.
  // Normalize only that grammar, retaining Python's rejection of hexadecimal.
  const value = score.value
    .replace(/^\p{White_Space}+|\p{White_Space}+$/gu, "")
    .replace(/\p{Decimal_Number}/gu, (c) => {
      const point = c.codePointAt(0)!;
      let zero = point;
      while (zero > 0 && /\p{Decimal_Number}/u.test(String.fromCodePoint(zero - 1))) zero--;
      return String((point - zero) % 10);
    });
  if (!/^[+-]?(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:e[+-]?\d(?:_?\d)*)?$/i.test(value)) return null;
  const number = Number(value.replaceAll("_", ""));
  return Number.isFinite(number) ? number : null;
}
const object = (v: unknown): Record<string, any> => {
  if (!v || typeof v !== "object" || Array.isArray(v)) throw new TypeError("Expected an object");
  return { ...v };
};
const string = (v: unknown): string => {
  if (typeof v !== "string") throw new TypeError("Expected a string");
  return v;
};
const nullable = <T>(v: unknown, parse: (v: any) => T): T | null => (v == null ? null : parse(v));
const array = <T>(v: unknown, parse: (v: any) => T): T[] => {
  if (!Array.isArray(v)) throw new TypeError("Expected an array");
  return v.map(parse);
};
const integer = (value: unknown, min = 0): number => {
  let v = value;
  if (typeof v === "boolean") v = Number(v);
  if (typeof v === "string") {
    const normalized = v.replace(/^\p{White_Space}+|\p{White_Space}+$/gu, "");
    if (/^[+-]?\d(?:_?\d)*(?:\.0+)?$/.test(normalized)) v = Number(normalized.replaceAll("_", ""));
  }
  if (typeof v !== "number" || !Number.isSafeInteger(v) || v < min) throw new TypeError("Invalid count");
  return v;
};
const status = <T extends string>(v: unknown, choices: T[]): T => {
  if (!choices.includes(v as T)) throw new TypeError("Invalid status");
  return v as T;
};
function summary(v: unknown): TraceSummary {
  const r = object(v);
  return {
    ...r,
    trace_id: uuid(r.trace_id),
    name: string(r.name),
    status: status(r.status, ["PENDING", "RUNNING", "COMPLETED", "ERROR"]),
    session_id: nullable(r.session_id, string),
    tags: array(r.tags === undefined ? [] : r.tags, string),
    started_at: string(r.started_at),
    ended_at: nullable(r.ended_at, string),
  };
}
export const decoders = {
  trace(v: unknown): Trace {
    const r = object(v);
    return {
      ...summary(r),
      project_id: uuid(r.project_id),
      input: r.input ?? null,
      output: r.output ?? null,
      metadata: object(r.metadata === undefined ? {} : r.metadata),
      spans: array(r.spans === undefined ? [] : r.spans, (v): Span => {
        const s = object(v);
        return {
          ...s,
          span_id: uuid(s.span_id),
          trace_id: uuid(s.trace_id),
          parent_span_id: nullable(s.parent_span_id, uuid),
          name: string(s.name),
          kind: string(s.kind),
          status: string(s.status),
          input: s.input ?? null,
          output: s.output ?? null,
          metadata: object(s.metadata === undefined ? {} : s.metadata),
          error: nullable(s.error, string),
        };
      }),
    };
  },
  page(v: unknown): TracePage {
    const r = object(v);
    return {
      ...r,
      items: array(r.items, summary),
      total: integer(r.total),
      limit: integer(r.limit, 1),
      offset: integer(r.offset),
    };
  },
  metric(v: unknown): Metric {
    const r = object(v);
    return { ...r, name: string(r.name), description: string(r.description), category: string(r.category) };
  },
  run(v: unknown): EvaluationRun {
    const r = object(v);
    return {
      ...r,
      id: uuid(r.id),
      status: status(r.status, ["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]),
      metric_names: array(r.metric_names, string),
      total_targets: integer(r.total_targets),
      evaluated_count: integer(r.evaluated_count),
      failed_count: integer(r.failed_count === undefined ? 0 : r.failed_count),
      target_type: string(r.target_type),
      error_message: nullable(r.error_message, string),
    };
  },
  score(v: unknown): Score {
    const r = object(v);
    if (!("value" in r)) throw new TypeError("Missing score value");
    return {
      ...r,
      id: uuid(r.id),
      trace_id: uuid(r.trace_id),
      name: string(r.name),
      value: nullable(r.value, string),
      status: status(r.status, ["PENDING", "SUCCESS", "FAILED"]),
      data_type: string(r.data_type),
      eval_run_id: nullable(r.eval_run_id, uuid),
      reason: nullable(r.reason, string),
      metadata: object(r.metadata === undefined ? {} : r.metadata),
    };
  },
};
export function decode<T>(parse: (v: unknown) => T, value: unknown): T {
  try {
    return parse(value);
  } catch {
    throw new InvalidResponseError("The platform returned an invalid response");
  }
}
export function decodeMany<T>(parse: (v: unknown) => T, value: unknown): T[] {
  return decode((v) => array(v, parse), value);
}
